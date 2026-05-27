#!/usr/bin/env python
"""
heatmap_tumor_dice.py

Compute Dice overlap between high-attention regions from a CLAM model and tumor annotations,
across multiple **range-percentage** cutoffs of the smoothed heatmap.

Locked-in assumptions (from previous script):
- Tile coordinates in the attention H5 are at slide level-0 (full resolution) and represent top-left corners.
- Only the **tile size** is adjusted for differing magnifications: tile_full_px = tile_size * (base_mag / extraction_mag).
- All mapping to the thumbnail uses floor for start indices.

Outputs
-------
1) An Excel workbook with two sheets, saved to --output_dir:
   - Sheet "per_slide": one row per slide with Dice at each cutoff, plus counts.
     Also pct_attn_in_*: for each percentile, % of binarized attention mask pixels
     that fall in each tissue compartment (tumor, nontumor, benign, normal when present).
   - Sheet "summary": mean ± std Dice (excluding NaN), and number of valid slides per cutoff.
     Also mean/std/n_valid for pct_attn_in_tumor, pct_attn_in_nontumor, and when present
     pct_attn_in_benign, pct_attn_in_normal.
2) (Optional) A CSV with the same per-slide table (for convenience).
3) data_type=rectal: tumor and non-tumor (and normal if present) only; no SRA class coverage.
4) data_type=rectal_new: for annotation dirs with only SRA class XMLs (no tumor). Slide validity is
   "has attention + at least one SRA class XML". Outputs: per_slide has cov_p{p}_{class} for each SRA class;
   sheet "sra_coverage_summary" gives mean ± std and n_valid per percentile and class (only over slides that
   have that class). No tumor dice/coverage columns.

Example
-------
python heatmap_tumor_dice.py \
  --slide_dir /path/to/WSIs \
  --attention_dir /path/to/attentions/20x_512px_0px_overlap/.../attentions \
  --annotation_dir /path/to/annotations  \
  --output_dir /path/to/out \
  --tile_size 512 \
  --blur_sigma 1.0 \
  --percentiles 25 50 60 70 80 90 

"""

import argparse
import sys
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import openslide
import h5py
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def infer_tiling_from_path(attn_dir: Path):
    """Infer (tile_size_px, extraction_mag_x) from path token like '20x_512px'.
    Only extraction magnification is used here; tile_size is provided via CLI.
    """
    m = re.search(r"(?P<mag>\d+)x[\-_](?P<size>\d+)px", str(attn_dir))
    if not m:
        return None, None
    return int(m.group("size")), int(m.group("mag"))


def load_attention(slide_stem: str, attention_dir: Path):
    """Load (coords Nx2, attn N) from H5 for a slide."""
    h5_path = attention_dir / f"{slide_stem}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Attention file not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        coords = np.array(f.get("coords", f.get("coord"))).reshape(-1, 2)
        attn = np.array(f.get("attention", f.get("attn"))).reshape(-1)
    return coords, attn


# Prostate annotation color constants (COLORREF format)
PROSTATE_TUMOR_COLOR = 65280   # Green (#00FF00)
PROSTATE_BENIGN_COLOR = 65535  # Yellow (#FFFF00)


def colorref_to_rgb(colorref):
    """Convert COLORREF (0x00BBGGRR) to RGB tuple.
    
    Args:
        colorref: Integer color value in COLORREF format, or None/string
        
    Returns:
        tuple: (R, G, B) values in range 0-255
    """
    if colorref is None:
        return None
    try:
        colorref = int(colorref)
        blue = (colorref >> 16) & 0xFF
        green = (colorref >> 8) & 0xFF
        red = colorref & 0xFF
        return (red, green, blue)
    except (ValueError, TypeError):
        return None


def find_tumor_xml_kidney(annotation_dir: Path, stem: str):
    """Return the first tumor-tagged XML for the slide (kidney format), or None."""
    hits = sorted(annotation_dir.glob(f"{stem}*tumor*.xml"))
    return hits[0] if hits else None


def find_tumor_xml_prostate(annotation_dir: Path, stem: str):
    """Return the XML file for the slide (prostate format), or None."""
    xml_path = annotation_dir / f"{stem}.xml"
    return xml_path if xml_path.exists() else None


def find_tumor_xml_rectal(annotation_dir: Path, stem: str):
    """Return the tumor XML file for rectal format (stored in subfolder), or None.
    
    Rectal annotations are stored as: annotation_dir/{stem}/{stem}.tumor.xml
    """
    xml_path = annotation_dir / stem / f"{stem}.tumor.xml"
    return xml_path if xml_path.exists() else None


def find_normal_xml_rectal(annotation_dir: Path, stem: str):
    """Return the normal XML file for rectal format (stored in subfolder), or None.
    
    Rectal annotations are stored as: annotation_dir/{stem}/{stem}.normal.xml
    """
    xml_path = annotation_dir / stem / f"{stem}.normal.xml"
    return xml_path if xml_path.exists() else None


def find_tumor_xml(annotation_dir: Path, stem: str, data_type: str):
    """Return the tumor annotation XML file path for given stem, or None.
    
    Args:
        annotation_dir: Directory containing annotation files
        stem: Slide filename stem (without extension)
        data_type: 'kidney', 'prostate', 'rectal', or 'rectal_new'
    
    Returns:
        Path to annotation file or None if not found
    """
    if data_type == 'kidney':
        return find_tumor_xml_kidney(annotation_dir, stem)
    elif data_type == 'prostate':
        return find_tumor_xml_prostate(annotation_dir, stem)
    elif data_type == 'rectal':
        return find_tumor_xml_rectal(annotation_dir, stem)
    elif data_type == 'rectal_new':
        return None  # rectal_new annotation dir has only SRA class XMLs, no tumor
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def find_normal_xml(annotation_dir: Path, stem: str, data_type: str):
    """Return the normal annotation XML file path for given stem, or None.
    
    Args:
        annotation_dir: Directory containing annotation files
        stem: Slide filename stem (without extension)
        data_type: 'kidney', 'prostate', 'rectal', or 'rectal_new'
    
    Returns:
        Path to annotation file or None if not found
    """
    if data_type == 'kidney':
        # Kidney format: look for files matching {stem}*normal*.xml in main dir
        hits = sorted(annotation_dir.glob(f"{stem}*normal*.xml"))
        return hits[0] if hits else None
    elif data_type == 'rectal':
        # Rectal format: look in subfolder
        return find_normal_xml_rectal(annotation_dir, stem)
    elif data_type == 'rectal_new':
        return None  # rectal_new annotation dir has only SRA class XMLs, no normal
    else:
        # Prostate format doesn't have separate normal annotations
        return None


# SRA tissue classes (output by SRA model; BACK omitted)
SRA_CLASSES = ['ADI', 'CSTR', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']


def find_sra_class_xml(annotation_dir: Path, stem: str, class_label: str):
    """Return the SRA class XML path for rectal format (subfolder per slide), or None.
    
    SRA annotations are stored as: annotation_dir/{stem}/{stem}.{class_label}.xml
    """
    xml_path = annotation_dir / stem / f"{stem}.{class_label}.xml"
    return xml_path if xml_path.exists() else None


def has_any_sra_annotation(annotation_dir: Path, stem: str) -> bool:
    """Return True if at least one SRA class XML exists for this slide under annotation_dir/{stem}/."""
    for cls in SRA_CLASSES:
        if find_sra_class_xml(annotation_dir, stem, cls) is not None:
            return True
    return False


def parse_tumor_polygons_kidney(xml_path: Path):
    """Parse tumor *.xml (kidney format, all regions are tumor). Return list of Nx2 float arrays."""
    polys = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Generic ASAP-like structure: Regions/Region/Vertices/Vertex
    for region in root.iter("Region"):
        verts = region.find("Vertices")
        if verts is None:
            continue
        pts = []
        for v in verts.iter("Vertex"):
            pts.append((float(v.get("X")), float(v.get("Y"))))
        if len(pts) >= 3:
            polys.append(np.array(pts, dtype=np.float32))
    return polys


def parse_tumor_polygons_prostate(xml_path: Path):
    """Parse prostate format XML with color support; return (tumor_polys, benign_polys) tuple.
    
    Prostate format structure:
    <Annotations> -> <Annotation> -> <Regions> -> <Region> -> <Vertices> -> <Vertex>
    
    Filters annotations by LineColor:
    - Green (65280) = Tumor
    - Yellow (65535) = Benign tissue
    - Other colors = Ignored
    
    Returns:
        tuple: (tumor_polys, benign_polys) where each is a list of polygons
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tumor_polys = []
    benign_polys = []
    
    # Iterate through all Annotation elements
    for annotation in root.findall('Annotation'):
        # Extract LineColor from annotation
        line_color = annotation.get('LineColor')
        colorref = int(line_color) if line_color else None
        
        # Iterate through all Region elements within this annotation
        for region in annotation.findall('.//Region'):
            pts = []
            verts = region.find('Vertices')
            if verts is None:
                continue
            for v in verts.iter('Vertex'):
                x = v.get('X')
                y = v.get('Y')
                if x is not None and y is not None:
                    pts.append((float(x), float(y)))
            if len(pts) >= 3:
                poly = np.array(pts, dtype=np.float32)
                # Filter by color
                if colorref == PROSTATE_TUMOR_COLOR:
                    tumor_polys.append(poly)
                elif colorref == PROSTATE_BENIGN_COLOR:
                    benign_polys.append(poly)
                # Other colors are ignored
    
    return tumor_polys, benign_polys


def parse_tumor_polygons(xml_path: Path, data_type: str):
    """Parse tumor XML based on data type.
    
    For prostate data_type: returns tuple (tumor_polys, benign_polys)
    For other data_types: returns list of tumor polygons
    
    Returns:
        For prostate: tuple (tumor_polys, benign_polys) where each is a list of polygons
        For others: list of Nx2 float arrays (polygons)
    """
    if data_type == 'kidney':
        return parse_tumor_polygons_kidney(xml_path)
    elif data_type == 'prostate':
        return parse_tumor_polygons_prostate(xml_path)
    elif data_type == 'rectal':
        # Rectal uses same XML format as prostate (Aperio-style) but without color filtering
        # For rectal, we still treat all annotations as tumor (backward compatibility)
        tumor_polys, benign_polys = parse_tumor_polygons_prostate(xml_path)
        # Combine all polygons as tumor for rectal (ignore benign)
        return tumor_polys + benign_polys
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def rasterize_polygons(polys, sx, sy, H, W):
    """Rasterize polygons (full-res coordinates) to a binary mask at thumbnail scale.
    Uses floor mapping for consistency and fills interiors.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    if not polys:
        return mask
    scaled = []
    for poly in polys:
        pts = np.stack([np.floor(poly[:, 0] * sx), np.floor(poly[:, 1] * sy)], axis=1)
        pts = pts.astype(np.int32)
        scaled.append(pts)
    mask = cv2.fillPoly(mask, scaled, 1)
    return mask


def build_smoothed_heatmap(coords, attn, W_full, H_full, sx, sy, tile_size, mag_ratio, H_thumb, W_thumb, blur_sigma):
    """Build the thumbnail-scale heatmap from tiles and apply Gaussian smoothing.

    Follows the locked-in logic:
    - coords are top-left at level-0
    - tile_full = tile_size * mag_ratio
    - start indices via floor; rect fill per tile
    - Gaussian smoothing, then clip to [0,1]
    """
    # Normalize attention to [0, 1]
    attn = attn.astype(np.float32)
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)

    # Tile footprint in thumbnail px
    tile_full = tile_size * mag_ratio
    tw = int(np.floor(tile_full * sx))
    th = int(np.floor(tile_full * sy))
    tw = max(tw, 1)
    th = max(th, 1)

    heat = np.zeros((H_thumb, W_thumb), dtype=np.float32)
    for (x_f, y_f), a in zip(coords, attn_norm):
        x0 = int(np.floor(x_f * sx))
        y0 = int(np.floor(y_f * sy))
        x1 = min(W_thumb, x0 + tw)
        y1 = min(H_thumb, y0 + th)
        if x0 < W_thumb and y0 < H_thumb and x1 > x0 and y1 > y0:
            heat[y0:y1, x0:x1] = a

    # Smoothing in thumbnail space
    if blur_sigma and blur_sigma > 0:
        heat = gaussian_filter(heat, sigma=blur_sigma)
    heat = np.clip(heat, 0, 1)
    return heat


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Dice = 2|A∩B| / (|A|+|B|). Returns np.nan if both masks empty."""
    A = mask_a.astype(bool)
    B = mask_b.astype(bool)
    inter = (A & B).sum()
    denom = A.sum() + B.sum()
    if denom == 0:
        return np.nan
    return 2.0 * inter / denom


def should_exclude_slide_rectal(slide_path: Path) -> bool:
    """Check if a slide should be excluded for rectal cancer analysis.
    
    Excludes:
    - HCM files (starting with "HCM-")
    - TCGA TS files (containing "-TS" in filename)
    
    Args:
        slide_path: Path to the slide file
        
    Returns:
        True if slide should be excluded, False otherwise
    """
    stem = slide_path.stem.upper()  # Case-insensitive matching
    # Exclude HCM files
    if stem.startswith("HCM-"):
        return True
    # Exclude TCGA files with -TS pattern
    if "TCGA" in stem and "-TS" in stem:
        return True
    return False


# ---------------------------------------------------------------------------
# Per-slide worker function (for multiprocessing)
# ---------------------------------------------------------------------------

def process_single_slide(slide_path, attn_dir, ann_dir, data_type, tile_size, 
                         target_dim, blur_sigma, percentiles, extraction_mag,
                         sra_coverage=False):
    """Process a single slide and return the results dictionary.
    
    This function is designed to be called by ProcessPoolExecutor.
    When data_type is 'rectal_new', computes only coverage of the heatmap over each
    SRA class (ADI, CSTR, DEB, etc.) per percentile (no tumor/normal annotations).
    """
    try:
        # Slide + metadata
        slide = openslide.OpenSlide(str(slide_path))
        W_full, H_full = slide.dimensions
        base_mag = float(slide.properties.get('openslide.objective-power', extraction_mag))
        mag_ratio = base_mag / extraction_mag

        # Thumbnail & scale
        scale = target_dim / max(W_full, H_full)
        thumb = slide.get_thumbnail((int(W_full * scale), int(H_full * scale)))
        slide.close()
        thumb_arr = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGBA2RGB)
        H_thumb, W_thumb = thumb_arr.shape[:2]
        sx, sy = W_thumb / W_full, H_thumb / H_full

        # Heatmap
        coords, attn = load_attention(slide_path.stem, attn_dir)
        heat = build_smoothed_heatmap(coords, attn, W_full, H_full, sx, sy,
                                      tile_size=tile_size, mag_ratio=mag_ratio,
                                      H_thumb=H_thumb, W_thumb=W_thumb, blur_sigma=blur_sigma)
        nz = heat[heat > 0]

        # rectal_new: annotation dir has only SRA class XMLs (no tumor/normal); compute SRA coverage only
        if data_type == 'rectal_new':
            slide_row = {
                'slide': slide_path.stem,
                'heat_nonzero_px': int(nz.size),
                'tumor_px': np.nan,
                'nontumor_px': np.nan,
                'normal_px': np.nan,
            }
            for p in percentiles:
                hmin = float(heat.min())
                hmax = float(heat.max())
                if hmax <= hmin:
                    heat_bin = np.zeros_like(heat, dtype=np.uint8)
                else:
                    cutoff = hmin + (p / 100.0) * (hmax - hmin)
                    heat_bin = (heat >= cutoff).astype(np.uint8)
                A = heat_bin.astype(bool)
                for cls in SRA_CLASSES:
                    sra_xml = find_sra_class_xml(ann_dir, slide_path.stem, cls)
                    if sra_xml is None:
                        slide_row[f'cov_p{p}_{cls}'] = np.nan
                        continue
                    sra_polys = parse_tumor_polygons_kidney(sra_xml)
                    if not sra_polys:
                        slide_row[f'cov_p{p}_{cls}'] = np.nan
                        continue
                    class_mask = rasterize_polygons(sra_polys, sx, sy, H_thumb, W_thumb)
                    class_sum = class_mask.sum()
                    if class_sum == 0:
                        slide_row[f'cov_p{p}_{cls}'] = np.nan
                    else:
                        cov = (np.logical_and(A, class_mask.astype(bool)).sum() / class_sum)
                        slide_row[f'cov_p{p}_{cls}'] = 100.0 * cov
            return slide_row

        # Tumor mask (and benign mask for prostate)
        xml_path = find_tumor_xml(ann_dir, slide_path.stem, data_type)
        if data_type == 'prostate':
            if xml_path:
                # Prostate: parse returns (tumor_polys, benign_polys) tuple
                tumor_polys, benign_polys = parse_tumor_polygons(xml_path, data_type)
                tumor_mask = rasterize_polygons(tumor_polys, sx, sy, H_thumb, W_thumb)
                benign_mask = rasterize_polygons(benign_polys, sx, sy, H_thumb, W_thumb)
                tumor_bool = tumor_mask.astype(bool)
                benign_bool = benign_mask.astype(bool)
                # Use benign mask instead of proxy for non-tumor
                non_tumor_bool = benign_bool
                non_tumor_mask = benign_mask
                has_benign = benign_mask.sum() > 0
            else:
                # No XML file found for prostate
                tumor_polys = []
                benign_polys = []
                tumor_mask = np.zeros((H_thumb, W_thumb), dtype=np.uint8)
                benign_mask = np.zeros((H_thumb, W_thumb), dtype=np.uint8)
                tumor_bool = tumor_mask.astype(bool)
                benign_bool = benign_mask.astype(bool)
                non_tumor_bool = benign_bool
                non_tumor_mask = benign_mask
                has_benign = False
        else:
            # Other data types: parse returns single list
            tumor_polys = parse_tumor_polygons(xml_path, data_type) if xml_path else []
            tumor_mask = rasterize_polygons(tumor_polys, sx, sy, H_thumb, W_thumb)
            tumor_bool = tumor_mask.astype(bool)
            benign_mask = None
            benign_bool = None
            has_benign = False
            # Non-tumor tissue mask (proxy): mean RGB > 220 AND not in tumor
            mean_img = thumb_arr.mean(axis=2)
            bright_bool = mean_img > 220.0
            non_tumor_bool = np.logical_and(bright_bool, np.logical_not(tumor_bool))
            non_tumor_mask = non_tumor_bool.astype(np.uint8)

        # Optional normal mask (only if *.normal.xml exists)
        normal_xml = find_normal_xml(ann_dir, slide_path.stem, data_type)
        has_normal = normal_xml is not None
        if has_normal:
            normal_polys = parse_tumor_polygons(normal_xml, data_type)
            # For prostate, parse_tumor_polygons returns tuple, but for normal we only need tumor_polys
            if data_type == 'prostate' and isinstance(normal_polys, tuple):
                normal_polys = normal_polys[0]  # Use tumor polygons from tuple
            normal_mask = rasterize_polygons(normal_polys, sx, sy, H_thumb, W_thumb)
            normal_bool = normal_mask.astype(bool)
        else:
            normal_mask = None
            normal_bool = None

        # Prepare non-zero heat pixels set (for diagnostics)
        nz = heat[heat > 0]
        slide_row = {
            'slide': slide_path.stem,
            'heat_nonzero_px': int(nz.size),
            'tumor_px': int(tumor_mask.sum()),
            'nontumor_px': int(non_tumor_mask.sum()),
            'benign_px': int(benign_mask.sum()) if has_benign else np.nan,
            'normal_px': int(normal_mask.sum()) if has_normal else np.nan,
        }

        # Range-based thresholds and metrics
        for p in percentiles:
            # Cutoff = min + (p/100)*(max-min)
            hmin = float(heat.min())
            hmax = float(heat.max())
            if hmax <= hmin:
                heat_bin = np.zeros_like(heat, dtype=np.uint8)
            else:
                cutoff = hmin + (p / 100.0) * (hmax - hmin)
                heat_bin = (heat >= cutoff).astype(np.uint8)

            # Dice vs tumor
            d = dice_score(heat_bin, tumor_mask)
            slide_row[f'dice_p{p}'] = d

            # Coverage of tumor by attention mask (percentage)
            A = heat_bin.astype(bool)
            T = tumor_bool
            NT = non_tumor_bool
            if T.sum() == 0:
                cov_tumor = np.nan
            else:
                cov_tumor = 100.0 * (np.logical_and(A, T).sum() / T.sum())
            if NT.sum() == 0:
                cov_nontumor = np.nan
            else:
                cov_nontumor = 100.0 * (np.logical_and(A, NT).sum() / NT.sum())
            slide_row[f'tumor_cov_p{p}'] = cov_tumor
            slide_row[f'nontumor_cov_p{p}'] = cov_nontumor

            # Benign overlap metrics (only for prostate with benign annotations)
            if has_benign:
                B = benign_bool
                db = dice_score(heat_bin, benign_mask)
                slide_row[f'dice_benign_p{p}'] = db
                if B.sum() == 0:
                    cov_benign = np.nan
                else:
                    cov_benign = 100.0 * (np.logical_and(A, B).sum() / B.sum())
                slide_row[f'benign_cov_p{p}'] = cov_benign
            else:
                slide_row[f'dice_benign_p{p}'] = np.nan
                slide_row[f'benign_cov_p{p}'] = np.nan

            # Normal overlap metrics (only if normal exists)
            if has_normal:
                N = normal_bool
                dn = dice_score(heat_bin, normal_mask)
                slide_row[f'dice_normal_p{p}'] = dn
                if N.sum() == 0:
                    cov_normal = np.nan
                else:
                    cov_normal = 100.0 * (np.logical_and(A, N).sum() / N.sum())
                slide_row[f'normal_cov_p{p}'] = cov_normal
            else:
                slide_row[f'dice_normal_p{p}'] = np.nan
                slide_row[f'normal_cov_p{p}'] = np.nan

            # Attention-in-compartment: % of binarized attention mask pixels in each compartment
            n_attn = int(A.sum())
            if n_attn == 0:
                slide_row[f'pct_attn_in_tumor_p{p}'] = np.nan
                slide_row[f'pct_attn_in_nontumor_p{p}'] = np.nan
                if has_benign:
                    slide_row[f'pct_attn_in_benign_p{p}'] = np.nan
                if has_normal:
                    slide_row[f'pct_attn_in_normal_p{p}'] = np.nan
            else:
                slide_row[f'pct_attn_in_tumor_p{p}'] = 100.0 * (np.logical_and(A, tumor_bool).sum() / n_attn)
                slide_row[f'pct_attn_in_nontumor_p{p}'] = 100.0 * (np.logical_and(A, non_tumor_bool).sum() / n_attn)
                if has_benign:
                    slide_row[f'pct_attn_in_benign_p{p}'] = 100.0 * (np.logical_and(A, benign_bool).sum() / n_attn)
                if has_normal:
                    slide_row[f'pct_attn_in_normal_p{p}'] = 100.0 * (np.logical_and(A, normal_bool).sum() / n_attn)

        return slide_row
    
    except Exception as e:
        # Return error info so we can report it
        return {'slide': slide_path.stem, 'error': str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate Dice overlap between heatmap and tumor.")
    ap.add_argument('--slide_dir',      required=True)
    ap.add_argument('--attention_dir',  required=True)
    ap.add_argument('--annotation_dir', required=True)
    ap.add_argument('--output_dir',     required=True)
    ap.add_argument('--data_type',      type=str, default='kidney', choices=['kidney', 'prostate', 'rectal', 'rectal_new'],
                    help='Dataset type: "kidney", "prostate", "rectal" (tumor+non-tumor only), or "rectal_new" (SRA class coverage only, no tumor)')
    ap.add_argument('--tile_size',      type=int, required=True, help='Tile size at extraction mag (px)')
    ap.add_argument('--target_dim',     type=int, default=2000, help='Longest side of thumbnail (px)')
    ap.add_argument('--blur_sigma',     type=float, default=1.0, help='Gaussian blur sigma (thumbnail px)')
    ap.add_argument('--percentiles',    type=int, nargs='+', default=[25, 50, 60, 70, 80, 90],
                    help='Range-percent cutoffs (0–100): threshold = min + (p/100)*(max−min) on smoothed heatmap')
    ap.add_argument('--save_csv',       action='store_true', help='Also save per-slide CSV next to Excel')
    ap.add_argument('--num_workers',    type=int, default=1, help='Number of parallel workers (default: 1)')
    ap.add_argument('--no_sra_coverage', action='store_true',
                    help='No-op (kept for compatibility). SRA coverage is only computed for data_type=rectal_new.')
    args = ap.parse_args()

    sra_coverage = (args.data_type == 'rectal_new')

    slide_dir = Path(args.slide_dir)
    attn_dir = Path(args.attention_dir)
    ann_dir = Path(args.annotation_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Infer extraction magnification
    _, extraction_mag = infer_tiling_from_path(attn_dir)
    if extraction_mag is None:
        sys.exit('Cannot infer extraction magnification from attention_dir path (needs token like 20x_512px)')

    slides = sorted(slide_dir.glob('*.svs'))
    if not slides:
        sys.exit('No SVS files found')

    # Apply rectal-specific filtering (rectal and rectal_new)
    if args.data_type in ('rectal', 'rectal_new'):
        original_count = len(slides)
        slides = [s for s in slides if not should_exclude_slide_rectal(s)]
        filtered_count = original_count - len(slides)
        if filtered_count > 0:
            print(f'Filtered out {filtered_count} slide(s) for rectal analysis (HCM and TCGA-TS files)')

    # Keep only slides with attention and required annotations
    valid = []
    for s in slides:
        if not (attn_dir / f"{s.stem}.h5").exists():
            continue
        if args.data_type == 'rectal_new':
            if not has_any_sra_annotation(ann_dir, s.stem):
                continue
        else:
            if not find_tumor_xml(ann_dir, s.stem, args.data_type):
                continue
        valid.append(s)

    if not valid:
        if args.data_type == 'rectal_new':
            sys.exit('No slides with both attention and at least one SRA class annotation found')
        sys.exit('No slides with both attention and tumor annotations found')

    print(f'Processing {len(valid)} slides with {args.num_workers} worker(s)...')
    records = []  # per-slide results

    if args.num_workers == 1:
        # Sequential processing
        for slide_path in tqdm(valid, desc='Slides'):
            result = process_single_slide(
                slide_path, attn_dir, ann_dir, args.data_type,
                args.tile_size, args.target_dim, args.blur_sigma,
                args.percentiles, extraction_mag, sra_coverage=sra_coverage
            )
            if 'error' in result:
                print(f"Error processing {result['slide']}: {result['error']}")
            else:
                records.append(result)
    else:
        # Parallel processing with ProcessPoolExecutor
        worker_fn = partial(
            process_single_slide,
            attn_dir=attn_dir,
            ann_dir=ann_dir,
            data_type=args.data_type,
            tile_size=args.tile_size,
            target_dim=args.target_dim,
            blur_sigma=args.blur_sigma,
            percentiles=args.percentiles,
            extraction_mag=extraction_mag,
            sra_coverage=sra_coverage
        )
        
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(worker_fn, slide_path): slide_path for slide_path in valid}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc='Slides'):
                result = future.result()
                if 'error' in result:
                    print(f"Error processing {result['slide']}: {result['error']}")
                else:
                    records.append(result)

    # Per-slide dataframe
    df = pd.DataFrame.from_records(records)

    # Summary (exclude NaNs per cutoff) for Dice, tumor coverage, non-tumor coverage, and normal metrics
    summary_rows = []
    for p in args.percentiles:
        # Dice (tumor) - may be missing for rectal_new
        col_d = f'dice_p{p}'
        vals_d = df[col_d].dropna().to_numpy(dtype=float) if col_d in df.columns else np.array([])
        mean_d = float(np.mean(vals_d)) if vals_d.size else np.nan
        std_d  = float(np.std(vals_d, ddof=1)) if vals_d.size > 1 else (0.0 if vals_d.size == 1 else np.nan)
        n_d    = int(vals_d.size)
        # Tumor coverage - may be missing for rectal_new
        col_t = f'tumor_cov_p{p}'
        vals_t = df[col_t].dropna().to_numpy(dtype=float) if col_t in df.columns else np.array([])
        mean_t = float(np.mean(vals_t)) if vals_t.size else np.nan
        std_t  = float(np.std(vals_t, ddof=1)) if vals_t.size > 1 else (0.0 if vals_t.size == 1 else np.nan)
        n_t    = int(vals_t.size)
        # Non-tumor coverage - may be missing for rectal_new
        col_n = f'nontumor_cov_p{p}'
        vals_n = df[col_n].dropna().to_numpy(dtype=float) if col_n in df.columns else np.array([])
        mean_n = float(np.mean(vals_n)) if vals_n.size else np.nan
        std_n  = float(np.std(vals_n, ddof=1)) if vals_n.size > 1 else (0.0 if vals_n.size == 1 else np.nan)
        n_n    = int(vals_n.size)
        # Normal Dice
        col_dn = f'dice_normal_p{p}'
        vals_dn = df[col_dn].dropna().to_numpy(dtype=float) if col_dn in df.columns else np.array([])
        mean_dn = float(np.mean(vals_dn)) if vals_dn.size else np.nan
        std_dn  = float(np.std(vals_dn, ddof=1)) if vals_dn.size > 1 else (0.0 if vals_dn.size == 1 else np.nan)
        n_dn    = int(vals_dn.size)
        # Normal coverage
        col_nc = f'normal_cov_p{p}'
        vals_nc = df[col_nc].dropna().to_numpy(dtype=float) if col_nc in df.columns else np.array([])
        mean_nc = float(np.mean(vals_nc)) if vals_nc.size else np.nan
        std_nc  = float(np.std(vals_nc, ddof=1)) if vals_nc.size > 1 else (0.0 if vals_nc.size == 1 else np.nan)
        n_nc    = int(vals_nc.size)
        # Benign Dice - only for prostate data_type
        col_db = f'dice_benign_p{p}'
        vals_db = df[col_db].dropna().to_numpy(dtype=float) if col_db in df.columns else np.array([])
        mean_db = float(np.mean(vals_db)) if vals_db.size else np.nan
        std_db  = float(np.std(vals_db, ddof=1)) if vals_db.size > 1 else (0.0 if vals_db.size == 1 else np.nan)
        n_db    = int(vals_db.size)
        # Benign coverage - only for prostate data_type
        col_bc = f'benign_cov_p{p}'
        vals_bc = df[col_bc].dropna().to_numpy(dtype=float) if col_bc in df.columns else np.array([])
        mean_bc = float(np.mean(vals_bc)) if vals_bc.size else np.nan
        std_bc  = float(np.std(vals_bc, ddof=1)) if vals_bc.size > 1 else (0.0 if vals_bc.size == 1 else np.nan)
        n_bc    = int(vals_bc.size)

        # Attention-in-compartment: mean/std/n_valid per percentile
        col_pt = f'pct_attn_in_tumor_p{p}'
        vals_pt = df[col_pt].dropna().to_numpy(dtype=float) if col_pt in df.columns else np.array([])
        mean_pt = float(np.mean(vals_pt)) if vals_pt.size else np.nan
        std_pt  = float(np.std(vals_pt, ddof=1)) if vals_pt.size > 1 else (0.0 if vals_pt.size == 1 else np.nan)
        n_pt    = int(vals_pt.size)
        col_pn = f'pct_attn_in_nontumor_p{p}'
        vals_pn = df[col_pn].dropna().to_numpy(dtype=float) if col_pn in df.columns else np.array([])
        mean_pn = float(np.mean(vals_pn)) if vals_pn.size else np.nan
        std_pn  = float(np.std(vals_pn, ddof=1)) if vals_pn.size > 1 else (0.0 if vals_pn.size == 1 else np.nan)
        n_pn    = int(vals_pn.size)
        col_pb = f'pct_attn_in_benign_p{p}'
        vals_pb = df[col_pb].dropna().to_numpy(dtype=float) if col_pb in df.columns else np.array([])
        mean_pb = float(np.mean(vals_pb)) if vals_pb.size else np.nan
        std_pb  = float(np.std(vals_pb, ddof=1)) if vals_pb.size > 1 else (0.0 if vals_pb.size == 1 else np.nan)
        n_pb    = int(vals_pb.size)
        col_pnc = f'pct_attn_in_normal_p{p}'
        vals_pnc = df[col_pnc].dropna().to_numpy(dtype=float) if col_pnc in df.columns else np.array([])
        mean_pnc = float(np.mean(vals_pnc)) if vals_pnc.size else np.nan
        std_pnc  = float(np.std(vals_pnc, ddof=1)) if vals_pnc.size > 1 else (0.0 if vals_pnc.size == 1 else np.nan)
        n_pnc    = int(vals_pnc.size)

        summary_rows.append({
            'percentile': p,
            'mean_dice': mean_d, 'std_dice': std_d, 'n_valid_dice': n_d,
            'mean_tumor_cov': mean_t, 'std_tumor_cov': std_t, 'n_valid_tumor_cov': n_t,
            'mean_nontumor_cov': mean_n, 'std_nontumor_cov': std_n, 'n_valid_nontumor_cov': n_n,
            'mean_dice_normal': mean_dn, 'std_dice_normal': std_dn, 'n_valid_dice_normal': n_dn,
            'mean_normal_cov': mean_nc, 'std_normal_cov': std_nc, 'n_valid_normal_cov': n_nc,
            'mean_dice_benign': mean_db, 'std_dice_benign': std_db, 'n_valid_dice_benign': n_db,
            'mean_benign_cov': mean_bc, 'std_benign_cov': std_bc, 'n_valid_benign_cov': n_bc,
            'mean_pct_attn_in_tumor': mean_pt, 'std_pct_attn_in_tumor': std_pt, 'n_valid_pct_attn_in_tumor': n_pt,
            'mean_pct_attn_in_nontumor': mean_pn, 'std_pct_attn_in_nontumor': std_pn, 'n_valid_pct_attn_in_nontumor': n_pn,
            'mean_pct_attn_in_benign': mean_pb, 'std_pct_attn_in_benign': std_pb, 'n_valid_pct_attn_in_benign': n_pb,
            'mean_pct_attn_in_normal': mean_pnc, 'std_pct_attn_in_normal': std_pnc, 'n_valid_pct_attn_in_normal': n_pnc,
        })
    df_summary = pd.DataFrame(summary_rows)

    # SRA coverage summary (rectal only): mean coverage per class per percentile over slides that have that class
    df_sra_summary = None
    if sra_coverage:
        sra_summary_rows = []
        for p in args.percentiles:
            for cls in SRA_CLASSES:
                col = f'cov_p{p}_{cls}'
                if col not in df.columns:
                    continue
                vals = df[col].dropna().to_numpy(dtype=float)
                mean_c = float(np.mean(vals)) if vals.size else np.nan
                std_c = float(np.std(vals, ddof=1)) if vals.size > 1 else (0.0 if vals.size == 1 else np.nan)
                n_c = int(vals.size)
                sra_summary_rows.append({
                    'percentile': p, 'class': cls,
                    'mean_coverage': mean_c, 'std_coverage': std_c, 'n_valid': n_c,
                })
        df_sra_summary = pd.DataFrame(sra_summary_rows) if sra_summary_rows else None

    # Save Excel with two sheets (pick an available engine) or fall back to CSVs
    def _pick_excel_engine():
        try:
            import xlsxwriter  # noqa: F401
            return 'xlsxwriter'
        except Exception:
            try:
                import openpyxl  # noqa: F401
                return 'openpyxl'
            except Exception:
                return None

    engine = _pick_excel_engine()
    if engine is not None:
        xlsx_path = out_dir / 'dice_results.xlsx'
        with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
            df.to_excel(writer, index=False, sheet_name='per_slide')
            df_summary.to_excel(writer, index=False, sheet_name='summary')
            if df_sra_summary is not None and not df_sra_summary.empty:
                df_sra_summary.to_excel(writer, index=False, sheet_name='sra_coverage_summary')
        print(f'Saved Excel ({engine}): {xlsx_path}')
    else:
        # neither xlsxwriter nor openpyxl available – write two CSVs instead
        per_csv = out_dir / 'dice_per_slide.csv'
        sum_csv = out_dir / 'dice_summary.csv'
        df.to_csv(per_csv, index=False)
        df_summary.to_csv(sum_csv, index=False)
        if df_sra_summary is not None and not df_sra_summary.empty:
            df_sra_summary.to_csv(out_dir / 'dice_sra_coverage_summary.csv', index=False)
        print('xlsxwriter/openpyxl not installed – saved CSVs instead:')
        print(f'  per-slide: {per_csv}')
        print(f'  summary:   {sum_csv}')
        if df_sra_summary is not None and not df_sra_summary.empty:
            print(f'  SRA summary: {out_dir / "dice_sra_coverage_summary.csv"}')

    # Optional CSV for per_slide (kept for backward compatibility)
    if args.save_csv and engine is not None:
        # If Excel was written, also save per-slide CSV when requested
        df.to_csv(out_dir / 'dice_per_slide.csv', index=False)



if __name__ == '__main__':
    main()
