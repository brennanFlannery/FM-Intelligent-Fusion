#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregated Tumor vs Normal t-SNE Analysis

Creates t-SNE visualizations by aggregating tiles across multiple slides, showing tiles 
from tumor and normal regions. Computes global metrics on the full aggregated dataset using 
bootstrapping (50 iterations, 80% sampling) and compares distributions between embedding 
models using Wilcoxon rank-sum tests.

Key features:
- Aggregates tiles from multiple slides per embedding model
- One t-SNE per embedding model (not per slide)
- Global metrics computed via bootstrapping for statistical robustness
- Wilcoxon rank-sum tests between model pairs
- Selective borders: red for tumor, blue for normal

Outputs:
- embedding_X.png: Aggregated t-SNE mosaic for each embedding
- global_metrics_bootstrap.csv: Bootstrapped metric scores (50 samples per embedding)
- model_comparison_wilcoxon.csv: Statistical comparison between models

Requirements:
  numpy, scipy, scikit-learn, pandas, h5py, openslide-python, pillow, tqdm, shapely

Example:
  python create_aggregated_tumor_normal_tsne.py \
      --h5_root /path/to/features_gigapath /path/to/features_uni \
      --svs_root /path/to/WSI \
      --annotation_dir /path/to/annotations \
      --output_dir /path/to/output \
      --max_slides 50 \
      --grid_rows 100 --grid_cols 100
"""

import argparse
import os
import re
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
import openslide

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from shapely.geometry import Point, Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

# Prostate annotation color constants (COLORREF format)
PROSTATE_TUMOR_COLOR = 65280   # Green (#00FF00)
PROSTATE_BENIGN_COLOR = 65535  # Yellow (#FFFF00)

# SRA tissue classes (rectal_sra; BACK omitted)
SRA_CLASSES = ['ADI', 'CSTR', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
SRA_EXCLUDE_DEFAULT = ['BACK', 'DEB']
# For rectal_sra silhouette only: MUS+STR vs TUM+NORM; other classes excluded
SRA_SILHOUETTE_GROUP_MUSCLE_STROMA = ('MUS', 'STR')
SRA_SILHOUETTE_GROUP_TUMOR_BENIGN = ('TUM', 'NORM')

# Per-subplot width (inches) for non-rectal_sra bar/box plots so they match rectal_sra subplot width
SUBPLOT_WIDTH_INCH = 1.6


def get_sra_class_color(class_label: str, dataset_name: str = 'kather19crctp') -> Tuple[int, int, int]:
    """Return (R, G, B) 0-255 for an SRA class label (for PIL/mosaic borders)."""
    if dataset_name == 'kather19':
        class_colors = {
            'ADI': [247, 129, 191], 'BACK': [153, 153, 153], 'DEB': [255, 255, 51],
            'LYM': [255, 0, 255], 'MUC': [23, 190, 192], 'MUS': [255, 127, 0],
            'NORM': [0, 255, 0], 'STR': [55, 126, 184], 'TUM': [228, 26, 28],
        }
    elif dataset_name == 'kather19crctp':
        class_colors = {
            'ADI': [247, 129, 191], 'BACK': [153, 153, 153], 'DEB': [255, 255, 51],
            'LYM': [255, 0, 255], 'MUC': [23, 190, 192], 'MUS': [255, 127, 0],
            'NORM': [0, 255, 0], 'STR': [55, 126, 184], 'TUM': [228, 26, 28],
            'CSTR': [0, 128, 0],
        }
    else:
        class_colors = {
            'ADI': [247, 129, 191], 'CSTR': [0, 128, 0], 'DEB': [255, 255, 51],
            'LYM': [255, 0, 255], 'MUC': [23, 190, 192], 'MUS': [255, 127, 0],
            'STR': [55, 126, 184], 'TUM': [228, 26, 28],
        }
    color = class_colors.get(class_label, [128, 128, 128])
    return tuple(color)


def find_sra_npy_file(annotation_dir: Path, stem: str):
    """Return annotation_dir/{stem}/{stem}.svs_classification_sra.npy, or None."""
    npy_path = annotation_dir / stem / f"{stem}.svs_classification_sra.npy"
    return npy_path if npy_path.exists() else None


def load_sra_classification_npy(npy_path: Path) -> dict:
    """Load NPY file; validate keys classification/metadata/classification_labels."""
    if not npy_path.exists():
        raise FileNotFoundError(f"NPY file not found: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()
    for key in ['classification', 'metadata', 'classification_labels']:
        if key not in data:
            raise KeyError(f"Required key '{key}' not found in NPY file: {npy_path}")
    return data


# Grid step for rectal_sra coarse class grid (pixels per cell)
SRA_GRID_STEP = 256


def build_sra_class_grid(
    npy_tx: np.ndarray,
    npy_ty: np.ndarray,
    s_src: int,
    predicted_classes: np.ndarray,
    class_labels,
    step: int = SRA_GRID_STEP,
):
    """Build a coarse 2D grid of majority-vote SRA class per cell from NPY patches.

    Returns:
        class_grid: (grid_rows, grid_cols) int32, value = index into active_labels or -1
        active_labels: list of class names (excluding SRA_EXCLUDE_DEFAULT)
        grid_rows, grid_cols: grid shape for bounds checking
    If no patches or no active classes, returns (None, [], 0, 0).
    """
    if npy_tx.size == 0:
        return None, [], 0, 0
    active_labels = sorted(
        set(class_labels[predicted_classes[i]] for i in range(len(predicted_classes))
            if class_labels[predicted_classes[i]] not in SRA_EXCLUDE_DEFAULT)
    )
    if not active_labels:
        return None, [], 0, 0
    label_to_id = {l: i for i, l in enumerate(active_labels)}
    n_classes = len(active_labels)
    max_tx = int(npy_tx.max()) + s_src
    max_ty = int(npy_ty.max()) + s_src
    grid_cols = (max_tx // step) + 1
    grid_rows = (max_ty // step) + 1
    grid_counts = np.zeros((grid_rows, grid_cols, n_classes), dtype=np.int32)
    for i in range(len(predicted_classes)):
        cls = class_labels[predicted_classes[i]]
        if cls in SRA_EXCLUDE_DEFAULT:
            continue
        tx, ty = int(npy_tx[i]), int(npy_ty[i])
        cid = label_to_id[cls]
        cx_lo = tx // step
        cx_hi = (tx + s_src - 1) // step + 1
        cy_lo = ty // step
        cy_hi = (ty + s_src - 1) // step + 1
        cx_lo = max(0, cx_lo)
        cx_hi = min(grid_cols, cx_hi)
        cy_lo = max(0, cy_lo)
        cy_hi = min(grid_rows, cy_hi)
        if cx_hi > cx_lo and cy_hi > cy_lo:
            grid_counts[cy_lo:cy_hi, cx_lo:cx_hi, cid] += 1
    class_grid = np.argmax(grid_counts, axis=-1).astype(np.int32)
    total = grid_counts.sum(axis=-1)
    class_grid[total == 0] = -1
    return class_grid, active_labels, grid_rows, grid_cols


def find_rectal_tumor_xml(annotation_dir: Path, stem: str):
    """Return rectal format tumor XML path: annotation_dir/{stem}/{stem}.tumor.xml, or None."""
    xml_path = annotation_dir / stem / f"{stem}.tumor.xml"
    return xml_path if xml_path.exists() else None


def find_rectal_normal_xml(annotation_dir: Path, stem: str):
    """Return rectal format normal XML path: annotation_dir/{stem}/{stem}.normal.xml, or None."""
    xml_path = annotation_dir / stem / f"{stem}.normal.xml"
    return xml_path if xml_path.exists() else None


# -----------------------------
# Utilities (from original script)
# -----------------------------

def extract_embedding_name(h5_root: str) -> str:
    """Extract embedding name from path."""
    m = re.search(r'features[_-](\w+)', h5_root, re.IGNORECASE)
    if m:
        return m.group(1)
    parts = Path(h5_root).parts
    for part in reversed(parts):
        if part and part not in ['.', '..', '/']:
            return part
    return "embedding"


def parse_mag_and_tile(path: str) -> Tuple[int, int]:
    """Infer magnification and tile size from path."""
    m = re.search(r'(\d+)\s*x[_/\\]+(\d+)\s*px', path, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.search(r'(\d+)\s*x.*?_(\d+)\s*px', path, re.IGNORECASE)
    if m2:
        return int(m2.group(1)), int(m2.group(2))
    return 20, 256


def basename_no_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def find_matching_svs(basename: str, svs_root: str) -> str:
    """Find matching whole slide image file."""
    candidates = glob.glob(os.path.join(svs_root, '**', basename + '.svs'), recursive=True)
    if candidates:
        return candidates[0]
    for ext in ['.tif', '.tiff', '.ndpi', '.svslide', '.mrxs']:
        candidates = glob.glob(os.path.join(svs_root, '**', basename + ext), recursive=True)
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Matching WSI not found for {basename} in {svs_root}")


def find_tumor_xml(annotation_dir: Path, stem: str):
    """Return the first tumor-tagged XML for the slide, or None."""
    hits = sorted(annotation_dir.glob(f"{stem}*tumor*.xml"))
    return hits[0] if hits else None


def find_normal_xml(annotation_dir: Path, stem: str):
    """Return the first normal-tagged XML for the slide, or None."""
    hits = sorted(annotation_dir.glob(f"{stem}*normal*.xml"))
    return hits[0] if hits else None


def find_prostate_xml(annotation_dir: Path, stem: str):
    """Return the single annotation XML for the slide (prostate format), or None."""
    xml_path = annotation_dir / f"{stem}.xml"
    return xml_path if xml_path.exists() else None


def parse_polygons_from_xml(xml_path: Path) -> List[np.ndarray]:
    """Parse annotation XML and return list of polygons."""
    polys = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
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


def parse_polygons_prostate(xml_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Parse prostate format XML with color support; return (tumor_polys, benign_polys).

    Prostate format: <Annotations> -> <Annotation LineColor=...> -> <Regions> -> <Region> -> <Vertices>.
    LineColor 65280 (Green) = Tumor, 65535 (Yellow) = Benign. Other colors are ignored.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tumor_polys = []
    benign_polys = []
    for annotation in root.findall('Annotation'):
        line_color = annotation.get('LineColor')
        colorref = int(line_color) if line_color else None
        for region in annotation.findall('.//Region'):
            verts = region.find('Vertices')
            if verts is None:
                continue
            pts = []
            for v in verts.iter('Vertex'):
                x, y = v.get('X'), v.get('Y')
                if x is not None and y is not None:
                    pts.append((float(x), float(y)))
            if len(pts) >= 3:
                poly = np.array(pts, dtype=np.float32)
                if colorref == PROSTATE_TUMOR_COLOR:
                    tumor_polys.append(poly)
                elif colorref == PROSTATE_BENIGN_COLOR:
                    benign_polys.append(poly)
    return tumor_polys, benign_polys


def point_in_polygon_simple(px: float, py: float, poly: np.ndarray) -> bool:
    """Simple ray-casting for point-in-polygon test."""
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def get_tile_bounds(coord: np.ndarray, tile_px: int, mag_ratio: float) -> Tuple[float, float, float, float]:
    """Get bounding box of a tile."""
    tile_full = tile_px * mag_ratio
    return coord[0], coord[1], coord[0] + tile_full, coord[1] + tile_full


def tile_overlaps_any_polygon(tile_bounds: Tuple[float, float, float, float], 
                              polygons: List[np.ndarray]) -> bool:
    """Check if a tile overlaps with any polygon."""
    x_min, y_min, x_max, y_max = tile_bounds
    
    if HAS_SHAPELY:
        from shapely.geometry import box
        tile_box = box(x_min, y_min, x_max, y_max)
        for poly_coords in polygons:
            try:
                poly = Polygon(poly_coords)
                if tile_box.intersects(poly):
                    return True
            except:
                continue
        return False
    else:
        # Fallback
        tile_corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        for poly in polygons:
            for cx, cy in tile_corners:
                if point_in_polygon_simple(cx, cy, poly):
                    return True
            for px, py in poly:
                if x_min <= px <= x_max and y_min <= py <= y_max:
                    return True
            poly_x_min, poly_y_min = poly.min(axis=0)
            poly_x_max, poly_y_max = poly.max(axis=0)
            if not (x_max < poly_x_min or x_min > poly_x_max or 
                    y_max < poly_y_min or y_min > poly_y_max):
                return True
        return False


def reduce_and_embed(features: np.ndarray, seed: int = 42, pca_threshold: int = 50,
                     tsne_perplexity: float = 30.0, tsne_iter: int = 1000) -> np.ndarray:
    """Reduce dimensionality and run t-SNE."""
    D = features.shape[1]
    X = features
    if D > pca_threshold:
        pca = PCA(n_components=pca_threshold, random_state=seed)
        X = pca.fit_transform(features)
    
    n_samples = X.shape[0]
    perplexity = min(tsne_perplexity, (n_samples - 1) / 3.0)
    perplexity = max(5.0, perplexity)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto',
                max_iter=tsne_iter, init='pca', random_state=seed)
    Y = tsne.fit_transform(X)
    return Y


def normalize_to_unit_square(X: np.ndarray) -> np.ndarray:
    """Normalize coordinates to [0,1]x[0,1]."""
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    span = np.maximum(mx - mn, 1e-12)
    return (X - mn) / span


def assign_points_to_grid_direct(points01: np.ndarray, rows: int, cols: int, rng=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct assignment: map each point to its natural grid cell.
    When multiple points map to the same cell, choose the one closest to the cell center.
    Returns: (cell_indices, cell_rc, selected_indices)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    N = points01.shape[0]
    r = np.clip((points01[:, 1] * rows).astype(int), 0, rows - 1)
    c = np.clip((points01[:, 0] * cols).astype(int), 0, cols - 1)
    
    # Compute cell centers
    cell_centers = np.zeros((rows, cols, 2))
    for ri in range(rows):
        for ci in range(cols):
            cell_centers[ri, ci, 0] = (ci + 0.5) / cols
            cell_centers[ri, ci, 1] = (ri + 0.5) / rows
    
    # Group points by cell
    cell_to_points = {}
    for i in range(N):
        key = (r[i], c[i])
        if key not in cell_to_points:
            cell_to_points[key] = []
        cell_to_points[key].append(i)
    
    # For each occupied cell, choose one point
    selected_indices = []
    out_rc = []
    
    for (ri, ci), point_indices in cell_to_points.items():
        if len(point_indices) == 1:
            selected_indices.append(point_indices[0])
            out_rc.append((ri, ci))
        elif len(point_indices) == 2:
            chosen = rng.choice(point_indices)
            selected_indices.append(chosen)
            out_rc.append((ri, ci))
        else:
            center = cell_centers[ri, ci]
            points_in_cell = points01[point_indices]
            dists = np.sum((points_in_cell - center) ** 2, axis=1)
            closest_idx = point_indices[np.argmin(dists)]
            selected_indices.append(closest_idx)
            out_rc.append((ri, ci))
    
    selected_indices = np.array(selected_indices, dtype=int)
    out_rc = np.array(out_rc, dtype=int)
    cell_indices = out_rc[:, 0] * cols + out_rc[:, 1]
    
    return cell_indices, out_rc, selected_indices


def assign_points_to_grid_greedy(points01: np.ndarray, rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy grid assignment with spiral search."""
    N = points01.shape[0]
    r = np.clip((points01[:, 1] * rows).astype(int), 0, rows - 1)
    c = np.clip((points01[:, 0] * cols).astype(int), 0, cols - 1)

    tree = cKDTree(points01)
    dists, _ = tree.query(points01, k=min(8, max(2, N)))
    local_score = dists[:, 1:].mean(axis=1)
    order = np.argsort(-local_score)

    occupied = np.full((rows, cols), False, dtype=bool)
    out_rc = np.zeros((N, 2), dtype=int)

    for idx in order:
        rr, cc = r[idx], c[idx]
        if not occupied[rr, cc]:
            occupied[rr, cc] = True
            out_rc[idx] = (rr, cc)
            continue
        found = False
        max_radius = max(rows, cols)
        for rad in range(1, max_radius):
            rmin, rmax = max(0, rr - rad), min(rows - 1, rr + rad)
            cmin, cmax = max(0, cc - rad), min(cols - 1, cc + rad)
            for rcur in range(rmin, rmax + 1):
                for ccur in (cmin, cmax):
                    if not occupied[rcur, ccur]:
                        occupied[rcur, ccur] = True
                        out_rc[idx] = (rcur, ccur)
                        found = True
                        break
                if found: break
            if found: continue
            for ccur in range(cmin + 1, cmax):
                for rcur in (rmin, rmax):
                    if not occupied[rcur, ccur]:
                        occupied[rcur, ccur] = True
                        out_rc[idx] = (rcur, ccur)
                        found = True
                        break
                if found: break
            if found: break
        if not found:
            out_rc[idx] = (rr, cc)

    cell_indices = out_rc[:, 0] * cols + out_rc[:, 1]
    return cell_indices, out_rc


def find_boundary_sides(labels: np.ndarray, rc: np.ndarray, rows: int, cols: int) -> List[Dict[str, bool]]:
    """Find which sides of each tile border a different group or empty space."""
    N = len(labels)
    grid_label = np.full((rows, cols), -1, dtype=int)
    
    for i in range(N):
        r, c = rc[i]
        grid_label[r, c] = labels[i]
    
    border_sides = []
    for i in range(N):
        r, c = rc[i]
        my_label = labels[i]
        sides = {'top': False, 'bottom': False, 'left': False, 'right': False}
        
        if r > 0:
            neighbor = grid_label[r-1, c]
            if neighbor == -1 or neighbor != my_label:
                sides['top'] = True
        else:
            sides['top'] = True
            
        if r < rows - 1:
            neighbor = grid_label[r+1, c]
            if neighbor == -1 or neighbor != my_label:
                sides['bottom'] = True
        else:
            sides['bottom'] = True
            
        if c > 0:
            neighbor = grid_label[r, c-1]
            if neighbor == -1 or neighbor != my_label:
                sides['left'] = True
        else:
            sides['left'] = True
            
        if c < cols - 1:
            neighbor = grid_label[r, c+1]
            if neighbor == -1 or neighbor != my_label:
                sides['right'] = True
        else:
            sides['right'] = True
        
        border_sides.append(sides)
    
    return border_sides


def add_selective_border(img: Image.Image, color: Tuple[int, int, int], 
                        sides: Dict[str, bool], border_width: int = 3) -> Image.Image:
    """Add colored borders only on specific sides."""
    img_with_border = img.copy()
    draw = ImageDraw.Draw(img_with_border)
    w, h = img.size
    
    if sides.get('top', False):
        for i in range(border_width):
            draw.line([(0, i), (w-1, i)], fill=color, width=1)
    
    if sides.get('bottom', False):
        for i in range(border_width):
            draw.line([(0, h-1-i), (w-1, h-1-i)], fill=color, width=1)
    
    if sides.get('left', False):
        for i in range(border_width):
            draw.line([(i, 0), (i, h-1)], fill=color, width=1)
    
    if sides.get('right', False):
        for i in range(border_width):
            draw.line([(w-1-i, 0), (w-1-i, h-1)], fill=color, width=1)
    
    return img_with_border


def extract_tile(slide: openslide.OpenSlide, top_left_xy: Tuple[int, int], tile_px: int) -> Image.Image:
    """Extract a tile from the slide at level 0."""
    x, y = int(top_left_xy[0]), int(top_left_xy[1])
    region = slide.read_region(location=(x, y), level=0, size=(tile_px, tile_px)).convert("RGB")
    return region


def stitch_mosaic(tiles: List[Image.Image], rc: np.ndarray, rows: int, cols: int, 
                  mosaic_tile_px: int, bg_color=(255, 255, 255)) -> Image.Image:
    """Stitch tiles into a mosaic grid."""
    H = rows * mosaic_tile_px
    W = cols * mosaic_tile_px
    canvas = Image.new('RGB', (W, H), color=bg_color)
    
    for img, (r, c) in zip(tiles, rc):
        if img is None:
            continue
        if img.size != (mosaic_tile_px, mosaic_tile_px):
            img = img.resize((mosaic_tile_px, mosaic_tile_px), resample=Image.BILINEAR)
        canvas.paste(img, (c * mosaic_tile_px, r * mosaic_tile_px))
    
    return canvas


def create_annotated_thumbnail(slide_path: Path, tumor_polys: List[np.ndarray], 
                               normal_polys: List[np.ndarray], target_dim: int = 2000) -> Image.Image:
    """Create a thumbnail with annotation outlines."""
    slide = openslide.OpenSlide(str(slide_path))
    W_full, H_full = slide.dimensions
    
    scale = target_dim / max(W_full, H_full)
    thumb = slide.get_thumbnail((int(W_full * scale), int(H_full * scale)))
    slide.close()
    
    thumb_rgb = thumb.convert('RGB')
    draw = ImageDraw.Draw(thumb_rgb)
    
    for poly in tumor_polys:
        scaled_poly = [(int(x * scale), int(y * scale)) for x, y in poly]
        draw.polygon(scaled_poly, outline=(230, 25, 75), width=3)
    
    for poly in normal_polys:
        scaled_poly = [(int(x * scale), int(y * scale)) for x, y in poly]
        draw.polygon(scaled_poly, outline=(0, 130, 200), width=3)
    
    return thumb_rgb


# -----------------------------
# Main pipeline
# -----------------------------

def collect_tiles_from_slides(h5_root: str, svs_root: Path, annotation_dir: Path,
                              valid_slides: List[Path], mag_x: int, tile_px: int,
                              embedding_name: str, args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Tuple], Dict[int, str]]:
    """
    Collect all tiles from all slides for one embedding type.
    Returns: (features, labels, slide_indices, slide_names, tile_coords_info, label_map)
    label_map: int -> class name (e.g. {0: 'tumor', 1: 'normal'} or SRA class names).
    """
    data_type = getattr(args, 'data_type', 'kidney')
    sra_dataset = getattr(args, 'sra_dataset', 'kather19crctp')
    all_features = []
    all_labels = []
    all_slide_indices = []
    all_coords_info = []
    slide_names = []
    # Binary modes: fixed label_map. rectal_sra: built incrementally.
    if data_type == 'prostate':
        label_map = {0: 'tumor', 1: 'benign'}
    elif data_type in ('kidney', 'rectal'):
        label_map = {0: 'tumor', 1: 'normal'}
    else:
        label_map = {}
    global_class_to_int = {}  # for rectal_sra: class_name -> int (union across slides)

    for slide_idx, slide_path in enumerate(tqdm(valid_slides, desc=f"[{embedding_name}] Loading tiles")):
        slide_stem = slide_path.stem

        if data_type == 'prostate':
            xml_path = find_prostate_xml(annotation_dir, slide_stem)
            if not xml_path:
                continue
            tumor_polys, normal_polys = parse_polygons_prostate(xml_path)
            if not tumor_polys or not normal_polys:
                continue
        elif data_type == 'rectal':
            tumor_xml = find_rectal_tumor_xml(annotation_dir, slide_stem)
            normal_xml = find_rectal_normal_xml(annotation_dir, slide_stem)
            if not tumor_xml or not normal_xml:
                continue
            tumor_polys = parse_polygons_from_xml(tumor_xml)
            normal_polys = parse_polygons_from_xml(normal_xml)
            if not tumor_polys or not normal_polys:
                continue
        elif data_type == 'rectal_sra':
            npy_path = find_sra_npy_file(annotation_dir, slide_stem)
            if not npy_path:
                continue
            h5_path = Path(h5_root) / f"{slide_stem}.h5"
            if not h5_path.exists():
                continue
            try:
                npy_data = load_sra_classification_npy(npy_path)
            except (FileNotFoundError, KeyError) as e:
                if args.debug:
                    print(f"[skip] {slide_stem}: NPY load error: {e}")
                continue
            try:
                with h5py.File(h5_path, 'r') as f:
                    features = f['features'][:]
                    coords = np.array(f.get('coords', f.get('coord'))).reshape(-1, 2)
            except Exception as e:
                if args.debug:
                    print(f"[skip] {slide_stem}: Error loading H5: {e}")
                continue
            if features.shape[0] != coords.shape[0]:
                continue
            predicted_classes = np.argmax(npy_data['classification'], axis=1)
            class_labels = npy_data['classification_labels']
            metadata = npy_data['metadata']
            npy_tx = metadata[:, 2].astype(int)
            npy_ty = metadata[:, 3].astype(int)
            # SRA patch size in pixels (metadata columns: mag, level, tx, ty, cx, cy, bx, by, s_src, s_tar)
            s_src = int(metadata[0, -2]) if metadata.size > 0 else 512
            class_grid, active_labels, grid_rows, grid_cols = build_sra_class_grid(
                npy_tx, npy_ty, s_src, predicted_classes, class_labels
            )
            tile_class_labels = []
            valid_tile_indices = []
            if class_grid is not None:
                step = SRA_GRID_STEP
                for i in range(len(coords)):
                    x, y = int(coords[i][0]), int(coords[i][1])
                    cx, cy = x // step, y // step
                    if 0 <= cx < grid_cols and 0 <= cy < grid_rows:
                        cid = class_grid[cy, cx]
                        if cid >= 0:
                            tile_class_labels.append(active_labels[cid])
                            valid_tile_indices.append(i)
            found_classes = sorted(set(tile_class_labels))
            if len(found_classes) < 2:
                if args.debug and len(valid_tile_indices) == 0 and len(coords) > 0:
                    print(f"  [{slide_stem}] rectal_sra: no H5 coords matched NPY grid (embedding {embedding_name} tile_px={tile_px}; SRA patch size={s_src}px)")
                continue
            for c in found_classes:
                if c not in global_class_to_int:
                    global_class_to_int[c] = len(global_class_to_int)
            class_to_int = global_class_to_int
            slide = openslide.OpenSlide(str(slide_path))
            base_mag = float(slide.properties.get('openslide.objective-power', mag_x))
            mag_ratio = base_mag / mag_x
            slide.close()
            rng = np.random.default_rng(args.seed + slide_idx)
            sampled_indices = []
            for c in found_classes:
                class_int = class_to_int[c]
                indices_this_class = [valid_tile_indices[j] for j in range(len(valid_tile_indices)) if tile_class_labels[j] == c]
                n_sample = max(1, int(len(indices_this_class) * args.sample_fraction))
                chosen = rng.choice(indices_this_class, size=min(n_sample, len(indices_this_class)), replace=False)
                sampled_indices.extend(chosen.tolist())
            sampled_indices = np.unique(sampled_indices)
            all_features.append(features[sampled_indices])
            slide_labels = np.array([class_to_int[tile_class_labels[valid_tile_indices.index(i)]] for i in sampled_indices])
            all_labels.append(slide_labels)
            all_slide_indices.append(np.full(len(sampled_indices), slide_idx, dtype=int))
            for idx in sampled_indices:
                all_coords_info.append((slide_idx, coords[idx][0], coords[idx][1], mag_ratio))
            slide_names.append(slide_stem)
            if args.debug:
                print(f"  [{slide_stem}] rectal_sra: {len(sampled_indices)} tiles from classes {found_classes}")
            continue

        else:
            # kidney
            tumor_xml = find_tumor_xml(annotation_dir, slide_stem)
            normal_xml = find_normal_xml(annotation_dir, slide_stem)
            if not tumor_xml or not normal_xml:
                continue
            tumor_polys = parse_polygons_from_xml(tumor_xml)
            normal_polys = parse_polygons_from_xml(normal_xml)
            if not tumor_polys or not normal_polys:
                continue

        # Load H5 (for non-rectal_sra we did not load yet in this loop iteration)
        h5_path = Path(h5_root) / f"{slide_stem}.h5"
        if not h5_path.exists():
            continue
        try:
            with h5py.File(h5_path, 'r') as f:
                features = f['features'][:]
                coords = f['coords'][:]
        except Exception as e:
            if args.debug:
                print(f"[skip] {slide_stem}: Error loading H5: {e}")
            continue
        if features.shape[0] != coords.shape[0]:
            continue

        slide = openslide.OpenSlide(str(slide_path))
        base_mag = float(slide.properties.get('openslide.objective-power', mag_x))
        mag_ratio = base_mag / mag_x
        slide.close()

        tumor_indices = []
        normal_indices = []
        for i, coord in enumerate(coords):
            tile_bounds = get_tile_bounds(coord, tile_px, mag_ratio)
            if tile_overlaps_any_polygon(tile_bounds, tumor_polys):
                tumor_indices.append(i)
            elif tile_overlaps_any_polygon(tile_bounds, normal_polys):
                normal_indices.append(i)
        if len(tumor_indices) == 0 or len(normal_indices) == 0:
            continue

        rng = np.random.default_rng(args.seed + slide_idx)
        n_tumor_sample = max(1, int(len(tumor_indices) * args.sample_fraction))
        n_normal_sample = max(1, int(len(normal_indices) * args.sample_fraction))
        tumor_sample_idx = rng.choice(tumor_indices, size=min(n_tumor_sample, len(tumor_indices)), replace=False)
        normal_sample_idx = rng.choice(normal_indices, size=min(n_normal_sample, len(normal_indices)), replace=False)

        all_features.append(features[tumor_sample_idx])
        all_features.append(features[normal_sample_idx])
        all_labels.append(np.zeros(len(tumor_sample_idx), dtype=int))
        all_labels.append(np.ones(len(normal_sample_idx), dtype=int))
        all_slide_indices.append(np.full(len(tumor_sample_idx), slide_idx, dtype=int))
        all_slide_indices.append(np.full(len(normal_sample_idx), slide_idx, dtype=int))
        for idx in tumor_sample_idx:
            all_coords_info.append((slide_idx, coords[idx][0], coords[idx][1], mag_ratio))
        for idx in normal_sample_idx:
            all_coords_info.append((slide_idx, coords[idx][0], coords[idx][1], mag_ratio))
        slide_names.append(slide_stem)
        if args.debug:
            other_label = "benign" if data_type == 'prostate' else "normal"
            print(f"  [{slide_stem}] {len(tumor_indices)} tumor ({len(tumor_sample_idx)} sampled), {len(normal_indices)} {other_label} ({len(normal_sample_idx)} sampled)")

    if len(all_features) == 0:
        return None, None, None, None, None, None

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    slide_indices = np.concatenate(all_slide_indices, axis=0)

    if data_type == 'rectal_sra':
        label_map = {i: c for c, i in global_class_to_int.items()}

    print(f"[{embedding_name}] Total: {len(features)} tiles from {len(slide_names)} slides (sampled at {args.sample_fraction*100:.0f}%)")
    if data_type == 'rectal_sra':
        for idx, cname in sorted(label_map.items()):
            print(f"  {cname}: {(labels == idx).sum()}")
    else:
        other_label = "Benign" if data_type == 'prostate' else "Normal"
        print(f"  Tumor: {(labels == 0).sum()}, {other_label}: {(labels == 1).sum()}")

    return features, labels, slide_indices, slide_names, all_coords_info, label_map


def compute_global_bootstrapped_metrics(Y: np.ndarray, labels: np.ndarray, features: np.ndarray,
                                       embedding_name: str, label_map: Dict[int, str],
                                       n_bootstrap: int = 50, bootstrap_fraction: float = 0.8,
                                       seed: int = 42, data_type: Optional[str] = None) -> List[Dict]:
    """
    Compute global metrics on full aggregated dataset using bootstrapping.
    Uses label_map for per-class compactness (binary or N-class).
    """
    results = []
    rng = np.random.default_rng(seed)
    n_total = len(Y)
    n_sample = int(n_total * bootstrap_fraction)
    label_map_rev = {v: k for k, v in label_map.items()}

    if len(np.unique(labels)) < 2:
        print(f"[{embedding_name}] Warning: Need at least 2 classes for metrics, skipping")
        return results

    print(f"[{embedding_name}] Computing global metrics with {n_bootstrap} bootstrap iterations (sampling {n_sample}/{n_total} tiles each)...")

    for bootstrap_idx in tqdm(range(n_bootstrap), desc=f"[{embedding_name}] Bootstrapping"):
        sample_indices = rng.choice(n_total, size=n_sample, replace=True)
        sample_Y = Y[sample_indices]
        sample_labels = labels[sample_indices]
        sample_features = features[sample_indices]
        if len(np.unique(sample_labels)) < 2:
            continue

        # rectal_sra silhouette: MUS+STR vs TUM+NORM only; other classes excluded
        if data_type == 'rectal_sra':
            sample_class_names = [label_map[int(l)] for l in sample_labels]
            in_sil = np.array([
                c in SRA_SILHOUETTE_GROUP_MUSCLE_STROMA or c in SRA_SILHOUETTE_GROUP_TUMOR_BENIGN
                for c in sample_class_names
            ])
            if in_sil.sum() == 0:
                sil_2d = np.nan
                sil_hd = np.nan
            else:
                Y_sub = sample_Y[in_sil]
                feat_sub = sample_features[in_sil]
                sil_labels_sub = np.array([
                    0 if sample_class_names[i] in SRA_SILHOUETTE_GROUP_MUSCLE_STROMA else 1
                    for i in range(len(sample_labels)) if in_sil[i]
                ])
                if len(np.unique(sil_labels_sub)) < 2:
                    sil_2d = np.nan
                    sil_hd = np.nan
                else:
                    try:
                        sil_2d = silhouette_score(Y_sub, sil_labels_sub, metric='euclidean')
                    except Exception:
                        sil_2d = np.nan
                    if feat_sub.shape[0] < 10000 and feat_sub.shape[1] < 2048:
                        try:
                            sil_hd = silhouette_score(feat_sub, sil_labels_sub, metric='euclidean')
                        except Exception:
                            sil_hd = np.nan
                    else:
                        sil_hd = np.nan
        else:
            try:
                sil_2d = silhouette_score(sample_Y, sample_labels, metric='euclidean')
            except Exception:
                sil_2d = np.nan
            if sample_features.shape[0] < 10000 and sample_features.shape[1] < 2048:
                try:
                    sil_hd = silhouette_score(sample_features, sample_labels, metric='euclidean')
                except Exception:
                    sil_hd = np.nan
            else:
                sil_hd = np.nan

        per_class_compactness = {}
        for label_int, class_name in label_map.items():
            mask = sample_labels == label_int
            if mask.sum() > 0:
                pts = sample_Y[mask]
                centroid = pts.mean(axis=0)
                per_class_compactness[class_name] = float(np.sqrt(((pts - centroid) ** 2).sum(axis=1)).mean())
            else:
                per_class_compactness[class_name] = np.nan

        row = {
            'bootstrap_idx': bootstrap_idx,
            'embedding': embedding_name,
            'silhouette_2d': float(sil_2d),
            'silhouette_highdim': float(sil_hd),
        }
        for class_name in label_map.values():
            label_int = label_map_rev[class_name]
            row[f'n_{class_name}_tiles'] = int((sample_labels == label_int).sum())
            row[f'{class_name}_compactness'] = per_class_compactness[class_name]
        results.append(row)

    print(f"[{embedding_name}] Completed {len(results)}/{n_bootstrap} successful bootstrap iterations")
    return results


def create_publication_boxplots(embedding_data, comparison_results, output_dir, data_type=None):
    """Create publication-ready box plots for all metrics in the style of compare_model_performance.py."""
    
    # Define model colors (matching compare_model_performance.py)
    model_colors = {
        'conch': '#0070C0',      # Blue
        'musk': '#FFC000',       # Yellow
        'hoptimus': '#C00000',   # Red
        'virchow': '#7030A0',    # Purple
        'gigapath': '#00B050',   # Green
        'pruned': '#FF6600',     # Orange
        'naive': '#FFFFFF',      # White with dots
        'uni': '#92D050',        # Light green
        'conch_v15': '#0070C0'   # Blue (variant)
    }
    
    plt.style.use('default')
    sns.set_palette("husl")
    all_metric_keys = set()
    for emb_data in embedding_data.values():
        all_metric_keys.update(emb_data.keys())
    compactness_keys = sorted(k for k in all_metric_keys if k.endswith('_compactness'))
    metrics = (['silhouette_2d'] if 'silhouette_2d' in all_metric_keys else []) + compactness_keys
    if not metrics:
        return
    metric_labels = {k: k.replace('_', ' ').title() for k in metrics}
    metric_labels['silhouette_2d'] = 'Silhouette Coefficient'
    y_ranges = {'silhouette_2d': (-1.0, 1.0)}
    if data_type != 'rectal_sra' and 'silhouette_2d' in metrics:
        y_ranges['silhouette_2d'] = (0.0, 0.5)
    for k in compactness_keys:
        y_ranges[k] = (0.0, None)
    
    # Get all models that exist in data
    all_models = list(embedding_data.keys())
    
    # Sort models: pruned first, naive second, then alphabetically
    def model_sort_key(model):
        if model == 'pruned':
            return (0, model)
        elif model == 'naive':
            return (1, model)
        else:
            return (2, model)
    
    models_in_plot = sorted(all_models, key=model_sort_key)
    
    print(f"Models to plot: {models_in_plot}")
    
    # Create subplots (thinner figure for non-rectal_sra to match rectal_sra per-subplot width)
    fig_width = (len(metrics) * SUBPLOT_WIDTH_INCH) if data_type != 'rectal_sra' else 16
    fig, axes = plt.subplots(1, len(metrics), figsize=(fig_width, 4))
    if len(metrics) == 1:
        axes = [axes]
    
    # Process each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Prepare data for box plot
        model_data_list = []
        models_with_data = []
        
        for model in models_in_plot:
            if model in embedding_data and metric in embedding_data[model]:
                values = embedding_data[model][metric]
                if len(values) > 0:
                    model_data_list.append(values)
                    models_with_data.append(model)
        
        if not model_data_list:
            continue
        
        # Create box plot
        box_plot = ax.boxplot(model_data_list,
                             labels=models_with_data,
                             patch_artist=True,
                             showfliers=False,  # Hide outliers for cleaner look
                             widths=0.4)  # Make boxes narrower
        
        # Apply colors and styling (matching compare_model_performance.py)
        for patch, model in zip(box_plot['boxes'], models_with_data):
            # Try to match model name (handle variants like hoptimus1, virchow2)
            base_model = model.rstrip('0123456789')  # Remove trailing numbers
            color = model_colors.get(model, model_colors.get(base_model, '#CCCCCC'))
            
            # Special styling for specific models
            if 'naive' in model.lower():
                patch.set_facecolor('white')
                patch.set_hatch('...')  # Dotted pattern
                patch.set_alpha(1.0)
            elif 'pruned' in model.lower():
                patch.set_facecolor('white')
                patch.set_alpha(1.0)
            else:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Make box borders thicker and black
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        
        # Make whiskers, caps, and medians thicker and black
        for whisker in box_plot['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(2)
        
        for cap in box_plot['caps']:
            cap.set_color('black')
            cap.set_linewidth(2)
        
        for median in box_plot['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # Add significance stars (single * for p < 0.05; comparison with pruned model)
        if 'pruned' in models_with_data:
            pruned_idx = models_with_data.index('pruned')
            
            for i, model in enumerate(models_with_data):
                if model != 'pruned':
                    # Find p-value for this comparison
                    p_value = None
                    comp_metric = '2D_silhouette' if metric == 'silhouette_2d' else metric
                    for comp in comparison_results:
                        if ((comp['model1'] == 'pruned' and comp['model2'] == model) or 
                            (comp['model1'] == model and comp['model2'] == 'pruned')) and \
                           comp['metric'] == comp_metric:
                            p_value = comp['p_value']
                            break
                    else:
                        p_value = None
                    if p_value is not None and p_value < 0.05:
                        star = '*'  # Single asterisk for any significant result
                        whisker_cap = box_plot['caps'][i*2 + 1]
                        whisker_y = whisker_cap.get_ydata()[0]
                        
                        # Position star above whisker
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        ax.text(i + 1, whisker_y + 0.02 * y_range, star,
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize the plot
        ax.set_title(metric_labels[metric], fontsize=7, fontweight='normal', pad=25)
        
        # Set y-axis range
        if y_ranges[metric][1] is not None:
            ax.set_ylim(y_ranges[metric])
        else:
            # Auto-range but start from specified minimum
            y_min = y_ranges[metric][0]
            y_max = max([max(data) for data in model_data_list]) * 1.1
            ax.set_ylim(y_min, y_max)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Remove grid lines
        ax.grid(False)
        
        # Remove x-axis ticks and labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        # Remove y-axis label
        ax.set_ylabel('')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'combined_metrics_boxplot.png'
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved combined box plot to {output_path}")


def create_publication_barplots(embedding_data, comparison_results, output_dir, data_type=None):
    """Create publication-ready bar plots for all metrics in the style of compare_model_performance.py."""
    
    model_colors = {
        'conch': '#0070C0', 'musk': '#FFC000', 'hoptimus': '#C00000', 'virchow': '#7030A0',
        'gigapath': '#00B050', 'pruned': '#FF6600', 'naive': '#FFFFFF', 'uni': '#92D050',
        'conch_v15': '#0070C0'
    }
    plt.style.use('default')
    sns.set_palette("husl")
    all_metric_keys = set()
    for emb_data in embedding_data.values():
        all_metric_keys.update(emb_data.keys())
    compactness_keys = sorted(k for k in all_metric_keys if k.endswith('_compactness'))
    metrics = (['silhouette_2d'] if 'silhouette_2d' in all_metric_keys else []) + compactness_keys
    if not metrics:
        return
    metric_labels = {k: k.replace('_', ' ').title() for k in metrics}
    metric_labels['silhouette_2d'] = 'Silhouette Coefficient'
    y_ranges = {'silhouette_2d': (-1.0, 1.0)}
    if data_type != 'rectal_sra' and 'silhouette_2d' in metrics:
        y_ranges['silhouette_2d'] = (0.0, 0.5)
    for k in compactness_keys:
        y_ranges[k] = (0.0, None)
    
    all_models = list(embedding_data.keys())
    
    # Sort models: pruned first, naive second, then alphabetically
    def model_sort_key(model):
        if model == 'pruned':
            return (0, model)
        elif model == 'naive':
            return (1, model)
        else:
            return (2, model)
    
    models_in_plot = sorted(all_models, key=model_sort_key)
    
    print(f"Models to plot in bar chart: {models_in_plot}")
    
    # Create subplots (thinner figure for non-rectal_sra to match rectal_sra per-subplot width)
    fig_width = (len(metrics) * SUBPLOT_WIDTH_INCH) if data_type != 'rectal_sra' else 16
    fig, axes = plt.subplots(1, len(metrics), figsize=(fig_width, 4))
    if len(metrics) == 1:
        axes = [axes]
    
    # Process each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Prepare data for bar plot
        model_means = []
        model_stds = []
        models_with_data = []
        
        for model in models_in_plot:
            if model in embedding_data and metric in embedding_data[model]:
                values = embedding_data[model][metric]
                if len(values) > 0:
                    model_means.append(np.mean(values))
                    model_stds.append(np.std(values))
                    models_with_data.append(model)
        
        if not model_means:
            continue
        
        # Determine bar bottom based on y-axis range
        # For silhouette (range -1 to 1), bars should start at -1
        # For compactness (range 0+), bars should start at 0
        bar_bottom = y_ranges[metric][0] if y_ranges[metric][0] is not None else 0.0
        
        # Create bar plot
        x_pos = np.arange(len(models_with_data))
        # Adjust means to be relative to bar_bottom
        bar_heights = [m - bar_bottom for m in model_means]
        bars = ax.bar(x_pos, bar_heights, bottom=bar_bottom,
                     width=0.6, edgecolor='black', linewidth=2)
        
        # Apply colors and styling (matching compare_model_performance.py)
        for bar, model in zip(bars, models_with_data):
            # Try to match model name (handle variants like hoptimus1, virchow2)
            base_model = model.rstrip('0123456789')  # Remove trailing numbers
            color = model_colors.get(model, model_colors.get(base_model, '#CCCCCC'))
            
            # Special styling for specific models
            if 'naive' in model.lower():
                bar.set_facecolor('white')
                bar.set_hatch('...')  # Dotted pattern
                bar.set_alpha(1.0)
            elif 'pruned' in model.lower():
                bar.set_facecolor('white')
                bar.set_alpha(1.0)
            else:
                bar.set_facecolor(color)
                bar.set_alpha(0.7)
            
            # Make bar borders thicker and black (already set above)
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
        
        # Customize the plot
        ax.set_title(metric_labels[metric], fontsize=7, fontweight='normal', pad=25)
        
        # Set y-axis range
        if y_ranges[metric][1] is not None:
            ax.set_ylim(y_ranges[metric])
        else:
            # Auto-range but start from specified minimum
            y_min = y_ranges[metric][0]
            y_max = max(model_means) * 1.1
            ax.set_ylim(y_min, y_max)
        
        # Add significance stars (single * for p < 0.05; comparison with pruned model)
        # Position after setting y-axis limits
        if 'pruned' in models_with_data:
            pruned_idx = models_with_data.index('pruned')
            
            for i, model in enumerate(models_with_data):
                if model != 'pruned':
                    # Find p-value for this comparison
                    p_value = None
                    # Map metric names between plot and comparison results
                    comp_metric = '2D_silhouette' if metric == 'silhouette_2d' else metric
                    for comp in comparison_results:
                        if ((comp['model1'] == 'pruned' and comp['model2'] == model) or 
                            (comp['model1'] == model and comp['model2'] == 'pruned')) and \
                           comp['metric'] == comp_metric:
                            p_value = comp['p_value']
                            break
                    else:
                        p_value = None
                    if p_value is not None and p_value < 0.05:
                        star = '*'  # Single asterisk for any significant result
                        bar_height = model_means[i]
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        star_y = bar_height + 0.02 * y_range
                        
                        # Position star above bar
                        ax.text(i, star_y, star,
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Remove grid lines
        ax.grid(False)
        
        # Remove x-axis ticks and labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        # Remove y-axis label
        ax.set_ylabel('')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'combined_metrics_barplot.png'
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved combined bar plot to {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Aggregated Tumor vs Normal t-SNE analysis.")
    ap.add_argument('--h5_root', required=True, nargs='+', 
                    help='Root folder(s) for H5 files (multiple for different embeddings).')
    ap.add_argument('--svs_root', required=True, help='Root folder containing WSI files.')
    ap.add_argument('--annotation_dir', required=True, help='Directory with tumor/normal XMLs.')
    ap.add_argument('--output_dir', required=True, help='Output directory.')
    ap.add_argument('--data_type', choices=['kidney', 'prostate', 'rectal', 'rectal_sra'], default='kidney',
                    help='Annotation format: kidney, prostate, rectal (subfolder tumor/normal XMLs), or rectal_sra (NPY SRA classes).')
    ap.add_argument('--sra_dataset', type=str, default='kather19crctp', choices=['kather19', 'kather19crctp'],
                    help='SRA dataset name for rectal_sra color mapping.')
    ap.add_argument('--max_slides', type=int, default=None, help='Maximum slides to process (-1 or None = all slides).')
    ap.add_argument('--sample_fraction', type=float, default=0.1, help='Fraction of tiles to sample per group per slide (0.1 = 10%%).')
    ap.add_argument('--grid_rows', type=int, default=100, help='Grid rows for mosaic (ignored if --auto_grid).')
    ap.add_argument('--grid_cols', type=int, default=100, help='Grid cols for mosaic (ignored if --auto_grid).')
    ap.add_argument('--auto_grid', action='store_true', 
                    help='Automatically size grid to fit all tiles (square grid based on sqrt of tile count).')
    ap.add_argument('--grid_method', choices=['greedy', 'direct'], default='direct',
                    help='Grid assignment method: direct (preserve t-SNE structure, recommended) or greedy (all tiles, but distorts t-SNE positions).')
    ap.add_argument('--mosaic_tile_px', type=int, default=64, help='Tile size in mosaic.')
    ap.add_argument('--border_width', type=int, default=3, help='Border width in pixels.')
    ap.add_argument('--seed', type=int, default=42, help='Random seed.')
    ap.add_argument('--tsne_perplexity', type=float, default=30.0, help='t-SNE perplexity.')
    ap.add_argument('--tsne_iter', type=int, default=2000, help='t-SNE iterations.')
    ap.add_argument('--thumbnail_size', type=int, default=2000, help='Thumbnail dimension.')
    ap.add_argument('--debug', action='store_true', help='Verbose debug prints.')
    args = ap.parse_args()
    
    svs_root = Path(args.svs_root)
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get valid slides
    slides = sorted(svs_root.glob('*.svs'))
    valid_slides = []
    for slide_path in slides:
        if args.data_type == 'prostate':
            if find_prostate_xml(annotation_dir, slide_path.stem):
                valid_slides.append(slide_path)
        elif args.data_type == 'rectal':
            if find_rectal_tumor_xml(annotation_dir, slide_path.stem) and find_rectal_normal_xml(annotation_dir, slide_path.stem):
                valid_slides.append(slide_path)
        elif args.data_type == 'rectal_sra':
            if find_sra_npy_file(annotation_dir, slide_path.stem):
                valid_slides.append(slide_path)
        else:
            if find_tumor_xml(annotation_dir, slide_path.stem) and find_normal_xml(annotation_dir, slide_path.stem):
                valid_slides.append(slide_path)
    
    if not valid_slides:
        if args.data_type == 'prostate':
            print("No slides with both tumor and benign annotations found")
        elif args.data_type == 'rectal':
            print("No slides with both tumor and normal annotations (rectal subfolder XMLs) found")
        elif args.data_type == 'rectal_sra':
            print("No slides with SRA classification NPY files found")
        else:
            print("No slides with both tumor and normal annotations found")
        return
    
    if args.max_slides is not None and args.max_slides > 0:
        valid_slides = valid_slides[:args.max_slides]
    
    if args.data_type == 'prostate':
        label_pair = "tumor and benign"
    elif args.data_type == 'rectal':
        label_pair = "tumor and normal (rectal)"
    elif args.data_type == 'rectal_sra':
        label_pair = "SRA classification"
    else:
        label_pair = "tumor and normal"
    print(f"Found {len(valid_slides)} slides with {label_pair} annotations")
    
    # Identify common slides across ALL embeddings BEFORE processing
    print("\nIdentifying common slides across all embeddings...")
    embedding_slide_sets = {}
    
    for h5_root in args.h5_root:
        embedding_name = extract_embedding_name(h5_root)
        h5_root_path = Path(h5_root)
        
        # Find which slides have H5 files for this embedding
        available_h5 = set()
        for slide_path in valid_slides:
            h5_path = h5_root_path / f"{slide_path.stem}.h5"
            if h5_path.exists():
                available_h5.add(slide_path.stem)
        
        embedding_slide_sets[embedding_name] = available_h5
        print(f"  {embedding_name}: {len(available_h5)} slides available")
    
    # Find intersection
    if len(embedding_slide_sets) > 1:
        common_slide_names = set.intersection(*embedding_slide_sets.values())
        print(f"\nCommon slides across all {len(args.h5_root)} embeddings: {len(common_slide_names)}")
        
        if len(common_slide_names) == 0:
            print("ERROR: No common slides across all embeddings!")
            return
        
        # Filter valid_slides to only common ones
        valid_slides = [s for s in valid_slides if s.stem in common_slide_names]
        print(f"Processing only {len(valid_slides)} common slides")
    else:
        print(f"Only one embedding provided, using all {len(valid_slides)} slides")
    
    # Process each embedding
    all_silhouette_results = []
    embedding_data = {}  # Store for comparison
    
    for h5_root in args.h5_root:
        embedding_name = extract_embedding_name(h5_root)
        mag_x, tile_px = parse_mag_and_tile(h5_root)
        
        print(f"\n{'='*60}")
        print(f"Processing embedding: {embedding_name}")
        print(f"{'='*60}")
        
        # Collect all tiles
        features, labels, slide_indices, slide_names, coords_info, label_map = collect_tiles_from_slides(
            h5_root, svs_root, annotation_dir, valid_slides, mag_x, tile_px, embedding_name, args
        )
        
        if features is None:
            print(f"[{embedding_name}] No tiles collected, skipping")
            if getattr(args, 'data_type', None) == 'rectal_sra':
                print(f"  (rectal_sra: H5 coords must match/overlap SRA NPY grid; SRA uses 512px patches, this embedding is {tile_px}px)")
            continue
        
        # Run t-SNE on aggregated data
        print(f"[{embedding_name}] Running t-SNE on {len(features)} tiles...")
        Y = reduce_and_embed(features, seed=args.seed, pca_threshold=50,
                            tsne_perplexity=args.tsne_perplexity, tsne_iter=args.tsne_iter)
        Y01 = normalize_to_unit_square(Y)
        
        # Grid assignment - auto-size if requested
        if args.auto_grid:
            n_tiles = len(features)
            grid_size = int(np.ceil(np.sqrt(n_tiles)))
            rows = cols = grid_size
            print(f"[{embedding_name}] Auto-sized grid: {rows}x{cols} for {n_tiles} tiles")
        else:
            rows, cols = args.grid_rows, args.grid_cols
        
        # Assign to grid using selected method
        n_tiles_before = len(features)
        if args.grid_method == 'direct':
            rng = np.random.default_rng(args.seed)
            _, rc, grid_selected = assign_points_to_grid_direct(Y01, rows, cols, rng=rng)
            # Filter data to selected tiles only
            features = features[grid_selected]
            labels = labels[grid_selected]
            slide_indices = slide_indices[grid_selected]
            Y = Y[grid_selected]
            coords_info = [coords_info[i] for i in grid_selected]
            print(f"[{embedding_name}] Direct method: kept {len(features)}/{n_tiles_before} tiles ({len(features)/n_tiles_before*100:.1f}%)")
        else:  # greedy
            _, rc = assign_points_to_grid_greedy(Y01, rows, cols)
            print(f"[{embedding_name}] Greedy method: using all {len(features)} tiles")
        
        # Compute global bootstrapped metrics
        bootstrap_results = compute_global_bootstrapped_metrics(
            Y, labels, features, embedding_name, label_map,
            n_bootstrap=50, bootstrap_fraction=0.5, seed=args.seed,
            data_type=getattr(args, 'data_type', None)
        )
        all_silhouette_results.extend(bootstrap_results)

        # Store for comparison (dynamic keys from bootstrap result)
        embedding_data[embedding_name] = {}
        if bootstrap_results:
            for key in bootstrap_results[0].keys():
                if key in ('bootstrap_idx', 'embedding'):
                    continue
                vals = [r[key] for r in bootstrap_results if key in r and (not isinstance(r[key], float) or not np.isnan(r[key]))]
                if vals:
                    embedding_data[embedding_name][key] = vals
        
        # Find boundary sides
        boundary_sides = find_boundary_sides(labels, rc, rows, cols)
        
        # Extract tiles using stored coordinates
        print(f"[{embedding_name}] Extracting tiles for visualization...")
        tiles = []
        
        # Group tiles by slide for efficient extraction
        slide_tile_groups = {}  # slide_idx -> list of (tile_idx, x, y, mag_ratio)
        for i, (sidx, x, y, mag_ratio) in enumerate(coords_info):
            if sidx not in slide_tile_groups:
                slide_tile_groups[sidx] = []
            slide_tile_groups[sidx].append((i, x, y, mag_ratio))
        
        # Create placeholder list
        tiles = [None] * len(coords_info)
        
        # Extract tiles slide by slide
        for sidx in tqdm(sorted(slide_tile_groups.keys()), desc=f"[{embedding_name}] Extracting"):
            slide_path = valid_slides[sidx]
            
            # Open slide
            slide_obj = openslide.OpenSlide(str(slide_path))
            
            # Extract all tiles for this slide
            data_type = getattr(args, 'data_type', 'kidney')
            sra_dataset = getattr(args, 'sra_dataset', 'kather19crctp')
            for tile_idx, x, y, mag_ratio in slide_tile_groups[sidx]:
                try:
                    img = extract_tile(slide_obj, (int(x), int(y)), tile_px)
                    # Resize to mosaic size before drawing border so border is visible in output
                    if img.size != (args.mosaic_tile_px, args.mosaic_tile_px):
                        img = img.resize((args.mosaic_tile_px, args.mosaic_tile_px), resample=Image.BILINEAR)
                    sides = boundary_sides[tile_idx]
                    if any(sides.values()):
                        class_name = label_map[int(labels[tile_idx])]
                        if data_type in ('kidney', 'rectal'):
                            color = (230, 25, 75) if class_name == 'tumor' else (0, 130, 200)
                        elif data_type == 'prostate':
                            color = (230, 25, 75) if class_name == 'tumor' else (0, 200, 50)
                        else:
                            color = get_sra_class_color(class_name, sra_dataset)
                        border_w = min(args.border_width, max(1, args.mosaic_tile_px // 2))
                        img = add_selective_border(img, color, sides, border_width=border_w)
                    tiles[tile_idx] = img
                except Exception as e:
                    if args.debug:
                        print(f"[warn] Failed to extract tile at ({x},{y}): {e}")
                    tiles[tile_idx] = None
            
            slide_obj.close()
        
        # Create mosaic
        print(f"[{embedding_name}] Creating mosaic...")
        mosaic = stitch_mosaic(tiles, rc, rows, cols, args.mosaic_tile_px)
        
        # Save
        mosaic_path = output_dir / f"{embedding_name}_aggregated.png"
        mosaic.save(mosaic_path, format='PNG', dpi=(400, 400))
        print(f"[{embedding_name}] Saved mosaic to {mosaic_path}")
    
    # Save bootstrapped metric results
    if all_silhouette_results:
        df_sil = pd.DataFrame(all_silhouette_results)
        sil_path = output_dir / "global_metrics_bootstrap.csv"
        df_sil.to_csv(sil_path, index=False)
        print(f"\nSaved bootstrapped global metrics to {sil_path}")
        
        # Wilcoxon rank-sum tests between models (using bootstrap distributions)
        print(f"\n{'='*60}")
        print("STATISTICAL COMPARISON (Wilcoxon Rank-Sum Test)")
        print(f"{'='*60}")
        
        model_names = list(embedding_data.keys())
        comparison_results = []
        all_keys = set()
        for emb_data in embedding_data.values():
            all_keys.update(emb_data.keys())
        metrics_to_test = {}
        if 'silhouette_2d' in all_keys:
            metrics_to_test['2D_silhouette'] = 'silhouette_2d'
        for k in sorted(all_keys):
            if k.endswith('_compactness'):
                metrics_to_test[k] = k

        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                for metric_label, metric_key in metrics_to_test.items():
                    scores1 = embedding_data[model1].get(metric_key, [])
                    scores2 = embedding_data[model2].get(metric_key, [])
                    
                    if len(scores1) > 0 and len(scores2) > 0:
                        stat, p = mannwhitneyu(scores1, scores2, alternative='two-sided')
                        
                        print(f"\n{model1} vs {model2} ({metric_label}):")
                        print(f"  {model1}: mean={np.mean(scores1):.4f}, std={np.std(scores1):.4f}, n={len(scores1)}")
                        print(f"  {model2}: mean={np.mean(scores2):.4f}, std={np.std(scores2):.4f}, n={len(scores2)}")
                        print(f"  U-statistic: {stat:.2f}, p-value: {p:.6f}")
                        
                        comparison_results.append({
                            'model1': model1,
                            'model2': model2,
                            'metric': metric_label,
                            'model1_mean': np.mean(scores1),
                            'model1_std': np.std(scores1),
                            'model1_n': len(scores1),
                            'model2_mean': np.mean(scores2),
                            'model2_std': np.std(scores2),
                            'model2_n': len(scores2),
                            'U_statistic': stat,
                            'p_value': p
                        })
        
        # Save comparison results
        if comparison_results:
            df_comp = pd.DataFrame(comparison_results)
            comp_path = output_dir / "model_comparison_wilcoxon.csv"
            df_comp.to_csv(comp_path, index=False)
            print(f"\nSaved statistical comparisons to {comp_path}")
            
            # Create box plots
            print(f"\n{'='*60}")
            print("CREATING VISUALIZATIONS")
            print(f"{'='*60}")
            # Convert comparison_results list to DataFrame for easier lookup
            create_publication_boxplots(embedding_data, comparison_results, output_dir,
                                        data_type=getattr(args, 'data_type', None))
            create_publication_barplots(embedding_data, comparison_results, output_dir,
                                        data_type=getattr(args, 'data_type', None))
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()

