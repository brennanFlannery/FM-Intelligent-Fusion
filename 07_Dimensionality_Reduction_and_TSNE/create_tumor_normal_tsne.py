#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tumor vs Normal t-SNE Analysis

Creates separate t-SNE visualizations for each slide, showing tiles from tumor and normal 
annotated regions. Computes Silhouette Coefficients to quantify separation between groups.

Key features:
- Processes multiple embedding types (multiple --h5_root paths)
- One t-SNE per slide (no cross-slide aggregation)
- Samples ALL tiles that overlap with tumor or normal regions (not just centroid-based)
- Supports greedy and direct grid assignment methods
- Auto-grid sizing to fit all tiles
- Selective borders: red for tumor, blue for normal (only on edges touching other group/whitespace)
- Computes Silhouette on both 2D t-SNE and original high-dim features
- Generates annotated slide thumbnails

Outputs:
- embedding_X/slide_Y.png: t-SNE mosaic for each slide and embedding
- thumbnails/slide_Y.png: Annotated slide thumbnails (400 DPI)
- summary_silhouette.csv: Silhouette scores and statistics for all slides/embeddings

Requirements:
  numpy, scipy, scikit-learn, pandas, h5py, openslide-python, pillow, tqdm, shapely

Example:
  python create_tumor_normal_tsne.py \
      --h5_root /path/to/features_gigapath /path/to/features_uni \
      --svs_root /path/to/WSI \
      --annotation_dir /path/to/annotations \
      --output_dir /path/to/output \
      --max_slides 50 \
      --grid_rows 60 --grid_cols 60
"""

import argparse
import os
import re
import glob
import math
from pathlib import Path
from typing import List, Tuple, Dict
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
import openslide
import cv2

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.spatial import cKDTree

try:
    from shapely.geometry import Point, Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("Warning: shapely not available, falling back to slower point-in-polygon check")

# -----------------------------
# Utilities
# -----------------------------

def extract_embedding_name(h5_root: str) -> str:
    """
    Extract embedding name from path like '.../features_gigapath/...'
    Falls back to basename if pattern not found.
    """
    m = re.search(r'features[_-](\w+)', h5_root, re.IGNORECASE)
    if m:
        return m.group(1)
    # Try to find any meaningful directory name
    parts = Path(h5_root).parts
    for part in reversed(parts):
        if part and part not in ['.', '..', '/']:
            return part
    return "embedding"


def parse_mag_and_tile(path: str) -> Tuple[int, int]:
    """
    Infer magnification and tile size from path like '.../20x_512px_0px_overlap/...'
    Returns (magnification_x, tile_px). If not found, defaults to (20, 256).
    """
    m = re.search(r'(\d+)\s*x[_/\\]+(\d+)\s*px', path, re.IGNORECASE)
    if m:
        mag = int(m.group(1))
        tile_px = int(m.group(2))
        return mag, tile_px
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


def parse_polygons_from_xml(xml_path: Path) -> List[np.ndarray]:
    """Parse annotation XML and return list of polygons as Nx2 arrays."""
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


def point_in_polygon_simple(px: float, py: float, poly: np.ndarray) -> bool:
    """Simple ray-casting algorithm for point-in-polygon test."""
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


def is_point_in_any_polygon(px: float, py: float, polygons: List[np.ndarray]) -> bool:
    """Check if point is inside any of the given polygons."""
    if HAS_SHAPELY:
        point = Point(px, py)
        for poly_coords in polygons:
            poly = Polygon(poly_coords)
            if poly.contains(point):
                return True
        return False
    else:
        # Fallback to simple implementation
        for poly in polygons:
            if point_in_polygon_simple(px, py, poly):
                return True
        return False


def get_tile_bounds(coord: np.ndarray, tile_px: int, mag_ratio: float) -> Tuple[float, float, float, float]:
    """
    Get bounding box of a tile.
    coord: top-left corner at level-0
    tile_px: tile size at extraction magnification
    mag_ratio: base_mag / extraction_mag
    Returns: (x_min, y_min, x_max, y_max)
    """
    tile_full = tile_px * mag_ratio
    x_min = coord[0]
    y_min = coord[1]
    x_max = coord[0] + tile_full
    y_max = coord[1] + tile_full
    return x_min, y_min, x_max, y_max


def tile_overlaps_any_polygon(tile_bounds: Tuple[float, float, float, float], 
                              polygons: List[np.ndarray]) -> bool:
    """
    Check if a tile (rectangle) overlaps with any of the given polygons.
    tile_bounds: (x_min, y_min, x_max, y_max)
    polygons: List of Nx2 arrays defining polygon vertices
    """
    x_min, y_min, x_max, y_max = tile_bounds
    
    if HAS_SHAPELY:
        from shapely.geometry import box
        tile_box = box(x_min, y_min, x_max, y_max)
        
        for poly_coords in polygons:
            try:
                poly = Polygon(poly_coords)
                if tile_box.intersects(poly):
                    return True
            except Exception:
                # If polygon is invalid, skip it
                continue
        return False
    else:
        # Fallback: check if any polygon point is inside tile OR tile corners inside polygon
        # Also check if any polygon edge intersects tile
        
        # Quick check: tile corners in any polygon
        tile_corners = [
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max)
        ]
        
        for poly in polygons:
            # Check if any tile corner is in polygon
            for cx, cy in tile_corners:
                if point_in_polygon_simple(cx, cy, poly):
                    return True
            
            # Check if any polygon point is in tile
            for px, py in poly:
                if x_min <= px <= x_max and y_min <= py <= y_max:
                    return True
            
            # Check bounding box overlap as a quick heuristic
            poly_x_min, poly_y_min = poly.min(axis=0)
            poly_x_max, poly_y_max = poly.max(axis=0)
            
            # Check if bounding boxes overlap
            if not (x_max < poly_x_min or x_min > poly_x_max or 
                    y_max < poly_y_min or y_min > poly_y_max):
                # Bounding boxes overlap, likely intersects
                return True
        
        return False


def reduce_and_embed(features: np.ndarray,
                     seed: int = 42,
                     pca_threshold: int = 50,
                     tsne_perplexity: float = 30.0,
                     tsne_iter: int = 1000) -> np.ndarray:
    """Reduce dimensionality with PCA if needed, then run t-SNE."""
    D = features.shape[1]
    X = features
    if D > pca_threshold:
        pca = PCA(n_components=pca_threshold, random_state=seed)
        X = pca.fit_transform(features)
    
    # Adjust perplexity if needed
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


def assign_points_to_grid_greedy(points01: np.ndarray, rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy nearest-open-cell assignment with spillover search (no overlap).
    Returns:
      cell_indices: (N,) linear indices into grid cells
      cell_rc: (N,2) row, col per point
    """
    N = points01.shape[0]
    # Convert to integer cell indices
    r = np.clip((points01[:, 1] * rows).astype(int), 0, rows - 1)
    c = np.clip((points01[:, 0] * cols).astype(int), 0, cols - 1)

    # Sort points by local density (sparser first helps reduce conflicts)
    # Estimate density via kNN distance
    tree = cKDTree(points01)
    dists, _ = tree.query(points01, k=min(8, max(2, N)))
    local_score = dists[:, 1:].mean(axis=1)  # higher => sparser
    order = np.argsort(-local_score)  # descending sparsity

    occupied = np.full((rows, cols), False, dtype=bool)
    out_rc = np.zeros((N, 2), dtype=int)

    for idx in order:
        rr, cc = r[idx], c[idx]
        if not occupied[rr, cc]:
            occupied[rr, cc] = True
            out_rc[idx] = (rr, cc)
            continue
        # Spiral search outward for nearest free cell
        found = False
        max_radius = max(rows, cols)
        for rad in range(1, max_radius):
            rmin, rmax = max(0, rr - rad), min(rows - 1, rr + rad)
            cmin, cmax = max(0, cc - rad), min(cols - 1, cc + rad)
            # Check border ring
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
            # Fallback: place at original even if occupied (should be rare)
            out_rc[idx] = (rr, cc)

    cell_indices = out_rc[:, 0] * cols + out_rc[:, 1]
    return cell_indices, out_rc


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


def find_boundary_sides(labels: np.ndarray, rc: np.ndarray, rows: int, cols: int) -> List[Dict[str, bool]]:
    """
    Find which sides of each tile border a different group or empty space.
    
    Args:
        labels: (N,) array of group labels (0=tumor, 1=normal)
        rc: (N, 2) array of (row, col) positions
        rows, cols: grid dimensions
    
    Returns:
        List of N dicts with keys 'top', 'bottom', 'left', 'right'
    """
    N = len(labels)
    grid_label = np.full((rows, cols), -1, dtype=int)  # -1 = empty
    
    for i in range(N):
        r, c = rc[i]
        grid_label[r, c] = labels[i]
    
    border_sides = []
    for i in range(N):
        r, c = rc[i]
        my_label = labels[i]
        sides = {'top': False, 'bottom': False, 'left': False, 'right': False}
        
        # Check neighbors
        if r > 0:
            neighbor = grid_label[r-1, c]
            if neighbor == -1 or neighbor != my_label:
                sides['top'] = True
        else:
            sides['top'] = True  # Edge of grid
            
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
    """Add colored borders only on specific sides of the image."""
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
                  mosaic_tile_px: int, bg_color=(255, 255, 255), debug=False) -> Image.Image:
    """Stitch tiles into a mosaic grid."""
    H = rows * mosaic_tile_px
    W = cols * mosaic_tile_px
    canvas = Image.new('RGB', (W, H), color=bg_color)
    
    pasted_count = 0
    skipped_none = 0
    
    for i, (img, (r, c)) in enumerate(zip(tiles, rc)):
        if img is None:
            skipped_none += 1
            continue
        if img.size != (mosaic_tile_px, mosaic_tile_px):
            img = img.resize((mosaic_tile_px, mosaic_tile_px), resample=Image.BILINEAR)
        
        # Paste position
        paste_x = c * mosaic_tile_px
        paste_y = r * mosaic_tile_px
        
        # Check bounds
        if paste_x >= W or paste_y >= H or paste_x < 0 or paste_y < 0:
            if debug:
                print(f"    Tile {i} at grid ({r},{c}) -> pixel ({paste_x},{paste_y}) OUT OF CANVAS BOUNDS ({W}x{H})")
            continue
            
        canvas.paste(img, (paste_x, paste_y))
        pasted_count += 1
    
    if debug or pasted_count < len(tiles) // 2:
        print(f"    Stitch summary: {pasted_count} tiles pasted, {skipped_none} None tiles skipped, canvas={W}x{H}")
    
    return canvas


def create_occupancy_grid(rc: np.ndarray, labels: np.ndarray, rows: int, cols: int, 
                         cell_size: int = 50) -> Image.Image:
    """
    Create a visualization showing occupancy count and group distribution per grid cell.
    Each cell shows: total count (and breakdown by tumor/normal if mixed).
    """
    from PIL import ImageFont
    
    H = rows * cell_size
    W = cols * cell_size
    canvas = Image.new('RGB', (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=max(8, cell_size // 4))
    except:
        font = ImageFont.load_default()
    
    # Count occupancy per cell
    grid_counts = {}  # (r, c) -> {'tumor': count, 'normal': count}
    
    for (r, c), label in zip(rc, labels):
        key = (r, c)
        if key not in grid_counts:
            grid_counts[key] = {'tumor': 0, 'normal': 0}
        if label == 0:
            grid_counts[key]['tumor'] += 1
        else:
            grid_counts[key]['normal'] += 1
    
    # Draw grid and counts
    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            
            # Draw cell border
            draw.rectangle([x0, y0, x1, y1], outline=(200, 200, 200), width=1)
            
            key = (r, c)
            if key in grid_counts:
                tumor_count = grid_counts[key]['tumor']
                normal_count = grid_counts[key]['normal']
                total = tumor_count + normal_count
                
                # Color cell based on majority class
                if tumor_count > normal_count:
                    # More tumor - light red
                    bg_color = (255, 200, 200)
                elif normal_count > tumor_count:
                    # More normal - light blue
                    bg_color = (200, 220, 255)
                else:
                    # Equal - light purple
                    bg_color = (230, 200, 230)
                
                # Fill with background color
                draw.rectangle([x0+1, y0+1, x1-1, y1-1], fill=bg_color)
                
                # Draw text showing counts - always show numbers
                if tumor_count > 0 and normal_count > 0:
                    # Mixed: show total and breakdown
                    text = f"{total}\nT{tumor_count}\nN{normal_count}"
                    text_color = (0, 0, 0)
                elif tumor_count > 0:
                    # Only tumor: show count
                    text = f"{tumor_count}"
                    text_color = (150, 0, 0)
                else:
                    # Only normal: show count
                    text = f"{normal_count}"
                    text_color = (0, 0, 150)
                
                # Center text in cell
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = x0 + (cell_size - text_w) // 2
                text_y = y0 + (cell_size - text_h) // 2
                
                draw.text((text_x, text_y), text, fill=text_color, font=font)
    
    return canvas


def create_annotated_thumbnail(slide_path: Path, tumor_polys: List[np.ndarray], 
                               normal_polys: List[np.ndarray], target_dim: int = 2000) -> Image.Image:
    """Create a thumbnail of the slide with annotation outlines."""
    slide = openslide.OpenSlide(str(slide_path))
    W_full, H_full = slide.dimensions
    
    # Get thumbnail
    scale = target_dim / max(W_full, H_full)
    thumb = slide.get_thumbnail((int(W_full * scale), int(H_full * scale)))
    slide.close()
    
    thumb_rgb = thumb.convert('RGB')
    draw = ImageDraw.Draw(thumb_rgb)
    
    # Draw tumor polygons in red
    for poly in tumor_polys:
        scaled_poly = [(int(x * scale), int(y * scale)) for x, y in poly]
        draw.polygon(scaled_poly, outline=(230, 25, 75), width=3)
    
    # Draw normal polygons in blue
    for poly in normal_polys:
        scaled_poly = [(int(x * scale), int(y * scale)) for x, y in poly]
        draw.polygon(scaled_poly, outline=(0, 130, 200), width=3)
    
    return thumb_rgb


# -----------------------------
# Main pipeline
# -----------------------------

def process_single_slide(slide_path: Path, h5_root: str, annotation_dir: Path,
                        args, mag_x: int, tile_px: int, embedding_name: str,
                        output_dir: Path, rng) -> Dict:
    """Process a single slide for one embedding type."""
    
    slide_stem = slide_path.stem
    
    # Load annotations
    tumor_xml = find_tumor_xml(annotation_dir, slide_stem)
    normal_xml = find_normal_xml(annotation_dir, slide_stem)
    
    if tumor_xml is None or normal_xml is None:
        return None
    
    tumor_polys = parse_polygons_from_xml(tumor_xml)
    normal_polys = parse_polygons_from_xml(normal_xml)
    
    if not tumor_polys or not normal_polys:
        return None
    
    # Load features and coords
    h5_path = Path(h5_root) / f"{slide_stem}.h5"
    if not h5_path.exists():
        if args.debug:
            print(f"[skip] {slide_stem}: H5 file not found at {h5_path}")
        return None
    
    try:
        with h5py.File(h5_path, 'r') as f:
            features = f['features'][:]
            coords = f['coords'][:]
    except Exception as e:
        if args.debug:
            print(f"[skip] {slide_stem}: Error loading H5 file: {e}")
        return None
    
    if features.shape[0] != coords.shape[0]:
        if args.debug:
            print(f"[skip] {slide_stem}: Feature/coord count mismatch: {features.shape[0]} vs {coords.shape[0]}")
        return None
    
    # Get magnification info
    slide = openslide.OpenSlide(str(slide_path))
    base_mag = float(slide.properties.get('openslide.objective-power', mag_x))
    mag_ratio = base_mag / mag_x
    slide.close()
    
    # Classify tiles as tumor or normal based on ANY overlap with annotations
    tumor_indices = []
    normal_indices = []
    
    for i, coord in enumerate(coords):
        tile_bounds = get_tile_bounds(coord, tile_px, mag_ratio)
        
        # Check overlap with tumor regions
        overlaps_tumor = tile_overlaps_any_polygon(tile_bounds, tumor_polys)
        # Check overlap with normal regions
        overlaps_normal = tile_overlaps_any_polygon(tile_bounds, normal_polys)
        
        # Priority: if overlaps both, classify as tumor
        # Otherwise classify by whichever it overlaps
        if overlaps_tumor:
            tumor_indices.append(i)
        elif overlaps_normal:
            normal_indices.append(i)
    
    tumor_indices = np.array(tumor_indices, dtype=int)
    normal_indices = np.array(normal_indices, dtype=int)
    
    if len(tumor_indices) == 0 or len(normal_indices) == 0:
        if args.debug:
            print(f"[skip] {slide_stem}: no tiles in tumor ({len(tumor_indices)}) or normal ({len(normal_indices)})")
        return None
    
    # Combine all selected tiles
    selected_indices = np.concatenate([tumor_indices, normal_indices])
    labels = np.concatenate([np.zeros(len(tumor_indices), dtype=int),
                            np.ones(len(normal_indices), dtype=int)])
    
    selected_features = features[selected_indices]
    selected_coords = coords[selected_indices]
    
    # Always print tile counts
    print(f"  [{slide_stem}] Found {len(tumor_indices)} tumor tiles, {len(normal_indices)} normal tiles (total: {len(selected_indices)})")
    
    # Check if we have enough samples for t-SNE
    if len(selected_indices) < 30:
        if args.debug:
            print(f"[skip] {slide_stem}: too few tiles ({len(selected_indices)}) for t-SNE")
        return None
    
    # Run t-SNE
    Y = reduce_and_embed(selected_features,
                        seed=args.seed,
                        pca_threshold=50,
                        tsne_perplexity=args.tsne_perplexity,
                        tsne_iter=args.tsne_iter)
    Y01 = normalize_to_unit_square(Y)
    
    # Grid assignment - auto-size if requested
    if args.auto_grid:
        # Size grid to fit all tiles (square grid)
        n_tiles = len(selected_indices)
        grid_size = int(np.ceil(np.sqrt(n_tiles)))
        rows = cols = grid_size
        if args.debug:
            print(f"  Auto-sized grid: {rows}x{cols} for {n_tiles} tiles")
    else:
        rows, cols = args.grid_rows, args.grid_cols
    
    max_cells = rows * cols
    n_tiles_before = len(selected_indices)
    
    # Map ALL t-SNE points to grid cells BEFORE selection (for pre-selection occupancy grid)
    r_all = np.clip((Y01[:, 1] * rows).astype(int), 0, rows - 1)
    c_all = np.clip((Y01[:, 0] * cols).astype(int), 0, cols - 1)
    rc_all_before_selection = np.column_stack([r_all, c_all])
    labels_all_before_selection = labels.copy()
    
    # Warn if grid is much larger than needed or much smaller
    if not args.auto_grid and n_tiles_before < max_cells * 0.3:
        print(f"  [WARNING] {slide_stem}: Grid is large ({rows}x{cols}={max_cells} cells) for only {n_tiles_before} tiles. Consider using --auto_grid or smaller --grid_rows/--grid_cols")
    elif not args.auto_grid and n_tiles_before > max_cells and args.grid_method == 'greedy':
        print(f"  [WARNING] {slide_stem}: {n_tiles_before} tiles exceeds grid capacity ({max_cells}). Tiles will be randomly sampled. Consider using --auto_grid")
    
    if args.grid_method == 'direct':
        # Direct method handles its own selection
        _, rc, grid_selected = assign_points_to_grid_direct(Y01, rows, cols, rng=rng)
        # Filter to only grid-selected tiles
        selected_indices = selected_indices[grid_selected]
        selected_coords = selected_coords[grid_selected]
        labels = labels[grid_selected]
        selected_features = selected_features[grid_selected]
        Y_selected = Y[grid_selected]
        n_tumor_final = int((labels == 0).sum())
        n_normal_final = int((labels == 1).sum())
        print(f"  [{slide_stem}] After grid (direct): {n_tumor_final} tumor, {n_normal_final} normal tiles plotted (kept {len(selected_indices)}/{n_tiles_before}, {len(selected_indices)/n_tiles_before*100:.1f}%)")
    else:  # greedy
        # Greedy assigns all points, reduce if needed
        if len(selected_indices) > max_cells:
            keep = rng.choice(len(selected_indices), size=max_cells, replace=False)
            selected_indices = selected_indices[keep]
            selected_coords = selected_coords[keep]
            labels = labels[keep]
            selected_features = selected_features[keep]
            Y01 = Y01[keep]
            Y_selected = Y[keep]
            n_tumor_final = int((labels == 0).sum())
            n_normal_final = int((labels == 1).sum())
            print(f"  [{slide_stem}] After grid (greedy): {n_tumor_final} tumor, {n_normal_final} normal tiles plotted (reduced from {n_tiles_before} to grid capacity)")
        else:
            Y_selected = Y
            n_tumor_final = int((labels == 0).sum())
            n_normal_final = int((labels == 1).sum())
            print(f"  [{slide_stem}] After grid (greedy): {n_tumor_final} tumor, {n_normal_final} normal tiles plotted (all tiles used)")
        _, rc = assign_points_to_grid_greedy(Y01, rows, cols)
    
    # Compute Silhouette scores
    if len(np.unique(labels)) == 2 and len(labels) > 2:
        sil_2d = silhouette_score(Y_selected, labels, metric='euclidean')
        # For high-dim, only compute if not too large (memory considerations)
        if selected_features.shape[0] < 10000 and selected_features.shape[1] < 2048:
            sil_highdim = silhouette_score(selected_features, labels, metric='euclidean')
        else:
            sil_highdim = np.nan
    else:
        sil_2d = np.nan
        sil_highdim = np.nan
    
    # Find boundary sides
    boundary_sides = find_boundary_sides(labels, rc, rows, cols)
    
    # Extract tiles and add borders
    slide = openslide.OpenSlide(str(slide_path))
    W_slide, H_slide = slide.dimensions
    tiles = []
    extraction_failures = 0
    out_of_bounds = 0
    
    for i in range(len(selected_indices)):
        x, y = selected_coords[i]
        
        # Check if coordinates are within slide bounds
        tile_full = tile_px * mag_ratio
        if x < 0 or y < 0 or x + tile_full > W_slide or y + tile_full > H_slide:
            out_of_bounds += 1
            if args.debug:
                print(f"[warn] {slide_stem}: tile at ({x},{y}) is out of bounds (slide: {W_slide}x{H_slide}, tile size: {tile_full})")
            tiles.append(None)
            continue
        
        try:
            img = extract_tile(slide, (int(x), int(y)), tile_px)
            
            # Add selective border
            sides = boundary_sides[i]
            if any(sides.values()):
                if labels[i] == 0:  # Tumor
                    color = (230, 25, 75)  # Red
                else:  # Normal
                    color = (0, 130, 200)  # Blue
                img = add_selective_border(img, color, sides, border_width=args.border_width)
        except Exception as e:
            extraction_failures += 1
            if args.debug:
                print(f"[warn] extract failed {slide_stem} @ ({x},{y}): {e}")
            img = None
        tiles.append(img)
    
    slide.close()
    
    # Count successful extractions
    successful_tiles = sum(1 for t in tiles if t is not None)
    total_failures = extraction_failures + out_of_bounds
    
    if total_failures > 0:
        print(f"  [WARNING] {slide_stem}: {total_failures}/{len(selected_indices)} tiles failed!")
        if out_of_bounds > 0:
            print(f"    - {out_of_bounds} tiles out of bounds")
        if extraction_failures > 0:
            print(f"    - {extraction_failures} tiles extraction errors")
        print(f"  [WARNING] Only {successful_tiles} tiles will appear in mosaic (expected {len(selected_indices)})")
    
    if args.debug:
        print(f"  [{slide_stem}] Tile extraction: {successful_tiles} succeeded, {out_of_bounds} out of bounds, {extraction_failures} extraction errors")
    
    # Diagnostic: Check tiles list vs rc array alignment
    print(f"  [{slide_stem}] Pre-mosaic check: {len(tiles)} tiles, {len(rc)} rc positions, {len(labels)} labels")
    non_none_tiles = sum(1 for t in tiles if t is not None)
    print(f"  [{slide_stem}] Tiles: {non_none_tiles} non-None, {len(tiles) - non_none_tiles} None")
    
    # Sample check: Are tiles valid images?
    if non_none_tiles > 0:
        sample_tiles = [t for t in tiles if t is not None][:5]
        white_tile_count = 0
        for t in [t for t in tiles if t is not None]:
            arr = np.array(t)
            if arr.mean() > 240:  # Mostly white
                white_tile_count += 1
        
        if args.debug:
            for idx, t in enumerate(sample_tiles):
                arr = np.array(t)
                mean_val = arr.mean()
                std_val = arr.std()
                print(f"    Sample tile {idx}: size={t.size}, mean={mean_val:.1f}, std={std_val:.1f}")
        
        if white_tile_count > non_none_tiles * 0.5:
            print(f"  [WARNING] {slide_stem}: {white_tile_count}/{non_none_tiles} tiles are mostly white (may be background/empty)")
    
    # Check for duplicate rc positions
    rc_tuples = [tuple(r) for r in rc]
    unique_rc = len(set(rc_tuples))
    if unique_rc < len(rc):
        print(f"  [WARNING] {slide_stem}: Duplicate grid positions! {len(rc)} tiles but only {unique_rc} unique positions")
        print(f"  [WARNING] Tiles will overwrite each other in the mosaic!")
    
    # Check if any rc values are outside grid bounds
    r_vals, c_vals = rc[:, 0], rc[:, 1]
    out_of_grid = ((r_vals < 0) | (r_vals >= rows) | (c_vals < 0) | (c_vals >= cols)).sum()
    if out_of_grid > 0:
        print(f"  [WARNING] {slide_stem}: {out_of_grid} tiles have grid positions outside bounds (grid is {rows}x{cols})")
        if args.debug:
            bad_indices = np.where((r_vals < 0) | (r_vals >= rows) | (c_vals < 0) | (c_vals >= cols))[0]
            for bidx in bad_indices[:5]:  # Show first 5
                print(f"    Tile {bidx}: grid position ({r_vals[bidx]}, {c_vals[bidx]}) is outside grid")
    
    # Create mosaic
    mosaic = stitch_mosaic(tiles, rc, rows, cols, args.mosaic_tile_px, bg_color=(255, 255, 255), debug=args.debug)
    
    # Verify mosaic content
    mosaic_arr = np.array(mosaic)
    white_pixels = ((mosaic_arr == 255).all(axis=2)).sum()
    total_pixels = mosaic_arr.shape[0] * mosaic_arr.shape[1]
    non_white_percent = 100 * (1 - white_pixels / total_pixels)
    print(f"  [{slide_stem}] Mosaic content: {non_white_percent:.1f}% non-white pixels (size: {mosaic.size})")
    
    # Save mosaic
    emb_dir = output_dir / embedding_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    mosaic_path = emb_dir / f"{slide_stem}.png"
    mosaic.save(mosaic_path, format='PNG', dpi=(400, 400))
    print(f"  [{slide_stem}] Saved mosaic to {mosaic_path}")
    
    # Create occupancy grid showing PRE-SELECTION mapping (before direct method filters)
    # This shows how many tiles originally mapped to each grid cell
    n_tumor_presel = int((labels_all_before_selection == 0).sum())
    n_normal_presel = int((labels_all_before_selection == 1).sum())
    print(f"  [{slide_stem}] Pre-selection occupancy input: {n_tumor_presel} tumor, {n_normal_presel} normal (total: {len(labels_all_before_selection)})")
    
    occupancy = create_occupancy_grid(rc_all_before_selection, labels_all_before_selection, rows, cols, cell_size=50)
    occupancy_path = emb_dir / f"{slide_stem}_occupancy_preselection.png"
    occupancy.save(occupancy_path, format='PNG', dpi=(150, 150))
    print(f"  [{slide_stem}] Saved pre-selection occupancy grid to {occupancy_path}")
    
    # Also create post-selection occupancy grid (only successfully extracted tiles)
    successful_mask = np.array([t is not None for t in tiles])
    rc_successful = rc[successful_mask]
    labels_successful = labels[successful_mask]
    
    occupancy_post = create_occupancy_grid(rc_successful, labels_successful, rows, cols, cell_size=50)
    occupancy_post_path = emb_dir / f"{slide_stem}_occupancy_postselection.png"
    occupancy_post.save(occupancy_post_path, format='PNG', dpi=(150, 150))
    print(f"  [{slide_stem}] Saved post-selection occupancy grid to {occupancy_post_path}")
    
    # Return statistics
    return {
        'slide': slide_stem,
        'embedding': embedding_name,
        'n_tumor_tiles': int((labels == 0).sum()),
        'n_normal_tiles': int((labels == 1).sum()),
        'tumor_normal_ratio': float((labels == 0).sum() / (labels == 1).sum()) if (labels == 1).sum() > 0 else np.nan,
        'silhouette_2d': float(sil_2d),
        'silhouette_highdim': float(sil_highdim),
    }


def main():
    ap = argparse.ArgumentParser(description="Tumor vs Normal t-SNE analysis per slide.")
    ap.add_argument('--h5_root', required=True, nargs='+', 
                    help='Root folder(s) containing per-slide H5 files (multiple for different embeddings).')
    ap.add_argument('--svs_root', required=True, help='Root folder containing original WSI files.')
    ap.add_argument('--annotation_dir', required=True, help='Directory containing tumor and normal annotation XMLs.')
    ap.add_argument('--output_dir', required=True, help='Output directory.')
    ap.add_argument('--max_slides', type=int, default=None, help='Maximum number of slides to process.')
    ap.add_argument('--grid_rows', type=int, default=60, help='Grid rows for mosaic (ignored if --auto_grid is used).')
    ap.add_argument('--grid_cols', type=int, default=60, help='Grid cols for mosaic (ignored if --auto_grid is used).')
    ap.add_argument('--auto_grid', action='store_true', 
                    help='Automatically size grid to fit all tiles (square grid based on sqrt of tile count).')
    ap.add_argument('--grid_method', choices=['greedy', 'direct'], default='greedy',
                    help='Grid assignment method: greedy (spiral search, all tiles) or direct (preserve t-SNE, pick centroid on collision).')
    ap.add_argument('--mosaic_tile_px', type=int, default=64, help='Tile size in final mosaic (downsampled).')
    ap.add_argument('--border_width', type=int, default=3, help='Width of colored borders in pixels.')
    ap.add_argument('--seed', type=int, default=42, help='Random seed.')
    ap.add_argument('--tsne_perplexity', type=float, default=30.0, help='t-SNE perplexity.')
    ap.add_argument('--tsne_iter', type=int, default=2000, help='t-SNE iterations.')
    ap.add_argument('--thumbnail_size', type=int, default=2000, help='Longest dimension for annotated thumbnails.')
    ap.add_argument('--debug', action='store_true', help='Verbose debug prints.')
    args = ap.parse_args()
    
    svs_root = Path(args.svs_root)
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(args.seed)
    
    # Get all slides with both tumor and normal annotations
    slides = sorted(svs_root.glob('*.svs'))
    if not slides:
        print("No SVS files found")
        return
    
    valid_slides = []
    for slide_path in slides:
        tumor_xml = find_tumor_xml(annotation_dir, slide_path.stem)
        normal_xml = find_normal_xml(annotation_dir, slide_path.stem)
        if tumor_xml and normal_xml:
            valid_slides.append(slide_path)
    
    if not valid_slides:
        print("No slides with both tumor and normal annotations found")
        return
    
    if args.max_slides is not None and args.max_slides > 0:
        valid_slides = valid_slides[:args.max_slides]
    
    print(f"Found {len(valid_slides)} slides with both tumor and normal annotations")
    
    # Process each embedding type
    all_results = []
    
    for h5_root in args.h5_root:
        embedding_name = extract_embedding_name(h5_root)
        mag_x, tile_px = parse_mag_and_tile(h5_root)
        
        print(f"\n{'='*60}")
        print(f"Processing embedding: {embedding_name}")
        print(f"  Path: {h5_root}")
        print(f"  Inferred: {mag_x}x, {tile_px}px tiles")
        print(f"{'='*60}\n")
        
        for slide_path in tqdm(valid_slides, desc=f"[{embedding_name}]"):
            try:
                result = process_single_slide(
                    slide_path, h5_root, annotation_dir,
                    args, mag_x, tile_px, embedding_name,
                    output_dir, rng
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                if args.debug:
                    import traceback
                    print(f"\n[ERROR] {slide_path.stem}: {e}")
                    traceback.print_exc()
                else:
                    print(f"\n[ERROR] {slide_path.stem}: {e}")
    
    # Create annotated thumbnails
    print("\nCreating annotated slide thumbnails...")
    thumb_dir = output_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    
    for slide_path in tqdm(valid_slides, desc="[thumbnails]"):
        tumor_xml = find_tumor_xml(annotation_dir, slide_path.stem)
        normal_xml = find_normal_xml(annotation_dir, slide_path.stem)
        
        tumor_polys = parse_polygons_from_xml(tumor_xml) if tumor_xml else []
        normal_polys = parse_polygons_from_xml(normal_xml) if normal_xml else []
        
        if tumor_polys or normal_polys:
            thumb = create_annotated_thumbnail(slide_path, tumor_polys, normal_polys, 
                                              target_dim=args.thumbnail_size)
            thumb_path = thumb_dir / f"{slide_path.stem}.png"
            thumb.save(thumb_path, format='PNG', dpi=(400, 400))
    
    # Save summary CSV
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = output_dir / "summary_silhouette.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        for emb in df['embedding'].unique():
            emb_df = df[df['embedding'] == emb]
            print(f"\n{emb}:")
            print(f"  Slides processed: {len(emb_df)}")
            print(f"  Avg tumor tiles: {emb_df['n_tumor_tiles'].mean():.1f} ± {emb_df['n_tumor_tiles'].std():.1f}")
            print(f"  Avg normal tiles: {emb_df['n_normal_tiles'].mean():.1f} ± {emb_df['n_normal_tiles'].std():.1f}")
            sil_2d_valid = emb_df['silhouette_2d'].dropna()
            if len(sil_2d_valid) > 0:
                print(f"  Silhouette (2D): {sil_2d_valid.mean():.4f} ± {sil_2d_valid.std():.4f} (n={len(sil_2d_valid)})")
            sil_hd_valid = emb_df['silhouette_highdim'].dropna()
            if len(sil_hd_valid) > 0:
                print(f"  Silhouette (HD): {sil_hd_valid.mean():.4f} ± {sil_hd_valid.std():.4f} (n={len(sil_hd_valid)})")
    else:
        print("\nNo results to save!")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print("\nOutput files:")
    print(f"  - t-SNE mosaics: {output_dir}/{{embedding}}/{{slide}}.png")
    print(f"  - Pre-selection occupancy: {output_dir}/{{embedding}}/{{slide}}_occupancy_preselection.png")
    print(f"  - Post-selection occupancy: {output_dir}/{{embedding}}/{{slide}}_occupancy_postselection.png")
    print(f"  - Slide thumbnails: {output_dir}/thumbnails/{{slide}}.png")
    print(f"  - Summary CSV: {output_dir}/summary_silhouette.csv")
    print("\nOccupancy grids explained:")
    print("  PRE-selection: Shows ALL tiles mapped to grid BEFORE filtering (reveals collisions)")
    print("  POST-selection: Shows only tiles that made it to final mosaic")
    print("  - Light red cells = tumor tiles (number = count)")
    print("  - Light blue cells = normal tiles (number = count)")
    print("  - Light purple cells = mixed (shows total, T#, N#)")
    print("  - Numbers > 1 in pre-selection = multiple tiles competing for same cell")


if __name__ == '__main__':
    main()

