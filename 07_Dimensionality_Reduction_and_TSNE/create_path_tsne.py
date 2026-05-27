#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gridified t-SNE mosaic of pathology tiles.

- Loads per-slide H5 files that contain:
  - features: (N_i, D)
  - coords:   (N_i, 2)  # top-left pixel coords at level-0 of the SVS
- Samples tiles (location-aware) from up to --max_slides slides.
- Reduces features (PCA->50 if D>50), runs t-SNE to 2D.
- Optional K-means clustering with colored borders on edge-sides that touch different clusters.
- Maps t-SNE points onto a KxK grid.
  - Method 'greedy' (default): nearest-open-cell assignment with spillover search (no overlap).
  - Method 'hungarian': SciPy optimal assignment on a selected subset of grid cells (no overlap).
  - Method 'direct': preserves exact t-SNE distribution; on collision, picks centroid tile (allows empty cells).
- Extracts tile images from matching .svs files and stitches a single PNG (400 DPI).

Requirements:
  numpy, scipy, scikit-learn, pandas (optional), h5py, openslide-python, pillow, tqdm

Example:
  python create_path_tsne.py \
      --h5_root /scratch/pioneer/users/bxf169/KidneyPathologyData/FM_Outputs/20x_512px_0px_overlap/features_gigapath \
      --svs_root /scratch/pioneer/users/bxf169/KidneyPathologyData/WSI \
      --output_png /scratch/pioneer/users/bxf169/figs/tsne_grid.png \
      --max_slides 200 \
      --max_tiles 3000 \
      --grid_rows 60 --grid_cols 60 \
      --grid_method direct \
      --n_clusters 5 \
      --border_width 3
"""

import argparse
import os
import re
import glob
import math
import random
from typing import List, Tuple, Dict

import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
import openslide

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# -----------------------------
# Utilities
# -----------------------------

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
    # Try alternate pattern with separators
    m2 = re.search(r'(\d+)\s*x.*?_(\d+)\s*px', path, re.IGNORECASE)
    if m2:
        return int(m2.group(1)), int(m2.group(2))
    return 20, 256


def list_h5_files(root: str, max_slides: int) -> List[str]:
    paths = sorted(glob.glob(os.path.join(root, '**', '*.h5'), recursive=True))
    if max_slides is not None and max_slides > 0:
        paths = paths[:max_slides]
    return paths


def basename_no_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def find_matching_svs(basename: str, svs_root: str) -> str:
    candidates = glob.glob(os.path.join(svs_root, '**', basename + '.svs'), recursive=True)
    if candidates:
        return candidates[0]
    # Try common extensions
    for ext in ['.tif', '.tiff', '.ndpi', '.svslide', '.mrxs']:
        candidates = glob.glob(os.path.join(svs_root, '**', basename + ext), recursive=True)
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Matching WSI not found for {basename} in {svs_root}")


def load_first_feature_dim(h5_path: str) -> int:
    with h5py.File(h5_path, 'r') as f:
        feats = f['features']
        return int(feats.shape[1])


def sample_tiles_by_location(coords: np.ndarray,
                             per_slide_max: int,
                             bins_xy: Tuple[int, int] = (32, 32),
                             rng: np.random.Generator = None) -> np.ndarray:
    """
    Location-aware subsampling: bucket coords into a 2D grid and pick at most one per bin,
    sweeping until reaching per_slide_max or exhausting bins.
    Returns selected indices relative to this slide's tiles.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = coords.shape[0]
    if per_slide_max >= n:
        return np.arange(n)

    # Normalize coords to [0,1]^2
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    span = np.maximum(mx - mn, 1.0)
    norm = (coords - mn) / span

    bx, by = bins_xy
    bins = {}
    order = rng.permutation(n)
    keep = []
    for idx in order:
        cx = int(min(bx - 1, max(0, math.floor(norm[idx, 0] * bx))))
        cy = int(min(by - 1, max(0, math.floor(norm[idx, 1] * by))))
        key = (cx, cy)
        if key not in bins:
            bins[key] = idx
            keep.append(idx)
            if len(keep) >= per_slide_max:
                break
    return np.array(keep, dtype=int)


def reduce_and_embed(features: np.ndarray,
                     seed: int = 42,
                     pca_threshold: int = 50,
                     tsne_perplexity: float = 30.0,
                     tsne_iter: int = 1000) -> np.ndarray:
    D = features.shape[1]
    X = features
    if D > pca_threshold:
        pca = PCA(n_components=pca_threshold, random_state=seed)
        X = pca.fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, learning_rate='auto',
                n_iter=tsne_iter, init='pca', random_state=seed)
    Y = tsne.fit_transform(X)
    return Y


def normalize_to_unit_square(X: np.ndarray) -> np.ndarray:
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    span = np.maximum(mx - mn, 1e-12)
    return (X - mn) / span


def make_grid_centers(rows: int, cols: int) -> np.ndarray:
    """
    Centers normalized to [0,1]x[0,1].
    """
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    cx = (xx + 0.5) / cols
    cy = (yy + 0.5) / rows
    centers = np.stack([cx, cy], axis=-1).reshape(-1, 2)
    return centers


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


def assign_points_to_grid_direct(points01: np.ndarray, rows: int, cols: int, rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct assignment: map each point to its natural grid cell.
    When multiple points map to the same cell, choose the one closest to the cell center.
    If only 2 points, randomly choose one.
    Returns one tile per occupied cell, preserving exact t-SNE distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    N = points01.shape[0]
    # Map points to grid cells
    r = np.clip((points01[:, 1] * rows).astype(int), 0, rows - 1)
    c = np.clip((points01[:, 0] * cols).astype(int), 0, cols - 1)
    
    # Compute cell centers in [0,1] space
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
            # Only one point, use it
            selected_indices.append(point_indices[0])
            out_rc.append((ri, ci))
        elif len(point_indices) == 2:
            # Two points, randomly choose one
            chosen = rng.choice(point_indices)
            selected_indices.append(chosen)
            out_rc.append((ri, ci))
        else:
            # Multiple points: choose the one closest to cell center
            center = cell_centers[ri, ci]
            points_in_cell = points01[point_indices]
            dists = np.sum((points_in_cell - center) ** 2, axis=1)
            closest_idx = point_indices[np.argmin(dists)]
            selected_indices.append(closest_idx)
            out_rc.append((ri, ci))
    
    # Build output arrays matching the selected points
    selected_indices = np.array(selected_indices, dtype=int)
    out_rc = np.array(out_rc, dtype=int)
    cell_indices = out_rc[:, 0] * cols + out_rc[:, 1]
    
    return cell_indices, out_rc, selected_indices


def assign_points_to_grid_hungarian(points01: np.ndarray, rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    SciPy optimal assignment on a subset of grid cells (size = N).
    We pick N cells that are the nearest-empty via a greedy uniqueness pass,
    then solve Hungarian on that square cost matrix.
    """
    centers = make_grid_centers(rows, cols)  # (C,2)
    C = centers.shape[0]
    N = points01.shape[0]
    tree = cKDTree(centers)
    # For each point, get k nearest grid cells
    k = min(64, C)
    dists, idxs = tree.query(points01, k=k)

    # Greedy selection of unique cells: walk points in random order
    rng = np.random.default_rng(42)
    order = rng.permutation(N)
    chosen = set()
    chosen_list = []
    chosen_for_point = np.full(N, -1, dtype=int)
    # try up to k neighbors
    for p in order:
        for j in range(k):
            cell = int(idxs[p, j])
            if cell not in chosen:
                chosen.add(cell)
                chosen_for_point[p] = cell
                chosen_list.append(cell)
                break
    # If still short (collisions too heavy), fill with any remaining cells
    if len(chosen_list) < N:
        remaining = [ci for ci in range(C) if ci not in chosen]
        fill = remaining[: (N - len(chosen_list))]
        chosen_list.extend(fill)
        # Assign leftover points arbitrarily (will be corrected by Hungarian)
        for p in range(N):
            if chosen_for_point[p] < 0:
                chosen_for_point[p] = fill.pop() if fill else chosen_list[0]

    # Build cost matrix between points and selected cells
    chosen_centers = centers[np.array(chosen_list)]
    # cost = squared Euclidean
    P = points01[:, None, :]  # (N,1,2)
    G = chosen_centers[None, :, :]  # (1,N,2)
    cost = ((P - G) ** 2).sum(axis=2)  # (N, N)

    ri, cj = linear_sum_assignment(cost)
    assigned_cells = np.array(chosen_list)[cj]  # (N,)
    rc = np.column_stack((assigned_cells // cols, assigned_cells % cols))
    return assigned_cells, rc


def extract_tile(slide: openslide.OpenSlide, top_left_xy: Tuple[int, int], tile_px: int) -> Image.Image:
    x, y = int(top_left_xy[0]), int(top_left_xy[1])
    region = slide.read_region(location=(x, y), level=0, size=(tile_px, tile_px)).convert("RGB")
    return region


def add_selective_border(img: Image.Image, color: Tuple[int, int, int], 
                        sides: Dict[str, bool], border_width: int = 3) -> Image.Image:
    """
    Add colored borders only on specific sides of the image.
    
    Args:
        img: PIL Image
        color: RGB color tuple
        sides: Dict with keys 'top', 'bottom', 'left', 'right' indicating which sides to border
        border_width: Width of border in pixels
    
    Returns:
        Image with selective borders
    """
    from PIL import ImageDraw
    img_with_border = img.copy()
    draw = ImageDraw.Draw(img_with_border)
    w, h = img.size
    
    # Draw borders on specified sides
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


def find_boundary_sides(cluster_labels: np.ndarray, rc: np.ndarray, rows: int, cols: int) -> List[Dict[str, bool]]:
    """
    Find which sides of each tile border a different cluster or empty space.
    
    Args:
        cluster_labels: (N,) array of cluster IDs for each tile
        rc: (N, 2) array of (row, col) positions for each tile
        rows, cols: grid dimensions
    
    Returns:
        List of N dicts, each with keys 'top', 'bottom', 'left', 'right' indicating which sides border different clusters or empty cells
    """
    N = len(cluster_labels)
    # Create a grid mapping (row, col) -> (tile_index, cluster_id)
    grid_cluster = np.full((rows, cols), -1, dtype=int)  # -1 = empty
    
    for i in range(N):
        r, c = rc[i]
        grid_cluster[r, c] = cluster_labels[i]
    
    # Check each tile's neighbors
    border_sides = []
    
    for i in range(N):
        r, c = rc[i]
        my_cluster = cluster_labels[i]
        sides = {'top': False, 'bottom': False, 'left': False, 'right': False}
        
        # Check top neighbor (row - 1)
        if r > 0:
            neighbor_cluster = grid_cluster[r-1, c]
            # Border if neighbor is empty OR has different cluster
            if neighbor_cluster == -1 or neighbor_cluster != my_cluster:
                sides['top'] = True
        
        # Check bottom neighbor (row + 1)
        if r < rows - 1:
            neighbor_cluster = grid_cluster[r+1, c]
            # Border if neighbor is empty OR has different cluster
            if neighbor_cluster == -1 or neighbor_cluster != my_cluster:
                sides['bottom'] = True
        
        # Check left neighbor (col - 1)
        if c > 0:
            neighbor_cluster = grid_cluster[r, c-1]
            # Border if neighbor is empty OR has different cluster
            if neighbor_cluster == -1 or neighbor_cluster != my_cluster:
                sides['left'] = True
        
        # Check right neighbor (col + 1)
        if c < cols - 1:
            neighbor_cluster = grid_cluster[r, c+1]
            # Border if neighbor is empty OR has different cluster
            if neighbor_cluster == -1 or neighbor_cluster != my_cluster:
                sides['right'] = True
        
        border_sides.append(sides)
    
    return border_sides


def get_cluster_colors(n_clusters: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for clusters.
    """
    if n_clusters <= 5:
        # Hand-picked distinct colors for up to 5 clusters
        colors = [
            (230, 25, 75),    # Red
            (60, 180, 75),     # Green
            (255, 225, 25),    # Yellow
            (0, 130, 200),     # Blue
            (245, 130, 48),    # Orange
        ]
        return colors[:n_clusters]
    else:
        # Generate colors using HSV color space for more clusters
        import colorsys
        colors = []
        for i in range(n_clusters):
            hue = i / n_clusters
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors


def stitch_mosaic(tiles: List[Image.Image], rc: np.ndarray, rows: int, cols: int, mosaic_tile_px: int,
                  bg_color=(255, 255, 255)) -> Image.Image:
    H = rows * mosaic_tile_px
    W = cols * mosaic_tile_px
    canvas = Image.new('RGB', (W, H), color=bg_color)
    for img, (r, c) in zip(tiles, rc):
        if img is None:  # safety
            continue
        if img.size != (mosaic_tile_px, mosaic_tile_px):
            img = img.resize((mosaic_tile_px, mosaic_tile_px), resample=Image.BILINEAR)
        canvas.paste(img, (c * mosaic_tile_px, r * mosaic_tile_px))
    return canvas


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Gridified t-SNE mosaic of pathology tiles.")
    ap.add_argument('--h5_root', required=True, help='Root folder containing per-slide H5 files.')
    ap.add_argument('--svs_root', required=True, help='Root folder containing original WSI files.')
    ap.add_argument('--output_png', required=True, help='Output PNG path.')
    ap.add_argument('--max_slides', type=int, default=30, help='Max slides to include.')
    ap.add_argument('--per_slide_max', type=int, default=5000, help='Max tiles to sample per slide before global cap.')
    ap.add_argument('--max_tiles', type=int, default=3600, help='Global cap on number of tiles (also used as grid target).')
    ap.add_argument('--coord_bins', type=int, nargs=2, default=[32, 32], help='2D bins for location-aware sampling per slide.')
    ap.add_argument('--grid_rows', type=int, default=60, help='Grid rows for final mosaic.')
    ap.add_argument('--grid_cols', type=int, default=60, help='Grid cols for final mosaic.')
    ap.add_argument('--mosaic_tile_px', type=int, default=64, help='Tile size (in pixels) to paste into the final mosaic (downsampled).')
    ap.add_argument('--grid_method', choices=['greedy', 'hungarian', 'direct'], default='greedy', 
                    help='Grid assignment method: greedy (spiral search), hungarian (optimal), direct (preserve t-SNE, pick centroid on collision).')
    ap.add_argument('--seed', type=int, default=42, help='Random seed.')
    ap.add_argument('--tsne_perplexity', type=float, default=30.0, help='t-SNE perplexity.')
    ap.add_argument('--tsne_iter', type=int, default=2000, help='t-SNE iterations.')
    ap.add_argument('--debug', action='store_true', help='Verbose debug prints.')
    ap.add_argument('--save_tsne_coords', type=str, default=None, help='Optional: save t-SNE coordinates to NPZ file for debugging.')
    ap.add_argument('--n_clusters', type=int, default=0, help='Number of K-means clusters for colored borders (0 = no clustering).')
    ap.add_argument('--border_width', type=int, default=3, help='Width of colored cluster borders in pixels.')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    mag_x, tile_px_infer = parse_mag_and_tile(args.h5_root)
    if args.debug:
        print(f"[infer] magnification ≈ {mag_x}x, tile_px ≈ {tile_px_infer}")

    h5_list = list_h5_files(args.h5_root, args.max_slides)
    if len(h5_list) == 0:
        raise RuntimeError(f"No H5 files found under {args.h5_root}")

    # Infer feature dim from first h5
    D = load_first_feature_dim(h5_list[0])
    if args.debug:
        print(f"[infer] feature dimension = {D}")

    # Load & sample per slide
    all_feats = []
    all_coords = []
    slide_keys = []
    svs_paths = {}

    pbar = tqdm(h5_list, desc="[load+sample per slide]")
    for h5p in pbar:
        base = basename_no_ext(h5p)
        try:
            svs_path = find_matching_svs(base, args.svs_root)
            svs_paths[base] = svs_path
        except Exception as e:
            if args.debug:
                print(f"[warn] SVS not found for {base}: {e}")
            continue

        with h5py.File(h5p, 'r') as f:
            feats = f['features'][:]  # (N,D)
            coords = f['coords'][:]   # (N,2)
        if feats.shape[0] != coords.shape[0]:
            if args.debug:
                print(f"[skip] Mismatch counts in {base}: feats {feats.shape[0]} vs coords {coords.shape[0]}")
            continue

        sel = sample_tiles_by_location(coords, per_slide_max=args.per_slide_max,
                                       bins_xy=tuple(args.coord_bins), rng=rng)
        feats = feats[sel]
        coords = coords[sel]

        all_feats.append(feats)
        all_coords.append(np.column_stack([coords, np.full(len(coords), len(slide_keys), dtype=int)]))  # append slide idx
        slide_keys.append(base)

    if len(all_feats) == 0:
        raise RuntimeError("No tiles loaded after sampling.")

    feats = np.concatenate(all_feats, axis=0)
    coords_slide = np.concatenate(all_coords, axis=0)  # (N, 3) => x, y, slide_idx
    if args.debug:
        print(f"[stats] sampled tiles total = {feats.shape[0]} across slides = {len(slide_keys)}")

    # Proportional sampling across slides to reach the target max_tiles
    per_slide_counts = [f.shape[0] for f in all_feats]
    total_available = int(np.sum(per_slide_counts))
    target = min(args.max_tiles, total_available)

    # If we already have <= target, keep all
    if total_available <= target:
        feats = np.concatenate(all_feats, axis=0)
        coords_slide = np.concatenate(all_coords, axis=0)
        if args.debug:
            print(f"[stats] using all {total_available} tiles (<= target {target}).")
    else:
        # Compute proportional quotas
        counts = np.array(per_slide_counts, dtype=float)
        proportions = counts / counts.sum()
        raw_quotas = proportions * target
        base_quotas = np.floor(raw_quotas).astype(int)
        remainder = int(target - base_quotas.sum())
        # Distribute remaining tiles by largest fractional parts
        frac = raw_quotas - base_quotas
        order = np.argsort(-frac)
        for i in order[:remainder]:
            base_quotas[i] += 1
        # Clip by availability per slide
        base_quotas = np.minimum(base_quotas, counts.astype(int))
        # If clipping reduced total, top up greedily
        deficit = target - base_quotas.sum()
        if deficit > 0:
            avail = counts.astype(int) - base_quotas
            add_order = np.argsort(-avail)
            for i in add_order:
                if deficit <= 0: break
                add = min(avail[i], deficit)
                base_quotas[i] += add
                deficit -= add
        # Now sample per slide according to quotas
        feats_sel = []
        coords_sel = []
        for sid, (f_slide, c_slide) in enumerate(zip(all_feats, all_coords)):
            q = int(base_quotas[sid])
            if q <= 0:
                continue
            if f_slide.shape[0] <= q:
                take_idx = np.arange(f_slide.shape[0])
            else:
                take_idx = rng.choice(f_slide.shape[0], size=q, replace=False)
            feats_sel.append(f_slide[take_idx])
            coords_sel.append(c_slide[take_idx])
        feats = np.concatenate(feats_sel, axis=0)
        coords_slide = np.concatenate(coords_sel, axis=0)
        if args.debug:
            print(f"[stats] proportional selection: target={target}, chosen={feats.shape[0]} across {len(feats_sel)} slides.")

    # Embed
    Y = reduce_and_embed(feats,
                         seed=args.seed,
                         pca_threshold=50,
                         tsne_perplexity=args.tsne_perplexity,
                         tsne_iter=args.tsne_iter)
    Y01 = normalize_to_unit_square(Y)
    
    # K-means clustering (optional)
    cluster_labels = None
    cluster_colors = None
    if args.n_clusters > 0:
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(Y01)  # Cluster on normalized t-SNE
        cluster_colors = get_cluster_colors(args.n_clusters)
        if args.debug:
            print(f"[cluster] K-means with {args.n_clusters} clusters")
            for i in range(args.n_clusters):
                count = int((cluster_labels == i).sum())
                print(f"  Cluster {i}: {count} tiles ({count/len(cluster_labels)*100:.1f}%)")
    
    # Optional: save t-SNE coordinates for debugging
    if args.save_tsne_coords:
        save_dict = {
            'tsne_raw': Y, 
            'tsne_normalized': Y01,
            'slide_indices': coords_slide[:, 2]
        }
        if cluster_labels is not None:
            save_dict['cluster_labels'] = cluster_labels
        np.savez(args.save_tsne_coords, **save_dict)
        if args.debug:
            print(f"[debug] saved t-SNE coords to {args.save_tsne_coords}")
            print(f"[debug] t-SNE range: x=[{Y[:, 0].min():.2f}, {Y[:, 0].max():.2f}], y=[{Y[:, 1].min():.2f}, {Y[:, 1].max():.2f}]")

    # Grid assignment
    rows, cols = args.grid_rows, args.grid_cols
    max_cells = rows * cols
    
    if args.grid_method == 'direct':
        # Direct method handles its own selection
        _, rc, selected_indices = assign_points_to_grid_direct(Y01, rows, cols, rng=rng)
        # Filter to only selected tiles
        coords_slide = coords_slide[selected_indices]
        if cluster_labels is not None:
            cluster_labels = cluster_labels[selected_indices]
        if args.debug:
            print(f"[grid-direct] selected {len(selected_indices)} tiles from {Y01.shape[0]} (kept {len(selected_indices)/Y01.shape[0]*100:.1f}%)")
    else:
        # Greedy and Hungarian need all points assigned
        if feats.shape[0] > max_cells:
            # reduce to max_cells (cannot place more than grid capacity)
            keep = rng.choice(feats.shape[0], size=max_cells, replace=False)
            Y01 = Y01[keep]
            coords_slide = coords_slide[keep]
            if cluster_labels is not None:
                cluster_labels = cluster_labels[keep]
            if args.debug:
                print(f"[grid] reduced to capacity {max_cells} tiles (grid {rows}x{cols}).")
        
        if args.grid_method == 'greedy':
            _, rc = assign_points_to_grid_greedy(Y01, rows, cols)
        else:
            _, rc = assign_points_to_grid_hungarian(Y01, rows, cols)
    
    # Find which sides of each tile border different clusters
    boundary_sides = None
    if cluster_labels is not None:
        boundary_sides = find_boundary_sides(cluster_labels, rc, rows, cols)
        if args.debug:
            n_boundary = sum(1 for sides in boundary_sides if any(sides.values()))
            print(f"[cluster] {n_boundary} tiles at cluster boundaries ({n_boundary/len(boundary_sides)*100:.1f}%)")

    # Extract tiles and stitch
    tile_px = tile_px_infer
    tiles = []
    # Group indices by slide
    # coords_slide columns: x, y, slide_idx
    slide_idx_to_ids = {}
    for i, (_, _, sid) in enumerate(coords_slide):
        slide_idx_to_ids.setdefault(int(sid), []).append(i)

    # Open slides once per slide
    for sid, id_list in tqdm(slide_idx_to_ids.items(), desc="[extract tiles]"):
        base = slide_keys[sid]
        svs_path = svs_paths[base]
        slide = openslide.OpenSlide(svs_path)
        try:
            for i in id_list:
                x, y, _ = coords_slide[i]
                try:
                    img = extract_tile(slide, (int(x), int(y)), tile_px)
                    # Add colored borders only on sides that touch different clusters
                    if (cluster_labels is not None and cluster_colors is not None and 
                        boundary_sides is not None and img is not None):
                        sides = boundary_sides[i]
                        # Only add border if at least one side needs it
                        if any(sides.values()):
                            cluster_id = int(cluster_labels[i])
                            color = cluster_colors[cluster_id]
                            img = add_selective_border(img, color, sides, border_width=args.border_width)
                except Exception as e:
                    img = None
                    if args.debug:
                        print(f"[warn] extract failed {base} @ ({x},{y}): {e}")
                tiles.append((i, img))
        finally:
            slide.close()

    # Reorder tiles to match rc order
    # tiles is list of (index_in_coords_slide, img)
    img_array = [None] * coords_slide.shape[0]
    for i, im in tiles:
        img_array[i] = im

    mosaic = stitch_mosaic(img_array, rc, rows, cols, args.mosaic_tile_px, bg_color=(255, 255, 255))

    # Ensure output directory
    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    # Save with requested DPI=400
    mosaic.save(args.output_png, format='PNG', dpi=(400, 400))
    print(f"[done] wrote {args.output_png} (grid {rows}x{cols}, tiles={len(img_array)}, tile_px={tile_px})")
    
    # Print cluster color legend if clustering was used
    if args.n_clusters > 0 and cluster_colors is not None:
        print(f"\n[cluster colors]")
        for i, color in enumerate(cluster_colors):
            print(f"  Cluster {i}: RGB{color}")


if __name__ == '__main__':
    main()
