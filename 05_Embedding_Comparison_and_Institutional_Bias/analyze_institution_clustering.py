#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Institution Clustering Analysis

Measures how well tile-level feature embeddings cluster by TCGA institution code,
separately for tumor and normal tissue. Uses silhouette score and Davies-Bouldin index
with bootstrapping, and Wilcoxon rank-sum tests between embedding models.

No t-SNE or visualizations; computational analysis only.

Outputs:
- institution_clustering_metrics_bootstrap.csv
- institution_clustering_wilcoxon.csv
- institution_clustering_barplot.png (when CSVs exist or after full run)

Requirements: numpy, scipy, scikit-learn, pandas, h5py, openslide-python, tqdm, shapely (optional), matplotlib
"""

import argparse
import math
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import openslide

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_palette("husl")
except ImportError:
    sns = None

try:
    from shapely.geometry import Polygon, box
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

import xml.etree.ElementTree as ET


# -----------------------------
# Institution code
# -----------------------------

def extract_institution_code(slide_stem: str) -> Optional[str]:
    """
    Extract TCGA institution code from slide stem.
    Pattern: TCGA-{institution_code}-{patient_id}_{...}
    Returns the second segment after splitting on '-', or None if not TCGA-like.
    """
    parts = slide_stem.split("-")
    if len(parts) >= 2 and parts[0].upper() == "TCGA":
        return parts[1]
    return None


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


# -----------------------------
# Annotations (kidney tumor/normal; prostate; rectal uses none)
# -----------------------------

# Prostate annotation color constants (COLORREF format)
PROSTATE_TUMOR_COLOR = 65280   # Green (#00FF00)
PROSTATE_BENIGN_COLOR = 65535  # Yellow (#FFFF00)


def find_tumor_xml(annotation_dir: Path, stem: str) -> Optional[Path]:
    """Return the first tumor-tagged XML for the slide, or None."""
    hits = sorted(annotation_dir.glob(f"{stem}*tumor*.xml"))
    return hits[0] if hits else None


def find_normal_xml(annotation_dir: Path, stem: str) -> Optional[Path]:
    """Return the first normal-tagged XML for the slide, or None."""
    hits = sorted(annotation_dir.glob(f"{stem}*normal*.xml"))
    return hits[0] if hits else None


def find_prostate_xml(annotation_dir: Path, stem: str) -> Optional[Path]:
    """Return the single annotation XML for the slide (prostate format), or None."""
    xml_path = annotation_dir / f"{stem}.xml"
    return xml_path if xml_path.exists() else None


def parse_polygons_prostate(xml_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Parse prostate format XML with color support; return (tumor_polys, benign_polys).
    LineColor 65280 (Green) = Tumor, 65535 (Yellow) = Benign. Other colors ignored.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tumor_polys = []
    benign_polys = []
    for annotation in root.findall("Annotation"):
        line_color = annotation.get("LineColor")
        colorref = int(line_color) if line_color else None
        for region in annotation.findall(".//Region"):
            verts = region.find("Vertices")
            if verts is None:
                continue
            pts = []
            for v in verts.iter("Vertex"):
                x, y = v.get("X"), v.get("Y")
                if x is not None and y is not None:
                    pts.append((float(x), float(y)))
            if len(pts) >= 3:
                poly = np.array(pts, dtype=np.float32)
                if colorref == PROSTATE_TUMOR_COLOR:
                    tumor_polys.append(poly)
                elif colorref == PROSTATE_BENIGN_COLOR:
                    benign_polys.append(poly)
    return tumor_polys, benign_polys


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


def tile_overlaps_any_polygon(
    tile_bounds: Tuple[float, float, float, float],
    polygons: List[np.ndarray],
) -> bool:
    """Check if a tile overlaps with any polygon."""
    x_min, y_min, x_max, y_max = tile_bounds
    if HAS_SHAPELY:
        tile_box = box(x_min, y_min, x_max, y_max)
        for poly_coords in polygons:
            try:
                poly = Polygon(poly_coords)
                if tile_box.intersects(poly):
                    return True
            except Exception:
                continue
        return False
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
        if not (x_max < poly_x_min or x_min > poly_x_max or y_max < poly_y_min or y_min > poly_y_max):
            return True
    return False


# -----------------------------
# Tile collection by institution and tissue
# -----------------------------

def collect_tiles_by_institution_and_tissue(
    h5_root: str,
    svs_root: Path,
    annotation_dir: Path,
    valid_slides: List[Path],
    mag_x: int,
    tile_px: int,
    embedding_name: str,
    args,
    data_type: str = "kidney",
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[List[str]],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[List[str]],
    Optional[Dict[int, str]],
]:
    """
    Collect tiles by institution and tissue. data_type: kidney (tumor/normal XMLs),
    prostate (single XML tumor/benign), rectal (no annotations, all H5 tiles).
    Returns tumor_* and normal_*; for rectal, tumor_* are None and normal_* hold
    the single stream with tissue_type "all".
    """
    all_tumor_features = []
    all_tumor_inst_codes = []
    all_tumor_slide_stems = []
    all_normal_features = []
    all_normal_inst_codes = []
    all_normal_slide_stems = []
    inst_to_int: Dict[str, int] = {}
    seed = getattr(args, "seed", 42)
    sample_frac = getattr(args, "sample_fraction", 0.1)

    for slide_idx, slide_path in enumerate(tqdm(valid_slides, desc=f"[{embedding_name}] Loading tiles")):
        slide_stem = slide_path.stem
        inst_code = extract_institution_code(slide_stem)
        if inst_code is None:
            if getattr(args, "debug", False):
                print(f"[skip] {slide_stem}: no TCGA institution code")
            continue

        h5_path = Path(h5_root) / f"{slide_stem}.h5"
        if not h5_path.exists():
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                features = f["features"][:]
                coords = np.array(f.get("coords", f.get("coord"))).reshape(-1, 2)
        except Exception as e:
            if getattr(args, "debug", False):
                print(f"[skip] {slide_stem}: H5 error: {e}")
            continue
        if features.shape[0] != coords.shape[0] or features.shape[0] == 0:
            continue

        if data_type == "rectal":
            # No annotations: use all tiles (or sample), label by institution only
            n_all = features.shape[0]
            rng = np.random.default_rng(seed + slide_idx)
            n_sample = max(1, int(n_all * sample_frac))
            sample_idx = rng.choice(n_all, size=min(n_sample, n_all), replace=False)
            if inst_code not in inst_to_int:
                inst_to_int[inst_code] = len(inst_to_int)
            all_normal_features.append(features[sample_idx])
            all_normal_inst_codes.extend([inst_code] * len(sample_idx))
            all_normal_slide_stems.extend([slide_stem] * len(sample_idx))
            continue

        # Kidney or prostate: need polygons
        if data_type == "kidney":
            tumor_xml = find_tumor_xml(annotation_dir, slide_stem)
            normal_xml = find_normal_xml(annotation_dir, slide_stem)
            if not tumor_xml or not normal_xml:
                continue
            tumor_polys = parse_polygons_from_xml(tumor_xml)
            normal_polys = parse_polygons_from_xml(normal_xml)
        else:  # prostate
            xml_path = find_prostate_xml(annotation_dir, slide_stem)
            if xml_path is None:
                continue
            tumor_polys, normal_polys = parse_polygons_prostate(xml_path)
        if not tumor_polys or not normal_polys:
            continue

        slide = openslide.OpenSlide(str(slide_path))
        base_mag = float(slide.properties.get("openslide.objective-power", mag_x))
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

        if inst_code not in inst_to_int:
            inst_to_int[inst_code] = len(inst_to_int)

        rng = np.random.default_rng(seed + slide_idx)
        n_tumor_sample = max(1, int(len(tumor_indices) * sample_frac))
        n_normal_sample = max(1, int(len(normal_indices) * sample_frac))
        tumor_sample_idx = rng.choice(
            tumor_indices, size=min(n_tumor_sample, len(tumor_indices)), replace=False
        )
        normal_sample_idx = rng.choice(
            normal_indices, size=min(n_normal_sample, len(normal_indices)), replace=False
        )

        all_tumor_features.append(features[tumor_sample_idx])
        all_tumor_inst_codes.extend([inst_code] * len(tumor_sample_idx))
        all_tumor_slide_stems.extend([slide_stem] * len(tumor_sample_idx))
        all_normal_features.append(features[normal_sample_idx])
        all_normal_inst_codes.extend([inst_code] * len(normal_sample_idx))
        all_normal_slide_stems.extend([slide_stem] * len(normal_sample_idx))

    if not all_tumor_features and not all_normal_features:
        return None, None, None, None, None, None, None

    label_map = {i: code for code, i in inst_to_int.items()}

    tumor_f = np.concatenate(all_tumor_features, axis=0) if all_tumor_features else None
    tumor_labels = np.array([inst_to_int[c] for c in all_tumor_inst_codes], dtype=int) if all_tumor_inst_codes else None
    tumor_stems = all_tumor_slide_stems if all_tumor_slide_stems else None

    normal_f = np.concatenate(all_normal_features, axis=0) if all_normal_features else None
    normal_labels = np.array([inst_to_int[c] for c in all_normal_inst_codes], dtype=int) if all_normal_inst_codes else None
    normal_stems = all_normal_slide_stems if all_normal_slide_stems else None

    return tumor_f, tumor_labels, tumor_stems, normal_f, normal_labels, normal_stems, label_map


def filter_by_min_slides_per_institution(
    features: np.ndarray,
    inst_labels: np.ndarray,
    slide_stems: List[str],
    label_map: Dict[int, str],
    min_slides_per_institution: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Keep only institutions that have at least min_slides_per_institution distinct slides.
    Returns (filtered_features, filtered_labels, new_label_map).
    """
    from collections import Counter
    inst_to_slides: Dict[int, set] = {}
    for i, stem in enumerate(slide_stems):
        lid = int(inst_labels[i])
        inst_to_slides.setdefault(lid, set()).add(stem)
    keep_ids = {lid for lid, stems in inst_to_slides.items() if len(stems) >= min_slides_per_institution}
    if len(keep_ids) < 2:
        return np.array([]), np.array([], dtype=int), {}
    mask = np.array([lid in keep_ids for lid in inst_labels], dtype=bool)
    feats = features[mask]
    labs = inst_labels[mask]
    old_to_new: Dict[int, int] = {}
    for lid in sorted(keep_ids):
        old_to_new[lid] = len(old_to_new)
    new_label_map = {old_to_new[lid]: label_map[lid] for lid in keep_ids}
    new_labs = np.array([old_to_new[int(l)] for l in labs], dtype=int)
    return feats, new_labs, new_label_map


# -----------------------------
# Bootstrapped metrics (silhouette + Davies-Bouldin on PCA)
# -----------------------------

def compute_institution_bootstrapped_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    embedding_name: str,
    tissue_type: str,
    label_map: Dict[int, str],
    n_bootstrap: int = 50,
    bootstrap_fraction: float = 0.5,
    pca_components: int = 50,
    seed: int = 42,
) -> List[Dict]:
    """
    Compute silhouette and Davies-Bouldin on PCA-reduced features with bootstrapping.
    Skips iterations where fewer than 2 institutions appear in the sample.
    """
    results = []
    rng = np.random.default_rng(seed)
    n_total = len(features)
    n_sample = int(n_total * bootstrap_fraction)
    if n_sample < 10:
        return results

    n_inst = len(label_map)
    if n_inst < 2:
        return results

    for bootstrap_idx in tqdm(range(n_bootstrap), desc=f"[{embedding_name}] {tissue_type} bootstrap"):
        sample_idx = rng.choice(n_total, size=n_sample, replace=True)
        X = features[sample_idx]
        y = labels[sample_idx]
        if len(np.unique(y)) < 2:
            continue

        # PCA
        n_comp = min(pca_components, X.shape[0] - 1, X.shape[1])
        if n_comp < 2:
            continue
        pca = PCA(n_components=n_comp, random_state=seed + bootstrap_idx)
        Xpca = pca.fit_transform(X)

        try:
            sil = float(silhouette_score(Xpca, y, metric="euclidean"))
        except Exception:
            sil = np.nan
        try:
            db = float(davies_bouldin_score(Xpca, y))
        except Exception:
            db = np.nan

        # Per-institution compactness (mean distance to centroid in PCA space, same as example script)
        per_inst_compactness = {}
        for label_int, code in label_map.items():
            mask = y == label_int
            if mask.sum() > 0:
                pts = Xpca[mask]
                centroid = pts.mean(axis=0)
                per_inst_compactness[code] = float(np.sqrt(((pts - centroid) ** 2).sum(axis=1)).mean())
            else:
                per_inst_compactness[code] = np.nan

        row = {
            "bootstrap_idx": bootstrap_idx,
            "embedding": embedding_name,
            "tissue_type": tissue_type,
            "silhouette_pca": sil,
            "davies_bouldin_pca": db,
            "n_institutions": len(np.unique(y)),
            "n_tiles": int(len(y)),
        }
        for code, val in per_inst_compactness.items():
            row[f"{code}_compactness"] = val
        results.append(row)
    return results


# -----------------------------
# Wilcoxon comparison between models
# -----------------------------

def run_wilcoxon_comparisons(
    all_bootstrap_rows: List[Dict],
    min_institutions: int = 2,
) -> List[Dict]:
    """Pairwise Wilcoxon rank-sum for each (embedding, tissue_type, metric) combination."""
    comparison_results = []
    df = pd.DataFrame(all_bootstrap_rows)
    if df.empty:
        return comparison_results

    model_names = df["embedding"].unique().tolist()
    base_metrics = ["silhouette_pca", "davies_bouldin_pca"]
    compactness_cols = [c for c in df.columns if c.endswith("_compactness")]
    metrics = base_metrics + compactness_cols
    for tissue in df["tissue_type"].unique():
        sub = df[df["tissue_type"] == tissue]
        for i, m1 in enumerate(model_names):
            for m2 in model_names[i + 1 :]:
                for metric in metrics:
                    s1 = sub.loc[sub["embedding"] == m1, metric].dropna()
                    s2 = sub.loc[sub["embedding"] == m2, metric].dropna()
                    if len(s1) > 0 and len(s2) > 0:
                        try:
                            stat, p = mannwhitneyu(s1, s2, alternative="two-sided")
                        except Exception:
                            continue
                        comparison_results.append({
                            "model1": m1,
                            "model2": m2,
                            "tissue_type": tissue,
                            "metric": metric,
                            "model1_mean": float(s1.mean()),
                            "model1_std": float(s1.std()),
                            "model2_mean": float(s2.mean()),
                            "model2_std": float(s2.std()),
                            "U_statistic": float(stat),
                            "p_value": float(p),
                        })
    return comparison_results


# -----------------------------
# Bar plots (publication style, same as create_aggregated_tumor_normal_tsne)
# -----------------------------

SUBPLOT_WIDTH_INCH = 1.6


def create_institution_barplots(
    bootstrap_df: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame],
    output_dir: Path,
    output_suffix: str = "",
) -> None:
    """
    Bar plot: one panel per (tissue_type, metric) from bootstrap_df (silhouette_pca, davies_bouldin_pca).
    Rectal: 2 panels (all × sil, all × db); kidney/prostate: 4 panels. Optional significance stars.
    output_suffix: appended before .png (e.g. _alltissues -> institution_clustering_barplot_alltissues.png).
    """
    model_colors = {
        "conch": "#0070C0",
        "musk": "#FFC000",
        "hoptimus": "#C00000",
        "virchow": "#7030A0",
        "gigapath": "#00B050",
        "pruned": "#FF6600",
        "naive": "#FFFFFF",
        "uni": "#92D050",
        "conch_v15": "#0070C0",
    }

    def model_sort_key(model: str):
        if model == "pruned":
            return (0, model)
        if model == "naive":
            return (1, model)
        return (2, model)

    tissue_types = sorted(bootstrap_df["tissue_type"].unique().tolist())
    metric_specs = [
        ("silhouette_pca", "Silhouette", (0.0, 0.5)),
        ("davies_bouldin_pca", "Davies-Bouldin", (0.0, None)),
    ]
    panels = [
        (tissue, metric_col, f"{metric_title} ({tissue})", y_range)
        for tissue in tissue_types
        for (metric_col, metric_title, y_range) in metric_specs
    ]
    if not panels:
        return

    plt.style.use("default")
    if sns is not None:
        sns.set_palette("husl")

    n_panels = len(panels)
    fig_width = n_panels * SUBPLOT_WIDTH_INCH
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4))
    if n_panels == 1:
        axes = [axes]

    all_models = sorted(bootstrap_df["embedding"].unique(), key=model_sort_key)

    for ax, (tissue_type, metric, title, y_range) in zip(axes, panels):
        sub = bootstrap_df[bootstrap_df["tissue_type"] == tissue_type]
        if sub.empty:
            ax.set_title(title)
            ax.set_visible(True)
            continue

        model_means = []
        model_stds = []
        models_with_data = []
        for model in all_models:
            vals = sub.loc[sub["embedding"] == model, metric].dropna()
            if len(vals) > 0:
                model_means.append(float(vals.mean()))
                model_stds.append(float(vals.std()))
                models_with_data.append(model)

        if not model_means:
            ax.set_title(title)
            continue

        bar_bottom = y_range[0] if y_range[0] is not None else 0.0
        x_pos = np.arange(len(models_with_data))
        bar_heights = [m - bar_bottom for m in model_means]
        bars = ax.bar(x_pos, bar_heights, bottom=bar_bottom, width=0.6, edgecolor="black", linewidth=2)

        for bar, model in zip(bars, models_with_data):
            base_model = model.rstrip("0123456789")
            color = model_colors.get(model, model_colors.get(base_model, "#CCCCCC"))
            if "naive" in model.lower():
                bar.set_facecolor("white")
                bar.set_hatch("...")
                bar.set_alpha(1.0)
            elif "pruned" in model.lower():
                bar.set_facecolor("white")
                bar.set_alpha(1.0)
            else:
                bar.set_facecolor(color)
                bar.set_alpha(0.7)
            bar.set_edgecolor("black")
            bar.set_linewidth(2)

        ax.set_title(title, fontsize=7, fontweight="normal", pad=25)
        if y_range[1] is not None:
            ax.set_ylim(y_range)
        else:
            ax.set_ylim(bar_bottom, max(model_means) * 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylabel("")

        # Significance stars (vs pruned)
        if comparison_df is not None and not comparison_df.empty and "pruned" in models_with_data:
            comp_sub = comparison_df[
                (comparison_df["tissue_type"] == tissue_type) & (comparison_df["metric"] == metric)
            ]
            for i, model in enumerate(models_with_data):
                if model == "pruned":
                    continue
                p_value = None
                for _, row in comp_sub.iterrows():
                    if (row["model1"] == "pruned" and row["model2"] == model) or (
                        row["model1"] == model and row["model2"] == "pruned"
                    ):
                        p_value = row["p_value"]
                        break
                if p_value is not None and p_value < 0.05:
                    star_y = model_means[i] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    ax.text(i, star_y, "*", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / f"institution_clustering_barplot{output_suffix}.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved bar plot to {out_path}")


def create_institution_all_metrics_barplots(
    bootstrap_df: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame],
    output_dir: Path,
    output_suffix: str = "",
) -> None:
    """
    One subplot per metric: 4 global (silhouette/db tumor/normal) + per-institution compactness per tissue.
    Multi-row layout when there are many metrics. Same styling as create_institution_barplots.
    output_suffix: appended before .png (e.g. _alltissues -> institution_clustering_metrics_barplot_alltissues.png).
    """
    model_colors = {
        "conch": "#0070C0",
        "musk": "#FFC000",
        "hoptimus": "#C00000",
        "virchow": "#7030A0",
        "gigapath": "#00B050",
        "pruned": "#FF6600",
        "naive": "#FFFFFF",
        "uni": "#92D050",
        "conch_v15": "#0070C0",
    }

    def model_sort_key(model: str):
        if model == "pruned":
            return (0, model)
        if model == "naive":
            return (1, model)
        return (2, model)

    # Build panels: silhouette/db per tissue_type from data, then per-institution compactness
    tissue_types = sorted(bootstrap_df["tissue_type"].unique().tolist())
    metric_specs = [
        ("silhouette_pca", "Silhouette", (0.0, 0.5)),
        ("davies_bouldin_pca", "Davies-Bouldin", (0.0, None)),
    ]
    panels = [
        (tissue, metric_col, f"{metric_title} ({tissue})", y_range)
        for tissue in tissue_types
        for (metric_col, metric_title, y_range) in metric_specs
    ]
    compactness_cols = [c for c in bootstrap_df.columns if c.endswith("_compactness")]
    for col in sorted(compactness_cols):
        code = col.replace("_compactness", "")
        for tissue in bootstrap_df["tissue_type"].unique():
            if bootstrap_df.loc[bootstrap_df["tissue_type"] == tissue, col].notna().any():
                panels.append((tissue, col, f"{code} compactness ({tissue})", (0.0, None)))

    if not panels:
        return
    all_models = sorted(bootstrap_df["embedding"].unique(), key=model_sort_key)
    n_metrics = len(panels)
    n_cols = min(6, n_metrics)
    n_rows = math.ceil(n_metrics / n_cols)
    fig_width = n_cols * SUBPLOT_WIDTH_INCH
    fig_height = 4 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    plt.style.use("default")
    if sns is not None:
        sns.set_palette("husl")

    for idx, (tissue_type, metric, title, y_range) in enumerate(panels):
        ax = axes_flat[idx]
        sub = bootstrap_df[bootstrap_df["tissue_type"] == tissue_type]
        if metric not in sub.columns:
            ax.set_visible(False)
            continue
        if sub.empty:
            ax.set_title(title)
            ax.set_visible(True)
            continue
        model_means = []
        model_stds = []
        models_with_data = []
        for model in all_models:
            vals = sub.loc[sub["embedding"] == model, metric].dropna()
            if len(vals) > 0:
                model_means.append(float(vals.mean()))
                model_stds.append(float(vals.std()))
                models_with_data.append(model)
        if not model_means:
            ax.set_title(title)
            continue
        bar_bottom = y_range[0] if y_range[0] is not None else 0.0
        x_pos = np.arange(len(models_with_data))
        bar_heights = [m - bar_bottom for m in model_means]
        bars = ax.bar(x_pos, bar_heights, bottom=bar_bottom, width=0.6, edgecolor="black", linewidth=2)
        for bar, model in zip(bars, models_with_data):
            base_model = model.rstrip("0123456789")
            color = model_colors.get(model, model_colors.get(base_model, "#CCCCCC"))
            if "naive" in model.lower():
                bar.set_facecolor("white")
                bar.set_hatch("...")
                bar.set_alpha(1.0)
            elif "pruned" in model.lower():
                bar.set_facecolor("white")
                bar.set_alpha(1.0)
            else:
                bar.set_facecolor(color)
                bar.set_alpha(0.7)
            bar.set_edgecolor("black")
            bar.set_linewidth(2)
        ax.set_title(title, fontsize=7, fontweight="normal", pad=25)
        if y_range[1] is not None:
            ax.set_ylim(y_range)
        else:
            ax.set_ylim(bar_bottom, max(model_means) * 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylabel("")
        if comparison_df is not None and not comparison_df.empty and "pruned" in models_with_data:
            comp_sub = comparison_df[
                (comparison_df["tissue_type"] == tissue_type) & (comparison_df["metric"] == metric)
            ]
            for i, model in enumerate(models_with_data):
                if model == "pruned":
                    continue
                p_value = None
                for _, row in comp_sub.iterrows():
                    if (row["model1"] == "pruned" and row["model2"] == model) or (
                        row["model1"] == model and row["model2"] == "pruned"
                    ):
                        p_value = row["p_value"]
                        break
                if p_value is not None and p_value < 0.05:
                    star_y = model_means[i] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    ax.text(i, star_y, "*", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    out_path = output_dir / f"institution_clustering_metrics_barplot{output_suffix}.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved all-metrics bar plot to {out_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Institution clustering analysis: silhouette and Davies-Bouldin by embedding, tumor vs normal."
    )
    ap.add_argument("--h5_root", required=True, nargs="+", help="Root folder(s) for H5 feature files (one per model).")
    ap.add_argument("--svs_root", required=True, help="Root folder containing WSI files.")
    ap.add_argument("--annotation_dir", required=True, help="Directory with tumor/normal or prostate XMLs (unused for rectal).")
    ap.add_argument("--data_type", choices=["kidney", "prostate", "rectal"], default="kidney",
                    help="Dataset type: kidney (tumor/normal XMLs), prostate (single XML tumor/benign), rectal (no annotations, all H5 tiles).")
    ap.add_argument("--output_dir", required=True, help="Output directory for CSVs.")
    ap.add_argument("--sample_fraction", type=float, default=0.1, help="Fraction of tiles to sample per group per slide.")
    ap.add_argument("--max_slides", type=int, default=None, help="Max slides to process (default: all).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_bootstrap", type=int, default=50)
    ap.add_argument("--bootstrap_fraction", type=float, default=0.5)
    ap.add_argument("--pca_components", type=int, default=50)
    ap.add_argument("--min_slides_per_institution", type=int, default=5)
    ap.add_argument("--min_institutions", type=int, default=2)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--force", action="store_true", help="Force full recomputation even if bootstrap CSV exists; ignore figures-only skip.")
    ap.add_argument("--all_tissues", action="store_true",
                    help="Also run analysis with all tissues combined (kidney/prostate only); outputs use _alltissues tag.")
    args = ap.parse_args()

    svs_root = Path(args.svs_root)
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_path = output_dir / "institution_clustering_metrics_bootstrap.csv"
    wilcoxon_path = output_dir / "institution_clustering_wilcoxon.csv"
    bootstrap_alltissues_path = output_dir / "institution_clustering_metrics_bootstrap_alltissues.csv"
    wilcoxon_alltissues_path = output_dir / "institution_clustering_wilcoxon_alltissues.csv"

    do_skip = (
        bootstrap_path.exists()
        and not getattr(args, "force", False)
        and (not getattr(args, "all_tissues", False) or bootstrap_alltissues_path.exists())
    )
    if do_skip:
        print("Bootstrap CSV found in output directory; skipping analysis and creating figures only.")
        bootstrap_df = pd.read_csv(bootstrap_path)
        comparison_df = pd.read_csv(wilcoxon_path) if wilcoxon_path.exists() else None
        create_institution_barplots(bootstrap_df, comparison_df, output_dir)
        create_institution_all_metrics_barplots(bootstrap_df, comparison_df, output_dir)
        if getattr(args, "all_tissues", False) and bootstrap_alltissues_path.exists():
            bootstrap_df_at = pd.read_csv(bootstrap_alltissues_path)
            comparison_df_at = pd.read_csv(wilcoxon_alltissues_path) if wilcoxon_alltissues_path.exists() else None
            create_institution_barplots(bootstrap_df_at, comparison_df_at, output_dir, output_suffix="_alltissues")
            create_institution_all_metrics_barplots(bootstrap_df_at, comparison_df_at, output_dir, output_suffix="_alltissues")
        print("Done.")
        return

    # Valid slides by data_type
    data_type = getattr(args, "data_type", "kidney")
    slides = sorted(svs_root.glob("*.svs"))
    valid_slides = []
    for slide_path in slides:
        stem = slide_path.stem
        if data_type == "kidney":
            if find_tumor_xml(annotation_dir, stem) and find_normal_xml(annotation_dir, stem):
                valid_slides.append(slide_path)
        elif data_type == "prostate":
            xml_path = find_prostate_xml(annotation_dir, stem)
            if xml_path is not None:
                tumor_polys, benign_polys = parse_polygons_prostate(xml_path)
                if tumor_polys and benign_polys:
                    valid_slides.append(slide_path)
        else:  # rectal: no annotation check, only institution code
            if extract_institution_code(stem) is not None:
                valid_slides.append(slide_path)
    if not valid_slides:
        if data_type == "kidney":
            print("No slides with both tumor and normal annotations found.")
        elif data_type == "prostate":
            print("No slides with prostate annotations found.")
        else:
            print("No slides with parseable institution code (TCGA) found.")
        return
    if args.max_slides is not None and args.max_slides > 0:
        valid_slides = valid_slides[: args.max_slides]
    if data_type == "kidney":
        print(f"Using {len(valid_slides)} slides with tumor and normal annotations.")
    elif data_type == "prostate":
        print(f"Using {len(valid_slides)} slides with prostate annotations.")
    else:
        print(f"Using {len(valid_slides)} slides (rectal, no annotation filter).")

    # Common slides across all embeddings
    embedding_slide_sets = {}
    for h5_root in args.h5_root:
        name = extract_embedding_name(h5_root)
        available = set()
        for slide_path in valid_slides:
            if (Path(h5_root) / f"{slide_path.stem}.h5").exists():
                available.add(slide_path.stem)
        embedding_slide_sets[name] = available
    if len(embedding_slide_sets) > 1:
        common = set.intersection(*embedding_slide_sets.values())
        if not common:
            print("No common slides across all embeddings. Aborting.")
            return
        valid_slides = [s for s in valid_slides if s.stem in common]
        print(f"Common slides across embeddings: {len(valid_slides)}")
    else:
        print(f"Single embedding; using {len(valid_slides)} slides.")

    all_bootstrap_results = []
    all_bootstrap_results_alltissues = []

    for h5_root in args.h5_root:
        embedding_name = extract_embedding_name(h5_root)
        mag_x, tile_px = parse_mag_and_tile(h5_root)
        print(f"\n{'='*60}\nEmbedding: {embedding_name}\n{'='*60}")

        tumor_f, tumor_labels, tumor_stems, normal_f, normal_labels, normal_stems, label_map = (
            collect_tiles_by_institution_and_tissue(
                h5_root, svs_root, annotation_dir, valid_slides, mag_x, tile_px, embedding_name, args,
                data_type=data_type,
            )
        )
        if label_map is None:
            print(f"[{embedding_name}] No tiles collected. Skipping.")
            continue

        if data_type == "rectal":
            tissue_list = [("all", normal_f, normal_labels, normal_stems)]
        elif data_type == "prostate":
            tissue_list = [
                ("tumor", tumor_f, tumor_labels, tumor_stems),
                ("benign", normal_f, normal_labels, normal_stems),
            ]
        else:
            tissue_list = [
                ("tumor", tumor_f, tumor_labels, tumor_stems),
                ("normal", normal_f, normal_labels, normal_stems),
            ]

        for tissue_type, features, labels, stems in tissue_list:
            if features is None or labels is None or len(features) == 0:
                print(f"[{embedding_name}] No {tissue_type} tiles. Skipping {tissue_type} analysis.")
                continue
            feats, labs, filt_map = filter_by_min_slides_per_institution(
                features, labels, stems, label_map, args.min_slides_per_institution
            )
            if len(filt_map) < args.min_institutions:
                print(f"[{embedding_name}] {tissue_type}: after min_slides_per_institution filter, "
                      f"only {len(filt_map)} institution(s). Skipping (min_institutions={args.min_institutions}).")
                continue
            print(f"[{embedding_name}] {tissue_type}: {feats.shape[0]} tiles, {len(filt_map)} institutions: {list(filt_map.values())}")
            rows = compute_institution_bootstrapped_metrics(
                feats, labs, embedding_name, tissue_type, filt_map,
                n_bootstrap=args.n_bootstrap,
                bootstrap_fraction=args.bootstrap_fraction,
                pca_components=args.pca_components,
                seed=args.seed,
            )
            all_bootstrap_results.extend(rows)
            if rows:
                sil_vals = [r["silhouette_pca"] for r in rows if not np.isnan(r["silhouette_pca"])]
                db_vals = [r["davies_bouldin_pca"] for r in rows if not np.isnan(r["davies_bouldin_pca"])]
                print(f"  silhouette_pca: mean={np.mean(sil_vals):.4f}, n={len(rows)} iters")
                print(f"  davies_bouldin_pca: mean={np.mean(db_vals):.4f}, n={len(rows)} iters")

        # All-tissues combined (kidney/prostate only)
        if getattr(args, "all_tissues", False) and data_type in ("kidney", "prostate"):
            if (tumor_f is not None and normal_f is not None and
                    len(tumor_f) > 0 and len(normal_f) > 0):
                comb_f = np.concatenate([tumor_f, normal_f], axis=0)
                comb_labels = np.concatenate([tumor_labels, normal_labels])
                comb_stems = tumor_stems + normal_stems
                feats, labs, filt_map = filter_by_min_slides_per_institution(
                    comb_f, comb_labels, comb_stems, label_map, args.min_slides_per_institution
                )
                if len(filt_map) >= args.min_institutions:
                    print(f"[{embedding_name}] all (combined tissues): {feats.shape[0]} tiles, "
                          f"{len(filt_map)} institutions: {list(filt_map.values())}")
                    rows = compute_institution_bootstrapped_metrics(
                        feats, labs, embedding_name, "all", filt_map,
                        n_bootstrap=args.n_bootstrap,
                        bootstrap_fraction=args.bootstrap_fraction,
                        pca_components=args.pca_components,
                        seed=args.seed,
                    )
                    all_bootstrap_results_alltissues.extend(rows)
                    if rows:
                        sil_vals = [r["silhouette_pca"] for r in rows if not np.isnan(r["silhouette_pca"])]
                        db_vals = [r["davies_bouldin_pca"] for r in rows if not np.isnan(r["davies_bouldin_pca"])]
                        print(f"  silhouette_pca: mean={np.mean(sil_vals):.4f}, n={len(rows)} iters")
                        print(f"  davies_bouldin_pca: mean={np.mean(db_vals):.4f}, n={len(rows)} iters")

    if all_bootstrap_results:
        df_boot = pd.DataFrame(all_bootstrap_results)
        out_boot = output_dir / "institution_clustering_metrics_bootstrap.csv"
        df_boot.to_csv(out_boot, index=False)
        print(f"\nSaved bootstrap metrics to {out_boot}")

        comparison_results = run_wilcoxon_comparisons(all_bootstrap_results, min_institutions=args.min_institutions)
        if comparison_results:
            df_comp = pd.DataFrame(comparison_results)
            out_comp = output_dir / "institution_clustering_wilcoxon.csv"
            df_comp.to_csv(out_comp, index=False)
            print(f"Saved Wilcoxon comparisons to {out_comp}")
            print("\nStatistical comparison (Wilcoxon rank-sum):")
            for r in comparison_results[:20]:
                print(f"  {r['model1']} vs {r['model2']} ({r['tissue_type']}, {r['metric']}): p={r['p_value']:.4f}")
            if len(comparison_results) > 20:
                print(f"  ... and {len(comparison_results) - 20} more rows in CSV.")
        else:
            df_comp = None
        create_institution_barplots(df_boot, df_comp, output_dir)
        create_institution_all_metrics_barplots(df_boot, df_comp, output_dir)

    if all_bootstrap_results_alltissues:
        df_boot_at = pd.DataFrame(all_bootstrap_results_alltissues)
        out_boot_at = output_dir / "institution_clustering_metrics_bootstrap_alltissues.csv"
        df_boot_at.to_csv(out_boot_at, index=False)
        print(f"\nSaved all-tissues bootstrap metrics to {out_boot_at}")
        comparison_at = run_wilcoxon_comparisons(
            all_bootstrap_results_alltissues, min_institutions=args.min_institutions
        )
        df_comp_at = pd.DataFrame(comparison_at) if comparison_at else None
        if df_comp_at is not None:
            out_comp_at = output_dir / "institution_clustering_wilcoxon_alltissues.csv"
            df_comp_at.to_csv(out_comp_at, index=False)
            print(f"Saved all-tissues Wilcoxon comparisons to {out_comp_at}")
        create_institution_barplots(df_boot_at, df_comp_at, output_dir, output_suffix="_alltissues")
        create_institution_all_metrics_barplots(df_boot_at, df_comp_at, output_dir, output_suffix="_alltissues")

    print("\nDone.")


if __name__ == "__main__":
    main()
