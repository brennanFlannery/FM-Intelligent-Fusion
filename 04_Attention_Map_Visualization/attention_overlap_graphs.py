#!/usr/bin/env python3
"""
Generate pairwise upper-triangle heatmaps of Mean Dice at selected percentiles.

• Input: CSV with columns: Model1, Model2, Percentile, Mean_Dice
• Output: One PNG per requested percentile (default: 50, 70, 90)
• Design: matplotlib-only; one chart per figure; 'hot' colormap; colorbar outside; no gridlines; shared color scale across figures

Example:
    python make_percentile_heatmaps.py \
        --csv path/to/data.csv \
        --percentiles 50 70 90 \
        --outdir ./figs \
        --title-prefix "Pairwise Mean Dice" \
        --sort-by 70

Notes:
- Upper triangle only (i<j) to avoid duplicated pairs.
- Shared vmin/vmax computed from all requested percentiles for fair comparison.
- Optional: sort model order by average Mean Dice at a specified percentile (e.g., --sort-by 70).
- Optional: annotate values in each cell (use --annotate) — small, readable text.
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

REQUIRED_COLS = ["Model1", "Model2", "Percentile", "Mean_Dice"]


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        _eprint(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        _eprint(f"Could not read CSV with default settings: {e}\nTrying with engine='python' and flexible separators...")
        df = pd.read_csv(csv_path, engine='python')
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        _eprint(f"ERROR: CSV is missing required columns: {missing}")
        _eprint(f"Found columns: {list(df.columns)}")
        sys.exit(1)
    # Ensure proper dtypes
    df = df.copy()
    df["Percentile"] = pd.to_numeric(df["Percentile"], errors="coerce")
    df["Mean_Dice"] = pd.to_numeric(df["Mean_Dice"], errors="coerce")
    # Drop rows with NaNs in critical fields
    df = df.dropna(subset=["Model1", "Model2", "Percentile", "Mean_Dice"]).reset_index(drop=True)
    return df


def build_upper_triangle_matrix(df: pd.DataFrame, percentile: float, model_order: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    sub = df[df["Percentile"] == percentile].copy()
    if sub.empty:
        _eprint(f"WARNING: No rows for Percentile={percentile}")
    models = sorted(set(sub["Model1"]).union(set(sub["Model2"])))
    if model_order is not None:
        # Keep only models present in this subset; preserve specified order
        models = [m for m in model_order if m in models]
    idx = {m: i for i, m in enumerate(models)}
    M = np.full((len(models), len(models)), np.nan)
    for _, r in sub.iterrows():
        # Get both model names (don't sort yet, to preserve original relationship)
        m1, m2 = str(r["Model1"]), str(r["Model2"])
        if m1 not in idx or m2 not in idx:
            continue
        i, j = idx[m1], idx[m2]
        val = float(r["Mean_Dice"])
        # Fill both symmetric positions so the matrix is complete
        M[i, j] = val
        M[j, i] = val
    return M, models


def compute_model_order_for_sorting(df: pd.DataFrame, percentile: float) -> List[str]:
    sub = df[df["Percentile"] == percentile].copy()
    if sub.empty:
        return sorted(set(df["Model1"]).union(set(df["Model2"])))
    # Build a symmetric table of averages per model (mean across pairings)
    models = sorted(set(sub["Model1"]).union(set(sub["Model2"])))
    scores = {m: [] for m in models}
    for _, r in sub.iterrows():
        a, b = str(r["Model1"]), str(r["Model2"]) 
        val = float(r["Mean_Dice"])
        scores[a].append(val)
        scores[b].append(val)
    avg = {m: (np.mean(v) if len(v) else np.nan) for m, v in scores.items()}
    # Sort descending by avg score; tie-breaker = name
    order = sorted(models, key=lambda m: (-(avg[m] if not np.isnan(avg[m]) else -np.inf), m))
    return order


def plot_upper_triangle_heatmap(M: np.ndarray, models: List[str], title: str, outpath: str, vmin: float, vmax: float,
                                annotate: bool = False) -> None:
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()

    im = ax.imshow(M, origin="upper", cmap=plt.cm.plasma, norm=Normalize(vmin=vmin, vmax=vmax))

    # tick labels
    n = len(models)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticklabels(models)
    
    # turn off grid
    ax.grid(False)

    # mask lower triangle
    for i in range(n):
        for j in range(n):
            if i >= j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="white", linewidth=0))

    # keep subtle outer frame, no internal gridlines were added
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    ax.set_title(title)

    # colorbar outside
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.4)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Mean Dice")

    # optional annotations (centered, 2–3 decimals)
    if annotate:
        for i in range(n):
            for j in range(n):
                if i < j and not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Percentile-specific Mean Dice heatmaps (upper triangle)")
    p.add_argument("--csv", required=True, help="Path to input CSV with columns: Model1, Model2, Percentile, Mean_Dice")
    p.add_argument("--percentiles", nargs="*", type=float, default=[50, 70, 90], help="Percentiles to plot (e.g., 50 70 90)")
    p.add_argument("--outdir", default="./figs", help="Directory to write PNGs")
    p.add_argument("--title-prefix", default="Pairwise Mean Dice", help="Title prefix for figures")
    p.add_argument("--sort-by", type=float, default=None, help="If set, sort model order by average Mean Dice at this percentile")
    p.add_argument("--annotate", action="store_true", help="Overlay numeric Mean Dice in cells")
    args = p.parse_args()

    df = load_data(args.csv)

    # Decide a model order (optional sorting)
    model_order = None
    if args.sort_by is not None:
        model_order = compute_model_order_for_sorting(df, args.sort_by)
        _eprint(f"Model order sorted by average Mean Dice @ p={args.sort_by}: {model_order}")

    # Build matrices for requested percentiles and collect values for global vmin/vmax
    matrices = []
    all_vals = []
    # Use a consistent union of models across selected percentiles if sorting is requested
    if model_order is None:
        # derive model list from full dataset (stable ordering)
        all_models = sorted(set(df["Model1"]).union(set(df["Model2"])))
    else:
        all_models = model_order

    for pctl in args.percentiles:
        M, models = build_upper_triangle_matrix(df, pctl, model_order=all_models)
        matrices.append((pctl, M, models))
        vals = M[~np.isnan(M)]
        if vals.size:
            all_vals.append(vals)

    if not all_vals:
        _eprint("ERROR: No valid Mean_Dice values found for the requested percentiles.")
        sys.exit(1)

    vmin = float(np.min(np.concatenate(all_vals)))
    vmax = float(np.max(np.concatenate(all_vals)))
    _eprint(f"Global color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

    # Plot
    for pctl, M, models in matrices:
        title = f"{args.title_prefix} @ {int(pctl)}th Percentile"
        fname = f"heatmap_mean_dice_{int(pctl)}.png"
        outpath = os.path.join(args.outdir, fname)
        plot_upper_triangle_heatmap(M, models, title, outpath, vmin=vmin, vmax=vmax, annotate=args.annotate)
        _eprint(f"Wrote: {outpath}")


if __name__ == "__main__":
    main()
