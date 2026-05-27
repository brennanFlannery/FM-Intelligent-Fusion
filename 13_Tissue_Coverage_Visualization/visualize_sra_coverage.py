#!/usr/bin/env python
"""
visualize_sra_coverage.py

Read SRA coverage summary CSV (from aggregate_dice_results.py --include_sra) and produce
a single multi-panel figure: heatmaps (models x tissues at each percentile), line plots
(coverage vs percentile by tissue), and ranking bump chart(s).

Usage:
    python visualize_sra_coverage.py --input path/to/dice_summary_all_models_sra.csv --output summary.png
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REQUIRED_COLUMNS = ["model", "percentile", "class", "mean_coverage", "std_coverage", "n_valid"]
TISSUE_ORDER = ["ADI", "CSTR", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def parse_args():
    p = argparse.ArgumentParser(description="Visualize SRA coverage summary CSV.")
    p.add_argument("--input", "--csv", dest="input", type=str, required=True,
                   help="Path to SRA summary CSV (e.g. dice_summary_all_models_sra.csv)")
    p.add_argument("--output", type=str, default=None,
                   help="Output path for figure (default: same dir as CSV, sra_coverage_summary.png)")
    p.add_argument("--figsize", type=str, default="16 14",
                   help="Figure size W H in inches (default: 16 14)")
    p.add_argument("--dpi", type=int, default=150, help="DPI for raster output (default: 150)")
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf", "both"],
                   help="Output format (default: png)")
    p.add_argument("--bump_tissues", type=str, nargs="*", default=["TUM"],
                   help="Tissues for ranking bump chart (default: TUM)")
    return p.parse_args()


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


# Fixed model colors and line styles (matching pruning_visualization.py palette)
MODEL_STYLE = {
    # Versioned names (used by rectal/prostate DiceResults dirs)
    "CONCH_V15": {"color": "#0070C0", "linestyle": "-"},
    "MUSK":      {"color": "#FFC000", "linestyle": "-"},
    "VIRCHOW2":  {"color": "#7030A0", "linestyle": "-"},
    "HOPTIMUS1": {"color": "#C00000", "linestyle": "-"},
    "GIGAPATH":  {"color": "#00B050", "linestyle": "-"},
    "NAIVE":     {"color": "black",   "linestyle": "--"},
    "PRUNED":    {"color": "black",   "linestyle": "-"},
    # Short-name aliases (used by kidney TumorAttentionAnalysis dirs)
    "CONCH":     {"color": "#0070C0", "linestyle": "-"},
    "VIRCHOW":   {"color": "#7030A0", "linestyle": "-"},
    "HOPTIMUS":  {"color": "#C00000", "linestyle": "-"},
}


def get_model_styles(models):
    """Return dict model -> {color, linestyle}. Fallback gray/solid for unknown models."""
    return {
        m: MODEL_STYLE.get(m, {"color": "gray", "linestyle": "-"})
        for m in models
    }


def clean_axis(ax):
    """Remove axis lines (spines) and tick labels from subplot."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def main():
    args = parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = load_and_validate(csv_path)
    models = sorted(df["model"].unique())
    classes = [c for c in TISSUE_ORDER if c in df["class"].unique()]
    if len(classes) < len(df["class"].unique()):
        extra = set(df["class"].unique()) - set(classes)
        classes = classes + sorted(extra)
    model_styles = get_model_styles(models)

    if args.output is None:
        out_path = csv_path.parent / "sra_coverage_summary.png"
    else:
        out_path = Path(args.output)
    out_stem = out_path.with_suffix("")

    figsize = tuple(float(x) for x in args.figsize.split())
    # Rank (bump) charts for all tissue types
    bump_tissues = classes

    # GridSpec: 3x3 line plots then 3x3 bump charts (all tissues)
    n_line_rows = (len(classes) + 2) // 3
    n_bump_rows = (len(bump_tissues) + 2) // 3
    n_rows = n_line_rows + n_bump_rows
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.6, wspace=0.3,
                  height_ratios=[1.5] * n_line_rows + [1.8] * n_bump_rows, width_ratios=[1, 1, 1])

    # ---- Line plots (all tissues) ----
    line_axes = []
    for i, tissue in enumerate(classes):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, col])
        line_axes.append(ax)
        sub = df[df["class"] == tissue].sort_values("percentile")
        for model in models:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue
            x = msub["percentile"].values
            y = msub["mean_coverage"].values
            sty = model_styles[model]
            ax.plot(x, y, color=sty["color"], linestyle=sty["linestyle"], label=model, linewidth=2)
        ax.set_title(tissue)
        ax.set_xlabel("Percentile")
        ax.set_ylabel("Mean coverage (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)
        clean_axis(ax)

    if line_axes:
        handles, labels = line_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(models), fontsize=9,
                  bbox_to_anchor=(0.5, -0.02))

    # ---- Bump charts (all tissues) ----
    for i, tissue in enumerate(bump_tissues):
        if tissue not in df["class"].values:
            continue
        row, col = n_line_rows + i // 3, i % 3
        ax = fig.add_subplot(gs[row, col])
        sub = df[df["class"] == tissue].copy()
        sub["rank"] = sub.groupby("percentile")["mean_coverage"].rank(ascending=False, method="first")
        for model in models:
            msub = sub[sub["model"] == model].sort_values("percentile")
            if msub.empty:
                continue
            sty = model_styles[model]
            ax.plot(msub["percentile"], msub["rank"], color=sty["color"], linestyle=sty["linestyle"],
                    label=model, linewidth=2, marker="o", markersize=4)
        ax.set_title(f"Rank: {tissue}")
        ax.set_xlabel("Percentile")
        ax.set_ylabel("Rank (1 = highest)")
        ax.invert_yaxis()
        ax.set_yticks(np.arange(1, len(models) + 1))
        ax.set_yticklabels(np.arange(1, len(models) + 1))
        # Legend outside to the right only on the last panel
        if i == len(bump_tissues) - 1:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    if args.format in ("png", "both"):
        fig.savefig(str(out_stem) + ".png", dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out_stem}.png")
    if args.format in ("pdf", "both"):
        fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight")
        print(f"Saved {out_stem}.pdf")


if __name__ == "__main__":
    main()
