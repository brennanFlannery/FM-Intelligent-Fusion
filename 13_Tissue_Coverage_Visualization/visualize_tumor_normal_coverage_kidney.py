#!/usr/bin/env python
"""
visualize_tumor_normal_coverage.py  (Kidney)

Read an aggregated summary CSV produced by aggregate_dice_results.py and produce
a figure with line plots (coverage vs percentile) and ranking bump charts for
the two tissue types present in kidney cancer data: Tumor and Normal.

Expected input columns (wide format, one row per model-percentile pair):
    model, percentile,
    mean_tumor_cov, std_tumor_cov, n_valid_tumor_cov,
    mean_normal_cov, std_normal_cov, n_valid_normal_cov,
    mean_pct_attn_in_tumor, std_pct_attn_in_tumor, n_valid_pct_attn_in_tumor,
    mean_pct_attn_in_normal, std_pct_attn_in_normal, n_valid_pct_attn_in_normal

Usage:
    python visualize_tumor_normal_coverage.py \\
        --input /path/to/dice_summary_all_models.csv \\
        --output kidney_tumor_normal.png
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory


# Tissue columns: (label, mean_col, std_col, n_col)
TISSUES = [
    ("Tumor",  "mean_tumor_cov",  "std_tumor_cov",  "n_valid_tumor_cov"),
    ("Normal", "mean_normal_cov", "std_normal_cov", "n_valid_normal_cov"),
]

# Attention-in-compartment: % of attention mask pixels in each tissue
TISSUES_PCT_ATTN = [
    ("Tumor",  "mean_pct_attn_in_tumor",  "std_pct_attn_in_tumor",  "n_valid_pct_attn_in_tumor"),
    ("Normal", "mean_pct_attn_in_normal", "std_pct_attn_in_normal", "n_valid_pct_attn_in_normal"),
]

REQUIRED_COLUMNS = [
    "model", "percentile",
    "mean_tumor_cov", "mean_normal_cov",
    "mean_pct_attn_in_tumor", "mean_pct_attn_in_normal",
]

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


def parse_args():
    p = argparse.ArgumentParser(description="Visualize tumor/normal coverage for kidney data.")
    p.add_argument("--input", "--csv", dest="input", type=str, required=True,
                   help="Aggregated summary CSV (output of aggregate_dice_results.py)")
    p.add_argument("--output", type=str, default=None,
                   help="Output path for figure (default: same dir as CSV, "
                        "kidney_tumor_normal_coverage.png)")
    p.add_argument("--figsize", type=str, default="18 4.7",
                   help="Figure size W H in inches (default: 18 4.7 — 4 panels wide)")
    p.add_argument("--dpi", type=int, default=150, help="DPI for raster output (default: 150)")
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf", "both"],
                   help="Output format (default: png)")
    return p.parse_args()


def get_model_style(model: str) -> dict:
    return MODEL_STYLE.get(model, {"color": "gray", "linestyle": "-"})


def get_model_order(models: list) -> list:
    """Return model order: PRUNED, NAIVE first, then the rest alphabetically."""
    priority = ["PRUNED", "NAIVE"]
    ordered = [m for m in priority if m in models]
    ordered += sorted([m for m in models if m not in priority])
    return ordered


def _broken_axis_transform(v):
    """Map benign/normal % (0-100) to y (0 to -100): 0-20 -> 0 to -80, 20-100 -> -80 to -100."""
    v = np.asarray(v, dtype=float)
    out = np.where(v <= 20, -80 * (v / 20), -80 - 20 * (v - 20) / 80)
    return out


def load_and_reshape(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load wide CSV and reshape to long format.
    Returns (df_coverage, df_pct_attn), each with columns:
    model, percentile, tissue, mean_coverage, std_coverage, n_valid.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    rows_cov = []
    rows_pct = []
    for _, row in df.iterrows():
        for label, mean_col, std_col, n_col in TISSUES:
            mean_val = row.get(mean_col, np.nan)
            std_val  = row.get(std_col,  np.nan)
            n_val    = row.get(n_col,    np.nan)
            rows_cov.append({
                "model":         row["model"],
                "percentile":    row["percentile"],
                "tissue":        label,
                "mean_coverage": mean_val,
                "std_coverage":  std_val,
                "n_valid":       n_val,
            })
        for label, mean_col, std_col, n_col in TISSUES_PCT_ATTN:
            mean_val = row.get(mean_col, np.nan)
            std_val  = row.get(std_col,  np.nan)
            n_val    = row.get(n_col,    np.nan)
            rows_pct.append({
                "model":         row["model"],
                "percentile":    row["percentile"],
                "tissue":        label,
                "mean_coverage": mean_val,
                "std_coverage":  std_val,
                "n_valid":       n_val,
            })
    return pd.DataFrame(rows_cov), pd.DataFrame(rows_pct)


def main():
    args = parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df, df_pct_attn = load_and_reshape(csv_path)
    models  = sorted(df["model"].unique())
    tissues = [t[0] for t in TISSUES]  # ["Tumor", "Normal"]

    if args.output is None:
        out_path = csv_path.parent / "kidney_tumor_normal_coverage.png"
    else:
        out_path = Path(args.output)
    out_stem = out_path.with_suffix("")

    figsize = tuple(float(x) for x in args.figsize.split())

    fig = plt.figure(figsize=figsize)
    # 2 rows: line plots (row 0) and bump charts (row 1)
    # 4 columns: coverage Tumor, coverage Normal, % in attn Tumor, % in attn Normal
    gs = GridSpec(2, 4, figure=fig, hspace=0.6, wspace=0.3,
                  height_ratios=[1.5, 1.8])

    # ---- Row 0: Line plots (cols 0-1 coverage, cols 2-3 % attention in tissue) ----
    line_axes = []
    for col_idx, tissue in enumerate(tissues):
        ax = fig.add_subplot(gs[0, col_idx])
        line_axes.append(ax)
        sub = df[df["tissue"] == tissue].sort_values("percentile")
        for model in models:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue
            x   = msub["percentile"].values
            y   = msub["mean_coverage"].values
            sty = get_model_style(model)
            ax.plot(x, y, color=sty["color"], linestyle=sty["linestyle"],
                    label=model, linewidth=2)
        ax.set_title(tissue, fontsize=11, fontweight="bold")
        ax.set_xlabel("Percentile cutoff")
        ax.set_ylabel("Mean coverage (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

    for col_idx, tissue in enumerate(tissues):
        ax = fig.add_subplot(gs[0, col_idx + 2])
        line_axes.append(ax)
        sub = df_pct_attn[df_pct_attn["tissue"] == tissue].sort_values("percentile")
        for model in models:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue
            x   = msub["percentile"].values
            y   = msub["mean_coverage"].values
            sty = get_model_style(model)
            ax.plot(x, y, color=sty["color"], linestyle=sty["linestyle"],
                    label=model, linewidth=2)
        ax.set_title(f"{tissue} (% in attn)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Percentile cutoff")
        ax.set_ylabel("Mean % of attention mask in tissue")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

    # Shared legend below line plots
    if line_axes:
        handles, labels = line_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(models),
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))

    # ---- Row 1: Bump (ranking) charts ----
    for col_idx, tissue in enumerate(tissues):
        ax = fig.add_subplot(gs[1, col_idx])
        sub = df[df["tissue"] == tissue].copy()
        sub["rank"] = sub.groupby("percentile")["mean_coverage"].rank(
            ascending=False, method="first"
        )
        for model in models:
            msub = sub[sub["model"] == model].sort_values("percentile")
            if msub.empty:
                continue
            sty = get_model_style(model)
            ax.plot(msub["percentile"], msub["rank"],
                    color=sty["color"], linestyle=sty["linestyle"],
                    label=model, linewidth=2, marker="o", markersize=4)
        ax.set_title(f"Rank: {tissue}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Percentile cutoff")
        ax.set_ylabel("Rank (1 = highest coverage)")
        ax.invert_yaxis()
        ax.set_yticks(np.arange(1, len(models) + 1))
        ax.set_yticklabels(np.arange(1, len(models) + 1))
        ax.grid(True, alpha=0.3)

    for col_idx, tissue in enumerate(tissues):
        ax = fig.add_subplot(gs[1, col_idx + 2])
        sub = df_pct_attn[df_pct_attn["tissue"] == tissue].copy()
        sub["rank"] = sub.groupby("percentile")["mean_coverage"].rank(
            ascending=False, method="first"
        )
        for model in models:
            msub = sub[sub["model"] == model].sort_values("percentile")
            if msub.empty:
                continue
            sty = get_model_style(model)
            ax.plot(msub["percentile"], msub["rank"],
                    color=sty["color"], linestyle=sty["linestyle"],
                    label=model, linewidth=2, marker="o", markersize=4)
        ax.set_title(f"Rank: {tissue} (% in attn)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Percentile cutoff")
        ax.set_ylabel("Rank (1 = highest % in tissue)")
        ax.invert_yaxis()
        ax.set_yticks(np.arange(1, len(models) + 1))
        ax.set_yticklabels(np.arange(1, len(models) + 1))
        ax.grid(True, alpha=0.3)

    plt.suptitle("Kidney Cancer — Tumor vs Normal Coverage by Model", fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if args.format in ("png", "both"):
        fig.savefig(str(out_stem) + ".png", dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out_stem}.png")
    if args.format in ("pdf", "both"):
        fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight")
        print(f"Saved {out_stem}.pdf")

    # ---- Double-bar figures: one per percentile (tumor up red, normal down green) ----
    model_order = get_model_order(list(df["model"].unique()))
    percentiles = sorted(df["percentile"].unique())
    n_models = len(model_order)
    x_pos = np.arange(n_models)
    bar_width = 0.4

    for p in percentiles:
        sub = df[df["percentile"] == p]
        tumor_ser = sub[sub["tissue"] == "Tumor"].set_index("model")["mean_coverage"]
        normal_ser = sub[sub["tissue"] == "Normal"].set_index("model")["mean_coverage"]
        tumor_vals = np.array([tumor_ser.get(m, np.nan) for m in model_order])
        normal_vals = np.array([normal_ser.get(m, np.nan) for m in model_order])
        tumor_vals = np.nan_to_num(tumor_vals, nan=0.0)
        normal_vals = np.nan_to_num(normal_vals, nan=0.0)

        use_broken = normal_vals.size == 0 or normal_vals.max() < 20

        fig_db, ax = plt.subplots(figsize=(max(8, n_models * 0.8), 2.5))
        ax.bar(x_pos, tumor_vals, bar_width, bottom=0, color="#FF0000", edgecolor="black", linewidth=0.8, label="Tumor")
        if use_broken:
            y_down = _broken_axis_transform(normal_vals)
            ax.bar(x_pos, y_down, bar_width, bottom=0, color="#00FF00", edgecolor="black", linewidth=0.8, label="Normal")
        else:
            ax.bar(x_pos, -normal_vals, bar_width, bottom=0, color="#00FF00", edgecolor="black", linewidth=0.8, label="Normal")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-100, 100)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([])
        # Single left axis: 100 at top, 0 at center, 100 at bottom; two labels (tumor top, normal bottom)
        ax.set_yticks([100, 80, 60, 40, 20, 0, -20, -40, -60, -80, -100])
        if use_broken:
            ax.set_yticklabels(["100", "80", "60", "40", "20", "0", "5", "10", "15", "20", "100"])
            # Draw "//" break symbol on top of the y-axis (spine at x=0), in the gap strip
            ax.plot([0, 0.012], [0.03, 0.06], "k-", linewidth=1, clip_on=False, transform=ax.transAxes)
            ax.plot([0, 0.012], [0.06, 0.09], "k-", linewidth=1, clip_on=False, transform=ax.transAxes)
        else:
            ax.set_yticklabels(["100", "80", "60", "40", "20", "0", "20", "40", "60", "80", "100"])
        ax.set_ylabel("% Tumor Coverage", fontsize=10)
        ax.yaxis.set_label_coords(-0.12, 0.78)
        ax.text(-0.18, 0.22, "% Normal coverage", rotation=90, transform=ax.transAxes, fontsize=10, va="center", ha="center")
        # Model color squares below x-axis (PRUNED=black, NAIVE=black+hatch, others=model color)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        sq_h = 0.12
        sq_w = 0.35
        for i, model in enumerate(model_order):
            sty = get_model_style(model)
            fc = sty["color"]
            rect = Rectangle((x_pos[i] - sq_w / 2, -0.14), sq_w, sq_h, transform=trans,
                             facecolor=fc, edgecolor="black", linewidth=0.5, clip_on=False)
            if model == "NAIVE":
                rect.set_hatch(".......")
                rect.set_facecolor("white")
                rect.set_edgecolor("black")
            ax.add_patch(rect)

        plt.tight_layout()

        db_stem = f"{out_stem}_doublebar_p{p}"
        if args.format in ("png", "both"):
            fig_db.savefig(f"{db_stem}.png", dpi=args.dpi, bbox_inches="tight")
            print(f"Saved {db_stem}.png")
        if args.format in ("pdf", "both"):
            fig_db.savefig(f"{db_stem}.pdf", bbox_inches="tight")
            print(f"Saved {db_stem}.pdf")
        plt.close(fig_db)


if __name__ == "__main__":
    main()
