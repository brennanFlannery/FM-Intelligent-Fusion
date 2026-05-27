#!/usr/bin/env python
"""
aggregated_ranking_plot.py

Read an aggregated summary CSV (output of aggregate_dice_results.py) and produce
a single bump chart of aggregated rank across percentiles. Aggregation: at each
percentile, rank models by tumor coverage and by normal/benign coverage separately;
add the two ranks; re-rank by this sum (lowest = best). Supports kidney
(tumor + normal) and prostate (tumor + benign).

Expected input columns (same as visualize_tumor_normal_coverage.py):
    model, percentile,
    mean_tumor_cov, (std/n_valid optional)
    and either mean_normal_cov (kidney) or mean_benign_cov (prostate)

Usage:
    python aggregated_ranking_plot.py --input /path/to/dice_summary_all_models.csv
    python aggregated_ranking_plot.py --input /path/to/dice_summary_all_models.csv --data_type prostate --output agg_rank.png
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_STYLE = {
    "CONCH_V15": {"color": "#0070C0", "linestyle": "-"},
    "MUSK":      {"color": "#FFC000", "linestyle": "-"},
    "VIRCHOW2":  {"color": "#7030A0", "linestyle": "-"},
    "HOPTIMUS1": {"color": "#C00000", "linestyle": "-"},
    "GIGAPATH":  {"color": "#00B050", "linestyle": "-"},
    "NAIVE":     {"color": "black",   "linestyle": "--"},
    "PRUNED":    {"color": "black",   "linestyle": "-"},
    "CONCH":     {"color": "#0070C0", "linestyle": "-"},
    "VIRCHOW":   {"color": "#7030A0", "linestyle": "-"},
    "HOPTIMUS":  {"color": "#C00000", "linestyle": "-"},
}


def get_model_style(model: str) -> dict:
    return MODEL_STYLE.get(model, {"color": "gray", "linestyle": "-"})


def detect_data_type(df: pd.DataFrame) -> str:
    """Infer kidney vs prostate from CSV columns."""
    if "mean_normal_cov" in df.columns:
        return "kidney"
    if "mean_benign_cov" in df.columns:
        return "prostate"
    raise ValueError(
        "Could not auto-detect data type: CSV must contain either "
        "'mean_normal_cov' (kidney) or 'mean_benign_cov' (prostate). "
        "Use --data_type kidney or --data_type prostate."
    )


def load_and_compute_aggregated_rank(
    csv_path: Path,
    data_type: str,
) -> pd.DataFrame:
    """
    Load wide CSV and add aggregated rank per (model, percentile).
    Requires: model, percentile, mean_tumor_cov, and mean_normal_cov (kidney) or mean_benign_cov (prostate).
    """
    df = pd.read_csv(csv_path)
    required = ["model", "percentile", "mean_tumor_cov"]
    if data_type == "kidney":
        other_col = "mean_normal_cov"
    else:
        other_col = "mean_benign_cov"
    required.append(other_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Per-percentile: rank by tumor (desc), rank by other (desc) then reverse so low coverage = best, sum, then rank by sum (asc)
    out_rows = []
    for pct, sub in df.groupby("percentile", sort=True):
        sub = sub.copy()
        sub["rank_tumor"] = sub["mean_tumor_cov"].rank(ascending=False, method="first")
        sub["rank_other"] = sub[other_col].rank(ascending=False, method="first")
        n = len(sub)
        sub["rank_other"] = n + 1 - sub["rank_other"]
        sub["combined_sum"] = sub["rank_tumor"] + sub["rank_other"]
        sub["agg_rank"] = sub["combined_sum"].rank(ascending=True, method="first").astype(int)
        out_rows.append(sub[["model", "percentile", "combined_sum", "agg_rank"]])
    return pd.concat(out_rows, ignore_index=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregated tumor + normal/benign ranking bump chart from dice summary CSV."
    )
    p.add_argument("--input", "--csv", dest="input", type=str, required=True,
                   help="Aggregated summary CSV (output of aggregate_dice_results.py)")
    p.add_argument("--output", type=str, default=None,
                   help="Output path for figure (default: next to CSV, aggregated_ranking_tumor_normal.png or _tumor_benign.png)")
    p.add_argument("--data_type", type=str, default=None, choices=["kidney", "prostate"],
                   help="kidney (tumor+normal) or prostate (tumor+benign); auto-detect from columns if omitted")
    p.add_argument("--figsize", type=str, default="10.7 2.8",
                   help="Figure size W H in inches (default: 10.7 2.8 — double width of one ranking panel)")
    p.add_argument("--dpi", type=int, default=150, help="DPI for raster output (default: 150)")
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf", "both"],
                   help="Output format (default: png)")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    data_type = args.data_type or detect_data_type(df_raw)
    df = load_and_compute_aggregated_rank(csv_path, data_type)

    models = sorted(df["model"].unique())
    n_models = len(models)

    if args.output is None:
        stem = "aggregated_ranking_tumor_benign" if data_type == "prostate" else "aggregated_ranking_tumor_normal"
        out_path = csv_path.parent / f"{stem}.png"
    else:
        out_path = Path(args.output)
    out_stem = out_path.with_suffix("")

    figsize = tuple(float(x) for x in args.figsize.split())
    fig, ax = plt.subplots(figsize=figsize)

    for model in models:
        msub = df[df["model"] == model].sort_values("percentile")
        if msub.empty:
            continue
        sty = get_model_style(model)
        ax.plot(
            msub["percentile"],
            msub["agg_rank"],
            color=sty["color"],
            linestyle=sty["linestyle"],
            label=model,
            linewidth=2,
            marker="o",
            markersize=4,
        )

    title = "Aggregated rank (Tumor + Benign)" if data_type == "prostate" else "Aggregated rank (Tumor + Normal)"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Percentile cutoff")
    ax.set_ylabel("Rank (1 = best)")
    ax.invert_yaxis()
    ax.set_yticks(np.arange(1, n_models + 1))
    ax.set_yticklabels(np.arange(1, n_models + 1))
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if args.format in ("png", "both"):
        fig.savefig(str(out_stem) + ".png", dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out_stem}.png")
    if args.format in ("pdf", "both"):
        fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight")
        print(f"Saved {out_stem}.pdf")

    # Save CSV with aggregated scores before re-ranking (combined_sum) and after re-ranking (agg_rank)
    scores_csv = out_stem.parent / f"{out_stem.name}_scores.csv"
    df.to_csv(scores_csv, index=False)
    print(f"Saved {scores_csv}")


if __name__ == "__main__":
    main()
