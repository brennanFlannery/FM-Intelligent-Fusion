#!/usr/bin/env python
"""
aggregate_dice_results.py

Read through all dice_results.xlsx files from heatmap_tumor_dice.py outputs
and create a combined summary table with metrics for each model at each threshold.

With --include_sra, also aggregates the sra_coverage_summary sheet (from rectal_new
runs) and writes {output_name}_sra.csv plus SRA sheets in the Excel.

Usage:
    python aggregate_dice_results.py --dice_results_dir /path/to/DiceResults
    python aggregate_dice_results.py --dice_results_dir /path/to/DiceResults --include_sra
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def aggregate_dice_results(dice_results_dir: Path, output_name: str = "dice_summary_all_models", include_sra: bool = False):
    """
    Aggregate dice results from all model subdirectories.

    Args:
        dice_results_dir: Parent directory containing model subdirectories
        output_name: Base name for output files (without extension)
        include_sra: If True, also read and aggregate sra_coverage_summary sheet from each model
    """
    dice_results_dir = Path(dice_results_dir)
    
    if not dice_results_dir.exists():
        raise FileNotFoundError(f"Dice results directory not found: {dice_results_dir}")
    
    # Find all subdirectories (model directories)
    model_dirs = [d for d in dice_results_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        raise ValueError(f"No model subdirectories found in {dice_results_dir}")
    
    all_summaries = []
    all_sra_summaries = []

    # Read summary sheet from each model's Excel file
    for model_dir in tqdm(sorted(model_dirs), desc="Reading model results"):
        model_name = model_dir.name
        excel_path = model_dir / "dice_results.xlsx"

        if not excel_path.exists():
            print(f"Warning: {excel_path} not found, skipping {model_name}")
            continue

        try:
            # Read the "summary" sheet
            df_summary = pd.read_excel(excel_path, sheet_name="summary")
            # Add model column
            df_summary.insert(0, "model", model_name)
            all_summaries.append(df_summary)
        except Exception as e:
            print(f"Warning: Failed to read {excel_path}: {e}")
            continue

        # Optionally read SRA coverage summary (rectal_new runs)
        if include_sra:
            try:
                df_sra = pd.read_excel(excel_path, sheet_name="sra_coverage_summary")
                df_sra.insert(0, "model", model_name)
                all_sra_summaries.append(df_sra)
            except Exception:
                pass  # Sheet missing or read failed; skip this model for SRA
    
    if not all_summaries:
        raise ValueError("No valid summary data found in any model directory")
    
    # Combine all summaries
    combined_df = pd.concat(all_summaries, ignore_index=True)
    
    # Reorder columns: model, percentile, then all metrics
    metric_cols = [c for c in combined_df.columns if c not in ["model", "percentile"]]
    combined_df = combined_df[["model", "percentile"] + metric_cols]
    
    # Sort by model name, then percentile
    combined_df = combined_df.sort_values(["model", "percentile"]).reset_index(drop=True)
    
    # Save as CSV
    csv_path = dice_results_dir / f"{output_name}.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")
    
    # Create pivot tables for each metric (models as rows, percentiles as columns)
    metric_cols = [c for c in combined_df.columns if c not in ["model", "percentile"]]
    pivot_tables = {}

    for metric in metric_cols:
        # Check if metric has any non-null values
        if combined_df[metric].notna().any():
            # Pivot: model as index, percentile as columns, metric as values
            pivot_df = combined_df.pivot_table(
                index="model",
                columns="percentile",
                values=metric,
                aggfunc="first"  # Should only be one value per model-percentile combo
            )
            # Sort models alphabetically
            pivot_df = pivot_df.sort_index()
            # Sort percentiles numerically
            pivot_df = pivot_df.sort_index(axis=1)
            pivot_tables[metric] = pivot_df

    # SRA aggregation (when include_sra and at least one model had the sheet)
    combined_sra = None
    sra_pivot_tables = {}
    if include_sra and all_sra_summaries:
        combined_sra = pd.concat(all_sra_summaries, ignore_index=True)
        sra_cols = ["model", "percentile", "class", "mean_coverage", "std_coverage", "n_valid"]
        combined_sra = combined_sra[[c for c in sra_cols if c in combined_sra.columns]]
        combined_sra = combined_sra.sort_values(["model", "percentile", "class"]).reset_index(drop=True)
        sra_csv_path = dice_results_dir / f"{output_name}_sra.csv"
        combined_sra.to_csv(sra_csv_path, index=False)
        print(f"Saved SRA CSV: {sra_csv_path}")
        # Pivot per class: model x percentile -> mean_coverage
        if "class" in combined_sra.columns and "mean_coverage" in combined_sra.columns:
            for cls in sorted(combined_sra["class"].unique()):
                sub = combined_sra[combined_sra["class"] == cls]
                if sub["mean_coverage"].notna().any():
                    pivot_sra = sub.pivot_table(index="model", columns="percentile", values="mean_coverage", aggfunc="first")
                    pivot_sra = pivot_sra.sort_index().sort_index(axis=1)
                    sheet_name = f"sra_mean_cov_{cls}"[:31]
                    sra_pivot_tables[sheet_name] = pivot_sra
    elif include_sra:
        print("No SRA coverage summary sheets found in any model; skipping SRA output.")

    # Try to save as Excel as well
    excel_path = dice_results_dir / f"{output_name}.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Original combined table
            combined_df.to_excel(writer, index=False, sheet_name="summary_all_models")

            # Add pivot tables as separate sheets
            for metric, pivot_df in pivot_tables.items():
                # Excel sheet names are limited to 31 characters
                sheet_name = metric[:31] if len(metric) > 31 else metric
                pivot_df.to_excel(writer, sheet_name=sheet_name)

            # SRA combined table and pivot sheets
            if combined_sra is not None and not combined_sra.empty:
                combined_sra.to_excel(writer, index=False, sheet_name="sra_coverage_all_models")
                for sheet_name, pivot_df in sra_pivot_tables.items():
                    pivot_df.to_excel(writer, sheet_name=sheet_name)

        print(f"Saved Excel: {excel_path}")
        print(f"  - Main sheet: 'summary_all_models'")
        print(f"  - Pivot sheets: {len(pivot_tables)} metrics")
        for metric in sorted(pivot_tables.keys()):
            print(f"    * {metric}")
        if combined_sra is not None and not combined_sra.empty:
            print(f"  - SRA sheet: 'sra_coverage_all_models'")
            print(f"  - SRA pivot sheets: {len(sra_pivot_tables)} classes")
            for name in sorted(sra_pivot_tables.keys()):
                print(f"    * {name}")
    except Exception as e:
        print(f"Note: Could not save Excel file (openpyxl may not be installed): {e}")
        print("CSV file saved successfully.")
    
    # Print summary statistics
    print(f"\nAggregated results from {len(all_summaries)} models:")
    print(f"Total rows: {len(combined_df)}")
    print(f"\nModels included: {', '.join(sorted(combined_df['model'].unique()))}")
    print(f"Percentiles: {sorted(combined_df['percentile'].unique())}")
    print(f"\nMetrics with data: {len(pivot_tables)}/{len(metric_cols)}")
    if include_sra and combined_sra is not None and not combined_sra.empty:
        n_sra_models = combined_sra["model"].nunique()
        print(f"\nSRA coverage: aggregated from {n_sra_models} model(s); see '{output_name}_sra.csv' and sheet 'sra_coverage_all_models'.")

    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate dice results from all model subdirectories"
    )
    parser.add_argument(
        "--dice_results_dir",
        type=str,
        required=True,
        help="Path to DiceResults parent directory containing model subdirectories"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="dice_summary_all_models",
        help="Base name for output files (default: dice_summary_all_models)"
    )
    parser.add_argument(
        "--include_sra",
        action="store_true",
        help="Also aggregate SRA coverage summary from each model (sheet sra_coverage_summary); use when aggregating rectal_new dice results."
    )

    args = parser.parse_args()

    aggregate_dice_results(
        dice_results_dir=Path(args.dice_results_dir),
        output_name=args.output_name,
        include_sra=args.include_sra
    )


if __name__ == "__main__":
    main()

