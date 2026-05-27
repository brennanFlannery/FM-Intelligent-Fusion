#!/usr/bin/env python3
"""
visualize_feature_selection_slide.py

Create a publication-quality figure showing slide-level feature selection across diseases and thresholds.

Figure structure:
- 7 columns (correlation thresholds: 0.1 - 0.7)
- 2 rows (top: 0.1-0.4, bottom: 0.5-0.7)
- Each cell: 4 horizontal bars (Kidney, Prostate, Rectum, Common) stacked vertically

Shows which features were selected (colored marks on unified feature space across slide-level models).
Models are inferred dynamically from feature_sources.npy (e.g., slide_features_chief/madeleine/titan).

Example:
  python visualize_feature_selection_slide.py \
    --kidney_features /scratch/pioneer/users/bxf169/NewFusionTracking_Slide/Kidney \
    --prostate_features /scratch/pioneer/users/bxf169/NewFusionTracking_Slide/Prostate \
    --rectal_features /scratch/pioneer/users/bxf169/NewFusionTracking_Slide/Rectum \
    --output_dir /scratch/pioneer/users/bxf169/FeatureSelectionSHAPVis_Slide
"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize slide-level feature selection across diseases"
    )
    parser.add_argument(
        "--kidney_features", type=str, required=True,
        help="Parent directory for kidney combined slide features with feature_sources.npy (thr_0p1, thr_0p2, etc.)"
    )
    parser.add_argument(
        "--prostate_features", type=str, required=True,
        help="Parent directory for prostate combined slide features with feature_sources.npy"
    )
    parser.add_argument(
        "--rectal_features", type=str, required=True,
        help="Parent directory for rectal combined slide features with feature_sources.npy"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save output figure"
    )
    parser.add_argument(
        "--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated list of correlation thresholds"
    )
    return parser.parse_args()


def infer_model_dims_across_all(feature_dirs: dict, threshold_strs: list):
    """Infer model names and original dimensions across all diseases and thresholds.

    Aggregates the maximum observed original feature index per model across any
    available feature_sources.npy, ensuring models like Titan are included even if
    absent at some thresholds/diseases.

    Returns:
        model_dims: dict[str,int]
        model_order: list[str] in order of first appearance across scan
    """
    model_dims: dict[str, int] = {}
    model_order: list[str] = []
    seen = set()

    # Scan all diseases and thresholds
    for disease, base_dir in feature_dirs.items():
        for thr_str in threshold_strs:
            thr_dir = os.path.join(base_dir, f'thr_{thr_str}')
            feat_src_path = os.path.join(thr_dir, 'feature_sources.npy')
            if not os.path.exists(feat_src_path):
                continue
            try:
                feat_sources = np.load(feat_src_path, allow_pickle=True)
            except Exception:
                continue
            if not hasattr(feat_sources, 'dtype') or not feat_sources.dtype.names:
                continue
            models = feat_sources['model']
            indices = feat_sources['feature_index']
            # Track first appearance order
            for m in models:
                if m not in seen:
                    seen.add(m)
                    model_order.append(m)
            # Update dims by max original index + 1
            for m in np.unique(models):
                mask = models == m
                if np.any(mask):
                    max_idx = int(np.max(indices[mask]))
                    dim = max_idx + 1
                    if m not in model_dims or dim > model_dims[m]:
                        model_dims[m] = dim

    if not model_dims:
        raise RuntimeError("Could not infer any models from feature_sources.npy across provided inputs")

    return model_dims, model_order


def create_unified_feature_mapping(model_dims, model_order):
    mapping = {}
    model_ranges = {}
    current_idx = 0
    for model in model_order:
        dim = int(model_dims.get(model, 0))
        start_idx = current_idx
        end_idx = current_idx + dim
        model_ranges[model] = (start_idx, end_idx)
        for feat_idx in range(dim):
            mapping[(model, feat_idx)] = current_idx
            current_idx += 1
    return mapping, current_idx, model_ranges


def load_selected_features(feature_dir, threshold_str, unified_mapping, total_features):
    thr_dir = os.path.join(feature_dir, f'thr_{threshold_str}')
    feat_src_path = os.path.join(thr_dir, 'feature_sources.npy')
    if not os.path.exists(feat_src_path):
        raise FileNotFoundError(f"Feature sources not found: {feat_src_path}")

    feat_sources = np.load(feat_src_path, allow_pickle=True)
    selected_unified = np.zeros(total_features, dtype=bool)

    if feat_sources.dtype.names:
        models = feat_sources['model']
        indices = feat_sources['feature_index']
        for model, idx in zip(models, indices):
            key = (model, int(idx))
            if key in unified_mapping:
                selected_unified[unified_mapping[key]] = True
    return selected_unified


def create_visualization(all_selections, thresholds, model_ranges, total_features, output_path):
    # Publication quality
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Slide model colors (from compare_slide_model_performance.py)
    slide_model_colors = {
        'titan': '#F6C6AD',
        'madeleine': '#A6CAEC',
        'chief': '#B4E5A2',
    }

    # Map feature index to slide model color
    feature_to_color = {}
    for model, (start, end) in model_ranges.items():
        lower = model.lower()
        if 'titan' in lower:
            color = slide_model_colors['titan']
        elif 'madeleine' in lower:
            color = slide_model_colors['madeleine']
        elif 'chief' in lower:
            color = slide_model_colors['chief']
        else:
            color = '#808080'  # unknown
        for i in range(start, end):
            feature_to_color[i] = color

    diseases = ['Kidney', 'Prostate', 'Rectum']

    # 2 rows: 0.1-0.4 on row 0; 0.5-0.7 on row 1
    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.1, left=0.05, right=0.98, top=0.95, bottom=0.08)

    for idx, threshold in enumerate(thresholds):
        if idx < 4:
            row_idx, col_idx = 0, idx
        else:
            row_idx, col_idx = 1, idx - 4
        ax = fig.add_subplot(gs[row_idx, col_idx])

        # Bar layout
        bar_height = 0.22
        y_positions = [0.78, 0.55, 0.32, 0.09]
        disease_labels = ['Kidney', 'Prostate', 'Rectum', 'Common']

        kidney_sel = all_selections['Kidney'][threshold]
        prostate_sel = all_selections['Prostate'][threshold]
        rectum_sel = all_selections['Rectum'][threshold]
        common_sel = kidney_sel & prostate_sel & rectum_sel

        # Background bars
        for disease_idx, _ in enumerate(diseases):
            y_pos = y_positions[disease_idx]
            ax.add_patch(mpatches.Rectangle((0, y_pos - bar_height/2), total_features, bar_height,
                                            facecolor='lightgray', edgecolor='black', linewidth=0.5))

        # Draw disease selections (wider rectangles for visibility)
        for disease_idx, disease in enumerate(diseases):
            y_pos = y_positions[disease_idx]
            selected = all_selections[disease][threshold]
            for i in range(total_features):
                if selected[i]:
                    feat_color = feature_to_color.get(i, '#FF0000')
                    ax.add_patch(mpatches.Rectangle((i-2, y_pos - bar_height/2), 5, bar_height,
                                                    facecolor=feat_color, edgecolor='none'))

        # Common bar
        y_pos = y_positions[3]
        ax.add_patch(mpatches.Rectangle((0, y_pos - bar_height/2), total_features, bar_height,
                                        facecolor='lightgray', edgecolor='black', linewidth=0.5))
        for i in range(total_features):
            if common_sel[i]:
                ax.add_line(plt.Line2D([i, i], [y_pos - bar_height/2, y_pos + bar_height/2],
                                       color='black', linewidth=1.0))

        # Axes formatting
        ax.set_xlim(-5, total_features + 5)
        ax.set_ylim(0, 1)
        if col_idx == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels(disease_labels, fontsize=14, fontweight='bold')
        else:
            ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f'Threshold = {threshold:.1f}', fontsize=16, fontweight='bold', pad=10)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Legend at bottom
    legend_items = []
    legend_labels = []
    for model, (start, end) in sorted(model_ranges.items(), key=lambda x: x[1][0]):
        lower = model.lower()
        if 'titan' in lower:
            color = slide_model_colors['titan']
            name = 'Titan'
        elif 'madeleine' in lower:
            color = slide_model_colors['madeleine']
            name = 'Madeleine'
        elif 'chief' in lower:
            color = slide_model_colors['chief']
            name = 'Chief'
        else:
            color = '#808080'
            name = model
        legend_items.append(mpatches.Patch(facecolor=color, edgecolor='black', linewidth=0.5))
        legend_labels.append(f"{name}: {start}-{end-1}")
    legend_items.append(plt.Line2D([0], [0], color='black', linewidth=2))
    legend_labels.append("Common (all diseases)")

    fig.legend(legend_items, legend_labels, loc='lower center', ncol=5, frameon=True, fontsize=11,
               bbox_to_anchor=(0.5, -0.01))

    # Save
    os.makedirs(output_path and os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none', format='png')
    plt.close()
    print(f"\nSaved visualization: {output_path}")


def main():
    args = parse_args()
    thresholds = [float(t) for t in args.thresholds.split(',')]
    threshold_strs = [f"{t:.1f}".replace('.', 'p') for t in thresholds]

    print("="*80)
    print("Slide-Level Feature Selection Visualization")
    print("="*80)
    print(f"Thresholds: {thresholds}")
    print(f"Diseases: Kidney, Prostate, Rectum")
    print("="*80)

    # Infer unified space across all diseases and thresholds
    print("\n1. Inferring unified feature space...")
    feature_dirs = {
        'Kidney': args.kidney_features,
        'Prostate': args.prostate_features,
        'Rectum': args.rectal_features,
    }
    model_dims, model_order = infer_model_dims_across_all(feature_dirs, threshold_strs)
    unified_mapping, total_features, model_ranges = create_unified_feature_mapping(model_dims, model_order)

    print(f"Total unified feature space: {total_features} features")
    print("Model ranges:")
    for model, (start, end) in model_ranges.items():
        print(f"  {model}: {start}-{end-1} ({end-start} features)")

    # Load selections
    print("\n2. Loading selected features...")
    feature_dirs = {
        'Kidney': args.kidney_features,
        'Prostate': args.prostate_features,
        'Rectum': args.rectal_features,
    }
    all_selections = {disease: {} for disease in feature_dirs}
    for disease, feature_dir in feature_dirs.items():
        print(f"\n  Loading {disease}...")
        for thr, thr_str in zip(thresholds, threshold_strs):
            try:
                sel = load_selected_features(feature_dir, thr_str, unified_mapping, total_features)
                all_selections[disease][thr] = sel
                print(f"    Threshold {thr:.1f}: {int(np.sum(sel))} features selected")
            except Exception as e:
                print(f"    Error loading threshold {thr:.1f}: {e}")
                all_selections[disease][thr] = np.zeros(total_features, dtype=bool)

    # Visualize
    print("\n3. Creating visualization...")
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'feature_selection_visualization_slide.png')
    create_visualization(all_selections, thresholds, model_ranges, total_features, output_path)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for thr in thresholds:
        print(f"\nThreshold {thr:.1f}:")
        for disease in ['Kidney','Prostate','Rectum']:
            n = int(np.sum(all_selections[disease][thr]))
            print(f"  {disease:10s}: {n:4d} features selected")
        common = all_selections['Kidney'][thr] & all_selections['Prostate'][thr] & all_selections['Rectum'][thr]
        print(f"  {'Common':10s}: {int(np.sum(common)):4d} features (shared by all)")


if __name__ == '__main__':
    main()
