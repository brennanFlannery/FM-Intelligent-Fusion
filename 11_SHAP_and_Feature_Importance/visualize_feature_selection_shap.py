#!/usr/bin/env python3
"""
visualize_feature_selection_shap.py

Create a publication-quality figure showing feature selection across diseases and thresholds.

Figure structure:
- 7 columns (correlation thresholds: 0.1 - 0.7)
- 1 row (feature selection)
- Each cell: 3 horizontal bars (Kidney, Prostate, Rectum) stacked vertically

Shows which features were selected (red marks on unified feature space across all 5 models).
Models are displayed in order: CONCH, GigaPath, Hoptimus1, Virchow2, MUSK.

Example:
  python visualize_feature_selection_shap.py \
    --kidney_features /path/to/NewFusionTracking/Kidney \
    --prostate_features /path/to/NewFusionTracking/Prostate \
    --rectal_features /path/to/NewFusionTracking/Rectum \
    --output_dir /path/to/output
"""

import argparse
import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize feature selection and SHAP values across diseases"
    )
    parser.add_argument(
        "--kidney_features", type=str, required=True,
        help="Parent directory for kidney combined features with feature_sources.npy (contains thr_0p1, thr_0p2, etc.)"
    )
    parser.add_argument(
        "--prostate_features", type=str, required=True,
        help="Parent directory for prostate combined features with feature_sources.npy"
    )
    parser.add_argument(
        "--rectal_features", type=str, required=True,
        help="Parent directory for rectal combined features with feature_sources.npy"
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


def load_original_dimensions(feature_dir, threshold_str='0p1'):
    """
    Load original model dimensions. Always returns all 5 models in order.
    Uses feature_sources.npy to verify, but falls back to standard dimensions.
    """
    # Standard dimensions for these models at 512px - ALWAYS use these to ensure all models appear
    standard_dims = {
        'features_conch_v15': 768,
        'features_gigapath': 1536,
        'features_hoptimus1': 1536,
        'features_virchow2': 2560,
        'features_musk': 1024
    }
    
    # Fixed model order
    model_order = ['features_conch_v15', 'features_gigapath', 'features_hoptimus1', 
                   'features_virchow2', 'features_musk']
    
    # Optionally verify from feature_sources.npy
    thr_dir = os.path.join(feature_dir, f'thr_{threshold_str}')
    feat_src_path = os.path.join(thr_dir, 'feature_sources.npy')
    
    if os.path.exists(feat_src_path):
        feat_sources = np.load(feat_src_path, allow_pickle=True)
        if feat_sources.dtype.names:
            models = feat_sources['model']
            indices = feat_sources['feature_index']
            
            # Verify dimensions for models that appear in feature_sources
            for model in np.unique(models):
                if model in standard_dims:
                    mask = models == model
                    max_idx = np.max(indices[mask])
                    observed_dim = max_idx + 1
                    if observed_dim > standard_dims[model]:
                        print(f"Warning: {model} has observed max index {max_idx}, "
                              f"but standard dim is {standard_dims[model]}. Using observed.")
                        standard_dims[model] = observed_dim
    
    return standard_dims, model_order


def create_unified_feature_mapping(model_dims, model_order):
    """
    Create a mapping from (model, feature_index) to unified index.
    
    Returns:
        mapping: dict[(model, idx)] -> unified_idx
        total_features: int, total number of features in unified space
        model_ranges: dict[model] -> (start_idx, end_idx)
    """
    mapping = {}
    model_ranges = {}
    current_idx = 0
    
    for model in model_order:
        dim = model_dims[model]
        start_idx = current_idx
        end_idx = current_idx + dim
        model_ranges[model] = (start_idx, end_idx)
        
        for feat_idx in range(dim):
            mapping[(model, feat_idx)] = current_idx
            current_idx += 1
    
    return mapping, current_idx, model_ranges


def load_selected_features(feature_dir, threshold_str, unified_mapping, total_features):
    """
    Load selected features for a given threshold and map to unified indices.
    
    Returns:
        selected_unified: np.ndarray of shape (total_features,), boolean array
    """
    thr_dir = os.path.join(feature_dir, f'thr_{threshold_str}')
    feat_src_path = os.path.join(thr_dir, 'feature_sources.npy')
    
    if not os.path.exists(feat_src_path):
        raise FileNotFoundError(f"Feature sources not found: {feat_src_path}")
    
    feat_sources = np.load(feat_src_path, allow_pickle=True)
    
    # Initialize selection array
    selected_unified = np.zeros(total_features, dtype=bool)
    
    # Map each selected feature to unified index
    if feat_sources.dtype.names:
        models = feat_sources['model']
        indices = feat_sources['feature_index']
        
        for model, idx in zip(models, indices):
            key = (model, int(idx))
            if key in unified_mapping:
                unified_idx = unified_mapping[key]
                selected_unified[unified_idx] = True
    
    return selected_unified


def load_and_aggregate_shap(feature_dir, shap_base_dir, threshold_str, unified_mapping, total_features, debug=True):
    """
    Load SHAP values, average across slides, and map to unified indices.
    
    Parameters:
        feature_dir: Directory containing feature_sources.npy
        shap_base_dir: Directory containing SHAP values
        threshold_str: Threshold string (e.g., '0p1')
        unified_mapping: Mapping from (model, idx) to unified index
        total_features: Total number of features in unified space
        debug: Print debug information
    
    Returns:
        shap_unified: np.ndarray of shape (total_features,), mean absolute SHAP values
    """
    thr_dir_features = os.path.join(feature_dir, f'thr_{threshold_str}')
    thr_dir_shap = os.path.join(shap_base_dir, f'thr_{threshold_str}')
    shap_dir = os.path.join(thr_dir_shap, 'SHAP')
    feat_src_path = os.path.join(thr_dir_features, 'feature_sources.npy')
    
    if debug:
        print(f"\n    DEBUG: SHAP Loading for threshold {threshold_str}")
        print(f"      Feature dir:     {thr_dir_features}")
        print(f"      Feature exists:  {os.path.exists(thr_dir_features)}")
        print(f"      SHAP base dir:   {shap_base_dir}")
        print(f"      SHAP thr dir:    {thr_dir_shap}")
        print(f"      SHAP thr exists: {os.path.exists(thr_dir_shap)}")
        
        # List contents of threshold directory if it exists
        if os.path.exists(thr_dir_shap):
            contents = os.listdir(thr_dir_shap)
            print(f"      Contents of {thr_dir_shap}:")
            for item in contents[:10]:  # Limit to first 10 items
                item_path = os.path.join(thr_dir_shap, item)
                if os.path.isdir(item_path):
                    print(f"        [DIR]  {item}")
                else:
                    print(f"        [FILE] {item}")
            if len(contents) > 10:
                print(f"        ... and {len(contents) - 10} more items")
        
        print(f"      SHAP dir:        {shap_dir}")
        print(f"      SHAP dir exists: {os.path.exists(shap_dir)}")
    
    if not os.path.exists(shap_dir):
        print(f"    Warning: SHAP directory not found: {shap_dir}")
        return np.zeros(total_features)
    
    if not os.path.exists(feat_src_path):
        raise FileNotFoundError(f"Feature sources not found: {feat_src_path}")
    
    # Load feature sources to get mapping
    feat_sources = np.load(feat_src_path, allow_pickle=True)
    
    # Get all SHAP files (try both .npy and .h5 formats)
    all_files = os.listdir(shap_dir)
    shap_files = [f for f in all_files if f.endswith('.npy') or f.endswith('.h5')]
    file_format = 'npy' if any(f.endswith('.npy') for f in shap_files) else 'h5'
    
    if debug:
        print(f"      Total files in SHAP dir: {len(all_files)}")
        print(f"      SHAP files found: {len(shap_files)} ({file_format} format)")
        if shap_files:
            print(f"      First few SHAP files: {shap_files[:3]}")
    
    if not shap_files:
        print(f"    Warning: No SHAP files found in {shap_dir}")
        if debug and all_files:
            print(f"    All files in directory: {all_files[:10]}")
        return np.zeros(total_features)
    
    # Accumulate SHAP values per unified index
    shap_sums = np.zeros(total_features)
    shap_counts = np.zeros(total_features)
    
    # Load feature sources mapping
    if feat_sources.dtype.names:
        models = feat_sources['model']
        indices = feat_sources['feature_index']
        
        # Create mapping from selected feature position to unified index
        pos_to_unified = []
        for model, idx in zip(models, indices):
            key = (model, int(idx))
            if key in unified_mapping:
                pos_to_unified.append(unified_mapping[key])
            else:
                pos_to_unified.append(-1)  # Invalid
        
        pos_to_unified = np.array(pos_to_unified)
    else:
        print("Warning: Could not parse feature sources")
        return np.zeros(total_features)
    
    # Load SHAP values from each slide
    for idx, shap_file in enumerate(tqdm(shap_files, desc=f"Loading SHAP", leave=False)):
        shap_path = os.path.join(shap_dir, shap_file)
        try:
            # Load based on file format
            if shap_file.endswith('.npy'):
                shap_vals = np.load(shap_path)
                
                # Debug first file only
                if debug and idx == 0:
                    print(f"      DEBUG: First SHAP file: {shap_file}")
                    print(f"      Array shape: {shap_vals.shape}")
                    print(f"      Array dtype: {shap_vals.dtype}")
                    print(f"      Expected features from feature_sources: {len(pos_to_unified)}")
                
            else:  # .h5 format
                with h5py.File(shap_path, 'r') as f:
                    # Debug first file only
                    if debug and idx == 0:
                        print(f"      DEBUG: First SHAP file: {shap_file}")
                        print(f"      Available keys: {list(f.keys())}")
                        for key in f.keys():
                            print(f"        '{key}': shape={f[key].shape}, dtype={f[key].dtype}")
                    
                    if 'shap_values' in f:
                        shap_vals = f['shap_values'][:]
                    elif 'features' in f:
                        shap_vals = f['features'][:]
                    else:
                        if debug and idx == 0:
                            print(f"      Warning: No 'shap_values' or 'features' dataset found in {shap_file}")
                        continue
            
            # Take absolute values and average if multiple samples
            if shap_vals.ndim > 1:
                shap_vals = np.abs(shap_vals).mean(axis=0)
            else:
                shap_vals = np.abs(shap_vals)
            
            # Map to unified indices
            for pos, unified_idx in enumerate(pos_to_unified):
                if unified_idx >= 0 and pos < len(shap_vals):
                    shap_sums[unified_idx] += shap_vals[pos]
                    shap_counts[unified_idx] += 1
        except Exception as e:
            if debug and idx == 0:
                print(f"      Error loading {shap_file}: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    # Average SHAP values
    shap_unified = np.zeros(total_features)
    valid_mask = shap_counts > 0
    shap_unified[valid_mask] = shap_sums[valid_mask] / shap_counts[valid_mask]
    
    if debug:
        n_with_shap = np.sum(valid_mask)
        mean_shap = np.mean(shap_unified[valid_mask]) if n_with_shap > 0 else 0
        print(f"      Successfully loaded SHAP for {n_with_shap}/{total_features} features")
        print(f"      Mean SHAP value: {mean_shap:.6f}")
    
    return shap_unified


def create_visualization(all_selections, thresholds, model_ranges, 
                        total_features, output_path):
    """
    Create the final visualization figure showing feature selection only.
    
    Parameters:
        all_selections: dict[disease][threshold] -> boolean array
        thresholds: list of float
        model_ranges: dict[model] -> (start, end)
        total_features: int
        output_path: str
    """
    # Set publication-quality style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # Define model colors (same as pruning_visualization.py)
    model_colors = {
        'conch': '#0070C0',      # Blue
        'musk': '#FFC000',       # Yellow/Gold
        'hoptimus': '#C00000',   # Red
        'hoptimus1': '#C00000',  # Red (alternative name)
        'h-optimus': '#C00000',  # Red (with hyphen)
        'virchow': '#7030A0',    # Purple
        'virchow2': '#7030A0',   # Purple (alternative name)
        'gigapath': '#00B050',   # Green
    }
    
    # Create mapping from feature index to model color
    feature_to_color = {}
    for model, (start, end) in model_ranges.items():
        # Extract base model name for color lookup
        model_lower = model.lower()
        
        # Determine color based on model name
        if 'conch' in model_lower:
            color = model_colors['conch']
        elif 'gigapath' in model_lower:
            color = model_colors['gigapath']
        elif 'hoptimus' in model_lower:
            color = model_colors['hoptimus1']
        elif 'virchow' in model_lower:
            color = model_colors['virchow2']
        elif 'musk' in model_lower:
            color = model_colors['musk']
        else:
            color = '#808080'  # Gray for unknown models
        
        for i in range(start, end):
            feature_to_color[i] = color
    
    diseases = ['Kidney', 'Prostate', 'Rectum']
    n_thresholds = len(thresholds)
    
    # Create figure - 2 rows for feature selection (4 bars per cell)
    # Row 1: thresholds 0.1-0.4 (4 columns)
    # Row 2: thresholds 0.5-0.7 (3 columns)
    fig = plt.figure(figsize=(24, 10))
    
    # Create grid: 2 rows × 4 columns
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.1,
                          left=0.05, right=0.98, top=0.95, bottom=0.08)
    
    # Create each subplot
    for idx, threshold in enumerate(thresholds):
        threshold_str = f"{threshold:.1f}".replace('.', 'p')
        
        # Determine row and column position
        if idx < 4:  # First 4 thresholds in row 0
            row_idx, col_idx = 0, idx
        else:  # Last 3 thresholds in row 1
            row_idx, col_idx = 1, idx - 4
        
        # Feature Selection subplot
        ax = fig.add_subplot(gs[row_idx, col_idx])
        
        # Create four stacked bars: 3 for each disease + 1 for common features
        bar_height = 0.22
        y_positions = [0.78, 0.55, 0.32, 0.09]  # Top to bottom: Kidney, Prostate, Rectum, Common
        disease_labels = ['Kidney', 'Prostate', 'Rectum', 'Common']
        
        # Get selections for all diseases
        kidney_sel = all_selections['Kidney'][threshold]
        prostate_sel = all_selections['Prostate'][threshold]
        rectum_sel = all_selections['Rectum'][threshold]
        
        # Calculate common features (intersection of all three)
        common_sel = kidney_sel & prostate_sel & rectum_sel
        
        # Draw bars for each disease
        for disease_idx, disease in enumerate(diseases):
            y_pos = y_positions[disease_idx]
            selected = all_selections[disease][threshold]
            
            # Draw background (all features in light gray)
            ax.add_patch(mpatches.Rectangle(
                (0, y_pos - bar_height/2), total_features, bar_height,
                facecolor='lightgray', edgecolor='black', linewidth=0.5, zorder=1
            ))
            
            # Draw selected features in their model colors
            n_drawn = 0
            drawn_info = []
            for i in range(total_features):
                if selected[i]:
                    feat_color = feature_to_color.get(i, '#FF0000')  # Default to red if not found
                    
                    # Draw wider rectangles for better visibility (width=5)
                    patch = mpatches.Rectangle(
                        (i-2, y_pos - bar_height/2), 5, bar_height,
                        facecolor=feat_color, edgecolor='none'
                    )
                    ax.add_patch(patch)
                    n_drawn += 1
                    if idx == 0 and n_drawn <= 5:  # Track first 5 for debug
                        drawn_info.append(f"idx={i}, color={feat_color}")
            
            if idx == 0:  # Debug first threshold only
                print(f"  {disease}: Drew {n_drawn} patches for {np.sum(selected)} selected features")
                if drawn_info:
                    print(f"    First features: {', '.join(drawn_info)}")
        
        # Draw common features bar (4th bar with black indicators)
        y_pos = y_positions[3]
        ax.add_patch(mpatches.Rectangle(
            (0, y_pos - bar_height/2), total_features, bar_height,
            facecolor='lightgray', edgecolor='black', linewidth=0.5, zorder=1
        ))
        
        # Draw common features as black vertical lines for clarity
        for i in range(total_features):
            if common_sel[i]:
                ax.add_line(plt.Line2D([i, i], [y_pos - bar_height/2, y_pos + bar_height/2],
                                       color='black', linewidth=1.0))
        
        # Format subplot - expand xlim slightly to avoid clipping
        ax.set_xlim(-5, total_features + 5)
        ax.set_ylim(0, 1)
        ax.set_clip_on(False)  # Don't clip patches at axes boundaries
        
        # Only show disease labels on leftmost plots of each row
        if col_idx == 0:  # First column of each row
            ax.set_yticks(y_positions)
            ax.set_yticklabels(disease_labels, fontsize=14, fontweight='bold')
        else:
            ax.set_yticks([])
        
        ax.set_xticks([])
        ax.set_title(f'Threshold = {threshold:.1f}', fontsize=16, fontweight='bold', pad=10)
        
        # Remove gridlines
        ax.grid(False)
        
        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Add legend showing model colors and ranges at the bottom
    model_legend_items = []
    for model, (start, end) in sorted(model_ranges.items(), key=lambda x: x[1][0]):
        # Clean up model names for display
        model_display = model.replace('features_', '').replace('_v15', '').replace('_', ' ').title()
        model_base = model.replace('features_', '').replace('_v15', '').replace('_', '').lower()
        color = model_colors.get(model_base, '#808080')
        
        # Create colored square for each model
        model_legend_items.append((model_display, color, start, end-1))
    
    # Create custom legend with colored patches
    legend_elements = []
    legend_labels = []
    for model_name, color, start, end in model_legend_items:
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', linewidth=0.5))
        legend_labels.append(f"{model_name}: {start}-{end}")
    
    # Add legend for common features as black line
    legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=2))
    legend_labels.append("Common (all diseases)")
    
    # Place legend at bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', ncol=6, 
              frameon=True, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    
    # Save figure with specific backend settings for better rendering
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close()
    
    print(f"\nSaved visualization: {output_path}")


def main():
    args = parse_args()
    
    # Parse thresholds
    thresholds = [float(t) for t in args.thresholds.split(',')]
    threshold_strs = [f"{t:.1f}".replace('.', 'p') for t in thresholds]
    
    print("="*80)
    print("Feature Selection and SHAP Visualization")
    print("="*80)
    print(f"Thresholds: {thresholds}")
    print(f"Diseases: Kidney, Prostate, Rectum")
    print("="*80)
    
    # Step 1: Infer unified feature space
    print("\n1. Inferring unified feature space...")
    model_dims, model_order = load_original_dimensions(args.kidney_features)
    unified_mapping, total_features, model_ranges = create_unified_feature_mapping(
        model_dims, model_order
    )
    
    print(f"Total unified feature space: {total_features} features")
    print("Model ranges:")
    for model, (start, end) in model_ranges.items():
        print(f"  {model}: {start}-{end-1} ({end-start} features)")
    
    # Step 2: Load selected features for all diseases and thresholds
    print("\n2. Loading selected features...")
    feature_dirs = {
        'Kidney': args.kidney_features,
        'Prostate': args.prostate_features,
        'Rectum': args.rectal_features
    }
    
    all_selections = {}
    for disease, feature_dir in feature_dirs.items():
        all_selections[disease] = {}
        print(f"\n  Loading {disease}...")
        for threshold, thr_str in zip(thresholds, threshold_strs):
            try:
                selected = load_selected_features(feature_dir, thr_str, 
                                                  unified_mapping, total_features)
                all_selections[disease][threshold] = selected
                n_selected = np.sum(selected)
                print(f"    Threshold {threshold:.1f}: {n_selected} features selected")
            except Exception as e:
                print(f"    Error loading threshold {threshold:.1f}: {e}")
                all_selections[disease][threshold] = np.zeros(total_features, dtype=bool)
    
    # Step 3: Create visualization
    print("\n3. Creating visualization...")
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'feature_selection_visualization.png')
    
    create_visualization(all_selections, thresholds, model_ranges,
                        total_features, output_path)
    
    # Step 4: Generate summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for threshold in thresholds:
        print(f"\nThreshold {threshold:.1f}:")
        for disease in ['Kidney', 'Prostate', 'Rectum']:
            n_sel = np.sum(all_selections[disease][threshold])
            print(f"  {disease:10s}: {n_sel:4d} features selected")
        
        # Calculate and show common features
        kidney_sel = all_selections['Kidney'][threshold]
        prostate_sel = all_selections['Prostate'][threshold]
        rectum_sel = all_selections['Rectum'][threshold]
        common_sel = kidney_sel & prostate_sel & rectum_sel
        n_common = np.sum(common_sel)
        print(f"  {'Common':10s}: {n_common:4d} features (shared by all)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

