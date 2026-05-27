#!/usr/bin/env python3
"""
pruning_visualization.py

Create a combined visualization showing:
1. Horizontal thermometer plots displaying feature pruning percentages for each model
2. Horizontal box plots showing F1 score distributions for pruned models

This script:
1. Reads pruning data from an Excel file with feature counts at different correlation thresholds
2. Loads pruned models and generates F1 score distributions using bootstrap sampling
3. Creates horizontal thermometer plots showing percentage of features retained
4. Creates horizontal box plots showing F1 score distributions with gradient colors
5. Combines both visualizations in a single publication-ready figure

Example:
  python pruning_visualization.py \
    --pruning_excel /path/to/tile_pruning.xlsx \
    --pruned_model_dirs /path/to/kidney_CLAM_combpruned01_sb_s1 /path/to/kidney_CLAM_combpruned02_sb_s1 ... \
    --pruned_feature_dirs /path/to/pruned01_features /path/to/pruned02_features ... \
    --labels_csv /path/to/labels.csv \
    --split_file /path/to/split_file.csv \
    --task task_kidney_grade \
    --output_dir /path/to/output \
    --n_classes 2 \
    --device cuda:0
"""
import argparse
import os
import re
import glob
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix
)
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon
import warnings
warnings.filterwarnings('ignore')

# For Excel input/output
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Import CLAM models
try:
    from models.model_clam import CLAM_SB, CLAM_MB
except ImportError:
    print("Warning: Could not import CLAM models. Make sure you're in the correct directory.")
    CLAM_SB = None
    CLAM_MB = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create pruning visualization with thermometer plots and F1 score box plots"
    )
    parser.add_argument(
        "--pruning_excel", type=str, required=True,
        help="Excel file containing pruning data with feature counts at different thresholds"
    )
    parser.add_argument(
        "--pruned_model_dirs", type=str, nargs='+', required=True,
        help="List of directories containing pruned models (thresholds 0.1-0.7)"
    )
    parser.add_argument(
        "--pruned_feature_dirs", type=str, nargs='+', required=True,
        help="List of directories containing feature files for pruned models (same order as model_dirs)"
    )
    parser.add_argument(
        "--labels_csv", type=str, required=True,
        help="CSV mapping slide_id to integer label"
    )
    parser.add_argument(
        "--split_file", type=str, required=True,
        help="CSV file with splits. Supports two formats: 1) train/val/test columns with slide IDs, or 2) slide_id/split columns (1=train, 2=val, 3=test). Only test slides will be evaluated."
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=[
            'task_kidney_grade',
            'task_prostate_grade', 
            'task_rectal_stage',
            'none'
        ],
        help="Label remapping task specification"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save visualization results"
    )
    parser.add_argument(
        "--model_type", choices=['clam_sb','clam_mb'], default='clam_sb',
        help="CLAM architecture type"
    )
    parser.add_argument(
        "--model_size", choices=['small','big'], default='small',
        help="CLAM model size"
    )
    parser.add_argument(
        "--n_classes", type=int, required=True,
        help="Number of output classes (after remapping)"
    )
    parser.add_argument(
        "--device", default='cuda:0',
        help="PyTorch device identifier"
    )
    parser.add_argument(
        "--bootstrap_iterations", type=int, default=50,
        help="Number of bootstrap iterations"
    )
    parser.add_argument(
        "--bootstrap_fraction", type=float, default=0.8,
        help="Fraction of data to use in each bootstrap sample"
    )
    return parser.parse_args()


def load_pruning_data(excel_file):
    """Load pruning data from Excel file and convert to percentages."""
    df = pd.read_excel(excel_file)
    
    # Extract model names from columns (excluding 'Threshold' and 'Total')
    model_columns = [col for col in df.columns if col not in ['Threshold', 'Total']]
    
    # Get original feature counts (OG Features row)
    og_row = df[df['Threshold'] == 'OG Features']
    if og_row.empty:
        raise ValueError("Excel file must contain 'OG Features' row")
    
    original_counts = {}
    total_original_features = 0
    for model in model_columns:
        original_counts[model] = og_row[model].iloc[0]
        total_original_features += original_counts[model]
    
    # Get threshold data (exclude OG Features row)
    threshold_data = df[df['Threshold'] != 'OG Features'].copy()
    
    # Convert to percentages and feature counts
    pruning_percentages = {}
    pruning_counts = {}
    total_percentages = {}
    thresholds = []
    
    for _, row in threshold_data.iterrows():
        threshold = float(row['Threshold'])
        thresholds.append(threshold)
        
        pruning_percentages[threshold] = {}
        pruning_counts[threshold] = {}
        total_retained = 0
        
        for model in model_columns:
            retained_count = row[model]
            pruning_counts[threshold][model] = retained_count
            total_retained += retained_count
        
        # Calculate percentages of total retained features at this threshold
        for model in model_columns:
            retained_count = row[model]
            percentage = (retained_count / total_retained) * 100 if total_retained > 0 else 0
            pruning_percentages[threshold][model] = percentage
        
        # Calculate percentage of total original features
        total_percentages[threshold] = (total_retained / total_original_features) * 100
    
    return pruning_percentages, pruning_counts, total_percentages, model_columns, thresholds


def extract_threshold_from_model_dir(model_dir):
    """Extract correlation threshold from model directory name."""
    model_name = os.path.basename(model_dir)
    
    # Expected format: {disease}_CLAM_combpruned0{threshold}_sb_s1
    # Extract threshold from combpruned0{threshold}
    match = re.search(r'combpruned0(\d)', model_name)
    if match:
        threshold_int = int(match.group(1))
        threshold_float = round(threshold_int * 0.1, 1)  # Convert 1->0.1, 2->0.2, etc. and round to 1 decimal
        return threshold_float
    
    raise ValueError(f"Could not extract threshold from model directory: {model_dir}")


def select_best_fold_for_model(model_dir, task):
    """Select the best fold for a model based on F1 score on test set."""
    metrics_file = os.path.join(model_dir, 'final_metric_summary.csv')
    if not os.path.exists(metrics_file):
        raise RuntimeError(f"Metrics file not found: {metrics_file}")
    
    df = pd.read_csv(metrics_file)
    test_df = df[df['split'] == 'test'].copy()
    
    if test_df.empty:
        raise RuntimeError(f"No test metrics found in {metrics_file}")
    
    # Select fold with highest F1 score
    best_fold = test_df.loc[test_df['f1_score'].idxmax(), 'fold']
    best_f1 = test_df.loc[test_df['f1_score'].idxmax(), 'f1_score']
    
    print(f"Model {os.path.basename(model_dir)}: Best fold {best_fold} with F1={best_f1:.4f}")
    return best_fold


def load_model_for_fold(model_dir, fold, model_type, model_size, embed_dim, n_classes, device):
    """Load a CLAM model for a specific fold."""
    if CLAM_SB is None or CLAM_MB is None:
        raise RuntimeError("CLAM models not available. Check import.")
    
    ckpt_file = os.path.join(model_dir, f's_{fold}_checkpoint.pt')
    if not os.path.exists(ckpt_file):
        raise RuntimeError(f"Checkpoint not found: {ckpt_file}")
    
    kwargs = dict(
        gate=True,
        size_arg=model_size,
        dropout=0.0,
        k_sample=1,
        n_classes=n_classes,
        subtyping=False,
        embed_dim=embed_dim
    )
    
    model = CLAM_SB(**kwargs) if model_type=='clam_sb' else CLAM_MB(**kwargs)
    ckpt = torch.load(ckpt_file, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def slide_prediction(model, feat_path, device):
    """Generate prediction for a single slide."""
    with h5py.File(feat_path, 'r') as f:
        feats = f['features'][:]
    feats_t = torch.from_numpy(feats).to(device).float()
    lbl = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        logits, prob, pred, _, _ = model(feats_t, lbl, instance_eval=False)
    return prob.cpu().numpy().ravel(), int(pred.item())


def load_and_remap_labels(labels_csv, task):
    """Load and remap labels based on task specification."""
    lbl_df = pd.read_csv(labels_csv, dtype={'slide_id': str, 'label': int})
    
    # Remap labels if needed
    if task == 'task_kidney_grade':
        lbl_df['label'] = lbl_df['label'].map({0: 0, 1: 0, 2: 1, 3: 1})
    elif task == 'task_prostate_grade':
        lbl_df['label'] = lbl_df['label'].map({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1})
    elif task == 'task_rectal_stage':
        lbl_df['label'] = lbl_df['label'].map({1: 0, 2: 0, 3: 1, 4: 1})
    # else 'none' - no remapping
    
    return dict(zip(lbl_df['slide_id'], lbl_df['label']))


def load_split_file(split_file):
    """Load split file and return test slide IDs."""
    split_df = pd.read_csv(split_file)
    
    # Check if this is the format with train/val/test columns
    if 'test' in split_df.columns:
        # Format: train, val, test columns with slide IDs
        test_slides = split_df['test'].dropna().tolist()
        train_slides = split_df['train'].dropna().tolist()
        val_slides = split_df['val'].dropna().tolist()
        
        print(f"Loaded split file (column format): {len(test_slides)} test slides")
        print(f"Split distribution: train={len(train_slides)}, val={len(val_slides)}, test={len(test_slides)}")
        
    elif 'split' in split_df.columns and 'slide_id' in split_df.columns:
        # Format: slide_id, split columns with 1/2/3 values
        test_slides = split_df[split_df['split'] == 3]['slide_id'].tolist()
        train_slides = split_df[split_df['split'] == 1]['slide_id'].tolist()
        val_slides = split_df[split_df['split'] == 2]['slide_id'].tolist()
        
        print(f"Loaded split file (row format): {len(test_slides)} test slides out of {len(split_df)} total slides")
        print(f"Split distribution: train={len(train_slides)}, val={len(val_slides)}, test={len(test_slides)}")
        
    else:
        raise ValueError("Split file must have either 'train','val','test' columns or 'slide_id','split' columns")
    
    return test_slides


def generate_predictions_for_model(model, test_slides, feature_dir, device, model_name=None):
    """Generate predictions for all test slides using a model."""
    predictions = []
    probabilities = []
    
    # Create a more descriptive progress bar
    desc = f"Predictions for {model_name}" if model_name else "Generating predictions"
    
    for slide_id in tqdm(test_slides, desc=desc):
        feat_path = os.path.join(feature_dir, f"{slide_id}.h5")
        prob, pred = slide_prediction(model, feat_path, device)
        predictions.append(pred)
        probabilities.append(prob)
    
    return np.array(predictions), np.array(probabilities)


def compute_metrics(y_true, y_pred, y_prob, n_classes):
    """Compute all metrics for given predictions."""
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1 Score
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # AUC
    if len(set(y_true)) > 1:
        if n_classes == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            except:
                metrics['auc'] = np.nan
        else:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except:
                metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan
    
    # Sensitivity and Specificity (binary only)
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.
    else:
        metrics['sensitivity'] = np.nan
        metrics['specificity'] = np.nan
    
    return metrics


def bootstrap_sample(data, fraction=0.8, random_state=None):
    """Create a bootstrap sample of the data."""
    np.random.seed(random_state)
    n_samples = len(data)
    n_bootstrap = int(n_samples * fraction)
    indices = np.random.choice(n_samples, size=n_bootstrap, replace=True)
    return indices


def perform_bootstrap_analysis(y_true, all_predictions, all_probabilities, n_classes, 
                              n_iterations=50, bootstrap_fraction=0.8):
    """Perform bootstrap analysis to generate performance distributions."""
    model_names = list(all_predictions.keys())
    metrics = ['f1_score']  # Only need F1 score for this visualization
    
    # Initialize results storage
    bootstrap_results = {}
    for model in model_names:
        bootstrap_results[model] = {}
        for metric in metrics:
            bootstrap_results[model][metric] = []
    
    print(f"Performing {n_iterations} bootstrap iterations...")
    
    for iteration in tqdm(range(n_iterations), desc="Bootstrap sampling"):
        # Create bootstrap sample
        bootstrap_indices = bootstrap_sample(y_true, fraction=bootstrap_fraction, 
                                           random_state=iteration)
        
        # Get bootstrap samples for true labels
        y_true_bootstrap = y_true[bootstrap_indices]
        
        # Compute metrics for each model
        for model_name in model_names:
            y_pred_bootstrap = all_predictions[model_name][bootstrap_indices]
            y_prob_bootstrap = all_probabilities[model_name][bootstrap_indices]
            
            metrics_dict = compute_metrics(y_true_bootstrap, y_pred_bootstrap, 
                                         y_prob_bootstrap, n_classes)
            
            for metric in metrics:
                bootstrap_results[model_name][metric].append(metrics_dict[metric])
    
    return bootstrap_results


def create_thermometer_plot(ax, pruning_percentages, pruning_counts, model_columns, thresholds, model_colors):
    """Create horizontal thermometer plots for pruning percentages."""
    
    # Set up the plot
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(thresholds) - 0.5)
    
    # Create thermometer bars for each threshold
    bar_height = 0.6
    y_positions = np.arange(len(thresholds))
    
    for i, threshold in enumerate(thresholds):
        y_pos = y_positions[i]
        
        # Create thermometer background (empty bar)
        ax.barh(y_pos, 100, height=bar_height, color='lightgray', alpha=0.3, edgecolor='black', linewidth=1)
        
        # Create stacked segments for each model
        x_start = 0
        for model in model_columns:
            percentage = pruning_percentages[threshold][model]
            count = pruning_counts[threshold][model]
            color = model_colors.get(model.lower(), '#CCCCCC')
            
            if percentage > 0:
                ax.barh(y_pos, percentage, height=bar_height, left=x_start, 
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add text showing feature count in the center of each segment
                if percentage > 8:  # Only show text if segment is large enough
                    text_x = x_start + percentage / 2
                    ax.text(text_x, y_pos, str(count), ha='center', va='center', 
                           fontsize=18, fontweight='bold', color='black')
                
                x_start += percentage
    
    # Set labels and formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'{t:.1f}' for t in thresholds], fontsize=20)
    ax.set_ylabel('Correlation Threshold', fontsize=24, fontweight='bold')
    ax.set_xlabel('Percentage of Retained Features (%)', fontsize=24, fontweight='bold')
    ax.set_title('Feature Distribution by Model', fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)
    
    # Remove grid
    ax.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_total_percentage_barplot(ax, total_percentages, thresholds):
    """Create horizontal bar plot showing percentage of total original features retained."""
    
    # Set up the plot
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(thresholds) - 0.5)
    
    # Create bars for each threshold
    bar_height = 0.6
    y_positions = np.arange(len(thresholds))
    
    for i, threshold in enumerate(thresholds):
        y_pos = y_positions[i]
        percentage = total_percentages[threshold]
        
        # Create bar showing percentage of total original features
        ax.barh(y_pos, percentage, height=bar_height, color='darkblue', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add text showing percentage
        ax.text(percentage + 2, y_pos, f'{percentage:.1f}%', ha='left', va='center', 
               fontsize=16, fontweight='bold')
    
    # Set labels and formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'{t:.1f}' for t in thresholds], fontsize=20)
    ax.set_xlabel('Total Features Retained (%)', fontsize=24, fontweight='bold')
    ax.set_title('Overall Feature Retention', fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)
    
    # Remove grid
    ax.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_f1_boxplot(ax, bootstrap_results, thresholds, model_colors):
    """Create horizontal box plots for F1 scores with gradient colors."""
    
    # Prepare data for plotting
    plot_data = []
    model_names = []
    threshold_labels = []
    
    for threshold in thresholds:
        threshold_str = f"{threshold:.1f}"
        if threshold_str in bootstrap_results:
            f1_values = np.array(bootstrap_results[threshold_str]['f1_score'])
            f1_clean = f1_values[~np.isnan(f1_values)]
            
            if len(f1_clean) > 0:
                plot_data.extend(f1_clean)
                model_names.extend([threshold_str] * len(f1_clean))
                threshold_labels.append(threshold_str)
    
    if not plot_data:
        return
    
    # Create horizontal box plot
    y_positions = np.arange(len(threshold_labels))
    
    # Calculate gradient colors based on threshold (0.1=black, 0.7=light gray)
    colors = []
    for threshold in thresholds:
        # Normalize threshold to 0-1 range (0.1 to 0.7 -> 0 to 1)
        normalized = (threshold - 0.1) / (0.7 - 0.1)
        # Create gradient from black (0) to very light gray (0.9) for more pronounced gradient
        gray_value = normalized * 0.9  # Max at 0.9 for very light gray
        colors.append((gray_value, gray_value, gray_value))
    
    # Create box plot data
    box_data = []
    for threshold_str in threshold_labels:
        threshold_data = [plot_data[i] for i, name in enumerate(model_names) if name == threshold_str]
        box_data.append(threshold_data)
    
    # Create horizontal box plot
    box_plot = ax.boxplot(box_data,
                         positions=y_positions,
                         labels=threshold_labels,
                         patch_artist=True,
                         showfliers=False,
                         widths=0.6,
                         vert=False)  # Horizontal box plot
    
    # Apply gradient colors
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style whiskers, caps, and medians
    for whisker in box_plot['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    
    for cap in box_plot['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)
    
    for median in box_plot['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # Set labels and formatting
    ax.set_xlim(0.4, 1)  # Start x-axis at 0.4 instead of 0
    ax.set_yticklabels(threshold_labels, fontsize=20)
    ax.set_xlabel('F1 Score', fontsize=24, fontweight='bold')
    ax.set_title('F1 Score Distribution by Correlation Threshold', fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)
    
    # Remove grid
    ax.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    args = parse_args()
    
    # Set font to Arial for publication-quality figures
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'  # Make axis labels bold by default
    plt.rcParams['axes.titleweight'] = 'bold'  # Make titles bold by default
    plt.rcParams['font.weight'] = 'normal'  # Keep regular text normal weight
    # Ensure matplotlib can synthesize bold fonts if needed
    plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts
    plt.rcParams['ps.fonttype'] = 42   # TrueType fonts
    
    print("="*80)
    print("Pruning Visualization with Thermometer Plots and F1 Score Box Plots")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of pruned models: {len(args.pruned_model_dirs)}")
    print("="*80)
    
    # Validate input arguments
    if len(args.pruned_model_dirs) != len(args.pruned_feature_dirs):
        raise RuntimeError("Number of pruned model directories must match number of pruned feature directories")
    
    # Step 1: Load pruning data from Excel
    print("\n1. Loading pruning data from Excel...")
    pruning_percentages, pruning_counts, total_percentages, model_columns, thresholds = load_pruning_data(args.pruning_excel)
    print(f"Found pruning data for thresholds: {thresholds}")
    print(f"Models: {model_columns}")
    
    # Step 2: Extract thresholds from model directories and validate
    print("\n2. Validating model directories and extracting thresholds...")
    model_thresholds = {}
    model_dirs_by_threshold = {}
    feature_dirs_by_threshold = {}
    
    for model_dir, feature_dir in zip(args.pruned_model_dirs, args.pruned_feature_dirs):
        threshold = extract_threshold_from_model_dir(model_dir)
        threshold_str = f"{threshold:.1f}"
        
        if threshold not in thresholds:
            print(f"Warning: Threshold {threshold} from {model_dir} not found in Excel data")
            continue
        
        if not os.path.exists(model_dir):
            raise RuntimeError(f"Model directory not found: {model_dir}")
        if not os.path.exists(feature_dir):
            raise RuntimeError(f"Feature directory not found: {feature_dir}")
        
        model_thresholds[threshold_str] = threshold
        model_dirs_by_threshold[threshold_str] = model_dir
        feature_dirs_by_threshold[threshold_str] = feature_dir
        
        print(f"  Threshold {threshold_str}: {os.path.basename(model_dir)} -> {os.path.basename(feature_dir)}")
    
    # Step 3: Load and process labels and split file
    print("\n3. Loading and processing labels and split file...")
    label_map = load_and_remap_labels(args.labels_csv, args.task)
    test_slides = load_split_file(args.split_file)
    
    # Filter test slides to only include those that have labels
    test_slides = [slide_id for slide_id in test_slides if slide_id in label_map]
    print(f"Found {len(test_slides)} test slides with labels")
    
    # Step 4: Infer embedding dimensions for each model
    print("\n4. Inferring embedding dimensions...")
    embed_dims = {}
    
    for threshold_str, feature_dir in feature_dirs_by_threshold.items():
        sample_files = [f for f in os.listdir(feature_dir) if f.endswith('.h5')]
        if sample_files:
            with h5py.File(os.path.join(feature_dir, sample_files[0]), 'r') as f:
                embed_dims[threshold_str] = f['features'].shape[1]
            print(f"Threshold {threshold_str} embedding dimension: {embed_dims[threshold_str]}")
        else:
            raise RuntimeError(f"No .h5 files found in {threshold_str} feature directory: {feature_dir}")
    
    # Step 5: Select best folds for each model
    print("\n5. Selecting best folds for each model...")
    best_folds = {}
    
    for threshold_str, model_dir in model_dirs_by_threshold.items():
        best_folds[threshold_str] = select_best_fold_for_model(model_dir, args.task)
    
    # Step 6: Get true labels for test slides
    print("\n6. Preparing test data...")
    y_true = np.array([label_map[slide_id] for slide_id in test_slides])
    
    # Step 7: Generate predictions for all pruned models
    print("\n7. Generating predictions for all pruned models...")
    all_predictions = {}
    all_probabilities = {}
    
    for threshold_str, model_dir in model_dirs_by_threshold.items():
        print(f"\nLoading model for threshold {threshold_str}...")
        print(f"  Model directory: {os.path.basename(model_dir)}")
        print(f"  Best fold: {best_folds[threshold_str]} (embedding dim: {embed_dims[threshold_str]})")
        
        model = load_model_for_fold(model_dir, best_folds[threshold_str], 
                                   args.model_type, args.model_size,
                                   embed_dims[threshold_str], args.n_classes, args.device)
        
        # Use the corresponding feature directory for this model
        feature_dir = feature_dirs_by_threshold[threshold_str]
        print(f"  Using features from: {os.path.basename(feature_dir)}")
        
        predictions, probabilities = generate_predictions_for_model(
            model, test_slides, feature_dir, args.device, f"threshold_{threshold_str}"
        )
        all_predictions[threshold_str] = predictions
        all_probabilities[threshold_str] = probabilities
    
    # Step 8: Perform bootstrap analysis
    print("\n8. Performing bootstrap analysis...")
    bootstrap_results = perform_bootstrap_analysis(
        y_true, all_predictions, all_probabilities, args.n_classes,
        args.bootstrap_iterations, args.bootstrap_fraction
    )
    
    # Step 9: Create visualization
    print("\n9. Creating combined visualization...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define model colors (same as compare_model_performance.py)
    model_colors = {
        'conch': '#0070C0',      # Blue
        'musk': '#FFC000',       # Yellow
        'hoptimus': '#C00000',   # Red
        'h-optimus': '#C00000',  # Red (with hyphen)
        'virchow': '#7030A0',    # Purple
        'gigapath': '#00B050',   # Green
    }
    
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), 
                                       gridspec_kw={'width_ratios': [2, 1, 2]})
    
    # Create thermometer plot (left panel)
    create_thermometer_plot(ax1, pruning_percentages, pruning_counts, model_columns, thresholds, model_colors)
    
    # Create total percentage bar plot (middle panel)
    create_total_percentage_barplot(ax2, total_percentages, thresholds)
    
    # Create F1 score box plot (right panel)
    create_f1_boxplot(ax3, bootstrap_results, thresholds, model_colors)
    
    # Add overall title
    fig.suptitle('Feature Pruning Analysis: Retention vs Performance', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the combined plot
    output_path = os.path.join(args.output_dir, 'pruning_visualization.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved combined visualization: {output_path}")
    
    # Step 10: Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Processed {len(thresholds)} correlation thresholds: {thresholds}")
    print(f"Analyzed {len(model_columns)} models: {model_columns}")
    print(f"Used {len(test_slides)} test slides for evaluation")
    print(f"Performed {args.bootstrap_iterations} bootstrap iterations")
    print("="*80)


if __name__ == '__main__':
    main()
