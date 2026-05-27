#!/usr/bin/env python3
"""
compare_slide_model_performance.py

Compare performance of multiple slide-level MLP models using bootstrap sampling and statistical testing.
Specifically compares a pruned model against individual models and a majority vote ensemble.

This script:
1. Validates provided model and feature directories and extracts model names from directory paths
2. Loads test slides from provided split file (supports train/val/test columns or slide_id/split format)
3. Selects best individual models based on AUC (or F1 if AUC unavailable) scores from summary.csv
4. Generates predictions for all models and creates a majority vote ensemble
5. Performs bootstrap sampling to create performance distributions
6. Conducts statistical tests (paired t-test, Wilcoxon signed-rank, and Mann-Whitney U) between distributions
7. Outputs comprehensive results to Excel file with multiple sheets using actual model names
8. Creates publication-ready box plots with custom styling

Example:
  python compare_slide_model_performance.py \
    --individual_model_dirs /path/to/titan_model /path/to/madeleine_model /path/to/chief_model \
    --individual_feature_dirs /path/to/titan_features /path/to/madeleine_features /path/to/chief_features \
    --pruned_model_dir /path/to/pruned_model \
    --pruned_feature_dir /path/to/pruned_features \
    --combined_model_dir /path/to/combined_model \
    --combined_feature_dir /path/to/combined_features \
    --labels_csv /path/to/labels.csv \
    --split_file /path/to/split_file.csv \
    --task task_kidney_grade \
    --output_dir /path/to/output \
    --n_classes 2 \
    --device cuda:0
"""
import argparse
import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
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

# For Excel output
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Model definition (must match train_slide_models.py)
# -----------------------------
class SlideMLP(nn.Module):
    def __init__(self, input_dim: int, depth: int, hidden_size: int = 512, dropout_rate: float = 0.5):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev = hidden_size
        layers.append(nn.Linear(prev, 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns shape [B], a single logit per example
        return self.net(x).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare slide-level MLP model performance with bootstrap sampling and statistical testing"
    )
    parser.add_argument(
        "--individual_model_dirs", type=str, nargs='+', required=True,
        help="List of directories containing individual model folders (titan, madeleine, chief)"
    )
    parser.add_argument(
        "--individual_feature_dirs", type=str, nargs='+', required=True,
        help="List of directories containing feature files corresponding to individual models (same order as model_dirs)"
    )
    parser.add_argument(
        "--pruned_model_dir", type=str, required=True,
        help="Directory containing the pruned model"
    )
    parser.add_argument(
        "--pruned_feature_dir", type=str, required=True,
        help="Directory containing feature files for the pruned model"
    )
    parser.add_argument(
        "--combined_model_dir", type=str, required=True,
        help="Directory containing the combined/naive model"
    )
    parser.add_argument(
        "--combined_feature_dir", type=str, required=True,
        help="Directory containing features for the combined/naive model"
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
        help="Directory to save comparison results"
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
        help="Fraction of data to use in each bootstrap sample (recommended: 0.8 for stable estimates with good variance)"
    )
    return parser.parse_args()


def select_best_fold_for_model(model_dir):
    """Select the best fold for a model based on test F1 score, with AUC as tiebreaker."""
    metrics_file = os.path.join(model_dir, 'final_metrics.csv')
    if not os.path.exists(metrics_file):
        raise RuntimeError(f"Metrics file not found: {metrics_file}")
    
    df = pd.read_csv(metrics_file)
    
    if 'split' not in df.columns or 'fold' not in df.columns:
        raise RuntimeError(f"final_metrics.csv must contain 'split' and 'fold' columns in {metrics_file}")
    
    test_df = df[df['split'] == 'test'].copy()
    
    if test_df.empty:
        raise RuntimeError(f"No test metrics found in {metrics_file}")
    
    # Select fold with highest test F1 score
    if 'f1' not in test_df.columns:
        raise RuntimeError(f"'f1' column not found in {metrics_file}")
    
    if not test_df['f1'].notna().any():
        raise RuntimeError(f"All f1 values are NaN in {metrics_file}")
    
    # Find maximum F1 score
    max_f1 = test_df['f1'].max()
    
    # Get all folds with the maximum F1 score (handles ties)
    top_folds = test_df[test_df['f1'] == max_f1].copy()
    
    # If there's a tie, use AUC as tiebreaker
    if len(top_folds) > 1:
        if 'auc' in top_folds.columns and top_folds['auc'].notna().any():
            # Select fold with highest AUC among tied folds
            best_idx = top_folds['auc'].idxmax()
            best_fold = top_folds.loc[best_idx, 'fold']
            best_f1 = top_folds.loc[best_idx, 'f1']
            best_auc = top_folds.loc[best_idx, 'auc']
            print(f"Model {os.path.basename(model_dir)}: Best fold {best_fold} with test F1={best_f1:.4f}, AUC={best_auc:.4f} (tiebreaker)")
        else:
            # No AUC available, just pick the first one
            best_fold = top_folds.iloc[0]['fold']
            best_f1 = top_folds.iloc[0]['f1']
            print(f"Model {os.path.basename(model_dir)}: Best fold {best_fold} with test F1={best_f1:.4f} (tied, no AUC for tiebreaker)")
    else:
        # No tie, single best fold
        best_fold = top_folds.iloc[0]['fold']
        best_f1 = top_folds.iloc[0]['f1']
        print(f"Model {os.path.basename(model_dir)}: Best fold {best_fold} with test F1={best_f1:.4f}")
    
    return int(best_fold)


def load_feature_vector(h5_path: str) -> np.ndarray:
    """Load a single feature vector from an H5 file."""
    with h5py.File(h5_path, 'r') as f:
        if 'features' not in f:
            raise KeyError(f"Dataset 'features' not found in {h5_path}")
        x = f['features'][:]
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError(f"Unexpected feature shape after squeeze: {x.shape} in {h5_path}")
    return x.astype(np.float32, copy=False)


def load_model_for_fold(model_dir, fold, input_dim, device):
    """Load a slide-level MLP model for a specific fold."""
    fold_dir = os.path.join(model_dir, f'fold_{fold}')
    ckpt_path = os.path.join(fold_dir, 'model.pt')
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    
    # Load to CPU first to avoid GPU memory issues during unpickling
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # training script saved {'state_dict': best_state, 'mlp_args': vars(args)}
    state_dict = ckpt.get('state_dict', ckpt)
    mlp_args = ckpt.get('mlp_args', {})
    
    depth = int(mlp_args.get('mlp_depth', 2))
    hidden = int(mlp_args.get('hidden_size', 512))
    dropout = float(mlp_args.get('dropout', 0.5))
    
    model = SlideMLP(input_dim=input_dim, depth=depth, hidden_size=hidden, dropout_rate=dropout)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def slide_prediction(model, feat_path, device):
    """Generate prediction for a single slide."""
    x = load_feature_vector(feat_path)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, D)
    
    with torch.no_grad():
        logit = model(x_tensor)  # (1,)
        prob = torch.sigmoid(logit)  # (1,)
        pred = (prob >= 0.5).long()  # (1,)
    
    # Return probabilities as [prob_class_0, prob_class_1]
    prob_class_1 = prob.cpu().numpy().item()
    prob_class_0 = 1.0 - prob_class_1
    
    return np.array([prob_class_0, prob_class_1]), int(pred.item())


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
        test_slides = split_df['test'].dropna().astype(str).tolist()
        train_slides = split_df['train'].dropna().astype(str).tolist()
        val_slides = split_df['val'].dropna().astype(str).tolist()
        
        print(f"Loaded split file (column format): {len(test_slides)} test slides")
        print(f"Split distribution: train={len(train_slides)}, val={len(val_slides)}, test={len(test_slides)}")
        
    elif 'split' in split_df.columns and 'slide_id' in split_df.columns:
        # Format: slide_id, split columns with 1/2/3 values
        test_slides = split_df[split_df['split'] == 3]['slide_id'].astype(str).tolist()
        train_slides = split_df[split_df['split'] == 1]['slide_id'].astype(str).tolist()
        val_slides = split_df[split_df['split'] == 2]['slide_id'].astype(str).tolist()
        
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
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file not found: {feat_path}")
        prob, pred = slide_prediction(model, feat_path, device)
        predictions.append(pred)
        probabilities.append(prob)
    
    return np.array(predictions), np.array(probabilities)


def create_majority_vote_ensemble(individual_predictions):
    """Create majority vote ensemble from individual model predictions."""
    # individual_predictions is a dict with model_name -> predictions array
    model_names = list(individual_predictions.keys())
    n_samples = len(individual_predictions[model_names[0]])
    
    ensemble_predictions = []
    for i in range(n_samples):
        votes = [individual_predictions[model][i] for model in model_names]
        majority_vote = max(set(votes), key=votes.count)
        ensemble_predictions.append(majority_vote)
    
    return np.array(ensemble_predictions)


def compute_metrics(y_true, y_pred, y_prob, n_classes):
    """Compute all metrics for given predictions."""
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1 Score
    metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
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
    # Get model names excluding ensemble (ensemble is processed separately)
    model_names = [name for name in all_predictions.keys() if name != 'ensemble']
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
    
    # Initialize results storage
    bootstrap_results = {}
    for model in model_names + ['ensemble']:
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
        
        # Compute ensemble metrics
        ensemble_preds_bootstrap = create_majority_vote_ensemble({
            model: all_predictions[model][bootstrap_indices] for model in model_names
        })
        # For ensemble, we need to compute probabilities differently
        # For simplicity, we'll use the mean of individual model probabilities
        ensemble_probs_bootstrap = np.mean([all_probabilities[model][bootstrap_indices] 
                                          for model in model_names], axis=0)
        
        
        ensemble_metrics = compute_metrics(y_true_bootstrap, ensemble_preds_bootstrap, 
                                         ensemble_probs_bootstrap, n_classes)
        
        for metric in metrics:
            bootstrap_results['ensemble'][metric].append(ensemble_metrics[metric])
    
    return bootstrap_results


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    # Remove NaN values
    g1_clean = group1[~np.isnan(group1)]
    g2_clean = group2[~np.isnan(group2)]
    
    if len(g1_clean) < 2 or len(g2_clean) < 2:
        return np.nan
    
    # Calculate pooled standard deviation
    n1, n2 = len(g1_clean), len(g2_clean)
    s1, s2 = np.std(g1_clean, ddof=1), np.std(g2_clean, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    if pooled_std == 0:
        return np.nan
    
    d = (np.mean(g1_clean) - np.mean(g2_clean)) / pooled_std
    return d


def calculate_confidence_intervals(data, confidence=0.95):
    """Calculate confidence intervals for bootstrap distributions."""
    clean_data = data[~np.isnan(data)]
    if len(clean_data) < 2:
        return np.nan, np.nan
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(clean_data, lower_percentile)
    upper_ci = np.percentile(clean_data, upper_percentile)
    
    return lower_ci, upper_ci


def perform_statistical_tests(bootstrap_results, reference_model, significance_level=0.05):
    """Perform t-tests and Mann-Whitney U tests between reference model and all others."""
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
    models = [model for model in bootstrap_results.keys() if model != reference_model]
    
    results = {}
    confidence_intervals = {}
    effect_sizes = {}
    
    for metric in metrics:
        results[metric] = {}
        confidence_intervals[metric] = {}
        effect_sizes[metric] = {}
        
        # Get reference model distribution
        reference_dist = np.array(bootstrap_results[reference_model][metric])
        
        # Calculate confidence intervals for reference model
        ref_lower_ci, ref_upper_ci = calculate_confidence_intervals(reference_dist)
        confidence_intervals[metric][reference_model] = {
            'lower_ci': ref_lower_ci,
            'upper_ci': ref_upper_ci
        }
        
        for model in models:
            model_dist = np.array(bootstrap_results[model][metric])
            
            # Remove NaN values for testing
            ref_clean = reference_dist[~np.isnan(reference_dist)]
            model_clean = model_dist[~np.isnan(model_dist)]
            
            # Calculate confidence intervals
            model_lower_ci, model_upper_ci = calculate_confidence_intervals(model_dist)
            confidence_intervals[metric][model] = {
                'lower_ci': model_lower_ci,
                'upper_ci': model_upper_ci
            }
            
            # Calculate effect size (Cohen's d)
            effect_d = cohens_d(ref_clean, model_clean)
            effect_sizes[metric][model] = effect_d
            
            # Check if we have enough data for statistical testing
            if len(ref_clean) < 3 or len(model_clean) < 3:
                results[metric][model] = {
                    'ttest_stat': np.nan, 'ttest_pvalue': np.nan,
                    'wilcoxon_stat': np.nan, 'wilcoxon_pvalue': np.nan,
                    'mannwhitney_stat': np.nan, 'mannwhitney_pvalue': np.nan,
                    'reference_mean': np.mean(ref_clean) if len(ref_clean) > 0 else np.nan,
                    'model_mean': np.mean(model_clean) if len(model_clean) > 0 else np.nan,
                    'effect_size': effect_d
                }
                continue
            
            # For paired t-test and Wilcoxon signed-rank test, we need to find common valid indices
            # (where both reference and model have non-NaN values)
            valid_mask = ~(np.isnan(reference_dist) | np.isnan(model_dist))
            if np.sum(valid_mask) < 3:
                # Not enough paired samples for paired tests, but we can still do Mann-Whitney
                ttest_stat, ttest_pvalue = np.nan, np.nan
                wilcoxon_stat, wilcoxon_pvalue = np.nan, np.nan
            else:
                # Paired t-test (since we're comparing the same bootstrap samples)
                try:
                    ref_paired = reference_dist[valid_mask]
                    model_paired = model_dist[valid_mask]
                    ttest_stat, ttest_pvalue = stats.ttest_rel(ref_paired, model_paired)
                except:
                    ttest_stat, ttest_pvalue = np.nan, np.nan
                
                # Wilcoxon signed-rank test (non-parametric paired test)
                try:
                    wilcoxon_stat, wilcoxon_pvalue = wilcoxon(
                        ref_paired, model_paired, alternative='two-sided'
                    )
                except:
                    wilcoxon_stat, wilcoxon_pvalue = np.nan, np.nan
            
            # Mann-Whitney U test (non-parametric) - can handle different sample sizes
            try:
                mannwhitney_stat, mannwhitney_pvalue = mannwhitneyu(
                    ref_clean, model_clean, alternative='two-sided'
                )
            except:
                mannwhitney_stat, mannwhitney_pvalue = np.nan, np.nan
            
            results[metric][model] = {
                'ttest_stat': ttest_stat,
                'ttest_pvalue': ttest_pvalue,
                'wilcoxon_stat': wilcoxon_stat,
                'wilcoxon_pvalue': wilcoxon_pvalue,
                'mannwhitney_stat': mannwhitney_stat,
                'mannwhitney_pvalue': mannwhitney_pvalue,
                'reference_mean': np.mean(ref_clean),
                'model_mean': np.mean(model_clean),
                'reference_std': np.std(ref_clean),
                'model_std': np.std(model_clean),
                'effect_size': effect_d
            }
    
    return results, confidence_intervals, effect_sizes


def save_results(bootstrap_results, statistical_results, confidence_intervals, effect_sizes, reference_model, output_dir):
    """Save all results to Excel file with multiple sheets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Excel workbook
    wb = openpyxl.Workbook()
    
    # Remove default sheet
    wb.remove(wb.active)
    
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
    
    # Sheet 1: Bootstrap distributions
    for metric in metrics:
        metric_data = []
        for model, distributions in bootstrap_results.items():
            for i, value in enumerate(distributions[metric]):
                metric_data.append({
                    'model': model,
                    'bootstrap_iteration': i,
                    'value': value
                })
        
        df = pd.DataFrame(metric_data)
        ws = wb.create_sheet(title=f'bootstrap_{metric}')
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)
    
    # Sheet 2: Statistical test results
    test_results = []
    for metric in metrics:
        for model, stats_dict in statistical_results[metric].items():
            test_results.append({
                'metric': metric,
                'comparison_model': model,
                'reference_model': reference_model,
                'reference_mean': stats_dict['reference_mean'],
                'reference_std': stats_dict['reference_std'],
                'model_mean': stats_dict['model_mean'],
                'model_std': stats_dict['model_std'],
                'ttest_statistic': stats_dict['ttest_stat'],
                'ttest_pvalue': stats_dict['ttest_pvalue'],
                'wilcoxon_statistic': stats_dict['wilcoxon_stat'],
                'wilcoxon_pvalue': stats_dict['wilcoxon_pvalue'],
                'mannwhitney_statistic': stats_dict['mannwhitney_stat'],
                'mannwhitney_pvalue': stats_dict['mannwhitney_pvalue'],
                'effect_size_cohens_d': stats_dict['effect_size'],
                'ttest_significant': stats_dict['ttest_pvalue'] < 0.05 if not np.isnan(stats_dict['ttest_pvalue']) else False,
                'wilcoxon_significant': stats_dict['wilcoxon_pvalue'] < 0.05 if not np.isnan(stats_dict['wilcoxon_pvalue']) else False,
                'mannwhitney_significant': stats_dict['mannwhitney_pvalue'] < 0.05 if not np.isnan(stats_dict['mannwhitney_pvalue']) else False
            })
    
    df_tests = pd.DataFrame(test_results)
    ws = wb.create_sheet(title='statistical_tests')
    for r in dataframe_to_rows(df_tests, index=False, header=True):
        ws.append(r)
    
    # Sheet 3: Summary statistics
    summary_data = []
    for model, distributions in bootstrap_results.items():
        for metric in metrics:
            values = np.array(distributions[metric])
            clean_values = values[~np.isnan(values)]
            
            summary_data.append({
                'model': model,
                'metric': metric,
                'mean': np.mean(clean_values) if len(clean_values) > 0 else np.nan,
                'std': np.std(clean_values) if len(clean_values) > 0 else np.nan,
                'median': np.median(clean_values) if len(clean_values) > 0 else np.nan,
                'q25': np.percentile(clean_values, 25) if len(clean_values) > 0 else np.nan,
                'q75': np.percentile(clean_values, 75) if len(clean_values) > 0 else np.nan,
                'min': np.min(clean_values) if len(clean_values) > 0 else np.nan,
                'max': np.max(clean_values) if len(clean_values) > 0 else np.nan
            })
    
    df_summary = pd.DataFrame(summary_data)
    ws = wb.create_sheet(title='summary_statistics')
    for r in dataframe_to_rows(df_summary, index=False, header=True):
        ws.append(r)
    
    # Sheet 4: Confidence intervals
    ci_data = []
    for metric in metrics:
        for model, ci_dict in confidence_intervals[metric].items():
            ci_data.append({
                'metric': metric,
                'model': model,
                'lower_ci_95': ci_dict['lower_ci'],
                'upper_ci_95': ci_dict['upper_ci'],
                'ci_width': ci_dict['upper_ci'] - ci_dict['lower_ci'] if not np.isnan(ci_dict['upper_ci']) and not np.isnan(ci_dict['lower_ci']) else np.nan
            })
    
    df_ci = pd.DataFrame(ci_data)
    ws = wb.create_sheet(title='confidence_intervals')
    for r in dataframe_to_rows(df_ci, index=False, header=True):
        ws.append(r)
    
    # Sheet 5: Effect sizes
    es_data = []
    for metric in metrics:
        for model, effect_d in effect_sizes[metric].items():
            es_data.append({
                'metric': metric,
                'comparison_model': model,
                'reference_model': reference_model,
                'cohens_d': effect_d,
                'effect_size_magnitude': 'negligible' if abs(effect_d) < 0.2 else 'small' if abs(effect_d) < 0.5 else 'medium' if abs(effect_d) < 0.8 else 'large'
            })
    
    df_es = pd.DataFrame(es_data)
    ws = wb.create_sheet(title='effect_sizes')
    for r in dataframe_to_rows(df_es, index=False, header=True):
        ws.append(r)
    
    # Save Excel file
    output_file = os.path.join(output_dir, 'slide_model_comparison_results.xlsx')
    wb.save(output_file)
    
    print(f"Results saved to {output_file}")
    print(f"  - bootstrap_* sheets: Bootstrap distributions for each metric")
    print(f"  - statistical_tests sheet: Statistical test results with effect sizes")
    print(f"  - summary_statistics sheet: Summary statistics for each model and metric")
    print(f"  - confidence_intervals sheet: 95% confidence intervals")
    print(f"  - effect_sizes sheet: Cohen's d effect sizes with magnitude interpretations")


def create_publication_boxplots(bootstrap_results, statistical_results, reference_model, output_dir):
    """Create individual publication-ready box plots for each metric with custom y-axis ranges."""
    
    # Define model colors for slide-level models
    model_colors = {
        'titan': '#F6C6AD',      # TITAN
        'madeleine': '#A6CAEC',  # MADELEINE
        'chief': '#B4E5A2',      # CHIEF
        'combined': '#FFFFFF',   # COMBINED (white with dots)
        'pruned': '#FFFFFF',     # PRUNED (white)
        'ensemble': '#FFFFFF'    # ENSEMBLE (white with cross hatches)
    }
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Remove accuracy from metrics
    metrics = ['auc', 'sensitivity', 'specificity', 'f1']
    metric_labels = {
        'auc': 'AUC',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'f1': 'F1 Score'
    }
    
    # Define custom y-axis ranges for each metric
    y_ranges = {
        'f1': (0.4, 1.0),
        'auc': (0.3, 1.0),
        'sensitivity': (0.0, 1.0),
        'specificity': (0.0, 1.0)
    }
    
    # Define the desired order for models: pruned, combined, ensemble, titan, madeleine, chief
    desired_order = ['pruned', 'combined', 'ensemble', 'titan', 'madeleine', 'chief']
    
    # Get all models that exist in the data
    all_models = set()
    for metric in metrics:
        for model in bootstrap_results.keys():
            values = np.array(bootstrap_results[model][metric])
            clean_values = values[~np.isnan(values)]
            if len(clean_values) > 0:
                all_models.add(model)
    
    # Sort models according to desired order, keeping only those that exist in the data
    models_in_plot = [model for model in desired_order if model in all_models]
    
    # Create subplots for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    if len(metrics) == 1:
        axes = [axes]  # Make it iterable for single metric case
    
    # Process each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Prepare data for plotting
        plot_data = []
        model_names = []
        
        # Add reference model (pruned) first
        if reference_model in bootstrap_results:
            ref_values = np.array(bootstrap_results[reference_model][metric])
            ref_clean = ref_values[~np.isnan(ref_values)]
            if len(ref_clean) > 0:
                plot_data.extend(ref_clean)
                model_names.extend([reference_model] * len(ref_clean))
        
        # Add other models
        for model_name in bootstrap_results.keys():
            if model_name != reference_model:
                model_values = np.array(bootstrap_results[model_name][metric])
                model_clean = model_values[~np.isnan(model_values)]
                if len(model_clean) > 0:
                    plot_data.extend(model_clean)
                    model_names.extend([model_name] * len(model_clean))
        
        if not plot_data:
            continue
        
        # Prepare data for each model
        model_data_list = []
        for model in models_in_plot:
            model_data = [plot_data[i] for i, name in enumerate(model_names) if name == model]
            model_data_list.append(model_data)
        
        # Create box plot
        box_plot = ax.boxplot(model_data_list,
                             labels=models_in_plot,
                             patch_artist=True,
                             showfliers=False,  # Hide outliers for cleaner look
                             widths=0.4)  # Make boxes narrower
        
        # Apply colors and styling
        for patch, model in zip(box_plot['boxes'], models_in_plot):
            color = model_colors.get(model, '#CCCCCC')  # Default gray if model not in color map
            
            # Special styling for specific models
            if model == 'combined':
                # COMBINED model: white with dots
                patch.set_facecolor('white')
                patch.set_hatch('...')  # Dotted pattern
                patch.set_alpha(1.0)
            elif model == 'pruned':
                # PRUNED model: pure white fill
                patch.set_facecolor('white')
                patch.set_alpha(1.0)
            elif model == 'ensemble':
                # ENSEMBLE model: white with cross hatches
                patch.set_facecolor('white')
                patch.set_hatch('xxx')  # Cross lines pattern
                patch.set_alpha(1.0)
            else:
                # Other models (titan, madeleine, chief): use original colors with alpha
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Make box borders thicker and black
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        
        # Make whiskers, caps, and medians thicker and black
        for whisker in box_plot['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(2)
        
        for cap in box_plot['caps']:
            cap.set_color('black')
            cap.set_linewidth(2)
        
        for median in box_plot['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # Add significance stars
        for i, model in enumerate(models_in_plot):
            if model != reference_model and model in statistical_results[metric]:
                stats_dict = statistical_results[metric][model]
                # Use Wilcoxon p-value for significance (most robust)
                p_value = stats_dict.get('wilcoxon_pvalue', np.nan)
                if not np.isnan(p_value) and p_value < 0.05:
                    # Determine star symbol based on p-value
                    if p_value < 0.001:
                        star = '***'
                    elif p_value < 0.01:
                        star = '**'
                    else:
                        star = '*'
                    
                    # Position star above the whisker cap
                    # Get the whisker cap position for this box
                    whisker_cap = box_plot['caps'][i*2 + 1]  # Upper whisker cap
                    whisker_y = whisker_cap.get_ydata()[0]  # Get y-coordinate of whisker cap
                    
                    # Position star slightly above the whisker cap
                    ax.text(i + 1, whisker_y + 0.02, star, 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize the plot
        ax.set_title(metric_labels[metric], fontsize=14, fontweight='bold', pad=25)
        
        # Set custom y-axis range for this metric
        y_min, y_max = y_ranges[metric]
        ax.set_ylim(y_min, y_max)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Remove grid lines
        ax.grid(False)
        
        # Remove x-axis ticks and labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        # Remove y-axis labels (keep only ticks)
        ax.set_ylabel('')
    
    
    # Adjust layout for the combined plot
    plt.tight_layout()
    
    # Save the single combined plot
    output_path = os.path.join(output_dir, 'combined_metrics_slide_boxplot.png')
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved combined box plot: {output_path}")


def main():
    args = parse_args()
    
    print("="*80)
    print("Slide-level MLP Model Performance Comparison with Bootstrap Analysis")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of individual models: {len(args.individual_model_dirs)}")
    print("="*80)
    
    # Validate input arguments
    if len(args.individual_model_dirs) != len(args.individual_feature_dirs):
        raise RuntimeError("Number of individual model directories must match number of individual feature directories")
    
    # Device handling
    requested_device = args.device
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(requested_device)
    
    # Step 1: Validate model directories and extract model names
    print("\n1. Validating model directories...")
    individual_folders = {}
    individual_feature_dirs = {}
    
    for i, (model_dir, feature_dir) in enumerate(zip(args.individual_model_dirs, args.individual_feature_dirs)):
        # Extract model name from directory path (e.g., /path/to/MADELEINE -> madeleine)
        model_name = os.path.basename(model_dir).lower()
        
        # Ensure unique model names (in case of duplicates)
        original_model_name = model_name
        counter = 1
        while model_name in individual_folders:
            model_name = f"{original_model_name}_{counter}"
            counter += 1
        
        if not os.path.exists(model_dir):
            raise RuntimeError(f"Individual model directory not found: {model_dir}")
        if not os.path.exists(feature_dir):
            raise RuntimeError(f"Individual feature directory not found: {feature_dir}")
        
        individual_folders[model_name] = model_dir
        individual_feature_dirs[model_name] = feature_dir
        print(f"  {model_name}: {os.path.basename(model_dir)} -> {os.path.basename(feature_dir)}")
    
    # Validate pruned model directory
    if not os.path.exists(args.pruned_model_dir):
        raise RuntimeError(f"Pruned model directory not found: {args.pruned_model_dir}")
    if not os.path.exists(args.pruned_feature_dir):
        raise RuntimeError(f"Pruned feature directory not found: {args.pruned_feature_dir}")
    
    print(f"  pruned: {os.path.basename(args.pruned_model_dir)} -> {os.path.basename(args.pruned_feature_dir)}")
    
    # Validate combined model directory
    if not os.path.exists(args.combined_model_dir):
        raise RuntimeError(f"Combined model directory not found: {args.combined_model_dir}")
    if not os.path.exists(args.combined_feature_dir):
        raise RuntimeError(f"Combined feature directory not found: {args.combined_feature_dir}")
    
    print(f"  combined: {os.path.basename(args.combined_model_dir)} -> {os.path.basename(args.combined_feature_dir)}")
    
    # Step 2: Load and process labels and split file
    print("\n2. Loading and processing labels and split file...")
    label_map = load_and_remap_labels(args.labels_csv, args.task)
    test_slides = load_split_file(args.split_file)
    
    # Filter test slides to only include those that have labels
    test_slides = [slide_id for slide_id in test_slides if slide_id in label_map]
    print(f"Found {len(test_slides)} test slides with labels")
    
    # Validate that test slides exist in feature directories
    print("Validating test slides exist in feature directories...")
    valid_test_slides = []
    for slide_id in test_slides:
        # Check if slide exists in any feature directory
        found = False
        for feature_dir in [args.pruned_feature_dir, args.combined_feature_dir] + list(individual_feature_dirs.values()):
            if os.path.exists(os.path.join(feature_dir, f"{slide_id}.h5")):
                found = True
                break
        if found:
            valid_test_slides.append(slide_id)
        else:
            print(f"Warning: Test slide {slide_id} not found in any feature directory")
    
    test_slides = valid_test_slides
    print(f"Using {len(test_slides)} test slides that exist in feature directories")
    
    # Step 3: Infer embedding dimensions for each model
    print("\n3. Inferring embedding dimensions...")
    embed_dims = {}
    
    # Get embedding dimension for pruned model
    sample_files = [f for f in os.listdir(args.pruned_feature_dir) if f.endswith('.h5')]
    if sample_files:
        sample_path = os.path.join(args.pruned_feature_dir, sample_files[0])
        embed_dims['pruned'] = load_feature_vector(sample_path).shape[0]
        print(f"Pruned model embedding dimension: {embed_dims['pruned']}")
    else:
        raise RuntimeError(f"No .h5 files found in pruned feature directory: {args.pruned_feature_dir}")
    
    # Get embedding dimension for combined model
    sample_files = [f for f in os.listdir(args.combined_feature_dir) if f.endswith('.h5')]
    if sample_files:
        sample_path = os.path.join(args.combined_feature_dir, sample_files[0])
        embed_dims['combined'] = load_feature_vector(sample_path).shape[0]
        print(f"Combined model embedding dimension: {embed_dims['combined']}")
    else:
        raise RuntimeError(f"No .h5 files found in combined feature directory: {args.combined_feature_dir}")
    
    # Get embedding dimensions for individual models
    for model_name, feature_dir in individual_feature_dirs.items():
        sample_files = [f for f in os.listdir(feature_dir) if f.endswith('.h5')]
        if sample_files:
            sample_path = os.path.join(feature_dir, sample_files[0])
            embed_dims[model_name] = load_feature_vector(sample_path).shape[0]
            print(f"{model_name} embedding dimension: {embed_dims[model_name]}")
        else:
            raise RuntimeError(f"No .h5 files found in {model_name} feature directory: {feature_dir}")
    
    # Step 4: Select best folds for each model
    print("\n4. Selecting best folds for each model...")
    best_folds = {}
    
    # Individual models
    for model_name, model_dir in individual_folders.items():
        best_folds[model_name] = select_best_fold_for_model(model_dir)
    
    # Pruned model
    best_folds['pruned'] = select_best_fold_for_model(args.pruned_model_dir)
    
    # Combined model
    best_folds['combined'] = select_best_fold_for_model(args.combined_model_dir)
    
    # Step 5: Get true labels for test slides
    print("\n5. Preparing test data...")
    y_true = np.array([label_map[slide_id] for slide_id in test_slides])
    
    print(f"Using {len(test_slides)} test slides for evaluation")
    
    # Step 6: Generate predictions for all models
    print("\n6. Generating predictions for all models...")
    all_predictions = {}
    all_probabilities = {}
    
    # Individual models
    for model_name, model_dir in individual_folders.items():
        print(f"\nLoading {model_name} model...")
        print(f"  Model directory: {os.path.basename(model_dir)}")
        print(f"  Best fold: {best_folds[model_name]} (embedding dim: {embed_dims[model_name]})")
        
        model = load_model_for_fold(model_dir, best_folds[model_name], 
                                   embed_dims[model_name], device)
        
        # Use the corresponding feature directory for this model
        feature_dir = individual_feature_dirs[model_name]
        print(f"  Using features from: {os.path.basename(feature_dir)}")
        
        predictions, probabilities = generate_predictions_for_model(
            model, test_slides, feature_dir, device, model_name
        )
        all_predictions[model_name] = predictions
        all_probabilities[model_name] = probabilities
    
    # Pruned model
    print(f"\nLoading pruned model...")
    print(f"  Model directory: {os.path.basename(args.pruned_model_dir)}")
    print(f"  Best fold: {best_folds['pruned']} (embedding dim: {embed_dims['pruned']})")
    
    pruned_model = load_model_for_fold(args.pruned_model_dir, best_folds['pruned'],
                                      embed_dims['pruned'], device)
    
    print(f"  Using features from: {os.path.basename(args.pruned_feature_dir)}")
    
    pruned_predictions, pruned_probabilities = generate_predictions_for_model(
        pruned_model, test_slides, args.pruned_feature_dir, device, "pruned"
    )
    all_predictions['pruned'] = pruned_predictions
    all_probabilities['pruned'] = pruned_probabilities
    
    # Combined model
    print(f"\nLoading combined model...")
    print(f"  Model directory: {os.path.basename(args.combined_model_dir)}")
    print(f"  Best fold: {best_folds['combined']} (embedding dim: {embed_dims['combined']})")
    
    combined_model = load_model_for_fold(args.combined_model_dir, best_folds['combined'],
                                     embed_dims['combined'], device)
    
    print(f"  Using features from: {os.path.basename(args.combined_feature_dir)}")
    
    combined_predictions, combined_probabilities = generate_predictions_for_model(
        combined_model, test_slides, args.combined_feature_dir, device, "combined"
    )
    all_predictions['combined'] = combined_predictions
    all_probabilities['combined'] = combined_probabilities
    
    # Step 7: Create majority vote ensemble
    print("\n7. Creating majority vote ensemble...")
    ensemble_predictions = create_majority_vote_ensemble({
        model: all_predictions[model] for model in individual_folders.keys()
    })
    all_predictions['ensemble'] = ensemble_predictions
    
    # For ensemble probabilities, use mean of individual model probabilities
    ensemble_probabilities = np.mean([all_probabilities[model] 
                                    for model in individual_folders.keys()], axis=0)
    all_probabilities['ensemble'] = ensemble_probabilities
    
    # Step 8: Compute full test set metrics for verification
    print("\n8. Computing full test set metrics for verification...")
    full_test_metrics = {}
    for model_name in all_predictions.keys():
        model_metrics = compute_metrics(y_true, all_predictions[model_name], 
                                       all_probabilities[model_name], args.n_classes)
        full_test_metrics[model_name] = model_metrics
    
    # Print full test set results
    print("\n" + "="*80)
    print("FULL TEST SET PERFORMANCE (for verification)")
    print("="*80)
    print(f"{'Model':<12} {'Accuracy':<10} {'F1 Score':<10} {'AUC':<10} {'Sensitivity':<12} {'Specificity':<12}")
    print("-" * 80)
    for model_name, metrics in full_test_metrics.items():
        print(f"{model_name:<12} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} "
              f"{metrics['auc']:<10.4f} {metrics['sensitivity']:<12.4f} {metrics['specificity']:<12.4f}")
    print("="*80)
    
    # Step 9: Perform bootstrap analysis
    print("\n9. Performing bootstrap analysis...")
    bootstrap_results = perform_bootstrap_analysis(
        y_true, all_predictions, all_probabilities, args.n_classes,
        args.bootstrap_iterations, args.bootstrap_fraction
    )
    
    # Step 10: Perform statistical tests
    print("\n10. Performing statistical tests...")
    statistical_results, confidence_intervals, effect_sizes = perform_statistical_tests(bootstrap_results, 'pruned')
    
    # Step 11: Save results
    print("\n11. Saving results...")
    save_results(bootstrap_results, statistical_results, confidence_intervals, effect_sizes, 'pruned', args.output_dir)
    
    # Step 12: Create publication-ready box plots
    print("\n12. Creating publication-ready box plots...")
    create_publication_boxplots(bootstrap_results, statistical_results, 'pruned', args.output_dir)
    
    # Step 13: Print summary
    print("\n" + "="*80)
    print("SUMMARY OF STATISTICAL TESTS (Pruned vs Others)")
    print("="*80)
    
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-" * 40)
        for model, stats_dict in statistical_results[metric].items():
            if not np.isnan(stats_dict['mannwhitney_pvalue']):
                # At least Mann-Whitney test was successful
                mw_sig = "***" if stats_dict['mannwhitney_pvalue'] < 0.001 else "**" if stats_dict['mannwhitney_pvalue'] < 0.01 else "*" if stats_dict['mannwhitney_pvalue'] < 0.05 else ""
                effect_size = stats_dict['effect_size']
                effect_magnitude = 'negligible' if abs(effect_size) < 0.2 else 'small' if abs(effect_size) < 0.5 else 'medium' if abs(effect_size) < 0.8 else 'large'
                
                print(f"{model:12} | Pruned: {stats_dict['reference_mean']:.4f}±{stats_dict['reference_std']:.4f} | "
                      f"{model}: {stats_dict['model_mean']:.4f}±{stats_dict['model_std']:.4f}")
                
                if not np.isnan(stats_dict['ttest_pvalue']) and not np.isnan(stats_dict['wilcoxon_pvalue']):
                    # All three tests were successful
                    ttest_sig = "***" if stats_dict['ttest_pvalue'] < 0.001 else "**" if stats_dict['ttest_pvalue'] < 0.01 else "*" if stats_dict['ttest_pvalue'] < 0.05 else ""
                    wilcoxon_sig = "***" if stats_dict['wilcoxon_pvalue'] < 0.001 else "**" if stats_dict['wilcoxon_pvalue'] < 0.01 else "*" if stats_dict['wilcoxon_pvalue'] < 0.05 else ""
                    print(f"{'':12} | t-test p={stats_dict['ttest_pvalue']:.4f}{ttest_sig} | "
                          f"Wilcoxon p={stats_dict['wilcoxon_pvalue']:.4f}{wilcoxon_sig} | "
                          f"Mann-Whitney p={stats_dict['mannwhitney_pvalue']:.4f}{mw_sig} | "
                          f"Cohen's d={effect_size:.3f} ({effect_magnitude})")
                elif not np.isnan(stats_dict['wilcoxon_pvalue']):
                    # Wilcoxon and Mann-Whitney tests were successful
                    wilcoxon_sig = "***" if stats_dict['wilcoxon_pvalue'] < 0.001 else "**" if stats_dict['wilcoxon_pvalue'] < 0.01 else "*" if stats_dict['wilcoxon_pvalue'] < 0.05 else ""
                    print(f"{'':12} | t-test p=N/A (insufficient paired samples) | "
                          f"Wilcoxon p={stats_dict['wilcoxon_pvalue']:.4f}{wilcoxon_sig} | "
                          f"Mann-Whitney p={stats_dict['mannwhitney_pvalue']:.4f}{mw_sig} | "
                          f"Cohen's d={effect_size:.3f} ({effect_magnitude})")
                else:
                    # Only Mann-Whitney test was successful
                    print(f"{'':12} | t-test p=N/A | Wilcoxon p=N/A | "
                          f"Mann-Whitney p={stats_dict['mannwhitney_pvalue']:.4f}{mw_sig} | "
                          f"Cohen's d={effect_size:.3f} ({effect_magnitude})")
            else:
                print(f"{model:12} | Insufficient data for statistical testing")
    
    print("\n" + "="*80)
    print("Legend: *** p<0.001, ** p<0.01, * p<0.05")
    print("="*80)


if __name__ == '__main__':
    main()

