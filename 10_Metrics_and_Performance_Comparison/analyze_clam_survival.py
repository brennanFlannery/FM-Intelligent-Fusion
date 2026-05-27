#!/usr/bin/env python3
"""
CLAM Survival Analysis Script

This script uses a CLAM model to predict high/low grade for patients, matches
predictions to survival data from the master Excel file, and generates survival
curves with statistical comparisons for each split (train/val/test).
"""

import argparse
import os
import sys
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Import CLAM models - adjust path as needed
try:
    from models.model_clam import CLAM_SB, CLAM_MB
except ImportError:
    # Try alternative import path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TRIDENT', 'CLAM', 'CLAM'))
    from models.model_clam import CLAM_SB, CLAM_MB


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze survival by CLAM-predicted grade groups"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to CLAM checkpoint file OR model directory containing multiple folds (will auto-select best fold if directory)"
    )
    parser.add_argument(
        "--feature_dir", type=str, required=True,
        help="Directory containing .h5 feature files"
    )
    parser.add_argument(
        "--splits_csv", type=str, default=None,
        help="Path to splits CSV file (default: kirc_splits/splits_0.csv)"
    )
    parser.add_argument(
        "--master_xlsx", type=str, default=None,
        help="Path to master Excel file (default: kca_master_hpc_cptacupdated.xlsx)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for output plots and CSV (default: survival_analysis_output/)"
    )
    parser.add_argument(
        "--model_type", choices=['clam_sb', 'clam_mb'], default='clam_sb',
        help="CLAM architecture type"
    )
    parser.add_argument(
        "--model_size", choices=['small', 'big'], default='small',
        help="Model size"
    )
    parser.add_argument(
        "--n_classes", type=int, default=2,
        help="Number of classes"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=None,
        help="Feature embedding dimension (will infer from .h5 file if not provided)"
    )
    parser.add_argument(
        "--device", type=str, default='cuda:0',
        help="PyTorch device identifier"
    )
    return parser.parse_args()


def load_splits_file(csv_path: str) -> pd.DataFrame:
    """Load the splits CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Splits file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        required_cols = ['train', 'val', 'test']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Splits file missing required columns: {missing_cols}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading splits file: {e}")


def load_master_file(excel_path: str) -> pd.DataFrame:
    """Load the master Excel file with survival data."""
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Master file not found: {excel_path}")
    
    try:
        df = pd.read_excel(excel_path)
        required_cols = ['PatientID', 'vital_status', 'days_to_last_followup', 'death_days_to']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Master file missing required columns: {missing_cols}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading master file: {e}")


def extract_patient_id(full_id: str) -> Optional[str]:
    """Extract short patient ID from full slide ID format."""
    if pd.isna(full_id) or not isinstance(full_id, str) or not full_id.strip():
        return None
    
    # Extract first three segments separated by dashes
    parts = full_id.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    
    # Fallback: try regex pattern
    import re
    match = re.match(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)', full_id)
    if match:
        return match.group(1)
    
    return None


def match_patients_to_master(splits_df: pd.DataFrame, master_df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], List[str]]:
    """Match slide IDs from splits to PatientID in master file."""
    # Create mapping from short patient ID to master file row
    master_mapping = {}
    for idx, row in master_df.iterrows():
        patient_id = row['PatientID']
        if pd.notna(patient_id):
            patient_id_str = str(patient_id).strip()
            master_mapping[patient_id_str] = row
    
    # Extract and match patients from splits
    slide_to_master = {}
    unmatched_slides = []
    
    # Collect all slide IDs from all splits
    all_slide_ids = set()
    for col in ['train', 'val', 'test']:
        if col in splits_df.columns:
            for slide_id in splits_df[col].dropna():
                all_slide_ids.add(slide_id)
    
    # Match each slide
    for full_id in all_slide_ids:
        short_id = extract_patient_id(full_id)
        if short_id is None:
            unmatched_slides.append(full_id)
            continue
        
        if short_id in master_mapping:
            slide_to_master[full_id] = master_mapping[short_id]
        else:
            unmatched_slides.append(full_id)
    
    return slide_to_master, unmatched_slides


def infer_embed_dim(feature_dir: str) -> int:
    """Infer embedding dimension from first .h5 file."""
    h5_files = [f for f in os.listdir(feature_dir) if f.endswith('.h5')]
    if not h5_files:
        raise RuntimeError(f"No .h5 files found in {feature_dir}")
    
    sample_file = os.path.join(feature_dir, h5_files[0])
    with h5py.File(sample_file, 'r') as f:
        if 'features' not in f:
            raise RuntimeError(f"No 'features' key in {sample_file}")
        embed_dim = f['features'].shape[1]
    return embed_dim


def select_best_fold_for_model(model_dir: str) -> int:
    """
    Select the best fold for a model based on F1 score on test set.
    
    Adapted from compare_model_performance.py
    """
    metrics_file = os.path.join(model_dir, 'final_metric_summary.csv')
    if not os.path.exists(metrics_file):
        raise RuntimeError(f"Metrics file not found: {metrics_file}. Cannot auto-select best fold.")
    
    df = pd.read_csv(metrics_file)
    test_df = df[df['split'] == 'test'].copy()
    
    if test_df.empty:
        raise RuntimeError(f"No test metrics found in {metrics_file}")
    
    # Select fold with highest F1 score
    if 'f1_score' not in test_df.columns:
        raise RuntimeError(f"'f1_score' column not found in {metrics_file}")
    
    if not test_df['f1_score'].notna().any():
        raise RuntimeError(f"All f1_score values are NaN in {metrics_file}")
    
    best_fold = test_df.loc[test_df['f1_score'].idxmax(), 'fold']
    best_f1 = test_df.loc[test_df['f1_score'].idxmax(), 'f1_score']
    
    print(f"Selected best fold {best_fold} with test F1={best_f1:.4f}")
    return int(best_fold)


def load_clam_model(model_path: str, model_type: str, model_size: str,
                    embed_dim: int, n_classes: int, device: str):
    """Load CLAM checkpoint from provided path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    kwargs = dict(
        gate=True,
        size_arg=model_size,
        dropout=0.0,
        k_sample=1,
        n_classes=n_classes,
        subtyping=False,
        embed_dim=embed_dim
    )
    model = CLAM_SB(**kwargs) if model_type == 'clam_sb' else CLAM_MB(**kwargs)
    
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def load_clam_model_from_dir(model_dir: str, model_type: str, model_size: str,
                             embed_dim: int, n_classes: int, device: str) -> Tuple[torch.nn.Module, int]:
    """
    Load CLAM model from directory, automatically selecting best fold if metrics file exists.
    
    Returns:
        Tuple of (model, fold_number)
    """
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Try to select best fold from metrics file
    metrics_file = model_dir_path / 'final_metric_summary.csv'
    if metrics_file.exists():
        try:
            fold = select_best_fold_for_model(str(model_dir_path))
        except Exception as e:
            print(f"Warning: Could not select best fold from metrics: {e}", file=sys.stderr)
            print("Falling back to finding first available checkpoint...", file=sys.stderr)
            fold = None
    else:
        fold = None
    
    # If no metrics file or selection failed, find first available checkpoint
    if fold is None:
        import glob
        ckpts = sorted(glob.glob(str(model_dir_path / 's_*_checkpoint.pt')))
        if not ckpts:
            raise RuntimeError(f"No checkpoints found in {model_dir}")
        # Extract fold number from first checkpoint
        import re
        match = re.match(r'.*s_(\d+)_checkpoint\.pt', os.path.basename(ckpts[0]))
        if not match:
            raise RuntimeError(f"Could not extract fold number from {ckpts[0]}")
        fold = int(match.group(1))
        print(f"Using fold {fold} (first available checkpoint)")
    
    # Load model for the selected fold
    ckpt_file = model_dir_path / f's_{fold}_checkpoint.pt'
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")
    
    model = load_clam_model(str(ckpt_file), model_type, model_size, embed_dim, n_classes, device)
    return model, fold


def slide_prediction(model, feat_path: str, device: str) -> Tuple[np.ndarray, int]:
    """Get CLAM prediction for a single slide."""
    with h5py.File(feat_path, 'r') as f:
        feats = f['features'][:]
    feats_t = torch.from_numpy(feats).to(device).float()
    lbl = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        logits, prob, pred, _, _ = model(feats_t, lbl, instance_eval=False)
    return prob.cpu().numpy().ravel(), int(pred.item())


def get_clam_predictions(model, feature_dir: str, slide_ids: List[str], device: str) -> Dict[str, Tuple[int, np.ndarray]]:
    """Get CLAM predictions for multiple slides."""
    predictions = {}
    missing_files = []
    
    for slide_id in tqdm(slide_ids, desc="Getting CLAM predictions"):
        feat_path = os.path.join(feature_dir, f"{slide_id}.h5")
        if not os.path.exists(feat_path):
            missing_files.append(slide_id)
            continue
        
        try:
            prob, pred = slide_prediction(model, feat_path, device)
            predictions[slide_id] = (pred, prob)
        except Exception as e:
            print(f"Warning: Failed to get prediction for {slide_id}: {e}", file=sys.stderr)
            missing_files.append(slide_id)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} slides missing feature files or failed prediction", file=sys.stderr)
    
    return predictions


def is_valid_value(value) -> bool:
    """Check if a value is valid (non-empty, non-zero)."""
    if pd.isna(value):
        return False
    
    if isinstance(value, (int, float)):
        return value != 0
    
    if isinstance(value, str):
        stripped = value.strip()
        try:
            num_val = float(stripped)
            return num_val != 0
        except ValueError:
            return len(stripped) > 0
    
    return value is not None


def prepare_survival_data(master_row: pd.Series) -> Optional[Tuple[float, int]]:
    """
    Extract time-to-event and event indicator from master file row.
    
    Returns:
        Tuple of (time_days, event) where event=1 if death, 0 if censored
        Returns None if data is invalid
    """
    death_days = master_row.get('death_days_to', None)
    followup_days = master_row.get('days_to_last_followup', None)
    
    # Check if death_days_to is valid
    if is_valid_value(death_days):
        try:
            time = float(death_days)
            if time > 0:
                return (time, 1)  # Event occurred (death)
        except (ValueError, TypeError):
            pass
    
    # Check if days_to_last_followup is valid
    if is_valid_value(followup_days):
        try:
            time = float(followup_days)
            if time > 0:
                return (time, 0)  # Censored (alive at last followup)
        except (ValueError, TypeError):
            pass
    
    return None


def create_survival_dataset(predictions: Dict[str, Tuple[int, np.ndarray]],
                            master_mapping: Dict[str, pd.Series],
                            splits_df: pd.DataFrame,
                            split_name: str) -> pd.DataFrame:
    """Combine predictions with survival data for a specific split."""
    slide_ids = splits_df[split_name].dropna().tolist()
    
    data = []
    for slide_id in slide_ids:
        if slide_id not in predictions:
            continue
        
        if slide_id not in master_mapping:
            continue
        
        pred, prob = predictions[slide_id]
        master_row = master_mapping[slide_id]
        
        survival_data = prepare_survival_data(master_row)
        if survival_data is None:
            continue
        
        time, event = survival_data
        patient_id = extract_patient_id(slide_id)
        
        data.append({
            'slide_id': slide_id,
            'patient_id': patient_id,
            'predicted_grade': pred,  # 0=low, 1=high
            'probability': prob,
            'time': time,
            'event': event
        })
    
    return pd.DataFrame(data)


def plot_survival_curves(survival_df: pd.DataFrame, split_name: str, output_dir: str) -> Tuple[Optional[str], Optional[float]]:
    """Plot Kaplan-Meier survival curves for high vs low grade groups."""
    if len(survival_df) == 0:
        print(f"Warning: No survival data for {split_name} split", file=sys.stderr)
        return None, None
    
    # Split by predicted grade
    low_grade = survival_df[survival_df['predicted_grade'] == 0]
    high_grade = survival_df[survival_df['predicted_grade'] == 1]
    
    if len(low_grade) == 0 or len(high_grade) == 0:
        print(f"Warning: Cannot plot {split_name} - one grade group is empty", file=sys.stderr)
        return None, None
    
    # Convert time from days to years
    time_low = low_grade['time'].values / 365.25
    time_high = high_grade['time'].values / 365.25
    event_low = low_grade['event'].values.astype(bool)
    event_high = high_grade['event'].values.astype(bool)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Fit Kaplan-Meier curves
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()
    
    # Low grade group (blue)
    kmf_low.fit(time_low, event_low, label=f'Low Grade (n={len(low_grade)})')
    kmf_low.plot_survival_function(ax=ax, color='blue', linewidth=3, ci_show=False)
    
    # High grade group (red)
    kmf_high.fit(time_high, event_high, label=f'High Grade (n={len(high_grade)})')
    kmf_high.plot_survival_function(ax=ax, color='red', linewidth=3, ci_show=False)
    
    # Perform log-rank test
    logrank_result = logrank_test(time_low, time_high, event_low, event_high)
    p_value = logrank_result.p_value
    
    # Formatting
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.set_title(
        f'Survival Analysis - {split_name.upper()} Set\n'
        f'Low Grade vs High Grade (log-rank p={p_value:.4f})',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(loc='best', fontsize=11)
    ax.grid(False)
    
    # Save figure
    output_path = Path(output_dir) / f'survival_curve_{split_name}.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path), p_value


def perform_logrank_test(survival_df: pd.DataFrame) -> Dict:
    """Perform log-rank test comparing high vs low grade groups."""
    if len(survival_df) == 0:
        return {'p_value': np.nan, 'test_statistic': np.nan}
    
    low_grade = survival_df[survival_df['predicted_grade'] == 0]
    high_grade = survival_df[survival_df['predicted_grade'] == 1]
    
    if len(low_grade) == 0 or len(high_grade) == 0:
        return {'p_value': np.nan, 'test_statistic': np.nan}
    
    time_low = low_grade['time'].values / 365.25
    time_high = high_grade['time'].values / 365.25
    event_low = low_grade['event'].values.astype(bool)
    event_high = high_grade['event'].values.astype(bool)
    
    try:
        logrank_result = logrank_test(time_low, time_high, event_low, event_high)
        return {
            'p_value': logrank_result.p_value,
            'test_statistic': logrank_result.test_statistic
        }
    except Exception as e:
        print(f"Warning: Log-rank test failed: {e}", file=sys.stderr)
        return {'p_value': np.nan, 'test_statistic': np.nan}


def analyze_survival_by_grade(model, feature_dir: str, splits_df: pd.DataFrame,
                               master_mapping: Dict[str, pd.Series],
                               output_dir: str, device: str) -> Dict:
    """Main analysis function for survival by predicted grade."""
    results = {}
    
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits_df.columns:
            results[split_name] = {
                'n_low_grade': 0,
                'n_high_grade': 0,
                'logrank_pvalue': np.nan,
                'logrank_statistic': np.nan,
                'median_survival_low': np.nan,
                'median_survival_high': np.nan,
                'plot_path': None
            }
            continue
        
        print(f"\nProcessing {split_name} split...")
        
        # Get slide IDs for this split
        slide_ids = splits_df[split_name].dropna().tolist()
        print(f"  Found {len(slide_ids)} slides in {split_name} split")
        
        # Get CLAM predictions
        predictions = get_clam_predictions(model, feature_dir, slide_ids, device)
        print(f"  Got predictions for {len(predictions)} slides")
        
        # Create survival dataset
        survival_df = create_survival_dataset(predictions, master_mapping, splits_df, split_name)
        print(f"  Matched {len(survival_df)} slides with valid survival data")
        
        if len(survival_df) == 0:
            results[split_name] = {
                'n_low_grade': 0,
                'n_high_grade': 0,
                'logrank_pvalue': np.nan,
                'logrank_statistic': np.nan,
                'median_survival_low': np.nan,
                'median_survival_high': np.nan,
                'plot_path': None
            }
            continue
        
        # Plot survival curves
        plot_path, p_value = plot_survival_curves(survival_df, split_name, output_dir)
        
        # Perform log-rank test
        logrank_stats = perform_logrank_test(survival_df)
        
        # Calculate median survival times
        low_grade = survival_df[survival_df['predicted_grade'] == 0]
        high_grade = survival_df[survival_df['predicted_grade'] == 1]
        
        median_low = np.nan
        median_high = np.nan
        
        if len(low_grade) > 0:
            kmf_low = KaplanMeierFitter()
            kmf_low.fit(low_grade['time'].values / 365.25, low_grade['event'].values.astype(bool))
            try:
                median_low = kmf_low.median_survival_time_
            except:
                pass
        
        if len(high_grade) > 0:
            kmf_high = KaplanMeierFitter()
            kmf_high.fit(high_grade['time'].values / 365.25, high_grade['event'].values.astype(bool))
            try:
                median_high = kmf_high.median_survival_time_
            except:
                pass
        
        results[split_name] = {
            'n_low_grade': len(low_grade),
            'n_high_grade': len(high_grade),
            'logrank_pvalue': logrank_stats['p_value'],
            'logrank_statistic': logrank_stats['test_statistic'],
            'median_survival_low': median_low,
            'median_survival_high': median_high,
            'plot_path': plot_path
        }
    
    return results


def save_results_csv(results: Dict, output_dir: str):
    """Save results to CSV file."""
    records = []
    for split_name in ['train', 'val', 'test']:
        if split_name in results:
            records.append({
                'split': split_name,
                'n_low_grade': results[split_name]['n_low_grade'],
                'n_high_grade': results[split_name]['n_high_grade'],
                'logrank_pvalue': results[split_name]['logrank_pvalue'],
                'logrank_statistic': results[split_name]['logrank_statistic'],
                'median_survival_low_years': results[split_name]['median_survival_low'],
                'median_survival_high_years': results[split_name]['median_survival_high']
            })
    
    df = pd.DataFrame(records)
    output_path = Path(output_dir) / 'survival_analysis_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


def print_summary(results: Dict):
    """Print formatted summary to console."""
    print("\n" + "=" * 70)
    print("Survival Analysis Summary")
    print("=" * 70)
    
    for split_name in ['train', 'val', 'test']:
        if split_name not in results:
            continue
        
        r = results[split_name]
        print(f"\n{split_name.upper()} Split:")
        print(f"  Low Grade:  n={r['n_low_grade']:3d},  Median survival: {r['median_survival_low']:.2f} years" if not pd.isna(r['median_survival_low']) else f"  Low Grade:  n={r['n_low_grade']:3d},  Median survival: N/A")
        print(f"  High Grade: n={r['n_high_grade']:3d},  Median survival: {r['median_survival_high']:.2f} years" if not pd.isna(r['median_survival_high']) else f"  High Grade: n={r['n_high_grade']:3d},  Median survival: N/A")
        
        p_val = r['logrank_pvalue']
        if not pd.isna(p_val):
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  Log-rank test: p={p_val:.4f} {sig}")
        else:
            print(f"  Log-rank test: N/A")
    
    print("\n" + "=" * 70)


def main():
    """Main function."""
    args = parse_args()
    
    # Set default paths
    script_dir = Path(__file__).parent
    if args.splits_csv is None:
        args.splits_csv = str(script_dir / 'kirc_splits' / 'splits_0.csv')
    if args.master_xlsx is None:
        args.master_xlsx = str(script_dir / 'kca_master_hpc_cptacupdated.xlsx')
    if args.output_dir is None:
        args.output_dir = str(script_dir / 'survival_analysis_output')
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load files
        print("Loading data files...")
        splits_df = load_splits_file(args.splits_csv)
        master_df = load_master_file(args.master_xlsx)
        print(f"Loaded {len(splits_df)} rows from splits file")
        print(f"Loaded {len(master_df)} rows from master file")
        
        # Match patients
        print("\nMatching patients to master file...")
        master_mapping, unmatched = match_patients_to_master(splits_df, master_df)
        print(f"Matched {len(master_mapping)} slides to master file")
        if unmatched:
            print(f"Warning: {len(unmatched)} slides could not be matched", file=sys.stderr)
        
        # Infer embed_dim if not provided
        if args.embed_dim is None:
            print("\nInferring embedding dimension from feature files...")
            args.embed_dim = infer_embed_dim(args.feature_dir)
            print(f"Inferred embed_dim: {args.embed_dim}")
        
        # Determine if model_path is a file or directory
        model_path_obj = Path(args.model_path)
        is_directory = model_path_obj.is_dir()
        
        # Load CLAM model
        if is_directory:
            print(f"\nLoading CLAM model from directory: {args.model_path}")
            print("Attempting to auto-select best fold from metrics...")
            model, selected_fold = load_clam_model_from_dir(
                args.model_path,
                args.model_type,
                args.model_size,
                args.embed_dim,
                args.n_classes,
                args.device
            )
            print(f"Model loaded successfully (using fold {selected_fold})")
            
            # If splits_csv not provided and we're using a directory, try to use splits from that directory
            if args.splits_csv is None or args.splits_csv == str(Path(__file__).parent / 'kirc_splits' / 'splits_0.csv'):
                splits_in_dir = model_path_obj / f'splits_{selected_fold}.csv'
                if splits_in_dir.exists():
                    print(f"Using splits file from model directory: {splits_in_dir}")
                    args.splits_csv = str(splits_in_dir)
        else:
            print(f"\nLoading CLAM model from checkpoint: {args.model_path}")
            model = load_clam_model(
                args.model_path,
                args.model_type,
                args.model_size,
                args.embed_dim,
                args.n_classes,
                args.device
            )
            print("Model loaded successfully")
        
        # Run analysis
        print("\nRunning survival analysis...")
        results = analyze_survival_by_grade(
            model,
            args.feature_dir,
            splits_df,
            master_mapping,
            args.output_dir,
            args.device
        )
        
        # Save results
        save_results_csv(results, args.output_dir)
        
        # Print summary
        print_summary(results)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
