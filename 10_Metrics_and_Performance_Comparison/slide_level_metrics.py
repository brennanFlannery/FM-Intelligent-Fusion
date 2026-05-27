#!/usr/bin/env python3
"""
slide_level_metrics.py

Calculate performance metrics for slide-level MLP models across all folds and splits.
For each model, computes metrics on train/val/test splits for folds 0, 1, and 2,
then saves results to final_metrics.csv in the model directory.

Example:
  python slide_level_metrics.py \
    --model_dirs /path/to/model1 /path/to/model2 /path/to/model3 \
    --feature_dirs /path/to/features1 /path/to/features2 /path/to/features3 \
    --splits_dir /path/to/splits \
    --labels_csv /path/to/labels.csv \
    --task task_kidney_grade \
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
import warnings
warnings.filterwarnings('ignore')


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
        description="Calculate slide-level MLP model metrics across all folds and splits"
    )
    parser.add_argument(
        "--model_dirs", type=str, nargs='+', required=True,
        help="List of model directories (each containing fold_0, fold_1, fold_2)"
    )
    parser.add_argument(
        "--feature_dirs", type=str, nargs='+', required=True,
        help="List of feature directories corresponding to models (same order as model_dirs)"
    )
    parser.add_argument(
        "--splits_dir", type=str, required=True,
        help="Directory containing splits_0.csv, splits_1.csv, splits_2.csv with train/val/test columns"
    )
    parser.add_argument(
        "--labels_csv", type=str, required=True,
        help="CSV mapping slide_id to integer label"
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
        "--n_classes", type=int, required=True,
        help="Number of output classes (after remapping)"
    )
    parser.add_argument(
        "--device", default='cuda:0',
        help="PyTorch device identifier"
    )
    parser.add_argument(
        "--num_folds", type=int, default=3,
        help="Number of folds to process (default: 3)"
    )
    return parser.parse_args()


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


def load_split_file(splits_dir, fold):
    """Load split file for a specific fold and return slide IDs for each split."""
    split_file = os.path.join(splits_dir, f'splits_{fold}.csv')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    split_df = pd.read_csv(split_file)
    
    # Check if this is the format with train/val/test columns
    if 'train' not in split_df.columns or 'val' not in split_df.columns or 'test' not in split_df.columns:
        raise ValueError(f"Split file must have 'train', 'val', 'test' columns: {split_file}")
    
    splits = {}
    for split_name in ['train', 'val', 'test']:
        splits[split_name] = split_df[split_name].dropna().astype(str).tolist()
    
    return splits


def generate_predictions_for_split(model, slide_ids, feature_dir, device, label_map):
    """Generate predictions for a split."""
    predictions = []
    probabilities = []
    true_labels = []
    valid_slide_ids = []
    
    for slide_id in slide_ids:
        # Skip slides without labels
        if slide_id not in label_map:
            continue
        
        feat_path = os.path.join(feature_dir, f"{slide_id}.h5")
        if not os.path.exists(feat_path):
            continue
        
        try:
            prob, pred = slide_prediction(model, feat_path, device)
            predictions.append(pred)
            probabilities.append(prob)
            true_labels.append(label_map[slide_id])
            valid_slide_ids.append(slide_id)
        except Exception as e:
            print(f"Warning: Failed to process {slide_id}: {e}")
            continue
    
    return (np.array(predictions), 
            np.array(probabilities), 
            np.array(true_labels),
            valid_slide_ids)


def compute_metrics(y_true, y_pred, y_prob, n_classes):
    """Compute all metrics for given predictions."""
    metrics = {}
    
    # Skip if no samples
    if len(y_true) == 0:
        return {
            'accuracy': np.nan,
            'f1': np.nan,
            'auc': np.nan,
            'sensitivity': np.nan,
            'specificity': np.nan
        }
    
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
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.
        except:
            metrics['sensitivity'] = np.nan
            metrics['specificity'] = np.nan
    else:
        metrics['sensitivity'] = np.nan
        metrics['specificity'] = np.nan
    
    return metrics


def process_model(model_dir, feature_dir, splits_dir, label_map, n_classes, num_folds, device):
    """Process a single model across all folds and splits."""
    model_name = os.path.basename(model_dir)
    print(f"\nProcessing model: {model_name}")
    print(f"  Model directory: {model_dir}")
    print(f"  Feature directory: {feature_dir}")
    
    # Infer input dimension
    sample_files = [f for f in os.listdir(feature_dir) if f.endswith('.h5')]
    if not sample_files:
        raise RuntimeError(f"No .h5 files found in feature directory: {feature_dir}")
    
    sample_path = os.path.join(feature_dir, sample_files[0])
    input_dim = load_feature_vector(sample_path).shape[0]
    print(f"  Input dimension: {input_dim}")
    
    # Store results for all folds
    all_results = []
    
    # Process each fold
    for fold in range(num_folds):
        print(f"\n  Processing fold {fold}...")
        
        # Check if fold exists
        fold_dir = os.path.join(model_dir, f'fold_{fold}')
        if not os.path.exists(fold_dir):
            print(f"    Warning: Fold directory not found: {fold_dir}")
            continue
        
        # Load splits for this fold
        try:
            splits = load_split_file(splits_dir, fold)
        except Exception as e:
            print(f"    Error loading split file: {e}")
            continue
        
        # Load model
        try:
            model = load_model_for_fold(model_dir, fold, input_dim, device)
        except Exception as e:
            print(f"    Error loading model: {e}")
            continue
        
        # Process each split
        for split_name in ['train', 'val', 'test']:
            slide_ids = splits[split_name]
            print(f"    Processing {split_name} split ({len(slide_ids)} slides)...")
            
            # Generate predictions
            predictions, probabilities, true_labels, valid_ids = generate_predictions_for_split(
                model, slide_ids, feature_dir, device, label_map
            )
            
            if len(predictions) == 0:
                print(f"      Warning: No valid predictions for {split_name}")
                continue
            
            # Compute metrics
            metrics = compute_metrics(true_labels, predictions, probabilities, n_classes)
            
            # Store results
            result = {
                'fold': fold,
                'split': split_name,
                'n_samples': len(valid_ids),
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity']
            }
            all_results.append(result)
            
            print(f"      Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
        
        # Clean up model
        del model
        torch.cuda.empty_cache()
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = os.path.join(model_dir, 'final_metrics.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\n  Results saved to: {output_path}")
        
        # Print summary
        print(f"\n  Summary:")
        print(results_df.to_string(index=False))
    else:
        print(f"\n  Warning: No results to save for {model_name}")


def main():
    args = parse_args()
    
    print("="*80)
    print("Slide-level MLP Model Metrics Calculation")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Number of models: {len(args.model_dirs)}")
    print(f"Number of folds: {args.num_folds}")
    print("="*80)
    
    # Validate input arguments
    if len(args.model_dirs) != len(args.feature_dirs):
        raise RuntimeError("Number of model directories must match number of feature directories")
    
    # Device handling
    requested_device = args.device
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(requested_device)
    
    print(f"Using device: {device}")
    
    # Load and remap labels
    print("\nLoading labels...")
    label_map = load_and_remap_labels(args.labels_csv, args.task)
    print(f"Loaded {len(label_map)} labeled slides")
    
    # Validate splits directory
    if not os.path.exists(args.splits_dir):
        raise RuntimeError(f"Splits directory not found: {args.splits_dir}")
    
    # Process each model
    for model_dir, feature_dir in zip(args.model_dirs, args.feature_dirs):
        if not os.path.exists(model_dir):
            print(f"\nWarning: Model directory not found, skipping: {model_dir}")
            continue
        
        if not os.path.exists(feature_dir):
            print(f"\nWarning: Feature directory not found, skipping: {feature_dir}")
            continue
        
        try:
            process_model(
                model_dir, 
                feature_dir, 
                args.splits_dir, 
                label_map, 
                args.n_classes, 
                args.num_folds, 
                device
            )
        except Exception as e:
            print(f"\nError processing model {os.path.basename(model_dir)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == '__main__':
    main()

