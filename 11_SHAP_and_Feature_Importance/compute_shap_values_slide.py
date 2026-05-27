#!/usr/bin/env python3
"""
compute_shap_values_mlp.py

Compute SHAP attributions for slide‑level MLP models trained by train_slide_models.py
using Captum's GradientShap.

What this script does
---------------------
1) Finds the best-performing fold by **test AUC** in <mlp_dir>/summary.csv (falls back to accuracy if AUC is NaN).
2) Rebuilds the SlideMLP exactly as trained (depth, hidden size, dropout) and loads the checkpoint from
   <mlp_dir>/fold_<k>/model.pt.
3) Samples --n_background slide feature vectors from the **training split** of that fold as the baseline distribution.
4) Computes GradientShap attributions **for every slide** (.h5) discovered in --features_dir.
5) Saves per‑slide SHAP vectors to **<features_dir>/SHAP/<slide_id>.npy** (shape: (D,)).
6) Aggregates global mean absolute SHAP across all processed slides and saves to **<mlp_dir>/global_shap.h5**
   (dataset name: 'feature_importance') to mirror the behavior of your CLAM SHAP script.

Notes
-----
• Input features are assumed to be stored in each H5 under the dataset key 'features', shaped either (D,) or (1, D).
• Binary task: model outputs a single logit; attributions are taken w.r.t. that scalar (no explicit target needed).
• Baseline distribution is a stack of training feature vectors of shape (n_background, D).

Example
-------
python compute_shap_values_mlp.py \
  --mlp_dir /path/to/mlp_run \
  --features_dir /path/to/slide_features \
  --splits_dir /path/to/splits \
  --device cuda:0 \
  --n_background 50 \
  --n_samples 50
"""
import os
import re
import glob
import json
import random
import argparse
from typing import List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from captum.attr import GradientShap

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

# -----------------------------
# Utilities
# -----------------------------
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_slide_ids(features_dir: str) -> List[str]:
    slide_ids = []
    for fname in os.listdir(features_dir):
        if fname.lower().endswith('.h5'):
            slide_ids.append(os.path.splitext(fname)[0])
    slide_ids.sort()
    return slide_ids


def load_feature_vector(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, 'r') as f:
        if 'features' not in f:
            raise KeyError(f"Dataset 'features' not found in {h5_path}")
        x = f['features'][:]
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError(f"Unexpected feature shape after squeeze: {x.shape} in {h5_path}")
    return x.astype(np.float32, copy=False)


def select_best_fold(mlp_dir: str) -> int:
    summary_path = os.path.join(mlp_dir, 'summary.csv')
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"summary.csv not found in {mlp_dir}")
    df = pd.read_csv(summary_path)
    if 'split' not in df.columns or 'fold' not in df.columns:
        raise RuntimeError("summary.csv must contain 'split' and 'fold' columns")
    df_test = df[df['split'] == 'test'].copy()
    if df_test.empty:
        raise RuntimeError("No rows with split=='test' found in summary.csv")

    if 'auc' in df_test.columns and df_test['auc'].notna().any():
        best_idx = df_test['auc'].idxmax()
    else:
        # Fallback to accuracy if AUC missing/NaN
        if 'accuracy' not in df_test.columns:
            raise RuntimeError("Neither 'auc' nor 'accuracy' available to select best fold")
        best_idx = df_test['accuracy'].idxmax()

    best_fold = int(df_test.loc[best_idx, 'fold'])
    return best_fold


def read_split_ids(splits_dir: str, fold: int) -> dict:
    split_path = os.path.join(splits_dir, f'splits_{fold}.csv')
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    df = pd.read_csv(split_path)
    ids = {}
    for split in ['train', 'val', 'test']:
        if split not in df.columns:
            raise RuntimeError(f"Column '{split}' missing in {split_path}")
        # Convert to strings, drop NaNs
        ids[split] = df[split].dropna().astype(str).tolist()
    return ids


def load_checkpoint_and_rebuild(mlp_dir: str, fold: int, input_dim: int, device: torch.device) -> SlideMLP:
    fold_dir = os.path.join(mlp_dir, f'fold_{fold}')
    ckpt_path = os.path.join(fold_dir, 'model.pt')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
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

# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compute SHAP values for slide‑level MLP models (GradientShap)")
    p.add_argument('--mlp_dir', required=True, help='Directory containing fold_* and summary.csv from training')
    p.add_argument('--features_dir', required=True, help='Directory of slide .h5 files with dataset "features"')
    p.add_argument('--splits_dir', required=True, help='Directory with splits_<k>.csv files (to sample train baselines)')
    p.add_argument('--device', default='cuda:0', help="PyTorch device, e.g., 'cuda:0' or 'cpu'")
    p.add_argument('--n_background', type=int, default=50, help='Number of train slides to use as baselines (default: 50)')
    p.add_argument('--n_samples', type=int, default=50, help='GradientShap n_samples per example (default: 50)')
    p.add_argument('--seed', type=int, default=0, help='Random seed for baseline sampling')
    return p.parse_args()


def main():
    args = parse_args()

    # Device handling
    requested_device = args.device
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(requested_device)

    # Reproducibility
    set_seeds(args.seed)

    # Slide discovery & input dimensionality
    slide_ids = list_slide_ids(args.features_dir)
    if not slide_ids:
        raise RuntimeError(f"No .h5 files found in features_dir: {args.features_dir}")

    # Determine input dim from the first slide
    first_feat = load_feature_vector(os.path.join(args.features_dir, f"{slide_ids[0]}.h5"))
    input_dim = int(first_feat.shape[0])
    print(f"[INFO] Discovered {len(slide_ids)} slides | feature dimension D={input_dim}")

    # Best fold by AUC (or accuracy fallback)
    best_fold = select_best_fold(args.mlp_dir)
    print(f"[INFO] Selected best fold: {best_fold}")

    # Read split IDs and sample train baselines
    split_ids = read_split_ids(args.splits_dir, best_fold)
    train_ids = split_ids['train']
    if not train_ids:
        raise RuntimeError(f"Train split is empty for fold {best_fold}")

    k = min(args.n_background, len(train_ids))
    if k < args.n_background:
        print(f"[WARN] Requested n_background={args.n_background} but train has only {len(train_ids)} slides; using {k}.")

    rng = random.Random(args.seed)
    background_ids = rng.sample(train_ids, k)
    print(f"[INFO] Using {k} training slides as baselines for GradientShap")

    # Load model
    model = load_checkpoint_and_rebuild(args.mlp_dir, best_fold, input_dim, device)

    # Build baseline tensor (k, D)
    baselines = []
    for sid in tqdm(background_ids, desc='Loading baselines', unit='slide'):
        path = os.path.join(args.features_dir, f"{sid}.h5")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Baseline slide features not found: {path}")
        vec = load_feature_vector(path)  # (D,)
        if vec.shape[0] != input_dim:
            raise ValueError(f"Baseline feature dim mismatch for {sid}: got {vec.shape[0]}, expected {input_dim}")
        baselines.append(torch.from_numpy(vec))
    baselines = torch.stack(baselines, dim=0).to(device=device, dtype=torch.float32)  # (k, D)

    # Captum GradientShap on the model directly (scalar output per example)
    explainer = GradientShap(model)

    # Output directories
    shap_dir = os.path.join(args.features_dir, 'SHAP')
    os.makedirs(shap_dir, exist_ok=True)

    # Accumulate global absolute SHAP for parity with CLAM script
    global_abs_list = []

    # Compute SHAP per slide
    for sid in tqdm(slide_ids, desc='Computing SHAP', unit='slide'):
        feat_path = os.path.join(args.features_dir, f"{sid}.h5")
        try:
            x = load_feature_vector(feat_path)  # (D,)
        except Exception as e:
            print(f"[ERROR] Skipping {sid}: {e}")
            continue
        if x.shape[0] != input_dim:
            print(f"[ERROR] Skipping {sid}: feature dim mismatch {x.shape[0]} != {input_dim}")
            continue

        x_tensor = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, D)
        # GradientShap returns same shape as input
        attr = explainer.attribute(
            x_tensor,
            baselines=baselines,
            n_samples=int(args.n_samples),
            # No target needed; model outputs a scalar logit per example
        )
        shap_vec = attr.squeeze(0).detach().cpu().numpy()  # (D,)

        # Save per‑slide SHAP vector
        out_path = os.path.join(shap_dir, f"{sid}.npy")
        np.save(out_path, shap_vec)

        global_abs_list.append(np.abs(shap_vec))

    # Save global mean absolute SHAP (parity with CLAM script behavior)
    if global_abs_list:
        global_arr = np.stack(global_abs_list, axis=0)  # (N, D)
        global_mean = np.mean(global_arr, axis=0)       # (D,)
        gpath = os.path.join(args.mlp_dir, 'global_shap.h5')
        with h5py.File(gpath, 'w') as hf:
            hf.create_dataset('feature_importance', data=global_mean)
        print(f"[INFO] Wrote global feature importance: {gpath}")

    print("[DONE] SHAP computation complete.")


if __name__ == '__main__':
    main()
