#!/usr/bin/env python3
"""
create_combined_fm_embeddings_slide.py

Slide-level version of the feature combiner/pruner.

Context: You extracted **one** feature vector per slide **per model** (e.g., via TRIDENT),
stored as an HDF5 file per slide with a dataset named `features` of shape `(D,)`.
This script combines multiple model-specific feature sets into one HDF5 per slide,
with two modes:

  • **Concatenate only** (default): simply concatenates all model features in the
    provided order and writes a single output to `--output_dir`.

  • **Two-stage pruning** (`--prune`):
      (1) Stage 1: per-model correlation pruning across slides.
      (2) Stage 2: merged correlation pruning across the concatenated survivors.
    For pruning, you must provide labels and a binary task mapping via `--labels_csv` and `--task`.
    Outputs are written to subdirectories per threshold (e.g., `thr_0p6`).

The script mirrors the CLI shape of the tile-level version where possible.
It also writes a `feature_sources.npy` mapping final feature indices to model names,
and, when pruning, a `pruning_report.csv` with counts removed at each stage per model.

Everything **except** the `features` dataset is copied from input H5s to the outputs.

---
Example (concatenate only):

  python create_combined_fm_embeddings_slide.py \
    --feature_dirs /path/to/UNI /path/to/CONCH /path/to/GIGAPATH \
    --output_dir /path/to/combined_slide_features

Example (prune with multiple thresholds):

  python create_combined_fm_embeddings_slide.py \
    --feature_dirs /path/to/UNI /path/to/CONCH /path/to/GIGAPATH \
    --output_dir /path/to/combined_slide_features \
    --prune --labels_csv /path/to/labels.csv --task task_kidney_grade \
    --corr_thresholds 0.4,0.5,0.6

Notes:
  • Assumes a **binary** task (mapping integer labels to {0,1}).
  • Keeps the original GPU-based correlation computation (CUDA required) like the prior code.
  • Slide set defaults to the intersection of slide IDs present in **all** feature dirs.
"""

import argparse
import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# -------------------------------
# Core selection routine (kept close to the prior behavior)
# -------------------------------

def pick_best_uncorrelated_features(data, classes, idx_pool, num_features=int(1e9), correlation_factor=0.6):
    """Greedy uncorrelated feature selection by two-sample t-test p-values.

    Parameters
    ----------
    data : np.ndarray, shape (N, D)
        Feature matrix across N slides and D features (for a single model or merged models).
    classes : np.ndarray, shape (N,)
        Class vector with values in {-1, +1} for binary task.
    idx_pool : Sequence[int]
        Feature indices to consider (0..D-1 for per-model stage; or 0..total-1 for merged stage).
    num_features : int
        Max number of features to select (defaults to very large -> select until correlation gate stops).
    correlation_factor : float
        Absolute correlation threshold for pruning.

    Returns
    -------
    selected_sorted : list[int]
        Sorted list of retained feature indices (relative to the original columns of `data`).
    sel_pvals : list[float]
        p-values for the selected features (aligned to `selected_sorted`).
    """
    import numpy as _np
    import torch as _torch
    from scipy import stats as _stats

    idx_arr = _np.array(idx_pool, dtype=int)
    mask_pos = classes == 1
    mask_neg = classes == -1

    # Two-sample t-test per feature within idx_pool
    p_vals = _np.array([_stats.ttest_ind(data[mask_pos, i], data[mask_neg, i], equal_var=False)[1]
                        for i in idx_arr])
    valid = _np.isfinite(p_vals)
    idx_good = idx_arr[valid]
    p_vals = p_vals[valid]

    # Correlation matrix (GPU, mirrors older behavior)
    X = _torch.from_numpy(data[:, idx_good].astype(_np.float32)).to('cuda')
    X = X - X.mean(dim=0, keepdim=True)
    # Compute covariance and convert to correlation
    cov = (X.t() @ X) / max(1, X.shape[0])
    std = _torch.sqrt(torch.clamp(torch.diag(cov), min=1e-12))
    corr = cov / (std[:, None] * std[None, :])
    corr = corr.abs().detach().cpu().numpy()

    selected = []
    alive = _np.ones(len(idx_good), dtype=bool)
    while len(selected) < num_features and alive.any():
        rem = _np.where(alive)[0]
        p_rem = p_vals[alive]
        best = rem[_np.argmin(p_rem)]
        selected.append(int(idx_good[best]))
        bad = _np.where(corr[best, :] > correlation_factor)[0]
        alive[bad] = False

    # Map to p-values in selection order
    idx_map = {int(idx_good[i]): i for i in range(len(idx_good))}
    sel_pvals = [float(p_vals[idx_map[j]]) for j in selected]
    return sorted(selected), sel_pvals


# -------------------------------
# Argument parsing
# -------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine and optionally prune slide-level feature .h5 sets into unified files"
    )
    parser.add_argument("--feature_dirs", nargs='+', required=True,
                        help="Directories containing slide-level .h5 files per model (one vector per slide)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory. If not pruning, results are written here. If pruning, subdirs per threshold are created.")
    parser.add_argument("--slides", nargs='*', default=None,
                        help="Optional explicit slide IDs (without .h5). Defaults to intersection across all models.")
    parser.add_argument("--prune", action='store_true',
                        help="Enable two-stage correlation-based feature pruning")
    parser.add_argument("--labels_csv", type=str,
                        help="CSV with columns: slide_id, label (required if --prune)")
    parser.add_argument("--task", type=str,
                        choices=['task_kidney_grade', 'task_prostate_grade', 'task_rectal_stage'],
                        help="Defines label-to-class mapping (required if --prune)")
    parser.add_argument("--corr_thresholds", type=str, default="0.6",
                        help="Comma-separated list of correlation cutoffs (e.g., 0.4,0.5,0.6). Used only if --prune.")
    return parser.parse_args()


# -------------------------------
# Helpers
# -------------------------------

def discover_common_slides(feature_dirs):
    sets = []
    for d in feature_dirs:
        if not os.path.isdir(d):
            print(f"Error: Feature dir not found: {d}", file=sys.stderr)
            sys.exit(1)
        files = [os.path.splitext(fn)[0] for fn in os.listdir(d) if fn.endswith('.h5')]
        sets.append(set(files))
    return sorted(set.intersection(*sets)) if sets else []


def read_feature_dim(h5_path):
    with h5py.File(h5_path, 'r') as f:
        feats = f['features'][:]
        if feats.ndim != 1:
            raise ValueError(f"Expected 'features' shape (D,), got {feats.shape} in {h5_path}")
        return int(feats.shape[0])


def load_all_model_matrices(slide_ids, model_dir):
    """Load per-slide features for one model as an (N, D_model) matrix.

    Parameters
    ----------
    slide_ids : list[str]
    model_dir : str

    Returns
    -------
    data : np.ndarray, shape (N, D_model)
        Features stacked in the order of slide_ids.
    others_first : dict
        Datasets (except 'features') from the FIRST slide file, to copy into outputs later.
    attrs_first : dict
        Attributes from the FIRST slide file, to copy into outputs later.
    """
    data_list = []
    others_first = {}
    attrs_first = {}

    for i, sid in enumerate(slide_ids):
        fp = os.path.join(model_dir, f"{sid}.h5")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing slide H5 for id {sid} in {model_dir}")
        with h5py.File(fp, 'r') as f:
            feats = f['features'][:]
            if feats.ndim != 1:
                raise ValueError(f"Expected 'features' shape (D,), got {feats.shape} in {fp}")
            data_list.append(feats.astype(np.float32))
            if i == 0:
                attrs_first = dict(f.attrs)
                # copy every dataset EXCEPT 'features'
                for k in f.keys():
                    if k != 'features':
                        others_first[k] = f[k][:]

    data = np.stack(data_list, axis=0)  # (N, D)
    return data, others_first, attrs_first


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# -------------------------------
# Main
# -------------------------------

def main():
    args = parse_args()

    # Validate/calc thresholds
    if args.prune:
        try:
            thresholds = [float(t) for t in args.corr_thresholds.split(',')]
        except ValueError:
            print("Error: --corr_thresholds must be comma-separated floats", file=sys.stderr)
            sys.exit(1)
    else:
        thresholds = [None]  # sentinel: single pass, no subdir

    # Determine slide ids
    if args.slides:
        slide_ids = list(args.slides)
    else:
        slide_ids = discover_common_slides(args.feature_dirs)
    if not slide_ids:
        print("Error: No common slides found across feature_dirs.", file=sys.stderr)
        sys.exit(1)

    # Labels / classes (for pruning)
    lbl_df = None
    if args.prune:
        if not args.labels_csv or not args.task:
            print("Error: --labels_csv and --task are required for pruning", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(args.labels_csv):
            print(f"Error: labels_csv not found: {args.labels_csv}", file=sys.stderr)
            sys.exit(1)
        lbl_df = pd.read_csv(args.labels_csv, dtype={'slide_id': str, 'label': int})
        if not {'slide_id', 'label'}.issubset(lbl_df.columns):
            print("Error: labels_csv must contain 'slide_id' and 'label' columns", file=sys.stderr)
            sys.exit(1)
        # Task mappings (binary)
        if args.task == 'task_kidney_grade':
            mapping = {0: 0, 1: 0, 2: 1, 3: 1}
        elif args.task == 'task_prostate_grade':
            mapping = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1}
        else:  # task_rectal_stage
            mapping = {1: 0, 2: 0, 3: 1, 4: 1}
        lbl_df['binary'] = lbl_df['label'].map(mapping)
        if lbl_df['binary'].isnull().any():
            print("Error: Some labels are not covered by the mapping.", file=sys.stderr)
            sys.exit(1)
        lbl_df['class'] = lbl_df['binary'].apply(lambda x: -1 if x == 0 else 1)
        labeled_ids = set(lbl_df['slide_id'])
        slide_ids = [s for s in slide_ids if s in labeled_ids]
        if not slide_ids:
            print("Error: No slides remain after intersecting with labels_csv.", file=sys.stderr)
            sys.exit(1)

    # Discover model dims and preload per-model matrices
    model_names = [os.path.basename(d.rstrip('/')) for d in args.feature_dirs]
    model_mats = []          # list of (N, D_m)
    model_others = []        # datasets from first file per model (dict)
    model_attrs = []         # attrs from first file per model (dict)
    original_dims = []

    # Validate CUDA availability if pruning (since correlation is on GPU, matching old behavior)
    if args.prune and not torch.cuda.is_available():
        print("Error: --prune requires CUDA (no GPU detected). This mirrors the prior script's behavior.", file=sys.stderr)
        sys.exit(1)

    for d in args.feature_dirs:
        # Read one file to get dimension
        sample_path = os.path.join(d, f"{slide_ids[0]}.h5")
        dim = read_feature_dim(sample_path)
        original_dims.append(dim)
        # Load matrix across slides
        mat, other, attr = load_all_model_matrices(slide_ids, d)
        model_mats.append(mat)
        model_others.append(other)
        model_attrs.append(attr)

    # If not pruning: single pass writing directly into output_dir
    # If pruning: iterate thresholds and write into subdirectories
    for corr in thresholds:
        if args.prune:
            subdir = f"thr_{str(corr).replace('.', 'p')}"
            out_dir = os.path.join(args.output_dir, subdir)
            ensure_dir(out_dir)
            print(f"Running pruning with correlation threshold = {corr} -> output in {out_dir}")
        else:
            out_dir = args.output_dir
            ensure_dir(out_dir)
            print(f"Concatenate only -> output in {out_dir}")

        # Stage 1: per-model pruning (or keep all)
        per_model_keep = []
        removed_stage1 = []
        classes_vec = None

        if args.prune:
            # Build N-vector of classes in the order of slide_ids
            # (We already filtered slide_ids to those present in lbl_df)
            classes_vec = lbl_df.set_index('slide_id').loc[slide_ids, 'class'].to_numpy()
            for name, mat, dim in zip(model_names, model_mats, original_dims):
                keep_indices, _ = pick_best_uncorrelated_features(
                    data=mat, classes=classes_vec, idx_pool=list(range(dim)), correlation_factor=corr
                )
                per_model_keep.append(keep_indices)
                removed_stage1.append(dim - len(keep_indices))
                print(f"Stage1 {name}: kept {len(keep_indices)} / {dim}")
        else:
            for dim in original_dims:
                per_model_keep.append(list(range(dim)))
                removed_stage1.append(0)

        # Stage 2: merged pruning (or pass-through)
        if args.prune:
            # Concatenate per-model survivors -> (N, sum_k |keep_k|)
            merged = np.concatenate([mat[:, keep]
                                     for mat, keep in zip(model_mats, per_model_keep)], axis=1)
            # Build source map for merged columns before stage 2 with original indices
            pre2_sources = []
            for name, keep in zip(model_names, per_model_keep):
                for orig_idx in keep:
                    pre2_sources.append((name, orig_idx))

            total_dim = merged.shape[1]
            keep2, _ = pick_best_uncorrelated_features(
                data=merged, classes=classes_vec, idx_pool=list(range(total_dim)), correlation_factor=corr
            )
            # Count Stage 2 removals per model
            removed_stage2 = []
            for name in model_names:
                init = sum(1 for s, _ in pre2_sources if s == name)
                surv = sum(1 for j in keep2 if pre2_sources[j][0] == name)
                removed_stage2.append(init - surv)
                print(f"Stage2 {name}: kept {surv} / {init}")

            # Save pruning report
            report = pd.DataFrame({
                'model': model_names,
                'removed_stage1': removed_stage1,
                'removed_stage2': removed_stage2,
            })
            report.to_csv(os.path.join(out_dir, 'pruning_report.csv'), index=False)

            # Final feature_sources after Stage 2 with model and original feature index
            final_sources = [pre2_sources[j] for j in keep2]
            feat_src_array = np.array(final_sources, dtype=[('model', 'U50'), ('feature_index', 'i4')])
            np.save(os.path.join(out_dir, 'feature_sources.npy'), feat_src_array)

            # Prepare per-slide final vectors for writing
            final_per_slide = merged[:, keep2]  # (N, D_final)
        else:
            # No pruning: direct concatenation
            final_sources = []
            for name, keep in zip(model_names, per_model_keep):
                for orig_idx in keep:
                    final_sources.append((name, orig_idx))
            feat_src_array = np.array(final_sources, dtype=[('model', 'U50'), ('feature_index', 'i4')])
            np.save(os.path.join(out_dir, 'feature_sources.npy'), feat_src_array)
            final_per_slide = np.concatenate([mat[:, keep]
                                              for mat, keep in zip(model_mats, per_model_keep)], axis=1)

        # -------------------------------
        # Write outputs per slide
        # -------------------------------
        # For copying, we take "others" and attrs of the FIRST model by convention.
        # (You can change this policy later if needed.)
        first_others = model_others[0]
        first_attrs = model_attrs[0]

        for i, sid in tqdm(list(enumerate(slide_ids)), desc=(f"Writing slides" + (f" @{corr}" if args.prune else ""))):
            out_path = os.path.join(out_dir, f"{sid}.h5")
            with h5py.File(out_path, 'w') as fo:
                # Copy attrs
                for k, v in first_attrs.items():
                    fo.attrs[k] = v
                # Copy datasets except 'features'
                for k, v in first_others.items():
                    fo.create_dataset(k, data=v)
                # Write final features as 1D vector (D_final,)
                fo.create_dataset('features', data=final_per_slide[i].astype(np.float32))

        if not args.prune:
            print("Completed concatenate-only run.")
        else:
            print(f"Completed pruning run for threshold {corr}.")


if __name__ == '__main__':
    main()

