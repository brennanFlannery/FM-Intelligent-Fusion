#!/usr/bin/env python3
"""
combine_clam_features.py

Combine multiple foundation-model feature sets into one HDF5 per slide,
optionally applying two-stage correlation-based feature pruning:
  1) prune each model's features individually (using top-100 attended tiles per slide)
  2) concatenate survivors and prune again
Additionally, save a global .npy mapping each final feature index to the originating model name,
and output a CSV report of features removed during each pruning stage per model.
Supports running multiple correlation thresholds in one invocation.

Example:
  python create_combined_fm_embeddings.py \
    --feature_dirs /path/to/UNI /path/to/CONCH . \
    --output_dir /path/to/combined_features \
    --prune --labels_csv labels.csv --task task_kidney_grade \
    --corr_thresholds 0.4,0.5,0.6
"""
import argparse
import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


def pick_best_uncorrelated_features(data, classes, idx_pool,
                                   num_features=int(1e9), correlation_factor=0.6):
    import numpy as _np
    import torch as _torch
    from scipy import stats as _stats

    idx_arr = _np.array(idx_pool, dtype=int)
    mask_pos = classes == 1
    mask_neg = classes == -1
    p_vals = _np.array([_stats.ttest_ind(data[mask_pos, i], data[mask_neg, i])[1]
                        for i in idx_arr])
    valid = _np.isfinite(p_vals)
    idx_good = idx_arr[valid]
    p_vals = p_vals[valid]

    X = _torch.from_numpy(data[:, idx_good].astype(_np.float32)).to('cuda')
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.t() @ X) / X.shape[0]
    std = _torch.sqrt(torch.diag(cov))
    corr = cov / (std[:, None] * std[None, :])
    corr = corr.abs().cpu().numpy()

    selected = []
    alive = _np.ones(len(idx_good), dtype=bool)
    while len(selected) < num_features and alive.any():
        rem = _np.where(alive)[0]
        p_rem = p_vals[alive]
        best = rem[_np.argmin(p_rem)]
        selected.append(int(idx_good[best]))
        bad = _np.where(corr[best, :] > correlation_factor)[0]
        alive[bad] = False

    sel_pvals = [float(p_vals[_np.where(idx_good == feat)[0][0]]) for feat in selected]
    return sorted(selected), sel_pvals


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine and optionally prune feature .h5 sets into unified files"
    )
    parser.add_argument("--feature_dirs", nargs='+', required=True,
                        help="Directories containing slide-level .h5 files per model")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory to contain subfolders per threshold")
    parser.add_argument("--slides", nargs='*', default=None,
                        help="Optional slide IDs (defaults to intersection across models)")
    parser.add_argument("--prune", action='store_true',
                        help="Enable two-stage correlation-based pruning")
    parser.add_argument("--labels_csv", type=str,
                        help="CSV of slide_id and label (required if --prune)")
    parser.add_argument("--task", type=str,
                        choices=['task_kidney_grade','task_prostate_grade','task_rectal_stage'],
                        help="Defines label-to-class mapping (required if --prune)")
    parser.add_argument("--corr_thresholds", type=str, default="0.6",
                        help="Comma-separated list of correlation cutoffs (e.g. 0.4,0.5,0.6)")
    return parser.parse_args()


def main():
    args = parse_args()
    # parse thresholds
    try:
        thresholds = [float(t) for t in args.corr_thresholds.split(',')]
    except ValueError:
        print("Error: --corr_thresholds must be comma-separated floats", file=sys.stderr)
        sys.exit(1)

    # discover common slide IDs
    if args.slides:
        slide_ids = args.slides
    else:
        sets = []
        for d in args.feature_dirs:
            if not os.path.isdir(d):
                print(f"Error: Feature dir not found: {d}", file=sys.stderr)
                sys.exit(1)
            files = [os.path.splitext(fn)[0] for fn in os.listdir(d) if fn.endswith('.h5')]
            sets.append(set(files))
        slide_ids = sorted(set.intersection(*sets)) if sets else []
    if not slide_ids:
        print("Error: No common slides found across feature_dirs.", file=sys.stderr)
        sys.exit(1)

    # prepare labels mapping
    lbl_df = None
    if args.prune:
        if not args.labels_csv or not args.task:
            print("Error: --labels_csv and --task are required for pruning", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(args.labels_csv):
            print(f"Error: labels_csv not found: {args.labels_csv}", file=sys.stderr)
            sys.exit(1)
        lbl_df = pd.read_csv(args.labels_csv, dtype={'slide_id':str,'label':int})
        if not {'slide_id','label'}.issubset(lbl_df.columns):
            print("Error: labels_csv must contain 'slide_id' and 'label' columns", file=sys.stderr)
            sys.exit(1)
        # define mapping
        if args.task == 'task_kidney_grade':
            mapping = {0:0,1:0,2:1,3:1}
        elif args.task == 'task_prostate_grade':
            mapping = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:1,9:1,10:1}
        else:
            mapping = {1:0,2:0,3:1,4:1}
        lbl_df['binary'] = lbl_df['label'].map(mapping)
        if lbl_df['binary'].isnull().any():
            print("Error: Some labels not in mapping", file=sys.stderr)
            sys.exit(1)
        lbl_df['class'] = lbl_df['binary'].apply(lambda x: -1 if x==0 else 1)
        slide_ids = [s for s in slide_ids if s in set(lbl_df['slide_id'])]
        if not slide_ids:
            print("Error: No slides match labels_csv after mapping", file=sys.stderr)
            sys.exit(1)

    # discover original dims per model
    original_dims = []
    for d in args.feature_dirs:
        with h5py.File(os.path.join(d, f"{slide_ids[0]}.h5"),'r') as f0:
            original_dims.append(f0['features'].shape[1])

    # loop over thresholds
    for corr in thresholds:
        subdir = f"thr_{str(corr).replace('.', 'p')}"
        out_dir = os.path.join(args.output_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Running pruning with correlation threshold = {corr} -> output in {out_dir}")

        # stats for report
        removed1 = []
        removed2 = []
        keep_lists = []
        model_names = []

        # Stage 1 pruning per model
        if args.prune:
            for d, dim in zip(args.feature_dirs, original_dims):
                model = os.path.basename(d)
                feats, cls_all = [], []
                for slide in slide_ids:
                    af = os.path.join(d,'attentions',f"{slide}.h5")
                    with h5py.File(af,'r') as fa:
                        attn = fa['attention'][:]
                    topk = np.argsort(attn)[::-1][:min(100,len(attn))]
                    with h5py.File(os.path.join(d,f"{slide}.h5"),'r') as f:
                        arr = f['features'][:][topk]
                    feats.append(arr)
                    c = lbl_df.loc[lbl_df['slide_id']==slide,'class'].iloc[0]
                    cls_all.append(np.full(arr.shape[0], c))
                data = np.vstack(feats)
                classes = np.concatenate(cls_all)
                keep, _ = pick_best_uncorrelated_features(
                    data, classes, list(range(dim)), correlation_factor=corr
                )
                keep_lists.append(keep)
                model_names.append(model)
                removed1.append(dim - len(keep))
                print(f"Stage1 {model}: removed {removed1[-1]} of {dim}")

            # Stage 2 merged pruning
            # prepare full source list
            full_src = []
            for model, keep in zip(model_names, keep_lists):
                full_src.extend([model]*len(keep))
            merged_feats, merged_cls = [], []
            for slide in slide_ids:
                parts = []
                for d, keep in zip(args.feature_dirs, keep_lists):
                    af = os.path.join(d,'attentions',f"{slide}.h5")
                    with h5py.File(af,'r') as fa:
                        attn = fa['attention'][:]
                    topk = np.argsort(attn)[::-1][:min(100,len(attn))]
                    with h5py.File(os.path.join(d,f"{slide}.h5"),'r') as f:
                        parts.append(f['features'][:][topk][:, keep])
                merged = np.concatenate(parts, axis=1)
                merged_feats.append(merged)
                c = lbl_df.loc[lbl_df['slide_id']==slide,'class'].iloc[0]
                merged_cls.append(np.full(merged.shape[0], c))
            all_data = np.vstack(merged_feats)
            all_cls = np.concatenate(merged_cls)
            total_dim = sum(len(k) for k in keep_lists)
            keep2, _ = pick_best_uncorrelated_features(
                all_data, all_cls, list(range(total_dim)), correlation_factor=corr
            )
            removed2 = []
            for model in model_names:
                init = sum(1 for m in full_src if m==model)
                surv = sum(1 for idx in keep2 if full_src[idx]==model)
                removed2.append(init - surv)
                print(f"Stage2 {model}: removed {removed2[-1]} of {init}")

            # write pruning report
            report = pd.DataFrame({
                'model': model_names,
                'removed_stage1': removed1,
                'removed_stage2': removed2
            })
            report.to_csv(os.path.join(out_dir,'pruning_report.csv'), index=False)

        else:
            # no pruning -> keep all dims
            keep_lists = [list(range(dim)) for dim in original_dims]
            model_names = [os.path.basename(d) for d in args.feature_dirs]
            for model, dim in zip(model_names, original_dims):
                print(f"No prune: {model} keeps all {dim}")
            keep2 = None

        # save feature_sources with model and original feature index
        feat_src = []
        for model, keep in zip(model_names, keep_lists):
            for orig_idx in keep:
                feat_src.append((model, orig_idx))
        if args.prune:
            feat_src = [feat_src[i] for i in keep2]
        # Create structured array with 'model' and 'feature_index' fields
        feat_src_array = np.array(feat_src, dtype=[('model', 'U50'), ('feature_index', 'i4')])
        np.save(os.path.join(out_dir,'feature_sources.npy'), feat_src_array)

        # write slides
        for slide in tqdm(slide_ids, desc=f"Writing slides @{corr}"):
            coords = None; parts=[]; attrs={}; others={}
            for d, keep in zip(args.feature_dirs, keep_lists):
                fp = os.path.join(d,f"{slide}.h5")
                with h5py.File(fp,'r') as f:
                    if coords is None:
                        coords = f['coords'][:]
                        attrs = dict(f.attrs)
                        for k in f.keys():
                            if k not in ('coords','features'): others[k]=f[k][:]
                    parts.append(f['features'][:, keep])
            merged = np.concatenate(parts, axis=1)
            if args.prune:
                merged = merged[:, keep2]
            outf = os.path.join(out_dir,f"{slide}.h5")
            with h5py.File(outf,'w') as fo:
                for k,v in attrs.items(): fo.attrs[k]=v
                for k,v in others.items(): fo.create_dataset(k,data=v)
                fo.create_dataset('coords',data=coords)
                fo.create_dataset('features',data=merged)
        print(f"Completed threshold {corr}")

if __name__=='__main__':
    main()
