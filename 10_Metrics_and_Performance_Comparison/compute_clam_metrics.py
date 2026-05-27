#!/usr/bin/env python3
"""
compute_clam_metrics.py

Compute classification metrics (accuracy, AUC, sensitivity, specificity, F1)
for each CLAM checkpoint and its corresponding data split, with optional label remapping by task.

For each checkpoint file `s_{n}_checkpoint.pt` in --model_dir:
  - Load split_{n}.csv to get slide IDs for train, val, test
  - Run the CLAM model on each slide's features to obtain bag-level predictions and probabilities
  - Compare against true labels from --labels_csv (remapped by --task if specified)
  - Compute metrics on train, val, and test subsets

Writes a CSV `final_metric_summary.csv` in --model_dir with columns:
  fold, split, accuracy, auc, sensitivity, specificity, f1_score

Example:
  python compute_clam_metrics.py \
    --model_dir /path/to/CLAM_musk \
    --feature_dir /path/to/features_musk \
    --labels_csv /path/to/labels.csv \
    --task task_kidney_grade \
    --model_type clam_sb \
    --model_size small \
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
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix
)
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_subclam import SubCLAM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute metrics for each CLAM checkpoint and split with label mappings"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Directory containing CLAM checkpoints and split CSVs"
    )
    parser.add_argument(
        "--feature_dir", type=str, required=True,
        help="Directory containing slide-level .h5 feature files"
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
        help="Optional label remapping to binary task"
    )
    parser.add_argument(
        "--model_type", choices=['clam_sb', 'clam_mb', 'subclam'], default='clam_sb',
        help="Which CLAM/SubCLAM architecture to load"
    )
    parser.add_argument(
        "--model_size", choices=['small', 'big'], default='small',
        help="Model size for CLAM (small or big)"
    )
    parser.add_argument(
        "--n_classes", type=int, required=True,
        help="Number of output classes in model (after remapping)"
    )
    parser.add_argument(
        "--cluster_cache_dir", type=str, default=None,
        help="Directory containing {slide_id}_clusters.h5 files (required when model_type=subclam)"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=5,
        help="Number of tissue clusters for SubCLAM (must match training)"
    )
    parser.add_argument(
        "--device", default='cuda:0',
        help="PyTorch device identifier"
    )
    return parser.parse_args()


def load_model_for_checkpoint(model_dir, fold, model_type, model_size,
                              embed_dim, n_classes, device, n_clusters=5):
    ckpt_file = os.path.join(model_dir, f's_{fold}_checkpoint.pt')
    if model_type == 'subclam':
        model = SubCLAM(
            n_clusters=n_clusters,
            gate=True,
            size_arg=model_size,
            dropout=0.25,
            n_classes=n_classes,
            embed_dim=embed_dim,
        )
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt, strict=False)
    else:
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
        ckpt = torch.load(ckpt_file, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def slide_prediction(model, feat_path, device):
    with h5py.File(feat_path,'r') as f:
        feats = f['features'][:]
    feats_t = torch.from_numpy(feats).to(device).float()
    lbl = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        logits, prob, pred, _, _ = model(feats_t, lbl, instance_eval=False)
    return prob.cpu().numpy().ravel(), int(pred.item())


def slide_prediction_subclam(model, feat_path, cluster_path, device):
    with h5py.File(feat_path, 'r') as f:
        feats = f['features'][:]
    with h5py.File(cluster_path, 'r') as f:
        cluster_ids = f['cluster_ids'][:]
    feats_t = torch.from_numpy(feats).to(device).float()
    cluster_ids_t = torch.from_numpy(cluster_ids).to(device).long()
    with torch.no_grad():
        logits, y_prob, y_hat, _, _ = model(feats_t, cluster_ids_t)
    return y_prob.cpu().numpy().ravel(), int(y_hat.item())


def compute_specificity(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    specs = []
    total = cm.sum()
    for i in range(n_classes):
        tp = cm[i,i]
        fn = cm[i,:].sum() - tp
        fp = cm[:,i].sum() - tp
        tn = total - tp - fp - fn
        specs.append(tn/(tn+fp) if (tn+fp)>0 else 0.)
    return float(np.mean(specs))


def main():
    args = parse_args()

    if args.model_type == 'subclam' and args.cluster_cache_dir is None:
        raise RuntimeError("--cluster_cache_dir is required when --model_type is subclam")

    # load true labels
    lbl_df = pd.read_csv(args.labels_csv, dtype={'slide_id':str,'label':int})
    # remap labels if needed
    if args.task == 'task_kidney_grade':
        lbl_df['label'] = lbl_df['label'].map({0:0,1:0,2:1,3:1})
    elif args.task == 'task_prostate_grade':
        lbl_df['label'] = lbl_df['label'].map({0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:1,9:1,10:1})
    elif args.task == 'task_rectal_stage':
        lbl_df['label'] = lbl_df['label'].map({1:0,2:0,3:1,4:1})
    # else 'none'
    label_map = dict(zip(lbl_df['slide_id'], lbl_df['label']))

    # gather slides
    slides = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(args.feature_dir)
        if f.endswith('.h5') and os.path.splitext(f)[0] in label_map
    ])
    if not slides:
        raise RuntimeError("No labeled .h5 slides found in feature_dir")

    # infer embed_dim
    sample = slides[0]
    with h5py.File(os.path.join(args.feature_dir, f"{sample}.h5"),'r') as f:
        embed_dim = f['features'].shape[1]

    # find checkpoints
    ckpts = sorted(glob.glob(os.path.join(args.model_dir, 's_*_checkpoint.pt')))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {args.model_dir}")

    records = []
    for ckpt_path in ckpts:
        fold = int(re.match(r'.*s_(\d+)_checkpoint\.pt', os.path.basename(ckpt_path)).group(1))
        model = load_model_for_checkpoint(
            args.model_dir, fold,
            args.model_type, args.model_size,
            embed_dim, args.n_classes,
            args.device,
            n_clusters=args.n_clusters,
        )
        df_split = pd.read_csv(os.path.join(args.model_dir, f'splits_{fold}.csv'))

        for split in ['train', 'val', 'test']:
            ids = [sid for sid in df_split[split].dropna().tolist() if sid in slides]
            y_true, y_pred, y_prob = [], [], []
            for sid in tqdm(ids, desc=f"Fold{fold}-{split}"):
                if args.model_type == 'subclam':
                    cluster_path = os.path.join(args.cluster_cache_dir, f"{sid}_clusters.h5")
                    prob, pred = slide_prediction_subclam(
                        model,
                        os.path.join(args.feature_dir, f"{sid}.h5"),
                        cluster_path,
                        args.device,
                    )
                else:
                    prob, pred = slide_prediction(
                        model,
                        os.path.join(args.feature_dir, f"{sid}.h5"),
                        args.device
                    )
                y_true.append(label_map[sid])
                y_pred.append(pred)
                y_prob.append(prob)
            y_prob = np.vstack(y_prob) if y_prob else np.empty((0,args.n_classes))

            acc = accuracy_score(y_true, y_pred) if y_true else np.nan
            # AUC
            if y_true and len(set(y_true))>1:
                if args.n_classes==2:
                    try:
                        auc = roc_auc_score(y_true, [p[1] for p in y_prob])
                    except:
                        auc = np.nan
                else:
                    try:
                        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                    except:
                        auc = np.nan
            else:
                auc = np.nan

            # binary sensitivity/specificity
            if y_true and args.n_classes==2:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                sens = tp/(tp+fn) if (tp+fn)>0 else 0.
                spec = tn/(tn+fp) if (tn+fp)>0 else 0.
            else:
                sens, spec = np.nan, np.nan

            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) if y_true else np.nan

            records.append({
                'fold': fold,
                'split': split,
                'accuracy': acc,
                'auc': auc,
                'sensitivity': sens,
                'specificity': spec,
                'f1_score': f1
            })

    out_df = pd.DataFrame(records)
    out_df.to_csv(os.path.join(args.model_dir, 'final_metric_summary.csv'), index=False)
    print(f"Wrote metrics to {args.model_dir}/final_metric_summary.csv")


if __name__ == '__main__':
    main()
