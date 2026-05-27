#!/usr/bin/env python
"""
Example CLI:

python ensemble_clam.py \
  --model_dirs /path/to/modelA /path/to/modelB /path/to/modelC \
  --feature_dirs /path/to/featA  /path/to/featB  /path/to/featC \
  --labels_csv /path/to/labels.csv \
  --task task_kidney_grade \
  --output_dir /path/to/output \
  --model_type clam_sb \
  --model_size small \
  --n_classes 2 \
  --device cuda
"""
import os
import argparse
import itertools
import pandas as pd
import h5py
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, f1_score
)

# Placeholder imports: replace 'models.model_clam' with your actual module path
from models.model_clam import CLAM_SB, CLAM_MB


def infer_embed_dim(feature_dir):
    """Open the first .h5 in feature_dir and return features.shape[1]."""
    for fn in os.listdir(feature_dir):
        if fn.endswith('.h5'):
            with h5py.File(os.path.join(feature_dir, fn), 'r') as f:
                return f['features'].shape[1]
    raise RuntimeError(f"No .h5 files found in {feature_dir}")


def load_best_model(model_dir, model_type, model_size, embed_dim, n_classes, device):
    df = pd.read_csv(os.path.join(model_dir, 'summary.csv'))
    best_fold = int(df['test_auc'].idxmax())
    ckpt_file = os.path.join(model_dir, f's_{best_fold}_checkpoint.pt')

    kwargs = dict(
        gate=True, size_arg=model_size, dropout=0.0,
        k_sample=1, n_classes=n_classes, subtyping=False,
        embed_dim=embed_dim
    )
    model = CLAM_SB(**kwargs) if model_type=='clam_sb' else CLAM_MB(**kwargs)
    ckpt = torch.load(ckpt_file, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, best_fold


def slide_prediction(model, feat_path, device):
    """
    Given a CLAM model and path to .h5 features, returns:
      - prob: numpy array of shape (n_classes,) of slide-level probabilities
      - pred: integer class prediction
    """
    with h5py.File(feat_path, 'r') as f:
        feats = f['features'][:]
    feats_t = torch.from_numpy(feats).to(device).float()
    lbl = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        _, prob, pred, _, _ = model(feats_t, lbl, instance_eval=False)
    return prob.cpu().numpy().ravel(), int(pred.item())


def predict_on_folder(model, feature_dir, device):
    """Returns a DataFrame with columns [slide_id, prob_0, …, prob_{C-1}, pred]."""
    rows = []
    slide_files = [fn for fn in os.listdir(feature_dir) if fn.endswith('.h5')]
    for fn in tqdm(slide_files, desc=f"Processing slides in {os.path.basename(feature_dir)}"):
        slide_id = os.path.splitext(fn)[0]
        fp = os.path.join(feature_dir, fn)
        probs, pred = slide_prediction(model, fp, device)
        row = {'slide_id': slide_id}
        for c, p in enumerate(probs):
            row[f'prob_{c}'] = p
        row['pred'] = pred
        rows.append(row)
    return pd.DataFrame(rows)


def compute_binary_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp+fn)>0 else 0.0
    spec = tn / (tn + fp) if (tn+fp)>0 else 0.0
    f1  = f1_score(y_true, y_pred)
    return acc, auc, sens, spec, f1


def main():
    p = argparse.ArgumentParser(description="Ensemble multiple CLAM models via majority vote")
    p.add_argument('--model_dirs', nargs='+', required=True,
                   help="List of directories, each with a summary.csv and checkpoints")
    p.add_argument('--feature_dirs', nargs='+', required=True,
                   help="List of feature-folders (must align 1:1 with --model_dirs)")
    p.add_argument('--labels_csv', required=True,
                   help="CSV mapping slide_id -> label (columns: slide_id,label)")
    p.add_argument('--task', choices=[
        'task_kidney_grade', 'task_prostate_grade', 'task_rectal_stage'
    ], required=True,
                   help="Which binary mapping to apply to labels")
    p.add_argument('--output_dir', required=True,
                   help="Where to write raw_predictions.csv & ensemble_metrics.csv")
    p.add_argument('--model_type', choices=['clam_sb','clam_mb'], required=True)
    p.add_argument('--model_size', required=True)
    p.add_argument('--n_classes', type=int, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_csv = os.path.join(args.output_dir, 'raw_predictions.csv')

    # Load and map labels to binary
    labels_df = pd.read_csv(args.labels_csv, usecols=['slide_id','label'])
    if args.task == 'task_kidney_grade':
        mapping = {0:0,1:0,2:1,3:1}
    elif args.task == 'task_prostate_grade':
        mapping = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:1,9:1,10:1}
    elif args.task == 'task_rectal_stage':
        mapping = {1:0,2:0,3:1,4:1}
    else:
        raise ValueError(f"Unknown task {args.task}")
    labels_df['label'] = labels_df['label'].map(mapping)

    # 1) Raw predictions: skip if exists or regenerate
    if os.path.exists(raw_csv):
        print(f"Found existing {raw_csv}, loading...")
        raw = pd.read_csv(raw_csv)
        # re-map any old labels
        raw['label'] = raw['label'].map(mapping)
        best_folds = None  # will load below if needed
    else:
        all_dfs = []
        best_folds = []
        model_pairs = list(zip(args.model_dirs, args.feature_dirs))
        for md, fd in tqdm(model_pairs, desc="Loading and predicting with each model", total=len(model_pairs)):
            embed_dim = infer_embed_dim(fd)
            model, fold = load_best_model(md, args.model_type,
                                          args.model_size, embed_dim,
                                          args.n_classes, args.device)
            best_folds.append(fold)
            name = f"{os.path.basename(md)}_fold{fold}"

            df = predict_on_folder(model, fd, args.device)
            df = df.merge(labels_df, on='slide_id')
            prob_cols = [c for c in df if c.startswith('prob_')]
            new_prob_cols = [f"{name}_{c}" for c in prob_cols]
            new_pred_col = f"{name}_pred"
            df = df.rename(columns={**{c: new for c, new in zip(prob_cols, new_prob_cols)},
                                     'pred': new_pred_col})
            df = df[['slide_id', 'label'] + new_prob_cols + [new_pred_col]]
            all_dfs.append(df)

        raw = all_dfs[0]
        for df in tqdm(all_dfs[1:], desc="Merging model predictions"):
            raw = raw.merge(df, on=['slide_id','label'], validate='one_to_one')
        cols = ['label','slide_id'] + [c for c in raw.columns if c not in ('label','slide_id')]
        raw = raw[cols]

        raw.to_csv(raw_csv, index=False)
        print(f"Wrote raw predictions to {raw_csv}")

    # Load splits from first model's best_fold
    if best_folds is None:
        # if loading raw, infer fold from filename of first pred column
        first_pred = [c for c in raw.columns if c.endswith('_pred')][0]
        fold_str = first_pred.split('_')[-1].replace('pred','')
        splits_fold = fold_str
        splits_file = os.path.join(args.model_dirs[0], f"splits_{splits_fold}.csv")
    else:
        splits_file = os.path.join(args.model_dirs[0], f"splits_{best_folds[0]}.csv")
    splits_df = pd.read_csv(splits_file)
    splits_long = splits_df.melt(value_vars=['train','val','test'],
                                 var_name='split', value_name='slide_id')
    splits_long = splits_long[['slide_id','split']]

    # merge split info into raw
    raw_split = raw.merge(splits_long, on='slide_id', how='left')

    # 2) Build ensembled combinations & compute metrics per split
    N = len(args.model_dirs)
    names = [c.rsplit('_',1)[0] for c in raw.columns if c.endswith('_pred')]
    combos = list(itertools.combinations(range(N), 3))
    combos.append(tuple(range(N)))

    rows = []
    for combo in tqdm(combos, desc="Computing ensemble metrics", total=len(combos)):
        picked = [names[i] for i in combo]
        # vote and probs across all slides
        vote_all = raw_split[[f"{n}_pred" for n in picked]].mode(axis=1)[0].astype(int)
        prob_array = [raw_split[[f"{n}_prob_{c}" for c in range(args.n_classes)]].values for n in picked]
        mean_probs_all = sum(prob_array) / len(picked)

        metrics = {'ensemble': '+'.join(picked)}
        # compute each split
        for phase in ['train','val','test']:
            mask = raw_split['split']==phase
            y_true = raw_split.loc[mask,'label'].values
            y_pred = vote_all.loc[mask].values
            if args.n_classes == 2:
                y_prob = mean_probs_all[mask,1]
            else:
                y_prob = mean_probs_all[mask].max(axis=1)
            acc, auc, sens, spec, f1 = compute_binary_metrics(y_true, y_pred, y_prob)
            metrics.update({
                f"{phase}_accuracy": acc,
                f"{phase}_auc": auc,
                f"{phase}_sensitivity": sens,
                f"{phase}_specificity": spec,
                f"{phase}_f1": f1
            })
        rows.append(metrics)

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, 'ensemble_metrics.csv')
    summary.to_csv(out_csv, index=False)
    print(f"Wrote ensemble metrics to {out_csv}")

if __name__ == '__main__':
    main()
