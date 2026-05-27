#!/usr/bin/env python
"""
Ensemble slide-level MLP models via tie-break by averaged probability.

Example CLI:
    python ensemble_slide_mlp.py \
        --model_dirs /path/to/modelA /path/to/modelB \
        --feature_dirs /path/to/featA /path/to/featB \
        --labels_csv /path/to/labels.csv \
        --splits_dir /path/to/splits_dir \
        --task task_kidney_grade \
        --output_dir /path/to/output \
        --device cuda

This version computes ensemble metrics for all pairs of models (combinations of 2)
and for the full ensemble of all models, using averaged probabilities
for tie-breaking (threshold at 0.5).
"""
import os
import argparse
import itertools
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

# import components from training script
from train_slide_models import load_features, SlideMLP


def infer_feat_dim(feature_dir):
    """Infer feature dimensionality by loading first .h5 file in directory."""
    for fn in os.listdir(feature_dir):
        if fn.endswith('.h5'):
            feats = load_features(os.path.join(feature_dir, fn))
            feats = np.squeeze(feats)
            return feats.shape[-1]
    raise RuntimeError(f"No .h5 files found in {feature_dir}")


def load_best_model(model_dir, embed_dim, device):
    """Load best-fold MLP checkpoint from model_dir, return model and fold index."""
    summary_csv = os.path.join(model_dir, 'summary.csv')
    df = pd.read_csv(summary_csv)
    df_test = df[df['split'] == 'test']
    if df_test.empty:
        raise RuntimeError(f"No test split entries found in {summary_csv}")
    best_idx = df_test['auc'].idxmax()
    best_fold = int(df_test.loc[best_idx, 'fold'])

    ckpt_path = os.path.join(model_dir, f"fold_{best_fold}", "model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt['mlp_args']

    model = SlideMLP(
        input_dim=embed_dim,
        depth=int(args['mlp_depth']),
        hidden_size=int(args['hidden_size']),
        dropout_rate=float(args['dropout'])
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model, best_fold


def slide_prediction(model, feat_path, device):
    """Run a single-slide MLP prediction: returns (prob, pred)."""
    feats = load_features(feat_path)
    feats = np.squeeze(feats)
    x = torch.from_numpy(feats).float().to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
    prob = float(probs.cpu().numpy().ravel()[0])
    pred = int(prob >= 0.5)
    return prob, pred


def predict_on_folder(model, feature_dir, device, name):
    """Generate DataFrame of predictions for all slides in feature_dir."""
    rows = []
    slide_files = [fn for fn in os.listdir(feature_dir) if fn.endswith('.h5')]
    for fn in tqdm(slide_files, desc=f"Processing {os.path.basename(feature_dir)}"):
        slide_id = os.path.splitext(fn)[0]
        prob, pred = slide_prediction(model, os.path.join(feature_dir, fn), device)
        rows.append({
            'slide_id': slide_id,
            f"{name}_prob": prob,
            f"{name}_pred": pred
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble slide-level MLP models via tie-break by averaged probability"
    )
    parser.add_argument('--model_dirs', nargs='+', required=True,
                        help="List of directories with summary.csv and fold_{n}/model.pt files")
    parser.add_argument('--feature_dirs', nargs='+', required=True,
                        help="List of feature dirs (one .h5 per slide), aligned with model_dirs")
    parser.add_argument('--labels_csv', required=True,
                        help="CSV mapping slide_id -> label (slide_id,label)")
    parser.add_argument('--splits_dir', required=True,
                        help="Directory containing splits_<fold>.csv for each fold")
    parser.add_argument('--task', choices=[
        'task_kidney_grade','task_prostate_grade','task_rectal_stage'
    ], required=True,
                        help="Which binary mapping to apply to labels")
    parser.add_argument('--output_dir', required=True,
                        help="Where to write raw_predictions.csv & ensemble_metrics.csv")
    parser.add_argument('--device', default='cpu',
                        help="Torch device (cpu or cuda)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_csv = os.path.join(args.output_dir, 'raw_predictions.csv')

    # load/map labels
    labels_df = pd.read_csv(args.labels_csv, usecols=['slide_id','label'])
    mapping = {
        'task_kidney_grade': {0:0,1:0,2:1,3:1},
        'task_prostate_grade': {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:1,9:1,10:1},
        'task_rectal_stage': {1:0,2:0,3:1,4:1}
    }[args.task]
    labels_df['label'] = labels_df['label'].map(mapping)

    # Raw predictions
    best_folds = []
    if os.path.exists(raw_csv):
        print(f"Found existing {raw_csv}, loading...")
        raw = pd.read_csv(raw_csv)
        raw['label'] = raw['label'].map(mapping)
        for md in args.model_dirs:
            df = pd.read_csv(os.path.join(md,'summary.csv'))
            df_test = df[df['split']=='test']
            best_folds.append(int(df_test.loc[df_test['auc'].idxmax(),'fold']))
    else:
        raw = labels_df[['slide_id','label']].copy()
        for md,fd in tqdm(zip(args.model_dirs,args.feature_dirs),
                           desc="Loading & predicting models",total=len(args.model_dirs)):
            dim = infer_feat_dim(fd)
            model,fold = load_best_model(md,dim,args.device)
            best_folds.append(fold)
            name = f"{os.path.basename(md)}_fold{fold}"
            dfp = predict_on_folder(model,fd,args.device,name)
            raw = raw.merge(dfp,on='slide_id',validate='one_to_one')
        raw.to_csv(raw_csv,index=False)
        print(f"Wrote raw predictions to {raw_csv}")

    # load splits for the first best_fold
    split_file = os.path.join(args.splits_dir,f"splits_{best_folds[0]}.csv")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Cannot find split file: {split_file}")
    splits = pd.read_csv(split_file)
    splits_long = pd.concat([
        pd.DataFrame({'slide_id':splits[s].dropna().astype(str),'split':s})
        for s in ['train','val','test']
    ],ignore_index=True)
    raw_split = raw.merge(splits_long,on='slide_id',how='left')

    # Ensemble metrics: pairs and full ensemble
    N = len(args.model_dirs)
    names = [c[:-5] for c in raw.columns if c.endswith('_pred')]
    combos = list(itertools.combinations(range(N),2)) + [tuple(range(N))]

    records = []
    for combo in tqdm(combos, desc="Computing ensemble metrics", total=len(combos)):
        picked = [names[i] for i in combo]
        # average probabilities for ensemble
        probs = raw_split[[f"{n}_prob" for n in picked]].mean(axis=1)
        # tie-break by averaged prob >= 0.5
        preds = (probs >= 0.5).astype(int)

        rec = {'ensemble':'+'.join(picked)}
        for phase in ['train','val','test']:
            mask = raw_split['split']==phase
            y_true = raw_split.loc[mask,'label']
            y_pred = preds[mask]
            y_prob = probs[mask]
            acc = accuracy_score(y_true,y_pred)
            try:
                auc = roc_auc_score(y_true,y_prob)
            except ValueError:
                auc = float('nan')
            tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()
            sens = tp/(tp+fn) if (tp+fn) else 0.0
            spec = tn/(tn+fp) if (tn+fp) else 0.0
            f1 = f1_score(y_true,y_pred)
            rec.update({
                f"{phase}_accuracy":acc,
                f"{phase}_auc":auc,
                f"{phase}_sensitivity":sens,
                f"{phase}_specificity":spec,
                f"{phase}_f1":f1
            })
        records.append(rec)

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(os.path.join(args.output_dir,'ensemble_metrics.csv'),index=False)
    print(f"Wrote ensemble metrics to {os.path.join(args.output_dir,'ensemble_metrics.csv')}")

if __name__ == '__main__':
    main()
