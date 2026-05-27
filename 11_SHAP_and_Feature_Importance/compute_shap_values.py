#!/usr/bin/env python3
"""
compute_shap_values.py

Compute SHAP attributions for a trained CLAM model's input features using Captum's GradientShap.

0. Finds the best-performing split/fold by `test_auc` from `summary.csv` in --model_dir
1. Loads that checkpoint (s_{n}_checkpoint.pt) into a CLAM_SB or CLAM_MB model
2. Samples --n_background slides *only* from the training split to build a background (baseline) set
3. Uses attention values (from `<feature_dir>/attentions`) to pick the lowest-`--top_n` tiles per background slide as baseline
4. For each target slide, uses attention values to pick the highest-`--top_n` tiles for SHAP inputs
5. Uses `GradientShap` with a custom forward-wrapper (`forward_flat`) that flattens features into [batch, top_n*embed_dim]
   and returns logits collapsed to [B, n_classes], then requests `target=1` (the positive-class logit) for attribution
6. Saves each slide's SHAP array as a .npy in `<feature_dir>/SHAP/`
7. Aggregates mean absolute SHAP across all slides to produce a global feature importance vector,
   saved as `global_shap.h5` in --model_dir

Example:
  python compute_shap_values.py \
    --model_dir /path/to/CLAM_musk_norm \
    --feature_dir /path/to/features_musk \
    --model_type clam_sb \
    --model_size small \
    --n_classes 2 \
    --n_background 50 \
    --top_n 100 \
    --device cuda:0
"""
import os, random, h5py, torch, numpy as np, pandas as pd
from tqdm import tqdm
from captum.attr import GradientShap
from models.model_clam import CLAM_SB, CLAM_MB

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Compute SHAP values for CLAM model inputs")
    p.add_argument("--model_dir",   type=str, required=True,
                   help="Directory with CLAM summary.csv and s_*_checkpoint.pt files")
    p.add_argument("--feature_dir", type=str, required=True,
                   help="Directory of slide-level .h5 feature bags")
    p.add_argument("--model_type", choices=['clam_sb','clam_mb'], default='clam_sb',
                   help="Which CLAM architecture to load")
    p.add_argument("--model_size", choices=['small','big'], default='small',
                   help="Model size for CLAM (small or big)")
    p.add_argument("--n_classes",  type=int, required=True,
                   help="Number of output classes in each model (binary uses positive class)")
    p.add_argument("--n_background", type=int, default=50,
                   help="Number of background slides to sample from train split")
    p.add_argument("--top_n", type=int, default=100,
                   help="Number of tiles per slide for SHAP input and baseline")
    p.add_argument("--device",     default='cuda:0',
                   help="PyTorch device identifier (e.g. 'cuda:0' or 'cpu')")
    return p.parse_args()

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

def get_background_ids(model_dir, best_fold, n_background, seed=0):
    df_split = pd.read_csv(os.path.join(model_dir, f'splits_{best_fold}.csv'))
    train_ids = [s for s in df_split['train'].dropna().tolist()]
    random.seed(seed)
    return random.sample(train_ids, min(n_background, len(train_ids)))

def load_features(path):
    with h5py.File(path,'r') as f:
        return f['features'][:]

def load_attention(path):
    with h5py.File(path,'r') as f:
        return f['attention'][:]

def main():
    args = parse_args()
    shap_dir = os.path.join(args.feature_dir, 'SHAP')
    os.makedirs(shap_dir, exist_ok=True)

    slides = [f[:-3] for f in os.listdir(args.feature_dir) if f.endswith('.h5')]
    if not slides:
        raise RuntimeError("No .h5 files found in feature_dir")
    embed_dim = load_features(os.path.join(args.feature_dir, slides[0] + '.h5')).shape[1]

    model, best_fold = load_best_model(
        args.model_dir, args.model_type, args.model_size,
        embed_dim, args.n_classes, args.device
    )
    bg_ids = get_background_ids(args.model_dir, best_fold, args.n_background)
    attention_dir = os.path.join(args.feature_dir, 'attentions')

    # Build baseline: lowest-attended tiles per background slide
    bg_flat = []
    for sid in tqdm(bg_ids, desc="Loading background slides"):
        feats = load_features(os.path.join(args.feature_dir, f"{sid}.h5"))
        atts  = load_attention(os.path.join(attention_dir, f"{sid}.h5"))
        idxs = np.argsort(atts)[:args.top_n]
        sel  = feats[idxs]
        if sel.shape[0] < args.top_n:
            sel = np.pad(sel, ((0, args.top_n - sel.shape[0]), (0,0)), constant_values=0)
        bg_flat.append(sel.reshape(-1))
    baselines_flat = torch.stack([torch.from_numpy(x) for x in bg_flat], dim=0)
    baselines_flat = baselines_flat.to(args.device).float()

    # Forward wrapper: reshape and return collapsed logits [B, n_classes]
    def forward_flat(x_flat):
        B = x_flat.shape[0]
        feats_batch = x_flat.view(B, args.top_n, embed_dim)
        out = []
        for i in range(B):
            logits = model(feats_batch[i], None)[0]        # [1, n_classes]
            out.append(logits)
        return torch.stack(out, dim=0).squeeze(1)        # [B, n_classes]

    explainer = GradientShap(forward_flat)

    global_accum = []
    for sid in tqdm(slides, desc="Processing slides"):
        feats = load_features(os.path.join(args.feature_dir, f"{sid}.h5"))
        atts  = load_attention(os.path.join(attention_dir, f"{sid}.h5"))
        idxs = np.argsort(atts)[-args.top_n:]
        sel  = feats[idxs]
        if sel.shape[0] < args.top_n:
            sel = np.pad(sel, ((0, args.top_n - sel.shape[0]), (0,0)), constant_values=0)
        input_flat = torch.from_numpy(sel.reshape(1, -1)).to(args.device).float()

        attributions_flat = explainer.attribute(
            input_flat,
            baselines=baselines_flat,
            target=1,
            n_samples=50
        )
        attributions = attributions_flat.view(args.top_n, embed_dim).cpu().numpy()
        np.save(os.path.join(shap_dir, f"{sid}.npy"), attributions)
        global_accum.append(np.abs(attributions))

    # Aggregate global SHAP
    global_mean = np.mean(np.concatenate(global_accum, axis=0), axis=0)
    with h5py.File(os.path.join(args.model_dir, 'global_shap.h5'), 'w') as hf:
        hf.create_dataset('feature_importance', data=global_mean)

    print("SHAP computation complete.")

if __name__ == '__main__':
    main()
