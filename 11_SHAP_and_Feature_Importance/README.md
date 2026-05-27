# Group 11 — SHAP and Feature Importance

Scripts for computing and visualizing SHAP attributions to interpret which features and tiles drive CLAM and slide-level MLP model decisions.

---

### compute_shap_values.py
**Purpose:** Compute SHAP attributions for a trained tile-level CLAM model using Captum's GradientShap, producing per-slide attribution arrays and a global feature importance summary.

**Inputs:**
- `--model_dir` (str, **required**): Directory containing `summary.csv` and fold checkpoint files (`s_*_checkpoint.pt`)
- `--feature_dir` (str, **required**): Directory with per-slide H5 feature bags
- `--n_classes` (int, **required**): Number of output classes
- `--model_type` (str, optional, default=`"clam_sb"`, choices: `clam_sb`, `clam_mb`): CLAM architecture
- `--model_size` (str, optional, default=`"small"`, choices: `small`, `big`): Model size
- `--n_background` (int, optional, default=`50`): Number of background slides to sample from the training split for the GradientShap baseline
- `--top_n` (int, optional, default=`100`): Number of top-attention tiles per slide to use as SHAP input and baseline
- `--device` (str, optional, default=`"cuda:0"`): Torch device
- File: `{model_dir}/summary.csv` — used to select the best fold by `test_auc`; `{model_dir}/splits_*.csv` — used to identify training slides; `{model_dir}/s_*_checkpoint.pt` — model weights; `{feature_dir}/{slide_id}.h5` — tile features; `{feature_dir}/attentions/{slide_id}.h5` — top-tile attention scores

**Outputs:**
- `{feature_dir}/SHAP/{slide_id}.npy` — per-slide SHAP attribution array of shape `(top_n, embed_dim)`
- `{model_dir}/global_shap.h5` — global mean absolute SHAP across all slides, stored under dataset key `feature_importance`

**Example:**
```bash
python compute_shap_values.py \
    --model_dir /path/to/clam_conch_grade \
    --feature_dir /path/to/features_conch \
    --n_classes 2 \
    --model_type clam_sb \
    --model_size small \
    --n_background 50 \
    --top_n 100 \
    --device cuda:0
```

---

### compute_shap_values_slide.py
**Purpose:** Compute SHAP attributions for trained slide-level MLP models using Captum's GradientShap, producing per-slide attribution vectors and a global importance summary.

**Inputs:**
- `--mlp_dir` (str, **required**): Directory containing `summary.csv` (or `final_metrics.csv`) and `fold_{k}/model.pt` checkpoints
- `--features_dir` (str, **required**): Directory with per-slide H5 files containing a `features` dataset (single vector per slide)
- `--splits_dir` (str, **required**): Directory with `splits_{k}.csv` files
- `--device` (str, optional, default=`"cuda:0"`): Torch device
- `--n_background` (int, optional, default=`50`): Number of training slides to use as GradientShap baselines
- `--n_samples` (int, optional, default=`50`): GradientShap `n_samples` parameter per example
- `--seed` (int, optional, default=`0`): Random seed for baseline sampling
- File: `{mlp_dir}/summary.csv` or `final_metrics.csv` — best fold selected by test AUC or accuracy; `{mlp_dir}/fold_{k}/model.pt` — model weights; `{features_dir}/{slide_id}.h5` — slide-level features; `{splits_dir}/splits_{k}.csv` — train/val/test assignments

**Outputs:**
- `{features_dir}/SHAP/{slide_id}.npy` — per-slide SHAP attribution vector of shape `(D,)`
- `{mlp_dir}/global_shap.h5` — global mean absolute SHAP across all slides under dataset key `feature_importance`

**Example:**
```bash
python compute_shap_values_slide.py \
    --mlp_dir /path/to/mlp_conch \
    --features_dir /path/to/slide_features_conch \
    --splits_dir /path/to/splits \
    --n_background 50 \
    --n_samples 50 \
    --device cuda:0
```

---

### visualize_feature_selection_shap.py
**Purpose:** Create a publication-quality multi-panel figure showing the number of features selected per model and disease at each correlation threshold (7 columns × 1 row, with stacked bars for kidney, prostate, and rectal).

**Inputs:**
- `--kidney_features` (str, **required**): Parent directory for kidney combined tile features — must contain `thr_0p{X}/feature_sources.npy` subdirectories
- `--prostate_features` (str, **required**): Parent directory for prostate combined tile features (same structure)
- `--rectal_features` (str, **required**): Parent directory for rectal combined tile features (same structure)
- `--output_dir` (str, **required**): Directory to save the output figure
- `--thresholds` (str, optional, default=`"0.1,0.2,0.3,0.4,0.5,0.6,0.7"`): Comma-separated list of correlation thresholds to visualize
- File: `feature_sources.npy` files from each threshold subdirectory (e.g., `thr_0p1/`, `thr_0p2/`, ...) for all three diseases

**Outputs:**
- `{output_dir}/feature_selection_visualization.png` — 400 DPI publication-ready figure

**Example:**
```bash
python visualize_feature_selection_shap.py \
    --kidney_features /path/combined_features/Kidney \
    --prostate_features /path/combined_features/Prostate \
    --rectal_features /path/combined_features/Rectum \
    --output_dir /path/figures \
    --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7
```

---

### visualize_feature_selection_slide.py
**Purpose:** Create a publication-quality multi-panel figure showing slide-level feature selection across diseases and thresholds (2 rows: thresholds 0.1–0.4 in row 0, 0.5–0.7 in row 1; 4 columns per row with per-disease and common-features bars).

**Inputs:**
- `--kidney_features` (str, **required**): Parent directory for kidney combined slide features — must contain `thr_0p{X}/feature_sources.npy` subdirectories
- `--prostate_features` (str, **required**): Parent directory for prostate combined slide features (same structure)
- `--rectal_features` (str, **required**): Parent directory for rectal combined slide features (same structure)
- `--output_dir` (str, **required**): Directory to save the output figure
- `--thresholds` (str, optional, default=`"0.1,0.2,0.3,0.4,0.5,0.6,0.7"`): Comma-separated correlation thresholds
- File: `feature_sources.npy` from each threshold subdirectory for all three diseases

**Outputs:**
- `{output_dir}/feature_selection_visualization_slide.png` — 400 DPI publication-ready figure

**Example:**
```bash
python visualize_feature_selection_slide.py \
    --kidney_features /path/slide_combined/Kidney \
    --prostate_features /path/slide_combined/Prostate \
    --rectal_features /path/slide_combined/Rectum \
    --output_dir /path/figures \
    --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7
```
