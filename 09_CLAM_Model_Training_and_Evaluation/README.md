# Group 9 — CLAM Model Training and Evaluation

Core multiple-instance learning training loop and model ensembling. (`eval.py` and `build_preset.py` are excluded from this library.)

---

### main.py
**Purpose:** Train CLAM (Clustering-constrained Attention Multiple Instance Learning) models for slide-level classification, with optional survival prediction mode.

**Inputs:**
- `--data_root_dir` (str, **required**): Root directory containing per-slide H5 feature files
- `--task` (str, **required**, choices: `task_1_tumor_vs_normal`, `task_2_tumor_subtyping`, `task_kidney_grade`, `task_prostate_grade`, `task_rectal_stage`): Classification task — determines which label CSV is loaded
- `--exp_code` (str, **required**): Experiment identifier string used for output naming
- `--embed_dim` (int, optional, default=`1024`): Feature embedding dimension
- `--max_epochs` (int, optional, default=`200`): Maximum training epochs per fold
- `--lr` (float, optional, default=`1e-4`): Learning rate
- `--reg` (float, optional, default=`1e-5`): L2 weight decay
- `--label_frac` (float, optional, default=`1.0`): Fraction of training labels to use
- `--seed` (int, optional, default=`1`): Random seed
- `--k` (int, optional, default=`10`): Number of cross-validation folds
- `--k_start` (int, optional, default=`-1`): First fold to train (inclusive; `-1` = start from 0)
- `--k_end` (int, optional, default=`-1`): Last fold to train (exclusive; `-1` = train all k folds)
- `--results_dir` (str, optional, default=`"./results"`): Output directory for results
- `--split_dir` (str, optional): Directory with `splits_{n}.csv` files; overrides default split path
- `--opt` (str, optional, default=`"adam"`, choices: `adam`, `sgd`): Optimizer
- `--drop_out` (float, optional, default=`0.25`): Dropout rate
- `--bag_loss` (str, optional, default=`"ce"`, choices: `svm`, `ce`): Bag-level loss function
- `--model_type` (str, optional, default=`"clam_sb"`, choices: `clam_sb`, `clam_mb`, `mil`): Model architecture
- `--model_size` (str, optional, default=`"small"`, choices: `small`, `big`): Model size variant
- `--weighted_sample` (flag, optional): Enable class-balanced weighted sampling
- `--early_stopping` (flag, optional): Enable early stopping based on validation loss
- `--log_data` (flag, optional): Enable TensorBoard logging
- `--no_inst_cluster` (flag, optional): Disable instance-level clustering loss
- `--inst_loss` (str, optional, default=`None`, choices: `svm`, `ce`, `None`): Instance clustering loss function
- `--subtyping` (flag, optional): Flag for subtyping (multi-class) problems
- `--bag_weight` (float, optional, default=`0.7`): Weight for bag-level loss relative to instance loss
- `--B` (int, optional, default=`8`): Number of patch samples for CLAM instance loss
- `--survival_mode` (flag, optional): Enable survival prediction (Cox proportional hazards) instead of classification
- `--survival_csv` (str, optional): Path to survival data CSV (required when `--survival_mode`)
- `--time_col` (str, optional, default=`"time"`): Column name for time-to-event in survival CSV
- `--event_col` (str, optional, default=`"event"`): Column name for event indicator in survival CSV
- `--cox_l1_reg` (float, optional, default=`0.0`): L1 regularization for Cox loss
- `--cox_l2_reg` (float, optional, default=`0.0`): L2 regularization for Cox loss
- `--survival_batch_size` (int, optional): Explicit batch size for survival training
- `--survival_batch_fraction` (float, optional, default=`0.25`): Fraction of dataset to use as batch size when `--survival_batch_size` is not set
- File: Label CSVs at hardcoded disease-specific paths; feature H5 files at `{data_root_dir}/{slide_id}.h5`; split CSVs at `splits/{task}_{label_frac}%/splits_{fold}.csv`

**Outputs:**
- `{results_dir}/{exp_code}_s{seed}/experiment_{exp_code}.txt` — saved run settings
- `{results_dir}/{exp_code}_s{seed}/split_{fold}_results.pkl` — per-fold results dictionary
- `{results_dir}/{exp_code}_s{seed}/summary.csv` — cross-validation summary with columns: `folds`, `test_auc`, `val_auc`, `test_acc`, `val_acc` (classification) or equivalent survival metrics

**Example:**
```bash
python main.py \
    --data_root_dir /path/to/features/conch_v15 \
    --task task_kidney_grade \
    --exp_code kidney_conch_grade \
    --embed_dim 512 \
    --k 5 \
    --lr 1e-4 \
    --model_type clam_sb \
    --model_size small \
    --seed 1
```

---

### main_stability.py
**Purpose:** Train CLAM with stability-aware features — either using stability metrics directly as additional features, or filtering FM features by their stability rank.

**Inputs:**

All standard CLAM arguments from `main.py` apply, plus:
- `--stability_mode` (str, **required**, choices: `features`, `filter`): Mode — `features` appends stability metrics as extra features; `filter` removes low-stability FM features before training
- `--stability_dir` (str, optional): Directory with per-slide stability H5 files
- `--stability_metric` (str, optional): Stability metric to use in `features` mode — choices: `icc`, `variance`, `norm_range`
- `--stability_category` (str, optional): Augmentation category — choices: `overall`, `spatial`, `color`, `noise`
- `--filter_metric` (str, optional, default=`"icc"`): Metric for `filter` mode — choices: `icc`, `variance`, `norm_range`
- `--filter_category` (str, optional, default=`"overall"`): Augmentation category for `filter` mode
- `--filter_direction` (str, optional, default=`"high"`): Keep `high` or `low` stability features
- `--filter_percentile` (float, optional, default=`80.0`): Percentile threshold (e.g., `80` = keep top 20% most stable features)
- `--filter_weighting` (flag, optional): Use soft stability weighting instead of hard binary masking
- `--models` (str, optional): Comma-separated model names for multi-model mode
- `--data_dirs` (str, optional): Comma-separated feature directories (for multi-model mode)
- `--stability_dirs` (str, optional): Comma-separated stability directories (for multi-model mode)
- `--patch_encoder` (str, optional): Single encoder name (for single-model mode)
- File: Stability H5 files at `{stability_dir}/{slide_id}.h5`; feature H5 files; split CSVs

**Outputs:** Same structure as `main.py`:
- `{results_dir}/{exp_code}_s{seed}/experiment_{exp_code}.txt`
- `{results_dir}/{exp_code}_s{seed}/split_{fold}_results.pkl`
- `{results_dir}/{exp_code}_s{seed}/summary.csv`

**Example:**
```bash
python main_stability.py \
    --data_root_dir /path/to/features \
    --stability_dir /path/to/stability_h5 \
    --task task_kidney_grade \
    --exp_code kidney_stable_icc80 \
    --stability_mode filter \
    --filter_metric icc \
    --filter_percentile 80 \
    --k 5 \
    --seed 1
```

---

### main_subclams.py
**Purpose:** Train SubCLAM — a hierarchical variant of CLAM that first clusters tiles into tissue groups, then applies attention within and across clusters.

**Inputs:**

Shares most arguments with `main.py`, plus:
- `--n_clusters` (int, optional, default=`5`): Number of tissue clusters (must match the cluster cache)
- `--cluster_cache_dir` (str, **required**): Directory with per-slide cluster H5 files (produced by `create_cluster_cache.py`)
- File: Feature H5 files; cluster H5 files at `{cluster_cache_dir}/{slide_id}_clusters.h5` with `cluster_ids` dataset; split CSVs

**Outputs:** Same structure as `main.py`, plus:
- `{results_dir}/{exp_code}_s{seed}/s_{fold}_checkpoint.pt` — model checkpoint per fold

**Example:**
```bash
python main_subclams.py \
    --data_root_dir /path/to/features \
    --cluster_cache_dir /path/to/cluster_cache \
    --task task_kidney_grade \
    --exp_code kidney_subclam_k5 \
    --n_clusters 5 \
    --k 5 \
    --seed 1
```

---

### ensemble_clam.py
**Purpose:** Ensemble multiple trained CLAM models via majority voting and compute comparative performance metrics across all model combinations.

**Inputs:**
- `--model_dirs` (str, nargs=`+`, **required**): Directories with trained checkpoints and `summary.csv` (one per model)
- `--feature_dirs` (str, nargs=`+`, **required**): Feature directories corresponding to each model (same order as `--model_dirs`)
- `--labels_csv` (str, **required**): CSV with columns `slide_id`, `label`
- `--task` (str, **required**, choices: `task_kidney_grade`, `task_prostate_grade`, `task_rectal_stage`): Task for label remapping
- `--output_dir` (str, **required**): Output directory
- `--model_type` (str, **required**, choices: `clam_sb`, `clam_mb`): CLAM architecture
- `--model_size` (str, **required**, choices: `small`, `big`): Model size
- `--n_classes` (int, **required**): Number of output classes after task remapping
- `--device` (str, optional, default=`"cpu"`): Torch device
- File: Best-fold checkpoint auto-selected from `{model_dir}/s_{fold}_checkpoint.pt` using `summary.csv`; feature H5 files; optional `{model_dir}/splits_{fold}.csv` for train/val/test delineation

**Outputs:**
- `{output_dir}/raw_predictions.csv` — per-slide predictions from every individual model
- `{output_dir}/ensemble_metrics.csv` — ensemble performance metrics broken down by split

**Example:**
```bash
python ensemble_clam.py \
    --model_dirs /path/model_conch /path/model_virchow2 /path/model_gigapath \
    --feature_dirs /path/feats_conch /path/feats_virchow2 /path/feats_gigapath \
    --labels_csv /path/grade_labels.csv \
    --task task_kidney_grade \
    --output_dir /path/ensemble_output \
    --model_type clam_sb \
    --model_size small \
    --n_classes 2
```

---

### ensemble_slide_mlps.py
**Purpose:** Ensemble slide-level MLP models using averaged probabilities with tie-breaking, computing metrics for all model combinations.

**Inputs:**
- `--model_dirs` (str, nargs=`+`, **required**): Directories containing `fold_{n}/model.pt` checkpoints (one per model)
- `--feature_dirs` (str, nargs=`+`, **required**): Feature directories corresponding to each model
- `--labels_csv` (str, **required**): CSV with columns `slide_id`, `label`
- `--splits_dir` (str, **required**): Directory with `splits_{fold}.csv` files
- `--task` (str, **required**): Task for label remapping
- `--output_dir` (str, **required**): Output directory
- `--device` (str, optional, default=`"cpu"`): Torch device
- File: Best-fold checkpoint auto-selected from `{model_dir}/final_metrics.csv`; feature H5 files with `features` dataset; split CSVs

**Outputs:**
- `{output_dir}/raw_predictions.csv` — per-slide predictions from all individual models
- `{output_dir}/ensemble_metrics.csv` — ensemble metrics for all model combinations

**Example:**
```bash
python ensemble_slide_mlps.py \
    --model_dirs /path/mlp_conch /path/mlp_virchow2 \
    --feature_dirs /path/slide_feats_conch /path/slide_feats_virchow2 \
    --labels_csv /path/grade_labels.csv \
    --splits_dir /path/splits \
    --task task_kidney_grade \
    --output_dir /path/ensemble_output
```
