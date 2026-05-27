# Group 6 — Feature Engineering and Pruning

Scripts for combining multiple foundation model feature sets into unified representations, and pruning redundant dimensions before model training.

---

### create_combined_fm_embeddings.py
**Purpose:** Combine tile-level features from multiple foundation models into unified H5 files, with optional two-stage correlation-based pruning (intra-model then inter-model).

**Inputs:**
- `--feature_dirs` (paths, **required**, nargs=`+`): Directories containing per-slide H5 feature files, one directory per model
- `--output_dir` (path, **required**): Root output directory; when pruning is enabled, subdirectories are created per threshold (e.g., `thr_0p6/`)
- `--slides` (str, nargs=`+`, optional): Explicit list of slide IDs to process — defaults to the intersection of slides found across all model directories
- `--prune` (flag, optional): Enable two-stage correlation-based pruning
- `--labels_csv` (path, **required if `--prune`**): CSV with columns `slide_id` and `label`, used to identify top-attention tiles
- `--task` (str, **required if `--prune`**, choices: `task_kidney_grade`, `task_prostate_grade`, `task_rectal_stage`): Task identifier for label remapping
- `--corr_thresholds` (str, optional, default=`"0.6"`): Comma-separated correlation thresholds to generate outputs for (e.g., `"0.4,0.5,0.6"`)
- File: H5 files with `features` dataset and optionally an `attentions/` subfolder (used to select top-100 tiles for pruning)

**Outputs:**
- Per threshold, a subdirectory `thr_0p{X}/` containing:
  - Combined H5 files per slide with concatenated/pruned features
  - `feature_sources.npy` — mapping of final feature indices to source model and original index
  - `pruning_report.csv` — counts of features removed at each pruning stage per model

**Example:**
```bash
python create_combined_fm_embeddings.py \
    --feature_dirs /path/features_conch /path/features_virchow2 /path/features_gigapath \
    --output_dir /path/combined_tile_features \
    --prune \
    --labels_csv /path/grade_labels.csv \
    --task task_kidney_grade \
    --corr_thresholds 0.4,0.5,0.6
```

---

### create_combined_slide_fm_embeddings.py
**Purpose:** Combine slide-level embeddings from multiple foundation models into unified H5 files, with optional two-stage correlation-based pruning.

**Inputs:**
- `--feature_dirs` (paths, **required**, nargs=`+`): Directories with per-slide H5 files (one feature vector per slide)
- `--output_dir` (path, **required**): Root output directory; pruning creates subdirectories per threshold
- `--slides` (str, nargs=`+`, optional): Explicit slide IDs — defaults to intersection across model directories
- `--prune` (flag, optional): Enable two-stage pruning
- `--labels_csv` (path, **required if `--prune`**): CSV with columns `slide_id` and `label`
- `--task` (str, **required if `--prune`**, choices: `task_kidney_grade`, `task_prostate_grade`, `task_rectal_stage`): Task for label remapping
- `--corr_thresholds` (str, optional, default=`"0.6"`): Comma-separated correlation thresholds
- File: H5 files with features of shape `(D,)` per slide

**Outputs:**
- Without pruning: Combined H5 files written directly to `--output_dir`
- With pruning: Per-threshold subdirectories (`thr_0p{X}/`) containing:
  - Combined H5 files per slide with pruned features
  - `feature_sources.npy` — feature index to source model mapping
  - `pruning_report.csv` — removal counts per pruning stage

**Example:**
```bash
python create_combined_slide_fm_embeddings.py \
    --feature_dirs /path/slide_feats_conch /path/slide_feats_virchow2 \
    --output_dir /path/combined_slide_features \
    --prune \
    --labels_csv /path/grade_labels.csv \
    --task task_kidney_grade \
    --corr_thresholds 0.4,0.5,0.6
```

---

### corr_prunning.py
**Purpose:** Standalone greedy correlation-based feature selection library — selects the best uncorrelated features using two-sample t-tests ranked by p-value, then greedily removes correlated redundancies. Intended as an imported module, not a standalone CLI script.

**Inputs (Python API):**
- `data` (ndarray, shape `N × D`): Feature matrix
- `classes` (ndarray, shape `N`): Class labels in `{1, -1}`
- `idx_pool` (list): Feature indices to consider for selection
- `num_features` (int, optional, default=`1e8`): Maximum number of features to select
- `correlation_factor` (float, optional, default=`0.6`): Absolute correlation threshold above which a feature is considered redundant
- `correlation_metric` (str, optional, default=`'spearman'`): Correlation metric to use — `'spearman'` or `'pearson'`

**Outputs (return values):**
- `selected` (list): Sorted list of selected feature indices
- `selected_pvals` (list): Corresponding p-values for selected features

**Example (internal import):**
```python
from corr_prunning import pick_best_uncorrelated_features

selected_indices, pvals = pick_best_uncorrelated_features(
    data=feature_matrix,        # shape (N, D)
    classes=labels,             # {1, -1}
    idx_pool=list(range(D)),
    correlation_factor=0.6,
    correlation_metric='spearman'
)
```
