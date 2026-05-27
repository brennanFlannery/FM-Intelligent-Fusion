# Group 12 — Pruning and Stability

Scripts for feature pruning analysis, stability filtering, and cluster caching after model training.

---

### pruning_visualization.py
**Purpose:** Create a combined figure showing tile-level feature pruning percentages (thermometer plots) alongside F1 score bootstrap distributions (box plots) for CLAM models at multiple pruning thresholds.

**Inputs:**
- `--pruning_excel` (str, **required**): Excel file (`tile_pruning.xlsx`) with feature counts at different thresholds per model
- `--pruned_model_dirs` (str, nargs=`+`, **required**): Directories of pruned CLAM models, one per threshold/model combination
- `--pruned_feature_dirs` (str, nargs=`+`, **required**): Feature directories corresponding to each pruned model (same order as `--pruned_model_dirs`)
- `--labels_csv` (str, **required**): CSV mapping `slide_id` to integer `label`
- `--split_file` (str, **required**): CSV with split assignments — supports `train`/`val`/`test` column format or `slide_id`/`split` row format
- `--task` (str, **required**, choices: `task_kidney_grade`, `task_prostate_grade`, `task_rectal_stage`, `none`): Task for label remapping
- `--output_dir` (str, **required**): Directory to save results
- `--n_classes` (int, **required**): Number of output classes after remapping
- `--model_type` (str, optional, default=`"clam_sb"`, choices: `clam_sb`, `clam_mb`): CLAM architecture
- `--model_size` (str, optional, default=`"small"`, choices: `small`, `big`): Model size
- `--device` (str, optional, default=`"cuda:0"`): Torch device
- `--bootstrap_iterations` (int, optional, default=`50`): Number of bootstrap iterations for F1 estimation
- `--bootstrap_fraction` (float, optional, default=`0.8`): Fraction of data per bootstrap sample
- File: Excel pruning file; `{model_dir}/summary.csv` and `s_*_checkpoint.pt`; `{feature_dir}/{slide_id}.h5`; labels CSV; split CSV

**Outputs:**
- `{output_dir}/pruning_visualization.png` — combined figure at 500 DPI with thermometer bars, total feature percentage, and bootstrapped F1 box plots

**Example:**
```bash
python pruning_visualization.py \
    --pruning_excel /path/tile_pruning.xlsx \
    --pruned_model_dirs /path/model_thr04 /path/model_thr05 /path/model_thr06 \
    --pruned_feature_dirs /path/feats_thr04 /path/feats_thr05 /path/feats_thr06 \
    --labels_csv /path/grade_labels.csv \
    --split_file /path/splits_0.csv \
    --task task_kidney_grade \
    --output_dir /path/pruning_output \
    --n_classes 2 \
    --bootstrap_iterations 50
```

---

### pruning_visualization_slide.py
**Purpose:** Create a combined figure showing slide-level feature pruning percentages and bootstrapped F1 score distributions for MLP models at multiple pruning thresholds.

**Inputs:**
- `--pruning_excel` (str, **required**): Excel file (`slide_pruning.xlsx`) with feature counts at different thresholds per model
- `--pruned_model_dirs` (str, nargs=`+`, **required**): Directories of pruned slide-level MLP models (one per threshold/model)
- `--pruned_feature_dirs` (str, nargs=`+`, **required**): Feature directories corresponding to each pruned model
- `--labels_csv` (str, **required**): CSV mapping `slide_id` to integer `label`
- `--split_file` (str, **required**): CSV with split assignments
- `--task` (str, **required**, choices: `task_kidney_grade`, `task_prostate_grade`, `task_rectal_stage`, `none`): Task for label remapping
- `--output_dir` (str, **required**): Output directory
- `--n_classes` (int, **required**): Number of output classes
- `--device` (str, optional, default=`"cuda:0"`): Torch device
- `--bootstrap_iterations` (int, optional, default=`50`): Bootstrap iterations
- `--bootstrap_fraction` (float, optional, default=`0.8`): Bootstrap fraction
- File: Excel pruning file; `{model_dir}/final_metrics.csv` and `fold_{k}/model.pt`; feature H5 files; labels and split CSVs

**Outputs:**
- `{output_dir}/pruning_visualization_slide.png` — combined figure at 500 DPI

**Example:**
```bash
python pruning_visualization_slide.py \
    --pruning_excel /path/slide_pruning.xlsx \
    --pruned_model_dirs /path/mlp_thr04 /path/mlp_thr05 /path/mlp_thr06 \
    --pruned_feature_dirs /path/slide_feats_thr04 /path/slide_feats_thr05 /path/slide_feats_thr06 \
    --labels_csv /path/grade_labels.csv \
    --split_file /path/splits_0.csv \
    --task task_kidney_grade \
    --output_dir /path/pruning_output \
    --n_classes 2
```

---

### stability_filters.py
**Purpose:** Re-exports the `StabilityFilter` class from `dataset_modules.dataset_stability` as a convenience import. This is a module, not a standalone script.

**Inputs:** None — import only.

**Outputs:** None — exposes `StabilityFilter` class.

**Usage:**
```python
from stability_filters import StabilityFilter
```

---

### create_cluster_cache.py
**Purpose:** Pre-compute tile cluster assignments for all slides using MiniBatchKMeans, saving centroids and per-slide cluster labels to disk for use by `main_subclams.py`.

**Inputs:**
- `--data_root_dir` (str, **required**): Directory containing per-slide H5 feature files with `features` and `coords` datasets
- `--csv_path` (str, **required**): CSV file with a `slide_id` column listing slides to process
- `--cluster_cache_dir` (str, **required**): Output directory for cluster cache files
- `--n_clusters` (int, optional, default=`5`): Number of global K-means clusters
- `--sample_frac` (float, optional, default=`0.1`): Fraction of tiles to sample per slide for global clustering
- `--max_tiles_per_slide` (int, optional, default=`1000`): Maximum tiles to sample per slide
- `--local_k` (int, optional, default=`10`): Number of local clusters per slide (used in stratified sampling)
- `--local_batch_size` (int, optional, default=`256`): MiniBatchKMeans batch size for local (per-slide) clustering
- `--batch_size` (int, optional, default=`10000`): MiniBatchKMeans batch size for global clustering
- `--chunk_size` (int, optional, default=`20000`): Chunk size for streaming global assignment (`0` = load all at once)
- `--seed` (int, optional, default=`42`): Random seed
- File: `{data_root_dir}/{slide_id}.h5` — per-slide H5 feature files

**Outputs:**
- `{cluster_cache_dir}/centroids.npy` — global K-means centroids of shape `(n_clusters, embed_dim)`
- `{cluster_cache_dir}/{slide_id}_clusters.h5` — per-slide cluster assignments and local centroids
- `{cluster_cache_dir}/cluster_cache_meta.json` — metadata: n_clusters, sample_frac, seed, tile counts, etc.
- `{cluster_cache_dir}/missing_slides.txt` — list of slide IDs whose H5 files were not found (if any)

**Example:**
```bash
python create_cluster_cache.py \
    --data_root_dir /path/to/features_conch \
    --csv_path /path/to/grade_labels.csv \
    --cluster_cache_dir /path/to/cluster_cache_conch \
    --n_clusters 5 \
    --sample_frac 0.1 \
    --seed 42
```
