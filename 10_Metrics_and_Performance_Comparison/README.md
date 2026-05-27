# Group 10 — Metrics and Performance Comparison

Scripts for computing, aggregating, and statistically comparing classification and survival metrics across CLAM and slide-level MLP models.

---

### compute_clam_metrics.py
**Purpose:** Compute per-fold classification metrics (accuracy, AUC, sensitivity, specificity, F1) for CLAM or SubCLAM models across all cross-validation splits.

**Inputs:**
- `--model_dir` (str, **required**): Directory containing model checkpoints (`s_{fold}_checkpoint.pt`) and split CSVs
- `--feature_dir` (str, **required**): Directory with per-slide H5 feature files
- `--labels_csv` (str, **required**): CSV with columns `slide_id`, `label` (integer)
- `--task` (str, **required**, choices: `task_kidney_grade`, `task_prostate_grade`, `task_rectal_stage`, `none`): Task for label remapping
- `--n_classes` (int, **required**): Number of output classes after remapping
- `--model_type` (str, optional, default=`"clam_sb"`, choices: `clam_sb`, `clam_mb`, `subclam`): Model architecture
- `--model_size` (str, optional, default=`"small"`, choices: `small`, `big`): Model size
- `--cluster_cache_dir` (str, optional): Required for SubCLAM — directory containing `{slide_id}_clusters.h5` files
- `--n_clusters` (int, optional, default=`5`): Number of clusters for SubCLAM
- `--device` (str, optional, default=`"cuda:0"`): Torch device
- File: Checkpoints at `{model_dir}/s_{fold}_checkpoint.pt`; split CSVs at `{model_dir}/splits_{fold}.csv`; feature H5 files with `features` dataset; cluster H5s (SubCLAM only)

**Outputs:**
- `{model_dir}/final_metric_summary.csv` — per-fold, per-split metrics with columns: `fold`, `split`, `accuracy`, `auc`, `sensitivity`, `specificity`, `f1_score`

**Example:**
```bash
python compute_clam_metrics.py \
    --model_dir /path/to/clam_model \
    --feature_dir /path/to/features \
    --labels_csv /path/to/grade_labels.csv \
    --task task_kidney_grade \
    --n_classes 2 \
    --model_type clam_sb \
    --device cuda:0
```

---

### slide_level_metrics.py
**Purpose:** Compute per-fold metrics (accuracy, AUC, F1, sensitivity, specificity) for slide-level MLP models across all cross-validation splits.

**Inputs:**
- `--model_dirs` (str, nargs=`+`, **required**): Directories with trained MLP models (each containing `fold_{k}/model.pt`)
- `--feature_dirs` (str, nargs=`+`, **required**): Feature directories corresponding to each model (same order as `--model_dirs`)
- `--splits_dir` (str, **required**): Directory with `splits_{fold}.csv` files
- `--labels_csv` (str, **required**): CSV with columns `slide_id`, `label`
- `--task` (str, **required**): Task for label remapping
- `--n_classes` (int, **required**): Number of output classes
- `--device` (str, optional, default=`"cuda:0"`): Torch device
- `--num_folds` (int, optional, default=`3`): Number of folds to evaluate
- File: Model checkpoints at `{model_dir}/fold_{k}/model.pt`; feature H5 files with single feature vector per slide; split CSVs with `train`/`val`/`test` columns

**Outputs:**
- `{model_dir}/final_metrics.csv` — per-fold, per-split metrics with columns: `fold`, `split`, `n_samples`, `accuracy`, `f1`, `auc`, `sensitivity`, `specificity`

**Example:**
```bash
python slide_level_metrics.py \
    --model_dirs /path/mlp_conch /path/mlp_virchow2 \
    --feature_dirs /path/slide_feats_conch /path/slide_feats_virchow2 \
    --splits_dir /path/splits \
    --labels_csv /path/grade_labels.csv \
    --task task_kidney_grade \
    --n_classes 2 \
    --num_folds 5
```

---

### compare_model_performance.py
**Purpose:** Compare tile-level CLAM model performance across individual models, a pruned combined model, and a naive combined model using bootstrap sampling and statistical tests (Wilcoxon, Friedman, effect sizes, confidence intervals).

**Inputs:**
- `--individual_model_dirs` (str, nargs=`+`, **required**): Directories for individual (non-combined) CLAM models
- `--individual_feature_dirs` (str, nargs=`+`, **required**): Feature directories for individual models (same order)
- `--pruned_model_dir` (str, **required**): Directory for the pruned combined model
- `--pruned_feature_dir` (str, **required**): Feature directory for the pruned model
- `--naive_model_dir` (str, **required**): Directory for the naive (concatenated, unpruned) combined model
- `--naive_feature_dir` (str, **required**): Feature directory for the naive model
- `--labels_csv` (str, **required**): CSV with columns `slide_id`, `label`
- `--split_file` (str, **required**): CSV with split assignments — supports both `train`/`val`/`test` column format and `slide_id`/`split` row format
- `--task` (str, **required**): Task for label remapping
- `--output_dir` (str, **required**): Output directory
- `--n_classes` (int, **required**): Number of classes after remapping
- `--model_type` (str, optional, default=`"clam_sb"`, choices: `clam_sb`, `clam_mb`): Model architecture
- `--model_size` (str, optional, default=`"small"`, choices: `small`, `big`): Model size
- `--device` (str, optional, default=`"cuda:0"`): Torch device
- `--bootstrap_iterations` (int, optional, default=`50`): Number of bootstrap iterations
- `--bootstrap_fraction` (float, optional, default=`0.8`): Fraction of data per bootstrap sample
- File: Best fold auto-selected from `{model_dir}/final_metric_summary.csv`; checkpoints at `{model_dir}/s_{fold}_checkpoint.pt`; feature H5 files

**Outputs:**
- `{output_dir}/model_comparison_results.xlsx` — Excel with sheets: `bootstrap_*`, `statistical_tests`, `summary_statistics`, `confidence_intervals`, `effect_sizes`
- `{output_dir}/combined_metrics_boxplot.png` — publication-ready box plots of per-metric bootstrap distributions

**Example:**
```bash
python compare_model_performance.py \
    --individual_model_dirs /m/conch /m/virchow2 /m/gigapath /m/musk /m/hoptimus \
    --individual_feature_dirs /f/conch /f/virchow2 /f/gigapath /f/musk /f/hoptimus \
    --pruned_model_dir /m/pruned \
    --pruned_feature_dir /f/pruned \
    --naive_model_dir /m/naive \
    --naive_feature_dir /f/naive \
    --labels_csv /path/grade_labels.csv \
    --split_file /path/splits_0.csv \
    --task task_kidney_grade \
    --output_dir /path/comparison_output \
    --n_classes 2 \
    --bootstrap_iterations 50
```

---

### compare_slide_model_performance.py
**Purpose:** Compare slide-level MLP model performance (individual vs. pruned vs. combined) using bootstrap sampling and statistical tests.

**Inputs:**
- `--individual_model_dirs` (str, nargs=`+`, **required**): Directories for individual MLP models
- `--individual_feature_dirs` (str, nargs=`+`, **required**): Feature directories for individual models
- `--pruned_model_dir` (str, **required**): Directory for the pruned combined MLP model
- `--pruned_feature_dir` (str, **required**): Feature directory for the pruned model
- `--combined_model_dir` (str, **required**): Directory for the naive combined MLP model
- `--combined_feature_dir` (str, **required**): Feature directory for the combined model
- `--labels_csv` (str, **required**): CSV with columns `slide_id`, `label`
- `--split_file` (str, **required**): CSV with split assignments
- `--task` (str, **required**): Task for label remapping
- `--output_dir` (str, **required**): Output directory
- `--n_classes` (int, **required**): Number of classes
- `--device` (str, optional, default=`"cuda:0"`): Torch device
- `--bootstrap_iterations` (int, optional, default=`50`): Bootstrap iterations
- `--bootstrap_fraction` (float, optional, default=`0.8`): Bootstrap fraction
- File: Best fold auto-selected from `{model_dir}/final_metrics.csv`; checkpoints at `{model_dir}/fold_{k}/model.pt`; feature H5 files

**Outputs:**
- `{output_dir}/slide_model_comparison_results.xlsx` — Excel with comprehensive statistical results
- `{output_dir}/combined_metrics_slide_boxplot.png` — publication-ready box plots

**Example:**
```bash
python compare_slide_model_performance.py \
    --individual_model_dirs /m/conch /m/virchow2 /m/gigapath \
    --individual_feature_dirs /f/conch /f/virchow2 /f/gigapath \
    --pruned_model_dir /m/pruned \
    --pruned_feature_dir /f/pruned \
    --combined_model_dir /m/combined \
    --combined_feature_dir /f/combined \
    --labels_csv /path/labels.csv \
    --split_file /path/splits_0.csv \
    --task task_kidney_grade \
    --output_dir /path/output \
    --n_classes 2
```

---

### aggregate_dice_results.py
**Purpose:** Collect Dice overlap result Excel files from multiple model subdirectories and aggregate them into a single summary CSV/Excel for downstream plotting.

**Inputs:**
- `--dice_results_dir` (str, **required**): Parent directory containing one subdirectory per model, each with a `dice_results.xlsx` file (output of `heatmap_tumor_dice.py`)
- `--output_name` (str, optional, default=`"dice_summary_all_models"`): Base name for output files
- `--include_sra` (flag, optional): Also aggregate the `sra_coverage_summary` sheet (for `rectal_new` runs)
- File: `{dice_results_dir}/{model_name}/dice_results.xlsx` with sheets `summary` (and optionally `sra_coverage_summary`)

**Outputs:**
- `{dice_results_dir}/{output_name}.csv` — aggregated long-format summary table
- `{dice_results_dir}/{output_name}.xlsx` — Excel with sheets `summary_all_models` and pivoted metric sheets per percentile
- `{dice_results_dir}/{output_name}_sra.csv` — aggregated SRA coverage summary (if `--include_sra`)

**Example:**
```bash
python aggregate_dice_results.py \
    --dice_results_dir /path/DiceResults \
    --output_name dice_summary_all_models \
    --include_sra
```

---

### aggregated_ranking_plot.py
**Purpose:** Generate a ranking bump chart of aggregated tumor + normal/benign attention coverage scores across percentile thresholds, one line per model.

**Inputs:**
- `--input` (str, **required**): Aggregated summary CSV from `aggregate_dice_results.py`; must contain columns `model`, `percentile`, `mean_tumor_cov`, and either `mean_normal_cov` (kidney) or `mean_benign_cov` (prostate)
- `--output` (str, optional, default=same directory as CSV): Output file path
- `--data_type` (str, optional, choices: `kidney`, `prostate`): Data type for axis labeling — auto-detected from CSV columns if not provided
- `--figsize` (str, optional, default=`"10.7 2.8"`): Figure size as `"W H"` in inches
- `--dpi` (int, optional, default=`150`): Output DPI
- `--format` (str, optional, default=`"png"`, choices: `png`, `pdf`, `both`): Output format

**Outputs:**
- `aggregated_ranking_tumor_normal.png` or `.pdf` (kidney), or `aggregated_ranking_tumor_benign.png` or `.pdf` (prostate)
- `aggregated_ranking_tumor_*.csv` — ranked score table per model and percentile

**Example:**
```bash
python aggregated_ranking_plot.py \
    --input /path/DiceResults/dice_summary_all_models.csv \
    --data_type kidney \
    --format both
```

---

### heatmap_tumor_dice.py
**Purpose:** Compute Dice overlap and coverage statistics between CLAM attention heatmaps and ground-truth tumor annotations across a range of percentile cutoffs.

**Inputs:**
- `--slide_dir` (str, **required**): Directory with `.svs` slide files
- `--attention_dir` (str, **required**): Directory with CLAM attention H5 files (one per slide)
- `--annotation_dir` (str, **required**): Directory with annotation XML files
- `--output_dir` (str, **required**): Output directory
- `--tile_size` (int, **required**): Tile size in pixels at the feature extraction magnification
- `--data_type` (str, optional, default=`"kidney"`, choices: `kidney`, `prostate`, `rectal`, `rectal_new`): Annotation format and parsing logic to use
- `--target_dim` (int, optional, default=`2000`): Longest dimension of the attention heatmap in pixels
- `--blur_sigma` (float, optional, default=`1.0`): Gaussian blur sigma applied to the attention map
- `--percentiles` (int, nargs=`+`, optional, default=`[25, 50, 60, 70, 80, 90]`): Range-percentage cutoffs for binary thresholding
- `--save_csv` (flag, optional): Also save a per-slide metrics CSV in addition to the Excel output
- `--num_workers` (int, optional, default=`1`): Number of parallel workers
- File: `.svs` slides; attention H5 files with `coords` and `attention`/`attn` datasets; annotation XMLs structured per `--data_type`:
  - `kidney`: `{slide_id}*.tumor.xml`, optionally `*normal.xml`
  - `prostate`: `{slide_id}.xml` with `LineColor` filtering
  - `rectal`: `{slide_id}/{slide_id}.tumor.xml`, optionally `.normal.xml`
  - `rectal_new`: `{slide_id}/{slide_id}.{sra_class}.xml` for each SRA class

**Outputs:**
- `{output_dir}/dice_results.xlsx` — Excel with sheets `per_slide` (raw per-slide metrics) and `summary` (mean/std per percentile), plus `sra_coverage_summary` for `rectal_new`
- `{output_dir}/dice_per_slide.csv` — per-slide CSV (if `--save_csv`)

**Example:**
```bash
python heatmap_tumor_dice.py \
    --slide_dir /path/to/wsi \
    --attention_dir /path/to/clam_attentions \
    --annotation_dir /path/to/annotations \
    --output_dir /path/DiceResults/conch_v15 \
    --tile_size 512 \
    --data_type kidney \
    --blur_sigma 1.0 \
    --percentiles 25 50 60 70 80 90 \
    --save_csv
```

---

### analyze_clam_survival.py
**Purpose:** Use a trained CLAM model to predict grade groups, then run Kaplan-Meier survival analysis comparing predicted high-grade vs. low-grade patients with log-rank tests.

**Inputs:**
- `--model_path` (str, **required**): Path to a checkpoint file, or to a model directory (best fold is auto-selected from `final_metric_summary.csv`)
- `--feature_dir` (str, **required**): Directory with per-slide H5 feature files
- `--splits_csv` (str, optional, default=`"kirc_splits/splits_0.csv"`): Split CSV with `train`/`val`/`test` columns
- `--master_xlsx` (str, optional, default=`"kca_master_hpc_cptacupdated.xlsx"`): Master Excel file with columns `PatientID`, `vital_status`, `days_to_last_followup`, `death_days_to`
- `--output_dir` (str, optional, default=`"survival_analysis_output/"`): Output directory
- `--model_type` (str, optional, default=`"clam_sb"`, choices: `clam_sb`, `clam_mb`): Model architecture
- `--model_size` (str, optional, default=`"small"`, choices: `small`, `big`): Model size
- `--n_classes` (int, optional, default=`2`): Number of classes
- `--embed_dim` (int, optional): Embedding dimension — inferred from checkpoint if not provided
- `--device` (str, optional, default=`"cuda:0"`): Torch device
- File: Checkpoint at `--model_path`; feature H5 files; split CSV; master Excel for survival data

**Outputs:**
- `{output_dir}/survival_curve_train.png` — Kaplan-Meier curves for training split
- `{output_dir}/survival_curve_val.png` — Kaplan-Meier curves for validation split
- `{output_dir}/survival_curve_test.png` — Kaplan-Meier curves for test split
- `{output_dir}/survival_analysis_results.csv` — log-rank test p-values and median survival per group per split

**Example:**
```bash
python analyze_clam_survival.py \
    --model_path /path/to/clam_model_dir \
    --feature_dir /path/to/features \
    --splits_csv /path/kirc_splits/splits_0.csv \
    --master_xlsx /path/kca_master_hpc_cptacupdated.xlsx \
    --output_dir /path/survival_output \
    --n_classes 2
```
