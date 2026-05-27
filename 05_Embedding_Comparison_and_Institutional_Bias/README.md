# Group 5 — Embedding Comparison and Institutional Bias

Scripts for comparing tile and slide embeddings across foundation models, and for quantifying how much institution identity leaks into embedding space.

---

### tile_embedding_comparisons.py
**Purpose:** Compare tile-level embeddings across two or more foundation models using multiple geometric and statistical metrics (CKA, SVCCA, Procrustes, k-NN overlap, Ridge R²).

**Inputs:**
- `--model_dirs` (paths, **required**, nargs=`+`): Directories of per-slide H5 feature files, one directory per model
- `--output` (path, optional, default=`"comparison_results.xlsx"`): Output Excel file path
- `--metrics` (str, nargs=`+`, optional, default=`["all"]`): Metrics to compute — choices: `cka`, `svcca`, `procrustes`, `knn`, `ridge`, or `all`
- `--sample_size` (int, optional, default=`50000`): Maximum number of tiles to sample per model pair
- `--seed` (int, optional, default=`0`): Random seed for sampling
- `--feature_key` (str, optional, default=`"features"`): H5 dataset key for embeddings
- `--coord_key` (str, optional, default=`"coords"`): H5 dataset key for tile coordinates
- `--pca_dim` (int, optional, default=`50`): Number of PCA dimensions for Procrustes alignment
- `--var_thres` (float, optional, default=`0.99`): Variance threshold for SVCCA dimensionality reduction
- `--knn_k` (int, optional, default=`10`): Number of neighbors for k-NN overlap metric
- `--ridge_alpha` (float, optional, default=`1.0`): Ridge regression regularization strength
- `--make_figures` (flag, optional): Generate per-pair visualizations (radar plots, k-NN histograms, neighborhood plots, summary plots)
- File: Per-slide H5 files in each `--model_dirs` folder, matched by slide filename across models

**Outputs:**
- `--output` Excel file with pairwise metric scores for all model pairs
- PNG visualizations per pair (if `--make_figures`): radar plots, k-NN histograms, neighborhood plots, summary figure

**Example:**
```bash
python tile_embedding_comparisons.py \
    --model_dirs /path/to/features_conch /path/to/features_virchow2 /path/to/features_gigapath \
    --output tile_comparison_results.xlsx \
    --metrics cka svcca knn \
    --sample_size 50000 \
    --make_figures
```

---

### slide_embedding_comparisons.py
**Purpose:** Compare slide-level embeddings across multiple foundation models using the same suite of metrics, with bootstrap-derived visualizations.

**Inputs:**
- `--model_dirs` (paths, **required**, nargs=`+`): Directories with per-slide H5 feature files
- `--output` (path, optional, default=`"slide_comparison.xlsx"`): Output Excel file path
- `--metrics` (str, nargs=`+`, optional, default=`["all"]`): Metrics to compute — choices: `cka`, `svcca`, `procrustes`, `knn`, `ridge`, or `all`
- `--pca_dim` (int, optional, default=`50`): PCA dimensions for Procrustes
- `--var_thres` (float, optional, default=`0.99`): Variance threshold for SVCCA
- `--knn_k` (int, optional, default=`10`): Neighbors for k-NN overlap
- `--ridge_alpha` (float, optional, default=`1.0`): Ridge regression regularization
- `--make_figures` (flag, optional): Generate per-pair and summary plots
- File: Per-slide H5 files (one feature vector of shape `(D,)` per slide), matched by filename across models

**Outputs:**
- `--output` Excel file with pairwise metric scores
- PNG visualizations (if `--make_figures`): PCA scatter plots, radar plots, metric heatmaps

**Example:**
```bash
python slide_embedding_comparisons.py \
    --model_dirs /path/to/slide_feats_conch /path/to/slide_feats_virchow2 \
    --output slide_comparison.xlsx \
    --make_figures
```

---

### analyze_institution_clustering.py
**Purpose:** Quantify how strongly tile embeddings cluster by TCGA institution of origin (a proxy for domain shift), using silhouette score, Davies-Bouldin index, and bootstrapped statistical comparisons between models.

**Inputs:**
- `--h5_root` (paths, **required**, nargs=`+`): Root directories with H5 feature files, one per model
- `--svs_root` (path, **required**): Directory containing WSI (`.svs`) files
- `--annotation_dir` (path, **required**): Directory containing tumor/normal annotation XMLs
- `--output_dir` (path, **required**): Output directory for CSVs and plots
- `--data_type` (str, optional, default=`"kidney"`, choices: `kidney`, `prostate`, `rectal`): Annotation format to use
- `--sample_fraction` (float, optional, default=`0.1`): Fraction of tiles per tissue group to sample per slide
- `--max_slides` (int, optional): Maximum number of slides to process
- `--seed` (int, optional, default=`42`): Random seed
- `--n_bootstrap` (int, optional, default=`50`): Number of bootstrap iterations
- `--bootstrap_fraction` (float, optional, default=`0.5`): Fraction of data to sample per bootstrap iteration
- `--pca_components` (int, optional, default=`50`): Number of PCA dimensions before computing clustering metrics
- `--min_slides_per_institution` (int, optional, default=`5`): Minimum slides required for an institution to be included
- `--all_tissues` (flag, optional): Also run combined tumor+normal analysis in addition to per-tissue analysis
- File: H5 files with `features` and `coords` datasets; XML annotations; SVS slides for institution code extraction

**Outputs:**
- `institution_clustering_metrics_bootstrap.csv` — silhouette score, Davies-Bouldin index, per-institution compactness for 50 bootstrap samples × model × tissue type
- `institution_clustering_wilcoxon.csv` — pairwise Wilcoxon statistical comparisons between models
- `institution_clustering_barplot.png` — bar plots of clustering metrics per model
- `institution_clustering_metrics_barplot.png` — extended bar plots including compactness
- `*_alltissues` variants of the above (if `--all_tissues`)

**Example:**
```bash
python analyze_institution_clustering.py \
    --h5_root /path/features_conch /path/features_virchow2 \
    --svs_root /path/to/wsi \
    --annotation_dir /path/to/annotations \
    --output_dir /path/to/output \
    --data_type kidney \
    --n_bootstrap 50 \
    --all_tissues
```
