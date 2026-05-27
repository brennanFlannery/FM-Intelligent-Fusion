# Cancer Pathology Pipeline — Script Library

Organized reference of all pipeline Python scripts across Kidney, Prostate, and Rectal cancer pathology projects, plus the shared CLAM/TRIDENT infrastructure. Files are copied from their source directories; originals remain in place.

## Dependencies

**Feature extraction** was performed using [TRIDENT](https://github.com/mahmoodlab/TRIDENT), an open-source whole-slide image processing framework available on GitHub. TRIDENT handles patch extraction, tissue segmentation, and embedding generation from foundation models.

**CLAM model training and evaluation** scripts in this library are designed to work in conjunction with the [CLAM repository](https://github.com/mahmoodlab/CLAM), also available on GitHub. The CLAM repo provides the core MIL model definitions, dataset modules, and utilities that these scripts depend on — clone it alongside this library and ensure it is on your Python path before running anything in Groups 8–12.

---

**Source directories:**
- `KidneyCancerPathology/`
- `ProstateCancerPathology/`
- `RectalCancerPathology/`
- `TRIDENT/CLAM/`
- `TRIDENT/CLAM/CLAM/`

---

## Group 1 — Foundation Model Inspection and Testing
`01_Foundation_Model_Inspection_and_Testing/`

Scripts for loading foundation models, inspecting their internals, and smoke-testing attention extraction. Mostly one-off debugging and validation tools.

| Script | Source | Description |
|---|---|---|
| `test_single_model.py` | KidneyCancerPathology | Tests attention extraction on a single tile for one model |
| `test_all_models.py` | KidneyCancerPathology | Batch-tests all five foundation models (MUSK, CONCH, Virchow2, H-optimus, GigaPath) |
| `simple_encoder_test.py` | KidneyCancerPathology | Duplicates TRIDENT's `encoder_factory` call for each model to verify loading |
| `simple_forward_methods.py` | KidneyCancerPathology | Prints the source code of attention block forward methods |
| `inspect_all_attention_methods.py` | KidneyCancerPathology | Inspects and compares ViT attention forward methods across all models |
| `quick_conch_gradcam_smoke.py` | TRIDENT/CLAM/CLAM | Quick smoke test for the CONCH Grad-CAM pipeline |

---

## Group 3 — Attention Map Extraction and Analysis
`03_Attention_Map_Extraction_and_Analysis/`

Scripts for pulling attention maps out of foundation models, comparing them across models, and studying their spatial properties.

| Script | Source | Description |
|---|---|---|
| `extract_attention_maps.py` | KidneyCancerPathology | Extracts ViT attention heatmaps from all five models; saves tiles and maps |
| `extract_multires_attention_maps.py` | KidneyCancerPathology | Extracts attention at multiple magnifications (5x/10x/20x) for the same physical area |
| `attention_from_h5.py` | KidneyCancerPathology | Extracts attention maps directly from H5 tile files via transformer monkey-patching |
| `analyze_attention_maps.py` | KidneyCancerPathology | Computes statistics (range, mean, std, histograms) across model attention maps |
| `compute_attention_overlap.py` | KidneyCancerPathology | Computes Dice score overlap between attention maps from pairs of models |
| `test_blur_attention.py` | KidneyCancerPathology | Tests and visualizes different blurring approaches applied to attention maps |

---

## Group 4 — Attention Map Visualization
`04_Attention_Map_Visualization/`

Turning attention maps into publication-ready figures, overlays, and grids.

| Script | Source | Description |
|---|---|---|
| `attention_overlap_graphs.py` | KidneyCancerPathology | Plots pairwise attention map overlap comparisons between models |
| `overlay_tile_attention.py` | KidneyCancerPathology | Overlays attention heatmaps onto original tile images |
| `create_heatmaps.py` | TRIDENT/CLAM/CLAM | Creates attention heatmaps highlighting the top tiles per prediction |
| `create_slide_attention_maps.py` | TRIDENT/CLAM/CLAM | Creates slide-level attention maps with Gaussian blur and percentile thresholding |
| `create_attention_map_grids.py` | TRIDENT/CLAM/CLAM | Grid visualizations of attention maps across multiple slides |
| `save_per_feature_gradcam.py` | TRIDENT/CLAM/CLAM | Saves per-feature Grad-CAM visualizations |

---

## Group 5 — Embedding Comparison and Institutional Bias
`05_Embedding_Comparison_and_Institutional_Bias/`

Comparing tile and slide embeddings across foundation models, and quantifying how much institution identity leaks into embedding space.

| Script | Source | Description |
|---|---|---|
| `tile_embedding_comparisons.py` | KidneyCancerPathology | Compares tile embeddings across models via PCA, CCA, nearest-neighbor, and correlation |
| `slide_embedding_comparisons.py` | KidneyCancerPathology | Same comparisons at the slide level |
| `analyze_institution_clustering.py` | KidneyCancerPathology | Measures how much tile embeddings cluster by TCGA institution using silhouette score and Davies-Bouldin index |

---

## Group 6 — Feature Engineering and Pruning
`06_Feature_Engineering_and_Pruning/`

Combining multiple feature sets and pruning redundant dimensions before model training.

| Script | Source | Description |
|---|---|---|
| `create_combined_fm_embeddings.py` | KidneyCancerPathology | Merges multiple FM feature sets into unified H5 files with two-stage correlation-based pruning |
| `create_combined_slide_fm_embeddings.py` | KidneyCancerPathology | Slide-level version of the above |
| `corr_prunning.py` | KidneyCancerPathology | Greedy uncorrelated feature selection using p-values and correlation thresholds |

---

## Group 7 — Dimensionality Reduction and t-SNE
`07_Dimensionality_Reduction_and_TSNE/`

Visualizing how tiles and slides distribute in embedding space, with a focus on tumor vs. normal tissue separation.

| Script | Source | Description |
|---|---|---|
| `create_tumor_normal_tsne.py` | KidneyCancerPathology | Per-slide t-SNE of tumor vs. normal tiles with silhouette scoring and grid mosaics |
| `create_aggregated_tumor_normal_tsne.py` | KidneyCancerPathology | Cross-slide aggregated t-SNE with bootstrapped metrics and Wilcoxon comparisons |
| `create_path_tsne.py` | KidneyCancerPathology | Gridified t-SNE mosaic of pathology tiles with optional K-means clustering |

---

## Group 8 — Dataset Preparation and Label Creation
`08_Dataset_Preparation_and_Label_Creation/`

Converting raw clinical files and annotations into CLAM-compatible CSVs, splits, and label sheets.

| Script | Source | Description |
|---|---|---|
| `create_clam_csv.py` | TRIDENT/CLAM | Converts clinical data and H5 bags into a CLAM-compatible CSV |
| `make_clam_splits.py` | TRIDENT/CLAM | Creates k-fold patient-stratified splits preventing patient leakage across folds |
| `create_splits_seq.py` | TRIDENT/CLAM/CLAM | Sequential split creation within CLAM |
| `create_clam_labels.py` | ProstateCancerPathology | Maps Gleason grades from XLSX to a CLAM label sheet (prostate-specific) |
| `create_survival_labels.py` | KidneyCancerPathology | Converts grade-based labels to survival labels by extracting time-to-event data from master Excel |
| `analyze_split_validation.py` | KidneyCancerPathology | Validates that all split patients have complete vital status and follow-up data |
| `analyze_annotation_colors.py` | ProstateCancerPathology | Parses XML annotation files and extracts unique annotation colors (Windows COLORREF to RGB) |

---

## Group 9 — CLAM Model Training and Evaluation
`09_CLAM_Model_Training_and_Evaluation/`

Core multiple-instance learning training loop and model ensembling. (`eval.py` and `build_preset.py` excluded.)

| Script | Source | Description |
|---|---|---|
| `main.py` | TRIDENT/CLAM/CLAM | Main CLAM training script with full CLI for MIL classification |
| `main_stability.py` | TRIDENT/CLAM/CLAM | Stability-focused CLAM training variant |
| `main_subclams.py` | TRIDENT/CLAM/CLAM | Hierarchical SubCLAM training variant |
| `ensemble_clam.py` | TRIDENT/CLAM/CLAM | Ensembles predictions across multiple CLAM models |
| `ensemble_slide_mlps.py` | KidneyCancerPathology | Ensembles slide-level MLP predictions via averaged probability tie-breaking |

---

## Group 10 — Metrics and Performance Comparison
`10_Metrics_and_Performance_Comparison/`

Computing, aggregating, and statistically comparing classification and survival metrics across models.

| Script | Source | Description |
|---|---|---|
| `compute_clam_metrics.py` | TRIDENT/CLAM/CLAM | Computes accuracy, F1, AUC, and confusion matrix for CLAM predictions |
| `slide_level_metrics.py` | TRIDENT/CLAM/CLAM | Computes slide-level performance metrics |
| `compare_model_performance.py` | TRIDENT/CLAM/CLAM | Statistical comparison of tile-level performance across models |
| `compare_slide_model_performance.py` | TRIDENT/CLAM/CLAM | Same comparison at the slide level |
| `aggregate_dice_results.py` | TRIDENT/CLAM/CLAM | Aggregates Dice and coverage results across models and percentile thresholds |
| `aggregated_ranking_plot.py` | TRIDENT/CLAM/CLAM | Creates model ranking visualizations across metrics |
| `heatmap_tumor_dice.py` | TRIDENT/CLAM/CLAM | Computes Dice overlap between attention maps and annotated tumor regions |
| `analyze_clam_survival.py` | KidneyCancerPathology | Kaplan-Meier survival analysis using CLAM grade predictions with log-rank tests |

---

## Group 11 — SHAP and Feature Importance
`11_SHAP_and_Feature_Importance/`

Interpreting which features and tiles drive model decisions using SHAP values.

| Script | Source | Description |
|---|---|---|
| `compute_shap_values.py` | TRIDENT/CLAM/CLAM | Tile-level SHAP value computation |
| `compute_shap_values_slide.py` | TRIDENT/CLAM/CLAM | Slide-level SHAP value computation |
| `visualize_feature_selection_shap.py` | KidneyCancerPathology | Publication-quality SHAP feature selection figures across diseases and thresholds |
| `visualize_feature_selection_slide.py` | KidneyCancerPathology | Slide-level feature selection visualization |

---

## Group 12 — Pruning and Stability
`12_Pruning_and_Stability/`

Feature pruning analysis and stability filtering after training.

| Script | Source | Description |
|---|---|---|
| `pruning_visualization.py` | TRIDENT/CLAM/CLAM | Visualizes tile-level feature pruning results |
| `pruning_visualization_slide.py` | TRIDENT/CLAM/CLAM | Slide-level feature pruning visualization |
| `stability_filters.py` | TRIDENT/CLAM/CLAM | Filters features based on stability metrics across runs |
| `create_cluster_cache.py` | TRIDENT/CLAM/CLAM | Pre-computes cluster assignments to speed up training |

---

## Group 13 — Tissue Coverage Visualization
`13_Tissue_Coverage_Visualization/`

Disease-specific scripts for visualizing how well model attention covers the right tissue types. Files are renamed to include the cancer type since two share the same original filename.

| Script | Source | Description |
|---|---|---|
| `visualize_tumor_normal_coverage_kidney.py` | KidneyCancerPathology | Attention coverage in tumor vs. normal regions — line plots and ranking bump charts (kidney) |
| `visualize_tumor_normal_coverage_prostate.py` | ProstateCancerPathology | Prostate-specific version of the above |
| `visualize_sra_coverage.py` | RectalCancerPathology | SRA tissue classification coverage for rectal cancer with heatmaps and bump charts |

---

## Excluded Groups

| Group | Reason |
|---|---|
| Group 2 — Feature & Patch Extraction | Excluded by request |
| Group 14 — Utilities | Excluded by request |
