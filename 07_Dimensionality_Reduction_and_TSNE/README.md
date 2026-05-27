# Group 7 — Dimensionality Reduction and t-SNE

Scripts for visualizing how tiles and slides distribute in embedding space, with a focus on tumor vs. normal tissue separation.

---

### create_tumor_normal_tsne.py
**Purpose:** Create per-slide t-SNE visualizations of tumor vs. normal tiles with a grid-based mosaic layout and silhouette scoring, one figure per slide per embedding.

**Inputs:**
- `--h5_root` (paths, **required**, nargs=`+`): Directories with per-slide H5 feature files, one per embedding/model
- `--svs_root` (path, **required**): Directory containing WSI (`.svs`) files
- `--annotation_dir` (path, **required**): Directory with tumor/normal annotation XMLs
- `--output_dir` (path, **required**): Root output directory
- `--max_slides` (int, optional): Limit on number of slides to process
- `--grid_rows` (int, optional, default=`60`): Mosaic grid rows (ignored if `--auto_grid`)
- `--grid_cols` (int, optional, default=`60`): Mosaic grid columns (ignored if `--auto_grid`)
- `--auto_grid` (flag, optional): Automatically size the grid as a square based on tile count
- `--grid_method` (str, optional, default=`"greedy"`, choices: `greedy`, `direct`): Tile-to-grid assignment algorithm
- `--mosaic_tile_px` (int, optional, default=`64`): Downsampled tile size in the output mosaic (pixels)
- `--border_width` (int, optional, default=`3`): Border width in pixels around each grid cell
- `--seed` (int, optional, default=`42`): Random seed
- `--tsne_perplexity` (float, optional, default=`30.0`): t-SNE perplexity
- `--tsne_iter` (int, optional, default=`2000`): t-SNE iterations
- File: H5 files with `features` and `coords` datasets; XML annotations; SVS slide files

**Outputs:**
- `{output_dir}/{embedding}/{slide}.png` — t-SNE mosaic per slide per embedding, with tumor/normal coloring
- `{output_dir}/{embedding}/{slide}_occupancy_preselection.png` — pre-assignment grid occupancy
- `{output_dir}/{embedding}/{slide}_occupancy_postselection.png` — post-assignment grid occupancy
- `{output_dir}/thumbnails/{slide}.png` — annotated slide thumbnail
- `{output_dir}/summary_silhouette.csv` — silhouette scores (2D and high-dim) per slide/embedding

**Example:**
```bash
python create_tumor_normal_tsne.py \
    --h5_root /path/features_conch /path/features_virchow2 \
    --svs_root /path/to/wsi \
    --annotation_dir /path/to/annotations \
    --output_dir /path/to/tsne_output \
    --max_slides 50 \
    --auto_grid \
    --grid_method direct \
    --tsne_perplexity 30 \
    --tsne_iter 2000
```

---

### create_aggregated_tumor_normal_tsne.py
**Purpose:** Aggregate tiles across all slides and create a single t-SNE per embedding with bootstrapped separation metrics and Wilcoxon statistical comparisons between models.

**Inputs:**
- `--h5_root` (paths, **required**, nargs=`+`): Multiple embedding directories
- `--svs_root` (path, **required**): WSI directory
- `--annotation_dir` (path, **required**): Annotation directory
- `--output_dir` (path, **required**): Output directory
- `--data_type` (str, optional, default=`"kidney"`, choices: `kidney`, `prostate`, `rectal`, `rectal_sra`): Annotation format
- `--sra_dataset` (str, optional, default=`"kather19crctp"`): SRA dataset name (only for `rectal_sra`)
- `--max_slides` (int, optional): Maximum number of slides to include
- `--sample_fraction` (float, optional, default=`0.1`): Fraction of tiles per tissue group per slide to sample
- `--grid_rows` (int, optional): Mosaic grid rows
- `--grid_cols` (int, optional): Mosaic grid columns
- `--auto_grid` (flag, optional): Auto-size grid
- `--grid_method` (str, optional, default=`"direct"`, choices: `direct`, `greedy`): Assignment method
- `--seed` (int, optional, default=`42`): Random seed
- `--tsne_perplexity` (float, optional, default=`30.0`): t-SNE perplexity
- `--tsne_iter` (int, optional, default=`2000`): t-SNE iterations
- File: H5 files, SVS files, XML or NPY annotations

**Outputs:**
- `{output_dir}/{embedding}_aggregated.png` — aggregated t-SNE mosaic per embedding with tissue coloring
- `{output_dir}/global_metrics_bootstrap.csv` — bootstrapped silhouette and compactness metrics (50 iterations × model)
- `{output_dir}/model_comparison_wilcoxon.csv` — pairwise Wilcoxon rank-sum tests between models
- `{output_dir}/combined_metrics_boxplot.png` — box plots of metrics per model
- `{output_dir}/combined_metrics_barplot.png` — bar plots of metrics per model

**Example:**
```bash
python create_aggregated_tumor_normal_tsne.py \
    --h5_root /path/features_conch /path/features_virchow2 /path/features_gigapath \
    --svs_root /path/to/wsi \
    --annotation_dir /path/to/annotations \
    --output_dir /path/to/agg_tsne \
    --data_type kidney \
    --max_slides 100 \
    --sample_fraction 0.1 \
    --auto_grid
```

---

### create_path_tsne.py
**Purpose:** Create a single gridified t-SNE mosaic aggregating tiles from multiple slides, with optional K-means cluster coloring.

**Inputs:**
- `--h5_root` (path, **required**): Directory with per-slide H5 feature files (single embedding)
- `--svs_root` (path, **required**): WSI directory
- `--output_png` (path, **required**): Output PNG file path
- `--max_slides` (int, optional, default=`30`): Maximum number of slides to include
- `--per_slide_max` (int, optional, default=`5000`): Maximum tiles to sample per slide before the global cap
- `--max_tiles` (int, optional, default=`3600`): Global tile count cap across all slides
- `--coord_bins` (int, nargs=`+`, optional, default=`[32, 32]`): Bin dimensions for location-aware sampling
- `--grid_rows` (int, optional, default=`60`): Mosaic grid rows
- `--grid_cols` (int, optional, default=`60`): Mosaic grid columns
- `--mosaic_tile_px` (int, optional, default=`64`): Tile size in the output mosaic (pixels)
- `--grid_method` (str, optional, default=`"greedy"`, choices: `greedy`, `hungarian`, `direct`): Grid assignment algorithm
- `--seed` (int, optional, default=`42`): Random seed
- `--tsne_perplexity` (float, optional, default=`30.0`): t-SNE perplexity
- `--tsne_iter` (int, optional, default=`2000`): t-SNE iterations
- `--n_clusters` (int, optional, default=`0`): Number of K-means clusters (`0` = no clustering/coloring)
- `--border_width` (int, optional, default=`3`): Border width in pixels around each grid cell
- `--save_tsne_coords` (path, optional): If provided, save t-SNE coordinates and cluster labels to an NPZ file
- File: H5 files with `features` and `coords` datasets; SVS slide files

**Outputs:**
- `--output_png` — single PNG mosaic at 400 DPI, with colored cluster borders if `--n_clusters > 0`
- NPZ file (if `--save_tsne_coords`): t-SNE coordinates and cluster labels

**Example:**
```bash
python create_path_tsne.py \
    --h5_root /path/features_gigapath \
    --svs_root /path/to/wsi \
    --output_png /path/to/output/tsne_grid_gigapath.png \
    --max_slides 200 \
    --max_tiles 3600 \
    --n_clusters 5 \
    --grid_method direct \
    --save_tsne_coords /path/to/output/tsne_coords.npz
```
