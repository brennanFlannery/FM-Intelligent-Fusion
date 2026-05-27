# Group 4 — Attention Map Visualization

Scripts for turning attention maps into publication-ready figures, overlays, and grids.

---

### attention_overlap_graphs.py
**Purpose:** Generate upper-triangle pairwise Dice heatmaps from a CSV of model-pair overlap scores, one figure per percentile.

**Inputs:**
- `--csv` (str, **required**): CSV file with columns `Model1`, `Model2`, `Percentile`, `Mean_Dice`
- `--percentiles` (float, nargs=`*`, optional, default=`[50, 70, 90]`): Which percentiles to plot
- `--outdir` (str, optional, default=`"./figs"`): Output directory for PNG figures
- `--title-prefix` (str, optional, default=`"Pairwise Mean Dice"`): Figure title prefix
- `--sort-by` (float, optional, default=`None`): Sort model order by average Dice at this percentile
- `--annotate` (flag, optional): Overlay numeric Dice values inside each cell

**Outputs:**
- `{outdir}/heatmap_mean_dice_{percentile}.png` — upper-triangle heatmap for each requested percentile

**Example:**
```bash
python attention_overlap_graphs.py \
    --csv /path/to/attention_overlap_dice_scores_percentile.csv \
    --percentiles 50 70 90 \
    --outdir ./figs \
    --annotate
```

---

### overlay_tile_attention.py
**Purpose:** Overlay attention maps onto original tile images using Gaussian smoothing, percentile thresholding, and JET colormap blending.

**Inputs:**
- `--input_dir` (str, **required**): Directory containing `Original/` and `{model}/simple/` subdirectories
- `--output_dir` (str, **required**): Output directory for overlaid images
- `--blur_sigma` (float, optional, default=`20.0`): Gaussian blur sigma applied to attention maps
- `--alpha` (float, optional, default=`0.4`): Transparency of the attention overlay (0=transparent, 1=opaque)
- `--dpi` (int, optional, default=`400`): Output image DPI
- `--num_tiles` (int, optional, default=`50`): Number of tiles to randomly select per model (`-1` for all)
- `--random_seed` (int, optional, default=`42`): Random seed for tile selection

**Outputs:**
- `{output_dir}/{model}/{original_filename}.png` — attention-overlaid tile images, one per model

**Example:**
```bash
python overlay_tile_attention.py \
    --input_dir /path/to/attention_output \
    --output_dir /path/to/overlays \
    --blur_sigma 20.0 \
    --alpha 0.4 \
    --num_tiles 100
```

---

### create_heatmaps.py
**Purpose:** Legacy CLAM heatmap inference script — generates WSI-level attention heatmaps driven by a YAML config file.

**Inputs:**
- `--save_exp_code` (str, optional, default=`None`): Override for experiment code
- `--overlap` (float, optional, default=`None`): Override patch overlap
- `--config_file` (str, optional, default=`"heatmap_config_template.yaml"`): Path to YAML config specifying all remaining parameters (model paths, slide dirs, patch settings, etc.)

> **Note:** Most configuration (slide directory, model path, patch size, etc.) is driven by the YAML config file rather than CLI flags.

**Outputs:**
- Per-slide heatmap PNGs in production and raw subdirectories
- CSV file with per-slide predictions and attention scores
- Optional annotation overlays on heatmap images

**Example:**
```bash
python create_heatmaps.py \
    --config_file heatmaps/configs/kidney_heatmap_config.yaml \
    --save_exp_code kidney_heatmap_v1
```

---

### create_slide_attention_maps.py
**Purpose:** Generate attention heatmaps overlaid on WSI thumbnails with optional tumor/normal/SRA tissue annotation outlines, supporting kidney, prostate, and rectal formats.

**Inputs:**
- `--slide_dir` (str, **required**): Directory containing `.svs` slides
- `--attention_dir` (str, **required**): Directory with CLAM `.h5` attention files (one per slide)
- `--annotation_dir` (str, **required**): Directory with annotation XML files
- `--output_dir` (str, **required**): Output directory for PNG heatmaps
- `--data_type` (str, optional, default=`"kidney"`): Dataset annotation format — choices: `kidney`, `prostate`, `prostate_colored`, `rectal`, `rectal_sra`
- `--num_slides` (int, optional, default=`10`): Number of slides to process (`-1` for all)
- `--target_dim` (int, optional, default=`4000`): Longest dimension of the output thumbnail in pixels
- `--tile_size` (int, **required**): Tile size in pixels at the extraction magnification
- `--blur_sigma` (float, optional, default=`1.0`): Gaussian blur sigma for attention smoothing
- `--outline_offset` (float, optional, default=`3.0`): Inward pixel offset for annotation polygon outlines
- `--sra_dataset` (str, optional, default=`"kather19crctp"`): SRA color mapping — `kather19` or `kather19crctp` (only used with `rectal_sra`)

**Outputs:**
- `{output_dir}/{slide_id}_attention.png` — slide thumbnail with attention heatmap overlay and annotation outlines
- `{output_dir}/{slide_id}_tumor_normal.png` — SRA tumor/normal region map (`rectal_sra` only)
- `{output_dir}/{slide_id}_other_tissues.png` — SRA other tissue class map (`rectal_sra` only)

**Example:**
```bash
python create_slide_attention_maps.py \
    --slide_dir /path/to/wsi \
    --attention_dir /path/to/clam_attentions \
    --annotation_dir /path/to/annotations \
    --output_dir /path/to/heatmap_output \
    --data_type kidney \
    --tile_size 224 \
    --num_slides 50 \
    --blur_sigma 1.0
```

---

### create_attention_map_grids.py
**Purpose:** Create 3×3 grid figures matching attention map images across 7 model folders (6 non-pruned models in the top rows + 1 pruned model centered in the bottom row).

**Inputs:**
- `--input_dir` (Path, **required**): Root directory containing exactly 7 subfolders: `CONCH_V15`, `GIGAPATH`, `HOPTIMUS1`, `MUSK`, `NAIVE`, `VIRCHOW2`, `PRUNED`
- `--output_dir` (Path, optional, default=`{input_dir}/grids`): Output directory for grid figures
- `--pattern` (str, optional, default=`None`): Glob pattern to filter image files — defaults to processing `*_attention.png`, `*_tumor_normal.png`, and `*_other_tissues.png`
- `--dpi` (int, optional, default=`300`): Output image DPI

**Outputs:**
- `{output_dir}/{stem}_grid.png` — one 3×3 grid PNG per matched slide/image stem

**Example:**
```bash
python create_attention_map_grids.py \
    --input_dir /path/to/AttentionMaps \
    --output_dir /path/to/AttentionMaps/grids \
    --pattern "*_attention.png" \
    --dpi 300
```

---

### save_per_feature_gradcam.py
**Purpose:** Compute per-feature token Grad-CAM maps for all encoder output dimensions across the top-100 tiles of each slide, using TRIDENT's `WSIPatcher`.

**Inputs:**
- `--slides_dir` (Path, **required**): Directory with WSI files
- `--tiles_dir` (Path, **required**): Directory with coords-only H5 files (one per slide)
- `--attn_dir` (Path, **required**): Directory with attention H5 files (one per slide)
- `--out_dir` (Path, **required**): Output directory root for `.npy` Grad-CAM arrays
- `--models` (str, optional, default=`"conch_v15"`): Comma-separated model names (e.g., `"conch_v15,virchow2"`)
- `--mag` (int, **required**): Target magnification (e.g., `20`)
- `--num_slides` (int, optional, default=`-1`): Number of slides to process (`-1` for all)
- `--tile_batch_size` (int, optional, default=`4`): Batch size for tile processing
- `--feature_chunk` (int, optional, default=`32`): Number of features to compute gradients for at once
- `--tile_size` (int, **required**): Output tile size in pixels
- `--device` (str, optional, default=`"cuda:0"`): Torch device
- `--precision` (str, optional, default=`"fp32"`): Forward precision — `fp32`, `bf16`, or `fp16`
- `--seed` (int, optional, default=`42`): Random seed

**Outputs:**
- `{out_dir}/{slide_id}/{x_y}/gradcam_{model}_{feature_idx}.npy` — float32 Grad-CAM array of shape `(tile_size, tile_size)` for each tile position and feature dimension

**Example:**
```bash
python save_per_feature_gradcam.py \
    --slides_dir /path/to/wsi \
    --tiles_dir /path/to/coords_h5 \
    --attn_dir /path/to/attn_h5 \
    --out_dir /path/to/gradcam_output \
    --models conch_v15,virchow2 \
    --mag 20 \
    --tile_size 512 \
    --feature_chunk 64 \
    --device cuda:0
```
