# Group 3 — Attention Map Extraction and Analysis

Scripts for pulling attention maps out of foundation models, comparing them across models, and studying their spatial properties and overlap.

---

### extract_attention_maps.py
**Purpose:** Extract ViT attention heatmaps from the top-scoring tiles of multiple pathology foundation models via TRIDENT.

**Inputs:**
- `--slides` / `-s` (str, **required**): Directory containing whole-slide images
- `--h5s` / `-a` (str, **required**): Directory containing HDF5 attention files (one per slide)
- `--output` / `-o` (str, **required**): Output directory root
- `--models` / `-m` (str, nargs=`+`, **required**): One or more model names — choices: `musk`, `conch_v15`, `virchow2`, `hoptimus1`, `gigapath`
- `--tile-size` (int, optional, default=`224`): Tile size in pixels
- `--top-k` (int, optional, default=`100`): Number of top-attention tiles to process per slide
- `--device` (str, optional, default=`"cpu"`): Torch device
- `--precision` (str, optional, default=`None`): Override precision — `float32`, `float16`, or `bfloat16`
- `--overwrite` (flag, optional): Overwrite existing output files
- `--simple` (flag, optional): Use single-layer extraction instead of multi-layer rollout
- `--layer-index` (int, optional, default=`-1`): Layer index for `--simple` mode
- `--head-index` (int, optional, default=`None`): Attention head index for `--simple` mode

**Outputs:**
- `{output}/Original/{tile_name}.png` — original extracted tile images
- `{output}/{model}/combined/{tile_name}.png` — multi-layer rollout attention maps (default mode)
- `{output}/{model}/first_layer/{tile_name}.png` — first layer attention only (rollout mode)
- `{output}/{model}/simple/{tile_name}_L{idx}_H{head}.png` — single-layer maps (`--simple` mode)

**Example:**
```bash
python extract_attention_maps.py \
    --slides /path/to/wsi \
    --h5s /path/to/attention_h5s \
    --output /path/to/output \
    --models conch_v15 virchow2 hoptimus1 \
    --top-k 100 \
    --device cuda
```

---

### extract_multires_attention_maps.py
**Purpose:** Extract attention maps at multiple magnifications (e.g., 5x/10x/20x) for the same physical tissue area, with optional random tissue sampling and overlay generation.

**Inputs:**
- `--slides` / `-s` (str, **required**): Directory containing WSIs
- `--h5s` / `-a` (str, optional): Directory with attention H5 files — required unless `--random-from-tissue` is set
- `--output` / `-o` (str, **required**): Output directory root
- `--models` / `-m` (str, nargs=`+`, **required**): Model names to process
- `--top-k` (int, optional, default=`100`): Top tiles per slide (H5-driven mode)
- `--device` (str, optional, default=`"cpu"`): Torch device
- `--precision` (str, optional, default=`None`): Override precision
- `--overwrite` (flag, optional): Overwrite existing files
- `--simple` (flag, optional): Single-layer extraction mode
- `--layer-index` (int, optional, default=`-1`): Layer index for simple mode
- `--head-index` (int, optional, default=`None`): Head index for simple mode
- `--base-tile-size` (int, optional, default=`224`): Base tile size from H5 coordinates
- `--magnifications` (int, nargs=`+`, optional, default=`[5, 10, 20]`): Target magnifications to extract
- `--sizes` (int, nargs=`+`, optional, default=`[256, 512, 1024]`): Output patch sizes per magnification (must match length of `--magnifications`)
- `--percentiles` (float, nargs=`+`, optional, default=`[50.0, 70.0, 90.0]`): Percentile thresholds for thresholded overlays
- `--random-from-tissue` (flag, optional): Sample random patches from tissue mask instead of using H5
- `--tiles-per-slide` (int, optional, default=`5`): Patches per slide in random mode
- `--seed` (int, optional, default=`42`): Random seed
- `--overlay-blur-sigma` (float, optional, default=`20.0`): Gaussian blur sigma for overlays
- `--overlay-alpha` (float, optional, default=`0.4`): Overlay transparency
- `--attention-normalize` (str, optional, default=`"zscore"`): Normalization method — `zscore` or `minmax`
- `--max-slides` (int, optional, default=`5`): Maximum number of slides to process

**Outputs:**
- `{output}/Original/{mag}x/{tile}.png` — original patches at each magnification
- `{output}/{model}/{mag}x/combined/{tile}.png` — combined attention overlays
- `{output}/{model}/{mag}x/first_layer/{tile}.png` — first-layer attention overlays
- `{output}/{model}/{mag}x/threshold_p{percentile}/{tile}.png` — percentile-thresholded overlays

**Example:**
```bash
python extract_multires_attention_maps.py \
    --slides /path/to/wsi \
    --h5s /path/to/attn_h5s \
    --output /path/to/output \
    --models conch_v15 virchow2 \
    --magnifications 5 10 20 \
    --sizes 256 512 1024 \
    --max-slides 20
```

---

### attention_from_h5.py
**Purpose:** Extract per-layer attention maps from an H5 file of embedded tile images using transformer monkey-patching.

**Inputs:**
- `h5_path` (str, hardcoded in `__main__`): Path to H5 file with an `images` or `imgs` dataset
- `out_dir` (str, hardcoded, default=`"attn_maps"`): Output directory
- `device` (str, hardcoded, default=`"cuda"`): Torch device
- `max_tiles` (int, hardcoded, default=`50`): Maximum number of tiles to process
- `use_hoptimus` (bool, hardcoded, default=`True`): Use H-optimus1 (`True`) or CONCH (`False`)
- `layer_idx` (int, hardcoded, default=`-1`): Layer index to extract
- `head_idx` (int, hardcoded, default=`None`): Attention head index (`None` = average all heads)

> **Note:** No CLI arguments. Edit the variables under `if __name__ == "__main__":` at the bottom of the file before running.

**Outputs:**
- `{out_dir}/tile{idx}_L{layer_idx}_H{head_idx|avg}.png` — attention overlays on each tile

**Example:**
```bash
# Edit h5_path, out_dir, and parameters in the __main__ block, then:
python attention_from_h5.py
```

---

### analyze_attention_maps.py
**Purpose:** Compute statistics (range, mean, std, histograms) across model attention maps stored in a combined output directory.

**Inputs:**
- `--output_dir` (str, **required**): Output directory containing model subfolders (e.g., produced by `extract_attention_maps.py`)
- `--models` (str, nargs=`+`, optional, default=`[musk, conch_v15, virchow2, hoptimus1, gigapath]`): Which model subfolders to analyze

**Outputs:**
- Console output: per-model statistics (min/max/mean/std/median, pixel value counts, normalized ranges, text histograms for first tile)

**Example:**
```bash
python analyze_attention_maps.py \
    --output_dir /path/to/attention_output \
    --models conch_v15 virchow2 hoptimus1
```

---

### compute_attention_overlap.py
**Purpose:** Compute Dice score overlaps between binary attention maps from all model pairs, with optional blurring and percentile thresholding.

**Inputs:**
- `--parent-folder` / `-p` (str, **required**): Parent directory with model subfolders and an `Original/` folder
- `--threshold` (float, optional, default=`0.5`): Fixed threshold for binarizing attention maps
- `--percentile-mode` (flag, optional): Use percentile-based thresholding instead of a fixed threshold
- `--percentiles` (float, nargs=`+`, optional, default=`[50.0, 60.0, 70.0, 80.0, 90.0]`): Percentiles to test (used with `--percentile-mode`)
- `--num-examples` (int, optional, default=`5`): Number of example overlay images to generate per model
- `--colormap` (str, optional, default=`"hot"`): Matplotlib colormap for overlay images
- `--test` (flag, optional): Test mode — process only the first 1000 tiles
- `--blur` (flag, optional): Apply blurring to attention maps before thresholding
- `--blur-method` (str, optional, default=`"gaussian"`): Blur type — `gaussian`, `uniform`, or `median`
- `--blur-strength` (float, optional, default=`1.0`): Blur sigma/strength
- `--simple` (flag, optional): Search `simple/` subdirectories instead of `combined/`

**Outputs:**
- `{parent}/attention_overlap_dice_scores.csv` (fixed mode) or `_percentile.csv` (percentile mode) — Dice scores with mean/std/min/max per pair
- `{parent}/attention_overlap_heatmap.png` or `_percentile.png` — pairwise Dice heatmap
- `{parent}/{model}_overlay_example_{1-5}.png` — example overlay images

**Example:**
```bash
python compute_attention_overlap.py \
    --parent-folder /path/to/attention_output \
    --percentile-mode \
    --percentiles 50 70 90 \
    --blur --blur-method gaussian --blur-strength 2.0
```

---

### test_blur_attention.py
**Purpose:** Test different blurring methods applied to the first tile from each model and save comparison grids.

**Inputs:**
- `--parent-folder` / `-p` (str, **required**): Parent directory containing model subfolders with attention maps

**Outputs:**
- `{parent}/blur_test/{model}_{method}.png` — individual blurred attention maps
- `{parent}/blur_test/{model}_blur_comparison.png` — side-by-side comparison grid (original + 6 blurred variants)

**Example:**
```bash
python test_blur_attention.py \
    --parent-folder /path/to/attention_output
```
