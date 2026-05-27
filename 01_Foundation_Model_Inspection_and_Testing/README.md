# Group 1 — Foundation Model Inspection and Testing

Scripts for loading foundation models, inspecting their internals, and smoke-testing attention extraction. Mostly one-off debugging and validation tools.

---

### test_single_model.py
**Purpose:** Debug attention extraction from a single vision transformer model on one slide tile.

**Inputs:**
- `--model` / `-m` (str, **required**): Model name — choices: `musk`, `conch_v15`, `virchow2`, `hoptimus1`, `gigapath`
- `--h5` (str, **required**): Path to an HDF5 file containing attention scores and tile coordinates
- `--slide` (str, **required**): Path to the whole-slide image (`.svs`, `.tif`, etc.)
- `--output` / `-o` (str, **required**): Output directory for saved images
- `--tile-size` (int, optional, default=`224`): Size in pixels of the square tile to extract
- `--device` (str, optional, default=`"cpu"`): Torch device (`cpu` or `cuda`)

**Outputs:**
- `{output}/{model}_original.png` — original extracted tile image
- `{output}/{model}_attention_combined.png` — multi-layer rollout attention map
- `{output}/{model}_attention_first_layer.png` — first layer attention only

**Example:**
```bash
python test_single_model.py \
    --model conch_v15 \
    --h5 /path/to/slide_attentions.h5 \
    --slide /path/to/slide.svs \
    --output ./debug_results \
    --tile-size 224 \
    --device cuda
```

---

### test_all_models.py
**Purpose:** Run attention extraction tests for all five foundation models sequentially on the same slide and H5 pair.

**Inputs:**
- `sys.argv[1]` (positional, **required**): Path to HDF5 file with attention scores and coordinates
- `sys.argv[2]` (positional, **required**): Path to whole-slide image
- `sys.argv[3]` (positional, **required**): Output directory root

**Outputs:**
- One subdirectory per model inside the output root, each containing the same outputs as `test_single_model.py`

**Example:**
```bash
python test_all_models.py \
    /path/to/slide_attentions.h5 \
    /path/to/slide.svs \
    ./test_results
```

---

### simple_encoder_test.py
**Purpose:** Test loading all ViT models via TRIDENT's `encoder_factory` and verify block structure and attention module details.

**Inputs:** None — model list is hardcoded in the script.

**Outputs:**
- Console output showing model type, transformer block count, and attention module details for each model

**Example:**
```bash
python simple_encoder_test.py
```

---

### simple_forward_methods.py
**Purpose:** Print the source code of the `Attention.forward()` method for each foundation model's attention blocks.

**Inputs:** None — model list is hardcoded.

**Outputs:**
- Console output with full Python source code of the forward method per model

**Example:**
```bash
python simple_forward_methods.py
```

---

### inspect_all_attention_methods.py
**Purpose:** Inspect and print the full `Attention` class source code and forward method implementation for all ViT models (excludes MUSK).

**Inputs:** None — model list is hardcoded.

**Outputs:**
- Console output showing the Attention class source and forward method per model

**Example:**
```bash
python inspect_all_attention_methods.py
```

---

### quick_conch_gradcam_smoke.py
**Purpose:** Smoke-test the per-feature token Grad-CAM pipeline on `conch_v15` using TRIDENT's `WSIPatcher` on the top-100 attention tiles.

**Inputs:**
- `--slides_dir` (Path, **required**): Directory of WSI files (`.svs`, `.tif`, `.ndpi`, etc.)
- `--tiles_dir` (Path, **required**): Directory of coords-only H5 files (one per slide)
- `--attn_dir` (Path, **required**): Directory of attention H5 files (one per slide)
- `--out_dir` (Path, **required**): Output directory root for Grad-CAM `.npy` files
- `--tile_size` (int, **required**): Tile size H=W in pixels
- `--mag` (int, **required**): Target magnification (e.g., `20`)
- `--device` (str, optional, default=`"cuda:0"`): Torch device
- `--precision` (str, optional, default=`"fp32"`, choices: `fp32`, `bf16`, `fp16`): Forward precision
- `--tile_batch_size` (int, optional, default=`8`): Batch size for tile processing
- `--seed` (int, optional, default=`42`): Random seed

**Outputs:**
- `.npy` files at `{out_dir}/{slide_id}/{x_y}/gradcam_conch_v15_0.npy` — float32 Grad-CAM arrays for feature 0 of each tile

**Example:**
```bash
python quick_conch_gradcam_smoke.py \
    --slides_dir /path/to/wsi \
    --tiles_dir /path/to/coords_h5 \
    --attn_dir /path/to/attn_h5 \
    --out_dir ./gradcam_output \
    --tile_size 512 \
    --mag 20 \
    --device cuda:0
```
