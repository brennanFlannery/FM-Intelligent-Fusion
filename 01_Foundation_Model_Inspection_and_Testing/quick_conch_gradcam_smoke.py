#!/usr/bin/env python3
"""
Quick smoke test for per-feature token Grad-CAM using TRIDENT's conch_v15 with WSIPatcher.

What this script does
---------------------
• Loads the TRIDENT patch encoder conch_v15.
• Pairs the first coords-only tiles H5 with its matching attention H5.
• Finds the matching WSI in --slides_dir and uses TRIDENT's WSIPatcher (OpenSlide backend)
  to lazily fetch only the needed tiles at --mag.
• Selects the top-100 tiles by attention.
• Computes a single Grad-CAM per tile for feature index 0.
• Saves .npy CAMs using the same structure as the full script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import h5py

# Reuse helpers from the main script
from save_per_feature_gradcam import (
    set_seed,
    match_h5_pairs_strict,
    _read_attention_h5,
    load_trident_encoder,
    _find_token_layer,
    TokenActHook,
    _reduce_cam,
    _infer_grid_from_tokens,
    _upsample,
    make_out_path,
    _find_wsi_for_slide,
)

# TRIDENT patcher (let it create the WSI internally)
from trident import load_wsi  # <-- instead of from trident import WSIPatcher


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Smoke test: Grad-CAM for conch_v15, feature 0, first slide (TRIDENT WSIPatcher, OpenSlide)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--slides_dir", type=Path, required=True, help="Directory of WSIs (e.g., .svs, .tif, .ndpi)")
    ap.add_argument("--tiles_dir", type=Path, required=True, help="Directory of coords-only H5 files (one per slide)")
    ap.add_argument("--attn_dir", type=Path, required=True, help="Directory of attention H5 files (one per slide)")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory root")
    ap.add_argument("--tile_size", type=int, required=True, help="Tile size (H=W) that the patcher must return")
    ap.add_argument("--mag", type=int, required=True, help="Destination magnification for tiles (e.g., 20)")
    ap.add_argument("--device", type=str, default="cuda:0", help="torch device (e.g., cuda:0 or cpu)")
    ap.add_argument("--precision", type=str, choices=["fp32", "bf16", "fp16"], default="fp32",
                    help="Forward precision (CAM math stays fp32)")
    ap.add_argument("--tile_batch_size", type=int, default=8, help="Tile batch size")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # Pair the first slide
    pairs: List[Tuple[Path, Path]] = match_h5_pairs_strict(args.tiles_dir, args.attn_dir)
    tiles_h5, attn_h5 = pairs[0]

    # Read coords & attention; verify coords equality vs tiles H5
    coords_attn, attn = _read_attention_h5(attn_h5)
    with h5py.File(tiles_h5, "r") as f:
        if "coords" not in f:
            raise ValueError(f"Missing 'coords' in {tiles_h5}")
        coords_tiles = f["coords"][...]
    if coords_tiles.shape != coords_attn.shape or not np.array_equal(coords_tiles, coords_attn):
        raise ValueError("Coords mismatch between tiles and attention H5 for first slide.")

    # Compute top-100 tiles by attention first to minimize coords loaded
    n = attn.shape[0]
    k = min(100, n)
    topk_idx_unsorted = np.argpartition(attn, -k)[-k:]
    topk_idx = topk_idx_unsorted[np.argsort(attn[topk_idx_unsorted])[::-1]]
    coords_top = coords_attn[topk_idx].astype(np.int32)

    # Find WSI and build TRIDENT patcher using only top-k coords
    slide_key = attn_h5.stem
    wsi_path = _find_wsi_for_slide(args.slides_dir, slide_key)
    wsi = load_wsi(str(wsi_path), lazy_init=False)
    
    # Core slide info that TRIDENT parses for you
    width, height = wsi.get_dimensions()
    mpp_x = getattr(wsi, "mpp_x", getattr(wsi, "mpp", None))
    mpp_y = getattr(wsi, "mpp_y", getattr(wsi, "mpp", None))
    mag   = getattr(wsi, "mag", None)
    
    # Build a patcher that respects desired output scale for just top-k coords
    patcher = wsi.create_patcher(
        patch_size=int(args.tile_size),
        dst_mag=int(args.mag),          # what you want (e.g., 20)
        src_pixel_size=float(mpp_x),    # avoids the src_mag=None code path
        coords_only=True,               # we're providing explicit coords
        custom_coords=coords_top,
        pil=True,
    )

    # Validate a tile size
    # Validate a tile size (use get_tile_xy when available; fall back to row API)
    try:
        xy0 = coords_top[0]
        test_im = patcher.get_tile_xy(int(xy0[0]), int(xy0[1]))
    except Exception:
        # Some TRIDENT builds expect a dataframe row for get_tile
        row0 = patcher.coords_df.iloc[0]
        test_im = patcher.get_tile(row0)
    if not isinstance(test_im, Image.Image):
        raise RuntimeError("WSIPatcher returned a non-PIL tile despite pil=True.")
    if test_im.size != (args.tile_size, args.tile_size):
        raise ValueError(f"Tile size mismatch: expected {args.tile_size}x{args.tile_size}, got {test_im.size}.")
        raise ValueError(f"Tile size mismatch: expected {args.tile_size}x{args.tile_size}, got {test_im.size}.")

    # (top-k computed above; using coords_top)

    # Load TRIDENT encoder (strict preprocessing enforced)
    encoder_name = "conch_v15"
    bundle = load_trident_encoder(encoder_name, device=device)
    model = bundle.model
    preproc = bundle.preproc

    # Token layer & dry-run for shapes
    cam_layer = _find_token_layer(model)
    x_test = preproc([test_im]).to(device)
    with TokenActHook(cam_layer) as hook:
        z_test = model(x_test)
        acts_test = hook.acts
    if acts_test is None or acts_test.ndim != 3:
        raise RuntimeError("Failed to capture token activations (expected (B,N,C)).")
    if z_test.shape[-1] < 1:
        raise RuntimeError("Model output has zero feature dimension; cannot compute j=0 Grad-CAM.")

    # Process tiles in small batches, computing ONLY feature j=0
    slide_id = slide_key
    tiles_pbar = tqdm(total=len(coords_top), desc=f"{slide_id} · {encoder_name} · tiles", leave=True)

    for t_start in range(0, len(coords_top), args.tile_batch_size):
        idx_batch = np.arange(len(coords_top))[t_start: t_start + args.tile_batch_size]
        coords_batch = coords_top[idx_batch]

        # Fetch a batch of tiles (prefer get_tile_xy; fall back to row-based get_tile)
        xys = coords_top[idx_batch]
        try:
            pil_batch = [patcher.get_tile_xy(int(x), int(y)) for (x, y) in xys]
        except Exception:
            rows = patcher.coords_df.iloc[idx_batch]
            pil_batch = [patcher.get_tile(rows.iloc[j]) for j in range(len(rows))]
        for im in pil_batch:
            if im.size != (args.tile_size, args.tile_size):
                raise ValueError(f"Tile size mismatch in batch: expected {args.tile_size}x{args.tile_size}, got {im.size}")
        x = preproc(pil_batch).to(device)

        # Forward once; target feature j=0
        with TokenActHook(cam_layer) as hook:
            if args.precision == "fp16":
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    z = model(x)
            elif args.precision == "bf16":
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    z = model(x)
            else:
                z = model(x)
            acts = hook.acts
        if acts is None:
            raise RuntimeError("Token activations missing; hook failed")
        acts = acts.detach().requires_grad_(True).to(torch.float32)

        B, Ntok, C = acts.shape
        gh, gw = _infer_grid_from_tokens(Ntok)

        t = z[:, 0].sum()
        (gradA,) = torch.autograd.grad(t, acts, retain_graph=False, allow_unused=False)

        grad_tokens = gradA[:, 1:, :]
        acts_tokens = acts[:, 1:, :]
        cam_tokens = _reduce_cam(grad_tokens, acts_tokens, reduction="grad_times_act")
        cams_grid = cam_tokens.reshape(B, gh, gw)
        cams_up = _upsample(cams_grid, (args.tile_size, args.tile_size))
        cams_up_np = cams_up.detach().cpu().numpy().astype(np.float32)

        for i in range(B):
            out_path = make_out_path(out_root, slide_id, coords_batch[i], encoder_name, 0)
            np.save(out_path, cams_up_np[i])

        del z, acts
        if device.type == "cuda":
            torch.cuda.empty_cache()

        tiles_pbar.update(len(idx_batch))

    tiles_pbar.close()
    print("Done. Saved feature-0 Grad-CAMs to:", str(out_root))


if __name__ == "__main__":
    main()
