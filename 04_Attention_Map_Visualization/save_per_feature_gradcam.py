#!/usr/bin/env python3
"""
Per‑feature token Grad‑CAM for ViT‑based pathology foundation models loaded via TRIDENT.

What this script does
---------------------
• Loads TRIDENT patch encoders (e.g., conch_v15, virchow2, gigapath, hoptimus1, musk).
• Pairs CLAM/TRIDENT *coords H5* (no images) with matched *attention H5* (verifies 1:1 `coords`).
• Locates the corresponding WSI in `--slides_dir` and uses **TRIDENT's OpenSlide-backed WSI + WSIPatcher**
  to lazily fetch tiles **on demand** at the requested magnification.
• Selects the **top‑100 tiles** by attention (hard‑coded by request).
• For each selected tile and for **every feature index j** of the encoder output `z∈R^D`,
  computes a **spatial Grad‑CAM map** by backpropagating from `z[j]` to a token‑level
  activation tensor near the end of the transformer.
• Upsamples CAMs to the **user‑specified tile size** and saves raw `.npy` arrays:

    out_root/
      <slide_id>/
        (x_y)/
          gradcam_<model>_<featureIdx>.npy   # float32 (H, W) where H=W=--tile_size

Notes & constraints
-------------------
• This is **gradient‑based** and therefore expensive: you will perform D backprops per
  tile per model. To keep it tractable, the script supports **feature‑chunking** and
  **small tile batches** and writes CAMs to disk immediately.
• We **fail hard** if models cannot be loaded, attention capture fails, or H5 schemas
  are not aligned (as requested).
• Default precision is **fp32** for stability; you can opt‑in to bf16/fp16 forwards but
  the CAM layer + gradients are kept / cast to fp32 before reduction.
• Downstream comparability: **all tiles must have the same size**, enforced via
  `--tile_size`. We validate this against tiles returned by TRIDENT's patcher.

Example usage
-------------
python save_per_feature_gradcam.py \
  --slides_dir /path/to/wsi \
  --tiles_dir /path/to/coords_h5 \
  --attn_dir /path/to/attn_h5 \
  --out_dir /path/to/out \
  --models conch_v15,virchow2,gigapath,hoptimus1,musk \
  --mag 20 \
  --num_slides 2 \
  --seed 42 \
  --tile_batch_size 4 \
  --feature_chunk 32 \
  --tile_size 512 \
  --device cuda:0

Dependencies
------------
• numpy, h5py, tqdm, pillow, torch
• **TRIDENT** installed and importable (we use encoder_factory and WSI/WSIPatcher).
• **OpenSlide** runtime (TRIDENT OpenSlide backend). No cuCIM required.
• TRIDENT recommends timm==0.9.16 for many backbones.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- TRIDENT imports (patch encoders + WSI) ---
try:
    from trident.patch_encoder_models.load import encoder_factory as patch_encoder_factory  # type: ignore
    from trident import OpenSlideWSI, WSIPatcher  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Failed to import TRIDENT components. Ensure TRIDENT is installed and on PYTHONPATH."
    ) from e

# -----------------------------
# Utility: reproducibility seeds
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------------
# H5 I/O utilities for attention
# ----------------------------------


def _read_attention_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read coords & attention vector from an attention H5 file.

    Assumptions per user brief:
    • `coords` dataset aligned 1:1 with `attention`.
    """
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f or "attention" not in f:
            raise ValueError(f"{h5_path} must contain 'coords' and 'attention' datasets")
        coords = f["coords"][...]
        attention = f["attention"][...]
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords of shape (N, 2), got {coords.shape} in {h5_path}")
    if attention.ndim != 1 or attention.shape[0] != coords.shape[0]:
        raise ValueError(
            f"attention must be 1D and aligned with coords. Got attention {attention.shape}, coords {coords.shape} in {h5_path}"
        )
    return coords, attention

# ----------------------------------
# Encoder loader & preprocessing glue
# ----------------------------------


def _best_transform_for_encoder(enc) -> Callable[[List[Image.Image]], torch.Tensor]:
    """Return callable mapping list[PIL.Image] -> Tensor (B,3,H,W).

    Prefer the transform exposed by TRIDENT's encoder wrapper; **error if missing**.
    This enforces that each encoder defines its own preprocessing/normalization.
    """
    for attr in ("transform", "preprocess", "preprocess_transform"):
        tr = getattr(enc, attr, None)
        if tr is not None and callable(tr):
            def _apply(imgs: List[Image.Image], tr=tr):
                return torch.stack([tr(im) for im in imgs], dim=0)
            return _apply
    raise RuntimeError(
        "Encoder is missing a callable preprocessing transform (e.g., '.transform'). "
        "Please define one in TRIDENT for this model to proceed."
    )


@dataclass
class EncoderBundle:
    name: str
    enc: object  # TRIDENT wrapper
    model: nn.Module
    preproc: Callable[[List[Image.Image]], torch.Tensor]


def load_trident_encoder(name: str, device: torch.device) -> EncoderBundle:
    enc = patch_encoder_factory(name)
    model = getattr(enc, "model", None)
    if model is None:
        raise RuntimeError(
            f"TRIDENT encoder '{name}' does not expose a '.model'. Update TRIDENT or use a supported version."
        )
    model = model.to(device)
    model.eval()
    preproc = _best_transform_for_encoder(enc)
    return EncoderBundle(name=name, enc=enc, model=model, preproc=preproc)

# -----------------
# Grad‑CAM internals
# -----------------

@dataclass
class GradCamCfg:
    cam_layer: str = "auto"     # 'auto' selects the last transformer block output
    reduction: str = "grad_times_act"  # {grad_times_act, grad_sum, grad_l2}
    feature_chunk: int = 32
    keep_graph_per_chunk: bool = False  # per‑feature retain_graph within chunk; graph freed between chunks


def _find_token_layer(model: nn.Module) -> nn.Module:
    """Select a token‑level layer returning (B,N,C) near the model's end.

    We only accept ViT‑style models exposing either `.blocks[-1]`
    or `.model.blocks[-1]`. Any other structure must be wired explicitly
    in TRIDENT; we fail fast otherwise.
    """
    if hasattr(model, "blocks") and hasattr(model.blocks, "__len__") and len(model.blocks) > 0:
        return model.blocks[-1]
    if hasattr(model, "model") and hasattr(model.model, "blocks") and hasattr(model.model.blocks, "__len__") and len(model.model.blocks) > 0:
        return model.model.blocks[-1]
    raise RuntimeError("Could not locate a transformer 'blocks' layer for CAM (expected '.blocks' or '.model.blocks').")


class TokenActHook:
    """Capture token activations (B,N,C) from a chosen layer during forward.

    After forward, `.acts` holds a tensor requiring grad.
    """
    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.acts: Optional[torch.Tensor] = None

    def __enter__(self):
        def _hook(_m, _inp, out):
            self.acts = out
        self.handle = self.layer.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            try:
                self.handle.remove()
            except Exception:
                pass
        self.handle = None
        return False


def _reduce_cam(grad: torch.Tensor, acts: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduce (B,N,C) -> (B,N) token importance.
    reduction in {grad_times_act, grad_sum, grad_l2}
    """
    if reduction == "grad_times_act":
        cam = (grad * acts).sum(dim=-1)
    elif reduction == "grad_sum":
        cam = grad.sum(dim=-1)
    elif reduction == "grad_l2":
        cam = grad.pow(2).sum(dim=-1).sqrt()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    cam = cam.clamp_min(0)
    return cam


def _infer_grid_from_tokens(n_tokens: int) -> Tuple[int, int]:
    if n_tokens <= 1:
        raise ValueError("Token count must be > 1 (expects CLS + patch tokens)")
    n_patches = n_tokens - 1
    s = int(round(math.sqrt(n_patches)))
    if s * s != n_patches:
        raise ValueError(
            f"Token count {n_tokens} is not 1 + square; can't form square grid for CAM."
        )
    return s, s


def _upsample(cam_tokens: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    cam = cam_tokens.unsqueeze(1)
    cam = F.interpolate(cam, size=size_hw, mode="bilinear", align_corners=False)
    return cam.squeeze(1)


def make_out_path(out_root: Path, slide_id: str, coord: Sequence[int], model_name: str, feat_idx: int) -> Path:
    x, y = int(coord[0]), int(coord[1])
    tile_dir = out_root / slide_id / f"({x}_{y})"
    tile_dir.mkdir(parents=True, exist_ok=True)
    return tile_dir / f"gradcam_{model_name}_{feat_idx}.npy"

# -----------------
# WSI helpers (strict matching, no silent fallbacks)
# -----------------

WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".bif", ".scn", ".svslide", ".czi", ".dcm"}


def _find_wsi_for_slide(slides_dir: Path, slide_key: str) -> Path:
    """Find a WSI whose stem contains `slide_key` or equals it.

    • If exactly one match is found, return it.
    • If multiple matches, raise error with candidates.
    • If none, raise error.
    """
    cands = []
    for p in slides_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in WSI_EXTS:
            continue
        st = p.stem
        if st == slide_key or (slide_key in st):
            cands.append(p)
    if len(cands) == 1:
        return cands[0]
    if len(cands) == 0:
        raise RuntimeError(f"No WSI in {slides_dir} matching slide key '{slide_key}'.")
    raise RuntimeError(f"Ambiguous WSI matches for key '{slide_key}': {[c.name for c in cands]}")

# -----------------
# Core worker logic (TRIDENT WSI + WSIPatcher)
# -----------------


def run_for_slide(
    slide_h5: Path,
    attn_h5: Path,
    slides_dir: Path,
    out_root: Path,
    encoders: List[EncoderBundle],
    device: torch.device,
    tile_batch_size: int,
    cfg: GradCamCfg,
    precision: str,
    tile_size: int,
    mag: int,
) -> None:
    # --- Read coords + attention (authoritative for N and order) ---
    coords_attn, attn = _read_attention_h5(attn_h5)

    # --- Verify tiles H5 contains coords aligned with attention H5 (fail hard) ---
    with h5py.File(slide_h5, "r") as f:
        if "coords" not in f:
            raise ValueError(f"Missing 'coords' in {slide_h5}")
        coords_tiles = f["coords"][...]
    if coords_tiles.shape != coords_attn.shape or not np.array_equal(coords_tiles, coords_attn):
        raise ValueError(
            f"coords arrays are not identical between tiles and attention for {slide_h5.name}"
        )

    # --- Resolve WSI path using the attention stem as key ---
    slide_key = attn_h5.stem
    wsi_path = _find_wsi_for_slide(slides_dir, slide_key)

    # --- Build TRIDENT OpenSlide WSI and patcher from legacy coords file ---
    wsi = OpenSlideWSI(str(wsi_path), lazy_init=True)
    try:
        patcher = WSIPatcher.from_legacy_coords_file(
            wsi=wsi,
            coords_file=str(slide_h5),
            patch_size=int(tile_size),
            dst_mag=int(mag),
            pil=True,
        )

        # Sanity: fetch one tile to validate tile size
        test_im = patcher.get_tile(0)
        if not isinstance(test_im, Image.Image):
            raise RuntimeError("WSIPatcher returned a non-PIL tile despite pil=True.")
        if test_im.size != (tile_size, tile_size):
            raise ValueError(
                f"Tile size mismatch: expected {tile_size}x{tile_size}, got {test_im.size} from patcher."
            )

        # --- Select top‑100 by attention ---
        n = attn.shape[0]
        k = min(100, n)
        topk_idx_unsorted = np.argpartition(attn, -k)[-k:]
        topk_idx = topk_idx_unsorted[np.argsort(attn[topk_idx_unsorted])[::-1]]

        # Per‑model loop with progress bars
        slide_id = slide_key
        for bundle in encoders:
            model = bundle.model
            preproc = bundle.preproc
            model_name = bundle.name

            # Determine CAM token layer
            cam_layer = _find_token_layer(model)

            # Determine embedding dimension D by a dry run on 1 tile
            test_img = test_im  # already validated
            x_test = preproc([test_img]).to(device)
            with TokenActHook(cam_layer) as hook:
                z_test = model(x_test)
                acts_test = hook.acts
            if acts_test is None:
                raise RuntimeError(f"Failed to capture token activations for model {model_name}")
            if acts_test.ndim != 3:
                raise RuntimeError(f"Expected token activations (B,N,C), got {tuple(acts_test.shape)}")
            D = int(z_test.shape[-1])

            tiles_pbar = tqdm(total=len(topk_idx), desc=f"{slide_id} · {model_name} · tiles", leave=False)

            # Process tiles in small batches
            for t_start in range(0, len(topk_idx), tile_batch_size):
                idx_batch = topk_idx[t_start : t_start + tile_batch_size]
                coords_batch = coords_attn[idx_batch]
                pil_batch = [patcher.get_tile(int(i)) for i in idx_batch]
                # Validate sizes (strict)
                for im in pil_batch:
                    if im.size != (tile_size, tile_size):
                        raise ValueError(
                            f"Tile size mismatch within batch: expected {tile_size}x{tile_size}, got {im.size}"
                        )
                x = preproc(pil_batch).to(device)

                # Feature chunks to control retain_graph window
                for f_start in tqdm(range(0, D, cfg.feature_chunk), desc="features", leave=False):
                    f_end = min(f_start + cfg.feature_chunk, D)

                    # Forward pass for this chunk (so we can free the graph between chunks)
                    with TokenActHook(cam_layer) as hook:
                        if precision == "fp16":
                            with torch.autocast(device_type=device.type, dtype=torch.float16):
                                z = model(x)
                        elif precision == "bf16":
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

                    # Loop features in this chunk; retain_graph within chunk
                    last_idx = f_end - 1
                    for j in range(f_start, f_end):
                        t = z[:, j].sum()
                        (gradA,) = torch.autograd.grad(
                            t, acts, retain_graph=(j != last_idx), allow_unused=False
                        )

                        grad_tokens = gradA[:, 1:, :]
                        acts_tokens = acts[:, 1:, :]
                        cam_tokens = _reduce_cam(grad_tokens, acts_tokens, cfg.reduction)
                        cams_grid = cam_tokens.reshape(B, gh, gw)
                        cams_up = _upsample(cams_grid, (tile_size, tile_size))
                        cams_up_np = cams_up.detach().cpu().numpy().astype(np.float32)

                        for i in range(B):
                            out_path = make_out_path(out_root, slide_id, coords_batch[i], model_name, j)
                            np.save(out_path, cams_up_np[i])

                    del z, acts
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                tiles_pbar.update(len(idx_batch))
            tiles_pbar.close()
    finally:
        # Ensure WSI is released even on error
        try:
            wsi.release()
        except Exception:
            pass

# ---------------
# CLI / Entrypoint
# ---------------

def list_h5(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() == ".h5"])  # stable order


def match_h5_pairs_strict(tiles_dir: Path, attn_dir: Path) -> List[Tuple[Path, Path]]:
    """Pair coords (tiles) and attention H5s strictly.

    Rule:
    1) Exact stem match wins.
    2) Otherwise, if an attention stem is a substring of the tiles stem, use it.
       If multiple attention stems match the same tiles stem, raise an error.
    3) If nothing matches, fail fast.
    """
    tiles = list_h5(tiles_dir)
    attns = list_h5(attn_dir)

    attn_map_exact = {p.stem: p for p in attns}
    attn_stems = [p.stem for p in attns]

    pairs: List[Tuple[Path, Path]] = []
    ambiguous: List[Tuple[str, List[str]]] = []

    for t in tiles:
        t_stem = t.stem
        # 1) exact
        a = attn_map_exact.get(t_stem)
        if a is not None:
            pairs.append((t, a))
            continue
        # 2) substring (attention stem contained within tiles stem)
        candidates = [s for s in attn_stems if s in t_stem]
        if len(candidates) == 1:
            pairs.append((t, attn_map_exact[candidates[0]]))
        elif len(candidates) > 1:
            ambiguous.append((t_stem, candidates))
        # else: no match for this tile

    if ambiguous:
        msg_lines = [
            "Ambiguous matches detected for some tiles (multiple attention stems contained):"
        ]
        for t_stem, cands in ambiguous[:10]:
            msg_lines.append(f"  • {t_stem} -> {cands}")
        raise RuntimeError("".join(msg_lines))

    if not pairs:
        raise RuntimeError(
            "No matching (tiles, attention) H5 pairs found using exact/substring rules."
        )

    return pairs


def parse_models_arg(s: str) -> List[str]:
    names = [x.strip() for x in s.split(",") if x.strip()]
    if not names:
        raise argparse.ArgumentTypeError("--models must list at least one encoder")
    return names


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Per‑feature token Grad‑CAM for ViT encoders via TRIDENT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--slides_dir", type=Path, required=True, help="Directory of WSIs (e.g., .svs, .tif)")
    ap.add_argument("--tiles_dir", type=Path, required=True, help="Directory of coords H5 files (one per slide)")
    ap.add_argument("--attn_dir", type=Path, required=True, help="Directory of attention H5 files (one per slide)")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory root")
    ap.add_argument(
        "--models",
        type=parse_models_arg,
        default="conch_v15,virchow2,gigapath,hoptimus1,musk",
        help="Comma‑separated TRIDENT patch encoder names",
    )
    ap.add_argument("--mag", type=int, required=True, help="Destination magnification for tiles (e.g., 20)")
    ap.add_argument("--num_slides", type=int, default=1, help="Number of slides to sample at random")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for slide sampling")
    ap.add_argument("--tile_batch_size", type=int, default=4, help="Tile batch size (keep small; grads per feature)")
    ap.add_argument("--feature_chunk", type=int, default=32, help="Number of feature indices per forward/retain_graph window")
    ap.add_argument("--device", type=str, default="cuda:0", help="torch device (e.g., cuda:0 or cpu)")
    ap.add_argument("--precision", type=str, choices=["fp32", "bf16", "fp16"], default="fp32", help="Forward precision (CAM math stays fp32)")
    ap.add_argument("--reduction", type=str, choices=["grad_times_act", "grad_sum", "grad_l2"], default="grad_times_act", help="Token CAM reduction rule")
    ap.add_argument("--tile_size", type=int, required=True, help="Required uniform tile size (H=W) returned by patcher")

    args = ap.parse_args(argv)

    set_seed(args.seed)

    device = torch.device(args.device)
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # Pair up coords H5s with attention H5s, sample N slides deterministically
    all_pairs = match_h5_pairs_strict(args.tiles_dir, args.attn_dir)
    rng = random.Random(args.seed)
    selected_pairs = rng.sample(all_pairs, k=min(args.num_slides, len(all_pairs)))

    # Load encoders (fail hard on any error as requested)
    encoders: List[EncoderBundle] = []
    for name in args.models:
        bundle = load_trident_encoder(name, device=device)
        encoders.append(bundle)

    cfg = GradCamCfg(
        cam_layer="auto",
        reduction=args.reduction,
        feature_chunk=int(args.feature_chunk),
        keep_graph_per_chunk=False,
    )

    # Progress over slides
    for tiles_h5, attn_h5 in tqdm(selected_pairs, desc="Slides", leave=True):
        run_for_slide(
            slide_h5=tiles_h5,
            attn_h5=attn_h5,
            slides_dir=args.slides_dir,
            out_root=out_root,
            encoders=encoders,
            device=device,
            tile_batch_size=int(args.tile_batch_size),
            cfg=cfg,
            precision=args.precision,
            tile_size=int(args.tile_size),
            mag=int(args.mag),
        )

    print("Done. Saved per‑feature Grad‑CAMs to:", str(out_root))


if __name__ == "__main__":
    main()
