#!/usr/bin/env python3
"""
Extract multi‑resolution, same‑physical‑area patches (5x/10x/20x by default) and compute
patch‑level/tile‑level attention maps from pathology foundation models.

This script reuses the model loading and attention extraction utilities already present in:
- extract_attention_maps.py (SimpleAttentionExtractor, AttentionRollout alias)
- test_single_model.py (get_model_config)

Key features:
- For each selected H5 tile (top‑K by attention score), derive the tile center (coords are top‑left)
- One OpenSlide crop at the highest requested magnification (typically 20×) at the corresponding
  output size. Lower magnifications are approximations produced by downsampling that crop:
  e.g. with ref 20×, 10× uses a 2× downsample and 5× uses a 4× downsample (same physical field of view).
- Compute attention maps for multiple foundation models using rollout or simple extraction.
- Attention overlays: by default z-normalize each patch (mean/std, clip ±3σ to [0,1]) so the
  colormap uses the full range; use `--attention-normalize minmax` for the previous min–max scaling.
- Save continuous attention maps and percentile‑thresholded binaries (default p50/p70/p90).

Output layout (rollout mode):
- Original/{5x,10x,20x}/<tile>.png
- {model}/{5x,10x,20x}/combined/<tile>.png
- {model}/{5x,10x,20x}/first_layer/<tile>.png
- {model}/{5x,10x,20x}/threshold_p{50,70,90}/<tile>.png

Output layout (simple mode):
- Original/{5x,10x,20x}/<tile>.png
- {model}/{5x,10x,20x}/simple/<tile>_L{idx}_H{head|avg}.png
- {model}/{5x,10x,20x}/threshold_p{50,70,90}/<tile>_L{idx}_H{head|avg}.png
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from scipy.ndimage import gaussian_filter

try:
	import h5py  # type: ignore
except ImportError as e:
	raise ImportError("h5py is required. Please install it before running this script.") from e

try:
	import openslide  # type: ignore
except ImportError as e:
	raise ImportError("openslide-python is required. Please install it before running this script.") from e

import torch

# TRIDENT encoder loader
from trident.patch_encoder_models import encoder_factory

# Reuse attention configs/utilities
import sys
sys.path.append('/mnt/vstor/Data7/bxf169/KidneyCancerPathology')
from test_single_model import get_model_config  # noqa: E402
from extract_attention_maps import (  # noqa: E402
	AttentionRollout,
	SimpleAttentionExtractor,
)


# -----------------------------
# H5 helpers (reuse semantics)
# -----------------------------
def _load_attention_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
	"""Load attention scores and coordinates from an HDF5 file."""
	with h5py.File(h5_path, "r") as f:
		score_candidates = ["attention_scores", "attention", "scores"]
		coord_candidates = ["coords", "coordinates"]
		score_key = next((k for k in score_candidates if k in f), None)
		coord_key = next((k for k in coord_candidates if k in f), None)
		if score_key is None:
			raise KeyError(f"None of {score_candidates} found in {h5_path}")
		if coord_key is None:
			raise KeyError(f"None of {coord_candidates} found in {h5_path}")
		scores = np.array(f[score_key]).reshape(-1)
		coords = np.array(f[coord_key])
		if coords.ndim != 2 or coords.shape[1] != 2:
			raise ValueError(f"Coordinate dataset {coord_key} must be (n,2), got {coords.shape}")
		return scores, coords


# -----------------------------
# OpenSlide / level math
# -----------------------------
def _get_objective_power(slide: "openslide.OpenSlide") -> Optional[float]:
	obj = slide.properties.get("openslide.objective-power", None)
	try:
		return float(obj) if obj is not None else None
	except Exception:
		return None


def _get_level_for_magnification(slide: "openslide.OpenSlide", target_mag: float) -> int:
	"""
	Choose the OpenSlide level such that its downsample is closest to (objective_power / target_mag).
	Fallback: choose level by minimizing |downsample - (level0_downsample_ratio)| assuming 20x base.
	"""
	level_downsamples = [slide.level_downsamples[i] for i in range(slide.level_count)]
	objective = _get_objective_power(slide)
	if objective is not None and target_mag > 0:
		target_downsample = objective / float(target_mag)
	else:
		# Fallback heuristic: assume level-0 ~ 20x, so target_downsample ≈ 20 / target_mag
		target_downsample = 20.0 / float(target_mag)
	level = int(np.argmin([abs(ds - target_downsample) for ds in level_downsamples]))
	return level


def _compute_level0_top_left(center_x0: float, center_y0: float, level: int, slide: "openslide.OpenSlide", size_px: int) -> Tuple[int, int]:
	"""
	Given a center in level-0 coords, an OpenSlide level, and desired output size (W=H=size_px) at that level,
	return a bounds-safe top-left in level-0 pixels suitable for read_region((x,y), level, (size,size)).
	"""
	down = float(slide.level_downsamples[level])
	radius0 = 0.5 * size_px * down
	x0 = int(round(center_x0 - radius0))
	y0 = int(round(center_y0 - radius0))
	# Clamp to non-negative; OpenSlide will pad if we go past max bounds anyway
	x0 = max(0, x0)
	y0 = max(0, y0)
	return x0, y0


def _percentile_threshold(mask_0_1: np.ndarray, percentile: float) -> np.ndarray:
	"""
	Binarize a normalized [0,1] attention map by its own percentile.
	Returns uint8 image with values in {0,255}.
	"""
	thr = float(np.percentile(mask_0_1.astype(np.float32), percentile))
	bin_mask = (mask_0_1 >= thr).astype(np.uint8) * 255
	return bin_mask


def _normalize_0_1(arr: np.ndarray) -> np.ndarray:
	arr = arr.astype(np.float32)
	min_v, max_v = float(arr.min()), float(arr.max())
	if max_v > min_v:
		return (arr - min_v) / (max_v - min_v)
	return np.zeros_like(arr, dtype=np.float32)


def _normalize_z_score_0_1(
	arr: np.ndarray,
	clip_low: float = -3.0,
	clip_high: float = 3.0,
) -> np.ndarray:
	"""
	Per-patch z-score, clip to [clip_low, clip_high] (σ units), then linear map to [0, 1].
	Spreads the colormap when raw attention has a narrow dynamic range.
	"""
	arr = np.asarray(arr, dtype=np.float32)
	arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
	flat = arr.ravel()
	if flat.size == 0:
		return arr
	mu = float(np.mean(flat))
	sd = float(np.std(flat))
	if sd < 1e-8:
		return np.zeros_like(arr, dtype=np.float32)
	z = (arr - mu) / sd
	z = np.clip(z, clip_low, clip_high)
	span = clip_high - clip_low
	if span <= 0:
		return np.zeros_like(arr, dtype=np.float32)
	return np.clip((z - clip_low) / span, 0.0, 1.0).astype(np.float32)


def _normalize_attention_for_display(arr: np.ndarray, mode: str) -> np.ndarray:
	if mode == "zscore":
		return _normalize_z_score_0_1(arr)
	if mode == "minmax":
		return _normalize_0_1(arr)
	raise ValueError(f"attention_normalize must be 'zscore' or 'minmax', got {mode!r}")


def _make_overlay_image(original_pil: Image.Image, attn_map_0_1: np.ndarray, blur_sigma: float, alpha: float, zero_below_median: bool = True) -> Image.Image:
	"""
	Adopt overlay parameters from overlay_tile_attention.py:
	- Convert attention to [0,255], apply Gaussian smoothing
	- Optionally zero out values below 50th percentile of non-zero pixels
	- Colorize with JET and alpha blend only non-zero mask
	"""
	# Convert original to BGR uint8 for OpenCV blending
	orig_rgb = np.array(original_pil.convert("RGB"), dtype=np.uint8)
	orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
	H, W = orig_bgr.shape[:2]

	# Prepare attention heat
	heat = attn_map_0_1.astype(np.float32)
	# Resize attention to match original if needed
	if heat.shape[0] != H or heat.shape[1] != W:
		heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
	# Smooth
	heat = gaussian_filter(heat, sigma=blur_sigma)
	heat = np.clip(heat, 0.0, 1.0)
	# Threshold at 50th percentile of non-zero
	if zero_below_median:
		nonzero = heat[heat > 0]
		if nonzero.size > 0:
			cutoff = float(np.percentile(nonzero, 50.0))
			heat[heat < cutoff] = 0.0
	# To uint8
	heat_uint = (heat * 255.0).astype(np.uint8)
	# Colorize and blend
	heat_color = cv2.applyColorMap(heat_uint, cv2.COLORMAP_JET)
	blended = cv2.addWeighted(orig_bgr, 1.0 - alpha, heat_color, alpha, 0.0)
	# Only write blended where mask > 0
	mask = heat_uint > 0
	out_bgr = orig_bgr.copy()
	out_bgr[mask] = blended[mask]
	# Back to RGB PIL
	out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
	return Image.fromarray(out_rgb, mode="RGB")


# -----------------------------
# Core pipeline
# -----------------------------
def extract_multires_attention_maps(
	slide_dir: Path,
	h5_dir: Optional[Path],
	output_dir: Path,
	model_names: Iterable[str],
	magnifications: List[int],
	out_sizes: List[int],
	percentiles: List[float],
	device: str = "cpu",
	precision_override: Optional[torch.dtype] = None,
	top_k: int = 100,
	overwrite: bool = False,
	simple_mode: bool = False,
	layer_index: int = -1,
	head_index: Optional[int] = None,
	base_tile_size: int = 224,
	# Random-from-tissue mode
	random_from_tissue: bool = False,
	tiles_per_slide: int = 5,
	random_seed: int = 42,
	thumb_max_size: int = 2048,
	tissue_sat_min: int = 20,
	tissue_val_max: int = 245,
	# Overlays
	overlay_blur_sigma: float = 20.0,
	overlay_alpha: float = 0.4,
	attention_normalize: str = "zscore",
	# Slide limiting
	max_slides: int = 5,
) -> None:
	"""
	Process slides and extract multi‑resolution attention maps.
	"""
	slide_dir = Path(slide_dir)
	h5_dir = Path(h5_dir) if h5_dir is not None else None
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	if attention_normalize not in ("zscore", "minmax"):
		raise ValueError("attention_normalize must be 'zscore' or 'minmax'")

	# Prepare output subdirectories scaffold
	original_dirs: Dict[str, Path] = {}
	for mag in magnifications:
		sub = output_dir / "Original" / f"{mag}x"
		sub.mkdir(parents=True, exist_ok=True)
		original_dirs[f"{mag}x"] = sub

	# Load models once (mirrors extract_attention_maps.py flow)
	if simple_mode:
		models: Dict[str, Tuple[torch.nn.Module, SimpleAttentionExtractor, int, torch.dtype, callable]] = {}
		mode_desc = "simple single-layer"
	else:
		models: Dict[str, Tuple[torch.nn.Module, AttentionRollout, int, torch.dtype, callable]] = {}
		mode_desc = "multi-layer rollout"

	print(f"Loading models for {mode_desc} extraction...")
	for m in tqdm(model_names, desc="Loading models"):
		try:
			model_config = get_model_config(m)
			# Use per-model initialization args for stability
			if m == "conch_v15":
				encoder = encoder_factory(m, img_size=448)  # type: ignore
			elif m == "hoptimus1":
				encoder = encoder_factory(m, timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False})  # type: ignore
			else:
				encoder = encoder_factory(m)  # type: ignore
			model = encoder.model
			base_model = getattr(model, "trunk", model)
			if simple_mode:
				extractor = SimpleAttentionExtractor(base_model, model_config)
			else:
				extractor = AttentionRollout(base_model, model_config)
			# Prefer float32 by default to avoid half/float bias mismatches
			if precision_override is not None:
				precision = precision_override
			else:
				precision = torch.float32
			# Ensure encoder/model weights match requested device and precision
			try:
				# Many TRIDENT encoders are nn.Module wrappers
				encoder.to(device=device, dtype=precision)  # type: ignore[attr-defined]
			except Exception:
				try:
					encoder.model.to(device=device, dtype=precision)
				except Exception:
					pass
			# Safety: set eval mode
			try:
				encoder.eval()
			except Exception:
				try:
					encoder.model.eval()
				except Exception:
					pass
			models[m] = (encoder, extractor, model_config['skip_tokens'], precision, encoder.eval_transforms)
		except Exception as e:
			print(f"ERROR: Failed to load model {m}: {e}")
			continue
	if not models:
		print("ERROR: No models loaded successfully. Exiting.")
		return

	# Slide discovery
	supported_exts = (".svs", ".tif", ".tiff", ".png", ".jpg", ".jpeg")
	slide_files = sorted([p for p in slide_dir.iterdir() if p.suffix.lower() in supported_exts])
	if not slide_files:
		print(f"No slide files found in {slide_dir}")
		return

	# If using H5 path (score-driven) mode, prepare H5s; else we will random sample per slide
	if not random_from_tissue:
		if h5_dir is None:
			raise ValueError("--h5s is required unless --random-from-tissue is set")
		h5_files = sorted(h5_dir.glob("*.h5"))
		if not h5_files:
			print(f"No .h5 files found in {h5_dir}")
			return
		print(f"Found {len(h5_files)} HDF5 files to process")

	# Pre-build per-model directory trees (model / mag / {combined, first_layer, simple, threshold_pXX})
	per_model_dirs: Dict[str, Dict[str, Dict[str, Path]]] = {}
	for model_name in models.keys():
		base = output_dir / model_name
		base.mkdir(exist_ok=True)
		per_model_dirs[model_name] = {}
		for mag in magnifications:
			mag_key = f"{mag}x"
			per_model_dirs[model_name][mag_key] = {}
			if simple_mode:
				simple_dir = base / mag_key / "simple"
				simple_dir.mkdir(parents=True, exist_ok=True)
				per_model_dirs[model_name][mag_key]["simple"] = simple_dir
			else:
				combined_dir = base / mag_key / "combined"
				first_layer_dir = base / mag_key / "first_layer"
				combined_dir.mkdir(parents=True, exist_ok=True)
				first_layer_dir.mkdir(parents=True, exist_ok=True)
				per_model_dirs[model_name][mag_key]["combined"] = combined_dir
				per_model_dirs[model_name][mag_key]["first_layer"] = first_layer_dir
			# Threshold directories (common for both modes)
			for p in percentiles:
				td = base / mag_key / f"threshold_p{int(p)}"
				td.mkdir(parents=True, exist_ok=True)
				per_model_dirs[model_name][mag_key][f"threshold_p{int(p)}"] = td

	def _make_and_save_outputs_for_center(stem: str, center_x0: float, center_y0: float, slide: "openslide.OpenSlide") -> None:
		# Single crop at max(magnifications); lower mags = downsample (same FOV, approximate optics)
		ref_mag = max(magnifications)
		ref_idx = magnifications.index(ref_mag)
		ref_out_size = int(out_sizes[ref_idx])
		ref_level = _get_level_for_magnification(slide, float(ref_mag))
		ref_down = float(slide.level_downsamples[ref_level])
		read_size = ref_out_size  # pixel size at ref_level covering ref_out_size * ref_down in level-0
		tlx0, tly0 = _compute_level0_top_left(center_x0, center_y0, ref_level, slide, read_size)
		tile_base = f"{stem}_{int(round(center_x0 - 0.5 * base_tile_size))}_{int(round(center_y0 - 0.5 * base_tile_size))}"
		try:
			patch_ref = slide.read_region((int(tlx0), int(tly0)), ref_level, (read_size, read_size)).convert("RGB")
		except Exception as e:
			tqdm.write(f"Warning: read_region failed at ({tlx0},{tly0}) L{ref_level} size {read_size}: {e}")
			return
		if patch_ref.size[0] != ref_out_size or patch_ref.size[1] != ref_out_size:
			patch_ref = patch_ref.resize((ref_out_size, ref_out_size), Image.BILINEAR)

		# For each requested magnification and output size, derive patch from ref crop and compute attention
		for mag, out_size in zip(magnifications, out_sizes):
			mag_key = f"{mag}x"
			if mag == ref_mag:
				patch = patch_ref.copy()
			else:
				factor = float(ref_mag) / float(mag)
				side = max(1, int(round(ref_out_size / factor)))
				patch = patch_ref.resize((side, side), Image.LANCZOS)
			if patch.size[0] != out_size or patch.size[1] != out_size:
				patch = patch.resize((out_size, out_size), Image.BILINEAR)

			# Save original crop
			orig_name = f"{tile_base}_S{out_size}_L{ref_level}.png"
			orig_path = original_dirs[mag_key] / orig_name
			if overwrite or not orig_path.exists():
				try:
					disp = patch if patch.size == (out_size, out_size) else patch.resize((out_size, out_size), Image.BILINEAR)
					disp.save(orig_path)
				except Exception as e:
					tqdm.write(f"Warning: Failed to save original tile: {e}")

			# For each model: preprocess, run encoder, compute attention
			for model_name, (encoder, extractor, _skip_tokens, precision, transform) in models.items():
				# Preprocess
				try:
					processed = transform(patch)
				except Exception as e:
					tqdm.write(f"Warning: Transforms failed for {model_name}: {e}")
					continue
				processed = processed.unsqueeze(0).to(device=device, dtype=precision)

				if simple_mode:
					# Extract from specific layer
					try:
						base_model = getattr(encoder.model, "trunk", encoder.model)
						simple_extractor = SimpleAttentionExtractor(base_model, get_model_config(model_name))
						att_mask = simple_extractor.get_attention_map(processed, layer_index, head_index)
					except Exception as e:
						tqdm.write(f"Warning: Simple attention failed for {model_name}: {e}")
						continue

					head_suffix = f"_H{head_index}" if head_index is not None else "_Havg"
					simple_name = f"{tile_base}_S{out_size}_L{layer_index}{head_suffix}.png"
					simple_path = per_model_dirs[model_name][mag_key]["simple"] / simple_name
					if overwrite or not simple_path.exists():
						try:
							disp_base = patch if patch.size == (out_size, out_size) else patch.resize((out_size, out_size), Image.BILINEAR)
							overlay_img = _make_overlay_image(
								disp_base, _normalize_attention_for_display(att_mask, attention_normalize), overlay_blur_sigma, overlay_alpha
							)
							overlay_img.save(simple_path)
						except Exception as e:
							tqdm.write(f"Warning: Save simple overlay failed for {model_name}: {e}")

					# Percentile thresholds
					norm_map = _normalize_attention_for_display(att_mask, attention_normalize)
					for p in percentiles:
						thr = float(np.percentile(norm_map, float(p)))
						th_map = norm_map.copy()
						th_map[th_map < thr] = 0.0
						disp_base = patch if patch.size == (out_size, out_size) else patch.resize((out_size, out_size), Image.BILINEAR)
						th_overlay = _make_overlay_image(disp_base, th_map, overlay_blur_sigma, overlay_alpha, zero_below_median=False)
						th_path = per_model_dirs[model_name][mag_key][f"threshold_p{int(p)}"] / simple_name
						if overwrite or not th_path.exists():
							try:
								th_overlay.save(th_path)
							except Exception as e:
								tqdm.write(f"Warning: Save percentile overlay failed ({p}) for {model_name}: {e}")
				else:
					# Rollout: combined + first_layer
					try:
						# Clear any previous attentions and run the encoder
						extractor._attentions.clear()  # type: ignore[attr-defined]
						with torch.no_grad():
							_ = encoder(processed)
						# debug prints disabled
						combined = extractor.compute_rollout()  # type: ignore[call-arg]
						first_layer = extractor.compute_first_layer_attention()  # type: ignore[call-arg]
					except Exception as e:
						tqdm.write(f"Warning: Rollout failed for {model_name}: {e}")
						continue

					comb_name = f"{tile_base}_S{out_size}.png"
					comb_path = per_model_dirs[model_name][mag_key]["combined"] / comb_name
					first_name = f"{tile_base}_S{out_size}.png"
					first_path = per_model_dirs[model_name][mag_key]["first_layer"] / first_name

					if overwrite or not comb_path.exists():
						try:
							disp_base = patch if patch.size == (out_size, out_size) else patch.resize((out_size, out_size), Image.BILINEAR)
							overlay_img = _make_overlay_image(
								disp_base, _normalize_attention_for_display(combined, attention_normalize), overlay_blur_sigma, overlay_alpha
							)
							overlay_img.save(comb_path)
						except Exception as e:
							tqdm.write(f"Warning: Save combined overlay failed for {model_name}: {e}")
					if overwrite or not first_path.exists():
						try:
							disp_base = patch if patch.size == (out_size, out_size) else patch.resize((out_size, out_size), Image.BILINEAR)
							overlay_first = _make_overlay_image(
								disp_base, _normalize_attention_for_display(first_layer, attention_normalize), overlay_blur_sigma, overlay_alpha
							)
							overlay_first.save(first_path)
						except Exception as e:
							tqdm.write(f"Warning: Save first_layer overlay failed for {model_name}: {e}")

					# Percentile thresholds from combined
					norm_map = _normalize_attention_for_display(combined, attention_normalize)
					for p in percentiles:
						thr = float(np.percentile(norm_map, float(p)))
						th_map = norm_map.copy()
						th_map[th_map < thr] = 0.0
						disp_base = patch if patch.size == (out_size, out_size) else patch.resize((out_size, out_size), Image.BILINEAR)
						th_overlay = _make_overlay_image(disp_base, th_map, overlay_blur_sigma, overlay_alpha, zero_below_median=False)
						th_path = per_model_dirs[model_name][mag_key][f"threshold_p{int(p)}"] / comb_name
						if overwrite or not th_path.exists():
							try:
								th_overlay.save(th_path)
							except Exception as e:
								tqdm.write(f"Warning: Save percentile overlay failed ({p}) for {model_name}: {e}")

	def _make_thumbnail_and_tissue_mask(slide: "openslide.OpenSlide") -> Tuple[Image.Image, np.ndarray]:
		w0, h0 = slide.level_dimensions[0]
		scale = max(w0, h0) / float(thumb_max_size) if max(w0, h0) > thumb_max_size else 1.0
		thumb_w = max(1, int(round(w0 / scale)))
		thumb_h = max(1, int(round(h0 / scale)))
		thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
		# Simple HSV-based tissue mask: require some saturation, not ultra-bright
		hsv = np.array(thumb.convert("HSV"), dtype=np.uint8)
		h = hsv[:, :, 0]  # unused
		s = hsv[:, :, 1]
		v = hsv[:, :, 2]
		mask = (s >= tissue_sat_min) & (v <= tissue_val_max)
		return thumb, mask

	def _sample_random_centers_from_mask(slide: "openslide.OpenSlide", mask: np.ndarray, n: int, seed: int) -> List[Tuple[float, float]]:
		# Map mask coords to level-0 coords
		h_t, w_t = mask.shape
		w0, h0 = slide.level_dimensions[0]
		scale_x = float(w0) / float(w_t)
		scale_y = float(h0) / float(h_t)
		ys, xs = np.where(mask)
		if len(xs) == 0:
			return []
		rng = np.random.default_rng(seed)
		choice_idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
		sel_xs = xs[choice_idx]
		sel_ys = ys[choice_idx]
		centers = []
		for tx, ty in zip(sel_xs, sel_ys):
			cx0 = (float(tx) + 0.5) * scale_x
			cy0 = (float(ty) + 0.5) * scale_y
			centers.append((cx0, cy0))
		return centers

	# Processing path selection
	if random_from_tissue:
		print(f"Random-from-tissue mode ON: sampling {tiles_per_slide} patches per slide")
		np.random.seed(random_seed)
		slide_counter = 0
		for slide_path in tqdm(slide_files, desc="Slides"):
			if slide_counter >= max_slides:
				break
			stem = slide_path.stem
			try:
				slide = openslide.OpenSlide(str(slide_path))
			except Exception as e:
				tqdm.write(f"Warning: Failed to open slide {slide_path}: {e}")
				continue
			try:
				thumb, tissue_mask = _make_thumbnail_and_tissue_mask(slide)
				centers = _sample_random_centers_from_mask(slide, tissue_mask, tiles_per_slide, random_seed)
				if not centers:
					tqdm.write(f"Warning: No tissue detected for {stem}; skipping.")
					slide.close()
					continue
				for cx0, cy0 in tqdm(centers, desc=f"Tiles in {stem}", leave=False):
					_make_and_save_outputs_for_center(stem, cx0, cy0, slide)
			finally:
				slide.close()
			slide_counter += 1
		print("Done. Random tissue sampling complete.")
		return

	# H5-driven path (original behavior)
	# Tile counting (for progress estimates)
	total_tiles = 0
	for h5_file in h5_files:
		try:
			scores, _ = _load_attention_h5(h5_file)
			total_tiles += min(len(scores), top_k)
		except Exception:
			continue
	print(f"Processing ~{total_tiles} tiles across {len(h5_files)} slides...")

	# Main loop
	slide_counter = 0
	for h5_file in tqdm(h5_files, desc="Slides"):
		if slide_counter >= max_slides:
			break
		try:
			scores, coords = _load_attention_h5(h5_file)
		except Exception as e:
			tqdm.write(f"Warning: Failed to load {h5_file.name}: {e}")
			continue
		if len(scores) == 0 or len(coords) == 0:
			tqdm.write(f"Warning: {h5_file.name} contains no data; skipping.")
			continue

		# Find slide path by stem
		stem = h5_file.stem
		possible_exts = [".svs", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]
		slide_path = None
		for ext in possible_exts:
			cand = slide_dir / f"{stem}{ext}"
			if cand.exists():
				slide_path = cand
				break
		if slide_path is None:
			tqdm.write(f"Warning: No matching slide found for {stem}")
			continue

		# Open slide once
		try:
			slide = openslide.OpenSlide(str(slide_path))
		except Exception as e:
			tqdm.write(f"Warning: Failed to open slide {slide_path}: {e}")
			continue

		# Select top‑K tiles
		sorted_idx = np.argsort(scores)
		k = min(len(sorted_idx), top_k)
		top_indices = sorted_idx[-k:][::-1]

		for idx in tqdm(top_indices, desc=f"Tiles in {stem}", leave=False):
			x0, y0 = coords[idx]  # top‑left at level‑0
			# Center from given convention (top-left) and base tile size
			center_x0 = float(x0) + 0.5 * float(base_tile_size)
			center_y0 = float(y0) + 0.5 * float(base_tile_size)
			_make_and_save_outputs_for_center(stem, center_x0, center_y0, slide)

		# Explicitly close slide
		slide.close()
		slide_counter += 1

	print("Done. Multi‑resolution attention extraction complete.")


# -----------------------------
# CLI
# -----------------------------
def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Extract multi‑resolution patches and attention maps: one crop at max magnification, lower mags by downsampling."
	)
	parser.add_argument("--slides", "-s", type=str, required=True, help="Directory with whole‑slide images")
	parser.add_argument("--h5s", "-a", type=str, required=False, default=None, help="Directory with HDF5 attention files (required unless --random-from-tissue)")
	parser.add_argument("--output", "-o", type=str, required=True, help="Output directory root")
	parser.add_argument(
		"--models", "-m", type=str, nargs="+", required=True,
		help="Models: musk, conch_v15, virchow2, hoptimus1, gigapath"
	)
	parser.add_argument("--top-k", type=int, default=100, help="Top K tiles per slide (H5-driven mode)")
	parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
	parser.add_argument(
		"--precision", type=str, default=None,
		help="Override model precision: float32 | float16 | bfloat16"
	)
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs")
	parser.add_argument("--simple", action="store_true", help="Use simple single-layer attention extraction")
	parser.add_argument("--layer-index", type=int, default=-1, help="Layer index for simple mode")
	parser.add_argument("--head-index", type=int, default=None, help="Head index for simple mode (default: avg)")
	parser.add_argument("--base-tile-size", type=int, default=224, help="Base tile size used by H5 coords (top-left)")

	# Multi-res specifics
	parser.add_argument("--magnifications", type=int, nargs="+", default=[5, 10, 20], help="Target magnifications")
	parser.add_argument(
		"--sizes", type=int, nargs="+", default=[256, 512, 1024],
		help="Output square patch sizes (pixels) per magnification; ref is max(--magnifications), others are downsampled then resized if needed"
	)
	parser.add_argument("--percentiles", type=float, nargs="+", default=[50.0, 70.0, 90.0], help="Percentiles to threshold")
	# Random-from-tissue mode (no H5 required)
	parser.add_argument("--random-from-tissue", action="store_true", help="Sample random patches from tissue mask instead of using H5 scores")
	parser.add_argument("--tiles-per-slide", type=int, default=5, help="Number of random tissue patches per slide when using --random-from-tissue")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
	parser.add_argument("--thumb-max", type=int, default=2048, help="Max thumbnail dimension (longest side)")
	parser.add_argument("--tissue-sat-min", type=int, default=20, help="HSV S threshold (min) to consider tissue")
	parser.add_argument("--tissue-val-max", type=int, default=245, help="HSV V threshold (max) to consider tissue (exclude bright background)")
	# Overlays
	parser.add_argument("--overlay-blur-sigma", type=float, default=20.0, help="Gaussian blur sigma for overlay smoothing")
	parser.add_argument("--overlay-alpha", type=float, default=0.4, help="Alpha for overlay blending")
	parser.add_argument(
		"--attention-normalize",
		type=str,
		default="zscore",
		choices=["zscore", "minmax"],
		help="Scale attention for JET overlay: zscore = per-patch (x-mu)/sigma clipped to ±3σ→[0,1]; minmax = global min-max",
	)
	# Slide limit
	parser.add_argument("--max-slides", type=int, default=5, help="Maximum number of slides to process")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	precision = None
	if args.precision:
		precision_map = {
			"float32": torch.float32,
			"float16": torch.float16,
			"bfloat16": torch.bfloat16,
		}
		if args.precision not in precision_map:
			raise ValueError(f"Unsupported precision '{args.precision}'. Use one of {list(precision_map.keys())}")
		precision = precision_map[args.precision]

	if len(args.magnifications) != len(args.sizes):
		raise ValueError("--magnifications and --sizes must have the same length")

	extract_multires_attention_maps(
		slide_dir=Path(args.slides),
		h5_dir=(Path(args.h5s) if hasattr(args, "h5s") and args.h5s else None),
		output_dir=Path(args.output),
		model_names=args.models,
		magnifications=list(args.magnifications),
		out_sizes=list(args.sizes),
		percentiles=list(args.percentiles),
		device=args.device,
		precision_override=precision,
		top_k=args.top_k,
		overwrite=args.overwrite,
		simple_mode=args.simple,
		layer_index=args.layer_index,
		head_index=args.head_index,
		base_tile_size=args.base_tile_size,
		random_from_tissue=args.random_from_tissue,
		tiles_per_slide=args.tiles_per_slide,
		random_seed=args.seed,
		thumb_max_size=args.thumb_max,
		tissue_sat_min=args.tissue_sat_min,
		tissue_val_max=args.tissue_val_max,
		overlay_blur_sigma=args.overlay_blur_sigma,
		overlay_alpha=args.overlay_alpha,
		attention_normalize=args.attention_normalize,
		max_slides=args.max_slides,
	)


if __name__ == "__main__":
	main()

