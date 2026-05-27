
# attention_from_h5.py
#please please please work this time
import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# choose your loader
from lora_hoptimus1_loader import load_hoptimus1
# from lora_conch_loader import load_conch_v15


# -----------------------------
# Utilities
# -----------------------------
def _find_blocks_container(model: torch.nn.Module):
    """
    Return the Sequential of transformer blocks for different FM wrappers.
    - Conch: model.trunk.blocks
    - H-Optimus (timm ViT wrapper): model.core.blocks
    Fallback: first nn.Sequential named '*blocks'.
    """
    if hasattr(model, "trunk") and hasattr(model.trunk, "blocks"):
        return model.trunk.blocks
    if hasattr(model, "core") and hasattr(model.core, "blocks"):
        return model.core.blocks
    for name, mod in model.named_modules():
        if name.endswith("blocks") and isinstance(mod, nn.Sequential):
            return mod
    raise AttributeError("Could not locate transformer blocks on this model.")


def _infer_grid_side_and_k(total_tokens: int, max_k: int = 16) -> tuple[int, int]:
    """
    Find (side, K) such that total_tokens = K + side*side, with small K (special tokens).
    Tries K=1..max_k; falls back to the largest square <= total_tokens.
    """
    for K in range(1, max_k + 1):
        P = total_tokens - K
        s = int(round(np.sqrt(P)))
        if s * s == P and s > 0:
            return s, K
    # Fallback: largest square <= N
    s = int(np.floor(np.sqrt(total_tokens)))
    P = s * s
    K = total_tokens - P
    return s, K


def _has_cls_token(tokens_len: int) -> bool:
    """Heuristic: CLS present if (tokens-1) is a perfect square."""
    sq = int(round(np.sqrt(max(tokens_len - 1, 0))))
    if sq * sq == tokens_len - 1:
        return True
    sq2 = int(round(np.sqrt(tokens_len)))
    return False if sq2 * sq2 == tokens_len else True


def _get_model_img_size(model, default: int = 224) -> int:
    """
    Read the model's configured input size from its patch_embed.img_size.
    Works for timm VisionTransformer (H-Optimus) and Conch.
    """
    pe = None
    if hasattr(model, "core") and hasattr(model.core, "patch_embed"):
        pe = model.core.patch_embed
    elif hasattr(model, "trunk") and hasattr(model.trunk, "patch_embed"):
        pe = model.trunk.patch_embed
    if pe is not None and hasattr(pe, "img_size"):
        s = pe.img_size
        return int(s[0]) if isinstance(s, (tuple, list)) else int(s)
    return default


def preprocess_for_model(img_pil: Image.Image,
                         model,
                         tfm,
                         device: str,
                         dtype: torch.dtype) -> tuple[torch.Tensor, np.ndarray]:
    """
    Force resize to model's expected img_size BEFORE applying tfm.
    Returns: (tensor ready for model, RGB numpy (H,W,3) for overlay)
    """
    target = _get_model_img_size(model, default=224)
    if img_pil.size != (target, target):
        img_pil = img_pil.resize((target, target), Image.BICUBIC)

    x = tfm(img_pil).unsqueeze(0).to(device, dtype=dtype)
    vis_rgb = np.array(img_pil, dtype=np.uint8)  # clean overlay base
    return x, vis_rgb


# -----------------------------
# Extract attention (monkey patch)
# -----------------------------
def get_attention_map(model, x, device, layer_idx: int = -1, head_idx: int | None = None) -> np.ndarray:
 
    blocks = _find_blocks_container(model)
    attn_weights = {}

    def patched_forward(self, x_in):
        # timm-style Attention: x_in is (B, tokens, C)
        B, N, C = x_in.shape
        qkv = self.qkv(x_in).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, heads, tokens, dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # (B, heads, tokens, tokens)
        attn_weights["attn"] = attn.detach().cpu()
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        if hasattr(self, "proj_drop"):
            out = self.proj_drop(out)
        return out

    # Patch forward on the target block's Attention module
    block_attn = blocks[layer_idx].attn
    orig_forward = block_attn.forward
    block_attn.forward = patched_forward.__get__(block_attn, type(block_attn))

    with torch.no_grad():
        _ = model(x.to(device))

    # Restore
    block_attn.forward = orig_forward

    if "attn" not in attn_weights:
        raise RuntimeError("Failed to capture attention weights (hook did not fire).")

    # attn: (B, heads, tokens, tokens) -> select batch 0
    # attn: (B, heads, tokens, tokens) -> select batch 0, then average (or pick head)
    attn = attn_weights["attn"][0]
    attn_tok = attn[head_idx] if head_idx is not None else attn.mean(0)  # (tokens, tokens)

    T = attn_tok.shape[0]
    side, K = _infer_grid_side_and_k(T, max_k=16)  # handles CLS + multiple reg tokens

    # Use CLS → patches attention: row 0 to columns [K : K+P]
    P = side * side
    if K + P > T:
        raise ValueError(f"Inferred K={K} and P={P} exceed tokens={T}.")

    grid = attn_tok[0, K:K + P].reshape(side, side)

    attn_map = F.interpolate(
        grid.unsqueeze(0).unsqueeze(0),
        size=(x.shape[2], x.shape[3]),  # match model input spatial size
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    return attn_map


# -----------------------------
# Overlay and save
# -----------------------------
def plot_attention_overlay_rgb(rgb_uint8: np.ndarray, attn_map: np.ndarray, save_path: str):
    img = rgb_uint8.astype(np.float32) / 255.0
    attn = attn_map.astype(np.float32)
    a_min, a_max = float(attn.min()), float(attn.max())
    if a_max > a_min:
        attn = (attn - a_min) / (a_max - a_min)
    else:
        attn.fill(0.0)

    plt.figure(figsize=(5, 5))
    plt.imshow(img, interpolation="nearest")
    plt.imshow(attn, cmap="jet", alpha=0.5, interpolation="nearest")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


# -----------------------------
# Main loop over H5
# -----------------------------
def generate_attention_from_h5(
    h5_path: str,
    out_dir="attn_maps",
    device="cuda",
    max_tiles=50,
    use_hoptimus=True,   # set False to use Conch loader instead
    layer_idx=-1,
    head_idx=None
):
    os.makedirs(out_dir, exist_ok=True)

    # Load FM + tfm
    if use_hoptimus:
        model, tfm, _ = load_hoptimus1(device=device, prefer_hf=True)
    else:
        from lora_conch_loader import load_conch_v15
        model, tfm, _ = load_conch_v15(device=device, prefer_hf=True)

    model.eval()
    param_dtype = next(model.parameters()).dtype  # match model params (fp16 on GPU per your loader, fp32 on CPU)

    # Read tiles (accept 'images' or 'imgs')
    with h5py.File(h5_path, "r") as f:
        if "images" in f:
            tiles = f["images"][:]
        elif "imgs" in f:
            tiles = f["imgs"][:]
        else:
            raise KeyError("H5 must contain 'images' or 'imgs' dataset.")

    N = min(max_tiles, len(tiles))
    for idx in range(N):
        tile = tiles[idx]

        # Convert to PIL
        if tile.ndim == 3 and tile.shape[-1] in (1, 3, 4):         # (H, W, C)
            img_pil = Image.fromarray(tile[..., :3].astype(np.uint8))
        elif tile.ndim == 3 and tile.shape[0] in (1, 3, 4):        # (C, H, W)
            img_pil = Image.fromarray(np.transpose(tile[:3], (1, 2, 0)).astype(np.uint8))
        elif tile.ndim == 2:                                       # grayscale
            img_pil = Image.fromarray(tile.astype(np.uint8)).convert("RGB")
        else:
            raise ValueError(f"Unsupported tile shape: {tile.shape}")

        # ✅ Force-resize to model's img_size, then apply tfm
        x, vis_rgb = preprocess_for_model(img_pil, model, tfm, device, dtype=param_dtype)

        # Attention map
        attn_map = get_attention_map(model, x, device, layer_idx=layer_idx, head_idx=head_idx)

        # Save
        save_path = os.path.join(out_dir, f"tile{idx}_L{layer_idx}_H{head_idx if head_idx is not None else 'avg'}.png")
        plot_attention_overlay_rgb(vis_rgb, attn_map, save_path)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{N} tiles")

    print("✅ Done! Attention maps saved to", out_dir)


if __name__ == "__main__":
    h5_path = "/scratch/pioneer/users/sxk2517/kich_patch_images/TCGA-KL-8323-01Z-00-DX1.01d72f6a-cc87-4082-8af4-738250ec4d9c_labeled.h5"
    generate_attention_from_h5(
        h5_path,
        out_dir="hoptimus_attn_maps",
        device="cuda",
        max_tiles=100,
        use_hoptimus=True,   # set False to switch to Conch
        layer_idx=-1,
        head_idx=None
    )
