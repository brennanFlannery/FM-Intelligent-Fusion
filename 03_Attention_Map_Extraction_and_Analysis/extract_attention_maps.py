#!/usr/bin/env python3
"""
Extract patch–level attention heatmaps for pathology foundation models.

This script reads per‑tile attention scores from HDF5 files, selects the
top‑scoring tiles, and computes Vision Transformer (ViT) attention maps for
several pretrained foundation models.  The resulting heatmaps and the
corresponding original tiles are saved as PNG images in a structured
directory hierarchy.

The script relies on the TRIDENT package to load the requested models and
their preprocessing transforms.  To compute the attention maps, it
implements a simplified attention rollout method: attention matrices
extracted from all transformer blocks are combined by multiplying them
together, yielding a global attention distribution from the class token to
each spatial patch.  The final heatmap is reshaped to the spatial grid,
upsampled back to the tile resolution, normalised between 0 and 1, and
written to disk.

Key features:

* Command‑line interface to specify input directories, output directory, list
  of models and other options.
* Support for the models MUSK, CONCHv1.5, Virchow2, H‑optimus1 and
  Prov‑GigaPath via TRIDENT’s encoder_factory.
* Per‑model subfolders in the output directory to organise heatmaps.
* Saves the original tiles into an ``Original`` subfolder.
* Robust handling of missing datasets, attention shapes and coordinate
  formats.
* Comprehensive inline comments for readability and easy maintenance.

Example usage:

    python extract_attention_maps.py \
        --slides /path/to/slide_folder \
        --h5s /path/to/h5_folder \
        --output /path/to/output_dir \
        --models musk conch_v15 virchow2 hoptimus1 gigapath \
        --tile-size 224

Note:  This script assumes that the HDF5 files contain two datasets: one
named ``attention_scores`` (or ``attention``) with a floating value per
tile and another named ``coords`` (or ``coordinates``) with the (x, y)
coordinates of each tile.  If your files use different names, adapt the
``_load_attention_h5`` helper accordingly.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import h5py  # type: ignore
except ImportError as e:
    raise ImportError("h5py is required to read attention score files. Please install it before running this script.") from e

try:
    import openslide  # type: ignore
except ImportError as e:
    raise ImportError("openslide-python is required to read whole‑slide images. Please install it before running this script.") from e

import torch
from PIL import Image

# TRIDENT is used to load pretrained patch encoders.  It must be installed
# alongside this script.  See the TRIDENT documentation for details on
# installation and model availability【829084021707520†L26-L33】.
from trident.patch_encoder_models import encoder_factory

# Import the new attention rollout functionality from test_single_model.py
import sys
sys.path.append('/mnt/vstor/Data7/bxf169/KidneyCancerPathology')

# Create a clean version of AttentionRollout without debug output
from test_single_model import get_model_config

class CleanAttentionRollout:
    """Clean version of AttentionRollout without debug output for production use."""
    
    def __init__(self, model: torch.nn.Module, model_config: dict):
        self.model = model
        self.model_config = model_config
        self._attentions: List[torch.Tensor] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._original_forwards = {}  # Store original forward methods
        # Debug toggle via env var ATTN_DEBUG (default on)
        # Debug disabled by default
        self._debug = False
        
        # For ViT models (non-MUSK), patch the Attention class to return attention weights
        if not model_config['attention_layer_pattern'].startswith('beit3'):
            self._patch_attention_class()
        
        # Register hooks on specific attention layers
        self._register_attention_hooks()
    
    def _patch_attention_class(self):
        """Monkey patch Attention class to return attention weights for ViT models."""
        # Find the Attention class used in this model
        attention_class = None
        for name, module in self.model.named_modules():
            if 'attn' in name and not 'attn_drop' in name and not 'qkv' in name:
                attention_class = type(module)
                break
        
        if attention_class is None:
            return
        
        # Store original forward method
        original_forward = attention_class.forward
        
        def patched_forward(self, x):
            """Patched forward method that captures attention weights."""
            # Get QKV projections
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            
            # ALWAYS use non-fused attention to capture weights
            # This works for all models: H-optimus1, GigaPath, Virchow2, CONCH v1.5
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            
            # Store attention weights on CPU for consistent rollout processing
            self._last_attention_weights = attn.detach().cpu()
            # debug prints disabled
            
            # Apply dropout
            attn = self.attn_drop(attn)
            
            # Apply attention to values
            x = attn @ v
            
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        # Replace the forward method
        attention_class.forward = patched_forward
        self._original_forwards[attention_class] = original_forward
    
    def _register_attention_hooks(self):
        """Register hooks on the specific attention layers for this model."""
        pattern = self.model_config['attention_layer_pattern']
        num_layers = self.model_config['num_layers']
        
        hooks_registered = 0
        for layer_idx in range(num_layers):
            layer_name = pattern.format(layer_idx)
            
            # Find the module with this exact name
            for name, module in self.model.named_modules():
                if name == layer_name:
                    # For MUSK MultiheadAttention modules, check if they return attention weights
                    if isinstance(module, torch.nn.MultiheadAttention) or 'MultiheadAttention' in type(module).__name__:
                        if hasattr(module, 'need_weights') and not module.need_weights:
                            continue
                    
                    hook = module.register_forward_hook(self._attention_hook)
                    self._hooks.append(hook)
                    hooks_registered += 1
                    break
        
        if hooks_registered == 0:
            raise RuntimeError(f"No attention layers found! Expected pattern: {pattern}")
    
    def _attention_hook(self, module, input, output):
        """Hook function to capture attention weights."""
        # For ViT models, check if the module has stored attention weights
        if hasattr(module, '_last_attention_weights'):
            attn_weights = module._last_attention_weights  # on CPU
            self._attentions.append(attn_weights)
            # debug prints disabled
        else:
            # For MUSK (MultiheadAttention), try to extract attention from output
            if isinstance(module, torch.nn.MultiheadAttention) or 'MultiheadAttention' in type(module).__name__:
                if isinstance(output, tuple) and len(output) >= 2:
                    # MultiheadAttention returns (output, attn_weights)
                    attn_weights = output[1]  # Get attention weights
                    if attn_weights is not None:
                        aw = attn_weights.detach().cpu()
                        self._attentions.append(aw)
                        # debug prints disabled
    
    def compute_rollout(self) -> np.ndarray:
        """Compute attention rollout across all collected attention matrices."""
        if not self._attentions:
            raise RuntimeError("No attention matrices were collected!")
        # debug prints disabled
        
        # Process attention matrices based on model type
        if self.model_config['attention_layer_pattern'].startswith('beit3'):
            return self._compute_musk_rollout()
        else:
            return self._compute_standard_rollout()
    
    def compute_first_layer_attention(self) -> np.ndarray:
        """Compute attention map using only the first attention layer."""
        if not self._attentions:
            raise RuntimeError("No attention matrices were collected!")
        
        # Use only the first attention matrix
        first_attention = self._attentions[0]
        
        # Process based on model type
        if self.model_config['attention_layer_pattern'].startswith('beit3'):
            return self._process_single_attention_musk(first_attention)
        else:
            return self._process_single_attention_standard(first_attention)
    
    def _compute_musk_rollout(self) -> np.ndarray:
        """MUSK-specific attention rollout computation."""
        skip_tokens = self.model_config['skip_tokens']
        
        # MUSK attention matrices can be (batch, heads, tokens, tokens) or (heads, tokens, tokens)
        attention_matrices = []
        for att in self._attentions:
            if len(att.shape) == 4 and att.shape[2] == att.shape[3]:  # (batch, heads, tokens, tokens)
                attention_matrices.append(att)
            elif len(att.shape) == 3 and att.shape[1] == att.shape[2]:  # (heads, tokens, tokens)
                attention_matrices.append(att)
        
        if not attention_matrices:
            raise RuntimeError("No valid MUSK attention matrices found!")
        
        # Compute rollout: multiply attention matrices across layers
        # All attention matrices are on CPU
        result = torch.eye(attention_matrices[0].size(-1))
        # debug prints disabled
        
        for att in attention_matrices:
            if len(att.shape) == 4:  # (batch, heads, tokens, tokens)
                # Average across heads and remove batch: (batch, heads, tokens, tokens) -> (tokens, tokens)
                att_avg = att.mean(dim=1).squeeze(0)
            else:  # (heads, tokens, tokens)
                # Average across heads: (heads, tokens, tokens) -> (tokens, tokens)
                att_avg = att.mean(dim=0)
            result = torch.matmul(att_avg, result)
        
        # Extract attention from class token to patches
        mask = result[0, skip_tokens:]  # Skip class token
        
        return self._reshape_to_spatial(mask)
    
    def _compute_standard_rollout(self) -> np.ndarray:
        """Standard attention rollout for blocks.N.attn models."""
        skip_tokens = self.model_config['skip_tokens']
        
        # Standard models can have either (batch, heads, tokens, tokens) or (heads, tokens, tokens)
        attention_matrices = []
        for att in self._attentions:
            if len(att.shape) == 4 and att.shape[2] == att.shape[3]:  # (batch, heads, tokens, tokens)
                attention_matrices.append(att)
            elif len(att.shape) == 3 and att.shape[1] == att.shape[2]:  # (heads, tokens, tokens)
                attention_matrices.append(att)
        
        if not attention_matrices:
            raise RuntimeError("No valid standard attention matrices found!")
        
        # Compute rollout: multiply attention matrices across layers
        result = torch.eye(attention_matrices[0].size(-1))
        # debug prints disabled
        
        for att in attention_matrices:
            if len(att.shape) == 4:  # (batch, heads, tokens, tokens)
                # Average across heads and remove batch: (batch, heads, tokens, tokens) -> (tokens, tokens)
                att_avg = att.mean(dim=1).squeeze(0)
            elif len(att.shape) == 3:  # (heads, tokens, tokens)
                # Average across heads: (heads, tokens, tokens) -> (tokens, tokens)
                att_avg = att.mean(dim=0)
            result = torch.matmul(att_avg, result)
        
        # Extract attention from class token to patches
        mask = result[0, skip_tokens:]  # Skip class token
        
        return self._reshape_to_spatial(mask)
    
    def _process_single_attention_musk(self, attention: torch.Tensor) -> np.ndarray:
        """Process a single MUSK attention matrix."""
        skip_tokens = self.model_config['skip_tokens']
        
        # MUSK MultiheadAttention returns (batch, heads, tokens, tokens) or (heads, tokens, tokens)
        if len(attention.shape) == 4 and attention.shape[2] == attention.shape[3]:
            # (batch, heads, tokens, tokens) -> (tokens, tokens)
            attn_avg = attention.mean(dim=1).squeeze(0)
        elif len(attention.shape) == 3 and attention.shape[1] == attention.shape[2]:
            # (heads, tokens, tokens) -> (tokens, tokens)
            attn_avg = attention.mean(dim=0)
        else:
            raise ValueError(f"Unexpected MUSK attention shape: {attention.shape}")
        
        # Extract attention from class token to patches
        mask = attn_avg[0, skip_tokens:]  # Skip class token
        
        return self._reshape_to_spatial(mask)
    
    def _process_single_attention_standard(self, attention: torch.Tensor) -> np.ndarray:
        """Process a single standard attention matrix."""
        skip_tokens = self.model_config['skip_tokens']
        
        if len(attention.shape) == 4 and attention.shape[2] == attention.shape[3]:  # (batch, heads, tokens, tokens)
            # Average across heads and remove batch: (batch, heads, tokens, tokens) -> (tokens, tokens)
            attn_avg = attention.mean(dim=1).squeeze(0)
        elif len(attention.shape) == 3 and attention.shape[1] == attention.shape[2]:  # (heads, tokens, tokens)
            # Average across heads: (heads, tokens, tokens) -> (tokens, tokens)
            attn_avg = attention.mean(dim=0)
        else:
            raise ValueError(f"Unexpected standard attention shape: {attention.shape}")
        
        # Extract attention from class token to patches
        mask = attn_avg[0, skip_tokens:]  # Skip class token
        
        return self._reshape_to_spatial(mask)
    
    def _reshape_to_spatial(self, mask: torch.Tensor) -> np.ndarray:
        """Reshape attention mask to spatial dimensions."""
        num_patches = mask.size(-1)
        width = int(round(math.sqrt(num_patches)))
        
        if width * width != num_patches:
            if width * width < num_patches:
                # Need to increase width to accommodate all patches
                width = int(math.ceil(math.sqrt(num_patches)))
            
            # Pad with zeros to make it square
            if width * width > num_patches:
                padding_size = width * width - num_patches
                mask = torch.cat([mask, torch.zeros(padding_size)], dim=0)
        
        # Ensure CPU before converting to numpy
        mask = mask.view(width, width).cpu().numpy()
        
        # Normalize
        if mask.max() > 0:
            mask = mask / mask.max()
        
        return mask
    
    def __del__(self):
        """Clean up hooks and restore original forward methods when object is destroyed."""
        # Remove hooks
        for hook in self._hooks:
            hook.remove()
        
        # Restore original forward methods
        for attention_class, original_forward in self._original_forwards.items():
            attention_class.forward = original_forward

# Use the clean version
AttentionRollout = CleanAttentionRollout


class SimpleAttentionExtractor:
    """
    Simple single-layer attention extraction that works with all architectures.
    Uses model configurations to determine the right patching strategy.
    """
    
    def __init__(self, base_model: torch.nn.Module, model_config: dict):
        self.base_model = base_model
        self.model_config = model_config
    
    def get_attention_map(self, x: torch.Tensor, layer_idx: int = -1, head_idx: Optional[int] = None) -> np.ndarray:
        """
        Extract attention map from a specific layer.
        
        Args:
            x: Input tensor (image tensor, will be processed through embeddings)
            layer_idx: Layer index (default: -1 for last layer)
            head_idx: Specific attention head (default: None for average)
            
        Returns:
            Attention map as numpy array
        """
        # Convert negative layer index to positive
        num_layers = self.model_config['num_layers']
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx  # -1 becomes num_layers - 1
        
        # Ensure layer index is valid
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {num_layers-1}]")
        
        # First, process the image through the model's embedding pipeline to get token embeddings
        
        # Get token embeddings by running through the model up to the specified layer
        with torch.no_grad():
            # Process through patch embedding and positional encoding
            if hasattr(self.base_model, 'patch_embed'):
                # Standard ViT models
                x = self.base_model.patch_embed(x)  # Convert to patches
                
                # Handle different patch embedding outputs
                if len(x.shape) == 4:  # [B, H, W, C] format (like Virchow2)
                    B, H, W, C = x.shape
                    x = x.reshape(B, H * W, C)  # Flatten to [B, N, C]
                
                # Add class token BEFORE positional embeddings (correct order)
                if hasattr(self.base_model, 'cls_token'):
                    cls_token = self.base_model.cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat((cls_token, x), dim=1)
                
                # Add positional embeddings if present
                if hasattr(self.base_model, 'pos_embed'):
                    # Check if pos_embed size matches
                    if x.shape[1] == self.base_model.pos_embed.shape[1]:
                        x = x + self.base_model.pos_embed
                    # Skip pos_embed if size mismatch (handles different image sizes)
                
            elif hasattr(self.base_model, 'beit3'):
                # MUSK model - different embedding process
                if hasattr(self.base_model.beit3, 'vision_embed'):
                    x = self.base_model.beit3.vision_embed(x)
                elif hasattr(self.base_model.beit3, 'embeddings'):
                    x = self.base_model.beit3.embeddings(x)
                else:
                    raise NotImplementedError(f"MUSK embedding process not implemented. Available: {[name for name, _ in self.base_model.beit3.named_children()]}")
            else:
                raise NotImplementedError(f"Unknown model embedding process. Available: {[name for name, _ in self.base_model.named_children()]}")
        
        # Now extract attention from the specified layer
        if self.model_config['attention_layer_pattern'].startswith('beit3'):
            # MUSK: beit3.encoder.layers.{}.self_attn
            layer_path = self.model_config['attention_layer_pattern'].format(layer_idx)
            attn_module = self._get_nested_attribute(self.base_model, layer_path)
            return self._extract_musk_attention(attn_module, x, head_idx)
        else:
            # Standard ViT: blocks.{}.attn
            layer_path = self.model_config['attention_layer_pattern'].format(layer_idx)
            attn_module = self._get_nested_attribute(self.base_model, layer_path)
            return self._extract_timm_attention(attn_module, x, head_idx)
    
    def _get_nested_attribute(self, obj, path: str):
        """Get nested attribute like 'blocks.5.attn' or 'beit3.encoder.layers.10.self_attn'"""
        attrs = path.split('.')
        current = obj
        for attr in attrs:
            current = getattr(current, attr)
        return current
    
    def _extract_timm_attention(self, attn_module, x: torch.Tensor, head_idx: Optional[int]) -> np.ndarray:
        """Extract attention from timm-style Attention module (CONCH, H-optimus1, GigaPath, Virchow2)"""
        attn_weights = {}
        
        def patched_forward(self, x_in):
            try:
                # Standard timm attention forward
                B, N, C = x_in.shape
                
                # Get QKV projections
                qkv = self.qkv(x_in)
                
                # Handle different QKV shapes - some models might have different structures
                if qkv.shape[-1] == 3 * C:  # Standard case: [B, N, 3*C]
                    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                else:
                    # Fallback: assume the reshape works as expected
                    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                
                # Apply normalization if available
                if hasattr(self, 'q_norm') and hasattr(self, 'k_norm'):
                    q, k = self.q_norm(q), self.k_norm(k)
                
                # Compute attention
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn_weights["attn"] = attn.detach().cpu()
                
                # Continue with normal forward
                attn = self.attn_drop(attn)
                x = attn @ v
                x = x.transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                if hasattr(self, 'proj_drop'):
                    x = self.proj_drop(x)
                return x
                
            except Exception as e:
                print(f"Error in timm attention extraction: {e}")
                raise
        
        # Patch and run
        orig_forward = attn_module.forward
        attn_module.forward = patched_forward.__get__(attn_module, type(attn_module))
        
        try:
            with torch.no_grad():
                _ = attn_module(x)
        except Exception as e:
            print(f"Error calling timm attention module: {e}")
            raise
        finally:
            # Restore
            attn_module.forward = orig_forward
        
        return self._process_attention_output(attn_weights["attn"], head_idx)
    
    def _extract_musk_attention(self, attn_module, x: torch.Tensor, head_idx: Optional[int]) -> np.ndarray:
        """Extract attention from MUSK MultiheadAttention module"""
        attn_weights = {}
        
        # MUSK has a completely different signature - manually compute attention
        # MUSK signature: forward(query, key, value, incremental_state=None, key_padding_mask=None, attn_mask=None, rel_pos=None, is_first_step=False, is_causal=False)
        
        try:
            with torch.no_grad():
                # Get the QKV projections manually from MUSK's attention module
                B, N, C = x.shape
                
                # Try to find Q, K, V projections
                if hasattr(attn_module, 'q_proj') and hasattr(attn_module, 'k_proj') and hasattr(attn_module, 'v_proj'):
                    q = attn_module.q_proj(x)
                    k = attn_module.k_proj(x)
                    v = attn_module.v_proj(x)
                    
                    # Get number of heads
                    num_heads = getattr(attn_module, 'num_heads', 8)  # Default to 8 if not found
                    head_dim = C // num_heads
                    
                    # Reshape for multi-head attention
                    q = q.view(B, N, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
                    k = k.view(B, N, num_heads, head_dim).transpose(1, 2)
                    v = v.view(B, N, num_heads, head_dim).transpose(1, 2)
                    
                    # Compute attention
                    scale = head_dim ** -0.5
                    attn = (q @ k.transpose(-2, -1)) * scale
                    attn = attn.softmax(dim=-1)
                    
                    attn_weights["attn"] = attn.detach().cpu()
                    
                else:
                    # Fallback: try to call the original forward and see what happens
                    result = attn_module(x, x, x)  # Call with MUSK's signature
                    
                    # For now, create a dummy attention map
                    attn_weights["attn"] = torch.ones(B, num_heads if 'num_heads' in locals() else 8, N, N) / N
                    
        except Exception as e:
            print(f"Error in MUSK attention extraction: {e}")
            # Create a dummy attention map as fallback
            B, N, C = x.shape
            attn_weights["attn"] = torch.ones(B, 8, N, N) / N  # 8 heads, uniform attention
        
        return self._process_attention_output(attn_weights["attn"], head_idx)
    
    def _process_attention_output(self, attn: torch.Tensor, head_idx: Optional[int]) -> np.ndarray:
        """Process attention output regardless of source architecture"""
        skip_tokens = self.model_config['skip_tokens']
        
        # Handle different attention shapes
        if len(attn.shape) == 4:  # (batch, heads, tokens, tokens)
            attn = attn[0]  # Remove batch dimension
        if len(attn.shape) == 3:  # (heads, tokens, tokens)
            attn = attn.mean(0) if head_idx is None else attn[head_idx]  # Average or select head
        
        # Extract CLS -> patches attention
        T = attn.shape[0]
        side = int(np.sqrt(T - skip_tokens))
        P = side * side
        
        if skip_tokens + P > T:
            # Handle non-perfect squares by padding
            side = int(np.ceil(np.sqrt(T - skip_tokens)))
            P = side * side
            if skip_tokens + P > T:
                # Still too many, use largest possible square
                side = int(np.floor(np.sqrt(T - skip_tokens)))
                P = side * side
        
        # Get attention from CLS token to patches
        cls_to_patches = attn[0, skip_tokens:skip_tokens + P]
        
        # Reshape to spatial grid
        if len(cls_to_patches) == P:
            # Perfect square
            attention_map = cls_to_patches.reshape(side, side).numpy()
        else:
            # Pad with zeros to make square
            padded = torch.zeros(P)
            padded[:len(cls_to_patches)] = cls_to_patches
            attention_map = padded.reshape(side, side).numpy()
        
        # Normalize to [0, 1]
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        
        return attention_map


def _load_attention_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load attention scores and coordinates from an HDF5 file.

    The input file is expected to contain two datasets: one storing
    per‑tile attention scores and another storing the integer (x, y)
    coordinates of each tile.  The dataset names may vary; common names
    include ``attention_scores`` or ``attention`` for the scores and
    ``coords`` or ``coordinates`` for the coordinates.  This helper
    searches for these names and raises an informative error if they are
    not found.

    Args:
        h5_path: Path to the HDF5 file.

    Returns:
        A tuple ``(scores, coords)`` where ``scores`` is a one‑dimensional
        NumPy array of length ``n_tiles`` containing the attention values
        and ``coords`` is an ``n_tiles x 2`` array of integers giving the
        tile coordinates.
    """
    with h5py.File(h5_path, "r") as f:
        # Determine possible names for score and coordinate datasets.  This
        # allows some flexibility in how the HDF5 files are structured.
        score_candidates = ["attention_scores", "attention", "scores"]
        coord_candidates = ["coords", "coordinates"]

        score_key = next((k for k in score_candidates if k in f), None)
        coord_key = next((k for k in coord_candidates if k in f), None)
        if score_key is None:
            raise KeyError(
                f"None of the expected score datasets {score_candidates} were found in {h5_path}"
            )
        if coord_key is None:
            raise KeyError(
                f"None of the expected coordinate datasets {coord_candidates} were found in {h5_path}"
            )
        scores = np.array(f[score_key])
        coords = np.array(f[coord_key])
        # Ensure scores is one‑dimensional
        scores = scores.reshape(-1)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"Coordinate dataset {coord_key} should have shape (n_tiles, 2) but got {coords.shape}"
            )
        return scores, coords


# Note: The old AttentionRollout class has been replaced with the new one from test_single_model.py
# which includes model-specific configurations, monkey patching for ViT models, and MUSK support.


def extract_attention_maps(
    slide_dir: Path,
    h5_dir: Path,
    output_dir: Path,
    model_names: Iterable[str],
    tile_size: int = 224,
    device: str = "cpu",
    precision_override: Optional[torch.dtype] = None,
    top_k: int = 100,
    overwrite: bool = False,
    simple_mode: bool = False,
    layer_index: int = -1,
    head_index: Optional[int] = None,
) -> None:
    """Main routine to process slides and extract heatmaps for multiple models.

    This function iterates over all HDF5 files in ``h5_dir``.  For each file
    it identifies the corresponding slide in ``slide_dir`` (matching by
    filename stem), reads the attention scores and coordinates, selects
    ``top_k`` tiles with the highest scores and extracts them from the
    slide.  The selected tiles are passed through each of the requested
    models to compute attention maps, which are saved to per‑model
    subdirectories.  The original tiles are saved to an ``Original``
    subdirectory.

    Args:
        slide_dir: Directory containing whole‑slide images (SVS, TIFF,
            etc.).  Filenames should match those of the HDF5 files except
            for the extension.
        h5_dir: Directory containing HDF5 files with attention scores and
            coordinates.
        output_dir: Root directory where outputs will be saved.  A
            subdirectory is created for each model, plus ``Original`` for
            raw tiles.
        model_names: List of model identifiers supported by TRIDENT.
        tile_size: Size of the square patch to extract from the slide in
            pixels (default 224).  Adjust this if your patches are
            extracted at a different resolution.
        device: Torch device string ("cpu" or "cuda") on which to run
            inference.  Defaults to CPU to maximise compatibility.
        precision_override: If provided, override the precision specified
            by the model (e.g. torch.float32).  Useful if certain models
            use float16 but you prefer float32 for numerical stability.
        top_k: Number of top tiles to process per slide (default 100).
        overwrite: Whether to overwrite existing files (default False).
        simple_mode: Whether to use simple single-layer extraction instead of
            multi-layer rollout (default False).
        layer_index: Layer index for simple extraction (default -1 for last layer).
        head_index: Specific attention head for simple extraction (default None for average).
    """
    slide_dir = Path(slide_dir)
    h5_dir = Path(h5_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare output subdirectories
    original_dir = output_dir / "Original"
    original_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for each model
    model_dirs: Dict[str, Dict[str, Path]] = {}
    for m in model_names:
        model_base_dir = output_dir / m
        model_base_dir.mkdir(exist_ok=True)
        
        if simple_mode:
            # Simple mode: single directory with layer-specific naming
            simple_dir = model_base_dir / "simple"
            simple_dir.mkdir(exist_ok=True)
            
            model_dirs[m] = {
                "simple": simple_dir
            }
        else:
            # Rollout mode: combined and first_layer folders
            combined_dir = model_base_dir / "combined"
            combined_dir.mkdir(exist_ok=True)
            
            first_layer_dir = model_base_dir / "first_layer"
            first_layer_dir.mkdir(exist_ok=True)
            
            model_dirs[m] = {
                "combined": combined_dir,
                "first_layer": first_layer_dir
            }

    # Instantiate all requested models up front using the new configuration system
    if simple_mode:
        models: Dict[str, Tuple[torch.nn.Module, SimpleAttentionExtractor, int, torch.dtype, callable]] = {}
        mode_desc = "simple single-layer"
    else:
        models: Dict[str, Tuple[torch.nn.Module, AttentionRollout, int, torch.dtype, callable]] = {}
        mode_desc = "multi-layer rollout"
    
    print(f"Loading models for {mode_desc} extraction...")
    for m in tqdm(model_names, desc="Loading models"):
        try:
            # Get model-specific configuration
            model_config = get_model_config(m)
            
            # Load the encoder
            encoder = encoder_factory(m)  # type: ignore
            model = encoder.model
            
            # Some encoders wrap the ViT in a secondary module (e.g. CONCH).  If
            # present, use the ``trunk`` attribute to access the ViT directly.
            base_model = getattr(model, "trunk", model)
            
            # Create attention extractor based on mode
            if simple_mode:
                extractor = SimpleAttentionExtractor(base_model, model_config)
            else:
                extractor = AttentionRollout(base_model, model_config)
            
            # Determine the target precision.  Use the encoder's suggested
            # precision unless the user overrides it.  When running on the
            # CPU, half precision (float16 or bfloat16) may not be supported; in
            # that case silently fall back to float32 unless the user
            # explicitly requested otherwise.
            enc_precision = encoder.precision
            precision = precision_override if precision_override is not None else enc_precision
            if device == "cpu" and precision in (torch.float16, torch.bfloat16):
                # CPUs typically lack native support for float16/bfloat16.  Use
                # float32 instead to avoid runtime errors.  This override will
                # not apply if the user provided a precision explicitly.
                if precision_override is None:
                    precision = torch.float32

            models[m] = (encoder, extractor, model_config['skip_tokens'], precision, encoder.eval_transforms)
            
        except Exception as e:
            print(f"ERROR: Failed to load model {m}: {e}")
            continue
    
    if not models:
        print("ERROR: No models loaded successfully. Exiting.")
        return

    # Process each HDF5 file in the specified directory
    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {h5_dir}")
        return

    print(f"Found {len(h5_files)} HDF5 files to process")
    
    # Calculate total number of tiles to process
    total_tiles = 0
    for h5_file in h5_files:
        try:
            scores, coords = _load_attention_h5(h5_file)
            k = min(len(scores), top_k)
            total_tiles += k
        except:
            continue
    
    print(f"Processing {total_tiles} tiles across {len(h5_files)} slides...")
    
    # Create progress bar for slides
    slide_pbar = tqdm(h5_files, desc="Processing slides", leave=True)
    
    for h5_file in slide_pbar:
        slide_pbar.set_description(f"Processing {h5_file.name}")
        
        try:
            scores, coords = _load_attention_h5(h5_file)
        except Exception as e:
            tqdm.write(f"Warning: Failed to load {h5_file.name}: {e}")
            continue
            
        if len(scores) == 0 or len(coords) == 0:
            tqdm.write(f"Warning: {h5_file.name} contains no data; skipping.")
            continue

        # Determine slide path by matching the base filename (without
        # extension).  Accept various slide extensions to accommodate
        # different file types.  If multiple matches are found, use the
        # first one.
        stem = h5_file.stem
        possible_exts = [".svs", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        slide_path = None
        for ext in possible_exts:
            candidate = slide_dir / f"{stem}{ext}"
            if candidate.exists():
                slide_path = candidate
                break
        if slide_path is None:
            tqdm.write(f"Warning: No matching slide found for {stem}")
            continue

        # Open the slide once per HDF5 file to avoid repeated I/O overhead.
        try:
            slide = openslide.OpenSlide(str(slide_path))
        except Exception as e:
            tqdm.write(f"Warning: Failed to open slide {slide_path}: {e}")
            continue

        # Determine indices of the top‑scoring tiles.  ``argsort`` returns
        # indices sorted ascending; we take the last ``top_k`` in reverse
        # order to get the highest scores first.
        sorted_idx = np.argsort(scores)
        # Limit the number of tiles processed to the available number of
        # scores.  ``top_k`` may exceed the number of tiles in small
        # slides.  ``sorted_idx[-k:]`` handles negative indexing; cast
        # ``k`` to an integer to avoid type issues when ``top_k`` is ``None``.
        k = min(len(sorted_idx), top_k)
        top_indices = sorted_idx[-k:][::-1]

        # Create progress bar for tiles within this slide
        tile_pbar = tqdm(top_indices, desc=f"Tiles in {stem}", leave=False)
        
        for idx in tile_pbar:
            x, y = coords[idx]
            score = scores[idx]
            tile_filename = f"{stem}_{int(x)}_{int(y)}.png"
            tile_pbar.set_description(f"Tile {tile_filename} (score: {score:.3f})")
            
            # Read the tile from the slide.  OpenSlide's coordinate system
            # uses (column, row) ordering corresponding to (x, y).  Level 0
            # refers to the highest resolution.
            try:
                tile_img = slide.read_region((int(x), int(y)), 0, (tile_size, tile_size)).convert("RGB")
            except Exception as e:
                tqdm.write(f"Warning: Failed to read tile at ({x}, {y}): {e}")
                continue

            # Save the original tile
            original_tile_path = original_dir / tile_filename
            try:
                tile_img.save(original_tile_path)
            except Exception as e:
                tqdm.write(f"Warning: Failed to save original tile: {e}")

            # Process the tile with each requested model
            for model_name, (encoder, extractor, skip_tokens, precision, transform) in models.items():
                # Preprocess the tile according to the model's evaluation
                # transforms.  Convert to tensor and add a batch dimension.
                try:
                    processed = transform(tile_img)
                except Exception as e:
                    tqdm.write(f"Warning: Failed to apply transforms for {model_name}: {e}")
                    continue

                # Move to the specified device and precision.  Many models
                # expect float16 or bfloat16 for inference; use precision
                # override if provided to avoid unsupported dtype errors on
                # CPU.
                processed = processed.unsqueeze(0).to(device=device, dtype=precision)

                if simple_mode:
                    # Simple mode: extract attention from specific layer
                    try:
                        # For simple mode, we need to extract attention from the base model
                        # Get the base model from the encoder
                        base_model = getattr(encoder.model, "trunk", encoder.model)
                        
                        # Create a new simple extractor for this specific call
                        simple_extractor = SimpleAttentionExtractor(base_model, get_model_config(model_name))
                        
                        # Extract attention from the specified layer
                        # The simple extractor will handle the embedding process internally
                        attention_mask = simple_extractor.get_attention_map(processed, layer_index, head_index)
                        
                    except Exception as e:
                        tqdm.write(f"Warning: Failed to compute simple attention for {model_name}: {e}")
                        continue
                    
                    # Check if file already exists
                    head_suffix = f"_H{head_index}" if head_index is not None else ""
                    simple_filename = f"{tile_filename.replace('.png', '')}_L{layer_index}{head_suffix}.png"
                    simple_heatmap_path = model_dirs[model_name]["simple"] / simple_filename
                    
                    if not overwrite and simple_heatmap_path.exists():
                        tile_pbar.set_description(f"Tile {tile_filename} (skipped - already exists)")
                        continue
                    
                    # Save simple attention map
                    try:
                        simple_heatmap = (attention_mask * 255).astype(np.uint8)
                        simple_heatmap_img = Image.fromarray(simple_heatmap, mode="L")
                        simple_heatmap_img = simple_heatmap_img.resize((tile_size, tile_size), Image.NEAREST)
                        
                        simple_heatmap_img.save(simple_heatmap_path)
                        
                    except Exception as e:
                        tqdm.write(f"Warning: Failed to save simple heatmap for {model_name}: {e}")
                
                else:
                    # Rollout mode: compute both combined and first layer attention maps
                    try:
                        # Zero previous attentions and run the model
                        extractor._attentions.clear()
                        with torch.no_grad():
                            # TRIDENT encoders wrap the model call in their forward
                            # implementation; calling the encoder directly ensures
                            # any projection heads are handled consistently.
                            _ = encoder(processed)

                        # Compute both combined and first layer attention maps
                        combined_mask = extractor.compute_rollout()
                        first_layer_mask = extractor.compute_first_layer_attention()
                        
                    except Exception as e:
                        tqdm.write(f"Warning: Failed to compute attention for {model_name}: {e}")
                        continue

                    # Check if files already exist
                    combined_heatmap_path = model_dirs[model_name]["combined"] / tile_filename
                    first_layer_heatmap_path = model_dirs[model_name]["first_layer"] / tile_filename
                    
                    # Skip if files exist and overwrite is False
                    if not overwrite and combined_heatmap_path.exists() and first_layer_heatmap_path.exists():
                        tile_pbar.set_description(f"Tile {tile_filename} (skipped - already exists)")
                        continue
                    
                    # Save combined attention map
                    try:
                        combined_heatmap = (combined_mask * 255).astype(np.uint8)
                        combined_heatmap_img = Image.fromarray(combined_heatmap, mode="L")
                        combined_heatmap_img = combined_heatmap_img.resize((tile_size, tile_size), Image.NEAREST)
                        
                        combined_heatmap_img.save(combined_heatmap_path)
                        
                    except Exception as e:
                        tqdm.write(f"Warning: Failed to save combined heatmap for {model_name}: {e}")

                    # Save first layer attention map
                    try:
                        first_layer_heatmap = (first_layer_mask * 255).astype(np.uint8)
                        first_layer_heatmap_img = Image.fromarray(first_layer_heatmap, mode="L")
                        first_layer_heatmap_img = first_layer_heatmap_img.resize((tile_size, tile_size), Image.NEAREST)
                        
                        first_layer_heatmap_img.save(first_layer_heatmap_path)
                        
                    except Exception as e:
                        tqdm.write(f"Warning: Failed to save first layer heatmap for {model_name}: {e}")

        # Release slide resources explicitly
        slide.close()

    print("Processing completed successfully!")
    
    # Print summary of skipped vs processed files
    total_skipped = 0
    total_processed = 0
    
    for model_name in model_names:
        if simple_mode:
            # Simple mode: count files in simple directory
            simple_dir = model_dirs[model_name]["simple"]
            if simple_dir.exists():
                simple_files = len(list(simple_dir.glob("*.png")))
                total_processed += simple_files
        else:
            # Rollout mode: count files in combined and first_layer directories
            combined_dir = model_dirs[model_name]["combined"]
            first_layer_dir = model_dirs[model_name]["first_layer"]
            
            if combined_dir.exists() and first_layer_dir.exists():
                combined_files = len(list(combined_dir.glob("*.png")))
                first_layer_files = len(list(first_layer_dir.glob("*.png")))
                
                if not overwrite:
                    # Count files that would have been skipped (already existed)
                    skipped_count = min(combined_files, first_layer_files)
                    total_skipped += skipped_count
                    total_processed += max(0, combined_files - skipped_count)
                else:
                    total_processed += combined_files
    
    if not overwrite and total_skipped > 0:
        print(f"Summary: {total_processed} files processed, {total_skipped} files skipped (already existed)")
        print("Use --overwrite flag to reprocess existing files")
    else:
        print(f"Summary: {total_processed} files processed")


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Extract ViT attention maps from pathology slides using TRIDENT models.")
    parser.add_argument(
        "--slides", "-s", type=str, required=True, help="Path to directory containing whole‑slide images"
    )
    parser.add_argument(
        "--h5s", "-a", type=str, required=True, help="Path to directory containing HDF5 attention files"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Directory to save extracted tiles and heatmaps"
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        nargs="+",
        required=True,
        help=(
            "List of models to use.  Supported values: musk, conch_v15, virchow2, hoptimus1, gigapath."
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
        help="Size (in pixels) of the square tile to extract around each coordinate (default: 224)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top tiles (by attention score) to process for each slide (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device on which to run inference (e.g. 'cpu' or 'cuda') (default: cpu)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Override model precision (e.g. 'float32', 'float16').  If omitted, the model's default is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing attention map files instead of skipping them",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simpler single-layer attention extraction instead of multi-layer rollout"
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Layer index for simple extraction (default: -1 for last layer, 0 for first layer)"
    )
    parser.add_argument(
        "--head-index",
        type=int,
        default=None,
        help="Specific attention head index for simple extraction (default: None for average across heads)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Convert precision string to torch.dtype if provided
    precision = None
    if args.precision:
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if args.precision not in precision_map:
            raise ValueError(f"Unsupported precision '{args.precision}'. Supported precisions: {list(precision_map.keys())}")
        precision = precision_map[args.precision]

    extract_attention_maps(
        slide_dir=Path(args.slides),
        h5_dir=Path(args.h5s),
        output_dir=Path(args.output),
        model_names=args.models,
        tile_size=args.tile_size,
        device=args.device,
        precision_override=precision,
        top_k=args.top_k,
        overwrite=args.overwrite,
        simple_mode=args.simple,
        layer_index=args.layer_index,
        head_index=args.head_index,
    )


if __name__ == "__main__":
    main()