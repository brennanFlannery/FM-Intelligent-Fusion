#!/usr/bin/env python3
"""
Simple test script to debug attention extraction for a single model on a single image tile.
This isolates the problem and makes debugging much easier.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import h5py
except ImportError as e:
    raise ImportError("h5py is required. Please install it before running this script.") from e

try:
    import openslide
except ImportError as e:
    raise ImportError("openslide-python is required. Please install it before running this script.") from e

from trident.patch_encoder_models import encoder_factory


def get_model_config(model_name: str) -> dict:
    """Get model-specific configuration for attention extraction."""
    
    configs = {
        "musk": {
            "attention_layer_pattern": "beit3.encoder.layers.{}.self_attn",
            "num_layers": 24,  # layers 0-23
            "skip_tokens": 1,
            "description": "MUSK model - uses beit3.encoder.layers.N.self_attn"
        },
        "conch_v15": {
            "attention_layer_pattern": "blocks.{}.attn",
            "num_layers": 24,  # layers 0-23
            "skip_tokens": 1,
            "description": "CONCH v1.5 model - uses blocks.N.attn"
        },
        "virchow2": {
            "attention_layer_pattern": "blocks.{}.attn",
            "num_layers": 32,  # layers 0-31
            "skip_tokens": 5,  # 4 registration tokens + 1 class token
            "description": "Virchow2 model - uses blocks.N.attn with 4 registration tokens"
        },
        "hoptimus1": {
            "attention_layer_pattern": "blocks.{}.attn",
            "num_layers": 40,  # layers 0-39
            "skip_tokens": 1,
            "description": "H-optimus1 model - uses blocks.N.attn"
        },
        "gigapath": {
            "attention_layer_pattern": "blocks.{}.attn",
            "num_layers": 40,  # layers 0-39
            "skip_tokens": 1,
            "description": "GigaPath model - uses blocks.N.attn"
        }
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(configs.keys())}")
    
    return configs[model_name]


def load_attention_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load attention scores and coordinates from an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        # Find the right datasets
        score_candidates = ["attention", "attention_scores", "scores"]
        coord_candidates = ["coords", "coordinates"]
        
        score_key = next((k for k in score_candidates if k in f), None)
        coord_key = next((k for k in coord_candidates if k in f), None)
        
        if score_key is None:
            raise KeyError(f"None of the expected score datasets {score_candidates} were found in {h5_path}")
        if coord_key is None:
            raise KeyError(f"None of the expected coordinate datasets {coord_candidates} were found in {h5_path}")
        
        scores = np.array(f[score_key]).reshape(-1)
        coords = np.array(f[coord_key])
        
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Coordinate dataset {coord_key} should have shape (n_tiles, 2) but got {coords.shape}")
        
        return scores, coords


class AttentionRollout:
    """Clean attention rollout implementation with specific layer targeting."""
    
    def __init__(self, model: torch.nn.Module, model_config: dict):
        self.model = model
        self.model_config = model_config
        self._attentions: List[torch.Tensor] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._original_forwards = {}  # Store original forward methods
        
        print(f"Setting up attention rollout for: {model_config['description']}")
        
        # For ViT models (non-MUSK), patch the Attention class to return attention weights
        if not model_config['attention_layer_pattern'].startswith('beit3'):
            self._patch_attention_class()
        
        # Register hooks on specific attention layers
        self._register_attention_hooks()
    
    def _patch_attention_class(self):
        """Monkey patch Attention class to return attention weights for ViT models."""
        print("DEBUG: Patching Attention class to return attention weights...")
        
        # Find the Attention class used in this model
        attention_class = None
        for name, module in self.model.named_modules():
            if 'attn' in name and not 'attn_drop' in name and not 'qkv' in name:
                attention_class = type(module)
                break
        
        if attention_class is None:
            print("DEBUG: Could not find Attention class to patch")
            return
        
        print(f"DEBUG: Found Attention class: {attention_class}")
        
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
            
            # Store attention weights for our hooks to capture
            self._last_attention_weights = attn.detach().cpu()
            
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
        
        print(f"DEBUG: Patched {attention_class} forward method")
    
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
                    print(f"Found attention layer: {name}")
                    
                    # For MUSK MultiheadAttention modules, check if they return attention weights
                    if isinstance(module, torch.nn.MultiheadAttention) or 'MultiheadAttention' in type(module).__name__:
                        if hasattr(module, 'need_weights'):
                            print(f"DEBUG: MUSK MultiheadAttention module - need_weights: {module.need_weights}")
                            if not module.need_weights:
                                print(f"DEBUG: WARNING - MultiheadAttention not configured to return weights!")
                                print(f"DEBUG: This will cause attention extraction to fail")
                        else:
                            print(f"DEBUG: MUSK MultiheadAttention module - no need_weights attribute (custom implementation)")
                    
                    hook = module.register_forward_hook(self._attention_hook)
                    self._hooks.append(hook)
                    hooks_registered += 1
                    break
        
        print(f"Registered {hooks_registered} attention hooks")
        
        if hooks_registered == 0:
            raise RuntimeError(f"No attention layers found! Expected pattern: {pattern}")
    
    def _attention_hook(self, module, input, output):
        """Hook function to capture attention weights."""
        print(f"DEBUG: Hook triggered for module: {type(module).__name__}")
        
        # For ViT models, check if the module has stored attention weights
        if hasattr(module, '_last_attention_weights'):
            attn_weights = module._last_attention_weights
            print(f"DEBUG: Found stored attention weights: {attn_weights.shape}")
            self._attentions.append(attn_weights)
        else:
            # For MUSK (MultiheadAttention), try to extract attention from output
            print(f"DEBUG: Module type: {type(module)}")
            print(f"DEBUG: Module name: {type(module).__name__}")
            print(f"DEBUG: Is MultiheadAttention: {isinstance(module, torch.nn.MultiheadAttention)}")
            
            if isinstance(module, torch.nn.MultiheadAttention) or 'MultiheadAttention' in type(module).__name__:
                print(f"DEBUG: Processing MultiheadAttention output")
                print(f"DEBUG: Output type: {type(output)}")
                print(f"DEBUG: Output length: {len(output) if isinstance(output, (tuple, list)) else 'N/A'}")
                
                if isinstance(output, tuple) and len(output) >= 2:
                    # MultiheadAttention returns (output, attn_weights)
                    attn_weights = output[1]  # Get attention weights
                    print(f"DEBUG: Attention weights type: {type(attn_weights)}")
                    print(f"DEBUG: Attention weights value: {attn_weights}")
                    
                    if attn_weights is not None:
                        print(f"DEBUG: Found MultiheadAttention weights: {attn_weights.shape}")
                        self._attentions.append(attn_weights)
                    else:
                        print(f"DEBUG: MultiheadAttention returned None weights")
                else:
                    print(f"DEBUG: Unexpected MultiheadAttention output format: {type(output)}")
                    if isinstance(output, tuple):
                        print(f"DEBUG: Tuple contents: {[type(x) for x in output]}")
            else:
                print(f"DEBUG: No stored attention weights found in module")
        
        print(f"DEBUG: Total attentions collected: {len(self._attentions)}")
    
    def compute_rollout(self) -> np.ndarray:
        """Compute attention rollout across all collected attention matrices."""
        if not self._attentions:
            raise RuntimeError("No attention matrices were collected!")
        
        print(f"Computing rollout from {len(self._attentions)} attention matrices")
        
        # Process attention matrices based on model type
        if self.model_config['attention_layer_pattern'].startswith('beit3'):
            return self._compute_musk_rollout()
        else:
            return self._compute_standard_rollout()
    
    def compute_first_layer_attention(self) -> np.ndarray:
        """Compute attention map using only the first attention layer."""
        if not self._attentions:
            raise RuntimeError("No attention matrices were collected!")
        
        print(f"Computing first layer attention from {len(self._attentions)} attention matrices")
        
        # Use only the first attention matrix
        first_attention = self._attentions[0]
        print(f"DEBUG: Using first attention matrix with shape: {first_attention.shape}")
        
        # Process based on model type
        if self.model_config['attention_layer_pattern'].startswith('beit3'):
            return self._process_single_attention_musk(first_attention)
        else:
            return self._process_single_attention_standard(first_attention)
    
    def _compute_musk_rollout(self) -> np.ndarray:
        """MUSK-specific attention rollout computation."""
        skip_tokens = self.model_config['skip_tokens']
        
        print(f"DEBUG: Processing {len(self._attentions)} MUSK attention tensors")
        for i, att in enumerate(self._attentions):
            print(f"DEBUG: MUSK Attention {i}: shape={att.shape}")
        
        # MUSK attention matrices can be (batch, heads, tokens, tokens) or (heads, tokens, tokens)
        attention_matrices = []
        for att in self._attentions:
            if len(att.shape) == 4 and att.shape[2] == att.shape[3]:  # (batch, heads, tokens, tokens)
                print(f"DEBUG: Found valid 4D MUSK attention matrix: {att.shape}")
                attention_matrices.append(att)
            elif len(att.shape) == 3 and att.shape[1] == att.shape[2]:  # (heads, tokens, tokens)
                print(f"DEBUG: Found valid 3D MUSK attention matrix: {att.shape}")
                attention_matrices.append(att)
            else:
                print(f"DEBUG: Skipping MUSK tensor - not square attention: {att.shape}")
        
        print(f"DEBUG: Found {len(attention_matrices)} valid MUSK attention matrices")
        
        if not attention_matrices:
            raise RuntimeError("No valid MUSK attention matrices found!")
        
        # Compute rollout: multiply attention matrices across layers
        result = torch.eye(attention_matrices[0].size(-1))
        
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
        
        print(f"DEBUG: Processing {len(self._attentions)} collected attention tensors")
        for i, att in enumerate(self._attentions):
            print(f"DEBUG: Attention {i}: shape={att.shape}")
        
        # Standard models can have either (batch, heads, tokens, tokens) or (heads, tokens, tokens)
        attention_matrices = []
        for att in self._attentions:
            print(f"DEBUG: Checking tensor shape: {att.shape}")
            if len(att.shape) == 4 and att.shape[2] == att.shape[3]:  # (batch, heads, tokens, tokens)
                print(f"DEBUG: Found valid 4D attention matrix: {att.shape}")
                attention_matrices.append(att)
            elif len(att.shape) == 3 and att.shape[1] == att.shape[2]:  # (heads, tokens, tokens)
                print(f"DEBUG: Found valid 3D attention matrix: {att.shape}")
                attention_matrices.append(att)
            else:
                print(f"DEBUG: Skipping tensor - not square attention: {att.shape}")
        
        print(f"DEBUG: Found {len(attention_matrices)} valid attention matrices")
        
        if not attention_matrices:
            raise RuntimeError("No valid standard attention matrices found!")
        
        # Compute rollout: multiply attention matrices across layers
        result = torch.eye(attention_matrices[0].size(-1))
        
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
        
        print(f"DEBUG: Processing MUSK attention - shape: {attention.shape}")
        
        # MUSK MultiheadAttention returns (batch, heads, tokens, tokens) or (heads, tokens, tokens)
        if len(attention.shape) == 4 and attention.shape[2] == attention.shape[3]:
            # (batch, heads, tokens, tokens) -> (tokens, tokens)
            attn_avg = attention.mean(dim=1).squeeze(0)
        elif len(attention.shape) == 3 and attention.shape[1] == attention.shape[2]:
            # (heads, tokens, tokens) -> (tokens, tokens)
            attn_avg = attention.mean(dim=0)
        else:
            raise ValueError(f"Unexpected MUSK attention shape: {attention.shape}")
        
        print(f"DEBUG: After averaging heads - shape: {attn_avg.shape}")
        print(f"DEBUG: Attention matrix range: [{attn_avg.min():.4f}, {attn_avg.max():.4f}]")
        
        # Extract attention from class token to patches
        mask = attn_avg[0, skip_tokens:]  # Skip class token
        print(f"DEBUG: Class-to-patches attention shape: {mask.shape}")
        print(f"DEBUG: Class-to-patches range: [{mask.min():.4f}, {mask.max():.4f}]")
        
        return self._reshape_to_spatial(mask)
    
    def _process_single_attention_standard(self, attention: torch.Tensor) -> np.ndarray:
        """Process a single standard attention matrix."""
        skip_tokens = self.model_config['skip_tokens']
        
        print(f"DEBUG: Processing standard attention - shape: {attention.shape}")
        print(f"DEBUG: Skip tokens: {skip_tokens}")
        
        if len(attention.shape) == 4 and attention.shape[2] == attention.shape[3]:  # (batch, heads, tokens, tokens)
            # Average across heads and remove batch: (batch, heads, tokens, tokens) -> (tokens, tokens)
            attn_avg = attention.mean(dim=1).squeeze(0)
        elif len(attention.shape) == 3 and attention.shape[1] == attention.shape[2]:  # (heads, tokens, tokens)
            # Average across heads: (heads, tokens, tokens) -> (tokens, tokens)
            attn_avg = attention.mean(dim=0)
        else:
            raise ValueError(f"Unexpected standard attention shape: {attention.shape}")
        
        print(f"DEBUG: After averaging heads - shape: {attn_avg.shape}")
        print(f"DEBUG: Attention matrix range: [{attn_avg.min():.4f}, {attn_avg.max():.4f}]")
        
        # Extract attention from class token to patches
        mask = attn_avg[0, skip_tokens:]  # Skip class token
        print(f"DEBUG: Class-to-patches attention shape: {mask.shape}")
        print(f"DEBUG: Class-to-patches range: [{mask.min():.4f}, {mask.max():.4f}]")
        print(f"DEBUG: Sample values: {mask[:10]}")
        
        return self._reshape_to_spatial(mask)
    
    def _reshape_to_spatial(self, mask: torch.Tensor) -> np.ndarray:
        """Reshape attention mask to spatial dimensions."""
        num_patches = mask.size(-1)
        width = int(round(math.sqrt(num_patches)))
        
        print(f"DEBUG: Reshaping {num_patches} patches to spatial grid")
        print(f"DEBUG: Calculated width: {width} (width² = {width*width})")
        
        if width * width != num_patches:
            print(f"DEBUG: Non-perfect square: {num_patches} patches")
            if width * width < num_patches:
                # Need to increase width to accommodate all patches
                width = int(math.ceil(math.sqrt(num_patches)))
                print(f"DEBUG: Adjusted width to {width} (width² = {width*width})")
            
            # Pad with zeros to make it square
            if width * width > num_patches:
                padding_size = width * width - num_patches
                print(f"DEBUG: Padding with {padding_size} zeros")
                mask = torch.cat([mask, torch.zeros(padding_size)], dim=0)
        
        mask = mask.view(width, width).numpy()
        print(f"DEBUG: Final spatial shape: {mask.shape}")
        
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


def test_single_model(
    model_name: str,
    h5_file: Path,
    slide_file: Path,
    output_dir: Path,
    tile_size: int = 224,
    device: str = "cpu"
):
    """Test a single model on a single tile."""
    
    print(f"=== Testing {model_name} ===")
    
    # Get model-specific configuration
    try:
        model_config = get_model_config(model_name)
        print(f"DEBUG: Model config: {model_config}")
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Load model
    print(f"Loading model {model_name}...")
    try:
        encoder = encoder_factory(model_name)
        model = encoder.model
        base_model = getattr(model, "trunk", model)
        print(f"DEBUG: Model loaded successfully. Type: {type(base_model).__name__}")
    except Exception as e:
        print(f"ERROR: Failed to load model {model_name}: {e}")
        return
    
    # Load attention data
    print(f"Loading attention data from {h5_file}...")
    try:
        scores, coords = load_attention_h5(h5_file)
        print(f"DEBUG: Loaded {len(scores)} tiles with scores")
    except Exception as e:
        print(f"ERROR: Failed to load attention data: {e}")
        return
    
    # Get the top tile
    top_idx = np.argmax(scores)
    x, y = coords[top_idx]
    score = scores[top_idx]
    print(f"DEBUG: Using top tile at ({x}, {y}) with score {score:.4f}")
    
    # Load slide and extract tile
    print(f"Loading slide {slide_file}...")
    try:
        slide = openslide.OpenSlide(str(slide_file))
        tile_img = slide.read_region((int(x), int(y)), 0, (tile_size, tile_size)).convert("RGB")
        slide.close()
        print(f"DEBUG: Extracted tile with size {tile_img.size}")
    except Exception as e:
        print(f"ERROR: Failed to load slide: {e}")
        return
    
    # Create attention rollout
    print(f"Creating attention rollout...")
    rollout = AttentionRollout(base_model, model_config)
    
    # Process tile
    print(f"Processing tile...")
    try:
        # Preprocess
        print(f"DEBUG: Original tile size: {tile_img.size}")
        processed = encoder.eval_transforms(tile_img)
        processed = processed.unsqueeze(0).to(device=device)
        print(f"DEBUG: Processed tensor shape: {processed.shape}")
        
        # Calculate expected patch dimensions
        B, C, H, W = processed.shape
        print(f"DEBUG: Input dimensions - Batch: {B}, Channels: {C}, Height: {H}, Width: {W}")
        
        # Try to determine patch size from the model
        patch_size = 'unknown'
        stride = 'unknown'
        
        if hasattr(encoder.model, 'patch_embed'):
            patch_embed = encoder.model.patch_embed
            if hasattr(patch_embed, 'proj'):
                patch_size = patch_embed.proj.kernel_size[0] if hasattr(patch_embed.proj, 'kernel_size') else 'unknown'
                stride = patch_embed.proj.stride[0] if hasattr(patch_embed.proj, 'stride') else 'unknown'
                print(f"DEBUG: Patch embed - kernel_size: {patch_size}, stride: {stride}")
        
        # Calculate expected number of patches
        if isinstance(patch_size, int) and isinstance(stride, int):
            expected_patches_h = H // stride
            expected_patches_w = W // stride
            expected_total_patches = expected_patches_h * expected_patches_w
            print(f"DEBUG: Expected patches - H: {expected_patches_h}, W: {expected_patches_w}, Total: {expected_total_patches}")
        else:
            print(f"DEBUG: Could not determine patch size from model")
        
        # Clear previous attentions and run model
        rollout._attentions.clear()
        with torch.no_grad():
            _ = encoder(processed)
        
        print(f"DEBUG: Collected {len(rollout._attentions)} attention matrices")
        
        # Debug attention matrix details
        if rollout._attentions:
            first_attn = rollout._attentions[0]
            print(f"DEBUG: First attention matrix shape: {first_attn.shape}")
            print(f"DEBUG: Attention matrix details - Batch: {first_attn.shape[0]}, Heads: {first_attn.shape[1]}, Tokens: {first_attn.shape[2]}")
            
            # Calculate expected spatial dimensions
            total_tokens = first_attn.shape[2]
            skip_tokens = model_config['skip_tokens']
            patch_tokens = total_tokens - skip_tokens
            spatial_size = int(math.sqrt(patch_tokens))
            print(f"DEBUG: Token calculation - Total: {total_tokens}, Skip: {skip_tokens}, Patches: {patch_tokens}")
            print(f"DEBUG: Expected spatial size: {spatial_size} (√{patch_tokens} = {math.sqrt(patch_tokens):.2f})")
        
        # Compute combined attention map (all layers)
        print("Computing combined attention rollout...")
        combined_mask = rollout.compute_rollout()
        print(f"DEBUG: Generated combined attention mask with shape: {combined_mask.shape}")
        
        # Compute first layer attention only
        print("Computing first layer attention...")
        first_layer_mask = rollout.compute_first_layer_attention()
        print(f"DEBUG: Generated first layer attention mask with shape: {first_layer_mask.shape}")
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original tile
        original_path = output_dir / f"{model_name}_original.png"
        tile_img.save(original_path)
        print(f"DEBUG: Saved original tile to {original_path}")
        
        # Save combined attention map
        combined_heatmap = (combined_mask * 255).astype(np.uint8)
        combined_heatmap_img = Image.fromarray(combined_heatmap, mode="L")
        combined_attention_path = output_dir / f"{model_name}_attention_combined.png"
        combined_heatmap_img.save(combined_attention_path)
        print(f"DEBUG: Saved combined attention map to {combined_attention_path}")
        
        # Save first layer attention map
        first_layer_heatmap = (first_layer_mask * 255).astype(np.uint8)
        first_layer_heatmap_img = Image.fromarray(first_layer_heatmap, mode="L")
        first_layer_attention_path = output_dir / f"{model_name}_attention_first_layer.png"
        first_layer_heatmap_img.save(first_layer_attention_path)
        print(f"DEBUG: Saved first layer attention map to {first_layer_attention_path}")
        
        print(f"SUCCESS: {model_name} test completed!")
        
    except Exception as e:
        print(f"ERROR: Failed to process tile: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test single model attention extraction")
    parser.add_argument("--model", "-m", type=str, required=True, 
                       help="Model name (musk, conch_v15, virchow2, hoptimus1, gigapath)")
    parser.add_argument("--h5", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--slide", type=str, required=True, help="Path to slide file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--tile-size", type=int, default=224, help="Tile size")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    
    args = parser.parse_args()
    
    # Map model names
    name_map = {
        "musk": "musk",
        "conch_v15": "conch_v15", 
        "virchow2": "virchow2",
        "hoptimus1": "hoptimus1",
        "gigapath": "gigapath",
    }
    
    if args.model not in name_map:
        print(f"ERROR: Unknown model {args.model}. Available: {list(name_map.keys())}")
        return
    
    test_single_model(
        model_name=name_map[args.model],
        h5_file=Path(args.h5),
        slide_file=Path(args.slide),
        output_dir=Path(args.output),
        tile_size=args.tile_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
