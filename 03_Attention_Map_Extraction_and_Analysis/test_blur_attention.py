#!/usr/bin/env python3
"""
Test script for blurring attention maps.

This script applies different blurring methods to the first tile from each model
and saves the results to a "blur_test" folder for inspection.

Usage:
    python test_blur_attention.py --parent-folder /path/to/parent/folder
"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Import the blur function from the main script
import sys
sys.path.append('/mnt/vstor/Data7/bxf169/KidneyCancerPathology')
from compute_attention_overlap import load_attention_map, blur_attention_map


def create_blur_comparison_grid(original: np.ndarray, blurred_maps: dict, 
                              tile_name: str, output_path: Path) -> None:
    """
    Create a comparison grid showing original and different blurred versions.
    
    Args:
        original: Original attention map
        blurred_maps: Dictionary mapping blur method names to blurred maps
        tile_name: Name of the tile for the title
        output_path: Path to save the comparison image
    """
    n_methods = len(blurred_maps) + 1  # +1 for original
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot original
    axes[0, 0].imshow(original, cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Original\n{tile_name}')
    axes[0, 0].axis('off')
    
    # Plot blurred versions
    row, col = 0, 1
    for method_name, blurred_map in blurred_maps.items():
        if col >= n_cols:
            row += 1
            col = 0
        
        axes[row, col].imshow(blurred_map, cmap='hot', vmin=0, vmax=1)
        axes[row, col].set_title(f'{method_name.title()} Blur')
        axes[row, col].axis('off')
        col += 1
    
    # Hide unused subplots
    for i in range(row * n_cols + col, n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        axes[r, c].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def test_blur_on_first_tiles(parent_folder: Path, models: List[str]) -> None:
    """
    Test blurring on the first tile from each model.
    
    Args:
        parent_folder: Parent directory containing model subdirectories
        models: List of model names
    """
    # Create output directory
    blur_test_dir = parent_folder / "blur_test"
    blur_test_dir.mkdir(exist_ok=True)
    
    print(f"Testing blur methods on first tiles from each model...")
    print(f"Results will be saved to: {blur_test_dir}")
    
    # Define blur methods to test
    blur_methods = {
        "gaussian_light": ("gaussian", 0.5),
        "gaussian_medium": ("gaussian", 1.0),
        "gaussian_heavy": ("gaussian", 2.0),
        "uniform_light": ("uniform", 1.0),
        "uniform_medium": ("uniform", 2.0),
        "median_light": ("median", 1.0),
    }
    
    for model in models:
        print(f"\nProcessing model: {model}")
        
        # Find the first tile for this model
        model_dir = parent_folder / model / "combined"
        if not model_dir.exists():
            print(f"  Warning: Model directory not found: {model_dir}")
            continue
        
        # Get first tile
        tile_files = sorted(list(model_dir.glob("*.png")))
        if not tile_files:
            print(f"  Warning: No tiles found in {model_dir}")
            continue
        
        first_tile = tile_files[0]
        tile_name = first_tile.name
        
        print(f"  Using tile: {tile_name}")
        
        try:
            # Load original attention map
            original_map = load_attention_map(first_tile)
            
            # Apply different blur methods
            blurred_maps = {}
            for method_name, (blur_type, blur_strength) in blur_methods.items():
                blurred_map = blur_attention_map(original_map, blur_type, blur_strength)
                blurred_maps[method_name] = blurred_map
                
                # Save individual blurred map
                blurred_img = (blurred_map * 255).astype(np.uint8)
                blurred_pil = Image.fromarray(blurred_img, mode='L')
                individual_path = blur_test_dir / f"{model}_{method_name}.png"
                blurred_pil.save(individual_path)
                print(f"    Saved: {individual_path}")
            
            # Create comparison grid
            comparison_path = blur_test_dir / f"{model}_blur_comparison.png"
            create_blur_comparison_grid(original_map, blurred_maps, tile_name, comparison_path)
            print(f"    Saved comparison: {comparison_path}")
            
        except Exception as e:
            print(f"  Error processing {model}: {e}")
            continue
    
    print(f"\nBlur testing complete! Check results in: {blur_test_dir}")


def main():
    """Main function to test blur methods."""
    parser = argparse.ArgumentParser(
        description="Test blurring methods on attention maps from foundation models"
    )
    parser.add_argument(
        "--parent-folder", "-p",
        type=str,
        required=True,
        help="Path to parent folder containing model subdirectories"
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    parent_folder = Path(args.parent_folder)
    
    if not parent_folder.exists():
        raise FileNotFoundError(f"Parent folder not found: {parent_folder}")
    
    # Define the 5 foundation models
    models = ["musk", "conch_v15", "virchow2", "hoptimus1", "gigapath"]
    
    print(f"Testing blur methods on attention maps in: {parent_folder}")
    print(f"Models: {', '.join(models)}")
    
    # Check that model directories exist
    available_models = []
    for model in models:
        model_dir = parent_folder / model / "combined"
        if model_dir.exists():
            available_models.append(model)
        else:
            print(f"Warning: Model directory not found: {model_dir}")
    
    if not available_models:
        raise ValueError("No model directories found with tiles")
    
    print(f"Using models: {', '.join(available_models)}")
    
    # Run blur tests
    test_blur_on_first_tiles(parent_folder, available_models)


if __name__ == "__main__":
    main()
