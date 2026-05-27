#!/usr/bin/env python
"""
Overlay attention maps onto original images.
Usage: python overlay_tile_attention.py --input_dir <path> --output_dir <path>
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def overlay_attention_on_image(original_img, attention_map, blur_sigma=20.0, alpha=0.4):
    """
    Overlay attention map onto original image using the method from create_slide_attention_maps.py
    
    Args:
        original_img: RGB image as numpy array
        attention_map: Grayscale attention map as numpy array
        blur_sigma: Gaussian blur sigma (default: 20.0)
        alpha: Transparency level for heatmap overlay (default: 0.4)
    
    Returns:
        Overlaid image as numpy array
    """
    # Ensure attention map is grayscale
    if len(attention_map.shape) == 3:
        attention_map = cv2.cvtColor(attention_map, cv2.COLOR_BGR2GRAY)
    
    # Normalize attention map to [0, 1]
    heat = attention_map.astype(np.float32) / 255.0
    
    # Apply Gaussian smoothing
    heat = gaussian_filter(heat, sigma=blur_sigma)
    heat = np.clip(heat, 0, 1)
    
    # Threshold: zero out values below the 50th percentile
    nonzero = heat[heat > 0]
    if nonzero.size > 0:
        cutoff = np.percentile(nonzero, 50)
        heat[heat < cutoff] = 0
    
    # Composite non-zero heat regions with translucency
    heat_uint = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint, cv2.COLORMAP_JET)
    overlay = original_img.copy()
    mask = heat_uint > 0
    blended = cv2.addWeighted(original_img, 1 - alpha, heat_color, alpha, 0)
    overlay[mask] = blended[mask]
    
    return overlay


def find_matching_attention_file(original_filename, attention_dir):
    """
    Find the matching attention map file for a given original image.
    Handles the _L-1 suffix that may be present in attention map filenames.
    
    Args:
        original_filename: Name of the original image file (e.g., 'image.png')
        attention_dir: Path to directory containing attention maps
    
    Returns:
        Path to matching attention map file, or None if not found
    """
    original_stem = Path(original_filename).stem
    
    # Try exact match first
    exact_match = attention_dir / original_filename
    if exact_match.exists():
        return exact_match
    
    # Try with _L-1 suffix (common pattern in attention maps)
    with_suffix = attention_dir / f"{original_stem}_L-1.png"
    if with_suffix.exists():
        return with_suffix
    
    # Try other common patterns
    for pattern in [f"{original_stem}_*.png"]:
        matches = list(attention_dir.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def process_overlays(input_dir, output_dir, blur_sigma=20.0, alpha=0.4, dpi=400, 
                     num_tiles=50, random_seed=42):
    """
    Process all models and create overlays of attention maps on original images.
    
    Args:
        input_dir: Path to input directory containing Original/ and model folders
        output_dir: Path to output directory for saving overlaid images
        blur_sigma: Gaussian blur sigma (default: 20.0)
        alpha: Transparency level for heatmap overlay (default: 0.4)
        dpi: DPI for output images (default: 400)
        num_tiles: Number of tiles to process (default: 50, -1 for all)
        random_seed: Random seed for reproducibility (default: 42)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find Original folder
    original_dir = input_path / "Original"
    if not original_dir.exists():
        raise ValueError(f"Original directory not found at {original_dir}")
    
    # Find all model directories (directories with a 'simple' subdirectory)
    model_dirs = []
    for item in input_path.iterdir():
        if item.is_dir() and item.name != "Original":
            simple_dir = item / "simple"
            if simple_dir.exists():
                model_dirs.append(item.name)
    
    if not model_dirs:
        raise ValueError(f"No model directories with 'simple' subdirectory found in {input_path}")
    
    print(f"Found {len(model_dirs)} model(s): {', '.join(model_dirs)}")
    
    # Get list of original images
    original_images = sorted(original_dir.glob("*.png"))
    if not original_images:
        raise ValueError(f"No PNG files found in {original_dir}")
    
    print(f"Found {len(original_images)} original images")
    
    # Randomly sample images if num_tiles is specified
    if num_tiles > 0 and num_tiles < len(original_images):
        np.random.seed(random_seed)
        selected_indices = np.random.choice(len(original_images), size=num_tiles, replace=False)
        original_images = [original_images[i] for i in sorted(selected_indices)]
        print(f"Randomly selected {num_tiles} images (seed={random_seed})")
    else:
        print(f"Processing all {len(original_images)} images")
    
    # Process each model
    for model_name in model_dirs:
        print(f"\nProcessing model: {model_name}")
        
        attention_dir = input_path / model_name / "simple"
        model_output_dir = output_path / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each original image
        successful = 0
        skipped = 0
        
        for original_path in tqdm(original_images, desc=f"{model_name}"):
            # Find matching attention map
            attention_path = find_matching_attention_file(original_path.name, attention_dir)
            
            if attention_path is None:
                skipped += 1
                continue
            
            # Load images
            original_img = cv2.imread(str(original_path))
            attention_map = cv2.imread(str(attention_path))
            
            if original_img is None:
                print(f"Warning: Could not load original image {original_path}")
                skipped += 1
                continue
            
            if attention_map is None:
                print(f"Warning: Could not load attention map {attention_path}")
                skipped += 1
                continue
            
            # Resize attention map to match original image if needed
            if original_img.shape[:2] != attention_map.shape[:2]:
                attention_map = cv2.resize(attention_map, 
                                          (original_img.shape[1], original_img.shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
            
            # Create overlay
            overlay = overlay_attention_on_image(original_img, attention_map, 
                                                blur_sigma=blur_sigma, alpha=alpha)
            
            # Save with specified DPI
            output_file = model_output_dir / original_path.name
            
            # OpenCV doesn't directly support DPI, so we use a workaround
            # Save the image and then modify metadata if needed
            cv2.imwrite(str(output_file), overlay, 
                       [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
            successful += 1
        
        print(f"Model {model_name}: {successful} overlays created, {skipped} skipped")
    
    print(f"\nDone! Overlaid images saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay attention maps onto original images for multiple models."
    )
    parser.add_argument('--input_dir', required=True,
                       help='Input directory containing Original/ and model folders')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for saving overlaid images')
    parser.add_argument('--blur_sigma', type=float, default=20.0,
                       help='Gaussian blur sigma for smoothing attention maps (default: 20.0)')
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='Transparency level for heatmap overlay (default: 0.4)')
    parser.add_argument('--dpi', type=int, default=400,
                       help='DPI for output images (default: 400)')
    parser.add_argument('--num_tiles', type=int, default=50,
                       help='Number of tiles to randomly select and process (default: 50, -1 for all)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible sampling (default: 42)')
    
    args = parser.parse_args()
    
    process_overlays(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        blur_sigma=args.blur_sigma,
        alpha=args.alpha,
        dpi=args.dpi,
        num_tiles=args.num_tiles,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()

