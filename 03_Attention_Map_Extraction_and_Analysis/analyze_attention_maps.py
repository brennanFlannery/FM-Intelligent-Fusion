#!/usr/bin/env python3
"""
Script to analyze attention maps from different models.
Loads the first output from each model subfolder and prints attention range and histogram.
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def analyze_attention_map(image_path: Path) -> dict:
    """Analyze a single attention map image."""
    try:
        # Load image as grayscale
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        
        # Convert to float and normalize to [0, 1]
        attention_values = img_array.astype(np.float32) / 255.0
        
        # Normalize to [0, 1] range (min-max normalization)
        min_val = np.min(attention_values)
        max_val = np.max(attention_values)
        if max_val > min_val:  # Avoid division by zero
            normalized_values = (attention_values - min_val) / (max_val - min_val)
        else:
            normalized_values = attention_values  # All values are the same
        
        # Calculate statistics on original values
        stats = {
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(np.mean(attention_values)),
            'std': float(np.std(attention_values)),
            'median': float(np.median(attention_values)),
            'shape': attention_values.shape,
            'total_pixels': attention_values.size,
            'non_zero_pixels': int(np.count_nonzero(attention_values)),
            'zero_pixels': int(np.count_nonzero(attention_values == 0)),
            'normalized_min': float(np.min(normalized_values)),
            'normalized_max': float(np.max(normalized_values))
        }
        
        return stats, normalized_values
        
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, None

def create_text_histogram(values: np.ndarray, bins: int = 10) -> str:
    """Create a text-based histogram."""
    # Create histogram
    hist, bin_edges = np.histogram(values, bins=bins)
    
    # Normalize histogram for display
    max_count = np.max(hist)
    if max_count == 0:
        return "No data to histogram"
    
    # Create text histogram
    lines = []
    lines.append("Histogram:")
    lines.append("Range        Count  Bar")
    lines.append("-" * 30)
    
    for i in range(len(hist)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        count = hist[i]
        
        # Create bar with proportional length
        bar_length = int((count / max_count) * 20)  # Max 20 chars
        bar = "█" * bar_length
        
        lines.append(f"[{bin_start:.3f}-{bin_end:.3f}] {count:6d} {bar}")
    
    return "\n".join(lines)

def analyze_model_outputs(output_dir: Path, model_names: list) -> None:
    """Analyze attention maps for all models."""
    
    print("=" * 80)
    print("ATTENTION MAP ANALYSIS (COMBINED MAPS ONLY)")
    print("=" * 80)
    
    for model_name in model_names:
        print(f"\n🔍 Analyzing {model_name.upper()}")
        print("-" * 60)
        
        # Only check combined directory
        model_dir = output_dir / model_name / "combined"
        
        if not model_dir.exists():
            print(f"❌ Directory not found: {model_dir}")
            continue
        
        # Find first PNG file
        png_files = list(model_dir.glob("*.png"))
        if not png_files:
            print(f"❌ No PNG files found in {model_dir}")
            continue
        
        # Sort to get consistent "first" file
        first_file = sorted(png_files)[0]
        print(f"📁 COMBINED maps from: {model_dir}")
        print(f"📄 Analyzing: {first_file.name}")
        
        # Analyze the attention map
        stats, normalized_values = analyze_attention_map(first_file)
        
        if stats is None:
            continue
        
        # Print statistics
        print(f"   Shape: {stats['shape']}")
        print(f"   Original Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"   Normalized Range: [{stats['normalized_min']:.4f}, {stats['normalized_max']:.4f}]")
        print(f"   Mean: {stats['mean']:.4f}")
        print(f"   Std: {stats['std']:.4f}")
        print(f"   Median: {stats['median']:.4f}")
        print(f"   Non-zero pixels: {stats['non_zero_pixels']}/{stats['total_pixels']} ({stats['non_zero_pixels']/stats['total_pixels']*100:.1f}%)")
        
        # Create and print histogram using normalized values
        if normalized_values is not None:
            histogram_text = create_text_histogram(normalized_values.flatten())
            print(f"\n   Normalized Histogram:")
            print(f"   {histogram_text}")
        
        print()

def main():
    parser = argparse.ArgumentParser(description='Analyze attention maps from different models')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory containing model subfolders')
    parser.add_argument('--models', nargs='+', 
                        default=['musk', 'conch_v15', 'virchow2', 'hoptimus1', 'gigapath'],
                        help='Models to analyze')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    analyze_model_outputs(output_dir, args.models)

if __name__ == "__main__":
    main()
