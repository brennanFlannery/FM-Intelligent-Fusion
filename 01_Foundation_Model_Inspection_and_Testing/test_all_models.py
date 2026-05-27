#!/usr/bin/env python3
"""
Test all models with a single command.
"""

import subprocess
import sys
from pathlib import Path

def test_all_models(h5_file: str, slide_file: str, output_dir: str):
    """Test all available models."""
    
    models = ["musk", "conch_v15", "virchow2", "hoptimus1", "gigapath"]
    
    print("=" * 60)
    print("TESTING ALL MODELS")
    print("=" * 60)
    
    for model in models:
        print(f"\n{'='*20} Testing {model.upper()} {'='*20}")
        
        try:
            cmd = [
                "python3", "test_single_model.py",
                "--model", model,
                "--h5", h5_file,
                "--slide", slide_file,
                "--output", f"{output_dir}/{model}"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {model.upper()} - SUCCESS")
                print("Last few lines of output:")
                print(result.stdout.split('\n')[-10:])
            else:
                print(f"❌ {model.upper()} - FAILED")
                print("Error output:")
                print(result.stderr[-500:])  # Last 500 chars
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {model.upper()} - TIMEOUT (5 minutes)")
        except Exception as e:
            print(f"💥 {model.upper()} - EXCEPTION: {e}")
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE")
    print(f"Check output directory: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 test_all_models.py <h5_file> <slide_file> <output_dir>")
        print("Example: python3 test_all_models.py data.h5 slide.svs ./test_results")
        sys.exit(1)
    
    h5_file = sys.argv[1]
    slide_file = sys.argv[2]
    output_dir = sys.argv[3]
    
    test_all_models(h5_file, slide_file, output_dir)
