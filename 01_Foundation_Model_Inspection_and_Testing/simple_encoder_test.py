#!/usr/bin/env python3
"""
Simple script that duplicates the exact encoder_factory call from TRIDENT.
This avoids all the parameter complexity and just loads models the way TRIDENT does.
"""

import torch
from trident.patch_encoder_models import encoder_factory

def test_all_models():
    """Test loading all models using the exact TRIDENT approach."""
    
    models = ["conch_v15", "virchow2", "hoptimus1", "gigapath"]
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Testing {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Use the exact same call as TRIDENT - no extra parameters
            encoder = encoder_factory(model_name)
            
            print(f"✅ Successfully loaded {model_name}")
            print(f"   Model type: {type(encoder.model)}")
            print(f"   Model class: {encoder.model.__class__.__name__}")
            
            # Check if it has blocks (ViT structure)
            if hasattr(encoder.model, 'blocks'):
                print(f"   Has blocks: ✅ ({len(encoder.model.blocks)} blocks)")
                
                # Try to get the first attention module
                try:
                    first_block = encoder.model.blocks[0]
                    if hasattr(first_block, 'attn'):
                        attn_module = first_block.attn
                        print(f"   First attention module: {type(attn_module).__name__}")
                        print(f"   Attention module class: {attn_module.__class__}")
                    else:
                        print(f"   First block structure: {[name for name, _ in first_block.named_children()]}")
                except Exception as e:
                    print(f"   Error accessing blocks: {e}")
            else:
                print(f"   Has blocks: ❌")
                print(f"   Available attributes: {[attr for attr in dir(encoder.model) if not attr.startswith('_')]}")
                
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")

if __name__ == "__main__":
    test_all_models()

