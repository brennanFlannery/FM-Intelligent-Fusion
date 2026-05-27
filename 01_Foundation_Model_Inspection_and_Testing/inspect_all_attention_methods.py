#!/usr/bin/env python3
"""
Script to inspect attention forward methods for all ViT models.
Duplicates the functionality from test_model_archs.py for each model.
"""

import torch
import inspect
from trident.patch_encoder_models import encoder_factory

def inspect_model_attention(model_name):
    """Inspect attention methods for a single model."""
    print(f"\n{'='*80}")
    print(f"INSPECTING MODEL: {model_name.upper()}")
    print(f"{'='*80}")
    
    try:
        # Use simple TRIDENT approach - no extra parameters
        encoder = encoder_factory(model_name)
        
        print(f"✅ Successfully loaded {model_name}")
        print(f"   Model type: {type(encoder.model)}")
        print(f"   Model class: {encoder.model.__class__.__name__}")
        
        # Find the Attention class in the model
        attention_class = None
        attention_module = None
        
        # Search through all modules to find an Attention class
        for name, module in encoder.model.named_modules():
            if 'attn' in name and not 'attn_drop' in name and not 'qkv' in name:
                attention_class = type(module)
                attention_module = module
                print(f"   Found attention module: {name} -> {attention_class.__name__}")
                break
        
        if attention_class is None:
            print(f"❌ Could not find Attention class in {model_name}")
            print(f"   Available modules: {[name for name, _ in encoder.model.named_modules() if 'attn' in name]}")
            return
        
        print(f"\n{'='*80}")
        print("SOURCE CODE FOR THE 'Attention' CLASS")
        print(f"{'='*80}")
        try:
            print(inspect.getsource(attention_class))
        except TypeError as e:
            print(f"Could not get source for Attention class: {e}")
        
        print(f"\n{'='*80}")
        print("SOURCE CODE FOR THE 'Attention.forward' METHOD")
        print(f"{'='*80}")
        try:
            print(inspect.getsource(attention_module.forward))
        except TypeError as e:
            print(f"Could not get source for forward method: {e}")
            
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")

def main():
    """Main function to inspect all models."""
    # Models to inspect (excluding MUSK since it uses different architecture)
    models = ["conch_v15", "virchow2", "hoptimus1", "gigapath"]
    
    print("🔍 INSPECTING ATTENTION FORWARD METHODS FOR ALL ViT MODELS")
    print("=" * 80)
    
    for model_name in models:
        inspect_model_attention(model_name)
    
    print(f"\n{'='*80}")
    print("INSPECTION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
