#!/usr/bin/env python3
"""
Simple script to print forward method source code for attention blocks.
"""

import inspect
from trident.patch_encoder_models import encoder_factory

def print_simple_forward_methods():
    """Print forward method source code for attention blocks."""
    
    models = ["conch_v15", "virchow2", "hoptimus1", "gigapath"]
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load model using the correct encoder_factory call pattern
            # Each model has specific parameter requirements for their _build method
            encoder = None
            
            if model_name == 'conch_v15':
                # Conchv15InferenceEncoder._build(self, img_size=448)
                encoder = encoder_factory(model_name, img_size=448)
            elif model_name == 'virchow2':
                # Virchow2InferenceEncoder._build(self, return_cls=False, timm_kwargs=...)
                encoder = encoder_factory(model_name, return_cls=False)
            elif model_name == 'gigapath':
                # GigaPathInferenceEncoder._build(self) - no parameters
                encoder = encoder_factory(model_name)
            elif model_name == 'hoptimus1':
                # HOptimus1InferenceEncoder._build(self, timm_kwargs=..., **kwargs)
                encoder = encoder_factory(model_name, timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False})
            else:
                print(f"❌ Unknown model: {model_name}")
                continue
            
            base_model = encoder.model
            
            # Find Attention class
            attention_class = None
            for name, module in base_model.named_modules():
                if 'attn' in name and not 'attn_drop' in name and not 'qkv' in name:
                    attention_class = type(module)
                    break
            
            if attention_class:
                # Get source code
                source_lines = inspect.getsourcelines(attention_class.forward)
                print("".join(source_lines[0]))
            else:
                print("Could not find Attention class")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print_simple_forward_methods()
