#!/usr/bin/env python3
"""
Test dimensions 96 and 160 with proper config
"""

import torch
from experiment_utils import get_base_config
from core import SimpleTransformer

def test_dimensions_proper():
    """Test dimensions 96 and 160 with complete config"""
    
    for hidden_dim in [96, 160]:
        print(f"\n{'='*60}")
        print(f"Testing hidden_dim = {hidden_dim}")
        print(f"{'='*60}")
        
        try:
            # Get base config and modify it
            config = get_base_config()
            config["hidden_dim"] = hidden_dim
            
            # Calculate num_heads like in experiment_utils.py
            target_head_dim = 16
            num_heads = max(1, int(round(hidden_dim / target_head_dim)))
            # Ensure hidden_dim is divisible by num_heads
            while hidden_dim % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            
            # Calculate num_layers
            base_hidden_dim = 32
            base_num_layers = 2
            layer_scale_ratio = base_num_layers / base_hidden_dim
            num_layers = max(1, int(round(hidden_dim * layer_scale_ratio)))
            
            config["num_heads"] = num_heads
            config["num_layers"] = num_layers
            
            print(f"Config: hidden_dim={hidden_dim}, num_heads={num_heads}, num_layers={num_layers}")
            print(f"head_dim = {hidden_dim // num_heads}")
            
            # Convert to object
            class Config:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            
            config_obj = Config(config)
            
            # Create model
            model = SimpleTransformer(
                vocab_size=50257,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=0.0,
                config=config_obj,
                tie_embeddings=True
            )
            
            print(f"✓ Model created successfully")
            
            # Test forward pass
            batch_size = 2
            seq_length = 16
            input_ids = torch.randint(0, 50257, (batch_size, seq_length))
            
            output = model(input_ids)
            print(f"✓ Forward pass successful! Output shape: {output.shape}")
            
            # Test with different sequence lengths
            for seq_len in [32, 64, 128]:
                input_ids_test = torch.randint(0, 50257, (batch_size, seq_len))
                output_test = model(input_ids_test)
                print(f"✓ Seq length {seq_len}: {output_test.shape}")
            
            # Test parameter count
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"❌ Error with hidden_dim={hidden_dim}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_dimensions_proper()
