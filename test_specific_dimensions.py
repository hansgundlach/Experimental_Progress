#!/usr/bin/env python3
"""
Test specific dimensions 96 and 160 to find the issue
"""

import torch
from experiment_utils import get_base_config, gen_experim
from core import SimpleTransformer

def test_specific_dimensions():
    """Test dimensions 96 and 160 specifically"""
    
    for hidden_dim in [96, 160]:
        print(f"\n{'='*60}")
        print(f"Testing hidden_dim = {hidden_dim}")
        print(f"{'='*60}")
        
        try:
            # Create experiment with specific dimension
            experiment = gen_experim(
                hidden_dim=hidden_dim,
                label=f"test_{hidden_dim}d",
                ff_ratio=4,
                modern_bias_0=False
            )
            
            print(f"✓ Experiment created successfully")
            
            # Get the config from the experiment
            config_dict = experiment[0]["subexperiments"][0]["overrides"]
            print(f"Config: {config_dict}")
            
            # Convert to object
            class Config:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            
            config_obj = Config(config_dict)
            
            # Create model
            model = SimpleTransformer(
                vocab_size=50257,
                hidden_dim=hidden_dim,
                num_heads=config_dict["num_heads"],
                num_layers=config_dict["num_layers"],
                dropout=0.0,
                config=config_obj,
                tie_embeddings=True
            )
            
            print(f"✓ Model created successfully")
            print(f"  - num_heads: {config_dict['num_heads']}")
            print(f"  - num_layers: {config_dict['num_layers']}")
            print(f"  - head_dim: {hidden_dim // config_dict['num_heads']}")
            
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
            
        except Exception as e:
            print(f"❌ Error with hidden_dim={hidden_dim}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_specific_dimensions()
