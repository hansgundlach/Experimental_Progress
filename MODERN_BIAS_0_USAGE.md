# Modern Bias-Free Architecture (`modern_bias_0`)

## Overview

The `modern_bias_0` parameter enables a modern transformer architecture that removes bias terms from layers followed by normalization. This approach is used in state-of-the-art models like LLaMA and reduces parameter count while maintaining or improving performance.

## What It Does

When `modern_bias_0=True`:

1. **Removes biases from attention projections**: QKV and output projections have no bias
2. **Removes biases from feedforward layers**: All MLP projections have no bias
3. **Removes bias from output layer**: Especially important when embeddings are tied
4. **Supports bias-free LayerNorm**: LayerNorm only keeps scale parameter (or use RMSNorm)

## Usage

### In Base Config

```python
from experiment_utils import get_base_config

config = get_base_config()
config["modern_bias_0"] = True  # Enable modern bias-free architecture
```

### In Experiments

```python
from experiment_utils import gen_experim

# Create experiment with modern bias-free architecture
experiment = gen_experim(
    hidden_dim=128,
    label="modern_128d",
    modern_bias_0=True,
    norm_type="rms",  # RMSNorm is recommended with modern_bias_0
)
```

## Parameter Savings

For a typical small transformer (64d, 2 layers):
- **With bias**: 3,349,824 parameters (2,304 bias parameters)
- **Without bias**: 3,347,904 parameters (384 bias parameters)
- **Reduction**: 1,920 parameters (0.06%)

The percentage reduction increases with model size.

## Compatibility

- ✅ Works with all activation functions (GELU, ReLU, SiLU, SwiGLU, GLU)
- ✅ Works with LayerNorm (bias-free mode) and RMSNorm
- ✅ Works with all positional encodings (rotary, sinusoidal, learned)
- ✅ Compatible with weight tying (`tie_embeddings=True`)
- ✅ Compatible with muP scaling

## Recommended Settings

For modern architectures, use:

```python
config = {
    "modern_bias_0": True,
    "norm_type": "rms",           # RMSNorm (already bias-free)
    "activation": "swiglu",        # Modern activation
    "norm_placement": "pre",       # Pre-normalization
    "pos_encoding": "rotary",      # Rotary positional encoding
    "tie_embeddings": True,        # Weight tying
}
```

## Backward Compatibility

The default value is `False`, which maintains the original behavior with all bias terms present. This ensures existing experiments continue to work exactly as before.

## Implementation Details

The implementation:
1. Adds `bias=False` to all Linear layers in attention and feedforward when `modern_bias_0=True`
2. Uses bias-free LayerNorm when `modern_bias_0=True` and `norm_type="layer"`
3. RMSNorm is already bias-free by design
4. All initialization code checks for bias existence before initializing
5. Output layer respects both `modern_bias_0` and `tie_embeddings` settings

## Testing

Run the test to verify functionality:

```python
python test_modern_bias_0.py
```

This will verify:
- Biases are correctly removed when `modern_bias_0=True`
- Biases are present when `modern_bias_0=False`
- Forward pass works correctly in both modes
- Compatible with different norm types
- Parameter counts are as expected
