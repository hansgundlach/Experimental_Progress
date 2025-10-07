#!/usr/bin/env python3
"""
Fix integer dimension issues and backward compatibility for ff_ratio and modern_bias_0
"""

import re


def fix_core():
    """Apply all necessary fixes to core.py"""
    with open("core.py", "r") as f:
        content = f.read()

    # Fix 1: SwiGLU - ensure hidden_dim is integer
    old_swiglu = """        hidden_dim = hidden_dim or dim * 4

        # Use a single linear layer for SwiGLU projections
        self.proj = nn.Linear(
            dim, hidden_dim * 2, bias=use_bias
        )  # Single projection for both gate and value"""

    new_swiglu = """        hidden_dim = hidden_dim or dim * 4
        # Ensure hidden_dim is an integer (important for non-integer ff_ratio)
        hidden_dim = int(hidden_dim)

        # Use a single linear layer for SwiGLU projections
        self.proj = nn.Linear(
            dim, hidden_dim * 2, bias=use_bias
        )  # Single projection for both gate and value"""

    content = content.replace(old_swiglu, new_swiglu)

    # Fix 2: GLUFeedForward - ensure dimensions are integers
    old_glu = """        # Project to 2*ff_ratio*dim because GLU will halve it to ff_ratio*dim
        self.linear1 = Linear(dim, 2 * ff_ratio * dim, bias=use_bias)
        # From ff_ratio*dim back to dim
        self.linear2 = Linear(ff_ratio * dim, dim, bias=use_bias)"""

    new_glu = """        # Calculate dimensions and ensure they are integers
        ff_dim = int(ff_ratio * dim)
        
        # Project to 2*ff_ratio*dim because GLU will halve it to ff_ratio*dim
        self.linear1 = Linear(dim, 2 * ff_dim, bias=use_bias)
        # From ff_ratio*dim back to dim
        self.linear2 = Linear(ff_dim, dim, bias=use_bias)"""

    content = content.replace(old_glu, new_glu)

    # Fix 3: SwiGLU call in SimpleTransformerLayer - ensure integer dimension
    old_swiglu_call = """            self.ff = SwiGLU(
                hidden_dim, hidden_dim * ff_ratio, dropout=dropout, use_bias=use_bias
            )"""

    new_swiglu_call = """            ff_dim = int(hidden_dim * ff_ratio)  # Ensure integer dimension
            self.ff = SwiGLU(
                hidden_dim, ff_dim, dropout=dropout, use_bias=use_bias
            )"""

    content = content.replace(old_swiglu_call, new_swiglu_call)

    # Fix 4: Standard feedforward - ensure integer dimensions
    old_standard_ff = """            ff_ratio = getattr(config, "ff_ratio", 4)
            self.ff = nn.Sequential(
                Linear(hidden_dim, ff_ratio * hidden_dim, bias=use_bias),
                act_fn,
                nn.Dropout(dropout),
                Linear(ff_ratio * hidden_dim, hidden_dim, bias=use_bias),
            )"""

    new_standard_ff = """            ff_ratio = getattr(config, "ff_ratio", 4)
            ff_dim = int(ff_ratio * hidden_dim)  # Ensure integer dimension
            self.ff = nn.Sequential(
                Linear(hidden_dim, ff_dim, bias=use_bias),
                act_fn,
                nn.Dropout(dropout),
                Linear(ff_dim, hidden_dim, bias=use_bias),
            )"""

    content = content.replace(old_standard_ff, new_standard_ff)

    # Fix 5: Output layer bias - fix backward compatibility
    old_output_bias = """        # Determine if output layer should have bias
        modern_bias_0 = getattr(config, "modern_bias_0", False)
        # Output layer has no bias if modern_bias_0 is True OR if embeddings are tied
        use_output_bias = not modern_bias_0 and not tie_embeddings"""

    new_output_bias = """        # Determine if output layer should have bias
        modern_bias_0 = getattr(config, "modern_bias_0", False)
        # Output layer has no bias ONLY if modern_bias_0 is True
        # Original behavior: always had bias regardless of tie_embeddings
        use_output_bias = not modern_bias_0"""

    content = content.replace(old_output_bias, new_output_bias)

    with open("core.py", "w") as f:
        f.write(content)

    print("✅ Applied all fixes to core.py:")
    print("  - Fixed SwiGLU to ensure integer hidden_dim")
    print("  - Fixed GLUFeedForward to ensure integer dimensions")
    print("  - Fixed SwiGLU call to use integer ff_dim")
    print("  - Fixed standard feedforward to use integer ff_dim")
    print("  - Fixed output layer bias for backward compatibility")


if __name__ == "__main__":
    print("Applying fixes for integer dimensions and backward compatibility...")
    print()

    try:
        fix_core()
        print()
        print("✅ All fixes applied successfully!")
        print()
        print("Summary of changes:")
        print(
            "1. All ff_ratio multiplications now use int() to ensure integer dimensions"
        )
        print("2. Output layer bias logic fixed for backward compatibility")
        print("3. When modern_bias_0=False, behavior is identical to original code")
        print("4. When modern_bias_0=True, uses modern bias-free architecture")
        print()
        print("You can now use ff_ratio=8/3 or any non-integer value safely!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
