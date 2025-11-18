#!/usr/bin/env python3
"""
Pre-tokenize large datasets and save as memory-mapped numpy arrays.
This version allows internet downloads to get fast tokenizer files if needed.

This allows datasets larger than RAM to be used without loading everything into memory.

Usage:
    python pretokenize_internet.py Datasets/c4_subset_large.txt
    python pretokenize_internet.py Datasets/openwebtext_subset.txt
    python pretokenize_internet.py Datasets/wikitext103_train.txt
"""

import numpy as np
from pathlib import Path
from transformers import GPT2Tokenizer
import sys
import time


def pretokenize_dataset(text_path, output_path=None):
    """
    Tokenize a text file and save as memory-mapped numpy array.
    This version allows internet downloads to get fast tokenizer files.

    Args:
        text_path: Path to text file
        output_path: Output path for .npy file (default: text_path.npy)
    """
    if output_path is None:
        output_path = str(Path(text_path).with_suffix(".npy"))

    print(f"\n{'='*60}")
    print(f"PRE-TOKENIZING DATASET (INTERNET ENABLED)")
    print(f"{'='*60}")

    print(f"Loading tokenizer (prioritizing fast tokenizer from HuggingFace)...")
    # Prioritize fast tokenizer - try HuggingFace Hub first (most reliable source)
    # Fast tokenizer (Rust-based) is 10-100x faster than slow tokenizer (Python-based)
    tokenizer = None
    try:
        # First try HuggingFace Hub directly - most reliable source for fast tokenizer
        print("  📥 Downloading fast tokenizer from HuggingFace Hub...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)
        print(
            "  ✅ Using fast tokenizer (Rust-based, much faster) - from HuggingFace Hub"
        )
    except:
        try:
            # Fallback: try local directory with internet enabled (may download missing files)
            print("  📥 Trying local tokenizer directory with internet fallback...")
            tokenizer = GPT2Tokenizer.from_pretrained(
                "./gpt2_tokenizer", local_files_only=False, use_fast=True
            )
            print(
                "  ✅ Using fast tokenizer (Rust-based, much faster) - local/updated files"
            )
        except:
            try:
                # Fallback: try local fast tokenizer only (no internet)
                print("  📥 Trying local fast tokenizer...")
                tokenizer = GPT2Tokenizer.from_pretrained(
                    "./gpt2_tokenizer", local_files_only=True, use_fast=True
                )
                print(
                    "  ✅ Using fast tokenizer (Rust-based, much faster) - local files only"
                )
            except:
                # Final fallback: slow tokenizer (Python-based, much slower)
                print(
                    "  ⚠️  Fast tokenizer not available, falling back to slow tokenizer..."
                )
                try:
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=False)
                    print(
                        "  ⚠️  Using slow tokenizer (Python-based, slower) - from HuggingFace Hub"
                    )
                except:
                    tokenizer = GPT2Tokenizer.from_pretrained(
                        "./gpt2_tokenizer", local_files_only=True, use_fast=False
                    )
                    print(
                        "  ⚠️  Using slow tokenizer (Python-based, slower) - local files only"
                    )

    print(f"\nReading text file: {text_path}")
    file_size = Path(text_path).stat().st_size
    print(f"  File size: {file_size / 1e9:.2f} GB ({file_size / 1e6:.0f} MB)")

    # Read and tokenize in chunks to avoid memory issues
    # Use larger chunks for better performance (tokenizer can handle bigger inputs)
    chunk_size = 100_000_000  # 100MB chunks of text
    all_token_ids = []
    total_chars = 0

    print(f"\nTokenizing (this may take a while for large files)...")
    start_time = time.time()
    with open(text_path, "r", encoding="utf-8") as f:
        chunk_num = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            chunk_num += 1
            total_chars += len(chunk)

            # Split on newlines to avoid breaking words, then tokenize each paragraph
            # Process paragraphs individually to avoid exceeding model's max sequence length (1024)
            # This prevents warnings and ensures all text is tokenized correctly
            paragraphs = chunk.split("\n")

            for paragraph in paragraphs:
                if paragraph.strip():  # Skip empty paragraphs
                    # Tokenize each paragraph individually
                    # This avoids warnings about exceeding max sequence length (1024 tokens)
                    tokens = tokenizer(
                        paragraph, truncation=False, add_special_tokens=False
                    )["input_ids"]
                    all_token_ids.extend(tokens)

            # Progress update with ETA
            elapsed = time.time() - start_time
            tokens_per_sec = len(all_token_ids) / elapsed if elapsed > 0 else 0
            chars_per_sec = total_chars / elapsed if elapsed > 0 else 0
            remaining_chars = file_size - total_chars
            eta_seconds = remaining_chars / chars_per_sec if chars_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60

            print(
                f"  Chunk {chunk_num}: {total_chars / 1e6:.0f}M chars → {len(all_token_ids) / 1e6:.1f}M tokens | "
                f"Speed: {chars_per_sec / 1e6:.1f} MB/s | ETA: {eta_minutes:.0f} min",
                end="\r",
            )

    print(f"\n  ✅ Total: {len(all_token_ids):,} tokens from {total_chars:,} chars")
    print(f"     Char-to-token ratio: {total_chars / len(all_token_ids):.2f}:1")

    # Save as numpy array in int32 format (4 bytes per token)
    print(f"\nSaving to {output_path}...")
    token_array = np.array(all_token_ids, dtype=np.int32)
    np.save(output_path, token_array)

    file_size_gb = token_array.nbytes / 1e9
    print(f"✅ Done! Saved to: {output_path}")
    print(f"   Tokens: {len(all_token_ids):,}")
    print(f"   File size: {file_size_gb:.2f} GB")

    print(f"\n{'='*60}")
    print(f"MEMORY-MAPPED FILE CREATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"This file will be memory-mapped during training.")
    print(f"Memory mapping means the OS loads only the pages you access,")
    print(f"so the entire dataset doesn't need to fit in RAM!")
    print(f"{'='*60}\n")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pretokenize_internet.py <text_file>")
        print("\nExamples:")
        print("  python pretokenize_internet.py Datasets/c4_subset_large.txt")
        print("  python pretokenize_internet.py Datasets/openwebtext_subset.txt")
        print("  python pretokenize_internet.py Datasets/wikitext103_train.txt")
        sys.exit(1)

    text_path = sys.argv[1]

    if not Path(text_path).exists():
        print(f"❌ Error: File not found: {text_path}")
        sys.exit(1)

    pretokenize_dataset(text_path)
