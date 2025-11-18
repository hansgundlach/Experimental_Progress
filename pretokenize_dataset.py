#!/usr/bin/env python3
"""
Pre-tokenize large datasets and save as memory-mapped numpy arrays.
This version allows internet downloads to get fast tokenizer files if needed.
Saves checkpoints every 200M tokens for safety.

Usage:
    python pretokenize_dataset.py Datasets/c4_subset_large.txt
    python pretokenize_dataset.py Datasets/openwebtext_subset.txt
    python pretokenize_dataset.py Datasets/wikitext103_train.txt
"""

import numpy as np
from pathlib import Path
from transformers import GPT2Tokenizer
import sys
import time


def pretokenize_dataset(text_path, output_path=None, checkpoint_interval=200_000_000):
    """
    Tokenize a text file and save as memory-mapped numpy array.
    Saves checkpoints periodically to avoid losing progress.

    Args:
        text_path: Path to text file
        output_path: Output path for .npy file (default: text_path.npy)
        checkpoint_interval: Save checkpoint every N tokens (default: 200M)
    """
    if output_path is None:
        output_path = str(Path(text_path).with_suffix(".npy"))

    print(f"\n{'='*60}")
    print(f"PRE-TOKENIZING DATASET (WITH CHECKPOINTS)")
    print(f"{'='*60}")

    print(f"Loading tokenizer (prioritizing fast tokenizer from HuggingFace)...")
    # Prioritize fast tokenizer - try HuggingFace Hub first (most reliable source)
    tokenizer = None
    try:
        print("  📥 Downloading fast tokenizer from HuggingFace Hub...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)
        print(
            "  ✅ Using fast tokenizer (Rust-based, much faster) - from HuggingFace Hub"
        )
    except:
        try:
            print("  📥 Trying local tokenizer directory with internet fallback...")
            tokenizer = GPT2Tokenizer.from_pretrained(
                "./gpt2_tokenizer", local_files_only=False, use_fast=True
            )
            print(
                "  ✅ Using fast tokenizer (Rust-based, much faster) - local/updated files"
            )
        except:
            try:
                print("  📥 Trying local fast tokenizer...")
                tokenizer = GPT2Tokenizer.from_pretrained(
                    "./gpt2_tokenizer", local_files_only=True, use_fast=True
                )
                print(
                    "  ✅ Using fast tokenizer (Rust-based, much faster) - local files only"
                )
            except:
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
    print(f"  Checkpoint interval: {checkpoint_interval / 1e6:.0f}M tokens")

    # Read and tokenize in chunks to avoid memory issues
    chunk_size = 100_000_000  # 100MB chunks of text
    all_token_ids = []
    total_chars = 0
    last_checkpoint_tokens = 0

    print(f"\nTokenizing (this may take a while for large files)...")
    start_time = time.time()
    checkpoint_path = output_path.replace(".npy", "_checkpoint.npy")

    with open(text_path, "r", encoding="utf-8") as f:
        chunk_num = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            chunk_num += 1
            total_chars += len(chunk)

            # Split on newlines to avoid breaking words
            paragraphs = chunk.split("\n")

            for paragraph in paragraphs:
                if paragraph.strip():  # Skip empty paragraphs
                    tokens = tokenizer(
                        paragraph, truncation=False, add_special_tokens=False
                    )["input_ids"]
                    all_token_ids.extend(tokens)

            # Checkpoint saving: Save every checkpoint_interval tokens
            current_tokens = len(all_token_ids)
            if current_tokens - last_checkpoint_tokens >= checkpoint_interval:
                print(
                    f"\n  💾 Saving checkpoint ({current_tokens / 1e6:.1f}M tokens)..."
                )
                checkpoint_start = time.time()
                token_array_checkpoint = np.array(all_token_ids, dtype=np.int32)
                np.save(checkpoint_path, token_array_checkpoint)
                checkpoint_time = time.time() - checkpoint_start
                file_size_gb = token_array_checkpoint.nbytes / 1e9
                print(
                    f"     ✅ Checkpoint saved: {file_size_gb:.2f} GB ({checkpoint_time:.1f}s)"
                )
                last_checkpoint_tokens = current_tokens

            # Progress update with ETA
            elapsed = time.time() - start_time
            tokens_per_sec = current_tokens / elapsed if elapsed > 0 else 0
            chars_per_sec = total_chars / elapsed if elapsed > 0 else 0
            remaining_chars = file_size - total_chars
            eta_seconds = remaining_chars / chars_per_sec if chars_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60

            print(
                f"  Chunk {chunk_num}: {total_chars / 1e6:.0f}M chars → {current_tokens / 1e6:.1f}M tokens | "
                f"Speed: {chars_per_sec / 1e6:.1f} MB/s | ETA: {eta_minutes:.0f} min",
                end="\r",
            )

    print(f"\n  ✅ Total: {len(all_token_ids):,} tokens from {total_chars:,} chars")
    print(f"     Char-to-token ratio: {total_chars / len(all_token_ids):.2f}:1")

    # Final save
    print(f"\nSaving final file to {output_path}...")
    save_start = time.time()
    token_array = np.array(all_token_ids, dtype=np.int32)
    np.save(output_path, token_array)
    save_time = time.time() - save_start

    file_size_gb = token_array.nbytes / 1e9
    print(f"✅ Done! Saved to: {output_path}")
    print(f"   Tokens: {len(all_token_ids):,}")
    print(f"   File size: {file_size_gb:.2f} GB (saved in {save_time:.1f}s)")

    # Remove checkpoint file if final save succeeded
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"   🗑️  Removed checkpoint file")

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
        print("Usage: python pretokenize_dataset.py <text_file> [checkpoint_interval]")
        print("\nExamples:")
        print("  python pretokenize_dataset.py Datasets/c4_subset_large.txt")
        print(
            "  python pretokenize_dataset.py Datasets/c4_subset_large.txt 100000000  # 100M tokens"
        )
        print("\nNote: Checkpoints are saved every N tokens (default: 200M)")
        sys.exit(1)

    text_path = sys.argv[1]

    # Optional checkpoint interval argument
    checkpoint_interval = 200_000_000  # Default: 200M tokens
    if len(sys.argv) >= 3:
        try:
            checkpoint_interval = int(sys.argv[2])
        except ValueError:
            print(f"⚠️  Invalid checkpoint interval '{sys.argv[2]}', using default 200M")
            checkpoint_interval = 200_000_000

    if not Path(text_path).exists():
        print(f"❌ Error: File not found: {text_path}")
        sys.exit(1)

    pretokenize_dataset(text_path, checkpoint_interval=checkpoint_interval)
