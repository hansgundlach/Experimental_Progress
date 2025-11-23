#!/usr/bin/env python3
"""
Setup script for Penn Treebank (PTB) dataset.

Downloads and prepares the PTB dataset for language modeling.
Outputs to ../Datasets/ directory.
"""

import os
from pathlib import Path
from datasets import load_dataset


def setup_ptb():
    """Download and setup Penn Treebank dataset."""
    # Output to parent Datasets folder
    output_dir = Path("..") / "Datasets"
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("Penn Treebank (PTB) Dataset Setup")
    print("="*60)

    # Download PTB from HuggingFace datasets
    print("\n📥 Downloading PTB dataset from HuggingFace...")
    try:
        dataset = load_dataset("ptb_text_only")
    except Exception as e:
        print(f"❌ Error downloading PTB: {e}")
        print("\nTrying alternative source...")
        try:
            dataset = load_dataset("ptb-text-only/ptb_text_only")
        except Exception as e2:
            print(f"❌ Alternative source also failed: {e2}")
            print("\n⚠️  You may need to manually download PTB")
            print("   Visit: https://huggingface.co/datasets/ptb_text_only")
            return

    print("✅ PTB dataset downloaded successfully!")

    # Save splits
    splits = ["train", "validation", "test"]
    output_files = {}

    for split in splits:
        split_name = split if split != "validation" else "valid"
        output_file = output_dir / f"ptb_{split_name}.txt"

        print(f"\n📝 Processing {split} split...")

        # Extract text from dataset
        texts = []
        for example in dataset[split]:
            # PTB has sentences in 'sentence' field
            sentence = example.get('sentence', '')
            if sentence.strip():
                texts.append(sentence.strip())

        # Write to file (one sentence per line)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(texts))

        num_lines = len(texts)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)

        print(f"   ✅ Saved to: {output_file}")
        print(f"   Lines: {num_lines:,}")
        print(f"   Size: {file_size_mb:.2f} MB")

        output_files[split_name] = output_file

    # Print summary
    print("\n" + "="*60)
    print("PTB Dataset Setup Complete!")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print("\nFiles created:")
    for split_name, filepath in output_files.items():
        print(f"  - {filepath.name}")

    print("\n📖 Penn Treebank Info:")
    print("  - Standard benchmark for language modeling")
    print("  - ~1 million tokens total")
    print("  - Vocabulary: ~10k words")
    print("  - Preprocessed and tokenized sentences")

    print("\n🚀 Next steps:")
    print("  1. Pre-tokenize for training:")
    print(f"     python pretokenize_dataset.py {output_files['train']}")
    print(f"  2. Pre-tokenize validation:")
    print(f"     python pretokenize_dataset.py {output_files['valid']}")
    print(f"  3. Pre-tokenize test:")
    print(f"     python pretokenize_dataset.py {output_files['test']}")
    print("="*60)


if __name__ == "__main__":
    setup_ptb()
