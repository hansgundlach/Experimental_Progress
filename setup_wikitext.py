from datasets import load_dataset
from transformers import GPT2Tokenizer
from pathlib import Path

# WikiText-103 is a standard language modeling benchmark
# Contains ~100M tokens from Wikipedia articles
# We'll download both train and validation/test splits

# Ensure target directory exists
Path("Datasets").mkdir(exist_ok=True)

# Load your local GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_tokenizer")

print("Downloading WikiText-103 dataset...")

# Load WikiText-103 (larger version, ~100M tokens)
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Process each split
for split_name in ["train", "validation", "test"]:
    print(f"\nProcessing {split_name} split...")

    split_data = dataset[split_name]

    # Collect all text
    all_text = []
    total_chars = 0

    for example in split_data:
        text = example["text"]
        # Skip empty lines (WikiText has many of these)
        if text.strip():
            all_text.append(text)
            total_chars += len(text)

    # Write out text file
    out_file = Path("Datasets") / f"wikitext103_{split_name}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    # Calculate approximate tokens
    approx_tokens = total_chars // 4

    print(f"  Saved {len(all_text)} examples")
    print(f"  ~{total_chars:,} characters (~{approx_tokens:,} tokens)")
    print(f"  Output: {out_file}")

print("\n" + "="*60)
print("WikiText-103 download complete!")
print("="*60)
print("Files created:")
print("  Datasets/wikitext103_train.txt       (training data)")
print("  Datasets/wikitext103_validation.txt  (validation data)")
print("  Datasets/wikitext103_test.txt        (test data)")
print("\nUse validation.txt for cross-dataset evaluation experiments.")
