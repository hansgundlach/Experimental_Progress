from datasets import load_dataset, DownloadConfig
from transformers import GPT2Tokenizer
from pathlib import Path

# Where to cache the shard downloads
CACHE_DIR = "./hf_cache"  # ← adjust as needed

# How many characters to collect (≈4:1 char:token ratio)
CHAR_LIMIT = 4 * 10**9  # ~1B tokens

# Ensure target directory exists
Path("../Datasets").mkdir(exist_ok=True)

# Load your local GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("../gpt2_tokenizer")

# Configure streaming download
download_config = DownloadConfig(use_etag=False, cache_dir=CACHE_DIR)

collected_texts = []
total_chars = 0

print(f"Downloading The Pile (~{CHAR_LIMIT//4:,} tokens)...")
print("Note: The Pile is very large (825GB). First download may take a while.")

# Stream through The Pile until we hit CHAR_LIMIT
# Using monology/pile-uncopyrighted which is hosted on HuggingFace directly
for example in load_dataset(
    "monology/pile-uncopyrighted",
    split="train",
    streaming=True,
    download_config=download_config,
):
    text = example["text"]

    if total_chars + len(text) > CHAR_LIMIT:
        break
    collected_texts.append(text)
    total_chars += len(text)

    if len(collected_texts) % 1000 == 0:
        print(f"  {len(collected_texts)} examples, {total_chars:,} chars ({total_chars//4:,} tokens)")

# Write out one big text file
out_file = Path("../Datasets") / "pile_subset.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(collected_texts))

print(
    f"Saved {len(collected_texts)} examples (~{total_chars:,} chars, ~{total_chars//4:,} tokens) to {out_file}"
)
