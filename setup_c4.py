from datasets import load_dataset, DownloadConfig
from transformers import GPT2Tokenizer
from pathlib import Path

# Where to cache the shard downloads
CACHE_DIR = "/your/real/cache/dir"  # ← adjust as needed

# How many GPT-2 tokens to collect (≈1e9)
TOKEN_LIMIT = 10**10
# 3*10**9 worked well

# Ensure target directory exists
Path("Datasets").mkdir(exist_ok=True)

# Load your local GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_tokenizer")

# Configure streaming download
download_config = DownloadConfig(use_etag=False, cache_dir=CACHE_DIR)

collected_texts = []
total_tokens = 0

# Stream through C4 until we hit TOKEN_LIMIT
for example in load_dataset(
    "c4",
    "en",
    split="train",
    streaming=True,
    download_config=download_config,
):
    text = example["text"]

    # Just count characters instead of tokens
    if total_tokens + len(text) > TOKEN_LIMIT:  # 4B characters ≈ 1B tokens
        break
    collected_texts.append(text)
    total_tokens += len(text)

# Write out one big text file
out_file = Path("Datasets") / "c4_subset.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(collected_texts))

print(
    f"Saved {len(collected_texts)} C4 examples (~{total_tokens:,} tokens) to {out_file}"
)
