from datasets import load_dataset
from pathlib import Path

# Download the official wikitext-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Create directory if it doesn't exist
Path("Datasets").mkdir(exist_ok=True)

# Save the dataset as a text file
with open("Datasets/wikitext.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(dataset["text"]))

print("Dataset saved to Datasets/wikitext.txt")
