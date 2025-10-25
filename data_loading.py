"""
data_loading.py - Dataset classes and data loading utilities

This module contains all dataset-related functionality including:
- TextDataset: Original in-memory dataset loading
- StreamingTextDataset: Memory-efficient streaming dataset for large files
- Smart dataset selection functions
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import random
from pathlib import Path
from transformers import GPT2Tokenizer


class TextDataset(Dataset):
    def __init__(self, text, seq_length, tokenizer, stride=1, random_offset=True):
        """
        Improved TextDataset with proper chunking for GPT2 tokenizer.
        """
        # First, break text into smaller chunks (aim for ~512 tokens per chunk)
        chunk_size = 2000  # characters, which should give roughly 400-500 tokens
        text_chunks = [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]

        all_tokens = []
        for chunk in text_chunks:
            chunk_tokens = tokenizer(chunk, truncation=True, max_length=1024)[
                "input_ids"
            ]
            all_tokens.extend(chunk_tokens)

        # Convert to tensor
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

        # Create sequences of seq_length (+1 for targets)
        total_length = self.tokens.size(0)
        self.seq_length = seq_length

        # Calculate number of complete sequences we can make
        n_seqs = (total_length - seq_length - 1) // stride

        self.sequences = []
        self.targets = []

        # Create sequences with stride
        for i in range(n_seqs):
            start_idx = i * stride
            end_idx = start_idx + seq_length
            self.sequences.append(self.tokens[start_idx:end_idx])
            self.targets.append(self.tokens[start_idx + 1 : end_idx + 1])

        self.sequences = torch.stack(self.sequences)
        self.targets = torch.stack(self.targets)

        print(f"Created {len(self.sequences)} sequences of length {seq_length}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader that pads sequences to same length.
        """
        # Separate sequences and targets
        sequences, targets = zip(*batch)

        # Stack into tensors
        sequences = torch.stack(sequences)
        targets = torch.stack(targets)

        return sequences, targets


class StreamingTextDataset(Dataset):
    def __init__(
        self, file_path, seq_length, tokenizer, max_tokens=None, split="train"
    ):
        """
        Memory-efficient streaming dataset that loads text chunks on-demand.

        Args:
            file_path: Path to the text file
            seq_length: Length of each sequence
            tokenizer: Tokenizer to use
            max_tokens: Maximum number of tokens to use (None = use all)
            split: 'train' or 'val' - determines which part of file to use
        """
        self.file_path = file_path
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.split = split

        # Get file size
        file_size = Path(file_path).stat().st_size

        # Determine which part of the file to use
        if split == "train":
            self.start_byte = 0
            if max_tokens:
                # Estimate characters needed (rough 4:1 char:token ratio)
                chars_needed = max_tokens * 4
                self.end_byte = min(chars_needed, int(file_size * 0.9))
            else:
                self.end_byte = int(file_size * 0.9)  # Use 90% for training
        elif split == "val":
            # Use last 10% for validation
            self.start_byte = int(file_size * 0.9)
            self.end_byte = file_size
            if max_tokens:
                # Limit validation size if specified
                chars_needed = max_tokens * 4
                self.end_byte = min(self.start_byte + chars_needed, file_size)
        else:
            self.start_byte = 0
            self.end_byte = file_size

        # Calculate approximate number of sequences
        usable_bytes = self.end_byte - self.start_byte
        self.approx_sequences = max(1, usable_bytes // (seq_length * 4))

        # Cache for file handle and recent reads
        self._file_handle = None
        self._cache = {}
        self._cache_size = 1000  # Cache up to 1000 recent reads

        print(
            f"StreamingDataset ({split}): ~{self.approx_sequences:,} sequences from {usable_bytes:,} bytes"
        )

    def __len__(self):
        return self.approx_sequences

    def __getitem__(self, idx):
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]

        # Calculate file position with some randomization for better data coverage
        position_range = self.end_byte - self.start_byte - (self.seq_length * 6)
        if position_range <= 0:
            position = self.start_byte
        else:
            # Use a deterministic but varied positioning based on idx
            position = self.start_byte + (idx * self.seq_length * 3) % position_range

        # Open file handle in BINARY mode to avoid UTF-8 decode issues with seeking
        if self._file_handle is None:
            self._file_handle = open(self.file_path, "rb")  # Changed to binary mode

        try:
            # Read chunk from file in binary mode
            self._file_handle.seek(position)
            raw_bytes = self._file_handle.read(
                self.seq_length * 8
            )  # Extra buffer for UTF-8 and safety

            if not raw_bytes:
                # Fallback to beginning of our section if we hit EOF
                self._file_handle.seek(self.start_byte)
                raw_bytes = self._file_handle.read(self.seq_length * 8)

            # Decode with error handling - skip invalid bytes
            chunk = raw_bytes.decode("utf-8", errors="ignore")

            # If we got very little text after decode, try a larger chunk
            if len(chunk) < self.seq_length:
                # Try reading more data
                additional_bytes = self._file_handle.read(self.seq_length * 8)
                chunk += additional_bytes.decode("utf-8", errors="ignore")

            # Tokenize the chunk
            tokens = self.tokenizer(
                chunk, truncation=True, max_length=self.seq_length + 10
            )["input_ids"]

            # Ensure we have enough tokens
            if len(tokens) < self.seq_length + 1:
                # Repeat tokens if needed (shouldn't happen often with large chunks)
                tokens = (tokens * 3)[: self.seq_length + 1]

            # Create sequence and target
            sequence = torch.tensor(tokens[: self.seq_length], dtype=torch.long)
            target = torch.tensor(tokens[1 : self.seq_length + 1], dtype=torch.long)

            result = (sequence, target)

            # Cache result if cache isn't full
            if len(self._cache) < self._cache_size:
                self._cache[idx] = result

            return result

        except Exception as e:
            print(f"Warning: Error reading from streaming dataset at idx {idx}: {e}")
            # Return a fallback sequence
            fallback_tokens = [self.tokenizer.pad_token_id or 0] * (self.seq_length + 1)
            sequence = torch.tensor(
                fallback_tokens[: self.seq_length], dtype=torch.long
            )
            target = torch.tensor(
                fallback_tokens[1 : self.seq_length + 1], dtype=torch.long
            )
            return sequence, target

    def __del__(self):
        """Clean up file handle"""
        if self._file_handle:
            self._file_handle.close()

    @staticmethod
    def collate_fn(batch):
        """Same collate function as TextDataset"""
        return TextDataset.collate_fn(batch)


def get_dataset(config):
    """
    Original dataset loading function - loads entire dataset into memory.

    Args:
        config: Configuration object containing dataset parameters

    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer, full_text)
    """
    # Move this block before get_dataset() call
    if hasattr(config, "get"):
        # Dictionary-like config
        seed = config.get("seed", 123)
        data_path = config.get("data_path", None)
        max_tokens_training = config.get("max_tokens_training", None)
        # Support old parameter name for backward compatibility
        if max_tokens_training is None:
            max_tokens_training = config.get("max_tokens", None)
    else:
        # Object-like config
        seed = getattr(config, "seed", 123)
        data_path = getattr(config, "data_path", None)
        max_tokens_training = getattr(config, "max_tokens_training", None)
        # Support old parameter name for backward compatibility
        if max_tokens_training is None:
            max_tokens_training = getattr(config, "max_tokens", None)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize GPT2 tokenizer from local files first (needed for token-based loading)
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_tokenizer")
    except:
        raise FileNotFoundError(
            "GPT2 tokenizer files not found in ./gpt2_tokenizer. "
            "Please download the tokenizer files first."
        )

    # Get the text data from configurable path (like LSTM system)
    if not data_path:
        raise ValueError("data_path must be specified in config to load dataset")

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset file not found at {data_path}. "
            "Please ensure you have downloaded and copied the dataset file."
        )

    print(f"Loading dataset from: {data_path}")

    # Smart loading: determine how much data we actually need before loading
    max_tokens_training = (
        config.get("max_tokens_training")
        if hasattr(config, "get")
        else getattr(config, "max_tokens_training", None)
    )
    # Support old parameter name for backward compatibility
    if max_tokens_training is None:
        max_tokens_training = (
            config.get("max_tokens")
            if hasattr(config, "get")
            else getattr(config, "max_tokens", None)
        )

    fixed_val_tokens = (
        config.get("fixed_val_tokens")
        if hasattr(config, "get")
        else getattr(config, "fixed_val_tokens", None)
    )

    if max_tokens_training:
        # Calculate exactly how much data we need BEFORE loading
        train_split = (
            config.get("train_split", 0.9)
            if hasattr(config, "get")
            else getattr(config, "train_split", 0.9)
        )
        char_to_token_ratio = config.get("char_to_token_ratio", 4.0)

        if fixed_val_tokens:
            total_tokens_needed = max_tokens_training + fixed_val_tokens
        else:
            total_tokens_needed = int(max_tokens_training / train_split)

        max_characters_needed = int(total_tokens_needed * char_to_token_ratio)

        # Get file size without loading entire file
        file_size = Path(data_path).stat().st_size

        if file_size > max_characters_needed * 2:  # If file is much larger than needed
            # Smart loading: read only what we need + buffer for random sampling
            buffer_multiplier = 2.0  # 100% extra for random positioning
            chars_to_load = min(
                int(max_characters_needed * buffer_multiplier), file_size
            )

            with open(data_path, "r", encoding="utf-8") as f:
                # Random start position for data variety
                max_start = max(0, file_size - chars_to_load)
                start_pos = random.randint(0, max_start) if max_start > 0 else 0
                f.seek(start_pos)
                # Skip partial line at start
                if start_pos > 0:
                    f.readline()

                text = f.read(chars_to_load)

            print(
                f"Smart loading: loaded {len(text):,} chars from {file_size:,}-char file (need ~{max_characters_needed:,})"
            )
        else:
            # File is reasonably sized, load normally
            text = Path(data_path).read_text(encoding="utf-8")
            print(f"Standard loading: {len(text):,} characters")
    else:
        # No token limit specified, load entire file
        text = Path(data_path).read_text(encoding="utf-8")
        print(f"Full loading (no limit): {len(text):,} characters")

    # Get split configuration first to determine how to limit the dataset
    if hasattr(config, "get"):
        # Dictionary-like config
        train_split = config.get("train_split", 0.9)
        fixed_val_tokens = config.get("fixed_val_tokens", None)
    else:
        # Object-like config
        train_split = getattr(config, "train_split", 0.9)
        fixed_val_tokens = getattr(config, "fixed_val_tokens", None)

    # Apply token-based limit if specified (convert to characters using 4:1 ratio)
    if max_tokens_training:
        if fixed_val_tokens:
            # When fixed_val_tokens is specified, we need exactly max_tokens_training + fixed_val_tokens
            total_tokens_needed = max_tokens_training + fixed_val_tokens
            print(
                f"Fixed validation mode: need exactly {max_tokens_training:,} training + {fixed_val_tokens:,} validation tokens"
            )
        else:
            # Calculate total tokens needed so that training portion = max_tokens_training
            total_tokens_needed = int(max_tokens_training / train_split)
            print(
                f"Percentage split mode: need {total_tokens_needed:,} total tokens for {max_tokens_training:,} training tokens"
            )

        # Convert token limit to character limit using configurable ratio
        char_to_token_ratio = config.get("char_to_token_ratio", 4.0)
        max_characters = int(total_tokens_needed * char_to_token_ratio)
        if len(text) > max_characters:
            # Random sampling for variety in training data
            start_idx = random.randint(0, max(0, len(text) - max_characters))
            text = text[start_idx : start_idx + max_characters]
            print(
                f"Limited dataset to {max_characters:,} characters (~{total_tokens_needed:,} tokens)"
            )
            print(
                f"  Sampled from {len(Path(data_path).read_text(encoding='utf-8')):,} total characters"
            )
        else:
            print(
                f"Using full dataset: {len(text):,} characters (~{int(len(text)/char_to_token_ratio):,} tokens)"
            )
    else:
        char_to_token_ratio = config.get("char_to_token_ratio", 4.0)
        print(
            f"Using full dataset: {len(text):,} characters (~{int(len(text)/char_to_token_ratio):,} tokens)"
        )

    # BETTER SPLIT: Random shuffle before splitting

    # Split into sentences/paragraphs first, then shuffle
    sentences = text.split("\n")
    random.shuffle(sentences)

    # Note: split configuration already loaded above

    # Validate split ratios
    if not (0.0 < train_split < 1.0):
        raise ValueError(f"train_split must be between 0 and 1, got: {train_split}")

    # When fixed_val_tokens is used, we don't use percentage splits for validation
    if fixed_val_tokens is None:
        # Calculate val_split from train_split (remaining portion)
        val_split = 1.0 - train_split
        if val_split <= 0:
            raise ValueError(
                f"train_split too large, no room for validation: {train_split}"
            )
    else:
        # Fixed validation tokens - percentage split validation not applicable
        val_split = None

    if fixed_val_tokens is not None:
        # Use fixed validation token size - take exactly the amounts specified
        if fixed_val_tokens <= 0:
            raise ValueError(
                f"fixed_val_tokens must be positive, got: {fixed_val_tokens}"
            )

        # Convert token counts to character counts (4:1 ratio)
        if max_tokens_training:
            training_chars_needed = int(max_tokens_training * 4)
        else:
            # If no max_tokens_training specified, use default percentage of available text
            training_chars_needed = int(len(text) * train_split)

        fixed_val_chars = int(fixed_val_tokens * 4)
        total_chars_needed = training_chars_needed + fixed_val_chars

        # Check if we have enough data
        if total_chars_needed > len(text):
            raise ValueError(
                f"Not enough data: need {total_chars_needed:,} characters "
                f"({training_chars_needed:,} training + {fixed_val_chars:,} validation) "
                f"but only have {len(text):,} characters available"
            )

        # Take exactly the specified amounts from separate parts of the dataset
        train_text = text[:training_chars_needed]  # First N characters for training
        val_text = text[
            training_chars_needed : training_chars_needed + fixed_val_chars
        ]  # Next M characters for validation

        print(f"Using fixed validation size with separate datasets:")
        print(
            f"  Training set: {len(train_text):,} characters (~{int(len(train_text)/char_to_token_ratio):,} tokens)"
        )
        print(
            f"  Validation set: {len(val_text):,} characters (~{int(len(val_text)/char_to_token_ratio):,} tokens)"
        )
        print(f"  Datasets are completely separate (no overlap)")
    else:
        # Use percentage-based split
        split_idx = int(len(sentences) * train_split)
        train_sentences = sentences[:split_idx]
        val_sentences = sentences[split_idx:]

        train_text = "\n".join(train_sentences)
        val_text = "\n".join(val_sentences)

        print(
            f"Using percentage-based split: {train_split*100:.1f}% train, {val_split*100:.1f}% validation"
        )
        print(
            f"Training set: {len(train_text):,} characters (~{int(len(train_text)/char_to_token_ratio):,} tokens)"
        )
        print(
            f"Validation set: {len(val_text):,} characters (~{int(len(val_text)/char_to_token_ratio):,} tokens)"
        )

    # Create datasets with GPT2 tokenizer
    if hasattr(config, "get"):
        # Dictionary-like config
        seq_length = config.get("seq_length", 128)
        stride = config.get("stride", 128)
    else:
        # Object-like config
        seq_length = getattr(config, "seq_length", 128)
        stride = getattr(config, "stride", 128)

    train_dataset = TextDataset(
        text=train_text,
        seq_length=seq_length,
        tokenizer=tokenizer,
        stride=stride,
        random_offset=True,
    )

    val_dataset = TextDataset(
        text=val_text,
        seq_length=seq_length,
        tokenizer=tokenizer,
        stride=stride,
        random_offset=False,
    )

    return train_dataset, val_dataset, tokenizer, text


def get_streaming_dataset(config):
    """
    Streaming dataset loading function - memory efficient for large datasets.

    Args:
        config: Configuration object containing dataset parameters

    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer, "streaming_text")
    """
    # Get configuration values
    if hasattr(config, "get"):
        seed = config.get("seed", 123)
        data_path = config.get("data_path", None)
        max_tokens_training = config.get("max_tokens_training", None)
        fixed_val_tokens = config.get("fixed_val_tokens", None)
        seq_length = config.get("seq_length", 128)
    else:
        seed = getattr(config, "seed", 123)
        data_path = getattr(config, "data_path", None)
        max_tokens_training = getattr(config, "max_tokens_training", None)
        fixed_val_tokens = getattr(config, "fixed_val_tokens", None)
        seq_length = getattr(config, "seq_length", 128)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_tokenizer")
    except:
        raise FileNotFoundError(
            "GPT2 tokenizer files not found in ./gpt2_tokenizer. "
            "Please download the tokenizer files first."
        )

    # Validate data path
    if not data_path:
        raise ValueError("data_path must be specified in config to load dataset")

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset file not found at {data_path}. "
            "Please ensure you have downloaded and copied the dataset file."
        )

    print(f"Loading streaming dataset from: {data_path}")

    # Create streaming datasets
    train_dataset = StreamingTextDataset(
        file_path=data_path,
        seq_length=seq_length,
        tokenizer=tokenizer,
        max_tokens=max_tokens_training,
        split="train",
    )

    val_dataset = StreamingTextDataset(
        file_path=data_path,
        seq_length=seq_length,
        tokenizer=tokenizer,
        max_tokens=fixed_val_tokens,
        split="val",
    )

    return train_dataset, val_dataset, tokenizer, "streaming_text"


def get_dataset_smart(config):
    """
    Smart dataset loader - automatically chooses streaming vs memory based on config.

    Args:
        config: Configuration object with use_streaming_dataset flag

    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer, text_or_indicator)
    """
    use_streaming = (
        config.get("use_streaming_dataset", False)
        if hasattr(config, "get")
        else getattr(config, "use_streaming_dataset", False)
    )

    if use_streaming:
        print("📡 Using streaming dataset (memory efficient)")
        return get_streaming_dataset(config)
    else:
        print("💾 Using in-memory dataset (original method)")
        return get_dataset(config)
