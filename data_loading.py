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


class TokenDataset(Dataset):
    """
    Simple dataset for pre-tokenized data (list of token IDs).
    Used by LSTM and can be used by any model working with token lists.
    Matches old LSTM TextDataset behavior exactly.
    """
    def __init__(self, token_list, seq_length, stride=1):
        """
        Args:
            token_list: List of token IDs
            seq_length: Length of each sequence
            stride: Stride for creating sequences
        """
        self.tokens = token_list
        self.seq_length = seq_length
        self.stride = stride

    def __len__(self):
        # Number of windows we can slide over the data (matches old LSTM exactly)
        # Need seq_length tokens for input + 1 for target = seq_length+1 total
        # Valid starting positions: 0 to len-seq_length-1 (inclusive)
        # That's (len - seq_length) positions total
        raw = (len(self.tokens) - self.seq_length) // self.stride
        return max(0, raw)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_length
        seq = torch.tensor(self.tokens[start:end], dtype=torch.long)
        tgt = torch.tensor(self.tokens[start + 1:end + 1], dtype=torch.long)
        return seq, tgt


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
        self, file_path, seq_length, tokenizer, max_tokens, stride=128, offset=0
    ):
        """
        Memory-efficient streaming dataset using memory-mapped files.

        If a pre-tokenized .npy file exists, uses memory mapping (no RAM usage!).
        Otherwise falls back to loading and tokenizing the text file (uses RAM).

        Args:
            file_path: Path to the text file (or .npy file if pre-tokenized)
            seq_length: Length of each sequence
            tokenizer: Tokenizer to use (only needed if loading text)
            max_tokens: Exact number of tokens to use (REQUIRED)
            stride: Stride for creating sequences (default: 128, same as TextDataset)
            offset: Number of tokens to skip from start (for validation split)
        """
        if not max_tokens:
            raise ValueError("max_tokens is required - percentage splits are no longer supported")

        self.seq_length = seq_length
        self.stride = stride
        self.tokenizer = tokenizer

        # Check if pre-tokenized .npy file exists
        npy_path = Path(file_path).with_suffix('.npy')

        if npy_path.exists():
            # MEMORY-MAPPED LOADING (uses almost no RAM!)
            print(f"🚀 Using memory-mapped file: {npy_path}")
            print(f"   (This uses ~0MB RAM regardless of dataset size!)")

            # Load as memory-mapped array (doesn't load into RAM)
            all_tokens_mmap = np.load(str(npy_path), mmap_mode='r')
            total_tokens = len(all_tokens_mmap)

            print(f"   Total tokens in file: {total_tokens:,}")

            # Handle negative offset (from end of file)
            if offset < 0:
                # Negative offset = from end (like old 90/10 split for validation)
                start_idx = max(0, total_tokens + offset)  # offset is negative, so this subtracts
                end_idx = total_tokens
                print(f"   Loading from END of file (last {max_tokens:,} tokens)")
            else:
                # Positive offset = from start
                start_idx = offset
                end_idx = min(offset + max_tokens, total_tokens)

            # Store the memory-mapped view (just a reference, no RAM used)
            self.tokens = all_tokens_mmap[start_idx:end_idx]
            total_length = len(self.tokens)

            print(f"   Using {total_length:,} tokens (indices {start_idx:,} to {end_idx:,})")

        else:
            # FALLBACK: Load text file (loads into RAM)
            print(f"⚠️  WARNING: No .npy file found at {npy_path}")
            print(f"   Falling back to loading text file (uses ~{Path(file_path).stat().st_size / 1e9:.1f}GB RAM)")
            print(f"   To avoid this, run: python pretokenize_dataset.py {file_path}")

            # Load exactly max_tokens worth of text
            chars_needed = max_tokens * 4  # Approximate 4 chars per token
            file_size = Path(file_path).stat().st_size
            chars_to_load = min(chars_needed, file_size)

            # Load and tokenize (uses RAM)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read(chars_to_load)

            chunk_size = 2000
            text_chunks = [
                text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
            ]

            all_tokens = []
            for chunk in text_chunks:
                chunk_tokens = tokenizer(chunk, truncation=True, max_length=1024)[
                    "input_ids"
                ]
                all_tokens.extend(chunk_tokens)

            # Limit to exactly max_tokens
            self.tokens = all_tokens[:max_tokens]
            total_length = len(self.tokens)

        # Calculate number of complete sequences (IDENTICAL to TextDataset)
        self.n_seqs = (total_length - seq_length - 1) // stride

        print(
            f"StreamingDataset: {self.n_seqs:,} sequences from {total_length:,} tokens"
        )

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        """
        Create sequences on-demand using IDENTICAL logic to TextDataset.
        This is where memory savings happen - we don't store all sequences.
        """
        # IDENTICAL to TextDataset: start_idx = i * stride
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_length

        # Extract sequence and target from token list
        sequence = torch.tensor(
            self.tokens[start_idx:end_idx], dtype=torch.long
        )
        target = torch.tensor(
            self.tokens[start_idx + 1 : end_idx + 1], dtype=torch.long
        )

        return sequence, target

    @staticmethod
    def collate_fn(batch):
        """Same collate function as TextDataset"""
        return TextDataset.collate_fn(batch)


class LSTMStatefulDataset(Dataset):
    """
    LSTM-specific stateful streaming dataset for Melis/Merity style training.

    Reshapes data into B contiguous streams where each batch consists of
    the next tokens from B continuous streams. This maintains hidden state
    continuity across batches for stateful LSTM training.

    EXACTLY matches old TextStreamingDataset behavior.
    """
    def __init__(self, token_list, seq_length, batch_size):
        """
        Args:
            token_list: List of token IDs
            seq_length: Length of each sequence
            batch_size: Number of parallel streams
        """
        self.seq_length = seq_length
        self.batch_size = batch_size

        # Calculate how many tokens we can use (must be divisible by batch_size)
        total_tokens = len(token_list)
        tokens_per_stream = total_tokens // batch_size
        usable_tokens = tokens_per_stream * batch_size

        if usable_tokens < batch_size * seq_length:
            raise ValueError(
                f"Dataset too small: need at least {batch_size * seq_length} tokens, "
                f"got {usable_tokens}"
            )

        # Reshape into [B, T_stream] where T_stream = tokens_per_stream
        # EXACT OLD BEHAVIOR: Store as tensor
        data_tensor = torch.tensor(token_list[:usable_tokens], dtype=torch.long)
        self.streams = data_tensor.view(batch_size, tokens_per_stream)

        # Number of sequence-length windows we can extract from each stream
        self.num_batches = (tokens_per_stream - 1) // seq_length

        print(
            f"LSTMStatefulDataset: {batch_size} streams, {tokens_per_stream} tokens per stream, "
            f"{self.num_batches} batches"
        )

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        """
        Return batch of sequences from B parallel streams.

        Returns:
            inputs: [batch_size, seq_length]
            targets: [batch_size, seq_length]
        """
        start_pos = idx * self.seq_length
        end_pos = start_pos + self.seq_length

        # Extract from all streams
        inputs = self.streams[:, start_pos:end_pos]
        targets = self.streams[:, start_pos + 1:end_pos + 1]

        return inputs, targets

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for stateful dataset.
        Since __getitem__ already returns proper batches, just return first item.
        """
        # batch will be a list with one element (the batch from __getitem__)
        return batch[0]


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
        val_data_path = config.get("val_data_path", None)  # NEW: separate validation dataset
        max_tokens_training = config.get("max_tokens_training", None)
        # Support old parameter name for backward compatibility
        if max_tokens_training is None:
            max_tokens_training = config.get("max_tokens", None)
    else:
        # Object-like config
        seed = getattr(config, "seed", 123)
        data_path = getattr(config, "data_path", None)
        val_data_path = getattr(config, "val_data_path", None)  # NEW: separate validation dataset
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

    # Check if separate validation dataset is specified
    if val_data_path:
        print(f"\nLoading separate validation dataset from: {val_data_path}")
        # Handle relative paths (for LSTM running from subdirectory)
        val_path_obj = Path(val_data_path)
        if not val_path_obj.exists():
            parent_relative = Path("..") / val_data_path
            if parent_relative.exists():
                val_data_path = str(parent_relative)
                print(f"  (adjusted path: {val_data_path})")

        val_text = Path(val_data_path).read_text(encoding="utf-8")
        print(f"  Validation set: {len(val_text):,} characters")

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
        val_data_path = config.get("val_data_path", None)  # NEW: separate validation dataset
        max_tokens_training = config.get("max_tokens_training", None)
        fixed_val_tokens = config.get("fixed_val_tokens", None)
        seq_length = config.get("seq_length", 128)
        stride = config.get("stride", 128)
    else:
        seed = getattr(config, "seed", 123)
        data_path = getattr(config, "data_path", None)
        val_data_path = getattr(config, "val_data_path", None)  # NEW: separate validation dataset
        max_tokens_training = getattr(config, "max_tokens_training", None)
        fixed_val_tokens = getattr(config, "fixed_val_tokens", None)
        seq_length = getattr(config, "seq_length", 128)
        stride = getattr(config, "stride", 128)

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

    # Validate fixed_val_tokens is provided
    if not fixed_val_tokens:
        raise ValueError("fixed_val_tokens is required - percentage splits are no longer supported")

    print(f"Loading streaming dataset from: {data_path}")
    print(f"  Training tokens: {max_tokens_training:,}")
    print(f"  Validation tokens: {fixed_val_tokens:,}")

    # Check if using separate validation file
    if val_data_path:
        print(f"\nUsing separate validation dataset: {val_data_path}")
        # Handle relative paths (for LSTM running from subdirectory)
        val_path_obj = Path(val_data_path)
        if not val_path_obj.exists():
            parent_relative = Path("..") / val_data_path
            if parent_relative.exists():
                val_data_path = str(parent_relative)
                print(f"  (adjusted path: {val_data_path})")

        # Create separate train and val datasets from different files
        train_dataset = StreamingTextDataset(
            file_path=data_path,
            seq_length=seq_length,
            tokenizer=tokenizer,
            max_tokens=max_tokens_training,
            stride=stride,
        )

        val_dataset = StreamingTextDataset(
            file_path=val_data_path,
            seq_length=seq_length,
            tokenizer=tokenizer,
            max_tokens=fixed_val_tokens,
            stride=stride,
        )
    else:
        # Create train and val datasets from same file
        # Training: first max_tokens_training tokens
        train_dataset = StreamingTextDataset(
            file_path=data_path,
            seq_length=seq_length,
            tokenizer=tokenizer,
            max_tokens=max_tokens_training,
            stride=stride,
            offset=0,
        )

        # Validation: last fixed_val_tokens tokens (from END of file, like old 90/10 split)
        print(f"\nCreating validation dataset from END of file (last {fixed_val_tokens:,} tokens)")
        # Need to get total tokens to calculate offset from end
        # Use "from_end" offset = -fixed_val_tokens to load from end
        val_dataset = StreamingTextDataset(
            file_path=data_path,
            seq_length=seq_length,
            tokenizer=tokenizer,
            max_tokens=fixed_val_tokens,
            stride=stride,
            offset=-fixed_val_tokens,  # Negative = from end of file
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


def load_and_tokenize_text(config):
    """
    Load and tokenize text without creating datasets.
    Returns tokenized data split into train/val/test for LSTM use.

    This function handles all the smart loading logic and returns token lists
    that can be used to create either regular TextDataset, StreamingTextDataset,
    or LSTMStatefulDataset.

    Args:
        config: Configuration dict with keys:
            - data_path: Path to text file
            - seed: Random seed
            - max_tokens_training: Target training tokens (REQUIRED)
            - fixed_val_tokens: Fixed validation tokens (REQUIRED)
            - val_data_path: Optional separate validation dataset
            - char_to_token_ratio: Chars per token estimate (default 4.0)

    Returns:
        Tuple of (train_tokens, val_tokens, test_tokens, tokenizer)
        where each tokens is a list of token IDs
    """
    # Get configuration values
    if hasattr(config, "get"):
        seed = config.get("seed", 123)
        data_path = config.get("data_path", None)
        val_data_path = config.get("val_data_path", None)
        max_tokens_training = config.get("max_tokens_training", None)
        fixed_val_tokens = config.get("fixed_val_tokens", None)
        tokenizer_path = config.get("tokenizer_path", "./gpt2_tokenizer")
    else:
        seed = getattr(config, "seed", 123)
        data_path = getattr(config, "data_path", None)
        val_data_path = getattr(config, "val_data_path", None)
        max_tokens_training = getattr(config, "max_tokens_training", None)
        fixed_val_tokens = getattr(config, "fixed_val_tokens", None)
        tokenizer_path = getattr(config, "tokenizer_path", "./gpt2_tokenizer")

    # Validate required parameters
    if not max_tokens_training:
        raise ValueError("max_tokens_training is required in config")
    if not fixed_val_tokens:
        raise ValueError("fixed_val_tokens is required in config - percentage splits are no longer supported")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize tokenizer (support both LSTM-style path and default)
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,  # LSTM compatibility: force offline mode
            use_fast=False,  # LSTM compatibility: use slow tokenizer
        )
    except:
        raise FileNotFoundError(
            f"GPT2 tokenizer files not found at {tokenizer_path}. "
            "Please download the tokenizer files first."
        )

    # Validate data path
    if not data_path:
        raise ValueError("data_path must be specified in config to load dataset")

    # Handle relative paths - try both as-is and relative to parent directory
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        # Try relative to parent directory (for LSTM running from subdirectory)
        parent_relative = Path("..") / data_path
        if parent_relative.exists():
            data_path = str(parent_relative)
            print(f"Found dataset at: {data_path} (adjusted for subdirectory)")
        else:
            raise FileNotFoundError(
                f"Dataset file not found at {data_path}. "
                f"Also tried: {parent_relative}. "
                "Please ensure you have downloaded and copied the dataset file."
            )
    else:
        print(f"Loading text from: {data_path}")

    # ============================================================
    # CHECK FOR MEMORY-MAPPED .npy FILE (SAME AS StreamingTextDataset)
    # ============================================================
    npy_path = Path(data_path).with_suffix('.npy')
    # Also check parent-relative path for LSTM
    if not npy_path.exists():
        parent_npy = Path("..") / data_path
        parent_npy = parent_npy.with_suffix('.npy')
        if parent_npy.exists():
            npy_path = parent_npy

    if npy_path.exists():
        # MEMORY-MAPPED LOADING (SAME AS StreamingTextDataset)
        print(f"🚀 LSTM: Using memory-mapped file: {npy_path}")
        print(f"   (This uses ~0MB RAM regardless of dataset size!)")

        # Load as memory-mapped array (doesn't load into RAM)
        all_tokens_mmap = np.load(str(npy_path), mmap_mode='r')
        total_tokens_available = len(all_tokens_mmap)
        print(f"   Total tokens in file: {total_tokens_available:,}")

        # Determine how many tokens to use (fixed sizes only)
        tokens_needed = int(max_tokens_training + fixed_val_tokens)
        tokens_to_use = int(min(tokens_needed, total_tokens_available))

        # Extract tokens from memory-mapped array
        tokens = all_tokens_mmap[:tokens_to_use].tolist()  # Convert to list for LSTM compatibility
        print(f"   Using {len(tokens):,} tokens from memory-mapped file")

    else:
        # FALLBACK: Load and tokenize text file (EXACTLY LIKE StreamingTextDataset)
        print(f"⚠️  LSTM: No .npy file found at {npy_path}")
        print(f"   Loading and tokenizing text file (uses RAM)")
        print(f"   To use memory mapping, run: python pretokenize_dataset.py {data_path}")

        # Smart loading logic (same as existing functions)
        if max_tokens_training:
            char_to_token_ratio = (
                config.get("char_to_token_ratio", 4.0)
                if hasattr(config, "get")
                else getattr(config, "char_to_token_ratio", 4.0)
            )

            # Calculate total tokens needed (fixed sizes only)
            total_tokens_needed = max_tokens_training + fixed_val_tokens
            max_characters_needed = int(total_tokens_needed * char_to_token_ratio)

            # Get file size
            file_size = Path(data_path).stat().st_size

            if file_size > max_characters_needed * 2:
                # Smart loading
                buffer_multiplier = 2.0
                chars_to_load = min(
                    int(max_characters_needed * buffer_multiplier), file_size
                )

                with open(data_path, "r", encoding="utf-8") as f:
                    max_start = max(0, file_size - chars_to_load)
                    start_pos = random.randint(0, max_start) if max_start > 0 else 0
                    f.seek(start_pos)
                    if start_pos > 0:
                        f.readline()
                    text = f.read(chars_to_load)

                print(
                    f"Smart loading: loaded {len(text):,} chars from {file_size:,}-char file"
                )
            else:
                text = Path(data_path).read_text(encoding="utf-8")
                print(f"Standard loading: {len(text):,} characters")
        else:
            text = Path(data_path).read_text(encoding="utf-8")
            print(f"Full loading (no limit): {len(text):,} characters")

        # Tokenize in 2000-char chunks (EXACTLY LIKE StreamingTextDataset/Transformer)
        print(f"Tokenizing {len(text):,} characters (using Transformer-style chunking)...")
        chunk_size = 2000  # SAME AS StreamingTextDataset
        text_chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        tokens = []
        for chunk in text_chunks:
            chunk_tokens = tokenizer(
                chunk,
                truncation=True,      # SAME AS StreamingTextDataset
                max_length=1024       # SAME AS StreamingTextDataset
                # add_special_tokens defaults to True (SAME AS StreamingTextDataset)
            )["input_ids"]
            tokens.extend(chunk_tokens)

        print(f"Tokenization complete: {len(tokens):,} tokens (Transformer-compatible)")

    # Check if separate validation dataset is specified
    if val_data_path:
        print(f"\nLoading separate validation dataset from: {val_data_path}")
        # Handle relative paths (for LSTM running from subdirectory)
        val_path_obj = Path(val_data_path)
        if not val_path_obj.exists():
            parent_relative = Path("..") / val_data_path
            if parent_relative.exists():
                val_data_path = str(parent_relative)
                print(f"  (adjusted path: {val_data_path})")

        # Check for .npy file for validation dataset too
        val_npy_path = Path(val_data_path).with_suffix('.npy')
        if not val_npy_path.exists():
            parent_val_npy = Path("..") / val_data_path
            parent_val_npy = parent_val_npy.with_suffix('.npy')
            if parent_val_npy.exists():
                val_npy_path = parent_val_npy

        if val_npy_path.exists():
            # Load from memory-mapped file
            print(f"  🚀 Using memory-mapped validation file: {val_npy_path}")
            val_tokens_mmap = np.load(str(val_npy_path), mmap_mode='r')
            val_tokens = val_tokens_mmap[:].tolist()  # Convert to list
            print(f"  Validation: {len(val_tokens):,} tokens (from memory-mapped file)")
        else:
            # Load and tokenize validation file (EXACTLY LIKE Transformer)
            val_text = Path(val_data_path).read_text(encoding="utf-8")
            print(f"  Tokenizing {len(val_text):,} characters (Transformer-style)...")

            # Use 2000-char chunks (SAME AS StreamingTextDataset)
            chunk_size = 2000
            text_chunks = [val_text[i : i + chunk_size] for i in range(0, len(val_text), chunk_size)]

            val_tokens = []
            for chunk in text_chunks:
                chunk_tokens = tokenizer(
                    chunk,
                    truncation=True,
                    max_length=1024
                )["input_ids"]
                val_tokens.extend(chunk_tokens)

            print(f"  Validation: {len(val_tokens):,} tokens (from separate file, Transformer-compatible)")

        # For train/test split, only split the training data
        train_tokens = tokens  # All tokens from training file
        test_tokens = []  # No test set when using separate validation
    else:
        val_tokens = None  # Will be split from training data below

    # Split into train/val/test (only if val_tokens not already set from separate file)
    if val_tokens is None:
        n = len(tokens)

        # Use exact token counts (required)
        n_train = int(max_tokens_training)
        n_val = int(fixed_val_tokens)

        if n_train + n_val > n:
            raise ValueError(
                f"Not enough tokenized data: need {n_train + n_val:,} tokens "
                f"({n_train:,} training + {n_val:,} validation) "
                f"but only have {n:,} tokens available"
            )

        # Training: first n_train tokens
        train_tokens = tokens[:n_train]

        # Validation: LAST n_val tokens (from END of file, like old 90/10 split)
        val_tokens = tokens[-n_val:]

        # Test: tokens between training and validation (middle section)
        test_tokens = tokens[n_train:-n_val] if n_val > 0 else tokens[n_train:]

        print(f"Dataset split with fixed token counts:")
        print(f"  Training: {len(train_tokens):,} tokens (from start)")
        print(f"  Validation: {len(val_tokens):,} tokens (from END, like old 90/10)")
        print(f"  Test: {len(test_tokens):,} tokens (middle section)")

    return train_tokens, val_tokens, test_tokens, tokenizer
