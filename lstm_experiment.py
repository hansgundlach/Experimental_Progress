import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import wandb
import os
from collections import Counter
from typing import Dict, List, Tuple

# Configuration
CONFIG = {
    "data_path": "Datasets/wikitext.txt",
    "vocab_size": 10000,  # Top K most frequent words
    "sequence_length": 128,
    "batch_size": 32,
    "hidden_size": 512,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "num_epochs": 20,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb_project": "lstm-wikitext",
    "wandb_offline": True,
    "print_every": 100,  # Print loss every N batches
}


class WikiTextDataset(Dataset):
    def __init__(self, text_data: List[int], sequence_length: int):
        self.data = text_data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx : idx + self.sequence_length], dtype=torch.long),
            torch.tensor(
                self.data[idx + 1 : idx + self.sequence_length + 1], dtype=torch.long
            ),
        )


class TextPreprocessor:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"

    def build_vocab(self, text: str) -> None:
        # Tokenize and count words
        words = text.lower().split()
        word_counts = Counter(words)

        # Get most frequent words
        most_common = word_counts.most_common(self.vocab_size - 2)  # -2 for UNK and PAD

        # Build vocabulary
        self.word_to_idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_word = {0: self.pad_token, 1: self.unk_token}

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        print(f"Built vocabulary with {len(self.word_to_idx)} words")

    def text_to_indices(self, text: str) -> List[int]:
        words = text.lower().split()
        return [self.word_to_idx.get(word, 1) for word in words]  # 1 is UNK token


class LSTMLanguageModel(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super(LSTMLanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.linear(lstm_out)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
        )


class FLOPCounter:
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.total_flops = 0

    def count_lstm_flops(self, batch_size: int, sequence_length: int) -> int:
        """Count FLOPs for LSTM layer"""
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]

        # LSTM has 4 gates (input, forget, cell, output)
        # Each gate: input_size * hidden_size + hidden_size * hidden_size + hidden_size (bias)
        # For each timestep and layer

        flops_per_gate = (
            (hidden_size * hidden_size) + (hidden_size * hidden_size) + hidden_size
        )
        flops_per_timestep = 4 * flops_per_gate  # 4 gates
        flops_per_layer = flops_per_timestep * sequence_length
        total_lstm_flops = flops_per_layer * num_layers * batch_size

        return total_lstm_flops

    def count_embedding_flops(self, batch_size: int, sequence_length: int) -> int:
        """Count FLOPs for embedding layer (mainly memory access, minimal compute)"""
        return batch_size * sequence_length  # Simplified

    def count_linear_flops(self, batch_size: int, sequence_length: int) -> int:
        """Count FLOPs for final linear layer"""
        hidden_size = self.config["hidden_size"]
        vocab_size = self.config["vocab_size"]
        return batch_size * sequence_length * hidden_size * vocab_size

    def count_forward_flops(self, batch_size: int, sequence_length: int) -> int:
        """Count total FLOPs for one forward pass"""
        embedding_flops = self.count_embedding_flops(batch_size, sequence_length)
        lstm_flops = self.count_lstm_flops(batch_size, sequence_length)
        linear_flops = self.count_linear_flops(batch_size, sequence_length)

        return embedding_flops + lstm_flops + linear_flops

    def add_batch_flops(self, batch_size: int, sequence_length: int):
        """Add FLOPs for one batch (forward + backward ≈ 3x forward)"""
        forward_flops = self.count_forward_flops(batch_size, sequence_length)
        total_batch_flops = forward_flops * 3  # Approximate backward pass
        self.total_flops += total_batch_flops
        return total_batch_flops


def load_and_preprocess_data(
    config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, TextPreprocessor]:
    """Load and preprocess the WikiText data"""
    print("Loading and preprocessing data...")

    # Load text data
    with open(config["data_path"], "r", encoding="utf-8") as f:
        text = f.read()

    # Initialize preprocessor
    preprocessor = TextPreprocessor(config["vocab_size"])
    preprocessor.build_vocab(text)

    # Convert text to indices
    indices = preprocessor.text_to_indices(text)

    # Split data
    total_len = len(indices)
    train_len = int(total_len * config["train_split"])
    val_len = int(total_len * config["val_split"])

    train_data = indices[:train_len]
    val_data = indices[train_len : train_len + val_len]
    test_data = indices[train_len + val_len :]

    # Create datasets
    train_dataset = WikiTextDataset(train_data, config["sequence_length"])
    val_dataset = WikiTextDataset(val_data, config["sequence_length"])
    test_dataset = WikiTextDataset(test_data, config["sequence_length"])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    print(
        f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
    )

    return train_loader, val_loader, test_loader, preprocessor


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: str
) -> float:
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            hidden = model.init_hidden(batch_size, device)
            outputs, _ = model(inputs, hidden)

            # Reshape for loss calculation
            outputs = outputs.view(-1, model.vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


def train_model(config: Dict):
    """Main training function"""
    # Initialize wandb
    if config["wandb_offline"]:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(project=config["wandb_project"], config=config)

    # Set device
    device = torch.device(config["device"])
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, preprocessor = load_and_preprocess_data(
        config
    )

    # Initialize model
    model = LSTMLanguageModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    # Initialize FLOP counter
    flop_counter = FLOPCounter(model, config)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_loader)} batches per epoch")

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size, sequence_length = inputs.size()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            hidden = model.init_hidden(batch_size, device)
            outputs, _ = model(inputs, hidden)

            # Reshape for loss calculation
            outputs = outputs.view(-1, config["vocab_size"])
            targets = targets.view(-1)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update FLOP counter
            batch_flops = flop_counter.add_batch_flops(batch_size, sequence_length)

            # Update epoch statistics
            epoch_loss += loss.item()
            epoch_batches += 1

            # Print batch statistics
            if batch_idx % config["print_every"] == 0:
                print(
                    f"Epoch {epoch+1}/{config['num_epochs']}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Total FLOPs: {flop_counter.total_flops:,}"
                )

        # Calculate epoch statistics
        avg_train_loss = epoch_loss / epoch_batches
        epoch_time = time.time() - epoch_start_time

        # Evaluate on validation set
        val_loss = evaluate_model(model, val_loader, criterion, device)

        # Calculate perplexity
        train_perplexity = np.exp(avg_train_loss)
        val_perplexity = np.exp(val_loss)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} Summary:")
        print(
            f"  Train Loss: {avg_train_loss:.4f} (Perplexity: {train_perplexity:.2f})"
        )
        print(f"  Val Loss: {val_loss:.4f} (Perplexity: {val_perplexity:.2f})")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print(f"  Total FLOPs: {flop_counter.total_flops:,}")
        print(
            f"  FLOPs per second: {flop_counter.total_flops/sum([time.time() - epoch_start_time for _ in range(epoch+1)]):.2e}"
        )
        print("-" * 60)

        # Log to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "train_perplexity": train_perplexity,
                "val_perplexity": val_perplexity,
                "total_flops": flop_counter.total_flops,
                "epoch_time": epoch_time,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss = evaluate_model(model, test_loader, criterion, device)
    test_perplexity = np.exp(test_loss)
    print(f"Test Loss: {test_loss:.4f} (Perplexity: {test_perplexity:.2f})")

    wandb.log(
        {
            "test_loss": test_loss,
            "test_perplexity": test_perplexity,
            "final_total_flops": flop_counter.total_flops,
        }
    )

    wandb.finish()

    return model, flop_counter.total_flops


def benchmark_model(model: nn.Module, config: Dict):
    """Benchmark model inference speed"""
    device = torch.device(config["device"])
    model.eval()

    # Create dummy input
    dummy_input = torch.randint(
        0, config["vocab_size"], (config["batch_size"], config["sequence_length"])
    ).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            hidden = model.init_hidden(config["batch_size"], device)
            _ = model(dummy_input, hidden)

    # Benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(100):
            hidden = model.init_hidden(config["batch_size"], device)
            _ = model(dummy_input, hidden)

    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / 100
    throughput = config["batch_size"] / avg_inference_time

    print(f"\nBenchmark Results:")
    print(f"  Average inference time: {avg_inference_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} samples/second")

    return avg_inference_time, throughput


if __name__ == "__main__":
    print("Starting LSTM training with FLOP counting...")
    print(f"Configuration: {CONFIG}")

    # Train model
    trained_model, total_flops = train_model(CONFIG)

    # Benchmark model
    inference_time, throughput = benchmark_model(trained_model, CONFIG)

    print(f"\nTraining completed!")
    print(f"Total FLOPs used in training: {total_flops:,}")
    print(
        f"Final benchmark - Inference time: {inference_time*1000:.2f}ms, Throughput: {throughput:.2f} samples/s"
    )
