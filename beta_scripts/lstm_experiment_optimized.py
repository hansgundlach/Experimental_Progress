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
from transformers import GPT2Tokenizer
from torch.profiler import profile, record_function, ProfilerActivity
import gc

# Configuration with optimizations
CONFIG = {
    "data_path": "Datasets/wikitext.txt",
    "tokenizer_path": "gpt2_tokenizer",
    "max_characters": 2e5,  # Maximum number of characters to use from dataset
    "sequence_length": 128,
    "batch_size": 64,  # Increased from 32
    "hidden_size": 512,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "num_epochs": 20,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb_project": "lstm-wikitext-optimized",
    "wandb_offline": True,
    "print_every": 100,  # Print loss every N batches
    "num_workers": 4,  # For DataLoader
    "pin_memory": True,  # For DataLoader
    "gradient_clip_val": 1.0,  # Gradient clipping
    "compile_model": True,  # Use torch.compile
    "mixed_precision": True,  # Use automatic mixed precision
    "accumulate_grad_batches": 1,  # Gradient accumulation
}


class OptimizedWikiTextDataset(Dataset):
    def __init__(self, text_data: List[int], sequence_length: int):
        # Convert to tensor once for efficiency
        self.data = torch.tensor(text_data, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        return (
            self.data[idx : idx + self.sequence_length],
            self.data[idx + 1 : idx + self.sequence_length + 1],
        )


class TextPreprocessor:
    def __init__(self, tokenizer_path: str):
        # Load tokenizer from local directory (offline mode)
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,  # Force offline mode
            use_fast=False,  # Use slow tokenizer to avoid potential issues
        )
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer)
        print(
            f"Loaded GPT-2 tokenizer from local path with vocabulary size: {self.vocab_size}"
        )

    def text_to_indices(self, text: str) -> List[int]:
        # Tokenize the text using GPT2 tokenizer
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens


class OptimizedLSTMLanguageModel(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super(OptimizedLSTMLanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability"""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.linear(lstm_out)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
        )


class SimpleFLOPCounter:
    """Simplified FLOP counter without profiling overhead"""

    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.total_flops = 0
        self.flops_per_batch = self._estimate_flops_per_batch()

    def _estimate_flops_per_batch(self) -> int:
        """Estimate FLOPs per batch using theoretical calculations"""
        batch_size = self.config["batch_size"]
        sequence_length = self.config["sequence_length"]
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]
        vocab_size = self.model.vocab_size

        # LSTM FLOPs (approximate)
        lstm_flops = (
            batch_size * sequence_length * num_layers * 8 * hidden_size * hidden_size
        )

        # Linear layer FLOPs
        linear_flops = batch_size * sequence_length * hidden_size * vocab_size

        # Total FLOPs for forward + backward (multiply by 3 for backward pass approximation)
        total_flops = (lstm_flops + linear_flops) * 3

        print(f"Estimated FLOPs per batch: {total_flops:.2e}")
        return total_flops

    def add_batch_flops(self):
        """Add FLOPs for one batch"""
        self.total_flops += self.flops_per_batch
        return self.flops_per_batch


def load_and_preprocess_data(
    config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, TextPreprocessor]:
    """Load and preprocess the WikiText data with optimized DataLoaders"""
    print("Loading and preprocessing data...")

    # Load text data
    with open(config["data_path"], "r", encoding="utf-8") as f:
        text = f.read()

    # Limit data size if specified
    if config["max_characters"] and len(text) > config["max_characters"]:
        text = text[: int(config["max_characters"])]
        print(f"Limited dataset to {config['max_characters']:.0e} characters")
    else:
        print(f"Using full dataset: {len(text)} characters")

    # Initialize preprocessor with GPT-2 tokenizer
    preprocessor = TextPreprocessor(config["tokenizer_path"])

    # Convert text to indices using GPT-2 tokenizer
    indices = preprocessor.text_to_indices(text)

    # Split data
    total_len = len(indices)
    train_len = int(total_len * config["train_split"])
    val_len = int(total_len * config["val_split"])

    train_data = indices[:train_len]
    val_data = indices[train_len : train_len + val_len]
    test_data = indices[train_len + val_len :]

    # Create optimized datasets
    train_dataset = OptimizedWikiTextDataset(train_data, config["sequence_length"])
    val_dataset = OptimizedWikiTextDataset(val_data, config["sequence_length"])
    test_dataset = OptimizedWikiTextDataset(test_data, config["sequence_length"])

    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        drop_last=True,  # For consistent batch sizes
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        drop_last=True,
    )

    print(
        f"Data split: Train={len(train_data)} tokens, Val={len(val_data)} tokens, Test={len(test_data)} tokens"
    )
    print(
        f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}"
    )

    return train_loader, val_loader, test_loader, preprocessor


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    scaler=None,
) -> float:
    """Evaluate model on validation/test set with mixed precision support"""
    model.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )
            batch_size = inputs.size(0)

            hidden = model.init_hidden(batch_size, device)

            if scaler is not None:
                with torch.autocast(device_type="cuda"):
                    outputs, _ = model(inputs, hidden)
                    outputs = outputs.view(-1, model.vocab_size)
                    targets = targets.view(-1)
                    loss = criterion(outputs, targets)
            else:
                outputs, _ = model(inputs, hidden)
                outputs = outputs.view(-1, model.vocab_size)
                targets = targets.view(-1)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


def train_model(config: Dict):
    """Main training function with optimizations"""
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

    # Initialize model using GPT-2 tokenizer vocab size
    model = OptimizedLSTMLanguageModel(
        vocab_size=preprocessor.vocab_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    # Compile model for optimization (PyTorch 2.0+)
    if config["compile_model"] and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Failed to compile model: {e}")

    # Initialize simplified FLOP counter
    flop_counter = SimpleFLOPCounter(model, config)

    # Loss and optimizer with optimizations
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )

    # Mixed precision scaler
    scaler = (
        torch.cuda.amp.GradScaler()
        if config["mixed_precision"] and device.type == "cuda"
        else None
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_loader)} batches per epoch")
    print(
        f"Optimizations enabled: Compile={config['compile_model']}, MixedPrecision={config['mixed_precision']}"
    )

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )
            batch_size, sequence_length = inputs.size()

            # Initialize hidden state
            hidden = model.init_hidden(batch_size, device)

            # Mixed precision training
            if scaler is not None:
                with torch.autocast(device_type="cuda"):
                    outputs, _ = model(inputs, hidden)
                    outputs = outputs.view(-1, model.vocab_size)
                    targets_reshaped = targets.view(-1)
                    loss = criterion(outputs, targets_reshaped)

                    # Scale loss for gradient accumulation
                    loss = loss / config["accumulate_grad_batches"]

                # Backward pass with scaling
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % config["accumulate_grad_batches"] == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["gradient_clip_val"]
                    )

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Regular training without mixed precision
                outputs, _ = model(inputs, hidden)
                outputs = outputs.view(-1, model.vocab_size)
                targets_reshaped = targets.view(-1)
                loss = criterion(outputs, targets_reshaped)

                # Scale loss for gradient accumulation
                loss = loss / config["accumulate_grad_batches"]

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % config["accumulate_grad_batches"] == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["gradient_clip_val"]
                    )

                    optimizer.step()
                    optimizer.zero_grad()

            # Add FLOPs for this batch
            flop_counter.add_batch_flops()

            # Update epoch statistics
            epoch_loss += (
                loss.item() * config["accumulate_grad_batches"]
            )  # Unscale for logging
            epoch_batches += 1

            # Print batch statistics
            if batch_idx % config["print_every"] == 0:
                print(
                    f"Epoch {epoch+1}/{config['num_epochs']}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item() * config['accumulate_grad_batches']:.4f}, "
                    f"Total FLOPs: {flop_counter.total_flops:.2e}"
                )

        # Step scheduler
        scheduler.step()

        # Calculate epoch statistics
        avg_train_loss = epoch_loss / epoch_batches
        epoch_time = time.time() - epoch_start_time

        # Evaluate on validation set
        val_loss = evaluate_model(model, val_loader, criterion, str(device), scaler)

        # Calculate perplexity
        train_perplexity = np.exp(avg_train_loss)
        val_perplexity = np.exp(val_loss)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} Summary:")
        print(
            f"  Training Loss: {avg_train_loss:.4f} (Perplexity: {train_perplexity:.2f})"
        )
        print(f"  Validation Loss: {val_loss:.4f} (Perplexity: {val_perplexity:.2f})")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print(f"  Total FLOPs: {flop_counter.total_flops:.2e}")
        print(f"  FLOPs per second: {flop_counter.total_flops/epoch_time:.2e}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
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
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        # Memory cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss = evaluate_model(model, test_loader, criterion, str(device), scaler)
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


def benchmark_model(model: nn.Module, config: Dict, tokenizer: GPT2Tokenizer):
    """Benchmark model inference speed"""
    device = torch.device(config["device"])
    model.eval()

    # Create dummy input using tokenizer vocab size
    dummy_input = torch.randint(
        0, len(tokenizer), (config["batch_size"], config["sequence_length"])
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
    print("Starting optimized LSTM training with FLOP counting...")
    print(f"Configuration: {CONFIG}")

    # Train model
    trained_model, total_flops = train_model(CONFIG)

    # Get the tokenizer for benchmarking (offline mode)
    tokenizer = GPT2Tokenizer.from_pretrained(
        CONFIG["tokenizer_path"], local_files_only=True, use_fast=False
    )

    # Benchmark model
    inference_time, throughput = benchmark_model(trained_model, CONFIG, tokenizer)

    print(f"\nTraining completed!")
    print(f"Total FLOPs used in training: {total_flops:.2e}")
    print(
        f"Final benchmark - Inference time: {inference_time*1000:.2f}ms, Throughput: {throughput:.2f} samples/s"
    )
