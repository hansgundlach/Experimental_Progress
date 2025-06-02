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

# Configuration
CONFIG = {
    "data_path": "../Datasets/wikitext.txt",
    "tokenizer_path": "../gpt2_tokenizer",
    "max_characters": 2e5,  # Maximum number of characters to use from dataset
    "sequence_length": 128,
    "batch_size": 32,
    "hidden_size": 512,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "lr_schedule": "constant",
    "step_size": 10,
    "gamma": 0.1,
    "num_epochs": 20,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb_project": "lstm-wikitext",
    "wandb_offline": True,
    "print_every": 100,  # Print loss every N batches
    # Gradient clipping settings
    "use_gradient_clipping": True,
    "gradient_clip_val": 1.0,
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
    def __init__(self, model: nn.Module, config: Dict, tokenizer: GPT2Tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.total_flops = 0
        self.flops_per_batch = None  # Will be set after profiling one batch
        self.profiled = False

    def profile_one_batch(self, inputs, targets, hidden, optimizer, criterion):
        """Profile one batch to get accurate FLOP count, then use for extrapolation"""
        print("  [PROFILER] Profiling one batch to determine FLOPs per batch...")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with record_function("forward_pass"):
                outputs, _ = self.model(inputs, hidden)
                outputs = outputs.view(-1, self.model.vocab_size)
                targets_reshaped = targets.view(-1)
                loss = criterion(outputs, targets_reshaped)

            with record_function("backward_pass"):
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                if self.config.get("use_gradient_clipping", False):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("gradient_clip_val", 1.0),
                    )
                optimizer.step()

        # Extract FLOP count from profiler
        total_flops = 0
        flop_events = 0
        for evt in prof.events():
            if evt.flops is not None and evt.flops > 0:
                total_flops += evt.flops
                flop_events += 1

        if flop_events > 0:
            self.flops_per_batch = total_flops
            print(
                f"  [PROFILER] Found {flop_events} FLOP events, FLOPs per batch: {self.flops_per_batch:.2e}"
            )
        else:
            # Fallback to manual calculation
            batch_size, sequence_length = inputs.size()
            self.flops_per_batch = (
                self.count_forward_flops_manual(batch_size, sequence_length) * 3
            )
            print(
                f"  [PROFILER] No FLOP events found, using manual estimate: {self.flops_per_batch:.2e}"
            )

        self.profiled = True
        self.total_flops += self.flops_per_batch
        return loss

    def count_forward_flops_manual(self, batch_size: int, sequence_length: int) -> int:
        """Count total FLOPs for one forward pass (manual approximation)"""
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]

        # LSTM FLOPs
        flops_per_gate = (
            (hidden_size * hidden_size) + (hidden_size * hidden_size) + hidden_size
        )
        flops_per_timestep = 4 * flops_per_gate  # 4 gates
        flops_per_layer = flops_per_timestep * sequence_length
        lstm_flops = flops_per_layer * num_layers * batch_size

        # Embedding FLOPs (minimal)
        embedding_flops = batch_size * sequence_length

        # Linear layer FLOPs
        linear_flops = batch_size * sequence_length * hidden_size * self.vocab_size

        return embedding_flops + lstm_flops + linear_flops

    def add_batch_flops(self):
        """Add FLOPs for one batch using the profiled value"""
        if self.flops_per_batch is not None:
            self.total_flops += self.flops_per_batch
        return self.flops_per_batch


def load_and_preprocess_data(
    config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, TextPreprocessor]:
    """Load and preprocess the WikiText data"""
    print("Loading and preprocessing data...")

    # Load text data
    with open(config["data_path"], "r", encoding="utf-8") as f:
        text = f.read()

    # Limit data size if specified
    if config["max_characters"] and len(text) > config["max_characters"]:
        text = text[: int(config["max_characters"])]
        print(
            f"Limited dataset to {config['max_characters']:.0e} characters (originally {len(text)} characters)"
        )
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
        f"Data split: Train={len(train_data)} tokens, Val={len(val_data)} tokens, Test={len(test_data)} tokens"
    )
    print(
        f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}"
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

    # Initialize model using GPT-2 tokenizer vocab size
    model = LSTMLanguageModel(
        vocab_size=preprocessor.vocab_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    # Initialize FLOP counter
    flop_counter = FLOPCounter(model, config, preprocessor.tokenizer)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Set up LR scheduler
    if config.get("lr_schedule") == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.1),
        )
    elif config.get("lr_schedule") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["num_epochs"],
            eta_min=config.get("min_lr", 0),
        )
    else:
        scheduler = None

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_loader)} batches per epoch")
    print(f"FLOP counting method: Profile first batch, then extrapolate")

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size, sequence_length = inputs.size()

            # Initialize hidden state
            hidden = model.init_hidden(batch_size, device)

            # Profile first batch, then extrapolate for the rest
            if not flop_counter.profiled:
                # Profile the first batch to get accurate FLOP count
                loss = flop_counter.profile_one_batch(
                    inputs, targets, hidden, optimizer, criterion
                )
            else:
                # Use profiled FLOP count and do normal training
                flop_counter.add_batch_flops()

                # Normal training step
                optimizer.zero_grad()

                # Forward pass
                outputs, _ = model(inputs, hidden)
                outputs = outputs.view(-1, model.vocab_size)
                targets_reshaped = targets.view(-1)
                loss = criterion(outputs, targets_reshaped)

                # Backward pass
                loss.backward()
                # Gradient clipping
                if config.get("use_gradient_clipping", False):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.get("gradient_clip_val", 1.0)
                    )
                optimizer.step()

            # Update epoch statistics
            epoch_loss += loss.item()
            epoch_batches += 1

            # Print batch statistics
            if batch_idx % config["print_every"] == 0:
                method = "Profiled" if flop_counter.profiled else "Profiling"
                print(
                    f"Epoch {epoch+1}/{config['num_epochs']}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Total FLOPs: {flop_counter.total_flops:.2e} [{method}]"
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
            f"  Training Loss: {avg_train_loss:.4f} (Perplexity: {train_perplexity:.2f})"
        )
        print(f"  Validation Loss: {val_loss:.4f} (Perplexity: {val_perplexity:.2f})")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print(f"  Total FLOPs: {flop_counter.total_flops:.2e}")
        print(f"  FLOPs per second: {flop_counter.total_flops/epoch_time:.2e}")
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

        # Step scheduler once per epoch
        if scheduler is not None:
            scheduler.step()

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
    print("Starting LSTM training with FLOP counting...")
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
