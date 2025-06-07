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
import multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import math
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Configuration
CONFIG = {
    "data_path": "../Datasets/wikitext.txt",
    "tokenizer_path": "../gpt2_tokenizer",
    "max_characters": 5 * 1e7,  # Maximum number of characters to use from dataset
    "sequence_length": 128,
    "batch_size": 256,  # Keep physical batch size small
    "hidden_size": 16,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001 * math.sqrt(4),  # Scale by sqrt of accumulation steps
    "lr_schedule": "cosine",
    "step_size": 10,
    "gamma": 0.1,
    "num_epochs": 5,
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
    # NEW: Data loading optimization settings
    "num_workers": "auto",  # Will be set automatically based on CPU cores
    "pin_memory": True,  # Faster GPU memory transfer
    "persistent_workers": True,  # Keep data loading workers alive between epochs
    "prefetch_factor": 4,  # Number of batches to prefetch per worker
    # NEW: Mixed precision settings
    "use_amp": False,  # Enable Automatic Mixed Precision
    "amp_opt_level": "O1",  # Not used with native AMP, but kept for reference
    # NEW: Gradient accumulation settings
    "gradient_accumulation_steps": 2,  # Simulate 4x larger batch size (32*4 = 128)# For tracking only - computed from batch_size * gradient_accumulation_steps
}

# old large 5-6M param config:
# CONFIG = {
#     "data_path": "../Datasets/wikitext.txt",
#     "tokenizer_path": "../gpt2_tokenizer",
#     "max_characters": 3 * 1e8,  # Maximum number of characters to use from dataset
#     "sequence_length": 128,
#     "batch_size": 32,  # Keep physical batch size small
#     "hidden_size": 64,
#     "num_layers": 2,
#     "dropout": 0.2,
#     "learning_rate": 0.001 * math.sqrt(4),  # Scale by sqrt of accumulation steps
#     "lr_schedule": "cosine",
#     "step_size": 10,
#     "gamma": 0.1,
#     "num_epochs": 4,
#     "train_split": 0.8,
#     "val_split": 0.1,
#     "test_split": 0.1,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "wandb_project": "lstm-wikitext",
#     "wandb_offline": True,
#     "print_every": 100,  # Print loss every N batches
#     # Gradient clipping settings
#     "use_gradient_clipping": True,
#     "gradient_clip_val": 1.0,
#     # NEW: Data loading optimization settings
#     "num_workers": "auto",  # Will be set automatically based on CPU cores
#     "pin_memory": True,  # Faster GPU memory transfer
#     "persistent_workers": True,  # Keep data loading workers alive between epochs
#     "prefetch_factor": 4,  # Number of batches to prefetch per worker
#     # NEW: Mixed precision settings
#     "use_amp": True,  # Enable Automatic Mixed Precision
#     "amp_opt_level": "O1",  # Not used with native AMP, but kept for reference
#     # NEW: Gradient accumulation settings
#     "gradient_accumulation_steps": 4,  # Simulate 4x larger batch size (32*4 = 128)
#     "effective_batch_size": 128,  # For tracking only - computed from batch_size * gradient_accumulation_steps
# }


cudnn.benchmark = True


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

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.flops_per_batch = None
        self.time_per_batch = None
        self.profiled = False

    def get_model_vocab_size(self):
        # Handle wrapped models (DataParallel or DistributedDataParallel)
        if hasattr(self.model, "module"):
            return self.model.module.vocab_size
        else:
            return self.model.vocab_size

    def profile_one_batch(
        self, inputs, targets, hidden, optimizer, criterion, scaler=None
    ):
        """Profile one batch with mixed precision support"""
        print(
            "  [PROFILER] Profiling one batch to determine FLOPs and timing per batch..."
        )

        # Move inputs and targets to the correct device
        device = next(self.model.parameters()).device
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Ensure data transfer is complete
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Start timing the entire batch
        batch_start_time = time.time()

        vocab_size = self.get_model_vocab_size()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with record_function("forward_pass"):
                # NEW: Use autocast for mixed precision
                if scaler is not None:
                    with autocast():
                        outputs, _ = self.model(inputs, hidden)
                        outputs = outputs.view(-1, vocab_size)
                        targets_reshaped = targets.view(-1)
                        loss = criterion(outputs, targets_reshaped)
                else:
                    outputs, _ = self.model(inputs, hidden)
                    outputs = outputs.view(-1, vocab_size)
                    targets_reshaped = targets.view(-1)
                    loss = criterion(outputs, targets_reshaped)

            with record_function("backward_pass"):
                optimizer.zero_grad()

                # NEW: Mixed precision backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # Gradient clipping with scaler
                    if self.config.get("use_gradient_clipping", False):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.get("gradient_clip_val", 1.0),
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # Regular gradient clipping
                    if self.config.get("use_gradient_clipping", False):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.get("gradient_clip_val", 1.0),
                        )
                    optimizer.step()

        # Synchronize CUDA operations before measuring time
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Calculate batch time
        batch_end_time = time.time()
        self.time_per_batch = batch_end_time - batch_start_time

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

        # Report timing information
        amp_status = "with AMP" if scaler is not None else "without AMP"
        print(
            f"  [PROFILER] Time per batch ({amp_status}): {self.time_per_batch:.4f} seconds"
        )
        print(
            f"  [PROFILER] FLOPs per second: {self.flops_per_batch / self.time_per_batch:.2e}"
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

    def get_timing_info(self):
        """Get timing information from profiling"""
        return {
            "time_per_batch": self.time_per_batch,
            "flops_per_batch": self.flops_per_batch,
            "flops_per_second": (
                self.flops_per_batch / self.time_per_batch
                if self.time_per_batch
                else None
            ),
        }


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

    # CONSERVATIVE: Determine optimal number of workers for Supercloud
    if config.get("num_workers") == "auto":
        # Be more conservative on shared systems like Supercloud
        num_workers = min(mp.cpu_count() // 4, 4)  # Changed: more conservative
        # On systems with issues, fall back to 0
        if mp.cpu_count() <= 4:
            num_workers = 0
    else:
        num_workers = config.get("num_workers", 0)

    # Check if we're on CUDA for pin_memory optimization
    pin_memory = config.get("pin_memory", False) and torch.cuda.is_available()

    # Only use persistent_workers if num_workers > 0 and not on potentially problematic systems
    persistent_workers = config.get("persistent_workers", False) and num_workers > 0

    # Prefetch factor (only relevant when num_workers > 0)
    prefetch_factor = config.get("prefetch_factor", 2) if num_workers > 0 else 2

    print(f"DataLoader optimization settings:")
    print(f"  num_workers: {num_workers}")
    print(f"  pin_memory: {pin_memory}")
    print(f"  persistent_workers: {persistent_workers}")
    print(f"  prefetch_factor: {prefetch_factor}")

    # Create optimized data loaders (use sampler under DDP)
    if dist.is_initialized():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
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
    device: torch.device,
) -> float:
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    total_batches = 0

    # Helper function to handle DataParallel or DDP wrapping
    def get_hidden(batch_size):
        if isinstance(model, nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ):
            return model.module.init_hidden(batch_size, device)
        else:
            return model.init_hidden(batch_size, device)

    # Helper function to get vocab size
    def get_vocab_size():
        if isinstance(model, nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ):
            return model.module.vocab_size
        else:
            return model.vocab_size

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            hidden = get_hidden(batch_size)
            outputs, _ = model(inputs, hidden)

            # Reshape for loss calculation
            outputs = outputs.view(-1, get_vocab_size())
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


def evaluate_model_amp(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    """Evaluate model with mixed precision support"""
    model.eval()
    total_loss = 0
    total_batches = 0

    # Helper function to handle DataParallel or DDP wrapping
    def get_hidden(batch_size):
        if isinstance(model, nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ):
            return model.module.init_hidden(batch_size, device)
        else:
            return model.init_hidden(batch_size, device)

    # Helper function to get vocab size
    def get_vocab_size():
        if isinstance(model, nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ):
            return model.module.vocab_size
        else:
            return model.vocab_size

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            hidden = get_hidden(batch_size)

            if use_amp:
                with autocast():
                    outputs, _ = model(inputs, hidden)
                    outputs = outputs.view(-1, get_vocab_size())
                    targets = targets.view(-1)
                    loss = criterion(outputs, targets)
            else:
                outputs, _ = model(inputs, hidden)
                outputs = outputs.view(-1, get_vocab_size())
                targets = targets.view(-1)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


def train_model(config: Dict, local_rank=0):
    """Main training function with mixed precision and gradient accumulation"""
    # Initialize wandb
    if config["wandb_offline"]:
        os.environ["WANDB_MODE"] = "offline"

    # Calculate effective batch size
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    effective_batch_size = config["batch_size"] * gradient_accumulation_steps
    config["effective_batch_size"] = effective_batch_size  # Store for reporting

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
        vocab_size=preprocessor.vocab_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    # Use DataParallel if multiple GPUs are available and NOT using DDP
    if torch.cuda.device_count() > 1 and not dist.is_initialized():
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    elif dist.is_initialized():
        print(f"Using DistributedDataParallel for training")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # Helper function to handle DataParallel or DDP wrapping
    def init_hidden(batch_size):
        if isinstance(model, nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ):
            return model.module.init_hidden(batch_size, device)
        else:
            return model.init_hidden(batch_size, device)

    # Initialize FLOP counter
    flop_counter = FLOPCounter(model, config, preprocessor.tokenizer)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Initialize mixed precision scaler
    use_amp = config.get("use_amp", False) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    print(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    print(
        f"Gradient Accumulation: {gradient_accumulation_steps} steps (Effective batch size: {effective_batch_size})"
    )

    # Set up LR scheduler
    if config.get("lr_schedule") == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.1),
        )
    elif config.get("lr_schedule") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["num_epochs"], eta_min=config.get("min_lr", 0)
        )
    else:
        scheduler = None

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_loader)} batches per epoch")

    # Add this function inside train_model
    def get_vocab_size():
        if isinstance(model, nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ):
            return model.module.vocab_size
        else:
            return model.vocab_size

    # Training loop
    for epoch in range(config["num_epochs"]):
        # reshuffle for DDP
        if dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_batches = 0
        optimizer.zero_grad()  # Zero gradients at the start of epoch

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_size, sequence_length = inputs.size()

            # Profile only the very first batch of training
            if not flop_counter.profiled:
                hidden = init_hidden(batch_size)
                loss = flop_counter.profile_one_batch(
                    inputs, targets, hidden, optimizer, criterion, scaler
                )
                # Re-zero gradients after profiling
                optimizer.zero_grad()
                # Update epoch statistics for profiled batch
                epoch_loss += loss.item()
                epoch_batches += 1
                continue

            # Move data to device
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )
            hidden = init_hidden(batch_size)

            # Track FLOP count
            flop_counter.add_batch_flops()

            # Forward pass with mixed precision
            if use_amp:
                with autocast():
                    outputs, _ = model(inputs, hidden)
                    outputs = outputs.view(-1, get_vocab_size())
                    targets_reshaped = targets.view(-1)
                    # Scale loss by accumulation steps
                    loss = (
                        criterion(outputs, targets_reshaped)
                        / gradient_accumulation_steps
                    )

                # Mixed precision backward pass
                scaler.scale(loss).backward()
            else:
                # Regular precision
                outputs, _ = model(inputs, hidden)
                outputs = outputs.view(-1, get_vocab_size())
                targets_reshaped = targets.view(-1)
                # Scale loss by accumulation steps
                loss = (
                    criterion(outputs, targets_reshaped) / gradient_accumulation_steps
                )
                loss.backward()

            # Update epoch statistics
            epoch_loss += (
                loss.item() * gradient_accumulation_steps
            )  # Scale back for logging
            epoch_batches += 1

            # Step optimizer after accumulation steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                batch_idx + 1 == len(train_loader)
            ):
                # Apply gradient clipping and optimizer step
                if use_amp:
                    if config.get("use_gradient_clipping", False):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.get("gradient_clip_val", 1.0)
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if config.get("use_gradient_clipping", False):
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.get("gradient_clip_val", 1.0)
                        )
                    optimizer.step()

                # Zero gradients after optimization step
                optimizer.zero_grad()

            # Print batch statistics
            if batch_idx % config["print_every"] == 0:
                accum_status = f"[{(batch_idx % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}]"
                amp_status = "[AMP]" if use_amp else ""
                status = (
                    "[Profiling]"
                    if not flop_counter.profiled
                    else f"{amp_status} {accum_status}"
                )

                print(
                    f"Epoch {epoch+1}/{config['num_epochs']}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item() * gradient_accumulation_steps:.4f} {status}"
                )

        # Calculate epoch statistics
        avg_train_loss = epoch_loss / epoch_batches
        epoch_time = time.time() - epoch_start_time

        # Evaluate on validation set (no gradient accumulation needed here)
        val_loss = evaluate_model_amp(model, val_loader, criterion, device, use_amp)

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
        print(
            f"  Effective Batch Size: {effective_batch_size} (Physical: {config['batch_size']} × {gradient_accumulation_steps})"
        )

        if flop_counter.profiled and flop_counter.time_per_batch:
            print(f"  Profiled compute time: {flop_counter.time_per_batch:.4f}s")
            print(
                f"  Profiled FLOPs per second: {flop_counter.flops_per_batch / flop_counter.time_per_batch:.2e}"
            )

        print(f"  Total FLOPs: {flop_counter.total_flops:.2e}")
        print(f"  FLOPs per second: {flop_counter.total_flops/epoch_time:.2e}")
        print("-" * 60)

        # Log to wandb
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "train_perplexity": train_perplexity,
            "val_perplexity": val_perplexity,
            "total_flops": flop_counter.total_flops,
            "epoch_time": epoch_time,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "mixed_precision": use_amp,
            "effective_batch_size": effective_batch_size,
        }

        if flop_counter.profiled and flop_counter.time_per_batch:
            log_dict.update(
                {
                    "profiled_compute_time": flop_counter.time_per_batch,
                    "profiled_flops_per_second": flop_counter.flops_per_batch
                    / flop_counter.time_per_batch,
                }
            )

        wandb.log(log_dict)

        # Step scheduler once per epoch
        if scheduler is not None:
            scheduler.step()

    # Final evaluation
    print("\nEvaluating on test set...")
    test_loss = evaluate_model_amp(model, test_loader, criterion, device, use_amp)
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

    # Function to handle DataParallel or DDP wrapping
    def init_hidden(batch_size):
        if isinstance(model, nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ):
            return model.module.init_hidden(batch_size, device)
        else:
            return model.init_hidden(batch_size, device)

    # Create dummy input using tokenizer vocab size
    dummy_input = torch.randint(
        0, len(tokenizer), (config["batch_size"], config["sequence_length"])
    ).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            hidden = init_hidden(config["batch_size"])
            _ = model(dummy_input, hidden)

    # Benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(100):
            hidden = init_hidden(config["batch_size"])
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
    # Check if we're running in distributed mode
    # Look for SLURM variables
    use_ddp = (
        "SLURM_JOB_ID" in os.environ and int(os.environ.get("SLURM_NTASKS", 1)) > 1
    ) or ("LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1)
    local_rank = 0

    if use_ddp:
        # Get local rank from SLURM or PyTorch
        if "SLURM_LOCALID" in os.environ:
            local_rank = int(os.environ["SLURM_LOCALID"])
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize process group
        torch.cuda.set_device(local_rank)
        # Make sure these environment variables are set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(
                os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1))
            ),
            rank=int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0))),
        )
        CONFIG["device"] = f"cuda:{local_rank}"
        print(f"Starting LSTM training under DDP: rank {local_rank}")
    else:
        # Single GPU mode
        CONFIG["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Starting LSTM training in single GPU mode…")

    print(f"Configuration: {CONFIG}")

    # Train model with local rank
    trained_model, total_flops = train_model(CONFIG, local_rank)

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
