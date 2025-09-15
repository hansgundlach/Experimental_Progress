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
import random  # NEW: for reproducible seeding
import copy
import csv


def get_mup_learning_rates(model, base_lr, use_mup=False, mup_base_width=64):
    """Get muP-scaled learning rates for different parameter groups"""
    if not use_mup:
        return [{"params": model.parameters(), "lr": base_lr}]

    # Calculate scaling factor
    if hasattr(model, "module"):  # Handle DataParallel/DDP
        hidden_size = model.module.hidden_size
        mup_scale = hidden_size / mup_base_width
        embedding = model.module.embedding
        lstm_layers = model.module.lstm_layers
        linear = model.module.linear
        tie_embeddings = getattr(model.module, "tie_embeddings", False)
    else:
        hidden_size = model.hidden_size
        mup_scale = hidden_size / mup_base_width
        embedding = model.embedding
        lstm_layers = model.lstm_layers
        linear = model.linear
        tie_embeddings = getattr(model, "tie_embeddings", False)

    param_groups = []

    # Handle embedding parameters
    if tie_embeddings:
        # When weight tying is enabled, tied weights serve both embedding and output functions
        # Use geometric mean of embedding scaling (1/scale) and output scaling (1.0)
        # This preserves muP scaling laws better than choosing one or the other
        embedding_lr = base_lr / mup_scale  # What embedding would use
        output_lr = base_lr  # What output would use
        tied_lr = (embedding_lr * output_lr) ** 0.5  # Geometric mean

        param_groups.append(
            {
                "params": embedding.parameters(),
                "lr": tied_lr,
                "name": "embedding_tied",
            }
        )
    else:
        # Separate embedding and output parameters when not tied
        # Embedding parameters: lr scaled by 1/scale
        param_groups.append(
            {
                "params": embedding.parameters(),
                "lr": base_lr / mup_scale,
                "name": "embedding",
            }
        )
        # Output layer parameters: no scaling (base lr)
        param_groups.append(
            {"params": linear.parameters(), "lr": base_lr, "name": "output"}
        )

    # LSTM parameters: lr scaled by 1/scale
    lstm_params = []
    for lstm_layer in lstm_layers:
        lstm_params.extend(list(lstm_layer.parameters()))
    param_groups.append(
        {"params": lstm_params, "lr": base_lr / mup_scale, "name": "lstm"}
    )

    if tie_embeddings:
        tied_lr = (base_lr / mup_scale * base_lr) ** 0.5  # Recalculate for printing
        print(
            f"muP learning rates (tied): embedding/output={tied_lr:.6f} (geometric mean), lstm={base_lr/mup_scale:.6f}"
        )
    else:
        print(
            f"muP learning rates: embedding={base_lr/mup_scale:.6f}, lstm={base_lr/mup_scale:.6f}, output={base_lr:.6f}"
        )

    return param_groups


cudnn.benchmark = True


def tbptt_forward_backward(
    model,
    inputs,
    targets,
    hidden,
    criterion,
    use_amp,
    scaler,
    tbptt_length,
    gradient_accumulation_steps,
    get_vocab_size_fn,
):
    """
    Perform truncated BPTT forward and backward pass.

    Args:
        model: LSTM model
        inputs: Input tokens [batch_size, sequence_length]
        targets: Target tokens [batch_size, sequence_length]
        hidden: Initial hidden state tuple (h, c)
        criterion: Loss function
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for mixed precision
        tbptt_length: Window length for truncated BPTT
        gradient_accumulation_steps: Number of gradient accumulation steps
        get_vocab_size_fn: Function to get vocabulary size

    Returns:
        total_loss: Accumulated loss over all windows
        final_hidden: Final hidden state (detached)
    """
    batch_size, full_sequence_length = inputs.size()
    total_loss = 0.0

    # Process sequence in windows of tbptt_length
    for start_idx in range(0, full_sequence_length, tbptt_length):
        end_idx = min(start_idx + tbptt_length, full_sequence_length)
        window_inputs = inputs[:, start_idx:end_idx]
        window_targets = targets[:, start_idx:end_idx]

        # Forward pass through this window
        if use_amp:
            with autocast():
                window_outputs, hidden = model(window_inputs, hidden)
                # Detach hidden state to truncate gradients at window boundary
                hidden = tuple(h.detach() for h in hidden)

                # Compute loss for this window
                window_outputs = window_outputs.reshape(-1, get_vocab_size_fn())
                window_targets_reshaped = window_targets.reshape(-1)
                window_loss = (
                    criterion(window_outputs, window_targets_reshaped)
                    / gradient_accumulation_steps
                )

            # Backward pass for this window
            scaler.scale(window_loss).backward()
        else:
            # Regular precision
            window_outputs, hidden = model(window_inputs, hidden)
            # Detach hidden state to truncate gradients at window boundary
            hidden = tuple(h.detach() for h in hidden)

            # Compute loss for this window
            window_outputs = window_outputs.reshape(-1, get_vocab_size_fn())
            window_targets_reshaped = window_targets.reshape(-1)
            window_loss = (
                criterion(window_outputs, window_targets_reshaped)
                / gradient_accumulation_steps
            )

            # Backward pass for this window
            window_loss.backward()

        total_loss += window_loss.item()

    return (
        total_loss * gradient_accumulation_steps,
        hidden,
    )  # Scale back loss for logging


class WikiTextDataset(Dataset):
    """Dataset of fixed‐length sequences with configurable stride."""

    def __init__(self, text_data: List[int], sequence_length: int, stride: int = 1):
        self.data = text_data
        self.sequence_length = sequence_length
        self.stride = stride

    def __len__(self):
        # number of windows we can slide over the data, never negative
        raw = (len(self.data) - self.sequence_length) // self.stride
        return max(0, raw)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.sequence_length
        seq = torch.tensor(self.data[start:end], dtype=torch.long)
        tgt = torch.tensor(self.data[start + 1 : end + 1], dtype=torch.long)
        return seq, tgt


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
        # FIXED: Use tokenizer directly to avoid sequence length validation
        # The encode() method can trigger model max_length validation
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_tensors=None,  # Return Python list
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        return tokens


class VariationalDropout(nn.Module):
    """
    Variational dropout that uses the same mask across all timesteps.
    Based on "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks"
    """

    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or self.dropout_rate == 0:
            return x

        # x shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.size()

        # Create mask that's constant across timesteps
        # Shape: (batch_size, 1, hidden_size) - broadcasts across seq_len
        mask = x.new_ones(batch_size, 1, hidden_size)
        mask = torch.bernoulli(mask * (1 - self.dropout_rate))

        # Scale by dropout probability to maintain expected value
        mask = mask / (1 - self.dropout_rate)

        # Apply same mask to all timesteps
        return x * mask


class LSTMCellWithRecurrentDropout(nn.Module):
    """
    Custom LSTM cell with recurrent (hidden-to-hidden) dropout.
    Applies dropout to the hidden state before it's used in the next timestep.
    """

    def __init__(self, input_size, hidden_size, recurrent_dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_dropout = recurrent_dropout

        # Standard LSTM parameters
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard LSTM practices"""
        # Input-to-hidden weights: Xavier uniform
        nn.init.xavier_uniform_(self.weight_ih)
        # Hidden-to-hidden weights: Orthogonal
        nn.init.orthogonal_(self.weight_hh)
        # Biases: zeros, except forget gate bias = 1
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
        self.bias_ih[self.hidden_size : 2 * self.hidden_size].fill_(1.0)  # forget gate
        self.bias_hh[self.hidden_size : 2 * self.hidden_size].fill_(1.0)  # forget gate

    def forward(self, input, hidden):
        """
        Args:
            input: (batch_size, input_size)
            hidden: tuple of (h_prev, c_prev) each of shape (batch_size, hidden_size)
        Returns:
            output: (batch_size, hidden_size)
            new_hidden: tuple of (h_new, c_new)
        """
        h_prev, c_prev = hidden
        batch_size = input.size(0)

        # Apply recurrent dropout to hidden state if training
        if self.training and self.recurrent_dropout > 0:
            # Create dropout mask for hidden state
            dropout_mask = torch.bernoulli(
                h_prev.new_ones(batch_size, self.hidden_size)
                * (1 - self.recurrent_dropout)
            ) / (1 - self.recurrent_dropout)
            h_prev = h_prev * dropout_mask

        # Compute gates
        gi = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        gh = torch.mm(h_prev, self.weight_hh.t()) + self.bias_hh
        gates = gi + gh

        # Split gates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # Apply activations
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        # Update cell state
        c_new = forgetgate * c_prev + ingate * cellgate

        # Compute new hidden state
        h_new = outgate * torch.tanh(c_new)

        return h_new, (h_new, c_new)


class MultiLayerLSTMWithRecurrentDropout(nn.Module):
    """
    Multi-layer LSTM using custom cells with recurrent dropout support.
    """

    def __init__(self, input_size, hidden_size, num_layers, recurrent_dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent_dropout = recurrent_dropout

        # Create LSTM cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(
                LSTMCellWithRecurrentDropout(
                    layer_input_size, hidden_size, recurrent_dropout
                )
            )

    def forward(self, input, hidden=None):
        """
        Args:
            input: (batch_size, seq_len, input_size)
            hidden: tuple of (h_0, c_0) each of shape (num_layers, batch_size, hidden_size)
        Returns:
            output: (batch_size, seq_len, hidden_size)
            new_hidden: tuple of (h_n, c_n)
        """
        batch_size, seq_len, _ = input.size()

        if hidden is None:
            h_0 = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
            c_0 = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
            hidden = (h_0, c_0)

        h_prev, c_prev = hidden

        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = input[:, t, :]  # (batch_size, input_size)

            # Process through layers
            h_new = []
            c_new = []
            for layer_idx, cell in enumerate(self.cells):
                layer_hidden = (h_prev[layer_idx], c_prev[layer_idx])
                h_out, (h_out, c_out) = cell(x_t, layer_hidden)
                h_new.append(h_out)
                c_new.append(c_out)
                x_t = h_out  # Use output as input for next layer

            # Stack layer outputs
            h_prev = torch.stack(h_new, dim=0)  # (num_layers, batch_size, hidden_size)
            c_prev = torch.stack(c_new, dim=0)  # (num_layers, batch_size, hidden_size)

            # Store output from last layer
            outputs.append(x_t)  # (batch_size, hidden_size)

        # Stack time outputs
        output = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)

        return output, (h_prev, c_prev)


class VanillaLSTMLanguageModel(nn.Module):
    """
    Advanced LSTM language model with comprehensive dropout support.
    Supports variational dropout, between-layers dropout, and recurrent dropout.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        output_dropout: float = 0.0,
        between_layers_dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        use_layer_norm: bool = False,
        layer_norm_position: str = "output",
        use_mup: bool = False,
        mup_base_width: int = 64,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.use_mup = use_mup
        self.mup_base_width = mup_base_width
        self.tie_embeddings = tie_embeddings

        # Calculate muP scaling factor
        self.mup_scale = hidden_size / mup_base_width if use_mup else 1.0

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Create LSTM layers - use custom recurrent dropout LSTM if needed
        if recurrent_dropout > 0:
            # Use custom LSTM with recurrent dropout
            self.lstm = MultiLayerLSTMWithRecurrentDropout(
                hidden_size, hidden_size, num_layers, recurrent_dropout
            )
            self.use_custom_lstm = True
        else:
            # Use standard PyTorch LSTM layers for better performance
            self.lstm_layers = nn.ModuleList()
            for i in range(num_layers):
                self.lstm_layers.append(
                    nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
                )
            self.use_custom_lstm = False

        # Different dropout types
        self.input_dropout = VariationalDropout(input_dropout)
        self.hidden_dropouts = nn.ModuleList(
            [VariationalDropout(hidden_dropout) for _ in range(num_layers - 1)]
        )
        self.between_layers_dropout = nn.Dropout(between_layers_dropout)
        self.output_dropout = VariationalDropout(output_dropout)

        self.linear = nn.Linear(hidden_size, vocab_size)

        # Tie input and output embeddings if specified
        if self.tie_embeddings:
            self.linear.weight = self.embedding.weight

        self.use_layer_norm = use_layer_norm
        self.layer_norm_position = layer_norm_position

        # Add LayerNorm layers based on config
        if use_layer_norm:
            if layer_norm_position in ["input", "both"]:
                self.input_layer_norm = nn.LayerNorm(hidden_size)
            if layer_norm_position in ["output", "both"]:
                self.output_layer_norms = nn.ModuleList(
                    [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
                )

        # Apply initialization based on configuration
        if use_mup:
            self._apply_mup_init()
        else:
            self._apply_standard_lstm_init()

    def _apply_standard_lstm_init(self):
        """Apply standard LSTM initialization recipe"""
        # Embedding layer: Xavier uniform initialization
        nn.init.xavier_uniform_(self.embedding.weight)

        # LSTM layers: follow standard LSTM initialization
        if self.use_custom_lstm:
            # Custom LSTM initialization is handled in the cell constructor
            pass
        else:
            # Standard LSTM layers initialization
            for lstm_layer in self.lstm_layers:
                for name, param in lstm_layer.named_parameters():
                    if "weight_ih" in name:
                        # Input-to-hidden weights: Xavier uniform
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        # Hidden-to-hidden weights: Orthogonal
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        # Set all biases to 0, then forget gate bias to +1
                        param.data.fill_(0.0)
                        hidden_size = param.size(0) // 4
                        param.data[hidden_size : 2 * hidden_size].fill_(
                            1.0
                        )  # forget gate bias

        # Output layer: Xavier uniform initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def _apply_mup_init(self):
        """Apply muP initialization with standard LSTM practices"""
        # Embedding layer: scale by 1/sqrt(width) but use uniform distribution
        std = 1 / math.sqrt(self.hidden_size)
        bound = math.sqrt(3.0) * std  # Convert to uniform bound
        nn.init.uniform_(self.embedding.weight, -bound, bound)

        # LSTM layers: combine muP scaling with standard LSTM practices
        if self.use_custom_lstm:
            # Custom LSTM initialization is handled in the cell constructor
            # We need to reinitialize the custom LSTM with muP scaling
            pass  # For now, custom LSTM uses standard initialization
        else:
            # Standard LSTM layers initialization
            for lstm_layer in self.lstm_layers:
                for name, param in lstm_layer.named_parameters():
                    if "weight_ih" in name:
                        # Input-to-hidden: muP scaled Xavier uniform
                        std = 1 / math.sqrt(self.hidden_size)
                        bound = math.sqrt(3.0) * std
                        nn.init.uniform_(param.data, -bound, bound)
                    elif "weight_hh" in name:
                        # Hidden-to-hidden: Orthogonal (standard LSTM practice)
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        # Standard LSTM bias initialization
                        param.data.fill_(0.0)
                        hidden_size = param.size(0) // 4
                        param.data[hidden_size : 2 * hidden_size].fill_(
                            1.0
                        )  # forget gate bias

        # Output layer: muP scaled initialization
        # Only initialize if not tied to embedding
        if not self.tie_embeddings:
            std = 1 / self.hidden_size
            bound = math.sqrt(3.0) * std
            nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)

        # Apply input LayerNorm if configured
        if self.use_layer_norm and self.layer_norm_position in ["input", "both"]:
            embedded = self.input_layer_norm(embedded)

        lstm_input = self.input_dropout(embedded)

        # Initialize hidden states if not provided
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        # Process through LSTM layers
        if self.use_custom_lstm:
            # Use custom LSTM with recurrent dropout
            lstm_out, (h_n, c_n) = self.lstm(lstm_input, hidden)
        else:
            # Use standard LSTM layers with variational dropout
            lstm_out = lstm_input
            new_hidden = []

            for i, lstm_layer in enumerate(self.lstm_layers):
                layer_hidden = (hidden[0][i : i + 1], hidden[1][i : i + 1])
                lstm_out, layer_new_hidden = lstm_layer(lstm_out, layer_hidden)
                new_hidden.append(layer_new_hidden)

                # Apply output LayerNorm if configured
                if self.use_layer_norm and self.layer_norm_position in [
                    "output",
                    "both",
                ]:
                    lstm_out = self.output_layer_norms[i](lstm_out)

                # Apply between-layers dropout (standard dropout between LSTM layers)
                if i < len(self.lstm_layers) - 1:
                    lstm_out = self.between_layers_dropout(lstm_out)

                # Apply hidden dropout (variational dropout, except for last layer)
                if i < len(self.lstm_layers) - 1:
                    lstm_out = self.hidden_dropouts[i](lstm_out)

            # Combine hidden states
            h_n = torch.cat([h[0] for h in new_hidden], dim=0)
            c_n = torch.cat([c[1] for c in new_hidden], dim=0)

        # Apply output dropout
        lstm_out = self.output_dropout(lstm_out)

        output = self.linear(lstm_out)

        # Apply muP output scaling
        if self.use_mup:
            output = output * self.mup_scale

        return output, (h_n, c_n)

    def init_hidden(self, batch_size: int, device: torch.device):
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

    def add_batch_flops(self, microbatches_in_step: int = 1):
        """Add FLOPs for a group of microbatches (one optimizer step).

        Assumes flops_per_batch represents one microbatch forward+backward (+ step overhead).
        We multiply by the number of microbatches contributing to this optimizer step.
        """
        if self.flops_per_batch is not None:
            self.total_flops += self.flops_per_batch * int(max(1, microbatches_in_step))
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

    # Load raw text
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

    # Create contiguous splits for stateful training, do not shuffle sentences.
    preprocessor = TextPreprocessor(config["tokenizer_path"])
    full_data = preprocessor.text_to_indices(text)
    n = len(full_data)
    n_train = int(n * config["train_split"])
    n_val = int(n * config["val_split"])

    train_data = full_data[:n_train]
    val_data = full_data[n_train : n_train + n_val]
    test_data = full_data[n_train + n_val :]

    # Create sliding‐window datasets with same stride as transformer
    stride = config.get("stride", 1)
    train_dataset = WikiTextDataset(
        train_data, config["sequence_length"], stride=stride
    )
    val_dataset = WikiTextDataset(val_data, config["sequence_length"], stride=stride)
    test_dataset = WikiTextDataset(test_data, config["sequence_length"], stride=stride)

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
            shuffle=False,  # Must be False for stateful training
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=False,  # Must be False for stateful training
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False,
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

    if total_batches == 0:
        return float("nan")
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

    if total_batches == 0:
        return float("nan")
    return total_loss / total_batches


def train_model(
    config: Dict, local_rank=0, run_name: str = None, csv_log_path: str = None
):
    """Main training function with mixed precision and gradient accumulation"""
    # Initialize wandb
    if config["wandb_offline"]:
        os.environ["WANDB_MODE"] = "offline"

    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    # CSV Logging Setup
    csv_log_interval = config.get("csv_log_interval")
    csv_file = None
    csv_writer = None
    if is_main_process and csv_log_path and csv_log_interval:
        try:
            os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
            csv_file = open(csv_log_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    "step",
                    "training_loss",
                    "validation_loss",
                    "total_flops_profiler",
                    "theoretical_flops",
                    "tokens",
                ]
            )
            csv_file.flush()  # Immediately write header to disk
            print(f"Logging training progress to {csv_log_path}")
        except IOError as e:
            print(
                f"Warning: Could not open {csv_log_path} for writing. CSV logging disabled. Error: {e}"
            )
            csv_writer = None

    # Calculate effective batch size
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    effective_batch_size = config["batch_size"] * gradient_accumulation_steps
    config["effective_batch_size"] = effective_batch_size  # Store for reporting

    wandb.init(project=config["wandb_project"], config=config, name=run_name)

    # === Seed everything exactly as in transformer.train() ===
    seed = wandb.config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.max_split_size_mb = 128

    # Set device
    device = torch.device(config["device"])
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, preprocessor = load_and_preprocess_data(
        config
    )

    # Initialize model
    model = VanillaLSTMLanguageModel(
        vocab_size=preprocessor.vocab_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        input_dropout=config["input_dropout"],
        hidden_dropout=config["hidden_dropout"],
        output_dropout=config["output_dropout"],
        between_layers_dropout=config.get("between_layers_dropout", 0.0),
        recurrent_dropout=config.get("recurrent_dropout", 0.0),
        use_layer_norm=config.get("use_layer_norm", False),
        layer_norm_position=config.get("layer_norm_position", "output"),
        use_mup=config.get("use_mup", False),
        mup_base_width=config.get("mup_base_width", 64),
        tie_embeddings=config.get("tie_embeddings", True),  # Default to True
    ).to(device)

    # Use DataParallel or DDP if requested
    if torch.cuda.device_count() > 1 and not dist.is_initialized():
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    elif dist.is_initialized():
        print(f"Using DistributedDataParallel for training")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # NEW: compile the model (must be after any parallel wrappers)
    if config.get("use_compile", False):
        print("🔧 Compiling model with torch.compile() …")
        # you can pass mode/backend args here, e.g. backend="inductor", mode="max-autotune"
        model = torch.compile(model)

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

    # Get parameter groups with muP learning rate scaling if enabled
    use_mup = config.get("use_mup", False)
    if use_mup:
        param_groups = get_mup_learning_rates(
            model,
            config["learning_rate"],
            use_mup=True,
            mup_base_width=config.get("mup_base_width", 64),
        )
    else:
        # Standard case: single parameter group
        param_groups = model.parameters()

    # Select optimizer based on config
    opt_name = config.get("optimizer", "adam").lower()
    if opt_name == "adam":
        if use_mup:
            optimizer = optim.Adam(param_groups)
        else:
            optimizer = optim.Adam(param_groups, lr=config["learning_rate"])
    elif opt_name == "adamw":
        if use_mup:
            optimizer = optim.AdamW(
                param_groups,
                weight_decay=config.get("weight_decay", 0.0),
            )
        else:
            optimizer = optim.AdamW(
                param_groups,
                lr=config["learning_rate"],
                weight_decay=config.get("weight_decay", 0.0),
            )
    elif opt_name == "sgd":
        if use_mup:
            optimizer = optim.SGD(
                param_groups,
                momentum=config.get("momentum", 0.9),
                weight_decay=config.get("weight_decay", 0.0),
            )
        else:
            optimizer = optim.SGD(
                param_groups,
                lr=config["learning_rate"],
                momentum=config.get("momentum", 0.9),
                weight_decay=config.get("weight_decay", 0.0),
            )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    # Initialize mixed precision scaler
    use_amp = config.get("use_amp", False) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    print(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    print(
        f"Gradient Accumulation: {gradient_accumulation_steps} steps (Effective batch size: {effective_batch_size})"
    )

    # Print TBPTT configuration
    use_tbptt = config.get("use_tbptt", True)
    if use_tbptt:
        tbptt_length = config.get("tbptt_length", 128)
        tbptt_stride = config.get("tbptt_stride", 128)
        tbptt_reset_hidden = config.get("tbptt_reset_hidden", True)
        print(
            f"Truncated BPTT: Enabled (length={tbptt_length}, stride={tbptt_stride}, reset_hidden={tbptt_reset_hidden})"
        )
    else:
        print("Truncated BPTT: Disabled (using full sequence with hidden detaching)")

    # Calculate total steps and warmup steps for step-based schedulers
    steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * config["num_epochs"])

    warmup_frac = config.get("warmup_frac", 0.0)
    if not (0.0 <= warmup_frac < 1.0):
        warmup_frac = 0.0

    warmup_steps = int(total_steps * warmup_frac)
    if warmup_frac > 0.0:
        warmup_steps = max(1, warmup_steps)
        print(
            f"Using {warmup_steps} warm-up step(s) ({warmup_steps/total_steps*100:.1f}% of total steps)"
        )
    else:
        print("No warm-up will be used.")

    # Set up LR scheduler
    if config.get("lr_schedule") == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.1),
        )
        scheduler_type = "epoch"
    elif config.get("lr_schedule") == "cosine":
        # Determine effective min_lr
        min_lr_multiplier = config.get("min_lr_multiplier", None)
        if min_lr_multiplier is not None:
            effective_min_lr = min_lr_multiplier * config["learning_rate"]
        else:
            effective_min_lr = config.get("min_lr", 0)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=effective_min_lr
        )
        scheduler_type = "step"
    elif config.get("lr_schedule") == "cosine_warmup":
        # Determine effective min_lr
        min_lr_multiplier = config.get("min_lr_multiplier", None)
        if min_lr_multiplier is not None:
            effective_min_lr = min_lr_multiplier * config["learning_rate"]
        else:
            effective_min_lr = config.get("min_lr", 0)

        # Calculate min_lr ratio for cosine decay
        min_lr_ratio = effective_min_lr / config["learning_rate"]

        def warmup_cosine_step(step):
            step = max(1, step)
            # Linear warmup
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # Cosine decay from 1.0 to min_lr_ratio
            if total_steps == warmup_steps:
                return 1.0
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_step)
        scheduler_type = "step"
    else:
        scheduler = None
        scheduler_type = None

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
    optimizer_step_counter = 0
    for epoch in range(config["num_epochs"]):
        # reshuffle for DDP
        if dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_batches = 0
        optimizer.zero_grad()  # Zero gradients at the start of epoch

        # Initialize hidden state at the start of each epoch
        hidden = None

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_size, sequence_length = inputs.size()

            # Initialize or resize hidden state if necessary
            if hidden is None or inputs.size(0) != hidden[0].size(1):
                hidden = init_hidden(batch_size)

            # Profile only the very first batch of training
            if not flop_counter.profiled:
                loss = flop_counter.profile_one_batch(
                    inputs, targets, hidden, optimizer, criterion, scaler
                )
                # Re-zero gradients after profiling
                optimizer.zero_grad()
                # Update epoch statistics for profiled batch
                epoch_loss += loss.item()
                epoch_batches += 1
                # After profiling, we don't update the hidden state to avoid polluting the next step
                hidden = init_hidden(batch_size)
                continue

            # Move data to device
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )

            # Track FLOPs: add FLOPs once per optimizer step, scaled by number of microbatches.
            is_optimizer_step = (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                batch_idx + 1 == len(train_loader)
            )
            if is_optimizer_step:
                # Determine how many microbatches contributed to this step
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    microbatches_this_step = gradient_accumulation_steps
                else:
                    # Last partial step of the epoch
                    remainder = (batch_idx + 1) % gradient_accumulation_steps
                    microbatches_this_step = (
                        remainder if remainder != 0 else gradient_accumulation_steps
                    )
                flop_counter.add_batch_flops(microbatches_this_step)

            # Choose forward/backward method based on TBPTT configuration
            if config.get("use_tbptt", True):
                # Use truncated BPTT
                tbptt_length = config.get("tbptt_length", 128)
                loss_value, hidden = tbptt_forward_backward(
                    model,
                    inputs,
                    targets,
                    hidden,
                    criterion,
                    use_amp,
                    scaler,
                    tbptt_length,
                    gradient_accumulation_steps,
                    get_vocab_size,
                )
                # Convert loss value to tensor for consistent interface
                loss = torch.tensor(
                    loss_value / gradient_accumulation_steps, device=device
                )
            else:
                # Original method: full sequence forward pass with hidden state detaching
                if use_amp:
                    with autocast():
                        outputs, hidden = model(inputs, hidden)
                        # Detach hidden state to treat it as a new input, preventing BPTT through all time
                        hidden = tuple(h.detach() for h in hidden)
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
                    outputs, hidden = model(inputs, hidden)
                    # Detach hidden state to treat it as a new input, preventing BPTT through all time
                    hidden = tuple(h.detach() for h in hidden)
                    outputs = outputs.view(-1, get_vocab_size())
                    targets_reshaped = targets.view(-1)
                    # Scale loss by accumulation steps
                    loss = (
                        criterion(outputs, targets_reshaped)
                        / gradient_accumulation_steps
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

                # Update step counter and step-based scheduler
                optimizer_step_counter += 1
                if scheduler is not None and scheduler_type == "step":
                    scheduler.step(optimizer_step_counter)

            # Log to CSV if enabled
            if csv_writer and (batch_idx + 1) % csv_log_interval == 0:
                current_step = batch_idx + 1
                current_train_loss = loss.item() * gradient_accumulation_steps

                # Temporarily switch to eval mode for validation
                model.eval()
                with torch.no_grad():
                    current_val_loss = evaluate_model_amp(
                        model, val_loader, criterion, device, use_amp
                    )
                model.train()

                total_flops = flop_counter.total_flops

                # Calculate theoretical FLOPs using 6ND formula (Chinchilla) - including embedding params
                effective_batch_size = (
                    config["batch_size"] * gradient_accumulation_steps
                )
                tokens_processed = (
                    current_step * effective_batch_size * config["sequence_length"]
                )

                # Count total parameters for theoretical FLOP calculation
                total_model_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                theoretical_flops_chinchilla = 6 * total_model_params * tokens_processed

                csv_writer.writerow(
                    [
                        current_step,
                        f"{current_train_loss:.4f}",
                        f"{current_val_loss:.4f}",
                        f"{total_flops:.2e}",
                        f"{theoretical_flops_chinchilla:.2e}",
                        f"{tokens_processed}",
                    ]
                )
                csv_file.flush()  # Immediately write row to disk

                # ALSO: log validation loss and cumulative FLOPs to W&B
                wandb.log(
                    {
                        "validation_loss": current_val_loss,
                        "total_flops_profiler": total_flops,
                    },
                    step=current_step,
                )

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

        # Step scheduler once per epoch (only for epoch-based schedulers)
        if scheduler is not None and scheduler_type == "epoch":
            scheduler.step()

    # Final evaluation
    print("\nEvaluating on test set...")
    test_loss = evaluate_model_amp(model, test_loader, criterion, device, use_amp)
    test_perplexity = np.exp(test_loss)
    print(f"Test Loss: {test_loss:.4f} (Perplexity: {test_perplexity:.2f})")

    # Calculate theoretical FLOPs
    # FIXED: Properly account for gradient accumulation
    flops_per_batch_manual = (
        flop_counter.count_forward_flops_manual(
            config["batch_size"], config["sequence_length"]
        )
        * 3
    )  # x3 for fwd+bwd
    # Count optimizer steps and microbatches precisely
    batches_per_epoch = len(train_loader)
    full_steps_per_epoch = batches_per_epoch // gradient_accumulation_steps
    remainder_microbatches = batches_per_epoch % gradient_accumulation_steps
    optimizer_steps_per_epoch = full_steps_per_epoch + (
        1 if remainder_microbatches > 0 else 0
    )
    total_optimizer_steps = optimizer_steps_per_epoch * config["num_epochs"]

    # Total microbatches processed over all epochs
    total_microbatches = batches_per_epoch * config["num_epochs"]
    # Theoretical FLOPs: per-microbatch fwd+bwd (x3) times total microbatches
    total_flops_theoretical = flops_per_batch_manual * total_microbatches

    # FIXED: Also calculate theoretical FLOPs using the standard formula
    # Theoretical FLOPs: 6 * non_embedding_params * total_tokens
    effective_batch_size = config["batch_size"] * gradient_accumulation_steps
    # Tokens are counted per microbatch
    total_tokens_processed = (
        total_microbatches * config["batch_size"] * config["sequence_length"]
    )

    # Count non-embedding parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Handle DDP wrapping
    if hasattr(model, "module"):
        num_embedding_params = model.module.embedding.weight.numel()
    else:
        num_embedding_params = model.embedding.weight.numel()
    num_non_embedding_params = num_params - num_embedding_params

    total_flops_theoretical_standard = (
        6 * num_non_embedding_params * total_tokens_processed
    )

    # Debug output for FLOPs calculations
    print(f"\n==== FLOPs Analysis ====")
    print(f"Profiler FLOPs per batch: {flop_counter.flops_per_batch:.2e}")
    print(f"Total profiler FLOPs: {flop_counter.total_flops:.2e}")
    print(f"Manual FLOPs per batch: {flops_per_batch_manual:.2e}")
    print(f"Total optimizer steps: {total_optimizer_steps}")
    print(f"Total batches: {len(train_loader) * config['num_epochs']}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Total tokens processed: {total_tokens_processed:.2e}")
    print(f"Non-embedding parameters: {num_non_embedding_params:,}")
    print(f"Theoretical FLOPs (manual): {total_flops_theoretical:.2e}")
    print(f"Theoretical FLOPs (standard): {total_flops_theoretical_standard:.2e}")
    print(f"========================\n")

    results = {
        "final_train_loss": avg_train_loss,
        "final_val_loss": val_loss,
        "test_loss": test_loss,
        "total_flops_profiler": flop_counter.total_flops,
        "total_flops_theoretical": total_flops_theoretical,
        "total_flops_theoretical_standard": total_flops_theoretical_standard,
    }

    wandb.log(
        {
            "test_loss": test_loss,
            "test_perplexity": test_perplexity,
            "final_total_flops": flop_counter.total_flops,
            "final_total_flops_theoretical": total_flops_theoretical,
            "final_total_flops_theoretical_standard": total_flops_theoretical_standard,
        }
    )

    if csv_file:
        csv_file.close()

    wandb.finish()
    return model, results


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
