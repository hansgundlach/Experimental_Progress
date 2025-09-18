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
from transformers import GPT2Tokenizer, PreTrainedTokenizer
from torch.profiler import profile, record_function, ProfilerActivity
import multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import math
import torch.backends.cudnn as cudnn
import random  # NEW: for reproducible seeding
import copy
import csv


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
    Perform truncated BPTT forward and backward pass with proper per-token loss scaling.

    Args:
        model: LSTM model
        inputs: Input tokens [batch_size, sequence_length]
        targets: Target tokens [batch_size, sequence_length]
        hidden: Initial hidden state tuple (h, c)
        criterion: Loss function (should use reduction='sum')
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for mixed precision
        tbptt_length: Window length for truncated BPTT
        gradient_accumulation_steps: Number of gradient accumulation steps
        get_vocab_size_fn: Function to get vocabulary size

    Returns:
        per_token_loss: Per-token loss for logging
        final_hidden: Final hidden state (detached)
    """
    batch_size, full_sequence_length = inputs.size()
    tokens_mb = batch_size * full_sequence_length  # Total tokens in microbatch
    total_loss_sum = 0.0

    # Per-token loss scaling for gradient accumulation
    scale = 1.0 / (tokens_mb * gradient_accumulation_steps)

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

                # Compute sum loss for this window
                window_outputs = window_outputs.reshape(-1, get_vocab_size_fn())
                window_targets_reshaped = window_targets.reshape(-1)
                loss_sum = criterion(window_outputs, window_targets_reshaped)

            # Backward pass with proper scaling
            scaled_loss = loss_sum * scale
            scaler.scale(scaled_loss).backward()
        else:
            # Regular precision
            window_outputs, hidden = model(window_inputs, hidden)
            # Detach hidden state to truncate gradients at window boundary
            hidden = tuple(h.detach() for h in hidden)

            # Compute sum loss for this window
            window_outputs = window_outputs.reshape(-1, get_vocab_size_fn())
            window_targets_reshaped = window_targets.reshape(-1)
            loss_sum = criterion(window_outputs, window_targets_reshaped)

            # Backward pass with proper scaling
            scaled_loss = loss_sum * scale
            scaled_loss.backward()

        total_loss_sum += loss_sum.item()

    # Return per-token loss for logging
    per_token_loss = total_loss_sum / tokens_mb
    return per_token_loss, hidden


class TextDataset(Dataset):
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


class TextStreamingDataset(Dataset):
    """
    Streaming dataset that reshapes data into B contiguous streams.
    Follows Melis/Merity style: data is reshaped into [B, T_total] where
    each batch consists of the next tokens from B continuous streams.
    """

    def __init__(self, text_data: List[int], sequence_length: int, batch_size: int):
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        # Calculate how many tokens we can use (must be divisible by batch_size)
        total_tokens = len(text_data)
        tokens_per_stream = total_tokens // batch_size
        usable_tokens = tokens_per_stream * batch_size

        if usable_tokens < batch_size * sequence_length:
            raise ValueError(
                f"Dataset too small: need at least {batch_size * sequence_length} tokens, got {usable_tokens}"
            )

        # Reshape into [B, T_stream] where T_stream = tokens_per_stream
        data_tensor = torch.tensor(text_data[:usable_tokens], dtype=torch.long)
        self.streams = data_tensor.view(batch_size, tokens_per_stream)

        # Number of sequence-length windows we can extract from each stream
        self.num_batches = (tokens_per_stream - 1) // sequence_length

        print(
            f"Streaming dataset: {batch_size} streams, {tokens_per_stream} tokens per stream, {self.num_batches} batches"
        )

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        # Extract sequence starting at position idx * sequence_length from all streams
        start_pos = idx * self.sequence_length
        end_pos = start_pos + self.sequence_length

        # Get inputs and targets for all streams at this time step
        inputs = self.streams[:, start_pos:end_pos]  # [B, seq_len]
        targets = self.streams[:, start_pos + 1 : end_pos + 1]  # [B, seq_len]

        return inputs, targets


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
            f"Loaded tokenizer from local path with vocabulary size: {self.vocab_size}"
        )

    def text_to_indices(self, text: str) -> List[int]:
        # Tokenize the text using the configured tokenizer
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
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings

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

        # Apply standard initialization
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
        # FIXED: Only initialize if not tied to embedding (avoid double initialization)
        if not self.tie_embeddings:
            nn.init.xavier_uniform_(self.linear.weight)
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

            # FIXED: Apply the same regularization as standard LSTM path
            # Note: Custom LSTM already handles layer-by-layer processing, so we need
            # to simulate the per-layer regularization that standard LSTM does
            # For simplicity, apply layer norm and dropouts to the final output
            if self.use_layer_norm and self.layer_norm_position in ["output", "both"]:
                # Apply layer norm from the last layer
                lstm_out = self.output_layer_norms[-1](lstm_out)

            # Apply between-layers dropout (equivalent to what happens between layers)
            if self.num_layers > 1:
                lstm_out = self.between_layers_dropout(lstm_out)

            # Apply hidden dropout (equivalent to what happens between layers)
            if self.num_layers > 1:
                lstm_out = self.hidden_dropouts[-1](lstm_out)
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

        # FIXED: Remove post-hoc muP logit scaling
        # muP scaling should be handled via learning rates and initialization only

        return output, (h_n, c_n)

    def init_hidden(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
        )


class FLOPCounter:
    def __init__(self, model: nn.Module, config: Dict, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.total_flops = 0
        self.flops_per_batch = None
        self.time_per_batch = None
        self.profiled = False

    def get_model_vocab_size(self):
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
                # FIXED: Profiler must not train - only forward+backward, no optimizer step
                # Clear gradients first for clean profiling
                optimizer.zero_grad()

                # Mixed precision backward pass (for FLOP counting only)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # Do NOT call scaler.step() or scaler.update() - we're only profiling
                else:
                    loss.backward()
                    # Do NOT call optimizer.step() - we're only profiling

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
        # FIXED: Don't increment total_flops in profiler - only set flops_per_batch
        # total_flops should only be incremented on real optimizer steps
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
    """Load and preprocess text data with optimized DataLoaders"""
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

    # Choose dataset type based on streaming config
    use_streaming = config.get("use_streaming", False)

    if use_streaming:
        print("Using streaming (Melis/Merity style) datasets")
        train_dataset = TextStreamingDataset(
            train_data, config["sequence_length"], config["batch_size"]
        )
        val_dataset = TextStreamingDataset(
            val_data, config["sequence_length"], config["batch_size"]
        )
        test_dataset = TextStreamingDataset(
            test_data, config["sequence_length"], config["batch_size"]
        )
    else:
        print("Using non-streaming (sliding window) datasets")
        stride = config.get("stride", 1)
        train_dataset = TextDataset(
            train_data, config["sequence_length"], stride=stride
        )
        val_dataset = TextDataset(val_data, config["sequence_length"], stride=stride)
        test_dataset = TextDataset(test_data, config["sequence_length"], stride=stride)

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

    # Create optimized data loaders (single-GPU only)
    # For streaming datasets, use batch_size=1 since dataset handles batching internally
    dataloader_batch_size = 1 if use_streaming else config["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader_batch_size,
        shuffle=False,  # Must be False for stateful training
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=dataloader_batch_size,
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
    config: Dict = None,
) -> float:
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    total_batches = 0

    # Helper function to get hidden state
    def get_hidden(batch_size):
        return model.init_hidden(batch_size, device)

    # Helper function to get vocab size
    def get_vocab_size():
        return model.vocab_size

    # Handle streaming evaluation policy
    use_streaming = config.get("use_streaming", False) if config else False
    eval_streaming_like_train = (
        config.get("eval_streaming_like_train", True) if config else True
    )
    use_eval_streaming = use_streaming and eval_streaming_like_train

    eval_hidden = None
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Handle tensor shapes - if 3D, squeeze first dimension (streaming mode)
            if inputs.dim() == 3:
                inputs = inputs.squeeze(0)
                targets = targets.squeeze(0)

            batch_size, sequence_length = inputs.size()
            tokens_in_batch = batch_size * sequence_length

            # Handle hidden state based on eval streaming policy
            if use_eval_streaming:
                # Streaming evaluation: carry hidden state across batches (like training)
                if eval_hidden is None:
                    hidden = get_hidden(batch_size)
                else:
                    # Use carried-over hidden state (detached to prevent gradients)
                    hidden = eval_hidden
            else:
                # Non-streaming evaluation: reset hidden state for each batch
                hidden = get_hidden(batch_size)
            outputs, new_hidden = model(inputs, hidden)

            # Update hidden state for streaming evaluation
            if use_eval_streaming:
                # Detach hidden state to prevent gradient flow but keep for next batch
                eval_hidden = tuple(h.detach() for h in new_hidden)

            # Reshape for loss calculation
            outputs = outputs.view(-1, get_vocab_size())
            targets = targets.view(-1)

            # Criterion now uses reduction='sum', so we get total loss for the batch
            loss_sum = criterion(outputs, targets)
            total_loss += loss_sum.item()
            total_tokens += tokens_in_batch
            total_batches += 1

    if total_tokens == 0:
        return float("nan")
    # Return per-token loss for consistent logging
    return total_loss / total_tokens


def evaluate_model_amp(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    config: Dict = None,
) -> float:
    """Evaluate model with mixed precision support"""
    model.eval()
    total_loss = 0
    total_batches = 0

    # Helper function to get hidden state
    def get_hidden(batch_size):
        return model.init_hidden(batch_size, device)

    # Helper function to get vocab size
    def get_vocab_size():
        return model.vocab_size

    # Handle streaming evaluation policy
    use_streaming = config.get("use_streaming", False) if config else False
    eval_streaming_like_train = (
        config.get("eval_streaming_like_train", True) if config else True
    )
    use_eval_streaming = use_streaming and eval_streaming_like_train

    eval_hidden = None
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Handle tensor shapes - if 3D, squeeze first dimension (streaming mode)
            if inputs.dim() == 3:
                inputs = inputs.squeeze(0)
                targets = targets.squeeze(0)

            batch_size, sequence_length = inputs.size()
            tokens_in_batch = batch_size * sequence_length

            # Handle hidden state based on eval streaming policy
            if use_eval_streaming:
                # Streaming evaluation: carry hidden state across batches (like training)
                if eval_hidden is None:
                    hidden = get_hidden(batch_size)
                else:
                    # Use carried-over hidden state (detached to prevent gradients)
                    hidden = eval_hidden
            else:
                # Non-streaming evaluation: reset hidden state for each batch
                hidden = get_hidden(batch_size)

            if use_amp:
                with autocast():
                    outputs, new_hidden = model(inputs, hidden)
                    outputs = outputs.view(-1, get_vocab_size())
                    targets = targets.view(-1)
                    loss_sum = criterion(outputs, targets)
            else:
                outputs, new_hidden = model(inputs, hidden)
                outputs = outputs.view(-1, get_vocab_size())
                targets = targets.view(-1)
                loss_sum = criterion(outputs, targets)

            # Update hidden state for streaming evaluation
            if use_eval_streaming:
                # Detach hidden state to prevent gradient flow but keep for next batch
                eval_hidden = tuple(h.detach() for h in new_hidden)

            total_loss += loss_sum.item()
            total_tokens += tokens_in_batch
            total_batches += 1

    if total_tokens == 0:
        return float("nan")
    # Return per-token loss for consistent logging
    return total_loss / total_tokens


def train_model(config: Dict, run_name: str = None, csv_log_path: str = None):
    """Main training function with mixed precision and gradient accumulation"""
    # Initialize wandb
    if config["wandb_offline"]:
        os.environ["WANDB_MODE"] = "offline"

    is_main_process = True  # Always main process in single-GPU training

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
        tie_embeddings=config.get("tie_embeddings", True),  # Default to True
    ).to(device)

    # Single-GPU training only
    print(f"Using single-GPU training")

    # NEW: compile the model (must be after any parallel wrappers)
    if config.get("use_compile", False):
        print("🔧 Compiling model with torch.compile() …")
        # you can pass mode/backend args here, e.g. backend="inductor", mode="max-autotune"
        model = torch.compile(model)

    # Helper function to get hidden state
    def init_hidden(batch_size):
        return model.init_hidden(batch_size, device)

    # Initialize FLOP counter
    flop_counter = FLOPCounter(model, config, preprocessor.tokenizer)

    # Loss and optimizer - use reduction='sum' for proper per-token scaling
    criterion = nn.CrossEntropyLoss(reduction="sum")

    # Select optimizer based on config
    opt_name = config.get("optimizer", "adam").lower()
    if opt_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
            eps=config.get("adam_epsilon", 1e-8),
        )
    elif opt_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.0),
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
            eps=config.get("adam_epsilon", 1e-8),
        )
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config.get("sgd_momentum", config.get("momentum", 0.9)),
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
        return model.vocab_size

    # Training loop
    optimizer_step_counter = 0

    # Initialize metrics counters
    tokens_cumulative = 0
    streaming_reset_count = 0
    effective_batch_tokens = (
        config["batch_size"] * gradient_accumulation_steps * config["sequence_length"]
    )

    # Initialize hidden state for streaming mode
    use_streaming = config.get("use_streaming", False)
    streaming_hidden = None
    streaming_reset_prob = config.get(
        "streaming_reset_prob", 0.01
    )  # 1% chance to reset

    for epoch in range(config["num_epochs"]):
        # Single-GPU training - no reshuffling needed
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_batches = 0
        optimizer.zero_grad()  # Zero gradients at the start of epoch

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Handle tensor shapes for streaming vs non-streaming datasets
            if use_streaming:
                # Streaming dataset returns [B, seq_len], DataLoader adds [1, B, seq_len]
                # So squeeze the first dimension
                inputs = inputs.squeeze(0)  # [B, seq_len]
                targets = targets.squeeze(0)  # [B, seq_len]

            batch_size, sequence_length = inputs.size()

            # Track cumulative tokens processed
            tokens_cumulative += batch_size * sequence_length

            # Handle hidden state based on streaming mode
            if use_streaming:
                # Streaming mode: carry hidden state across batches, reset occasionally
                if streaming_hidden is None:
                    # Initialize hidden state at start
                    hidden = init_hidden(batch_size)
                    streaming_hidden = hidden
                else:
                    # Use carried-over hidden state
                    hidden = streaming_hidden

                    # Optional: randomly reset hidden state with small probability
                    if torch.rand(1).item() < streaming_reset_prob:
                        print(f"  Randomly resetting hidden state at batch {batch_idx}")
                        hidden = init_hidden(batch_size)
                        streaming_reset_count += 1
            else:
                # Non-streaming baseline: reset hidden state at EVERY batch
                # This ensures we don't carry hidden state across DataLoader batches
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
                # Use truncated BPTT with proper per-token loss scaling
                tbptt_length = config.get("tbptt_length", 128)
                per_token_loss, hidden = tbptt_forward_backward(
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
                # Handle hidden state for streaming mode
                if use_streaming:
                    # In streaming mode, save the hidden state for next batch
                    streaming_hidden = hidden
                # Use per-token loss directly for logging (already properly scaled)
                loss = torch.tensor(per_token_loss, device=device)
            else:
                # Full sequence forward pass with proper per-token loss scaling
                batch_size, sequence_length = inputs.size()
                tokens_mb = batch_size * sequence_length
                scale = 1.0 / (tokens_mb * gradient_accumulation_steps)

                if use_amp:
                    with autocast():
                        outputs, hidden = model(inputs, hidden)
                        # Handle hidden state based on streaming mode
                        if use_streaming:
                            # Streaming: detach only for TBPTT (gradients within sequence)
                            # but keep for cross-batch state continuation
                            streaming_hidden = tuple(h.detach() for h in hidden)
                        else:
                            # Non-streaming: detach to prevent BPTT through all time
                            hidden = tuple(h.detach() for h in hidden)
                        outputs = outputs.view(-1, get_vocab_size())
                        targets_reshaped = targets.view(-1)
                        # Compute sum loss and scale properly
                        loss_sum = criterion(outputs, targets_reshaped)
                        scaled_loss = loss_sum * scale

                    # Mixed precision backward pass
                    scaler.scale(scaled_loss).backward()
                    # Store per-token loss for logging
                    loss = torch.tensor(loss_sum.item() / tokens_mb, device=device)
                else:
                    # Regular precision
                    outputs, hidden = model(inputs, hidden)
                    # Handle hidden state based on streaming mode
                    if use_streaming:
                        # Streaming: detach only for TBPTT (gradients within sequence)
                        # but keep for cross-batch state continuation
                        streaming_hidden = tuple(h.detach() for h in hidden)
                    else:
                        # Non-streaming: detach to prevent BPTT through all time
                        hidden = tuple(h.detach() for h in hidden)
                    outputs = outputs.view(-1, get_vocab_size())
                    targets_reshaped = targets.view(-1)
                    # Compute sum loss and scale properly
                    loss_sum = criterion(outputs, targets_reshaped)
                    scaled_loss = loss_sum * scale
                    scaled_loss.backward()
                    # Store per-token loss for logging
                    loss = torch.tensor(loss_sum.item() / tokens_mb, device=device)

            # Update epoch statistics - loss is already per-token, no need to scale
            epoch_loss += loss.item()
            epoch_batches += 1

            # Step optimizer after accumulation steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                batch_idx + 1 == len(train_loader)
            ):
                # Apply gradient clipping and optimizer step
                grad_norm_preclip = 0.0
                clipped_step = 0
                clip_val = config.get("gradient_clip_val", 1.0)

                if use_amp:
                    if config.get("use_gradient_clipping", False):
                        scaler.unscale_(optimizer)
                        grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), clip_val
                        )
                        clipped_step = int(grad_norm_preclip > clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if config.get("use_gradient_clipping", False):
                        grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), clip_val
                        )
                        clipped_step = int(grad_norm_preclip > clip_val)
                    optimizer.step()

                # Zero gradients after optimization step
                optimizer.zero_grad()

                # Update step counter and step-based scheduler
                optimizer_step_counter += 1
                if scheduler is not None and scheduler_type == "step":
                    scheduler.step(optimizer_step_counter)

                # Log high-leverage metrics to wandb after each optimizer step
                current_lr = optimizer.param_groups[0]["lr"]
                flops_per_batch = (
                    flop_counter.flops_per_batch if flop_counter.profiled else 0
                )
                total_flops = flop_counter.total_flops

                step_log_dict = {
                    "learning_rate": current_lr,
                    "train_loss_per_token": loss.item(),
                    "grad_norm_preclip": float(grad_norm_preclip),
                    "clipped_step": clipped_step,
                    "effective_batch_tokens": effective_batch_tokens,
                    "tokens_cumulative": tokens_cumulative,
                    "flops_per_batch": flops_per_batch,
                    "total_flops": total_flops,
                    "streaming_reset_count": streaming_reset_count,
                    "optimizer_step": optimizer_step_counter,
                }

                if not config.get("wandb_offline", False):
                    wandb.log(step_log_dict, step=optimizer_step_counter)

            # Log to CSV if enabled
            if csv_writer and (batch_idx + 1) % csv_log_interval == 0:
                current_step = batch_idx + 1
                current_train_loss = loss.item() * gradient_accumulation_steps

                # Temporarily switch to eval mode for validation
                model.eval()
                with torch.no_grad():
                    current_val_loss = evaluate_model_amp(
                        model, val_loader, criterion, device, use_amp, config
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
                        "val_loss_per_token": current_val_loss,
                        "validation_loss": current_val_loss,  # Keep for backwards compatibility
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
        val_loss = evaluate_model_amp(
            model, val_loader, criterion, device, use_amp, config
        )

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
    test_loss = evaluate_model_amp(
        model, test_loader, criterion, device, use_amp, config
    )
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


def benchmark_model(model: nn.Module, config: Dict, tokenizer: PreTrainedTokenizer):
    """Benchmark model inference speed"""
    device = torch.device(config["device"])
    model.eval()

    # Function to get hidden state
    def init_hidden(batch_size):
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
