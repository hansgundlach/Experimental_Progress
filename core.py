import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
import numpy as np
import math
import requests
from pathlib import Path
import random
import os
import copy
import datetime
import csv
import multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import torch.profiler
from torch.nn.functional import scaled_dot_product_attention
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Dropout, ModuleList
from transformers import GPT2Tokenizer
import time
import random
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

os.environ["WANDB_MODE"] = "offline"

os.environ["WANDB_MODE"] = "offline"


def get_wikitext_data(limit=100000):
    """Load WikiText-2 dataset from local file"""
    file_path = Path("Datasets/wikitext.txt")

    if not file_path.exists():
        raise FileNotFoundError(
            "WikiText dataset file not found at Datasets/wikitext.txt. "
            "Please ensure you have downloaded and copied the dataset file."
        )

    # Load the data
    print("Loading WikiText from local file...")
    text = file_path.read_text(encoding="utf-8")

    # Ensure limit is an integer to avoid float issues with random.randint
    limit = int(limit)

    # Limit to a reasonable sample size for quick testing
    sample_size = min(limit, len(text))  # Use at most limit characters

    # Start from a random position for variety
    if len(text) > sample_size:
        start_idx = random.randint(0, len(text) - sample_size - 1)
        sampled_text = text[start_idx : start_idx + sample_size]
    else:
        sampled_text = text

    print(f"Loaded WikiText sample: {len(sampled_text)} characters")
    return sampled_text


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


def evaluate_perplexity(model, dataloader, criterion, device):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad(), autocast():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Reshape output and target for loss calculation
            output = output[:, :-1, :]  # Remove last prediction
            target = target[:, :-1]  # Remove last target token

            output = output.contiguous().view(-1, output.size(-1))
            target = target.contiguous().view(-1)

            # Calculate loss
            loss = criterion(output, target)

            # Accumulate loss and token count
            total_loss += loss.item() * target.size(0)
            total_tokens += target.size(0)

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def evaluate_loss(model, dataloader, criterion, device, vocab_size, use_amp):
    """Calculate loss on a dataset with optional AMP"""
    model.eval()
    total_loss = 0
    total_batches = 0
    with torch.no_grad(), autocast(enabled=use_amp):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output[:, :-1, :].contiguous().view(-1, vocab_size)
            target = target[:, :-1].contiguous().view(-1)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        return float("nan")

    return total_loss / total_batches


# swiglu implmentation adds extra computation so not directly comparable to gelu, relu, silu, glu
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        # Use a single linear layer for SwiGLU projections
        self.proj = nn.Linear(
            dim, hidden_dim * 2
        )  # Single projection for both gate and value
        self.act = nn.SiLU()  # Swish activation
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        # Project input and split into gate and value
        x = self.proj(x)
        gate, value = x.chunk(2, dim=-1)

        # Apply activation to gate and element-wise multiply
        x = self.act(gate) * value
        x = self.dropout(x)
        return self.to_out(x)


class GLUFeedForward(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # Project to 8*dim because GLU will halve it to 4*dim
        self.linear1 = Linear(dim, 8 * dim)
        # From 4*dim back to dim
        self.linear2 = Linear(4 * dim, dim)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        # Project up (8x)
        x = self.linear1(x)
        # Split into two halves
        a, b = x.chunk(2, dim=-1)
        # GLU operation
        x = a * torch.sigmoid(b)
        # Dropout
        x = self.dropout(x)
        # Project back to original dimension
        x = self.linear2(x)
        return x


# Add SinusoidalPositionalEmbedding class BEFORE SimpleTransformerLayer and SimpleTransformer
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional encoding to the embedding
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class RMSNorm(nn.Module):
    """Root‐Mean‐Square Layer Norm: a lightweight LayerNorm variant."""

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute RMS over last dimension
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, config=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.activation_type = config.activation
        self.dropout = nn.Dropout(dropout)
        self.use_rotary = config.pos_encoding == "rotary"
        self.norm_placement = getattr(config, "norm_placement", "pre")

        # Choose normalization type
        if getattr(config, "norm_type", "layer") == "rms":
            norm_layer = RMSNorm
        else:
            norm_layer = nn.LayerNorm

        self.norm1 = norm_layer(hidden_dim)
        self.norm2 = norm_layer(hidden_dim)

        self.qkv = Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)

        if config.activation == "swiglu":
            # SwiGLU is a complete feed-forward module
            self.ff = SwiGLU(hidden_dim, hidden_dim * 4, dropout=dropout)
        elif config.activation == "glu":
            self.ff = GLUFeedForward(hidden_dim, dropout=dropout)
        else:
            # Standard feed-forward with normal activations
            if config.activation == "gelu":
                act_fn = nn.GELU()
            elif config.activation == "relu":
                act_fn = nn.ReLU()
            elif config.activation == "silu" or config.activation == "swish":
                act_fn = nn.SiLU()
            else:
                raise ValueError(
                    f"Unsupported activation function: {config.activation}"
                )

            self.ff = nn.Sequential(
                Linear(hidden_dim, 4 * hidden_dim),
                act_fn,
                nn.Dropout(dropout),
                Linear(4 * hidden_dim, hidden_dim),
            )

        # Check if Flash Attention is available
        self.use_flash_attention = False
        if torch.cuda.is_available():
            try:
                from flash_attn import flash_attn_func

                self.flash_attn_func = flash_attn_func
                self.use_flash_attention = True
                print("Flash Attention is available and will be used on CUDA")
            except ImportError:
                print(
                    "Flash Attention not available, falling back to standard attention"
                )

        # Add rotary embedding if specified
        if self.use_rotary:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=config.seq_length, base=10000
            )

    def standard_attention(self, q, k, v, dropout_p=0.0):
        # Standard scaled dot-product attention
        B, H, L, D = q.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        attn = F.softmax(scores, dim=-1)
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, v)

    def forward(self, x):
        B, L, D = x.shape

        if self.norm_placement == "pre":
            # Pre-Normalization
            normed_x = self.norm1(x)
            qkv = self.qkv(normed_x).chunk(3, dim=-1)
        else:
            # Post-Normalization
            qkv = self.qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2), qkv
        )

        # Apply rotary position embedding if enabled
        if self.use_rotary:
            # Create position ids for the sequence
            position_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            cos, sin = self.rotary_emb(v, L)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Choose attention implementation based on device and availability
        if self.use_flash_attention and x.device.type == "cuda":
            # Reshape for Flash Attention
            q = q.transpose(1, 2)  # [B, H, L, D] -> [B, L, H, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Flash Attention expects contiguous tensors
            q, k, v = map(lambda t: t.contiguous(), (q, k, v))

            # Use Flash Attention
            attn_output = self.flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True,
            )

            # Reshape back
            attn_output = attn_output.transpose(1, 2)  # [B, L, H, D] -> [B, H, L, D]
        else:
            # Use standard attention for CPU/MPS or when Flash Attention is not available
            try:
                attn_output = scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True,
                )
            except (RuntimeError, AttributeError):
                attn_output = self.standard_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )

        # Rest of the forward pass remains the same
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)

        if self.norm_placement == "pre":
            # Pre-Normalization
            x = x + self.dropout(attn_output)
            x = x + self.dropout(self.ff(self.norm2(x)))
        else:
            # Post-Normalization
            x = self.norm1(x + self.dropout(attn_output))
            x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_heads,
        num_layers,
        dropout=0.1,
        config=None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # MODIFY THIS SECTION - Add positional embeddings based on config
        self.pos_encoding = config.pos_encoding
        if self.pos_encoding == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEmbedding(hidden_dim)
        elif self.pos_encoding == "learned":
            # Max sequence length for learned positional embeddings
            max_seq_len = config.seq_length
            self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
            nn.init.normal_(self.pos_emb, std=0.01)
        elif self.pos_encoding == "rotary":
            # Rotary embeddings are handled in the attention layers
            self.pos_emb = None
        else:
            raise ValueError(f"Unsupported positional encoding: {self.pos_encoding}")

        # Create transformer layers
        self.layers = ModuleList(
            [
                SimpleTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    config=config,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(hidden_dim)
        self.fc = Linear(hidden_dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        x = self.embedding(x)

        # MODIFY THIS SECTION - Apply positional encoding if configured
        if self.pos_encoding == "sinusoidal":
            x = self.pos_emb(x)
        elif self.pos_encoding == "learned":
            x = x + self.pos_emb[:, :L, :]
        # rotary doesn't add anything here - it's handled in attention layers

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.fc(x)
        return x


def get_device(gpu_id=None):
    """Get appropriate device based on available hardware."""
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train(gpu_id=None, csv_log_path=None):
    # Set memory optimization flags for V100
    if torch.cuda.is_available():
        # just added to let cuda optimize kernel selection
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.max_split_size_mb = 128  # Reduce fragmentation

    # Get appropriate device
    device = get_device(gpu_id)
    print(f"Training on device: {device}")

    # Remove the wandb.init() call from here since it's now handled in run_experiments_on_gpu
    config = wandb.config  # This will use the existing run's config

    # CSV Logging Setup
    csv_writer = None
    csv_file = None
    csv_log_interval = config.get("csv_log_interval")

    if csv_log_path and csv_log_interval:
        try:
            os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
            csv_file = open(csv_log_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                ["step", "training_loss", "validation_loss", "total_flops_profiler"]
            )
            csv_file.flush()  # Immediately write header to disk
            print(f"Logging training progress to {csv_log_path}")
        except (IOError, OSError) as e:
            print(
                f"Warning: Could not open {csv_log_path} for writing. CSV logging disabled. Error: {e}"
            )
            csv_writer = None

    # random seeding to make experiments more reproducable
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Add after seeding
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Only use GradScaler when using CUDA
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # Load dataset using new HF-style loading
    train_dataset, val_dataset, primary_tokenizer, primary_text = get_dataset(config)

    num_workers = min(mp.cpu_count() // 2, 8)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        collate_fn=TextDataset.collate_fn,
        persistent_workers=True,
        prefetch_factor=config.prefetch_factor,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        collate_fn=TextDataset.collate_fn,
        persistent_workers=True,
        prefetch_factor=config.prefetch_factor,
    )

    # === Calculate total steps and warmup steps for schedulers ===
    steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    total_steps = max(1, steps_per_epoch * config.max_epochs)  # safety-guard

    warmup_frac = getattr(config, "warmup_frac", 0.0)
    if not (0.0 <= warmup_frac < 1.0):
        raise ValueError("warmup_frac must be in the interval [0.0, 1.0).")

    warmup_steps = int(total_steps * warmup_frac)
    # Make sure there is at least *one* warm-up step when warmup_frac > 0
    if warmup_frac > 0.0:
        warmup_steps = max(1, warmup_steps)
        print(
            f"Using {warmup_steps} warm-up step(s) "
            f"({warmup_steps/total_steps*100:.1f}% of total steps)"
        )
    else:
        print("No warm-up will be used.")

    # Initialize model
    model = SimpleTransformer(
        vocab_size=len(primary_tokenizer),
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        config=config,
    )

    # Count non-embedding parameters for theoretical FLOPs calculation
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_embedding_params = model.embedding.weight.numel()
    if model.pos_encoding == "learned" and hasattr(model, "pos_emb"):
        num_embedding_params += model.pos_emb.numel()
    num_non_embedding_params = num_params - num_embedding_params

    # Apply initialization based on config.init_scheme
    if config.init_scheme == "xavier_uniform":
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight, gain=1.0)

    elif config.init_scheme == "kaiming_normal":
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    elif config.init_scheme == "transformer_scaled":
        # Modern transformer initialization with layer-dependent scaling
        init_scale = 0.02
        num_layers = len(model.layers)

        for i, layer in enumerate(model.layers):
            # Scale by depth for better gradient flow
            layer_scale = init_scale / math.sqrt(2.0 * num_layers)

            # Initialize attention weights
            nn.init.normal_(layer.qkv.weight, mean=0.0, std=layer_scale)
            nn.init.zeros_(layer.qkv.bias)
            nn.init.normal_(layer.out_proj.weight, mean=0.0, std=layer_scale)
            nn.init.zeros_(layer.out_proj.bias)

            # Initialize feedforward weights based on the type of the feedforward network
            if isinstance(layer.ff, SwiGLU):
                # Initialize SwiGLU weights
                nn.init.normal_(layer.ff.proj.weight, mean=0.0, std=layer_scale)
                nn.init.zeros_(layer.ff.proj.bias)
                nn.init.normal_(layer.ff.to_out.weight, mean=0.0, std=layer_scale)
                nn.init.zeros_(layer.ff.to_out.bias)
            elif isinstance(layer.ff, GLUFeedForward):
                # Initialize GLUFeedForward weights
                nn.init.normal_(layer.ff.linear1.weight, mean=0.0, std=layer_scale)
                nn.init.zeros_(layer.ff.linear1.bias)
                nn.init.normal_(layer.ff.linear2.weight, mean=0.0, std=layer_scale)
                nn.init.zeros_(layer.ff.linear2.bias)
            elif isinstance(layer.ff, nn.Sequential):
                # Initialize standard sequential feed-forward weights
                # First find the Linear layers by iterating through the sequential modules
                linear_layers = [m for m in layer.ff if isinstance(m, nn.Linear)]

                # Initialize each Linear layer found
                for linear in linear_layers:
                    nn.init.normal_(linear.weight, mean=0.0, std=layer_scale)
                    nn.init.zeros_(linear.bias)
            else:
                print(f"Warning: Unknown feed-forward network type: {type(layer.ff)}")

        # Initialize embedding with small uniform
        nn.init.normal_(model.embedding.weight, mean=0.0, std=init_scale)

        # Initialize positional embedding if using learned positional embeddings
        if model.pos_encoding == "learned" and hasattr(model, "pos_emb"):
            nn.init.normal_(model.pos_emb, mean=0.0, std=init_scale)

        # Initialize final layer with small weights
        nn.init.normal_(
            model.fc.weight, mean=0.0, std=init_scale / math.sqrt(num_layers)
        )
        nn.init.zeros_(model.fc.bias)
    elif config.init_scheme == "default":
        pass
    else:
        raise ValueError(f"Unsupported initialization scheme: {config.init_scheme}")

    # Add conditional compilation for GPU
    if (
        device.type == "cuda"
        and torch.__version__ >= "2.0.0"
        and config.compile == True
    ):
        try:
            print("Compiling model for GPU acceleration...")
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compilation successful")
        except Exception as e:
            print(
                f"Model compilation failed, falling back to default model. Error: {e}"
            )

    model.to(device)

    if config.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Choose optimizer based on config
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,  # Standard momentum value
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    # After optimizer creation, add step-based schedulers
    if config.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,  # T_max should be total steps for cosine
            eta_min=config.min_lr,
        )
        scheduler_type = "step"
    elif config.lr_schedule == "cosine_warmup":

        def warmup_cosine_step(step):
            step = max(1, step)
            # Linear warmup
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # Cosine decay
            if total_steps == warmup_steps:
                return 1.0
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_step)
        scheduler_type = "step"
    elif config.lr_schedule == "inverse_sqrt":

        def inverse_sqrt_with_warmup(step):
            step = max(1, step)
            if warmup_steps > 0:
                if step < warmup_steps:
                    return float(step) / float(warmup_steps)
                else:
                    return (float(warmup_steps) / step) ** 0.5
            else:
                return 1.0  # Constant LR multiplier if no warmup

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=inverse_sqrt_with_warmup
        )
        scheduler_type = "step"
    elif config.lr_schedule == "one_cycle":
        # Calculate total steps for the scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_frac if hasattr(config, "warmup_frac") else 0.3,
            div_factor=25,
            final_div_factor=1000,
        )
        scheduler_type = "step"
    elif config.lr_schedule == "transformer":
        # Transformer schedule from "Attention Is All You Need"
        # Requires warm-up; warmup_steps is guaranteed ≥ 1 when warmup_frac > 0.
        if warmup_steps == 0:
            raise ValueError(
                "Set warmup_frac > 0.0 when using 'transformer' LR schedule"
            )

        d_model = config.hidden_dim

        def transformer_lr_lambda(step):
            # Transformer learning rate schedule from "Attention Is All You Need"
            # lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
            step = max(1, step)  # Avoid division by zero
            return (d_model**-0.5) * min(step**-0.5, step * (warmup_steps**-1.5))

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=transformer_lr_lambda
        )
        scheduler_type = "step"
    elif config.lr_schedule == "linear_warmup":

        def linear_with_warmup(step):
            if warmup_steps > 0:
                if step < warmup_steps:
                    return float(step) / float(warmup_steps)
                else:
                    return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
            else:
                return max(0.0, (total_steps - step) / total_steps)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_with_warmup)
        scheduler_type = "step"
    else:
        scheduler = None
        scheduler_type = None

    # === Prepare profiler log directory and estimate FLOPs per training step ===
    profiler_dir = "profiler_logs"
    os.makedirs(profiler_dir, exist_ok=True)

    # === Estimate FLOPs per training step (forward+backward+optimizer) ===
    # Grab a single batch
    data_batch, target_batch = next(iter(train_dataloader))
    data_batch, target_batch = data_batch.to(device), target_batch.to(device)
    # Zero grads and profile one full train step
    optimizer.zero_grad()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
    ) as prof:
        # forward
        output = model(data_batch)
        output = output[:, :-1, :].contiguous().view(-1, primary_tokenizer.vocab_size)
        tgt = target_batch[:, :-1].contiguous().view(-1)
        loss = criterion(output, tgt)
        # backward + step
        loss.backward()
        optimizer.step()
    # Sum up the FLOPs and extrapolate to full run
    flops_per_step = sum(
        evt.flops for evt in prof.key_averages() if hasattr(evt, "flops")
    )
    steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.max_epochs
    total_flops = flops_per_step * total_steps
    flops_per_epoch = flops_per_step * steps_per_epoch
    print(f"\n==== FLOPs per step: {flops_per_step:.2e}")
    print(f"==== FLOPs per epoch: {flops_per_epoch:.2e}")
    print(
        f"==== Estimated total training FLOPs: {total_flops:.2e} ({total_steps} steps) ====\n"
    )

    # Initialize training variables
    best_val_loss = float("inf")
    patience_counter = 0
    min_delta = 1e-4
    patience = 10
    best_model_state = None

    # now continue with your normal epoch loop
    # initialize step counter before it's used below
    optimizer_step_counter = 0
    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0

        if epoch == 0:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=True,  # <-- enable FLOPS counting
            )
            profiler.start()

        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            if epoch == 0 and batch_idx == 0:
                with profiler:
                    output = model(data)
                    # drop last position, flatten for loss
                    output = (
                        output[:, :-1, :]
                        .contiguous()
                        .view(-1, primary_tokenizer.vocab_size)
                    )
                    target_flat = target[:, :-1].contiguous().view(-1)
                    raw_loss = criterion(output, target_flat)
                    scaled_loss = raw_loss / config.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()

            else:
                # Forward pass and compute raw and scaled losses
                if use_amp:
                    with autocast():
                        output = model(data)
                        # drop last position, flatten for loss
                        output = (
                            output[:, :-1, :]
                            .contiguous()
                            .view(-1, primary_tokenizer.vocab_size)
                        )
                        target_flat = target[:, :-1].contiguous().view(-1)
                        raw_loss = criterion(output, target_flat)
                        scaled_loss = raw_loss / config.gradient_accumulation_steps
                    scaler.scale(scaled_loss).backward()
                else:
                    output = model(data)
                    output = (
                        output[:, :-1, :]
                        .contiguous()
                        .view(-1, primary_tokenizer.vocab_size)
                    )
                    target_flat = target[:, :-1].contiguous().view(-1)
                    raw_loss = criterion(output, target_flat)
                    scaled_loss = raw_loss / config.gradient_accumulation_steps
                    scaled_loss.backward()

            # Only update weights after accumulating gradients
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_val
                    )
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                optimizer_step_counter += 1

                # Update step-based scheduler after optimization step
                if scheduler is not None and scheduler_type == "step":
                    scheduler.step(optimizer_step_counter)
                    current_lr = optimizer.param_groups[0]["lr"]
                    step_metrics = {
                        "learning_rate": current_lr,
                        "optimizer_step": optimizer_step_counter,
                    }
                    wandb.log(step_metrics)

                # Log to CSV if enabled (uses optimizer steps)
                if csv_writer and (optimizer_step_counter % csv_log_interval == 0):
                    current_train_loss = raw_loss.item()
                    # Run validation
                    current_val_loss = evaluate_loss(
                        model,
                        val_dataloader,
                        criterion,
                        device,
                        len(primary_tokenizer),
                        use_amp,
                    )
                    model.train()  # Switch back to train mode

                    cumulative_flops_profiler = flops_per_step * optimizer_step_counter

                    csv_writer.writerow(
                        [
                            optimizer_step_counter,
                            f"{current_train_loss:.4f}",
                            f"{current_val_loss:.4f}",
                            f"{cumulative_flops_profiler:.2e}",
                        ]
                    )
                    csv_file.flush()

            # accumulate the _raw_ loss for logging
            total_loss += raw_loss.item()

            # Log batch metrics
            if batch_idx % 100 == 0:
                wandb.log(
                    {
                        "batch": batch_idx + epoch * len(train_dataloader),
                        "batch_loss": raw_loss.item(),
                        "optimizer": config.optimizer,
                    }
                )

            # Log to CSV if enabled
            if csv_writer and (optimizer_step_counter % csv_log_interval == 0):
                current_train_loss = raw_loss.item()
                # Run validation
                current_val_loss = evaluate_loss(
                    model,
                    val_dataloader,
                    criterion,
                    device,
                    len(primary_tokenizer),
                    use_amp,
                )
                model.train()  # Switch back to train mode

                cumulative_flops_profiler = flops_per_step * optimizer_step_counter

                csv_writer.writerow(
                    [
                        optimizer_step_counter,
                        f"{current_train_loss:.4f}",
                        f"{current_val_loss:.4f}",
                        f"{cumulative_flops_profiler:.2e}",
                    ]
                )
                csv_file.flush()

        # Update metrics with loss after training loop
        avg_train_loss = total_loss / len(train_dataloader)
        metrics = {
            "epoch": epoch,
            "optimizer": config.optimizer,
            "dataset": "wikitext",
            "train_loss": avg_train_loss,
        }

        # Validation phase
        model.eval()
        val_loss = evaluate_loss(
            model, val_dataloader, criterion, device, len(primary_tokenizer), use_amp
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # don't need to save the model state dict, because we don't use it for anything
            # best_model_state = copy.deepcopy(model.state_dict())

        # Early stopping check
        if epoch >= config.min_epochs:  # Only start checking after min_epochs
            if val_loss < (best_val_loss - min_delta):  # Meaningful improvement
                # best_val_loss = val_loss
                # best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if epoch >= config.max_epochs:
                print(f"Max epochs reached at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

        metrics.update({"val_loss": val_loss, "best_val_loss": best_val_loss})
        wandb.log(metrics)

        # Step epoch-based scheduler at the END of each epoch
        if scheduler is not None and scheduler_type == "epoch":
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        # Generate sample text
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad(), autocast():
                start_text = primary_text[:50]  # Might want to use more text for BPE
                input_ids = (
                    torch.tensor(primary_tokenizer(start_text)["input_ids"])
                    .unsqueeze(0)
                    .to(device)
                )
                generated_text = start_text

                # Define max_gen_length here
                max_gen_length = (
                    config.seq_length
                )  # Use same length as training sequences

                for _ in range(100):
                    if input_ids.size(1) > max_gen_length:
                        input_ids = input_ids[:, -max_gen_length:]

                    output = model(input_ids)
                    next_token_logits = output[0, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / 0.7, dim=0), 1
                    )
                    generated_text = primary_tokenizer.decode(
                        input_ids[0].tolist()
                    )  # Decode the full sequence
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                wandb.log({"generated_text": generated_text})
                print(f"\nGenerated text (epoch {epoch}):\n{generated_text}\n")

        # Update print statement to include validation loss
        print(
            f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        if epoch == 0 and batch_idx == 0:
            profiler.step()
            profiler.stop()
            # extract and print total FLOPS for this batch
            ka = profiler.key_averages()
            total_flops = sum(evt.flops for evt in ka if hasattr(evt, "flops"))
            print(f"**** Total FLOPS for batch 0: {total_flops:,} ****")

    # At the end of training, return the final values
    # Close CSV file if it was opened
    if csv_file:
        csv_file.close()
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Theoretical FLOPs: 6 * non_embedding_params * total_tokens
    # Total tokens = total_optimizer_steps * effective_batch_size * seq_len
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    total_tokens_processed = total_steps * effective_batch_size * config.seq_length
    total_flops_theoretical = 6 * num_non_embedding_params * total_tokens_processed

    print(f"\n==== FLOPs per optimizer step (Profiler): {flops_per_step:.2e}")
    print(
        f"==== Estimated total training FLOPs (Profiler): {total_flops:.2e} ({total_steps} optimizer steps) ===="
    )
    print(
        f"==== Estimated total training FLOPs (Theoretical): {total_flops_theoretical:.2e} ====\n"
    )

    return {
        "final_train_loss": avg_train_loss,
        "final_val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "total_flops_profiler": total_flops,
        "total_flops_theoretical": total_flops_theoretical,
    }


def get_dataset(config):
    """Load and prepare dataset using PyTorch Dataset"""
    # Move this block before get_dataset() call
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Get the text data
    text = get_wikitext_data(limit=config.wikitext_limit)

    # Initialize GPT2 tokenizer from local files
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_tokenizer")
    except:
        raise FileNotFoundError(
            "GPT2 tokenizer files not found in ./gpt2_tokenizer. "
            "Please download the tokenizer files first."
        )

    # BETTER SPLIT: Random shuffle before splitting

    # Split into sentences/paragraphs first, then shuffle
    sentences = text.split("\n")
    random.shuffle(sentences)

    # Now split
    split_idx = int(len(sentences) * 0.9)
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]

    train_text = "\n".join(train_sentences)
    val_text = "\n".join(val_sentences)

    # Create datasets with GPT2 tokenizer
    train_dataset = TextDataset(
        text=train_text,
        seq_length=config.seq_length,
        tokenizer=tokenizer,
        stride=config.stride,
        random_offset=True,
    )

    val_dataset = TextDataset(
        text=val_text,
        seq_length=config.seq_length,
        tokenizer=tokenizer,
        stride=config.stride,
        random_offset=False,
    )

    return train_dataset, val_dataset, tokenizer, text
