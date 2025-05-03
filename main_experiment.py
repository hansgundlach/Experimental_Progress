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
import datetime
import wandb
import csv
import multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

# amp stands for automatic mixed precision
import copy
from torch.nn.functional import scaled_dot_product_attention
from torch.nn import LayerNorm, Linear, Dropout, ModuleList
from transformers import GPT2Tokenizer
import time  # Add at the top with other imports

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


class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, config=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.activation_type = config.activation
        self.dropout = nn.Dropout(dropout)

        # Choose normalization type
        norm_layer = (
            nn.RMSNorm
            if getattr(config, "norm_type", "layer") == "rms"
            else nn.LayerNorm
        )
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

        normed = self.norm1(x)
        qkv = self.qkv(normed).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2), qkv
        )

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
                    is_causal=True,
                )

        # Rest of the forward pass remains the same
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        out = x + self.dropout(self.out_proj(attn_output))
        out = out + self.dropout(self.ff(self.norm2(out)))
        return out


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

        # Add positional embeddings based on config
        self.pos_encoding = config.pos_encoding
        if self.pos_encoding == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEmbedding(hidden_dim)
        elif self.pos_encoding == "learned":
            # Max sequence length for learned positional embeddings
            max_seq_len = config.seq_length
            self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
            nn.init.normal_(self.pos_emb, std=0.01)

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

        # Apply positional encoding if configured
        if self.pos_encoding == "sinusoidal":
            x = self.pos_emb(x)
        elif self.pos_encoding == "learned":
            x = x + self.pos_emb[:, :L, :]

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


def train(gpu_id=None):

    # Get appropriate device
    device = get_device(gpu_id)
    print(f"Training on device: {device}")

    # Remove the wandb.init() call from here since it's now handled in run_experiments_on_gpu
    config = wandb.config  # This will use the existing run's config
    # random seeding to make experiments more reproducable
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Only use GradScaler when using CUDA
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

    # Dynamically set warmup_steps based on fraction of epochs
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.max_epochs
    if hasattr(config, "warmup_epochs_frac"):
        config.warmup_steps = int(total_steps * config.warmup_epochs_frac)
        print(
            f"Using {config.warmup_steps} warmup steps ({config.warmup_epochs_frac*100:.1f}% of total steps)"
        )

    # Initialize model
    model = SimpleTransformer(
        vocab_size=len(primary_tokenizer),
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        config=config,
    )

    # Apply initialization based on config.init_scheme
    if config.init_scheme == "xavier_normal":
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight, gain=1.0)

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
            T_max=config.max_epochs,
            eta_min=config.min_lr,
        )
        scheduler_type = "epoch"
    elif config.lr_schedule == "cosine_warmup":

        def warmup_cosine(epoch):
            if epoch < config.warmup_epochs:
                return epoch / config.warmup_epochs
            else:
                progress = (epoch - config.warmup_epochs) / (
                    config.max_epochs - config.warmup_epochs
                )
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
        scheduler_type = "epoch"
    elif config.lr_schedule == "inverse_sqrt":

        def inverse_sqrt_with_warmup(epoch):
            # Linear warmup for warmup_epochs
            if epoch < config.warmup_epochs:
                return float(epoch) / float(max(1, config.warmup_epochs))
            # Inverse square root decay after warmup
            # This formula ensures learning rate = 1.0 right after warmup
            # and then decays with inverse square root
            return (config.warmup_epochs**0.5) * ((epoch + 1) ** -0.5)

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=inverse_sqrt_with_warmup
        )
        scheduler_type = "epoch"
    elif config.lr_schedule == "one_cycle":
        # Calculate total steps for the scheduler
        steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
        total_steps = steps_per_epoch * config.max_epochs

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
        )
        scheduler_type = "step"
    elif config.lr_schedule == "transformer":
        # Implementation of the transformer paper's learning rate schedule
        d_model = config.hidden_dim
        warmup_steps = config.warmup_steps if hasattr(config, "warmup_steps") else 4000

        def transformer_lr_lambda(step):
            # Transformer learning rate schedule from "Attention Is All You Need"
            # lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
            step = max(1, step)  # Avoid division by zero
            return (d_model**-0.5) * min(step**-0.5, step * (warmup_steps**-1.5))

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=transformer_lr_lambda
        )
        scheduler_type = "step"
    else:
        scheduler = None
        scheduler_type = None

    # Create a descriptive name
    # Create a descriptive name with timestamp

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"{config.num_layers}L-{config.activation}-{config.pos_encoding}-{config.hidden_dim}d-{config.optimizer}-{config.lr_schedule}-{timestamp}"

    # Set the run name
    wandb.run.name = run_name
    wandb.run.save()

    best_val_loss = float("inf")
    best_model_state = None
    patience = 15  # Increased patience
    patience_counter = 0
    min_delta = 1e-4  # Minimum improvement threshold
    # min_epochs = 20  # Minimum number of epochs before early stopping

    global_step = 0

    for epoch in range(config.max_epochs):
        # Training phase
        model.train()
        total_loss = 0

        # Initialize metrics dictionary
        metrics = {
            "epoch": epoch,
            "optimizer": config.optimizer,
            "dataset": "wikitext",
        }

        # Step epoch-based scheduler at the beginning of each epoch
        if scheduler is not None and scheduler_type == "epoch":
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            metrics.update({"learning_rate": current_lr})
            wandb.log(metrics)

        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            # Mixed precision training logic...
            if use_amp:
                with autocast():
                    output = model(data)
                    output = output[:, :-1, :]
                    target = target[:, :-1]
                    output = output.contiguous().view(-1, primary_tokenizer.vocab_size)
                    target = target.contiguous().view(-1)
                    loss = criterion(output, target)
                    # Scale the loss
                    loss = loss / config.gradient_accumulation_steps

                scaler.scale(loss).backward()

                # Only update weights after accumulating gradients
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    if config.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.gradient_clip_val
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # Update step-based scheduler after optimization step
                    global_step += 1
                    if scheduler is not None and scheduler_type == "step":
                        scheduler.step()
                        current_lr = optimizer.param_groups[0]["lr"]
                        step_metrics = {
                            "learning_rate": current_lr,
                            "global_step": global_step,
                        }
                        wandb.log(step_metrics)
            else:
                output = model(data)
                output = output[:, :-1, :]
                target = target[:, :-1]
                output = output.contiguous().view(-1, primary_tokenizer.vocab_size)
                target = target.contiguous().view(-1)
                loss = criterion(output, target)
                # Scale the loss
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    if config.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.gradient_clip_val
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                    # Update step-based scheduler here too
                    global_step += 1
                    if scheduler is not None and scheduler_type == "step":
                        scheduler.step()

            total_loss += loss.item()

            # Log batch metrics
            if batch_idx % 100 == 0:
                wandb.log(
                    {
                        "batch": batch_idx + epoch * len(train_dataloader),
                        "batch_loss": loss.item(),
                        "optimizer": config.optimizer,
                    }
                )

        # Update metrics with loss after training loop
        avg_loss = total_loss / len(train_dataloader)
        metrics.update({"train_loss": avg_loss})

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                output = (
                    output[:, :-1, :]
                    .contiguous()
                    .view(-1, primary_tokenizer.vocab_size)
                )
                target = target[:, :-1].contiguous().view(-1)
                val_loss += criterion(output, target).item()

        val_loss /= len(val_dataloader)
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
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")

    # At the end of training, return the final values
    return {"val_loss": val_loss, "best_val_loss": best_val_loss}


def run_experiments_on_gpu(gpu_id, experiments):
    print(f"\nGPU {gpu_id} STARTING:")
    print(f"Number of experiments on GPU: {len(experiments)}")
    print("Experiments:", [(exp[parameter], exp["seed"]) for exp in experiments])
    start_time = time.time()
    results = {}

    for exp in experiments:
        try:
            with wandb.init(
                project=project_name, config={**base_config, **exp}, reinit=True
            ) as run:
                training_results = train(gpu_id=gpu_id)

                # Use the current parameter type instead of hardcoding "activation"
                param_value = exp[parameter]
                seed = exp["seed"]
                if param_value not in results:
                    results[param_value] = {}
                results[param_value][seed] = {
                    "final_loss": training_results["val_loss"],
                    "best_loss": training_results["best_val_loss"],
                }

                wandb.log(
                    {
                        "final_val_loss": training_results["val_loss"],
                        "best_val_loss": training_results["best_val_loss"],
                    }
                )
                run.finish()

                print(f"Completed experiment: {param_value} seed {seed}")
                print(
                    f"Results: final_loss={training_results['val_loss']:.4f}, best_loss={training_results['best_val_loss']:.4f}"
                )

        except Exception as e:
            print(f"Error in experiment {exp} on GPU {gpu_id}: {str(e)}")
            param_value = exp[parameter]
            seed = exp["seed"]
            if param_value not in results:
                results[param_value] = {}
            results[param_value][seed] = {
                "final_loss": float("nan"),
                "best_loss": float("nan"),
            }

    elapsed = time.time() - start_time
    print(f"GPU {gpu_id} completed all experiments in {elapsed:.2f} seconds")
    return results


class TransformerDataset(Dataset):
    def __init__(self, sequences, targets):
        """
        Args:
            sequences: Tensor of input sequences
            targets: Tensor of target sequences
        """
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Ensure idx is an integer
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self.sequences[idx], self.targets[idx]


def get_dataset(config):
    """Load and prepare dataset using PyTorch Dataset"""
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

    # Split into train/val (90/10)
    split_point = int(len(text) * 0.9)
    train_text = text[:split_point]
    val_text = text[split_point:]

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


# def setup_experiment_configs():
#     comparison_setup = {
#         "parameter": "activation",
#         "options": ["gelu", "relu", "swiglu"],
#         "base_changes": {
#             "gelu": {"activation": "gelu"},
#             "relu": {"activation": "relu"},
#             "swiglu": {"activation": "swiglu"},
#             "glu": {"activation": "glu"},
#         },
#     }
#     return comparison_setup


if __name__ == "__main__":

    wandb.login()
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    project_name = f"transformer_experiments_{timestamp}"

    # Detect available compute resources
    n_gpus = torch.cuda.device_count()
    # Base configuration for all experiments
    base_config = {
        "dataset": "wikitext",
        "batch_size": 64,
        "learning_rate": 0.0005,
        "min_lr": 0.0001,
        "lr_schedule": "inverse_sqrt",  # Options: "cosine", "cosine_warmup", "inverse_sqrt", "one_cycle", "transformer"
        "warmup_epochs": 3,  # For "cosine_warmup" and "inverse_sqrt"
        "warmup_epochs_frac": 0.2,  # 20% of total epochs as warmup
        "pct_start": 0.3,  # For "one_cycle" - percentage of training spent in warmup phase
        "weight_decay": 0.05,
        "hidden_dim": 128,
        "num_layers": 8,
        "num_heads": 8,
        "dropout": 0.2,
        "seq_length": 128,
        "wikitext_limit": 1000000,
        "pos_encoding": "sinusoidal",
        "init_scheme": "transformer_scaled",
        "stride": 64,
        "pin_memory": True,
        "compile": False,
        "prefetch_factor": 8,
        "min_epochs": 15,
        "max_epochs": 15,
        "use_gradient_clipping": True,
        "gradient_clip_val": 0.5,
        "label_smoothing": 0.1,
        "gradient_accumulation_steps": 4,
        "optimizer": "adamw",
        "activation": "gelu",  # Default activation choices are gelu, relu, glu, swiglu
        "norm_type": "layer",  # Options: "layer" or "rms"
    }

    # Setup experiments
    seeds = [42, 123, 789, 1000]

    # comparing activation functions
    comparison_activation = {
        "parameter": "activation",
        "options": ["gelu", "relu", "swiglu"],
        "base_changes": {
            "gelu": {"activation": "gelu"},
            "relu": {"activation": "relu"},
            "swiglu": {"activation": "swiglu"},
            "glu": {"activation": "glu"},
        },
    }
    # comparing lr_schedulers
    comparison_lr_schedule = {
        "parameter": "lr_schedule",
        "options": [
            "cosine",
            "cosine_warmup",
            "inverse_sqrt",
            "one_cycle",
            "transformer",
        ],
        "base_changes": {
            "cosine": {"lr_schedule": "cosine"},
            "cosine_warmup": {"lr_schedule": "cosine_warmup"},
            "inverse_sqrt": {"lr_schedule": "inverse_sqrt"},
            "one_cycle": {"lr_schedule": "one_cycle"},
            "transformer": {"lr_schedule": "transformer"},
        },
    }
    comparison_optimizer = {
        "parameter": "optimizer",
        "options": ["adamw", "adam", "sgd"],
        "base_changes": {
            "adamw": {"optimizer": "adamw"},
            "adam": {"optimizer": "adam"},
            "sgd": {"optimizer": "sgd"},
        },
    }
    comparison_init_scheme = {
        "parameter": "init_scheme",
        "options": ["transformer_scaled", "transformer_uniform", "xavier_uniform"],
        "base_changes": {
            "transformer_scaled": {"init_scheme": "transformer_scaled"},
            "transformer_uniform": {"init_scheme": "transformer_uniform"},
            "xavier_uniform": {"init_scheme": "xavier_uniform"},
        },
    }
    comparison_gradient_clipping = {
        "parameter": "use_gradient_clipping",
        "options": [True, False],
        "base_changes": {
            True: {"use_gradient_clipping": True},
            False: {"use_gradient_clipping": False},
        },
    }
    comparison_dropout = {
        "parameter": "dropout",
        "options": [0.0, 0.1],
        "base_changes": {
            "0.0": {"dropout": 0.0},
            "0.1": {"dropout": 0.1},
        },
    }
    comparison = comparison_activation
    parameter = comparison["parameter"]
    options = comparison["options"]
    base_changes = comparison["base_changes"]

    experiments = []
    for seed in seeds:
        for option in options:
            exp_config = {
                "seed": seed,
                **base_changes[option],
            }
            experiments.append(exp_config)

    final_losses = {option: {} for option in options}

    if n_gpus > 1:
        total_start_time = time.time()
        print(f"\nRunning {len(experiments)} total experiments across {n_gpus} GPUs")
        processes = []
        results_queue = mp.Queue()
        experiments_per_gpu = len(experiments) // n_gpus

        # Track which GPUs are assigned what
        for gpu_id in range(n_gpus):
            start_idx = gpu_id * experiments_per_gpu
            end_idx = (
                start_idx + experiments_per_gpu
                if gpu_id < n_gpus - 1
                else len(experiments)
            )
            gpu_experiments = experiments[start_idx:end_idx]
            print(f"\nGPU {gpu_id} assigned experiments {start_idx} to {end_idx-1}:")
            print(
                f"Experiments: {[(exp[parameter], exp['seed']) for exp in gpu_experiments]}"
            )

            p = mp.Process(
                target=lambda q, gid, exps: q.put(
                    (gid, run_experiments_on_gpu(gid, exps))
                ),  # Include GPU ID in results
                args=(results_queue, gpu_id, gpu_experiments),
            )
            p.daemon = False
            processes.append(p)
            p.start()

        # Collect results from all processes
        final_losses = {option: {} for option in options}
        print("\nCollecting results from GPUs:")
        for _ in range(len(processes)):
            gpu_id, gpu_results = results_queue.get()  # Unpack the tuple correctly
            print(f"\nReceived results from GPU {gpu_id}:")
            # gpu_results is the dictionary we want to process
            for act_type, seeds_dict in gpu_results.items():  # Process the results dict
                if act_type not in final_losses:
                    final_losses[act_type] = {}
                final_losses[act_type].update(seeds_dict)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        total_elapsed = time.time() - total_start_time
        print(
            f"\nAll experiments completed in {total_elapsed:.2f} seconds"
        )  # Add timing

    elif torch.cuda.is_available():
        # Single GPU setup
        print("Running on single GPU")
        device = torch.device("cuda:0")
        for exp in experiments:
            config = {**base_config, **exp}
            with wandb.init(project=project_name, config=config):
                train(gpu_id=0)
                final_loss = wandb.run.summary["val_loss"]
                best_val_loss = wandb.run.summary["best_val_loss"]
                final_losses[exp[parameter]][exp["seed"]] = {
                    "final_loss": final_loss,
                    "best_loss": best_val_loss,
                }

    else:
        # CPU setup
        print("Running on CPU")
        for exp in experiments:
            config = {**base_config, **exp}
            with wandb.init(project=project_name, config=config):
                train()
                final_loss = wandb.run.summary["val_loss"]
                best_val_loss = wandb.run.summary["best_val_loss"]
                final_losses[exp[parameter]][exp["seed"]] = {
                    "final_loss": final_loss,
                    "best_loss": best_val_loss,
                }

    # Modified CSV output
    csv_data = [[parameter, "Seed", "Final Val Loss", "Best Val Loss"]]
    param_losses = {option: {"final": [], "best": []} for option in options}

    for option in options:
        for seed in seeds:
            result = final_losses[option].get(seed)
            if result is not None:
                final_loss = result["final_loss"]
                best_loss = result["best_loss"]
                param_losses[option]["final"].append(final_loss)
                param_losses[option]["best"].append(best_loss)
                csv_data.append(
                    [
                        option,
                        seed,
                        f"{final_loss:.4f}",
                        f"{best_loss:.4f}",
                    ]
                )

        # Add summary statistics
        if param_losses[option]["final"]:
            mean_final = np.mean(param_losses[option]["final"])
            std_final = np.std(param_losses[option]["final"])
            mean_best = np.mean(param_losses[option]["best"])
            std_best = np.std(param_losses[option]["best"])
            csv_data.append(
                [
                    f"{option}_summary",
                    "N/A",
                    f"{mean_final:.4f} ± {std_final:.4f}",
                    f"{mean_best:.4f} ± {std_best:.4f}",
                ]
            )

    # Save results with parameter type in filename
    csv_file_path = f"experiment_results_{parameter}_{timestamp}.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"\nResults saved to {csv_file_path}")

    # Print summary statistics
    print(f"\nComparing different {parameter} values:")
    for option in options:
        final_loss_values = [
            result["final_loss"] for result in final_losses[option].values()
        ]
        best_loss_values = [
            result["best_loss"] for result in final_losses[option].values()
        ]

        if final_loss_values:
            mean_final = np.mean(final_loss_values)
            std_final = np.std(final_loss_values)
            mean_best = np.mean(best_loss_values)
            std_best = np.std(best_loss_values)
            print(f"{option}:")  # Remove .upper() to preserve exact parameter values
            print(f"  Final: {mean_final:.4f} ± {std_final:.4f}")
            print(f"  Best:  {mean_best:.4f} ± {std_best:.4f}")

    print("\nTOTAL EXPERIMENTS CREATED:")
    print(f"Number of experiments: {len(experiments)}")
    print("Experiments:", [(exp[parameter], exp["seed"]) for exp in experiments])
