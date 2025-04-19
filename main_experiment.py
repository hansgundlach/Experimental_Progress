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
import copy
from torch.nn.functional import scaled_dot_product_attention
from torch.nn import LayerNorm, Linear, Dropout, ModuleList

os.environ["WANDB_MODE"] = "offline"


#     return sampled_text
#  Add rotary positional encoding implementation
# class RotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_seq_len=1000):
#         super().__init__()
#         self.dim = dim
#         self.max_seq_len = max_seq_len

#         # Initialize the frequencies for different dimensions
#         freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#         positions = torch.arange(0, max_seq_len).float()

#         # Compute the sine and cosine values
#         freqs = torch.outer(positions, freqs)
#         cos = torch.cos(freqs)
#         sin = torch.sin(freqs)

#         # Store sin and cos values for later use
#         self.register_buffer("cos", cos)
#         self.register_buffer("sin", sin)

#     def forward(self, x, seq_dim=1):
#         seq_len = x.shape[seq_dim]

#         # Get the appropriate sine and cosine values for this sequence length
#         cos = self.cos[:seq_len, :]
#         sin = self.sin[:seq_len, :]

#         # Reshape for broadcasting
#         if seq_dim == 1:
#             # (batch, seq, dim) -> need cos/sin of shape (1, seq, dim)
#             cos = cos.unsqueeze(0)
#             sin = sin.unsqueeze(0)
#         else:
#             # (seq, batch, dim) -> need cos/sin of shape (seq, 1, dim)
#             cos = cos.unsqueeze(1)
#             sin = sin.unsqueeze(1)

#         return cos, sin

#     def apply_rotary_pos_emb(x, cos, sin):
#         # x shape: (batch, seq_len, dim) or (seq_len, batch, dim)
#         # Assuming dim divisible by 2
#         # Split x into even and odd dimensions
#         x_shape = x.shape
#         x = x.reshape(*x_shape[:-1], -1, 2)

#         # Apply rotation
#         x1, x2 = x[..., 0], x[..., 1]

#         if cos.dim() == 3 and x.dim() == 4:
#             # Need to reshape cos/sin to match x's dimensionality
#             cos = cos.unsqueeze(2)
#             sin = sin.unsqueeze(2)

#         # Rotate even and odd dimensions with sine and cosine
#         rotated_x = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

#         # Reshape back
#         rotated_x = rotated_x.reshape(*x_shape)
#         return rotated_x


class CharacterTokenizer:
    def __init__(self, text):
        # Get unique characters from text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Create character to index and index to character mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return "".join([self.idx_to_char[idx] for idx in indices])


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
        Improved TextDataset with concatenation, chunking, and random offsets.

        Args:
            text (str): Input text
            seq_length (int): Length of each sequence
            tokenizer: Tokenizer object with encode/decode methods
            stride (int): Stride length for sliding window (default=1)
            random_offset (bool): Whether to use random offset at start (default=True)
        """
        # Encode all text at once
        tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        # Calculate effective length that's divisible by (seq_length + 1)
        total_length = tokens.size(0)
        chunk_size = seq_length + 1  # +1 for target shift
        n_chunks = (
            total_length - 1
        ) // chunk_size  # -1 to ensure we have room for target

        # Trim to make evenly divisible
        effective_length = n_chunks * chunk_size

        # Apply random offset if requested
        if random_offset and total_length > effective_length:
            max_offset = total_length - effective_length
            offset = random.randint(0, max_offset)
            tokens = tokens[offset : offset + effective_length]
        else:
            tokens = tokens[:effective_length]

        # Reshape into chunks
        self.data = tokens.view(-1, chunk_size)

        # Calculate number of sequences based on stride
        self.stride = stride
        self.seq_length = seq_length
        self.n_sequences = (self.data.size(0) - 1) // stride + 1

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        # Calculate starting chunk and position
        chunk_idx = (idx * self.stride) // (self.seq_length + 1)

        # Get sequence and target
        sequence = self.data[chunk_idx, :-1]  # all but last token
        target = self.data[chunk_idx, 1:]  # all but first token

        return sequence, target

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


class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, use_rotary=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rotary = use_rotary

        self.qkv = Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout = Dropout(dropout)

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
                causal=False,
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
                )
            except (RuntimeError, AttributeError):
                attn_output = self.standard_attention(
                    q, k, v, dropout_p=self.dropout.p if self.training else 0.0
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
        use_rotary=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Create transformer layers
        self.layers = ModuleList(
            [
                SimpleTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_rotary=use_rotary,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(hidden_dim)
        self.fc = Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)

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
    # Initialize wandb
    wandb.init(mode="offline")
    config = wandb.config

    # Get appropriate device
    device = get_device(gpu_id)
    print(f"Training on device: {device}")

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
        prefetch_factor=2,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        collate_fn=TextDataset.collate_fn,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Initialize model
    model = SimpleTransformer(
        vocab_size=primary_tokenizer.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_rotary=config.pos_encoding == "rotary",
    )

    # Add conditional compilation for GPU
    if (
        device.type == "cuda"
        and torch.__version__ >= "2.0.0"
        and config.compile == True
    ):
        try:
            print("Compiling model for GPU acceleration...")
            model = torch.compile(model)
            print("Model compilation successful")
        except Exception as e:
            print(
                f"Model compilation failed, falling back to default model. Error: {e}"
            )

    model.to(device)

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

    # Add learning rate scheduler AFTER optimizer is created
    if config.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr,
        )
    elif config.lr_schedule == "cosine_warmup":

        def warmup_cosine(epoch):
            if epoch < config.warmup_epochs:
                return epoch / config.warmup_epochs
            else:
                progress = (epoch - config.warmup_epochs) / (
                    config.epochs - config.warmup_epochs
                )
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    else:
        scheduler = None

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

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        total_loss = 0

        # Initialize metrics dictionary at the start of each epoch
        metrics = {
            "epoch": epoch,
            "optimizer": config.optimizer,
            "dataset": "wikitext",
        }

        current_lr = optimizer.param_groups[0]["lr"]
        metrics.update({"learning_rate": current_lr})

        # Step the scheduler at the end of each epoch
        if scheduler is not None:
            scheduler.step()

        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Conditional AMP usage
            if use_amp:
                with autocast():
                    output = model(data)
                    output = output[:, :-1, :]
                    target = target[:, :-1]
                    output = output.contiguous().view(-1, primary_tokenizer.vocab_size)
                    target = target.contiguous().view(-1)
                    loss = criterion(output, target)

                # Use scaled versions
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training path for CPU/MPS
                output = model(data)
                output = output[:, :-1, :]
                target = target[:, :-1]
                output = output.contiguous().view(-1, primary_tokenizer.vocab_size)
                target = target.contiguous().view(-1)
                loss = criterion(output, target)

                # Standard backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

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
                # Use a sample from the training data instead of hardcoded text
                start_text = primary_text[:20]
                input_ids = (
                    torch.tensor(primary_tokenizer.encode(start_text))
                    .unsqueeze(0)
                    .to(device)
                )
                generated_text = start_text

                # Limit input sequence length during generation
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
                    generated_text += primary_tokenizer.decode([next_token.item()])
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                wandb.log({"generated_text": generated_text})
                print(f"\nGenerated text (epoch {epoch}):\n{generated_text}\n")

        # Update print statement to include validation loss
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Save model #not necesary for experiments
    # model_path = (
    #     f"../models/{dataset_choice}_model_{config.optimizer}_{config.activation}.pth"
    # )
    # torch.save(
    #     {
    #         "model_state_dict": model.state_dict(),
    #         "tokenizer_chars": primary_tokenizer.chars,
    #         "config": {
    #             "hidden_dim": config.hidden_dim,
    #             "num_heads": config.num_heads,
    #             "num_layers": config.num_layers,
    #             "dropout": config.dropout,
    #             "optimizer": config.optimizer,
    #             "dataset": dataset_choice,
    #         },
    #     },
    #     model_path,
    # )
    # wandb.save(model_path)


def run_experiments_on_gpu(gpu_id, experiments):
    """Run a subset of experiments on a specific GPU"""
    final_losses = {}
    for exp in experiments:
        config = {**base_config, **exp}
        with wandb.init(project=project_name, config=config):
            train(gpu_id=gpu_id)  # Pass the GPU ID to the train function
            # Change this to use validation loss instead of training loss
            final_loss = wandb.run.summary["val_loss"]  # Changed from train_loss
            best_val_loss = wandb.run.summary[
                "best_val_loss"
            ]  # Also track best validation loss
            depth = exp["num_layers"]
            optimizer = exp["optimizer"]
            seed = exp["seed"]

            if depth not in final_losses:
                final_losses[depth] = {"adam": {}, "adamw": {}}
            final_losses[depth][optimizer][seed] = {
                "final_loss": final_loss,
                "best_loss": best_val_loss,
            }

    return final_losses


def create_dataset_from_local_file(file_path, seq_length, stride=1):
    """Create sequences and targets from local text file"""
    # Read the text file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Create initial tokenization
    tokenizer = CharacterTokenizer(text)

    # Convert text to token ids
    tokens = tokenizer.encode(text)

    # Create sequences with stride
    sequences = []
    targets = []

    for i in range(0, len(tokens) - seq_length, stride):
        sequences.append(tokens[i : i + seq_length])
        targets.append(tokens[i + 1 : i + seq_length + 1])

    # Convert to torch tensors directly
    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return sequences, targets, tokenizer


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

    # Split into train/val (90/10)
    split_point = int(len(text) * 0.9)
    train_text = text[:split_point]
    val_text = text[split_point:]

    # Create tokenizer (on full text to ensure same vocabulary)
    tokenizer = CharacterTokenizer(text)

    # Create datasets
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
        stride=config.seq_length,  # Non-overlapping for validation
        random_offset=False,
    )

    return train_dataset, val_dataset, tokenizer, text


if __name__ == "__main__":
    wandb.login()
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    project_name = f"transformer_experiments_{timestamp}"

    # Detect available compute resources
    use_multi_gpu = torch.cuda.device_count() > 1
    use_mps = torch.backends.mps.is_available()

    # base_config = {
    #     "dataset": "wikitext",
    #     "batch_size": 128,
    #     "learning_rate": 0.001,
    #     "min_lr": 0.0001,
    #     "lr_schedule": "cosine_warmup",  # More sophisticated schedule
    #     "activation": "gelu",
    #     "warmup_epochs": 2,  # Add proper warmup
    #     "weight_decay": 0.1,  # Increase weight decay to make difference more visible
    #     "hidden_dim": 128,  # Larger model
    #     "num_heads": 4,
    #     "num_layers": 4,  # More layers
    #     "dropout": 0.1,
    #     "epochs": 15,  # Train longer
    #     "seq_length": 256,  # Longer sequences
    #     "wikitext_limit": 100000,  # More data
    #     "pos_encoding": "rotary",
    #     "init_scheme": "xavier_normal",  # Better initialization
    # }
    # config for fast testing
    base_config = {
        "dataset": "wikitext",
        "batch_size": 128,
        "learning_rate": 0.001,
        "min_lr": 0.0001,
        "lr_schedule": "cosine_warmup",  # More sophisticated schedule
        "activation": "gelu",
        "warmup_epochs": 5,  # Add proper warmup
        "weight_decay": 0.001,  # Increase weight decay to make difference more visible
        "hidden_dim": 128,  # Larger model
        "num_heads": 4,
        "num_layers": 4,  # More layers
        "dropout": 0.2,
        "epochs": 200,  # Train longer
        "seq_length": 128,  # Longer sequences
        "wikitext_limit": 1000000,  # More data
        "pos_encoding": "rotary",
        "init_scheme": "xavier_normal",  # Better initialization
        "stride": 16,
        "num_workers": 4,  # Adjust based on CPU cores
        "pin_memory": True,
        "compile": False,
        "min_epochs": 100,
        "max_epochs": 100,
    }

    depths = [6]
    seeds = [42, 123, 456]
    option1 = "adam"
    option2 = "adamw"
    parameter = "optimizer"
    experiments = []
    for depth in depths:
        for seed in seeds:
            experiments.append({"num_layers": depth, parameter: option1, "seed": seed})
            experiments.append({"num_layers": depth, parameter: option2, "seed": seed})

    final_losses = {depth: {option1: {}, option2: {}} for depth in depths}

    if use_multi_gpu:
        # Multi-GPU setup
        n_gpus = torch.cuda.device_count()
        experiments_per_gpu = len(experiments) // n_gpus

        with mp.Pool(n_gpus) as pool:
            gpu_assignments = []
            for i in range(n_gpus):
                start_idx = i * experiments_per_gpu
                end_idx = (
                    start_idx + experiments_per_gpu
                    if i < n_gpus - 1
                    else len(experiments)
                )
                gpu_assignments.append((i, experiments[start_idx:end_idx]))

            results = pool.starmap(run_experiments_on_gpu, gpu_assignments)

        # Combine results
        for gpu_results in results:
            for depth, optimizer_results in gpu_results.items():
                for optimizer, results in optimizer_results.items():
                    for seed, result in results.items():
                        final_losses[depth][optimizer][seed] = result

    elif torch.cuda.is_available():
        # Single GPU setup
        device = torch.device("cuda:0")
        for exp in experiments:
            config = {**base_config, **exp}
            with wandb.init(project=project_name, config=config):
                train(gpu_id=0)  # Use first GPU
                final_loss = wandb.run.summary["val_loss"]
                best_val_loss = wandb.run.summary["best_val_loss"]
                final_losses[exp["num_layers"]][exp[parameter]][exp["seed"]] = {
                    "final_loss": final_loss,
                    "best_loss": best_val_loss,
                }

    else:
        # Single device setup (CPU or MPS)
        for exp in experiments:
            config = {**base_config, **exp}
            with wandb.init(project=project_name, config=config):
                train()  # No gpu_id needed for CPU/MPS
                final_loss = wandb.run.summary["val_loss"]
                best_val_loss = wandb.run.summary["best_val_loss"]
                final_losses[exp["num_layers"]][exp[parameter]][exp["seed"]] = {
                    "final_loss": final_loss,
                    "best_loss": best_val_loss,
                }

    # Save detailed results to CSV
    csv_data = [["Depth", parameter, "Seed", "Final Val Loss", "Best Val Loss"]]

    for depth in depths:
        optimizer_losses = {
            option1: {"final": [], "best": []},
            option2: {"final": [], "best": []},
        }

        for optimizer in [option1, option2]:
            for seed in seeds:
                result = final_losses[depth][optimizer].get(seed)
                if result is not None:
                    final_loss = result["final_loss"]
                    best_loss = result["best_loss"]
                    optimizer_losses[optimizer]["final"].append(final_loss)
                    optimizer_losses[optimizer]["best"].append(best_loss)
                    csv_data.append(
                        [
                            depth,
                            optimizer,
                            seed,
                            f"{final_loss:.4f}",
                            f"{best_loss:.4f}",
                        ]
                    )

        # Add summary statistics for each optimizer
        if optimizer_losses[optimizer]["final"]:
            mean_final = np.mean(optimizer_losses[optimizer]["final"])
            std_final = np.std(optimizer_losses[optimizer]["final"])
            mean_best = np.mean(optimizer_losses[optimizer]["best"])
            std_best = np.std(optimizer_losses[optimizer]["best"])
            csv_data.append(
                [
                    depth,
                    f"{optimizer}_summary",
                    "N/A",
                    f"{mean_final:.4f} ± {std_final:.4f}",
                    f"{mean_best:.4f} ± {std_best:.4f}",
                ]
            )

    # Save to CSV #not timestamping for now
    # timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    csv_file_path = f"new_experiment_results.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"\nResults saved to {csv_file_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    for depth in depths:
        print(f"\nDepth {depth}:")
        for optimizer in [option1, option2]:
            # Change variable names to avoid conflict
            final_loss_values = [
                result["final_loss"]
                for result in final_losses[depth][optimizer].values()
            ]
            best_loss_values = [
                result["best_loss"]
                for result in final_losses[depth][optimizer].values()
            ]

            if final_loss_values:  # Check if we have any values
                mean_final = np.mean(final_loss_values)
                std_final = np.std(final_loss_values)
                mean_best = np.mean(best_loss_values)
                std_best = np.std(best_loss_values)
                print(f"{optimizer.upper()}:")
                print(f"  Final: {mean_final:.4f} ± {std_final:.4f}")
                print(f"  Best:  {mean_best:.4f} ± {std_best:.4f}")
