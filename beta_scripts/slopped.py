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
from transformers import GPT2Tokenizer

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


class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, config=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.activation_type = config.activation

        self.qkv = Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network setup
        if config.activation == "swiglu":
            self.ff = SwiGLU(hidden_dim, hidden_dim * 4, dropout=dropout)
        elif config.activation == "glu":
            self.ff = GLUFeedForward(hidden_dim, dropout=dropout)
        else:
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

    def forward(self, x):
        # First norm and attention
        shortcut = x
        x = self.norm1(x)

        # Use PyTorch's attention (rotary is handled internally if enabled)
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = shortcut + x

        # Second norm and feed-forward
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = shortcut + x

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
            nn.init.normal_(layer.self_attn.in_proj_weight, mean=0.0, std=layer_scale)
            nn.init.zeros_(layer.self_attn.in_proj_bias)

            # Initialize feedforward weights based on type
            if isinstance(layer.ff, SwiGLU):
                # Initialize SwiGLU weights
                nn.init.normal_(layer.ff.proj.weight, mean=0.0, std=layer_scale)
                nn.init.zeros_(layer.ff.proj.bias)
                nn.init.normal_(layer.ff.to_out.weight, mean=0.0, std=layer_scale)
                nn.init.zeros_(layer.ff.to_out.bias)
            elif isinstance(layer.ff, GLUFeedForward):
                # Initialize GLU weights
                nn.init.normal_(layer.ff.linear1.weight, mean=0.0, std=layer_scale)
                nn.init.zeros_(layer.ff.linear1.bias)
                nn.init.normal_(layer.ff.linear2.weight, mean=0.0, std=layer_scale)
                nn.init.zeros_(layer.ff.linear2.bias)
            elif isinstance(layer.ff, nn.Sequential):
                # Initialize standard sequential feed-forward weights
                nn.init.normal_(layer.ff[0].weight, mean=0.0, std=layer_scale)
                nn.init.zeros_(layer.ff[0].bias)
                if len(layer.ff) > 2:
                    nn.init.normal_(layer.ff[2].weight, mean=0.0, std=layer_scale)
                    nn.init.zeros_(layer.ff[2].bias)
            else:
                print(f"Warning: Unknown feed-forward network type: {type(layer.ff)}")

        # Initialize embedding with small uniform
        nn.init.normal_(model.embedding.weight, mean=0.0, std=init_scale)

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

            if use_amp:
                with autocast():
                    output = model(data)
                    output = output[:, :-1, :]
                    target = target[:, :-1]
                    output = output.contiguous().view(-1, primary_tokenizer.vocab_size)
                    target = target.contiguous().view(-1)
                    loss = criterion(output, target)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_val
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                output = output[:, :-1, :]
                target = target[:, :-1]
                output = output.contiguous().view(-1, primary_tokenizer.vocab_size)
                target = target.contiguous().view(-1)
                loss = criterion(output, target)

                loss.backward()
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_val
                    )
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


# old character level tokenizatio form
# def create_dataset_from_local_file(file_path, seq_length, stride=1):
#     """Create sequences and targets from local text file"""
#     # Read the text file
#     with open(file_path, "r", encoding="utf-8") as f:
#         text = f.read()

#     # Create initial tokenization
#     tokenizer = CharacterTokenizer(text)

#     # Convert text to token ids
#     tokens = tokenizer.encode(text)

#     # Create sequences with stride
#     sequences = []
#     targets = []

#     for i in range(0, len(tokens) - seq_length, stride):
#         sequences.append(tokens[i : i + seq_length])
#         targets.append(tokens[i + 1 : i + seq_length + 1])

#     # Convert to torch tensors directly
#     sequences = torch.tensor(sequences, dtype=torch.long)
#     targets = torch.tensor(targets, dtype=torch.long)

#     return sequences, targets, tokenizer


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
        "batch_size": 64,
        "learning_rate": 0.0001,
        "min_lr": 0.00001,
        "lr_schedule": "inverse_sqrt",  # More sophisticated schedule
        "activation": "gelu",
        "warmup_epochs": 5,  # Add proper warmup
        "weight_decay": 0.05,  # Increase weight decay to make difference more visible
        "hidden_dim": 128,  # Might want larger model for BPE
        "num_layers": 8,
        "num_heads": 8,
        "dropout": 0.2,
        "epochs": 200,  # Train longer
        "seq_length": 128,  # Might want longer sequences for BPE
        "wikitext_limit": 1000000,  # More data
        "pos_encoding": "none",  # Changed from "rotary" to "none"
        "init_scheme": "transformer_scaled",  # Better initialization
        "stride": 64,
        "num_workers": 4,  # Adjust based on CPU cores
        "pin_memory": True,
        "compile": False,
        "min_epochs": 50,
        "max_epochs": 50,
        "use_gradient_clipping": True,
        "gradient_clip_val": 0.5,
    }
    depths = [8]

    # Replace the optimizer-specific setup with a more general comparison setup
    def setup_experiment_configs():
        # Define what you want to compare
        comparison_setup = {
            "parameter": "activation",  # What parameter you're varying
            "options": ["glu", "gelu", "relu"],  # The values to compare
            "base_changes": {  # Any changes needed to base_config for each option
                "glu": {"activation": "glu"},
                "gelu": {"activation": "gelu"},
                "relu": {"activation": "relu"},
            },
            "required_base_params": {  # Add any required parameters that aren't being compared
                "optimizer": "adamw"  # Default optimizer
            },
        }
        return comparison_setup

    # Modified experiment generation
    # depths = [4]
    seeds = [42, 123]
    comparison = setup_experiment_configs()
    parameter = comparison["parameter"]
    options = comparison["options"]
    base_changes = comparison["base_changes"]

    experiments = []
    # for depth in depths:
    for seed in seeds:
        for option in options:
            exp_config = {
                "seed": seed,
                **comparison["required_base_params"],  # Add required parameters
                **base_changes[option],  # Add the varying parameter
            }
            experiments.append(exp_config)
    # Modified results storage
    final_losses = {option: {} for option in options}

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
            for optimizer, results in gpu_results.items():
                for seed, result in results.items():
                    final_losses[optimizer][seed] = result

    elif torch.cuda.is_available():
        # Single GPU setup
        device = torch.device("cuda:0")
        for exp in experiments:
            config = {**base_config, **exp}
            with wandb.init(project=project_name, config=config):
                train(gpu_id=0)  # Use first GPU
                final_loss = wandb.run.summary["val_loss"]
                best_val_loss = wandb.run.summary["best_val_loss"]
                final_losses[exp[parameter]][exp["seed"]] = {
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
                final_losses[exp[parameter]][exp["seed"]] = {
                    "final_loss": final_loss,
                    "best_loss": best_val_loss,
                }

    # Modified CSV output
    csv_data = [[parameter, "Seed", "Final Val Loss", "Best Val Loss"]]

    optimizer_losses = {option: {"final": [], "best": []} for option in options}

    for option in options:
        for seed in seeds:
            result = final_losses[option].get(seed)
            if result is not None:
                final_loss = result["final_loss"]
                best_loss = result["best_loss"]
                optimizer_losses[option]["final"].append(final_loss)
                optimizer_losses[option]["best"].append(best_loss)
                csv_data.append(
                    [
                        option,
                        seed,
                        f"{final_loss:.4f}",
                        f"{best_loss:.4f}",
                    ]
                )

        # Add summary statistics
        if optimizer_losses[option]["final"]:
            mean_final = np.mean(optimizer_losses[option]["final"])
            std_final = np.std(optimizer_losses[option]["final"])
            mean_best = np.mean(optimizer_losses[option]["best"])
            std_best = np.std(optimizer_losses[option]["best"])
            csv_data.append(
                [
                    f"{option}_summary",
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
            print(f"{option.upper()}:")
            print(f"  Final: {mean_final:.4f} ± {std_final:.4f}")
            print(f"  Best:  {mean_best:.4f} ± {std_best:.4f}")


# Add SinusoidalPositionalEmbedding class
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
