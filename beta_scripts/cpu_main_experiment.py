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

os.environ["WANDB_MODE"] = "offline"

# Add WikiText download and processing functions
# def get_wikitext_data():
#     """Download and prepare WikiText-2 dataset"""
#     file_path = Path("Datasets/wikitext-2-raw-v1/wiki.train.raw")

#     if not file_path.exists():
#         # Create directory if it doesn't exist
#         file_path.parent.mkdir(parents=True, exist_ok=True)

#         # Download WikiText-2 data
#         url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
#         print(f"Downloading WikiText-2 dataset from {url}")

#         import requests
#         import zipfile
#         import io

#         response = requests.get(url)
#         z = zipfile.ZipFile(io.BytesIO(response.content))
#         z.extractall("Datasets/")

#         print("Downloaded and extracted WikiText-2 dataset")

#     # Load the data
#     text = file_path.read_text(encoding='utf-8')

#     # Limit to a reasonable sample size for quick testing
#     sample_size = min(100000, len(text))  # Use at most 100k characters

#     # Start from a random position for variety
#     if len(text) > sample_size:
#         start_idx = random.randint(0, len(text) - sample_size - 1)
#         sampled_text = text[start_idx:start_idx + sample_size]
#     else:
#         sampled_text = text


#     return sampled_text
#  Add rotary positional encoding implementation
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Initialize the frequencies for different dimensions
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(0, max_seq_len).float()

        # Compute the sine and cosine values
        freqs = torch.outer(positions, freqs)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Store sin and cos values for later use
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]

        # Get the appropriate sine and cosine values for this sequence length
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]

        # Reshape for broadcasting
        if seq_dim == 1:
            # (batch, seq, dim) -> need cos/sin of shape (1, seq, dim)
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        else:
            # (seq, batch, dim) -> need cos/sin of shape (seq, 1, dim)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        return cos, sin

    def apply_rotary_pos_emb(x, cos, sin):
        # x shape: (batch, seq_len, dim) or (seq_len, batch, dim)
        # Assuming dim divisible by 2
        # Split x into even and odd dimensions
        x_shape = x.shape
        x = x.reshape(*x_shape[:-1], -1, 2)

        # Apply rotation
        x1, x2 = x[..., 0], x[..., 1]

        if cos.dim() == 3 and x.dim() == 4:
            # Need to reshape cos/sin to match x's dimensionality
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)

        # Rotate even and odd dimensions with sine and cosine
        rotated_x = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        # Reshape back
        rotated_x = rotated_x.reshape(*x_shape)
        return rotated_x


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
    def __init__(self, text, seq_length, tokenizer):
        self.text = text
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_length - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return x, y


def evaluate_perplexity(model, dataloader, criterion, device):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
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


# class TransformerModel(nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         hidden_dim,
#         num_heads,
#         num_layers,
#         dropout,
#         activation="gelu",
#         pos_encoding="rotary",
#         max_seq_length=1000,
#     ):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, hidden_dim)
#         self.pos_encoding_type = pos_encoding

#         # Positional encoding setup
#         if pos_encoding == "sinusoidal":
#             # Standard sinusoidal positional encoding
#             position = torch.arange(max_seq_length).unsqueeze(1)
#             div_term = torch.exp(
#                 torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
#             )
#             pe = torch.zeros(max_seq_length, hidden_dim)
#             pe[:, 0::2] = torch.sin(position * div_term)
#             pe[:, 1::2] = torch.cos(position * div_term)
#             self.register_buffer("pe", pe)
#             self.use_rotary = False
#         elif pos_encoding == "rotary":
#             # Rotary positional encoding
#             self.rotary_emb = RotaryEmbedding(hidden_dim, max_seq_length)
#             self.use_rotary = True
#         else:
#             raise ValueError(f"Unsupported positional encoding: {pos_encoding}")

#         self.dropout = nn.Dropout(dropout)


#         # Custom positional encoding
#         position = torch.arange(max_seq_length).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
#         )
#         pe = torch.zeros(max_seq_length, hidden_dim)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe)


#         # Use GELU as default activation
#         if activation == "gelu":
#             activation = nn.GELU()
#         elif activation == "relu":
#             activation = nn.ReLU()
#         elif activation == "swish":
#             activation = nn.SiLU()
#         else:
#             raise ValueError(f"Unsupported activation: {activation}")

#         encoder_layers = nn.TransformerEncoderLayer(
#             hidden_dim,
#             num_heads,
#             hidden_dim * 4,
#             dropout,
#             activation=activation,
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
#         self.fc = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, src):
#         # Embed the input and add positional encoding
#         src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
#         src = src + self.pe[: src.size(1), :]
#         src = self.dropout(src)

#         # Transform for transformer (seq_len, batch, hidden_dim)
#         src = src.transpose(0, 1)

#         # Pass through transformer
#         output = self.transformer_encoder(src)

#         # Transform back (batch, seq_len, hidden_dim)
#         output = output.transpose(0, 1)

#         # Project to vocabulary size
#         output = self.fc(output)

#         return output


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_heads,
        num_layers,
        dropout,
        activation="gelu",
        pos_encoding="sinusoidal",
        max_seq_length=1000,
        weight_init="default",
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding_type = pos_encoding

        # Move method outside of __init__
        self._initialize_weights(weight_init)

        # Positional encoding setup
        if pos_encoding == "sinusoidal":
            # Standard sinusoidal positional encoding
            position = torch.arange(max_seq_length).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
            )
            pe = torch.zeros(max_seq_length, hidden_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)
            self.use_rotary = False
        elif pos_encoding == "rotary":
            # Rotary positional encoding
            self.rotary_emb = RotaryEmbedding(hidden_dim, max_seq_length)
            self.use_rotary = True
        else:
            raise ValueError(f"Unsupported positional encoding: {pos_encoding}")

        self.dropout = nn.Dropout(dropout)

        # Activation function selection
        if activation == "gelu":
            activation_fn = nn.GELU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "swish":
            activation_fn = nn.SiLU()
        elif activation == "leaky_relu":
            activation_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Custom transformer layer with RoPE support if needed
        if self.use_rotary:
            # We'll need to modify the attention mechanism for RoPE
            # This is a simplified implementation - a full one would modify the
            # MultiheadAttention class to apply rotary embeddings
            # directly to Q and K matrices before computing attention

            # In this simplified version, we apply dropout and normalization
            # similar to the TransformerEncoderLayer but inject the rotary
            # position encoding in between

            encoder_layers = []
            for _ in range(num_layers):
                layer = nn.TransformerEncoderLayer(
                    hidden_dim,
                    num_heads,
                    hidden_dim * 4,
                    dropout,
                    activation=activation_fn,
                )
                encoder_layers.append(layer)
            self.transformer_layers = nn.ModuleList(encoder_layers)
        else:
            # Standard implementation
            encoder_layers = nn.TransformerEncoderLayer(
                hidden_dim,
                num_heads,
                hidden_dim * 4,
                dropout,
                activation=activation_fn,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def _initialize_weights(self, scheme):
        """Initialize network weights based on specified scheme"""
        if scheme == "default":
            return  # PyTorch default initialization

        for name, p in self.named_parameters():
            if "weight" in name:
                if scheme == "xavier_uniform":
                    nn.init.xavier_uniform_(p)
                elif scheme == "xavier_normal":
                    nn.init.xavier_normal_(p)
                elif scheme == "kaiming_uniform":
                    nn.init.kaiming_uniform_(p)
                elif scheme == "kaiming_normal":
                    nn.init.kaiming_normal_(p)
                elif scheme == "orthogonal":
                    nn.init.orthogonal_(p)
                else:
                    raise ValueError(f"Unsupported initialization scheme: {scheme}")
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, src):
        # Embed the input
        src = self.embedding(src) * math.sqrt(self.hidden_dim)

        # Apply positional encoding
        if not self.use_rotary:
            # Standard sinusoidal positional encoding
            src = src + self.pe[: src.size(1), :]
            src = self.dropout(src)

            # Transform for transformer (seq_len, batch, hidden_dim)
            src = src.transpose(0, 1)

            # Pass through transformer
            output = self.transformer_encoder(src)
        else:
            # Rotary positional encoding
            src = self.dropout(src)

            # Transform for transformer (seq_len, batch, hidden_dim)
            src = src.transpose(0, 1)

            # Get rotary embeddings
            cos, sin = self.rotary_emb(src)

            # For each layer, manually apply RoPE to the queries and keys
            output = src
            for layer in self.transformer_layers:
                # This is a simplified approach - a complete implementation
                # would modify the MultiheadAttention module to apply RoPE
                # directly to Q and K matrices before computing attention

                # In this simplified version, we apply dropout and normalization
                # similar to the TransformerEncoderLayer but inject the rotary
                # position encoding in between

                # Self-attention with RoPE
                # First get the residual
                residual = output

                # Apply the first normalization
                output = layer.norm1(output)

                # Here we would normally compute self-attention, but instead
                # we're simulating it with rotary embeddings
                # Apply rotary embeddings to the whole tensor (simplified)
                rotary_output = RotaryEmbedding.apply_rotary_pos_emb(output, cos, sin)

                # Continue with the rest of the self-attention computation
                # Since we can't modify PyTorch's built-in MultiheadAttention,
                # we use the original layer's self-attention but with our rotary-encoded inputs
                output = layer.self_attn(rotary_output, rotary_output, rotary_output)[0]
                output = residual + layer.dropout1(output)

                # Feed-forward part is unchanged
                residual = output
                output = layer.norm2(output)
                output = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(output)))
                )
                output = residual + layer.dropout2(output)

        # Transform back (batch, seq_len, hidden_dim)
        output = output.transpose(0, 1)

        # Project to vocabulary size
        output = self.fc(output)

        return output


def train():
    # Initialize wandb
    wandb.init(mode="offline")
    config = wandb.config

    # Simplified dataset handling - only wikitext
    wikitext_data = get_wikitext_data(limit=config.wikitext_limit)
    primary_tokenizer = CharacterTokenizer(wikitext_data)
    primary_text = wikitext_data

    # Create dataset for training
    seq_length = config.seq_length
    train_dataset = TextDataset(primary_text, seq_length, primary_tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )

    # Initialize model
    model = TransformerModel(
        vocab_size=primary_tokenizer.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        activation=config.activation,
        max_seq_length=seq_length,
        weight_init=config.init_scheme,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS available, using MPS")
    else:
        print("No GPU available or MPS not available, using CPU")
        device = torch.device("cpu")

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
    run_name = f"{config.activation}-{config.pos_encoding}-{config.hidden_dim}d-{config.optimizer}-{config.lr_schedule}-{timestamp}"

    # Set the run name
    wandb.run.name = run_name
    wandb.run.save()

    for epoch in range(config.epochs):
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

            # Forward pass
            output = model(data)

            # Reshape output and target for loss calculation
            output = output[:, :-1, :]  # Remove last prediction
            target = target[:, :-1]  # Remove last target token

            output = output.contiguous().view(-1, primary_tokenizer.vocab_size)
            target = target.contiguous().view(-1)

            # Calculate loss
            loss = criterion(output, target)
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

        wandb.log(metrics)

        # Generate sample text
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                start_text = "The article discusses "
                input_ids = (
                    torch.tensor(primary_tokenizer.encode(start_text))
                    .unsqueeze(0)
                    .to(device)
                )
                generated_text = start_text

                # Limit input sequence length during generation
                max_gen_length = seq_length  # Use same length as training sequences

                for _ in range(100):
                    # Trim input_ids if it gets too long
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

        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

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


if __name__ == "__main__":
    wandb.login()
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    project_name = f"transformer_experiments_{timestamp}"

    # Define sweep configuration with dataset options
    # longer more accurate benchmark
    # sweep_config = {
    #     "method": "grid",
    #     "parameters": {
    #         "dataset": {"values": ["wikitext"]},
    #         "optimizer": {"values": ["adam"]},
    #         "activation": {"values": ["relu", "gelu", "swish"]},
    #         "batch_size": {"value": 64},
    #         "learning_rate": {"value": 0.001},
    #         "weight_decay": {"value": 0.01},  # Only used for AdamW
    #         "hidden_dim": {"value": 256},
    #         "num_heads": {"value": 8},
    #         "num_layers": {"value": 4},
    #         "dropout": {"value": 0.2},
    #         "epochs": {"value": 30},
    #         "seq_length": {"value": 64},
    #         "shakespeare_limit": {"value": 10000},  # Limit Shakespeare dataset size
    #     },
    # }

    # Configuration for small models/datasets for quick testing
    sweep_config = {
        "method": "grid",
        "parameters": {
            "dataset": {"values": ["wikitext"]},
            "optimizer": {"values": ["adam"]},
            "activation": {"values": ["gelu"]},
            "pos_encoding": {"values": ["sinusoidal", "rotary"]},
            "batch_size": {"value": 32},  # Smaller batch size
            "learning_rate": {"value": 0.001},
            "init_scheme": {
                "values": [
                    "default",
                    "xavier_normal",
                    "kaiming_normal",
                    "orthogonal",
                ]
            },
            "min_lr": {"value": 0.0001},
            "lr_schedule": {"values": ["cosine", "cosine_warmup"]},
            "warmup_epochs": {"value": 5},
            "weight_decay": {"value": 0.01},
            "hidden_dim": {"value": 64},  # Much smaller hidden dimension
            "num_heads": {"value": 2},  # Fewer attention heads
            "num_layers": {"value": 8},  # Fewer transformer layers
            "dropout": {"value": 0.1},
            "epochs": {"value": 30},  # Just 5 epochs for testing
            "seq_length": {"value": 32},  # Shorter sequences
            "wikitext_limit": {
                "value": 10000
            },  # Limit text size to 5000 chars for testing
        },
    }

    # # Initialize sweep
    # sweep_id = wandb.sweep(sweep_config, project="transformer_new_activation_test")

    # # Start sweep agent
    # wandb.agent(sweep_id, train)

    # Define specific experiment variations as diffs from the base config
    # Define a proper base config as a dictionary
    # large base config
    # base_config = {
    #     "dataset": "wikitext",
    #     "batch_size": 128,
    #     "learning_rate": 0.001,
    #     "min_lr": 0.0001,
    #     "lr_schedule": "default",
    #     "warmup_epochs": 0,
    #     "weight_decay": 0.01,
    #     "hidden_dim": 128,
    #     "num_heads": 4,
    #     "num_layers": 8,
    #     "dropout": 0.1,
    #     "epochs": 30,
    #     "seq_length": 128,
    #     "wikitext_limit": 10000,
    #     "pos_encoding": "sinusoidal",
    #     "init_scheme": "xavier_normal",
    #     # Add any missing parameters from your sweep config
    #     "shakespeare_limit": 10000,  # In case you need this
    # }
    # medium size base configbase_config = {

    # small size base config
    # base_config = {
    #     "dataset": "wikitext",
    #     "batch_size": 64,
    #     "learning_rate": 0.001,
    #     "min_lr": 0.0001,
    #     "lr_schedule": "default",
    #     "warmup_epochs": 0,
    #     "weight_decay": 0.01,
    #     "hidden_dim": 64,
    #     "num_heads": 4,
    #     "num_layers": 4,
    #     "dropout": 0.1,
    #     "epochs": 15,
    #     "seq_length": 64,
    #     "wikitext_limit": 10000,
    #     "pos_encoding": "sinusoidal",
    #     "init_scheme": "default",
    #     "optimizer": "adamw",
    # }
    base_config = {
        "dataset": "wikitext",
        "batch_size": 64,
        "learning_rate": 0.001,
        "min_lr": 0.0001,
        "lr_schedule": "default",
        "activation": "gelu",
        "warmup_epochs": 0,
        "weight_decay": 0.01,
        "hidden_dim": 128,
        "num_heads": 8,
        "num_layers": 4,
        "dropout": 0.1,
        "epochs": 15,
        "seq_length": 64,
        "wikitext_limit": 20000,
        "pos_encoding": "rotary",
        "init_scheme": "default",
        "optimizer": "adamw",
    }

    # experiments = [
    #     {
    #         # Experiment 1: Adam with ReLU
    #         "optimizer": "adam",
    #         "activation": "gelu",
    #         "pos_encoding": "rotary",
    #         # All other parameters inherited from base_config
    #     },
    #     # You can add more variants easily:
    #     {
    #         "activation": "gelu",
    #         "pos_encoding": "rotary",  # Override the scheduling method
    #     },
    # ]

    # Run each experiment, this for chronological experiment order experiments
    depths = [2, 4, 6, 8, 10]  # Example depths
    experiments = []

    for depth in depths:
        # Without GELU
        experiments.append(
            {
                "num_layers": depth,
                "optimizer": "adam",
            }
        )
        # With GELU
        experiments.append(
            {
                "num_layers": depth,
                "optimizer": "adamw",
            }
        )
    # Assuming final_losses is already populated with the results
    final_losses = {depth: {"adam": None, "adamw": None} for depth in depths}

    # updates as gpt to speed this up for mac os
    # update so that intermediate results are saved to csv

    # Run experiments and populate final_losses
    for exp in experiments:
        # Merge with base config
        config = {**base_config, **exp}

        # Initialize wandb run
        with wandb.init(project=project_name, config=config):
            train()  # Run the training with this config

            # Record the final loss
            final_loss = wandb.run.summary["train_loss"]
            # Use optimizer as the key instead of activation
            final_losses[config["num_layers"]][config["optimizer"]] = final_loss

    # Prepare data for CSV
    csv_data = [["Depth", "Final Loss with Adam", "Final Loss with AdamW"]]
    for depth in depths:
        adam_loss = final_losses[depth]["adam"]
        adamw_loss = final_losses[depth]["adamw"]
        csv_data.append([depth, adam_loss, adamw_loss])

    # Write to CSV
    csv_file_path = "experiment_results.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"Results saved to {csv_file_path}")

    # for exp_idx, exp_diff in enumerate(experiments):
    #     # Create a full config by merging the base config with experiment-specific diffs
    #     full_config = {**base_config, **exp_diff}

    #     # Add a descriptive name
    #     full_config_str = (
    #         f"Exp{exp_idx+1}_{exp_diff['optimizer']}_{exp_diff['activation']}"
    #     )
    #     print(f"Running experiment: {full_config_str}")

    #     # Initialize a wandb run for this experiment
    #     with wandb.init(
    #         project=project_name,
    #         config=full_config,
    #         name=full_config_str,
    #     ):
    #         train()  # Run the training with this config

    #     print(f"Completed experiment: {full_config_str}\n")

    # experiments = [
    #     {
    #         "dataset": "wikitext",
    #         "optimizer": "adam",
    #         "activation": "relu",
    #         "batch_size": 32,
    #         "learning_rate": 0.001,
    #         "weight_decay": 0.01,
    #         "hidden_dim": 64,
    #         "num_heads": 2,
    #         "num_layers": 8,
    #         "dropout": 0.1,
    #         "epochs": 30,
    #         "seq_length": 32,
    #         "wikitext_limit": 10000,
    #         "pos_encoding": "sinusoidal",
    #     },
    #     {
    #         "dataset": "wikitext",
    #         "optimizer": "adam",
    #         "activation": "gelu",
    #         # ... rest of the config same as above ...
    #     },
    #     {
    #         "dataset": "wikitext",
    #         "optimizer": "adamw",
    #         "activation": "gelu",
    #         # ... rest of the config same as above ...
    #     },
    # ]

    # running through configurations in historical order
