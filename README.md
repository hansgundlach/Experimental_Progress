# Model Organism of Algorithmic Progress

How much influence have new AI algorithms had on improveements in training efficiency?

# Here We Examine a Few Important Improvements

activation functions: ReLU, GeLU, SwiGLU
optimizers: sgd(heavy ball momentum), adam, adamW
positional encodings: sinusoidal, learned, and rotary encodings
learning rate schedules: linear decay, cosine decay
normalization: layernorm, rmsnorm
architectures: LSTM, Transformer

## Stack

- pytorch
- weights and biases
- MIT supercloud V100s (these are offline)

# Current Optionality

activation functions: relu, gelu, silu/swish, GLU, swigGLU
optimizers: sgd(heavy ball momentum), adam, adamW
learning rate schedule: none, warmup+cosine annealing, warmup+inverse square root
positional encodings: sinusoidal, rotary encodings
initializations: xavier normal, kaiming normal, transformer (layer dependent) initalization
regularization options: early stopping, gradient clipping

## Optimizations

- torch.compile
- autocast mixed precision
- flash attention
- multiprocessing experiments
- data loading optimization: keeping workers alive, pinning memory, prefetch factor

## Running guidlines

run setup_dataset.py to setup wikitext and GTP2Tokenizer
sbatch main.sh

## Analyzing Learning Rate Sweeps

After running learning rate experiments, use `analyze_best_lr.py` to find the optimal learning rates for each model dimension:

```bash
# Basic usage - analyzes experimental_data_folder by default
python analyze_best_lr.py

# Specify a custom folder
python analyze_best_lr.py --folder path/to/results

# Show detailed per-folder breakdown
python analyze_best_lr.py --detailed

# Verbose output during processing
python analyze_best_lr.py --verbose
```

The script:
- Scans CSV files from learning rate sweep experiments
- Extracts model dimensions (32d, 64d, 128d, etc.) and learning rates from filenames
- Identifies the best learning rate for each dimension based on final validation loss
- Outputs results in formatted tables showing best LRs across different experimental folders
