# Model Organism of Algorithmic Progress

How much influence have AI algorithms had on training efficiency?

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
- V100s (these are offline)

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

## Running Guidelines

Run `setup_dataset.py` to setup wikitext and GPT2Tokenizer, then submit experiments via SLURM.

## Workflow

### 1. Define Experiments

Experiments are defined in `experiment_definitions.py` (and `non_scaling_experim_def.py`) using `gen_experim()` from `experiment_utils.py`. Each call specifies a hidden dimension, learning rate, optimizer, etc. and produces a structured dict. Experiments are composed by concatenating `gen_experim()` calls with `+`.

Key experiment types:
- **Scaling studies** — sweep hidden dims (32, 48, 64, 96, 128, 160, 192, 256+) at fixed or tuned LRs for transformers, LSTMs, SGD variants, muP, etc.
- **LR sweeps** — `create_multi_lr_experiments()` wraps a base experiment with multiple learning rates at reduced token budgets, producing `*_lr_sweep` folders to find the optimal LR before full runs.
- **Ablation studies** — activation functions, rotary embeddings, weight decay, optimizers (Adam vs SGD vs RMSProp), etc.
- **Multi-seed runs** — `create_multi_seed_experiments()` for variance estimation.

The active experiment set is selected by setting `GRAND_EXPERIMENT` in `experiment_definitions.py`, which `experiments.py` imports as `EXPERIMENTS`.

### 2. Submit to SLURM

- **Transformers**: `./submit_job.sh -50` submits a SLURM array job. Each array task gets 1 V100 GPU via `main.sh`. The task ID and total count slice the experiment list so each GPU runs a subset.
- **LSTMs**: `./submit_lstm_job.sh -25` does the same via `LSTM_model/lstm.sh`, running `lstm_experiments.py` with the same slicing pattern.
- Concurrency is capped at 8 GPUs by default. Logs go to `logs/DD-HH/`.
- Each experiment runs `train()` from `core.py`, logs to W&B (when online), and writes a CSV to `experimental_data_folder/<folder_name>/<label>.csv` with columns: `step, training_loss, validation_loss, total_flops_profiler, theoretical_flops, tokens`.

### 3. Iterate (LR Sweep then Full Run)

The typical pattern for a new model size or architecture variant:

1. **LR sweep first** — Run `create_multi_lr_experiments()` with short token budgets across many LRs. This populates `*_lr_sweep` folders.
2. **Pick optimal LR** — Use `analyze_best_lr.py` or inspect W&B/CSVs to find the best LR for each hidden dim.
3. **Full scaling run** — Plug the optimal LRs back into `gen_experim()` calls for the full token-budget run.

This is reflected in paired folder names: `vanilla_scaling` + `vanilla_scaling_optimal_lr`, `sgd_scaling` + `optimal_lr_sgd_scaling`, `lstm_scaling` + `lstm_scaling_lr_sweep`, etc.

### 4. Sync Results Locally

Results are synced from the cluster using `rsync`. There is also `sync_wandb_runs.sh` for pulling W&B data.

### 5. Analysis

All analysis happens in `experimental_analysis/`:

- **`nextgen_lstmvtransformer copy.py`** — Contains `TrainingCurveAnalyzer`, which loads CSVs, extracts final losses, plots loss-vs-compute curves, and fits power laws (`L = E + A * C^alpha`).
- **`fit_hitchhikers_loss.py`** — Chinchilla-style fitting (`L(N,D) = E + A*N^alpha + B*D^beta`) with Huber loss across all runs.
- **`plot_analysis_neurips_final.py`** — Publication figure generation (e.g. two-panel transformer vs LSTM scaling comparisons).
- **`compute_multiplier_by_loss.py`**, **`single_panel_lstm_layers.py`** — Specific figure scripts for compute multiplier analysis and LSTM layer comparisons.
- **Notebooks** — `lr_analysis.ipynb`, `main_ablation.ipynb`, `kaplan_vs_chinchilla.ipynb`, and `how-bitter-graphs/` for exploratory analysis.

### 6. Figures

Final outputs land in `Figures/` and `experimental_analysis/Figures/` as PDFs and PNGs for publication.

### Summary

```
experiment_definitions.py  (define what to run)
        |
submit_job.sh / submit_lstm_job.sh  (sbatch array jobs)
        |
main.sh -> experiments.py -> core.py  (train on V100s, log CSVs + W&B)
        |
rsync CSVs to local machine
        |
experimental_analysis/ scripts & notebooks  (fit scaling laws, plot)
        |
Figures/*.pdf  (publication-ready output)
```

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
