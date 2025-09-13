# CLAUDE.md - Experimental Progress ML Research Project

This repository implements machine learning experiments with Transformer and LSTM models, focusing on scaling laws, optimization techniques, and distributed training.

## 🎯 Project Overview

**Primary Goal**: Research scaling laws, muP (Maximal Update Parametrization), and optimization techniques across different model architectures and sizes.

**Key Features**:
- **Transformer Models**: Standard transformer architecture with configurable scaling
- **LSTM Models**: RNN-based models with comparable parameter scaling
- **muP Scaling**: Maximal Update Parametrization for hyperparameter transfer
- **Distributed Training**: DDP support for multi-GPU training
- **Scaling Law Analysis**: Chinchilla-style power law fitting and analysis

## 📁 Repository Structure

### Core Training Scripts
- `core.py` - Main transformer training with comprehensive logging and memory optimization
- `experiments.py` - Experiment runner that processes experiment definitions
- `LSTM_model/lstm_training.py` - LSTM training with DDP support

### Configuration and Experiment Definition
- `experiment_definitions.py` - Predefined experiment configurations for scaling studies
- `experiment_utils.py` - Utilities for generating transformer experiments with memory optimization
- `LSTM_model/lstm_experiment_utils.py` - LSTM experiment generation utilities

### Analysis and Evaluation
- `experimental_analysis/` - Scaling law fitting and analysis tools
- `experimental_analysis/fit_scaling_analysis.py` - Chinchilla scaling law replication

### Optimization Utilities
- `cpu_optimization_utils.py` - Intelligent CPU utilization for data loading across different machines

### SLURM Job Submission
- `submit_job.sh` - Transformer experiment submission (1 GPU per experiment)
- `submit_lstm_job.sh` - LSTM experiment submission (2 GPUs per experiment for DDP)
- `lstm.sh` - SLURM array job script for LSTM experiments

## 🚀 Quick Start

### Running Transformer Experiments
```bash
# Single experiment
python experiments.py

# SLURM submission for multiple experiments
bash submit_job.sh -50  # Run 50 experiments
```

### Running LSTM Experiments
```bash
# SLURM submission (DDP training)
bash submit_lstm_job.sh -25  # Run 25 experiments (2 GPUs each)
```

### Testing Experiment Generation
```bash
python test_gen_experim_output.py  # Test gen_experim function output
```

## 🔧 Key Technical Features

### Memory Optimization
- **Automatic Memory Estimation**: Calculates optimal per-step batch size and gradient accumulation
- **Model-Size Scaling**: Conservative memory thresholds for larger models (64d+, 128d+)
- **GPU Memory Management**: Prevents OOM errors through intelligent batch size adjustment

### CPU Optimization
- **Adaptive DataLoader Configuration**: Automatically detects system capabilities
- **Shared System Detection**: Scales conservatively on shared computing resources
- **Model-Specific Optimization**: Different configurations for transformer vs LSTM models

### muP Implementation
- **Weight Tying Support**: Geometric mean scaling for tied embedding/output weights
- **Parameter Group Management**: Proper learning rate scaling for different parameter types
- **Base Width Configuration**: Configurable base model for muP scaling transfer

### Distributed Training
- **Transformer**: Single GPU per experiment
- **LSTM**: DDP training with 2 GPUs per experiment
- **SLURM Integration**: Job array submission with proper resource allocation

## 📊 Experiment Types

### Scaling Experiments
```python
# Transformer scaling across hidden dimensions
TRANSFORMER_SCALING_EXPERIMENTS_OPTIMAL_LR = (
    gen_experim(32, label="vanilla_32d", learning_rate=1e-2) +
    gen_experim(64, label="vanilla_64d", learning_rate=10**(-2.5)) +
    gen_experim(128, label="vanilla_128d", learning_rate=10**(-3))
)
```

### Learning Rate Sweeps
```python
# Multi-LR experiments with muP
TRANSFORMER_LR_TUNE_MUP_STANDARD = create_multi_lr_experiments(
    gen_experim(32, label="32d_mup", use_mup=True, mup_base_width=32),
    NARROW_LR_SWEEP
)
```

### Activation and Initialization Studies
```python
# Different activation functions
ACTIVATION_EXPERIMENTS = (
    gen_experim(16, label="16d_activation", activation="gelu") +
    gen_experim(16, label="16d_activation", activation="relu") +
    gen_experim(16, label="16d_activation", activation="swiglu")
)
```

## 🛠 Configuration Parameters

### Model Settings
```yaml
hidden_dim: 64              # Model width (scales other parameters)
num_layers: 6               # Depth (scales with width^0.5)
num_heads: 8                # Attention heads
seq_length: 512             # Sequence length
tie_embeddings: true        # Share input/output embeddings
```

### Training Parameters (Auto-adjusted)
```yaml
batch_size: 32              # Per-step batch size (calculated)
gradient_accumulation_steps: 4  # Accumulation (calculated)
learning_rate: 0.01         # Base learning rate
target_effective_batch_size: 128  # Target total batch size
```

### muP Configuration
```yaml
use_mup: true
mup_base_width: 32          # Base model width for scaling
mup_scale_lr: true          # Scale learning rate with width
```

### Memory Optimization
```yaml
force_conservative_cpu: false  # Force conservative CPU usage
target_gpu_memory_usage: 0.6   # Target GPU memory utilization
```

## 📈 Expected Results and Monitoring

### Output Structure
```
results/
├── experiment_name/
│   ├── 32d_experiment.csv          # Training metrics
│   ├── 64d_experiment.csv
│   └── config_32d_experiment.json  # Experiment configuration
└── lr_sweep_folder/
    ├── 32d_experiment_lr_001.csv
    └── 32d_experiment_lr_003.csv
```

### Key Metrics Logged
- `loss`: Training loss per step
- `tokens_processed`: Cumulative tokens (accounts for gradient accumulation)
- `learning_rate`: Current learning rate (with scheduling)
- `theoretical_flops_chinchilla`: 6ND FLOP calculation
- `optimizer_step`: True optimizer steps (not micro-batches)

### Memory and Performance
- Automatic batch size calculation prevents OOM errors
- Conservative thresholds for 64d+ models (45% memory) and 128d+ models (35% memory)
- Gradient accumulation maintains constant effective batch size
- CPU utilization scaled based on system detection

## 🔍 Common Operations

### Adding New Experiments
```python
# In experiment_definitions.py
NEW_EXPERIMENT = gen_experim(
    hidden_dim=48,
    label="my_experiment",
    folder_name="custom_folder",  # Optional: custom output folder
    learning_rate=0.005,
    use_mup=True,
    mup_base_width=32
)
```

### Running Scaling Analysis
```python
# Generate experiments for different model sizes
scaling_experiments = (
    gen_experim(32, label="scale_32d") +
    gen_experim(64, label="scale_64d") +
    gen_experim(128, label="scale_128d")
)
```

### Memory Troubleshooting
- **OOM Errors**: The system should automatically prevent these, but if they occur:
  - Check `gradient_accumulation_steps` calculation in logs
  - Verify `target_effective_batch_size` is reasonable
  - Consider using `force_conservative_cpu=True` for shared systems

### Folder Organization
- Use `folder_name` parameter in `gen_experim()` for custom output directories
- Results appear as `folder_name/experiment_label.csv`
- Learning rate sweeps create: `folder_name_lr_sweep/experiment_label_lr_XXX.csv`

## 🚨 Important Notes

### Resource Management
- **Transformer**: 1 GPU per experiment
- **LSTM**: 2 GPUs per experiment (DDP training)
- CPU workers auto-scale based on system detection
- Memory estimation prevents OOM through gradient accumulation

### Remote Execution
- Code designed for remote execution on various machines
- Adaptive CPU detection works across different cluster environments
- Results can be synced with: `rsync -avz user@host:path/results/ local_folder/`

### muP Scaling
- Properly handles weight tying with geometric mean learning rates
- Base width should match smallest model in scaling study
- Learning rates scale as `base_lr / sqrt(width / base_width)`

### Token Counting Accuracy
- Accounts for gradient accumulation: `tokens = optimizer_steps × effective_batch_size × seq_length`
- `effective_batch_size = batch_size × gradient_accumulation_steps`
- Logging happens on optimizer steps, not micro-batch steps

## 💡 Development Guidelines

### Adding New Model Architectures
1. Create model-specific training script (follow `lstm_training.py` pattern)
2. Add experiment utilities (follow `lstm_experiment_utils.py`)
3. Update CPU optimization for new model type
4. Create SLURM submission script with appropriate GPU allocation

### Memory Optimization Rules
- Always calculate per-step batch size first, then gradient accumulation
- Use conservative thresholds for larger models
- Test memory estimation on target hardware before large runs
- Monitor actual vs estimated memory usage in initial runs

### Experiment Organization
- Use descriptive labels for experiment tracking
- Specify `folder_name` for organized output structure
- Include key hyperparameters in experiment labels
- Keep base configurations consistent across scaling studies

This repository is optimized for academic research with focus on scaling laws, distributed training, and systematic hyperparameter studies across different model architectures.