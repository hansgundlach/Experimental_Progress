# Learning Rate Analysis

Automated pipeline for finding optimal learning rates across model sizes. Runs LR sweeps on SLURM, fits a power law (`lr = c * dim^b`), and outputs a table of optimal LRs for any hidden dimension.

## Quick Start

From the **project root** (`Experimental_Progress/`):

```bash
# List available experiment groups
bash learning_rate_analysis/submit_lr.sh --list

# Run a sweep
bash learning_rate_analysis/submit_lr.sh modern_transformer

# Limit to 4 concurrent GPUs (default is 8)
bash learning_rate_analysis/submit_lr.sh modern_transformer -c 4
```

Use `bash`, not `sbatch` — the script calls `sbatch` internally.

## What Happens

1. A SLURM array job is submitted (1 GPU per task, up to 8 concurrent V100s)
2. Each task trains a subset of (hidden_dim, learning_rate) combinations at ~5% of full token budget
3. After all tasks finish, a dependency job automatically runs the analysis
4. The analysis finds the best LR per dimension, fits `log(lr) = a + b*log(dim)`, and outputs:
   - A plot of LR vs hidden dimension with the fit line
   - A table of fitted optimal LRs for dims 16-512 (step 16)
   - A copy-paste Python dict for use in `experiment_definitions.py`

## Output Locations

```
learning_rate_analysis/
├── results/<group_name>/    # Training CSVs (one per dim/lr combo)
│   ├── 32d_lr_1e-02.csv
│   ├── 32d_lr_3e-02.csv
│   └── ...
└── plots/<group_name>/      # Plots and analysis log
    ├── <group_name>_lr_scaling.png
    ├── <group_name>_lr_scaling.pdf
    └── analysis_output.log
```

## Re-running Analysis Manually

If you want to re-run the analysis without re-running the sweep (e.g., after adding more results):

```bash
cd learning_rate_analysis
python analyze_lr.py --group modern_transformer

# Or analyze all groups that have results
python analyze_lr.py --all-groups
```

## Defining New Experiment Groups

Edit `lr_experiment_groups.py` and add an entry to `LR_EXPERIMENT_GROUPS`:

```python
"my_new_setup": {
    "description": "Description of what this tests",
    "hidden_dims": [32, 64, 96, 128, 192, 256],
    "learning_rates": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    "model_overrides": {
        "activation": "swiglu",
        "pos_encoding": "rotary",
        "optimizer": "adamw",
    },
    "token_budget_fraction": 0.05,
    "csv_log_interval": 100,
},
```

Then run it:

```bash
bash learning_rate_analysis/submit_lr.sh my_new_setup
```

## Monitoring

```bash
squeue -u $(whoami)          # Check job status
scancel <job_id>             # Cancel a sweep
```

## Files

| File | Purpose |
|------|---------|
| `submit_lr.sh` | Main entry point. Submits sweep + analysis jobs to SLURM |
| `lr_experiment_groups.py` | Defines experiment groups (dims, LRs, model config) |
| `run_lr_sweep.py` | Training runner called by each SLURM array task |
| `analyze_lr.py` | Reads CSVs, fits power law, makes plots, prints LR table |
| `lr_job.sh` | SLURM script for each sweep task (1 GPU) |
| `analyze_job.sh` | SLURM script for the analysis step (no GPU) |
