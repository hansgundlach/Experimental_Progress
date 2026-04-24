# Wandb logging, syncing, and gradient-instability checks

## How wandb runs get created

Both trainers run wandb in **offline mode** on SuperCloud (compute nodes have no internet):

- `core.py` sets `os.environ["WANDB_MODE"] = "offline"` at import time, so every transformer run from `experiments.py` is offline.
- `lstm_training.py` honors `config["wandb_offline"]`, which defaults to `True` in `LSTM_model/lstm_experiment_utils.py`.

A run is started inside each trainer:

- Transformer: `wandb.init(project=folder_name, config=config, name=sub_label, reinit=True)` in `experiments.py:51`.
- LSTM: `wandb.init(project=folder_name_or_config, config=config, name=run_name)` in `LSTM_model/lstm_training.py:1077`.

The wandb **project name** = the experiment's `folder_name` (e.g. `x2_lstm_layer2`), and the **run name** = the sub-experiment label (e.g. `128d`, `64d_lr_-2.5`).

## Where the offline run files live

Wandb writes `offline-run-<timestamp>-<id>/` directories into a `wandb/` folder **next to whatever script was launched**. Because different entry points run from different CWDs, runs end up in three places in this repo:

| Source | Offline runs land in |
|---|---|
| `python experiments.py` (transformer) | `./wandb/` |
| `python LSTM_model/lstm_experiments.py` (LSTM) | `LSTM_model/wandb/` |
| `python learning_rate_analysis/run_lr_sweep.py` | `learning_rate_analysis/wandb/` |

Each run directory contains:
- `files/config.yaml` — the full config snapshot
- `files/output.log` — stdout/stderr
- `run-*.wandb` — the binary metric stream (what `wandb sync` uploads)

## Syncing to wandb.ai

Use `sync_wandb_runs.sh` from the repo root. It now scans all three wandb directories.

```bash
# 5 most recent runs across all dirs (default)
./sync_wandb_runs.sh

# 20 most recent across all dirs
./sync_wandb_runs.sh 20

# Scope to one source
./sync_wandb_runs.sh 20 lstm    # LSTM_model/wandb
./sync_wandb_runs.sh 20 root    # ./wandb (transformer)
./sync_wandb_runs.sh 20 lr      # learning_rate_analysis/wandb

# Everything that's still on disk
./sync_wandb_runs.sh all
```

The script `cd`s into each run's parent and calls `wandb sync <run-name>`. Auth is read from `~/.netrc` (already set up — verified via `wandb --version`).

After a sync, runs show up at `https://wandb.ai/<your-entity>/<project>`, where `<project>` is the experiment folder name.

### If you are running on a compute node without internet

Sync from a login node after the job finishes:

```bash
ssh <login-node>
cd ~/Experimental_Progress
./sync_wandb_runs.sh all
```

## Gradient instability: what is logged and how to read it

Both trainers log two gradient metrics to wandb **every optimizer step** (not to CSV — this is why you must sync to inspect them):

- `grad_norm_preclip` — the **pre-clip** global grad norm returned by `torch.nn.utils.clip_grad_norm_`. This is the true size of the gradient before clipping reins it in.
- `clipped_step` — 0/1 indicator that `grad_norm_preclip > gradient_clip_val` on that step.

Source: `core.py:1266-1305` and `LSTM_model/lstm_training.py:1425-1475`.

> **Note on older LSTM runs.** Before 2026-04-23 the LSTM trainer had a gate `if not config.get("wandb_offline", False): wandb.log(step_log_dict, ...)` that skipped per-step metrics whenever offline mode was on — which is every run on SuperCloud. So `grad_norm_preclip` and `clipped_step` **do not exist** in LSTM runs from before that fix, even after syncing. The gate is removed now; re-run the experiment to collect them.

`grad_norm_preclip` is only populated when `config.use_gradient_clipping == True`. If you want to observe raw instability with clipping **off**, temporarily set `use_gradient_clipping=True` with a very large `gradient_clip_val` (e.g. `1e9`) — the clip won't bite, but the pre-clip norm will still be logged.

### Diagnosing instability from the wandb UI

After syncing, open the project and look at the run:

1. **Plot `grad_norm_preclip` vs `optimizer_step` on a log y-axis.**
   - Healthy: bounded band, maybe slowly decaying, no spikes more than ~5× the median.
   - Unstable: spikes orders of magnitude above baseline, or a trend that grows over training.
2. **Plot `clipped_step` as a running mean** (wandb "smooth" slider, or group by bins).
   - `<1%` clipped: your `gradient_clip_val` is loose — clipping isn't doing much, and isn't needed unless you see rare catastrophic spikes.
   - `1–20%` clipped: clipping is actively stabilizing. Typical for aggressive LRs.
   - `>50%` clipped: the clip threshold is below your normal grad norm — you are effectively training with a smaller LR than configured. Either raise `gradient_clip_val` or lower the LR.
3. **Correlate with `train_loss_per_token`.** A grad-norm spike followed by a loss spike = instability. Grad-norm spikes with no loss response = clipping is doing its job.

### Rules of thumb for adjusting `gradient_clip_val`

- If you see rare (<5%) but very large spikes (10×+ median) **and** the loss reacts to them → **lower** `gradient_clip_val` to ~median grad norm.
- If clipping fires on nearly every step → **raise** `gradient_clip_val` or reduce LR.
- If grad norm drifts upward through training → instability is building; lower LR first, then consider clipping.

Both configs expose the knobs:

- Transformer: `config.use_gradient_clipping`, `config.gradient_clip_val` (see `core.py`).
- LSTM: `config["use_gradient_clipping"]`, `config["gradient_clip_val"]` (default `1.0`, see `LSTM_model/lstm_training.py:1427`).

### Quick wandb API check for a single run (no UI)

```python
import wandb
api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")  # run_id is the suffix after offline-run-...-
hist = run.history(keys=["grad_norm_preclip", "clipped_step", "optimizer_step"])
print(hist["grad_norm_preclip"].describe())
print("fraction clipped:", hist["clipped_step"].mean())
print("max / median ratio:", hist["grad_norm_preclip"].max() / hist["grad_norm_preclip"].median())
```

A `max/median` ratio above ~10 is the clearest signal that clipping is worth tightening.
