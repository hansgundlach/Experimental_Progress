# Dynamic Experiment Submission System

This system automatically counts your experiments and creates the appropriate SLURM job array size.

## Files Created

1. **`count_experiments.py`** - Counts total experiments defined in experiments.py
2. **`submit_experiments.sh`** - Dynamically submits jobs with correct array size
3. **`main.sh`** - Original SLURM job script (unchanged)

## Usage

### Quick Start

Instead of manually submitting with `sbatch main.sh`, now use:

```bash
./submit_experiments.sh
```

### What It Does

1. **Counts experiments**: Runs `count_experiments.py` to get total experiment count
2. **Calculates array size**: Creates array `0-(N-1)%8` where N is experiment count
3. **Creates temporary job script**: Copies `main.sh` with updated array size
4. **Submits job**: Uses `sbatch` to submit the dynamically-sized array
5. **Cleans up**: Removes temporary files

### Current Configuration

- **Max concurrent jobs**: 8 (adjustable in `submit_experiments.sh`)
- **Current experiment count**: 18 experiments
- **Generated array**: `0-17%8`

### Customization

**To change max concurrent jobs:**
Edit `MAX_CONCURRENT=8` in `submit_experiments.sh`

**To change which experiments run:**
Modify the `EXPERIMENTS = scaling_experiments` line in `experiments.py`

**To add new experiment types:**

1. Add experiments to `experiment_definitions.py`
2. Update the experiment list in both `experiments.py` and `count_experiments.py`

### Benefits

- ✅ **No wasted resources**: Only requests exactly the GPUs needed
- ✅ **Automatic sizing**: No manual counting or array size calculation
- ✅ **Safe**: Validates experiment count before submission
- ✅ **Portable**: Works on both Linux clusters and local development

### Example Output

```bash
$ ./submit_experiments.sh
=== Dynamic Experiment Submission Script ===
Counting experiments...
Found 18 experiments
Will use array 0-17%8
Created temporary job script: main_dynamic.sh
Array configuration: #SBATCH --array=0-17%8

=== Job Configuration ===
#SBATCH --job-name=gpu_job_array
#SBATCH --array=0-17%8
#SBATCH --partition=xeon-g6-volta
...

Submitting job array...
Submitted batch job 12345
Job submitted successfully!
```
