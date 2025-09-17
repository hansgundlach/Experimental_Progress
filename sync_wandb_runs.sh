#!/bin/bash

# Script to sync the N most recent offline wandb runs for Transformer experiments
# Adapted for Transformer language modeling project

# Default to 5 runs if no parameter provided
N_RUNS=${1:-5}

# Validate input
if ! [[ "$N_RUNS" =~ ^[0-9]+$ ]] || [ "$N_RUNS" -lt 1 ]; then
    echo "❌ Error: Please provide a positive number of runs to sync"
    echo "Usage: $0 [number_of_runs]"
    echo "Example: $0 10    # Sync the 10 most recent runs"
    echo "Example: $0        # Sync the 5 most recent runs (default)"
    exit 1
fi

echo "🔄 Finding and syncing the $N_RUNS most recent offline Transformer wandb runs..."
echo

# Check if wandb directory exists
if [ ! -d "wandb" ]; then
    echo "❌ Error: wandb directory not found in current directory"
    echo "Current directory: $(pwd)"
    echo "Contents:"
    ls -la | head -10
    exit 1
fi

# Change to wandb directory
cd wandb

# Find all offline-run directories, sort by modification time (newest first) and take the first N
recent_runs=$(find . -maxdepth 1 -type d -name "offline-run-*" -printf '%T@ %p\n' | sort -rn | head -$N_RUNS | cut -d' ' -f2-)

# Check if any runs were found
if [ -z "$recent_runs" ]; then
    echo "❌ No offline wandb runs found in wandb/ directory"
    echo "Looking for directories with pattern: offline-run-*"
    echo
    echo "📁 Available directories:"
    ls -la | grep "^d" | head -10
    exit 1
fi

echo "📋 Found the following recent offline Transformer runs:"
echo "$recent_runs" | sed 's|^\.\/||' | nl -w2 -s'. '
echo

# Show project mapping info
echo "📊 Transformer Project Mapping Information:"
echo "   - Each run will be synced to the 'transformer_experiments_MMDD_HHMM' project format"
echo "   - Project names are timestamp-based (e.g., 'transformer_experiments_0815_1430')"
echo "   - Run names include experiment labels and model configurations"
echo "   - Transformer scaling experiments from experiments.py → timestamped projects"
echo "   - Run names typically include model size (e.g., '64d_4h', '128d_8h', '256d_16h')"
echo

# Sync each run
count=0
success_count=0
for run_dir in $recent_runs; do
    count=$((count + 1))
    run_name=$(basename "$run_dir")
    echo "🚀 [$count/$N_RUNS] Syncing: $run_name"
    
    # Try to extract experiment info from the run
    if [ -f "$run_dir/files/config.yaml" ]; then
        # Look for project name
        project_name=$(grep "project:" "$run_dir/files/config.yaml" | head -1 | sed 's/.*project:[[:space:]]*//' | tr -d '"' | tr -d "'")
        if [ ! -z "$project_name" ]; then
            echo "   📍 Will sync to project: $project_name"
        fi
        
        # Look for experiment details
        hidden_dim=$(grep "hidden_dim:" "$run_dir/files/config.yaml" | head -1 | sed 's/.*hidden_dim:[[:space:]]*//' | tr -d '"')
        num_heads=$(grep "num_heads:" "$run_dir/files/config.yaml" | head -1 | sed 's/.*num_heads:[[:space:]]*//' | tr -d '"')
        num_layers=$(grep "num_layers:" "$run_dir/files/config.yaml" | head -1 | sed 's/.*num_layers:[[:space:]]*//' | tr -d '"')
        learning_rate=$(grep "learning_rate:" "$run_dir/files/config.yaml" | head -1 | sed 's/.*learning_rate:[[:space:]]*//' | tr -d '"')
        pos_encoding=$(grep "pos_encoding:" "$run_dir/files/config.yaml" | head -1 | sed 's/.*pos_encoding:[[:space:]]*//' | tr -d '"')
        
        if [ ! -z "$hidden_dim" ]; then
            echo "   🧠 Transformer config: ${hidden_dim}d hidden, ${num_heads}h heads, ${num_layers}L layers, lr=${learning_rate}, pos=${pos_encoding}"
        fi
    elif [ -f "$run_dir/files/wandb-metadata.json" ]; then
        # Fallback: check metadata file
        project_name=$(grep '"project"' "$run_dir/files/wandb-metadata.json" | sed 's/.*"project":[[:space:]]*"//' | sed 's/".*//')
        if [ ! -z "$project_name" ]; then
            echo "   📍 Will sync to project: $project_name"
        fi
    else
        echo "   ⚠️ No config found, using default project"
    fi
    
    # Check the run's age
    if [ -d "$run_dir" ]; then
        age_days=$(( ($(date +%s) - $(stat -c %Y "$run_dir")) / 86400 ))
        echo "   📅 Run age: $age_days days old"
    fi
    
    if wandb sync "$run_name"; then
        echo "   ✅ Successfully synced: $run_name"
        success_count=$((success_count + 1))
    else
        echo "   ❌ Failed to sync: $run_name"
    fi
    echo
done

echo "🎉 Finished syncing $N_RUNS offline Transformer wandb runs!"
echo "   ✅ Successfully synced: $success_count/$count runs"
echo

echo "💡 Tips for Transformer experiments:"
echo "   - All runs appear in timestamped projects like 'transformer_experiments_0815_1430'"
echo "   - Look for experiments with labels like '64d_4h', '128d_8h', '256d_16h' (hidden_dim_heads)"
echo "   - Different positional encodings (rotary vs sinusoidal) are tracked separately"
echo "   - Check for activation function variations (gelu, swiglu, etc.)"
echo "   - Recent experiments include attention scaling studies and Complete-P configurations"
echo

echo "📝 Usage: $0 [number_of_runs]"
echo "   Example: $0 10    # Sync the 10 most recent runs"
echo "   Example: $0        # Sync the 5 most recent runs (default)"
echo

echo "🔗 View your synced runs at: https://wandb.ai/[your-username]/transformer_experiments_[timestamp]"