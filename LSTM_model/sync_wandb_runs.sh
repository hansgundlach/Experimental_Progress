#!/bin/bash

# Script to sync the N most recent offline wandb runs for LSTM experiments
# Adapted for LSTM language modeling project

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

echo "🔄 Finding and syncing the $N_RUNS most recent offline LSTM wandb runs..."
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

echo "📋 Found the following recent offline LSTM runs:"
echo "$recent_runs" | sed 's|^\./||' | nl -w2 -s'. '
echo

# Show project mapping info
echo "📊 LSTM Project Mapping Information:"
echo "   - Each run will be synced to the 'lstm-wikitext' project"
echo "   - Run names will include experiment labels and timestamps"
echo "   - LSTM scaling experiments from lstm_experiments.py → 'lstm-wikitext'"
echo "   - Run names typically include model size (e.g., '32d', '48d', '64d')"
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
        project_name=$(grep "wandb_project:" "$run_dir/files/config.yaml" | head -1 | sed 's/.*wandb_project:[[:space:]]*//' | tr -d '"' | tr -d "'")
        if [ ! -z "$project_name" ]; then
            echo "   📍 Will sync to project: $project_name"
        fi
        
        # Look for experiment details
        hidden_size=$(grep "hidden_size:" "$run_dir/files/config.yaml" | head -1 | sed 's/.*hidden_size:[[:space:]]*//' | tr -d '"')
        batch_size=$(grep "batch_size:" "$run_dir/files/config.yaml" | head -1 | sed 's/.*batch_size:[[:space:]]*//' | tr -d '"')
        learning_rate=$(grep "learning_rate:" "$run_dir/files/config.yaml" | head -1 | sed 's/.*learning_rate:[[:space:]]*//' | tr -d '"')
        use_streaming=$(grep "use_streaming:" "$run_dir/files/config.yaml" | head -1 | sed 's/.*use_streaming:[[:space:]]*//' | tr -d '"')
        
        if [ ! -z "$hidden_size" ]; then
            echo "   🧠 LSTM config: ${hidden_size}d hidden, batch=${batch_size}, lr=${learning_rate}, streaming=${use_streaming}"
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

echo "🎉 Finished syncing $N_RUNS offline LSTM wandb runs!"
echo "   ✅ Successfully synced: $success_count/$count runs"
echo

echo "💡 Tips for LSTM experiments:"
echo "   - All runs should appear in the 'lstm-wikitext' project on wandb.ai"
echo "   - Look for experiments with labels like '32d', '48d', '64d' (hidden sizes)"
echo "   - Streaming vs non-streaming experiments are now tracked separately"
echo "   - Check for dropout variations and TBPTT configurations"
echo "   - Recent experiments include learning rate sweeps and scaling studies"
echo
echo "📝 Usage: $0 [number_of_runs]"
echo "   Example: $0 10    # Sync the 10 most recent runs"
echo "   Example: $0        # Sync the 5 most recent runs (default)"
echo
echo "🔗 View your synced runs at: https://wandb.ai/[your-username]/lstm-wikitext"