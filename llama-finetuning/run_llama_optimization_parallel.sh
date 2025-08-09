#!/bin/bash

# LLaMA QLoRA Parallel Optimization Script
# This script runs the hyperparameter optimization using Optuna's n_jobs=3 for parallel trials

# Configuration
SCRIPT_NAME="optimize_llama_qlora_parallel.py"
CONFIG_FILE="gpu_config.yaml"
LOG_DIR="./logs"
OUTPUT_DIR="./outputs"

# Load configuration from YAML file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found"
    exit 1
fi

# Parse YAML configuration using Python with conda environment
eval $(source ~/miniconda3/etc/profile.d/conda.sh && conda activate llama-env && python3 -c "
import yaml
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f'PHYSICAL_GPUS=\"{config[\"physical_gpus\"]}\"')
    print(f'NUM_GPUS={config[\"num_gpus\"]}')
    print(f'CLUSTER_LABELING={str(config[\"training\"][\"cluster_labeling\"]).lower()}')
    print(f'DEBUG={str(config[\"training\"][\"debug\"]).lower()}')
except Exception as e:
    print(f'Error parsing config file: {e}', file=sys.stderr)
    sys.exit(1)
")

echo "Loaded configuration from $CONFIG_FILE:"
echo "  Physical GPUs: $PHYSICAL_GPUS"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Cluster labeling: $CLUSTER_LABELING"

# Create directories
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

# Log file name based on cluster labeling
if [ "$CLUSTER_LABELING" = true ]; then
    LOG_FILE="$LOG_DIR/llama_optimization_parallel_cluster.log"
else
    LOG_FILE="$LOG_DIR/llama_optimization_parallel_binary.log"
fi

# Kill any existing optimize_llama processes
echo "Killing any existing optimize_llama processes..."
pkill -f "optimize_llama_qlora_parallel.py" 2>/dev/null || true

# Kill any existing tmux sessions with llama_optimization
echo "Killing any existing llama optimization tmux sessions..."
tmux kill-session -t "llama_optimization_parallel_cluster" 2>/dev/null || true
tmux kill-session -t "llama_optimization_parallel_binary" 2>/dev/null || true
tmux kill-session -t "llama_optimization_parallel" 2>/dev/null || true

# Clear the log file
echo "Clearing log file..."
> $LOG_FILE

echo "Starting LLaMA QLoRA Parallel Optimization"
echo "Log file: $LOG_FILE"
echo "Cluster labeling: $CLUSTER_LABELING"
echo "Number of GPUs: $NUM_GPUS"
echo "Physical GPUs: $PHYSICAL_GPUS"
echo "Parallel trials: 3 (n_jobs=3)"

# Check if conda environment exists
echo "Checking if conda environment exists..."
if ! conda env list | grep -q "llama-env"; then
    echo "Error: conda environment llama-env not found"
    exit 1
fi
echo "Conda environment llama-env found"

# Create tmux session name based on cluster labeling
if [ "$CLUSTER_LABELING" = true ]; then
    SESSION_NAME="llama_optimization_parallel_cluster"
else
    SESSION_NAME="llama_optimization_parallel_binary"
fi
echo "Creating tmux session: $SESSION_NAME"

# Create new session
tmux new-session -d -s $SESSION_NAME

# Build the command with optional cluster labeling flag
if [ "$CLUSTER_LABELING" = true ]; then
    CLUSTER_FLAG="--cluster_labeling"
else
    CLUSTER_FLAG=""
fi

# Run the parallel optimization script
# Note: We don't need DeepSpeed or distributed training for parallel optimization
# Each worker will use a single GPU, and Optuna will handle the parallelization
echo "Starting parallel optimization..."
echo "Using physical GPUs: $PHYSICAL_GPUS"
echo "Number of GPUs: $NUM_GPUS"

# Check if we have enough GPUs for parallel execution
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Warning: Only $NUM_GPUS GPU(s) available. Parallel optimization may not be optimal."
    echo "Consider using more GPUs for better parallel performance."
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "Single GPU detected. Running sequential optimization (n_jobs=1)."
elif [ "$NUM_GPUS" -eq 2 ]; then
    echo "Two GPUs detected. Running with 2 parallel jobs."
elif [ "$NUM_GPUS" -eq 3 ]; then
    echo "Three GPUs detected. Running with 3 parallel jobs."
elif [ "$NUM_GPUS" -eq 4 ]; then
    echo "Four GPUs detected. Running with 4 parallel jobs."
elif [ "$NUM_GPUS" -gt 4 ]; then
    echo "$NUM_GPUS GPUs detected. Running with $NUM_GPUS parallel jobs."
fi

echo "Each trial will run on a single GPU, with up to $NUM_GPUS trials running in parallel"

# Set CUDA_VISIBLE_DEVICES and run the script
# The script will handle GPU assignment internally via Optuna's n_jobs
tmux send-keys -t $SESSION_NAME "source ~/miniconda3/etc/profile.d/conda.sh && conda activate llama-env && CUDA_VISIBLE_DEVICES=$PHYSICAL_GPUS python3 $SCRIPT_NAME $CLUSTER_FLAG 2>&1 | tee $LOG_FILE" Enter

echo "Parallel optimization started in tmux session: $SESSION_NAME"
echo "To monitor progress:"
echo "  tmux attach-session -t $SESSION_NAME"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Note: This will run up to $NUM_GPUS trials in parallel, each using a single GPU."
echo "Total optimization time should be approximately 1/$NUM_GPUS of the sequential version."
echo ""
echo "To change GPU configuration:"
echo "  ./change_gpus.sh \"1,2,3\"  # Use GPUs 1, 2, and 3"
echo "  ./change_gpus.sh \"0,1\"    # Use GPUs 0 and 1"
echo "  ./change_gpus.sh \"2\"      # Use only GPU 2" 