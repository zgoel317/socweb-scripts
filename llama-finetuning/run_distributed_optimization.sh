#!/bin/bash

# Configuration
SCRIPT_NAME="optimize_llama_qlora_deepspeed_optuna.py"
NUM_GPUS=3  # Set to your number of GPUs

# 1. Cleanup: Kill any lingering python processes to free VRAM
echo "Cleaning up old processes..."
pkill -f $SCRIPT_NAME

# 2. Setup Environment
# We use all GPUs together as a single resource
export CUDA_VISIBLE_DEVICES=0,1,2

# CRITICAL: Limit CPU threads. 
# Since torchrun spawns 3 processes that communicate heavily, 
# allowing them to use all CPU cores causes massive slowdowns.
export OMP_NUM_THREADS=1 

echo "Starting Distributed Optimization..."
echo "Configuration: 1 Trial is being split across $NUM_GPUS GPUs."

# 3. The Launcher
# --nproc_per_node: Must match the number of GPUs you want to use
# --master_port: Port for the GPUs to talk to each other
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    $SCRIPT_NAME \
    2>&1 | tee logs/distributed_training.log