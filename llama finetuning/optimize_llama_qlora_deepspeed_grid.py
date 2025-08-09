import os
import sys
import json
import pickle
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from datasets import Dataset
import optuna
from optuna.pruners import MedianPruner
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import bitsandbytes as bnb

# DeepSpeed and Accelerate imports
import deepspeed
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)

# Global debug flag
DEBUG = True

def debug_print(message, rank=None, important=False):
    """Debug print function with rank information - only print important messages by default"""
    if DEBUG and important:
        if rank is not None:
            print(f"[DEBUG][Rank {rank}] {message}", flush=True)
        else:
            print(f"[DEBUG] {message}", flush=True)

def debug_gpu_memory(rank=None, prefix="", important=False):
    """Debug function to print GPU memory usage - only when important"""
    if DEBUG and important and torch.cuda.is_available():
        available_gpus = get_available_gpu_count()
        for i in range(available_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free = total - reserved
            debug_print(f"{prefix}GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free, {total:.2f}GB total", rank, important)

def check_memory_before_trial(trial_num, hyperparams):
    """Check if we have enough memory for the trial"""
    if torch.cuda.is_available():
        device = get_current_device()  # Use our safe device getter
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free = total - reserved
        
        # Estimate memory needed for this trial
        batch_size = hyperparams.get("per_device_train_batch_size", 1)
        lora_r = hyperparams.get("lora_r", 8)
        
        # More realistic estimate for LLaMA model with larger batch sizes
        # Base model: ~16GB, Batch size factor: ~2GB per sample, LoRA factor: ~0.5GB per rank
        estimated_memory = 16.0 + (batch_size * 2.0) + (lora_r * 0.5)  # GB
        
        debug_print(f"[Trial {trial_num}] Memory check: {free:.2f}GB free, estimated need: {estimated_memory:.2f}GB", important=True)
        
        if free < estimated_memory:
            debug_print(f"[Trial {trial_num}] WARNING: Low memory! Free: {free:.2f}GB, Estimated need: {estimated_memory:.2f}GB", important=True)
            return False
        return True
    return True

def load_config():
    """Load configuration from YAML file"""
    import yaml
    config_file = "gpu_config.yaml"
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Default configuration
        return {
            "physical_gpus": "1,2,3",
            "num_gpus": 3,
            "training": {
                "cluster_labeling": True,
                "debug": True
            },
            "deepspeed": {
                "enabled": True,
                "config_file": "ds_config.json"
            }
        }

def parse_args():
    debug_print("parse_args() called")
    parser = argparse.ArgumentParser(description="DeepSpeed LLaMA QLoRA Optimization")
    parser.add_argument("--cluster_labeling", action="store_true", help="Run cluster labeling optimization")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config_file", type=str, default="gpu_config.yaml", help="Configuration file path")
    args = parser.parse_args()
    
    # Load configuration from file
    config = load_config()
    
    # Override with command line arguments if provided
    if args.cluster_labeling:
        config["training"]["cluster_labeling"] = True
    
    debug_print(f"parse_args() returned: {args}")
    debug_print(f"Loaded config: {config}")
    return args, config

def setup_distributed():
    """Setup distributed training with DeepSpeed"""
    debug_print("setup_distributed() called", important=True)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # DeepSpeed will handle distributed setup automatically
    # We just need to get the rank and world size
    rank = get_rank()
    world_size = get_world_size()
    
    debug_print(f"setup_distributed() - Rank: {rank}, World size: {world_size}", important=True)
    print(f"[Rank {rank}] World size: {world_size}")
    
    # Check CUDA availability and respect CUDA_VISIBLE_DEVICES
    debug_print(f"CUDA available: {torch.cuda.is_available()}", important=True)
    if torch.cuda.is_available():
        available_gpus = get_available_gpu_count()
        current_device = get_current_device()
        debug_print(f"Available GPUs (respecting CUDA_VISIBLE_DEVICES): {available_gpus}", important=True)
        debug_print(f"Current CUDA device: {current_device}", important=True)
        
        # Only check GPUs that are actually available to us
        for i in range(available_gpus):
            debug_print(f"GPU {i}: {torch.cuda.get_device_name(i)}", important=True)
        
        # Monitor GPU memory for available GPUs only
        debug_gpu_memory(rank, "setup_distributed() - ", important=True)
        
        # Verify we're not trying to use GPU 0 if it's not available
        if current_device >= available_gpus:
            debug_print(f"WARNING: Current device {current_device} >= available GPUs {available_gpus}", important=True)
    
    return None  # No accelerator needed for DeepSpeed

def load_dataset(cluster_labeling=False):
    """Load the synthetic DMS dataset"""
    debug_print(f"load_dataset() called with cluster_labeling={cluster_labeling}", important=True)
    print("Loading dataset...")
    
    try:
        # Load the CSV files
        lonely_df = pd.read_csv("training_data/lonely_dms_0.5_0.7.csv")
        not_lonely_df1 = pd.read_csv("training_data/serious_dms_0.7_0.7.csv")
        not_lonely_df2 = pd.read_csv("training_data/casual_dms_0.7_0.7.csv")

        if cluster_labeling:
            lonely_with_clusters = pd.read_csv("training_data/lonely_with_clusters.csv")
            serious_with_clusters = pd.read_csv("training_data/serious_with_clusters.csv")
            casual_with_clusters = pd.read_csv("training_data/casual_with_clusters.csv")
            print("Using cluster-based labeling with full datasets:")
            print(f"  Lonely  : {len(lonely_df)}")
            print(f"  Serious : {len(not_lonely_df1)}")
            print(f"  Casual  : {len(not_lonely_df2)}")
            lonely_df['label'] = (1 - lonely_with_clusters['cluster'].iloc[:len(lonely_df)].values).astype(np.float32)
            not_lonely_df1['label'] = (1 - serious_with_clusters['cluster'].iloc[:len(not_lonely_df1)].values).astype(np.float32)
            not_lonely_df2['label'] = (1 - casual_with_clusters['cluster'].iloc[:len(not_lonely_df2)].values).astype(np.float32)
        else:
            print("Using simple binary labeling with full datasets:")
            print("  Lonely samples -> label = 1")
            print("  Non-lonely samples -> label = 0")
            lonely_df['label'] = 1.0
            not_lonely_df1['label'] = 0.0
            not_lonely_df2['label'] = 0.0
            print(f"\nDataset sizes:")
            print(f"  Lonely  : {len(lonely_df)}")
            print(f"  Serious : {len(not_lonely_df1)}")
            print(f"  Casual  : {len(not_lonely_df2)}")

        full_df = pd.concat([lonely_df, not_lonely_df1, not_lonely_df2], ignore_index=True)
        debug_print(f"Full dataset size: {len(full_df)}", important=True)
        
        full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df, val_df = train_test_split(
            full_df,
            test_size=0.2,
            stratify=full_df['label'],
            random_state=42
        )
        debug_print(f"Train/val split: {len(train_df)}/{len(val_df)}", important=True)
        
        print("\nFinal dataset sizes:")
        print(f"  Full    : {len(full_df)}")
        print(f"  Train   : {len(train_df)}")
        print(f"  Valid   : {len(val_df)}")
        print("\nTrain label distribution:")
        print(train_df['label'].value_counts(normalize=True))
        print("\nValidation label distribution:")
        print(val_df['label'].value_counts(normalize=True))
        
        debug_print("load_dataset() completed successfully", important=True)
        return train_df, val_df
        
    except Exception as e:
        debug_print(f"ERROR in load_dataset(): {e}", important=True)
        raise e

def create_model_and_tokenizer(model_name=None, num_labels=2):
    """Create model and tokenizer with LoRA configuration"""
    debug_print(f"create_model_and_tokenizer() called with model_name={model_name}, num_labels={num_labels}", important=True)
    
    if model_name is None:
        model_name = MODEL_PATH
    
    print(f"Loading model: {model_name}")
    debug_print(f"Loading model from: {model_name}", important=True)
    
    try:
        # Load tokenizer
        debug_print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        debug_print(f"Tokenizer loaded, vocab size: {len(tokenizer)}", important=True)
        
        if tokenizer.pad_token is None:
            # define pad_token so batching works
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            # wire up the ID too
            tokenizer.pad_token_id = tokenizer.eos_token_id
        debug_print(f"Pad token: {tokenizer.pad_token}  ID: {tokenizer.pad_token_id}")
        
        # Load model with 8-bit quantization (more compatible with DeepSpeed)
        debug_print("Loading model with 8-bit quantization...", important=True)
        
        # Ensure we're not trying to use GPU 0 if it's not available
        if torch.cuda.is_available():
            available_gpus = get_available_gpu_count()
            debug_print(f"Loading model with {available_gpus} available GPUs", important=True)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.float16,
            device_map=None,  # Disable automatic device mapping - let DeepSpeed handle sharding
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        )
        debug_print("Model loaded successfully", important=True)
        
        # Don't manually move model - let DeepSpeed handle device placement and sharding
        debug_print("Model ready for DeepSpeed sharding", important=True)
        
        # Prepare model for k-bit training
        debug_print("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        debug_print("Model prepared for k-bit training")
        
        # LoRA configuration
        debug_print("Creating LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,  # Will be optimized
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        debug_print(f"LoRA config created: {lora_config}")
        
        # Apply LoRA
        debug_print("Applying LoRA to model...")
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        debug_print("LoRA applied successfully and caching disabled", important=True)
        
        # Move the model onto that exact device (safely)
        device = torch.device(f"cuda:{get_local_rank()}")
        model = model.to(device)
        debug_print(f"Model moved to device {device}", important=True)
        
        # Debug: Verify that only LoRA adapters and classifier head are trainable
        trainable_params = 0
        all_params = 0
        trainable_names = []
        
        for name, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_names.append(name)
        
        print(f"PARAMETER CHECK: {trainable_params:,} trainable / {all_params:,} all parameters ({100 * trainable_params / all_params:.2f}%)")
        debug_print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)", important=True)
        print("Trainable parameters:")
        for name in trainable_names[:10]:  # Show first 10
            print(f"  {name}")
        if len(trainable_names) > 10:
            print(f"  ... and {len(trainable_names) - 10} more")
        
        debug_print("create_model_and_tokenizer() completed successfully", important=True)
        return model, tokenizer
        
    except Exception as e:
        debug_print(f"ERROR in create_model_and_tokenizer(): {e}", important=True)
        raise e

def tokenize_function(examples, tokenizer, max_length=256):  # Reduced from 512 to 256
    """Tokenize the examples"""
    try:
        # Clean the text data - convert to strings and handle NaN values
        texts = []
        for text in examples["DM"]:
            if pd.isna(text) or text is None:
                texts.append("")
            else:
                texts.append(str(text))
        
        result = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None  # Return lists, not tensors - much faster!
        )
        
        # Ensure labels are preserved as floats
        if "label" in examples:
            result["labels"] = [float(label) for label in examples["label"]]
        
        return result
        
    except Exception as e:
        debug_print(f"ERROR in tokenize_function(): {e}", important=True)
        raise e

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    debug_print("compute_metrics() called")
    
    try:
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        
        result = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        debug_print(f"compute_metrics() result: {result}")
        return result
        
    except Exception as e:
        debug_print(f"ERROR in compute_metrics(): {e}")
        raise e

# Get rank from environment variables (works with both DeepSpeed and torchrun)
def get_rank():
    import os
    # For DeepSpeed, use LOCAL_RANK as the primary rank identifier
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    debug_print(f"get_rank() returned: {local_rank}")
    return local_rank

def get_world_size():
    import os
    # For DeepSpeed, use CUDA_VISIBLE_DEVICES or infer from environment
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count() if torch.cuda.is_available() else 1))
    debug_print(f"get_world_size() returned: {world_size}")
    return world_size

def get_local_rank():
    import os
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    debug_print(f"get_local_rank() returned: {local_rank}")
    return local_rank

def get_available_gpu_count():
    """Get the number of available GPUs, respecting CUDA_VISIBLE_DEVICES"""
    import os
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def get_current_device():
    """Get the current device index, respecting CUDA_VISIBLE_DEVICES"""
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0

def validate_gpu_setup():
    """Validate that GPU setup is correct and we're not trying to use unavailable GPUs"""
    if not torch.cuda.is_available():
        debug_print("CUDA not available", important=True)
        return False
    
    available_gpus = get_available_gpu_count()
    current_device = get_current_device()
    
    debug_print(f"GPU validation: {available_gpus} GPUs available, current device: {current_device}", important=True)
    
    # Check CUDA_VISIBLE_DEVICES to understand the mapping
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    debug_print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}", important=True)
    
    if current_device >= available_gpus:
        debug_print(f"ERROR: Current device {current_device} >= available GPUs {available_gpus}", important=True)
        return False
    
    # Check if we can actually use the current device
    try:
        test_tensor = torch.zeros(1, device=f"cuda:{current_device}")
        del test_tensor
        debug_print(f"GPU {current_device} is accessible", important=True)
        
        # Additional check: verify we're not accidentally using the physical GPU 0
        if cuda_visible_devices != 'Not set':
            visible_gpus = [int(x.strip()) for x in cuda_visible_devices.split(',')]
            if 0 not in visible_gpus:
                debug_print(f"✓ Physical GPU 0 is NOT in visible devices: {visible_gpus}", important=True)
            else:
                debug_print(f"⚠ WARNING: Physical GPU 0 IS in visible devices: {visible_gpus}", important=True)
        
        return True
    except Exception as e:
        debug_print(f"ERROR: Cannot access GPU {current_device}: {e}", important=True)
        return False

def setup_tokenizer_padding(tokenizer):
    """Ensure tokenizer has proper padding configuration"""
    if tokenizer.pad_token is None:
        debug_print("Setting pad_token to eos_token", important=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        debug_print(f"Pad token already set: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})", important=True)
    
    # Verify the setup
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        debug_print("ERROR: Failed to set pad_token!", important=True)
        return False
    
    debug_print(f"Tokenizer padding configured: pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}", important=True)
    return True

def is_main_process():
    # For DeepSpeed, rank 0 is the main process
    is_main = get_local_rank() == 0
    debug_print(f"is_main_process() returned: {is_main}")
    return is_main

def is_distributed_available():
    """Check if distributed training is available (DeepSpeed or torch.distributed)"""
    # Check for DeepSpeed environment variables
    deepspeed_available = (
        os.environ.get('LOCAL_RANK') is not None and 
        os.environ.get('WORLD_SIZE') is not None
    )
    
    # Check for torch.distributed
    torch_distributed_available = torch.distributed.is_initialized()
    
    debug_print(f"is_distributed_available() - DeepSpeed: {deepspeed_available}, torch.distributed: {torch_distributed_available}")
    return deepspeed_available or torch_distributed_available

MODEL_PATH = "your model path here"
def precompile_deepspeed_extensions():
    """Precompile DeepSpeed extensions to avoid runtime compilation"""
    debug_print("Precompiling DeepSpeed extensions...", important=True)
    try:
        import deepspeed.ops.adam
        # This will trigger compilation if needed
        _ = deepspeed.ops.adam.FusedAdam
        debug_print("DeepSpeed extensions precompiled successfully", important=True)
    except Exception as e:
        debug_print(f"Warning: Could not precompile DeepSpeed extensions: {e}", important=True)

def create_deepspeed_config():
    """Create DeepSpeed configuration file"""
    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",  # Let TrainingArguments set this dynamically
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,  # Use stage 2 for better PEFT compatibility
            "allgather_partitions": True,
            "allgather_bucket_size": 5e7,  # Reduced bucket size for speed
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,  # Reduced bucket size for speed
            "contiguous_gradients": True,
            "cpu_offload": False,  # Disable CPU offloading for PEFT compatibility
            "cpu_offload_use_pin_memory": False
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "activation_checkpointing": {
            "partition_activations": False,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        },
        "wall_clock_breakdown": False,
        "memory_efficient_linear": False,  # Disable for PEFT compatibility
        "gradient_checkpointing": False,  # Disabled for speed
    }
    
    with open("ds_config.json", "w") as f:
        json.dump(ds_config, f, indent=2)
    
    print("Created DeepSpeed configuration: ds_config.json")

# Tokenization for classification head approach
def tokenize(batch):
    return tokenizer(
        [str(x) for x in batch["DM"]],
        padding="max_length",
        truncation=True,
        max_length=256,  # Reduced from 512 to 256
    )

class QLoRATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Move inputs to the model's device safely
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Use default DataCollatorWithPadding - let Trainer handle device placement

class IncrementalCSVLoggerCallback(TrainerCallback):
    def __init__(self, trial_number, hyperparams, log_path="metrics_log_llama_qlora.csv"):
        self.trial_number = trial_number
        self.hyperparams = hyperparams
        self.log_path = log_path
        self.columns = [
            "trial_number", "step", "epoch", "train_loss", "eval_loss", "eval_accuracy",
            "eval_precision", "eval_recall", "eval_f1", "learning_rate", "weight_decay", "batch_size", "lora_rank", "gradient_accumulation_steps", "num_epochs", "probability_decision_boundary", "is_best"
        ]
        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=self.columns).to_csv(self.log_path, index=False)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        is_best = False
        if os.path.exists("llama_qlora_optuna.csv"):
            df_summary = pd.read_csv("llama_qlora_optuna.csv")
            if not df_summary.empty and logs.get("eval_f1") is not None:
                max_f1 = df_summary["eval_f1"].max()
                is_best = logs.get("eval_f1") >= max_f1
        row = {
            "trial_number": self.trial_number,
            "step": state.global_step,
            "epoch": state.epoch,
            "train_loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
            "eval_accuracy": logs.get("eval_accuracy"),
            "eval_precision": logs.get("eval_precision"),
            "eval_recall": logs.get("eval_recall"),
            "eval_f1": logs.get("eval_f1"),
            "learning_rate": self.hyperparams.get("learning_rate"),
            "weight_decay": self.hyperparams.get("weight_decay"),
            "batch_size": self.hyperparams.get("per_device_train_batch_size"),
            "lora_rank": self.hyperparams.get("lora_r"),
            "gradient_accumulation_steps": self.hyperparams.get("gradient_accumulation_steps"),
            "num_epochs": self.hyperparams.get("num_train_epochs"),
            "probability_decision_boundary": self.hyperparams.get("probability_decision_boundary"),
            "is_best": is_best
        }
        df_existing = pd.read_csv(self.log_path)
        df_new = pd.DataFrame([row])[self.columns]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(self.log_path, index=False)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Ensure inputs are numpy arrays and handle device placement
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    probs = torch.sigmoid(torch.tensor(logits)).squeeze(-1).numpy()
    
    # Use the decision boundary from hyperparameters (will be passed via global variable)
    global current_decision_boundary
    decision_boundary = current_decision_boundary if 'current_decision_boundary' in globals() else 0.5    
    preds = (probs > decision_boundary).astype(int)
    
    # Ensure labels is a numpy array
    labels = np.array(labels)
    
    # Calculate accuracy
    acc = (preds == labels).mean()
    
    # Calculate precision, recall, F1
    try:
        precision = precision_score(labels, preds, average='binary', zero_division=0)
        recall = recall_score(labels, preds, average='binary', zero_division=0)
        f1 = f1_score(labels, preds, average='binary', zero_division=0)
    except:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def model_init(lora_config=None, tokenizer=None):
    print(f"[Rank {get_rank()}] Initializing model...")
    
    # Ensure we're on the correct device
    device = torch.device(f"cuda:{get_local_rank()}")
    print(f"[Rank {get_rank()}] Loading model on device: {device}")
    
    print(f"[Rank {get_rank()}] Loading model with 8-bit quantization...", flush=True)
    start_time = time.time()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=None,  # Disable automatic device mapping - let DeepSpeed handle sharding
        num_labels=1,
        problem_type="single_label_classification",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    )
    
    load_time = time.time() - start_time
    print(f"[Rank {get_rank()}] Model loaded successfully in {load_time:.1f} seconds", flush=True)
    
    print(f"[Rank {get_rank()}] Model loaded, preparing for kbit training...")
    model = prepare_model_for_kbit_training(model)
    
    # Resize embeddings to match tokenizer vocabulary size if tokenizer is provided
    if tokenizer is not None:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        # Set padding token on the model
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply LoRA if config is provided
    if lora_config is not None:
        print(f"[Rank {get_rank()}] Applying LoRA configuration...")
        model = get_peft_model(model, lora_config)
    
    # Disable caching
    model.config.use_cache = False
    print(f"[Rank {get_rank()}] Model caching disabled")
    
    # Move the model onto that exact device
    device = torch.device(f"cuda:{get_local_rank()}")
    model = model.to(device)
    
    print(f"[Rank {get_rank()}] Model initialization complete")
    
    # Debug: Check if model is on the correct device
    if torch.cuda.is_available():
        print(f"[Rank {get_rank()}] Model device: {next(model.parameters()).device}")
        print(f"[Rank {get_rank()}] Current CUDA device: {get_current_device()}")
    
    # Debug: Verify that only LoRA adapters and classifier head are trainable
    trainable_params = 0
    all_params = 0
    trainable_names = []
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)
    
    print(f"[Rank {get_rank()}] PARAMETER CHECK:")
    print(f"  Total parameters: {all_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable percentage: {100 * trainable_params / all_params:.2f}%")
    print(f"  Trainable parameter names:")
    for name in trainable_names:
        print(f"    - {name}")
    
    return model

def hp_space(trial):
    """Hyperparameter space for Optuna optimization - OPTIMIZED FOR 3 L40S GPUs (46GB VRAM each)"""
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 2e-4, 3e-4]),  # Added higher learning rate
        "weight_decay": trial.suggest_categorical("weight_decay", [0.01, 0.05]),  # Added more weight decay options
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [32, 48, 64]),  # Much larger batch sizes for 46GB VRAM
        "lora_r": trial.suggest_categorical("lora_r", [16, 32, 64]),  # Higher LoRA ranks for better capacity
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2]),  # Reduced since we have larger batches
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [1, 2]),  # Added option for 2 epochs
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.03, 0.05]),  # Added higher warmup ratio
        "probability_decision_boundary": trial.suggest_float("probability_decision_boundary", 0.5, 0.5),  # Fixed to 0.5
    }

def save_trial_callback(study, trial):
    try:
        is_best = study.best_trial.number == trial.number
    except ValueError:
        is_best = False
    
    # Log trial results to console instead of CSV
    print(f"[Rank {get_rank()}] Trial {trial.number} completed:")
    print(f"  Parameters: {trial.params}")
    print(f"  Value: {trial.value}")
    print(f"  Is Best: {is_best}")
    print("  " + "-"*50)

def run_single_trial(hyperparams, tokenizer, train_dataset, val_dataset, trial_num):
    """Run a single trial with given hyperparameters across all GPUs"""
    debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Starting trial...", important=True)
    
    try:
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Hyperparameters: {hyperparams}", important=True)
        
        # Ensure tokenizer is properly configured for this trial
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Verifying tokenizer configuration...", important=True)
        if not setup_tokenizer_padding(tokenizer):
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] ERROR: Tokenizer padding setup failed!", important=True)
            return 0.0
        
        # Validate hyperparameters
        required_keys = ["learning_rate", "weight_decay", "per_device_train_batch_size", "lora_r", 
                        "gradient_accumulation_steps", "num_train_epochs", "warmup_ratio", "probability_decision_boundary"]
        
        if not hyperparams or not all(key in hyperparams for key in required_keys):
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] ERROR: Missing required hyperparameters", important=True)
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Required: {required_keys}", important=True)
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Received: {list(hyperparams.keys()) if hyperparams else 'None'}", important=True)
            return 0.0
        
        # Save trial data (only on main process)
        if is_main_process():
            trial_data = {
                "trial_number": trial_num,
                "hyperparams": hyperparams,
                "start_time": datetime.now().isoformat()
            }
            
            with open(f"trial_{trial_num}_data.pkl", "wb") as f:
                pickle.dump(trial_data, f)
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Trial data saved")
        
        # Synchronize all processes before starting trial
        if torch.distributed.is_initialized():
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Waiting for distributed barrier...")
            torch.distributed.barrier()
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Distributed barrier passed")
        
        # Create a fresh model for each trial to avoid PEFT conflicts
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Creating fresh model...", important=True)
        
        # Check CUDA before model loading
        if torch.cuda.is_available():
            debug_gpu_memory(get_rank(), f"[Trial {trial_num}] Before model loading - ", important=True)
        
        try:
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Loading model with 8-bit quantization...", important=True)
            
            # Ensure we're not trying to use GPU 0 if it's not available
            if torch.cuda.is_available():
                available_gpus = get_available_gpu_count()
                debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Loading model with {available_gpus} available GPUs", important=True)
            
            fresh_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16,
                device_map=None,  # Disable automatic device mapping - let DeepSpeed handle sharding
                num_labels=1,
                problem_type="single_label_classification",
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
            )
            
            # Set the classifier head to output single value for binary classification
            fresh_model.classifier = nn.Linear(fresh_model.config.hidden_size, 1)
            
            # Ensure pad_token is set on the model config
            if tokenizer.pad_token is not None:
                fresh_model.config.pad_token_id = tokenizer.pad_token_id
                debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Set pad_token_id to {tokenizer.pad_token_id}", important=True)
            else:
                debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] WARNING: tokenizer.pad_token is None!", important=True)
            
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Model loaded successfully!", important=True)
            
            # Don't manually move model - let DeepSpeed handle device placement and sharding
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Model ready for DeepSpeed sharding", important=True)
            
            # Monitor GPU memory after model loading
            if torch.cuda.is_available():
                debug_gpu_memory(get_rank(), f"[Trial {trial_num}] After model loading - ", important=True)
                
        except Exception as e:
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Error loading model: {e}", important=True)
            raise e
        
        # Prepare for kbit training
        fresh_model = prepare_model_for_kbit_training(fresh_model)
        
        # Create LoRA config with trial hyperparameters
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=hyperparams["lora_r"],
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Apply LoRA to fresh model
        fresh_model = get_peft_model(fresh_model, lora_config)
        fresh_model.config.use_cache = False
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] LoRA applied successfully and caching disabled", important=True)
        
        # Move the model onto that exact device before training (safely)
        device = torch.device(f"cuda:{get_local_rank()}")
        fresh_model = fresh_model.to(device)
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Model moved to device {device}", important=True)
        
        # Training arguments with DeepSpeed
        training_args = TrainingArguments(
            output_dir=f"tuned_model_llama_qlora_trial_{trial_num}",
            learning_rate=hyperparams["learning_rate"],
            per_device_train_batch_size=hyperparams["per_device_train_batch_size"],
            per_device_eval_batch_size=hyperparams["per_device_train_batch_size"],
            num_train_epochs=hyperparams["num_train_epochs"],
            weight_decay=hyperparams["weight_decay"],
            warmup_ratio=hyperparams["warmup_ratio"],
            gradient_accumulation_steps=hyperparams["gradient_accumulation_steps"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=50,  # Less frequent logging
            save_total_limit=0,  # Don't save any checkpoints
            dataloader_pin_memory=True,  # Enable for better performance
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
            # DeepSpeed configuration
            deepspeed="ds_config.json",
            fp16=True,
            local_rank=get_local_rank(),
            dataloader_num_workers=0,  # Disable multiprocessing for speed - GPU-bound training doesn't benefit from workers
            # Device placement settings
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=25,
            # Additional settings for speed
            dataloader_drop_last=True,
            group_by_length=False,
            # Performance optimizations
            # Removed dataloader settings that conflict with num_workers=0
        )
        
        # Data collator - use default, let Trainer handle device placement
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Debug: Check dataset format before creating trainer
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Train dataset format: {train_dataset.format}", important=True)
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Val dataset format: {val_dataset.format}", important=True)
        

        
        # Create trainer with custom loss function
        trainer = QLoRATrainer(
            model=fresh_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train the model
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Starting training...", important=True)
        print(f"[Rank {get_rank()}] [Trial {trial_num}] Starting training...")
        
        # Debug: Check model device before training
        if torch.cuda.is_available():
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Model device before training: {next(fresh_model.parameters()).device}", important=True)
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Current CUDA device: {get_current_device()}", important=True)
        
        train_result = trainer.train()
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Training completed", important=True)
        
        # Evaluate
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Evaluating...", important=True)
        print(f"[Rank {get_rank()}] [Trial {trial_num}] Evaluating...")
        eval_result = trainer.evaluate()
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Evaluation completed", important=True)
        
        # Get the best metric
        best_metric = eval_result.get("eval_f1", 0.0)
        
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Trial completed. Best metric: {best_metric}", important=True)
        print(f"[Rank {get_rank()}] [Trial {trial_num}] Trial completed. Best metric: {best_metric}")
        
        # Clean up hyperparameter file
        hyperparams_file = f"trial_{trial_num}_hyperparams.json"
        if os.path.exists(hyperparams_file):
            try:
                os.remove(hyperparams_file)
                debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Cleaned up {hyperparams_file}")
            except Exception as e:
                debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Warning: Could not clean up {hyperparams_file}: {e}")
        
        # Clear GPU cache after each trial
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] GPU cache cleared and synchronized after trial", important=True)
        
        # Save metrics (only on main process)
        if is_main_process():
            metrics_data = {
                "trial_number": trial_num,
                "hyperparams": hyperparams,
                "train_loss": train_result.training_loss,
                "eval_metrics": eval_result,
                "best_metric": best_metric,
                "end_time": datetime.now().isoformat()
            }
            
            # Append to CSV
            df = pd.DataFrame([metrics_data])
            if os.path.exists("metrics_log_llama_deepspeed.csv"):
                df.to_csv("metrics_log_llama_deepspeed.csv", mode='a', header=False, index=False)
            else:
                df.to_csv("metrics_log_llama_deepspeed.csv", index=False)
            debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Metrics saved to CSV")
        
        return best_metric
        
    except Exception as e:
        debug_print(f"[Rank {get_rank()}] [Trial {trial_num}] Trial failed with error: {e}", important=True)
        print(f"[Rank {get_rank()}] [Trial {trial_num}] Trial failed with error: {e}")
        return 0.0

def objective(trial, model, tokenizer, train_dataset, val_dataset, accelerator):
    """Legacy Optuna objective function - kept for compatibility"""
    return run_single_trial(hp_space(trial), tokenizer, train_dataset, val_dataset, trial.number)
    
    try:
        # Get hyperparameters
        hyperparams = hp_space(trial)
        debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Hyperparameters: {hyperparams}", important=True)
        
        # Save trial data (only on main process)
        if is_main_process():
            trial_data = {
                "trial_number": trial.number,
                "hyperparams": hyperparams,
                "start_time": datetime.now().isoformat()
            }
            
            with open(f"trial_{trial.number}_data.pkl", "wb") as f:
                pickle.dump(trial_data, f)
            debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Trial data saved")
        
        # Synchronize all processes before starting trial
        if torch.distributed.is_initialized():
            debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Waiting for distributed barrier...")
            torch.distributed.barrier()
            debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Distributed barrier passed")
        
        # Create a fresh model for each trial to avoid PEFT conflicts
        debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Creating fresh model...", important=True)
        
        # Check CUDA before model loading
        if torch.cuda.is_available():
            debug_gpu_memory(get_rank(), f"[Trial {trial.number}] Before model loading - ", important=True)
        
        try:
            debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Loading model with 8-bit quantization...", important=True)
            fresh_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16,
                device_map=None,  # Disable automatic device mapping - let DeepSpeed handle sharding
                num_labels=1,
                problem_type="single_label_classification",
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
            )
            debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Model loaded successfully!", important=True)
            
            # Don't manually move model - let DeepSpeed handle device placement and sharding
            debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Model ready for DeepSpeed sharding", important=True)
            
            # Monitor GPU memory after model loading
            if torch.cuda.is_available():
                debug_gpu_memory(get_rank(), f"[Trial {trial.number}] After model loading - ", important=True)
                
        except Exception as e:
            debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Error loading model: {e}", important=True)
            raise e
        
        # Prepare for kbit training
        fresh_model = prepare_model_for_kbit_training(fresh_model)
        
        # Create LoRA config with trial hyperparameters
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=hyperparams["lora_r"],
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Apply LoRA to fresh model
        fresh_model = get_peft_model(fresh_model, lora_config)
        debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] LoRA applied successfully", important=True)
        
        # Training arguments with DeepSpeed - OPTIMIZED FOR SPEED
        training_args = TrainingArguments(
            output_dir=f"tuned_model_llama_qlora_trial_{trial.number}",
            learning_rate=hyperparams["learning_rate"],
            per_device_train_batch_size=hyperparams["per_device_train_batch_size"],
            per_device_eval_batch_size=hyperparams["per_device_train_batch_size"],
            num_train_epochs=hyperparams["num_train_epochs"],
            weight_decay=hyperparams["weight_decay"],
            warmup_ratio=hyperparams["warmup_ratio"],
            gradient_accumulation_steps=hyperparams["gradient_accumulation_steps"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=100,  # Less frequent logging
            save_total_limit=1,  # Don't save any checkpoints
            dataloader_pin_memory=True,  # Enable for better performance
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
            # DeepSpeed configuration
            deepspeed="ds_config.json",
            fp16=True,
            dataloader_num_workers=0,  # Disable multiprocessing for speed
        )
        
        # Data collator with proper device handling
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Create trainer with custom loss function
        trainer = QLoRATrainer(
            model=fresh_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train the model
        debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Starting training...", important=True)
        print(f"[Rank {get_rank()}] [Trial {trial.number}] Starting training...")
        train_result = trainer.train()
        debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Training completed", important=True)
        
        # Evaluate
        debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Evaluating...", important=True)
        print(f"[Rank {get_rank()}] [Trial {trial.number}] Evaluating...")
        eval_result = trainer.evaluate()
        debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Evaluation completed", important=True)
        
        # Get the best metric
        best_metric = eval_result.get("eval_f1", 0.0)
        
        debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Trial completed. Best metric: {best_metric}", important=True)
        print(f"[Rank {get_rank()}] [Trial {trial.number}] Trial completed. Best metric: {best_metric}")
        
        # Save metrics (only on main process)
        if is_main_process():
            metrics_data = {
                "trial_number": trial.number,
                "hyperparams": hyperparams,
                "train_loss": train_result.training_loss,
                "eval_metrics": eval_result,
                "best_metric": best_metric,
                "end_time": datetime.now().isoformat()
            }
            
            # Append to CSV
            df = pd.DataFrame([metrics_data])
            if os.path.exists("metrics_log_llama_deepspeed.csv"):
                df.to_csv("metrics_log_llama_deepspeed.csv", mode='a', header=False, index=False)
            else:
                df.to_csv("metrics_log_llama_deepspeed.csv", index=False)
            debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Metrics saved to CSV")
        
        return best_metric
        
    except Exception as e:
        debug_print(f"[Rank {get_rank()}] [Trial {trial.number}] Trial failed with error: {e}", important=True)
        print(f"[Rank {get_rank()}] [Trial {trial.number}] Trial failed with error: {e}")
        return 0.0

def run_optuna_optimization(cluster_labeling=False):
    """Run Optuna optimization with DeepSpeed"""
    debug_print(f"run_optuna_optimization() called with cluster_labeling={cluster_labeling}", important=True)
    print(f"[Rank {get_rank()}] Starting DeepSpeed Optuna optimization...")
    
    try:
        # Setup distributed training
        debug_print("Setting up distributed training...", important=True)
        setup_distributed()
        
        # Precompile DeepSpeed extensions
        debug_print("Precompiling DeepSpeed extensions...", important=True)
        precompile_deepspeed_extensions()
        
        # Create DeepSpeed config
        debug_print("Creating DeepSpeed config...", important=True)
        create_deepspeed_config()
        
        # Load dataset
        debug_print("Loading dataset...", important=True)
        train_df, val_df = load_dataset(cluster_labeling)
        debug_print(f"Dataset loaded: train={len(train_df)}, val={len(val_df)}", important=True)
        
        # Create tokenizer only (model will be created in each trial)
        debug_print("Creating tokenizer...", important=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        debug_print(f"Tokenizer created, vocab size: {len(tokenizer)}", important=True)
        
        # Ensure proper padding configuration
        if not setup_tokenizer_padding(tokenizer):
            debug_print("ERROR: Failed to setup tokenizer padding!", important=True)
            raise RuntimeError("Tokenizer padding setup failed")
        
        debug_print(f"Pad token: {tokenizer.pad_token}  ID: {tokenizer.pad_token_id}")
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        debug_print(f"Datasets created: train={len(train_dataset)}, val={len(val_dataset)}", important=True)
        
        # Tokenize datasets
        debug_print(f"[Rank {get_rank()}] Starting tokenization of training dataset...", important=True)
        train_dataset = train_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=[col for col in train_dataset.column_names if col != 'label']
        )
        debug_print(f"[Rank {get_rank()}] Training dataset tokenization completed", important=True)
        
        val_dataset = val_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=[col for col in val_dataset.column_names if col != 'label']
        )
        debug_print(f"[Rank {get_rank()}] Validation dataset tokenization completed", important=True)
        
        # Set dataset format to PyTorch tensors to ensure proper device placement
        debug_print(f"[Rank {get_rank()}] Setting dataset format to PyTorch tensors...", important=True)
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        debug_print(f"[Rank {get_rank()}] Dataset format set to PyTorch tensors", important=True)
        
        # Clear GPU cache before starting optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            debug_print(f"[Rank {get_rank()}] GPU cache cleared", important=True)
        
        # All ranks participate in the same trials to avoid deadlocks
        debug_print("All ranks participating in optimization...", important=True)
        print(f"[Rank {get_rank()}] Participating in optimization...")
        
        # Create hyperparameter grid for systematic search
        def create_hyperparameter_grid():
            """Create a grid of hyperparameters for systematic search - optimized for 3 L40S GPUs (46GB VRAM each)"""
            return {
                "learning_rate": [1e-4, 2e-4, 3e-4],  # Added higher learning rate
                "per_device_train_batch_size": [32, 48, 64],  # Much larger batch sizes for 46GB VRAM
                "gradient_accumulation_steps": [1, 2],  # Reduced since we have larger batches
                "lora_r": [16, 32, 64],  # Higher LoRA ranks for better capacity
                "probability_decision_boundary": [0.5],  # Fixed to reduce trials
                # Fixed values:
                "num_train_epochs": [1, 2],  # Added option for 2 epochs
                "weight_decay": [0.01, 0.05],  # Added more weight decay options
            }
        
        # Generate all combinations from the grid
        import itertools
        grid = create_hyperparameter_grid()
        grid_keys = list(grid.keys())
        grid_values = list(grid.values())
        
        # Generate all combinations
        all_combinations = list(itertools.product(*grid_values))
        hyperparam_sets = []
        
        for combination in all_combinations:
            hyperparams = dict(zip(grid_keys, combination))
            # Add fixed values that aren't in the grid
            hyperparams["warmup_ratio"] = 0.03  # Fixed warmup ratio
            hyperparam_sets.append(hyperparams)
        
        total_trials = len(hyperparam_sets)
        debug_print(f"Generated {total_trials} hyperparameter combinations for grid search", important=True)
        
        if is_main_process():
            print(f"Grid search will run {total_trials} trials:")
            print(f"  Learning rates: {grid['learning_rate']}")
            print(f"  Batch sizes: {grid['per_device_train_batch_size']}")
            print(f"  Gradient accumulation: {grid['gradient_accumulation_steps']}")
            print(f"  LoRA ranks: {grid['lora_r']}")
            print(f"  Decision boundaries: {grid['probability_decision_boundary']}")
        
        # Synchronize all processes before starting optimization
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Run grid search
        best_metric_overall = 0.0
        best_hyperparams = None
        
        for trial_num, hyperparams in enumerate(hyperparam_sets):
            debug_print(f"[Rank {get_rank()}] Starting trial {trial_num}/{total_trials}...", important=True)
            print(f"[Rank {get_rank()}] Starting trial {trial_num}/{total_trials} with hyperparams: {hyperparams}")
            
            # Memory check removed - let training proceed and fail naturally if needed
            
            # All ranks run the same training with the same hyperparameters
            best_metric = run_single_trial(hyperparams, tokenizer, train_dataset, val_dataset, trial_num)
            
            # Track best result (only main process saves)
            if is_main_process():
                if best_metric > best_metric_overall:
                    best_metric_overall = best_metric
                    best_hyperparams = hyperparams
                debug_print(f"[Rank {get_rank()}] Trial {trial_num} completed with metric: {best_metric} (best so far: {best_metric_overall})", important=True)
            
            debug_print(f"[Rank {get_rank()}] Trial {trial_num} completed with metric: {best_metric}", important=True)
        
        if is_main_process():
            debug_print("Optimization completed", important=True)
            print("Optimization completed!")
            print(f"Best metric: {best_metric_overall}")
            print(f"Best hyperparameters: {best_hyperparams}")
            
            # Save best hyperparameters
            debug_print("Saving best hyperparameters...", important=True)
            save_best_hyperparameters_simple(best_hyperparams, best_metric_overall, "cluster" if cluster_labeling else "binary")
            
            # Print completion message for shell script monitoring
            print("Optimization and training completed!")
        
        return best_metric_overall
            
    except Exception as e:
        debug_print(f"ERROR in run_optuna_optimization(): {e}", important=True)
        raise e



def save_best_hyperparameters(study, labeling_suffix):
    """Save best hyperparameters to file"""
    best_trial = study.best_trial
    best_hyperparams = best_trial.params
    
    # Save to pickle for better serialization
    results = {
        "best_trial_number": best_trial.number,
        "best_trial_value": best_trial.value,
        "best_hyperparams": best_hyperparams,
        "labeling_suffix": labeling_suffix
    }
    
    with open(f"best_hyperparams_{labeling_suffix}.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Also save as text for human readability
    with open(f"best_hyperparams_{labeling_suffix}.txt", "w") as f:
        f.write(f"Best Trial Number: {best_trial.number}\n")
        f.write(f"Best Trial Value: {best_trial.value}\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_hyperparams.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"[Rank {get_rank()}] Best hyperparameters saved for {labeling_suffix} labeling")
    return best_hyperparams

def save_best_hyperparameters_simple(best_hyperparams, best_metric, labeling_suffix):
    """Save best hyperparameters to file (simple version)"""
    # Save to pickle for better serialization
    results = {
        "best_metric": best_metric,
        "best_hyperparams": best_hyperparams,
        "labeling_suffix": labeling_suffix
    }
    
    with open(f"best_hyperparams_{labeling_suffix}.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Also save as text for human readability
    with open(f"best_hyperparams_{labeling_suffix}.txt", "w") as f:
        f.write(f"Best Metric: {best_metric}\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_hyperparams.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"[Rank {get_rank()}] Best hyperparameters saved for {labeling_suffix} labeling")
    return best_hyperparams

def load_best_hyperparameters(labeling_suffix):
    """Load best hyperparameters from file"""
    try:
        with open(f"best_hyperparams_{labeling_suffix}.pkl", "rb") as f:
            results = pickle.load(f)
        return results["best_hyperparams"]
    except FileNotFoundError:
        print(f"[Rank {get_rank()}] Warning: Could not find best hyperparameters file for {labeling_suffix}")
        return None

def run_final_training(best_hyperparams, labeling_suffix, tokenizer=None):
    """Run final training with best hyperparameters"""
    print("=" * 50)
    print("FINAL FINETUNING WITH BEST HYPERPARAMETERS")
    print("=" * 50)
    
    if best_hyperparams is None:
        print(f"[Rank {get_rank()}] Error: No best hyperparameters found for {labeling_suffix}")
        return
    
    final_output_dir = f"./final_llama_qlora_{labeling_suffix}"
    print(f"Output directory: {final_output_dir}")
    print(f"Best hyperparameters: {best_hyperparams}")
    
    # Set global decision boundary for final training
    global current_decision_boundary
    current_decision_boundary = best_hyperparams["probability_decision_boundary"]
    print(f"Using decision boundary: {current_decision_boundary}")
    
    # Load and prepare datasets
    train_df, val_df = load_dataset("cluster" in labeling_suffix)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=[col for col in train_dataset.column_names if col != 'label']
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=[col for col in val_dataset.column_names if col != 'label']
    )
    
    # Set dataset format to PyTorch tensors
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            # define pad_token so batching works
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            # wire up the ID too
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Calculate training parameters
    total_batch_size = best_hyperparams["per_device_train_batch_size"] * best_hyperparams["gradient_accumulation_steps"]
    total_steps = len(train_dataset) // best_hyperparams["per_device_train_batch_size"] * best_hyperparams["num_train_epochs"]
    warmup_steps = int(total_steps * best_hyperparams["warmup_ratio"])
    
    # No DeepSpeed configuration needed for PyTorch distributed training
    
    # Prepare final training arguments
    final_training_kwargs = {
        "output_dir": final_output_dir,
        "learning_rate": best_hyperparams["learning_rate"],
        "weight_decay": best_hyperparams["weight_decay"],
        "per_device_train_batch_size": best_hyperparams["per_device_train_batch_size"],
        "per_device_eval_batch_size": best_hyperparams["per_device_train_batch_size"],
        "num_train_epochs": best_hyperparams["num_train_epochs"],
        "gradient_accumulation_steps": best_hyperparams["gradient_accumulation_steps"],
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_dir": f"./logs_llama_qlora_final_{labeling_suffix}",
        "logging_steps": 50,
        "report_to": "none",
        "disable_tqdm": False,
        "fp16": True,
        "dataloader_pin_memory": True,
        "ddp_find_unused_parameters": False,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": "linear",
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 3,
        # Critical distributed training settings for PyTorch
        "local_rank": get_local_rank(),  # Get from environment variables
        "ddp_backend": "nccl",  # Use NCCL for GPU communication
        "ddp_timeout": 3600,  # 60 minutes timeout for DDP initialization
        "ddp_bucket_cap_mb": 25,  # Reduce bucket size for better compatibility
        "dataloader_num_workers": 0,  # Disable multiprocessing for dataloader
    }
    
    final_training_args = TrainingArguments(**final_training_kwargs)
    
    # Final LoRA configuration
    final_lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=best_hyperparams["lora_r"],
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    # Final trainer
    final_trainer = QLoRATrainer(
        model_init=model_init,
        args=final_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)]
    )
    
    # Apply LoRA to the model
    final_trainer.model = get_peft_model(final_trainer.model, final_lora_config)
    
    # Move the model onto that exact device
    device = torch.device(f"cuda:{get_local_rank()}")
    final_trainer.model = final_trainer.model.to(device)
    
    print(f"[Rank {get_rank()}] Starting final training with {best_hyperparams['num_train_epochs']} epochs...", flush=True)
    final_trainer.train()
    
    # Evaluate final model
    print("\nEvaluating final model...")
    final_metrics = final_trainer.evaluate()
    
    print(f"\nFinal Model Results:")
    print(f"  Accuracy: {final_metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {final_metrics.get('precision', 0):.4f}")
    print(f"  Recall: {final_metrics.get('recall', 0):.4f}")
    print(f"  F1 Score: {final_metrics.get('f1', 0):.4f}")
    print(f"  Loss: {final_metrics.get('eval_loss', 0):.4f}")
    
    # Save final model
    print(f"\nSaving final model to {final_output_dir}...")
    final_trainer.save_model()
    
    # Log final metrics to console
    print(f"\n[Rank {get_rank()}] Final Model Results:")
    print(f"  Labeling Approach: {labeling_suffix}")
    print(f"  Final Accuracy: {final_metrics.get('accuracy', 0):.4f}")
    print(f"  Final Precision: {final_metrics.get('precision', 0):.4f}")
    print(f"  Final Recall: {final_metrics.get('recall', 0):.4f}")
    print(f"  Final F1 Score: {final_metrics.get('f1', 0):.4f}")
    print(f"  Final Loss: {final_metrics.get('eval_loss', 0):.4f}")

def main():
    """Main function for DeepSpeed optimization"""
    debug_print("main() called", important=True)
    
    try:
        args, config = parse_args()
        debug_print(f"main() - args: {args}", important=True)
        debug_print(f"main() - config: {config}", important=True)
        
        # Safety check: Ensure we're not trying to use GPU 0 when it's not available
        if torch.cuda.is_available():
            available_gpus = get_available_gpu_count()
            debug_print(f"Available GPUs: {available_gpus}", important=True)
            
            # If we have fewer GPUs than expected, adjust the local rank
            if args.local_rank != -1 and args.local_rank >= available_gpus:
                debug_print(f"WARNING: Local rank {args.local_rank} >= available GPUs {available_gpus}, adjusting", important=True)
                args.local_rank = args.local_rank % available_gpus
                debug_print(f"Adjusted local rank to: {args.local_rank}", important=True)
        
        # Pin the process to its GPU immediately
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            debug_print(f"Process {args.local_rank} pinned to GPU {get_current_device()}", important=True)
        
        # Validate GPU setup
        if not validate_gpu_setup():
            debug_print("ERROR: GPU setup validation failed", important=True)
            raise RuntimeError("GPU setup validation failed - check CUDA_VISIBLE_DEVICES and GPU availability")
        
        print("Starting DeepSpeed LLaMA QLoRA optimization...")
        print(f"Cluster labeling: {config['training']['cluster_labeling']}")
        print(f"Physical GPUs: {config['physical_gpus']}")
        print(f"Number of GPUs: {config['num_gpus']}")
        
        # Check CUDA availability at startup
        debug_print(f"CUDA available at startup: {torch.cuda.is_available()}", important=True)
        if torch.cuda.is_available():
            available_gpus = get_available_gpu_count()
            current_device = get_current_device()
            debug_print(f"Available GPUs (respecting CUDA_VISIBLE_DEVICES): {available_gpus}", important=True)
            debug_print(f"Current CUDA device: {current_device}", important=True)
            for i in range(available_gpus):
                debug_print(f"GPU {i}: {torch.cuda.get_device_name(i)}", important=True)
            
            # Monitor GPU memory at startup
            debug_gpu_memory(None, "main() startup - ", important=True)
        
        # Only main process runs Optuna (on CPU), all processes participate in training
        if is_main_process():
            debug_print("Main process: Running Optuna optimization and coordinating distributed training...", important=True)
            print("Main process: Running Optuna optimization and coordinating distributed training...")
            
            # Run the optimization (Optuna runs on CPU, training uses all GPUs)
            study = run_optuna_optimization(cluster_labeling=config['training']['cluster_labeling'])
            
            debug_print("Main process: Optimization completed", important=True)
            print("Optimization completed!")
            print("Optimization completed for cluster labeling!" if config['training']['cluster_labeling'] else "Optimization completed for binary labeling!")
        else:
            debug_print(f"Non-main process: Participating in distributed training...", important=True)
            print(f"[Rank {get_rank()}] Non-main process: Participating in distributed training...")
            
            # For DeepSpeed, non-main processes should just participate in the same function
            # DeepSpeed will handle the coordination automatically
            run_optuna_optimization(cluster_labeling=config['training']['cluster_labeling'])
            
        debug_print("main() completed successfully", important=True)
        
    except Exception as e:
        debug_print(f"ERROR in main(): {e}", important=True)
        raise e

if __name__ == "__main__":
    debug_print("Script started - __main__ block")
    main()
    debug_print("Script completed")