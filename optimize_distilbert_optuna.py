import os
import argparse
import shutil
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import gc

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertConfig,
    DistilBertModel,
    DistilBertPreTrainedModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from datasets import Dataset
import optuna
from optuna.pruners import MedianPruner

def tokenize(batch):
    # Ensure DM column contains strings
    texts = [str(text) if text is not None else "" for text in batch["DM"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

class DistilBertBinaryClassifier(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)  # Single logit output
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]  # (batch_size, seq_len, hidden_size)
        pooled_output = hidden_state[:, 0]  # CLS token
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())  # force float label

        return {"loss": loss, "logits": logits}

# Use standard Trainer - the custom model handles the loss computation

class F1EarlyStoppingCallback(EarlyStoppingCallback):
    """Custom early stopping callback based on F1 score instead of loss."""
    
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.best_f1 = -float('inf')
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        
        eval_f1 = metrics.get("eval_f1")
        if eval_f1 is None:
            return
        
        if eval_f1 > self.best_f1 + self.early_stopping_threshold:
            self.best_f1 = eval_f1
            self.best_model_checkpoint = state.best_model_checkpoint
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
        
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

# Removed IncrementalCSVLoggerCallback for faster execution

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).squeeze().numpy()
    preds = (probs > 0.5).astype(int)
    acc = (preds == np.array(labels)).mean()
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

def model_init():
    # Re-initialize the model each time
    torch.cuda.empty_cache()
    config = DistilBertConfig.from_pretrained(MODEL_PATH)
    config.problem_type = "single_label_classification"
    config.num_labels = 1
    return DistilBertBinaryClassifier.from_pretrained(MODEL_PATH, config=config)

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"]),
        "adam_beta1": trial.suggest_float("adam_beta1", 0.85, 0.95),
        "adam_beta2": trial.suggest_float("adam_beta2", 0.98, 0.999),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-8, 1e-6, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 20),
        "early_stopping_patience": trial.suggest_int("early_stopping_patience", 2, 5)
    }

def save_trial_callback(study, trial):
    # Removed CSV writing for faster execution
    pass

def objective(trial, study=None, cluster_labeling=False):
    global current_trial_hp
    gc.collect()
    torch.cuda.empty_cache()
    hp = hp_space(trial)
    current_trial_hp = hp

    # Create output directory for this trial with labeling method
    model_suffix = "_cluster" if cluster_labeling else "_binary"
    output_dir = f"./tuned_model_distilbert_trial_{trial.number}{model_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up previous checkpoints from this trial directory
    # In distributed training, only do this before training starts to avoid race conditions
    if not torch.distributed.is_initialized():
        # Single GPU training - safe to clean up
        checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if len(checkpoint_dirs) > 3:  # Keep last 3 checkpoints (most recent + 2 before it)
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
            for old_checkpoint in checkpoint_dirs[:-3]:
                try:
                    shutil.rmtree(old_checkpoint)
                    print(f"Deleted old checkpoint: {old_checkpoint}")
                except (OSError, FileNotFoundError) as e:
                    print(f"Warning: Could not delete {old_checkpoint}: {e}")
    elif torch.distributed.get_rank() == 0:
        # Distributed training - only clean up existing checkpoints before starting
        # Don't clean up during training to avoid race conditions
        checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if len(checkpoint_dirs) > 3:  # Keep last 3 checkpoints (most recent + 2 before it)
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
            for old_checkpoint in checkpoint_dirs[:-3]:
                try:
                    shutil.rmtree(old_checkpoint)
                    print(f"Deleted old checkpoint before training: {old_checkpoint}")
                except (OSError, FileNotFoundError) as e:
                    print(f"Warning: Could not delete {old_checkpoint}: {e}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        num_train_epochs=hp["num_train_epochs"],
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_dir="./logs_distilbert",
        logging_steps=50,
        report_to="none",
        disable_tqdm=True,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        gradient_accumulation_steps=1,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True
    )

    # Create early stopping callback
    early_stopping = F1EarlyStoppingCallback(
        early_stopping_patience=hp["early_stopping_patience"],
        early_stopping_threshold=0.001
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    trainer.train()
    metrics = trainer.evaluate()
    
    # Clean up all checkpoints after training is complete
    # In distributed training, synchronize all processes before cleanup
    if torch.distributed.is_initialized():
        # Wait for all processes to finish training
        torch.distributed.barrier()
    
    # Only rank 0 does the cleanup to avoid conflicts
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        for checkpoint in checkpoint_dirs:
            try:
                shutil.rmtree(checkpoint)
                print(f"Deleted checkpoint after training: {checkpoint}")
            except (OSError, FileNotFoundError) as e:
                print(f"Warning: Could not delete {checkpoint}: {e}")

    try:
        is_best = study.best_trial.number == trial.number
    except ValueError:
        is_best = False

    trial_summary = {
        "trial_number": trial.number,
        **hp,
        "eval_accuracy": metrics.get("eval_accuracy"),
        "eval_f1": metrics.get("eval_f1"),
        "eval_loss": metrics.get("eval_loss"),
        "is_best": is_best
    }
    
    # Removed CSV writing for faster execution

    return metrics.get("eval_f1", 0.0)

def train_final_model(best_hyperparams, cluster_labeling=False):
    """Train the final model with the best hyperparameters."""
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*60)
    print(f"Best hyperparameters: {best_hyperparams}")
    
    # Set global hyperparams for model_init
    global current_trial_hp
    current_trial_hp = best_hyperparams
    
    # Determine model name based on cluster labeling
    model_suffix = "_cluster" if cluster_labeling else "_binary"
    final_model_name = f"final_distilbert_model{model_suffix}"
    
    # Create output directory for final model
    output_dir = f"./{final_model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up any existing checkpoints in final model directory (only if not in distributed training)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if len(checkpoint_dirs) > 3:  # Keep last 3 checkpoints (most recent + 2 before it)
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
            for old_checkpoint in checkpoint_dirs[:-3]:
                try:
                    shutil.rmtree(old_checkpoint)
                    print(f"Deleted old checkpoint in final model: {old_checkpoint}")
                except (OSError, FileNotFoundError) as e:
                    print(f"Warning: Could not delete {old_checkpoint}: {e}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=best_hyperparams["learning_rate"],
        weight_decay=best_hyperparams["weight_decay"],
        per_device_train_batch_size=best_hyperparams["per_device_train_batch_size"],
        num_train_epochs=best_hyperparams["num_train_epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_dir=f"./logs_final_distilbert{model_suffix}",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        gradient_accumulation_steps=1,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True
    )

    # Create early stopping callback
    early_stopping = F1EarlyStoppingCallback(
        early_stopping_patience=best_hyperparams["early_stopping_patience"],
        early_stopping_threshold=0.001
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    print("Starting final training...")
    trainer.train()
    
    # Evaluate final model
    final_metrics = trainer.evaluate()
    print(f"\nFinal model evaluation:")
    print(f"Accuracy: {final_metrics.get('eval_accuracy', 0):.4f}")
    print(f"F1 Score: {final_metrics.get('eval_f1', 0):.4f}")
    print(f"Loss: {final_metrics.get('eval_loss', 0):.4f}")
    
    # Clean up all checkpoints after final training
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    for checkpoint in checkpoint_dirs:
        shutil.rmtree(checkpoint)
        print(f"Deleted checkpoint after final training: {checkpoint}")
    
    # Save final model
    final_model_path = f"./{final_model_name}_best"
    trainer.save_model(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    return final_metrics

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DistilBERT hyperparameter optimization with cluster labeling option')
    parser.add_argument('--cluster_labeling', action='store_true', 
                       help='Use cluster-based labeling (True) or simple binary labeling (False)')
    args = parser.parse_args()
    
    print(f"Using cluster labeling: {args.cluster_labeling}")
    
    # Set up logging
    import logging
    model_suffix = "_cluster" if args.cluster_labeling else "_binary"
    log_filename = f"distilbert_optimization{model_suffix}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting DistilBERT optimization with cluster_labeling={args.cluster_labeling}")
    
    # Load and prepare data
    lonely_df = pd.read_csv("lonely_ckpt_0.5_0.7.csv")
    not_lonely_df1 = pd.read_csv("serious_ckpt_0.7_0.7.csv")
    not_lonely_df2 = pd.read_csv("casual_ckpt_0.7_0.7.csv")
    
    if args.cluster_labeling:
        # Load cluster data for cluster-based labeling
        lonely_with_clusters = pd.read_csv("lonely_with_clusters.csv")
        serious_with_clusters = pd.read_csv("serious_with_clusters.csv")
        casual_with_clusters = pd.read_csv("casual_with_clusters.csv")

        print("Using cluster-based labeling with full datasets:")
        print(f"  Lonely  : {len(lonely_df)}")
        print(f"  Serious : {len(not_lonely_df1)}")
        print(f"  Casual  : {len(not_lonely_df2)}")

        # Cluster-based labeling (using full datasets)
        lonely_df['label'] = (1 - lonely_with_clusters['cluster'].iloc[:len(lonely_df)].values).astype(np.float32)
        not_lonely_df1['label'] = (1 - serious_with_clusters['cluster'].iloc[:len(not_lonely_df1)].values).astype(np.float32)
        not_lonely_df2['label'] = (1 - casual_with_clusters['cluster'].iloc[:len(not_lonely_df2)].values).astype(np.float32)
        
    else:
        # Simple binary labeling
        print("Using simple binary labeling with full datasets:")
        print("  Lonely samples -> label = 1")
        print("  Non-lonely samples -> label = 0")
        
        # Simple binary labeling: lonely = 1, not lonely = 0
        lonely_df['label'] = 1.0
        not_lonely_df1['label'] = 0.0
        not_lonely_df2['label'] = 0.0
        
        print(f"\nDataset sizes:")
        print(f"  Lonely  : {len(lonely_df)}")
        print(f"  Serious : {len(not_lonely_df1)}")
        print(f"  Casual  : {len(not_lonely_df2)}")

    full_df = pd.concat([lonely_df, not_lonely_df1, not_lonely_df2], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        stratify=full_df['label'],
        random_state=42
    )

    print("\nFinal dataset sizes:")
    print(f"  Full    : {len(full_df)}")
    print(f"  Train   : {len(train_df)}")
    print(f"  Valid   : {len(val_df)}")

    print("\nTrain label distribution:")
    print(train_df['label'].value_counts(normalize=True))

    print("\nValidation label distribution:")
    print(val_df['label'].value_counts(normalize=True))

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    MODEL_PATH = "./local-distilbert"
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.map(lambda x: {"label": np.float32(x["label"])})
    val_dataset = val_dataset.map(lambda x: {"label": np.float32(x["label"])})

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Hyperparameter optimization
    results_csv = "optuna_trials_log_distilbert.csv"
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    )

    print("\n" + "="*60)
    print("STARTING HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    logging.info("Starting hyperparameter optimization with 50 trials")
    study.optimize(lambda trial: objective(trial, study, args.cluster_labeling), n_trials=50, callbacks=[save_trial_callback])

    # Get best trial results
    best_trial = study.best_trial
    best_hyperparams = best_trial.params
    print(f"\nBest trial number: {best_trial.number}")
    print(f"Best F1 score: {best_trial.value:.4f}")
    print("Best hyperparameters:", best_hyperparams)
    logging.info(f"Best trial: {best_trial.number}, F1 score: {best_trial.value:.4f}")
    logging.info(f"Best hyperparameters: {best_hyperparams}")
    
    # Removed CSV writing for faster execution
    
    # Train final model with best hyperparameters
    final_metrics = train_final_model(best_hyperparams, args.cluster_labeling)
    
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best F1 Score: {best_trial.value:.4f}")
    print(f"Final Model F1 Score: {final_metrics.get('eval_f1', 0):.4f}")
    print(f"Final Model Accuracy: {final_metrics.get('eval_accuracy', 0):.4f}")
    logging.info("Hyperparameter optimization complete")
    logging.info(f"Best F1 Score: {best_trial.value:.4f}")
    logging.info(f"Final Model F1 Score: {final_metrics.get('eval_f1', 0):.4f}")
    logging.info(f"Final Model Accuracy: {final_metrics.get('eval_accuracy', 0):.4f}") 