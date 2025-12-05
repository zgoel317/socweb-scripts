#imports
import os
# Allow use of GPUs 0–3 for faster multi-GPU training
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import gc
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import json

if torch.cuda.is_available():
    print(f"[GPU] Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("[GPU] CUDA not available – running on CPU")

#data paths
casual_data_path = "training_data/casual_dms_0.7_0.7_noemojis_zero_shot_clusters.csv"
serious_data_path = "training_data/serious_dms_0.7_0.7_noemojis_zero_shot_clusters.csv"
lonely_data_path = "training_data/lonely_dms_0.5_0.7_noemojis_zero_shot_clusters.csv"

data_paths = [casual_data_path, serious_data_path, lonely_data_path]

#column names for all csvs
dm_column_name = "DM"
zero_shot_label_column_name = "zero_shot_label"


#model path
MODEL_PATH = "/home/shared_models/base_models/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"

#output paths
model_save_path = "results/few_shot_zero_shot_labels"
best_model_path = os.path.join(model_save_path, "best")


#path with all the parquets to run inference on:
inference_dir = "demojized_parquets"

#output directory for scored parquets
output_dir = "few_shot_zero_shot_labels_scoring"

#functions
def load_data(data_paths):
    """Loads CSVs, prints per-file stats, concatenates, and returns a clean DataFrame."""
    print("Loading data from CSV files...")

    from tqdm import tqdm
    dfs = []
    for path in tqdm(data_paths, desc="Reading CSVs"):
        print(f"\nReading: {path}")
        try:
            df_i = pd.read_csv(path, usecols=[dm_column_name, zero_shot_label_column_name])
        except Exception as e:
            print(f"  -> Error reading {path}: {e}. Skipping this file.")
            continue

        # Drop NaN DMs and strip whitespace-only rows
        initial_rows = len(df_i)
        df_i = df_i.dropna(subset=[dm_column_name])
        df_i = df_i[df_i[dm_column_name].astype(str).str.strip() != ""]

        # Drop NaN labels and cast to int
        df_i = df_i.dropna(subset=[zero_shot_label_column_name])
        df_i[zero_shot_label_column_name] = df_i[zero_shot_label_column_name].astype(int)

        final_rows = len(df_i)
        removed = initial_rows - final_rows
        print(f"  Rows loaded: {initial_rows} | Rows after cleaning: {final_rows} | Removed: {removed}")
        print("  Label distribution:\n", df_i[zero_shot_label_column_name].value_counts())

        dfs.append(df_i)

    # Concatenate all cleaned DataFrames
    if not dfs:
        raise ValueError("No valid CSV files were loaded.")

    df = pd.concat(dfs, ignore_index=True)

    print("\n=== Combined Dataset Summary ===")
    print(f"Total rows: {len(df)}")
    print("Label distribution:\n", df[zero_shot_label_column_name].value_counts())

    return df


# ---------------------------------------------------------------------------
# Prompt construction (hard-coded few-shot examples)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE =  """You are a binary classifier for loneliness detection.                                     
Return the label only, either 1 (lonely) or 0 (not lonely).

Label 1  (lonely)  – The speaker describes feeling a lack of social connection
                    OR longing for reciprocation / closer bonds **right now**.
Label 0  (not)     – Any other emotion or topic.  Trauma, anxiety, depression,
                    self-reflection, anger, etc. **alone** are NOT enough.

Heuristics
• Keywords (“lonely”, “alone”, “ghost”, “ignored”) help but AREN’T required.
• A single mention of family/friends/etc. ≠ loneliness unless it shows disconnection.
• Talking about other people’s loneliness → label 0.

# FEW-SHOT EXAMPLES – follow this format exactly
Message: Been feeling mad lonely lately like I haven’t talked to anyone in forever… just wanna feel close to someone again fr. It’s been rough, ngl  
Answer: Feels lonely and wants closeness.
1

Message: Being homeschooled is so lonely fr like I can't talk to ppl without overthinking everything. Public school kids talk about stuff I don’t get, and homeschoolers can be kinda weird… idk I feel like I’m stuck.  
Answer: Feels lonely and struggles to talk to peers.
1

Message: Is it just me or is making friends so hard these days? Like idk if it’s me or ppl just don’t vibe anymore  
Answer: Trouble making friends; feels isolated.
1

Message: Ever feel like you’re closer to someone than they are to you? Like they’re cool w how things are but you want more? Idk it kinda hurts but you don’t wanna mess it up by pushing too much. Just me?  
Answer: Wants deeper connection than they have.
1

Message: Tbh idk why ppl act like college is all fun & vibes when it’s just stress and deadlines  like… where’s the ‘memorable’ part 
Answer: Complains about college stress, not loneliness.
0

Message: Yo, had the weirdest breakdown ever the other day… thought I was legit hallucinating, and it got so bad I ended up hurting myself  idk what’s going on fr
Answer: Emotional breakdown, but no loneliness.
0

Message: Yo, these hurricanes hitting red states right before elections… kinda sus ngl  
Answer: Comments on hurricanes; no loneliness.
0

Message: Omg desserts are life idk if I could pick but probs crème brûlée or choc-covered strawberries wbu??  
Answer: Loves desserts; no loneliness sentiment.
0

Message: Yo my cousin's wife is super sick and he’s been taking care of her for years, but now he’s out here holding hands w this other woman idk what to think… like I get he’s lonely but still, it feels so messy  
Answer: Comments on cousin's situation; not lonely.
0

Now, classify the following message:
DM: "{target_text}"
Classification:"""


def create_prompt(target_text):
    """Return the full prompt with the target_text inserted."""
    return PROMPT_TEMPLATE.format(target_text=target_text)

def compute_metrics(eval_pred):
    """Computes accuracy, precision, recall, and F1-score."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    print(f"\n[Metrics] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_model(train_df, val_df):
    """Trains a QLoRA model for few-shot classification."""
    print("Starting model training...")

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        prompts = [create_prompt(text) for text in examples[dm_column_name]]
        return tokenizer(prompts, truncation=True, padding="longest", max_length=3000)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    print("Tokenization complete.")
    
    tokenized_train_dataset = tokenized_train_dataset.rename_column(zero_shot_label_column_name, "labels")
    tokenized_val_dataset = tokenized_val_dataset.rename_column(zero_shot_label_column_name, "labels")

    print("Setting up BitsAndBytes config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        num_labels=2,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Preparing model for k-bit training and applying LoRA...")
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    # Freeze the backbone – train only LoRA adapters and classification head
    for name, param in model.named_parameters():
        if "lora" not in name.lower() and not name.startswith("score."):
            param.requires_grad = False
    trainable = [n for n,p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable)} / {sum(1 for _ in model.parameters())}")
    print("Model LoRA setup complete.")

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,
        weight_decay=0.001,
        warmup_steps=100,
        lr_scheduler_type="linear",
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,
        torch_compile=False,
        optim="adamw_torch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        max_grad_norm=0.3,
        label_smoothing_factor=0.0
    )

    print("Computing class weights...")
    label_counts = train_df[zero_shot_label_column_name].value_counts().sort_index()
    total_samples = len(train_df)
    raw_weights = torch.tensor([total_samples / (len(label_counts) * count) for count in label_counts], dtype=torch.float32)
    class_weights = torch.sqrt(raw_weights)
    print(f"Class weights (softened): {class_weights.tolist()}")

    print("Initializing Trainer...")
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        class_weights=class_weights
    )

    print("Starting training...")
    trainer.train()

    print("Training finished. Saving best model.")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    return model, tokenizer, trainer


def run_inference(model, tokenizer):
    """Runs inference on parquet files and saves scored CSVs."""
    print("Starting inference...")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(inference_dir):
        print(f"Inference directory '{inference_dir}' not found. Skipping inference.")
        return

    parquet_files = [f for f in os.listdir(inference_dir) if f.endswith('.parquet')]
    print(f"Found {len(parquet_files)} parquet files to process.")
    
    model.eval()
    
    for filename in tqdm(parquet_files, desc="Scoring parquet files"):
        input_path = os.path.join(inference_dir, filename)
        output_filename = f"{os.path.splitext(filename)[0]}.csv"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping {filename}, output already exists.")
            continue

        try:
            # Use progress bar per parquet file
            print(f"\nProcessing {filename}...")
            df = pd.read_parquet(input_path)
            if "Message" not in df.columns:
                print(f"Skipping {filename}, no 'Message' column.")
                continue

            texts = df["Message"].tolist()
            predictions = []

            for text in tqdm(texts, desc="Classifying", leave=False):
                prompt = create_prompt(text)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000).to(model.device)

                with torch.no_grad():
                    logits = model(**inputs).logits
                    pred = torch.argmax(logits, dim=-1).item()
                    predictions.append(pred)

            df['few_shot_prediction'] = predictions
            df.to_csv(output_path, index=False)
            print(f"Finished processing {filename}. Results saved to {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Inference complete.")


def plot_training_metrics(trainer, output_dir):
    """Generate and save training/validation plots and statistics."""
    print("\n=== Generating Training Plots and Statistics ===")

    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Extract training history from trainer logs
    log_history = trainer.state.log_history

    # Separate training and eval logs
    train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log]

    # Set seaborn style
    sns.set_style("whitegrid")

    # 1. Plot Training Loss
    if train_logs:
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = [log['step'] for log in train_logs if 'step' in log]
        losses = [log['loss'] for log in train_logs if 'loss' in log]

        ax.plot(steps, losses, marker='o', linestyle='-', linewidth=2, markersize=4)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training loss plot to {plots_dir}/training_loss.png")

    # 2. Plot Validation Metrics
    if eval_logs:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        epochs = [log.get('epoch', i+1) for i, log in enumerate(eval_logs)]

        metrics = {
            'Loss': ([log.get('eval_loss', 0) for log in eval_logs], 'lower'),
            'Accuracy': ([log.get('eval_accuracy', 0) for log in eval_logs], 'upper'),
            'Precision': ([log.get('eval_precision', 0) for log in eval_logs], 'upper'),
            'Recall': ([log.get('eval_recall', 0) for log in eval_logs], 'upper'),
        }

        for idx, (metric_name, (values, _)) in enumerate(metrics.items()):
            ax = axes[idx // 2, idx % 2]
            ax.plot(epochs, values, marker='o', linestyle='-', linewidth=2, markersize=8, color=f'C{idx}')
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'Validation {metric_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Annotate final value
            if values:
                final_val = values[-1]
                ax.annotate(f'{final_val:.4f}',
                           xy=(epochs[-1], final_val),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                           fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'validation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved validation metrics plot to {plots_dir}/validation_metrics.png")

    # 3. Generate Final Statistics Report
    stats = {
        'training': {},
        'validation': {}
    }

    # Final training stats
    if train_logs:
        final_train = train_logs[-1]
        stats['training'] = {
            'final_loss': final_train.get('loss', None),
            'final_step': final_train.get('step', None),
            'total_steps': len(train_logs)
        }

    # Final validation stats
    if eval_logs:
        final_eval = eval_logs[-1]
        stats['validation'] = {
            'final_loss': final_eval.get('eval_loss', None),
            'accuracy': final_eval.get('eval_accuracy', None),
            'precision': final_eval.get('eval_precision', None),
            'recall': final_eval.get('eval_recall', None),
            'f1': final_eval.get('eval_f1', None),
            'epoch': final_eval.get('epoch', None)
        }

    # Save statistics to JSON
    stats_file = os.path.join(plots_dir, 'training_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_file}")

    # Print final statistics
    print("\n" + "="*60)
    print("FINAL TRAINING STATISTICS")
    print("="*60)
    if stats['training']:
        print("\nTraining Set:")
        print(f"  Final Loss:     {stats['training'].get('final_loss', 'N/A'):.6f}" if stats['training'].get('final_loss') else "  Final Loss:     N/A")
        print(f"  Total Steps:    {stats['training'].get('total_steps', 'N/A')}")

    if stats['validation']:
        print("\nValidation Set:")
        print(f"  Final Loss:     {stats['validation'].get('final_loss', 'N/A'):.6f}" if stats['validation'].get('final_loss') else "  Final Loss:     N/A")
        print(f"  Accuracy:       {stats['validation'].get('accuracy', 'N/A'):.4f}" if stats['validation'].get('accuracy') else "  Accuracy:       N/A")
        print(f"  Precision:      {stats['validation'].get('precision', 'N/A'):.4f}" if stats['validation'].get('precision') else "  Precision:      N/A")
        print(f"  Recall:         {stats['validation'].get('recall', 'N/A'):.4f}" if stats['validation'].get('recall') else "  Recall:         N/A")
        print(f"  F1 Score:       {stats['validation'].get('f1', 'N/A'):.4f}" if stats['validation'].get('f1') else "  F1 Score:       N/A")
    print("="*60 + "\n")

    return stats


def generate_confusion_matrix(model, tokenizer, val_df, output_dir):
    """Generate confusion matrix for validation set."""
    print("\n=== Generating Confusion Matrix ===")

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    model.eval()
    predictions = []
    true_labels = val_df[zero_shot_label_column_name].tolist()

    print("Running predictions on validation set...")
    for text in tqdm(val_df[dm_column_name].tolist(), desc="Predicting"):
        prompt = create_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append(pred)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Lonely (0)', 'Lonely (1)'],
                yticklabels=['Not Lonely (0)', 'Lonely (1)'],
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to {cm_path}")

    # Print confusion matrix details
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")

    return cm


#run
def main():
    """Main function to run the few-shot training and inference pipeline."""
    df = load_data(data_paths)
    print("Splitting data into training and validation sets...")
    full_train_df, full_val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df[zero_shot_label_column_name])
    # Debug: label balance after split
    print("[DEBUG] Training label counts:\n", full_train_df[zero_shot_label_column_name].value_counts())
    print("[DEBUG] Validation label counts:\n", full_val_df[zero_shot_label_column_name].value_counts())
    print(f"Full train size: {len(full_train_df)}, full val size: {len(full_val_df)}")

    pos_samples = full_train_df[full_train_df[zero_shot_label_column_name]==1].sample(n=1000, random_state=42)
    neg_samples = full_train_df[full_train_df[zero_shot_label_column_name]==0].sample(n=1000, random_state=42)
    train_df = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

    val_subset_size = min(500, len(full_val_df))
    val_df = full_val_df.sample(n=val_subset_size, random_state=42)
    print(f"Few-shot train size: {len(train_df)}, validation subset size: {len(val_df)}")

    model, tokenizer, trainer = train_model(train_df, val_df)

    # Generate plots and statistics
    plot_training_metrics(trainer, model_save_path)

    # Generate confusion matrix
    generate_confusion_matrix(model, tokenizer, val_df, model_save_path)

    run_inference(model, tokenizer)

    print("Pipeline finished. Cleaning up GPU memory...")
    # Clean up GPU memory
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import sys
    from datetime import datetime

    log_file = "few_shot_zero_shot.log"

    class TeeOutput:
        def __init__(self, file, original_stdout):
            self.file = file
            self.original_stdout = original_stdout
        
        def write(self, text):
            self.file.write(text)
            self.file.flush()
            self.original_stdout.write(text)
            self.original_stdout.flush()
        
        def flush(self):
            self.file.flush()
            self.original_stdout.flush()

    with open(log_file, 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(f, original_stdout)
        
        print(f"=== Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        print(f"All output is being logged to {os.path.abspath(log_file)}")
        
        try:
            main()
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"=== Script finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            sys.stdout = original_stdout
    
    print(f"Script finished. Log saved to {log_file}")