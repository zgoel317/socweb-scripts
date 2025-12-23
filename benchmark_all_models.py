#imports
import os
# Allow use of GPUs 0–3 for faster multi-GPU training
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from datasets import Dataset
from peft import PeftModel
import gc
from tqdm import tqdm

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

#model paths - updated to match downloaded models
LLAMA_PATH = "/home/shared_models/base_models/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
GEMMA_PATH = "/home/shared_models/base_models/google--gemma-2-9b-it"
MISTRAL_PATH = "/home/shared_models/base_models/mistralai--Mistral-7B-Instruct-v0.3"
QWEN_PATH = "/home/shared_models/base_models/Qwen--Qwen2.5-7B-Instruct"

# Finetuned model path
FINETUNED_MODEL_PATH = "results/few_shot_zero_shot_labels/best"

# Output paths
benchmark_results_path = "benchmark_results"
os.makedirs(benchmark_results_path, exist_ok=True)

# Prompt template - same as in the original script
PROMPT_TEMPLATE = """You are a binary classifier for loneliness detection.
Return the label only, either 1 (lonely) or 0 (not lonely).

Label 1  (lonely)  – The speaker describes feeling a lack of social connection
                    OR longing for reciprocation / closer bonds **right now**.
Label 0  (not)     – Any other emotion or topic.  Trauma, anxiety, depression,
                    self-reflection, anger, etc. **alone** are NOT enough.

Heuristics
• Keywords ("lonely", "alone", "ghost", "ignored") help but AREN'T required.
• A single mention of family/friends/etc. ≠ loneliness unless it shows disconnection.
• Talking about other people's loneliness → label 0.

# FEW-SHOT EXAMPLES – follow this format exactly
Message: Been feeling mad lonely lately like I haven't talked to anyone in forever… just wanna feel close to someone again fr. It's been rough, ngl
Answer: Feels lonely and wants closeness.
1

Message: Being homeschooled is so lonely fr like I can't talk to ppl without overthinking everything. Public school kids talk about stuff I don't get, and homeschoolers can be kinda weird… idk I feel like I'm stuck.
Answer: Feels lonely and struggles to talk to peers.
1

Message: Is it just me or is making friends so hard these days? Like idk if it's me or ppl just don't vibe anymore
Answer: Trouble making friends; feels isolated.
1

Message: Ever feel like you're closer to someone than they are to you? Like they're cool w how things are but you want more? Idk it kinda hurts but you don't wanna mess it up by pushing too much. Just me?
Answer: Wants deeper connection than they have.
1

Message: Tbh idk why ppl act like college is all fun & vibes when it's just stress and deadlines  like… where's the 'memorable' part
Answer: Complains about college stress, not loneliness.
0

Message: Yo, had the weirdest breakdown ever the other day… thought I was legit hallucinating, and it got so bad I ended up hurting myself  idk what's going on fr
Answer: Emotional breakdown, but no loneliness.
0

Message: Yo, these hurricanes hitting red states right before elections… kinda sus ngl
Answer: Comments on hurricanes; no loneliness.
0

Message: Omg desserts are life idk if I could pick but probs crème brûlée or choc-covered strawberries wbu??
Answer: Loves desserts; no loneliness sentiment.
0

Message: Yo my cousin's wife is super sick and he's been taking care of her for years, but now he's out here holding hands w this other woman idk what to think… like I get he's lonely but still, it feels so messy
Answer: Comments on cousin's situation; not lonely.
0

Now, classify the following message:
DM: "{target_text}"
Classification:"""


def create_prompt(target_text):
    """Return the full prompt with the target_text inserted."""
    return PROMPT_TEMPLATE.format(target_text=target_text)


def load_data(data_paths):
    """Loads CSVs, prints per-file stats, concatenates, and returns a clean DataFrame."""
    print("Loading data from CSV files...")

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


def compute_metrics_from_predictions(labels, predictions, model_name):
    """Computes accuracy, precision, recall, and F1-score."""
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    print(f"\n[{model_name}] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_base_model(model_path, model_name):
    """Load a base model for inference."""
    print(f"\n{'='*80}")
    print(f"Loading {model_name} from {model_path}")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        num_labels=2,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    print(f"{model_name} loaded successfully!")
    return model, tokenizer


def load_finetuned_model(base_model_path, adapter_path, model_name):
    """Load the finetuned model with PEFT adapters."""
    print(f"\n{'='*80}")
    print(f"Loading {model_name}")
    print(f"Base model: {base_model_path}")
    print(f"Adapters: {adapter_path}")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        num_labels=2,
        device_map="auto"
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print(f"{model_name} loaded successfully!")
    return model, tokenizer


def run_inference_on_eval_set(model, tokenizer, eval_df, model_name, use_chat_template=False):
    """Run inference on evaluation set and return predictions.

    Args:
        use_chat_template: If True, use the tokenizer's chat template.
                          Only use for finetuned models that were trained with chat templates.
                          Base models doing sequence classification should use raw prompts.
    """
    print(f"\n{'='*80}")
    print(f"Running inference with {model_name}")
    print(f"Using chat template: {use_chat_template}")
    print(f"{'='*80}")

    predictions = []
    texts = eval_df[dm_column_name].tolist()

    for text in tqdm(texts, desc=f"{model_name} - Classifying"):
        if use_chat_template and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            # Use the model's native chat template (for finetuned models)
            messages = [
                {"role": "user", "content": create_prompt(text)}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Use raw prompt (for base models doing sequence classification)
            prompt = create_prompt(text)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append(pred)

    return predictions


def cleanup_model(model):
    """Clean up GPU memory after using a model."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory cleaned up.")


def main():
    """Main function to benchmark all models."""
    print("="*80)
    print("BENCHMARK: Comparing Base Models vs Finetuned Model")
    print("="*80)

    # Load data and create same eval set used in training
    df = load_data(data_paths)
    print("\nSplitting data into training and validation sets...")
    full_train_df, full_val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df[zero_shot_label_column_name]
    )

    # Use the same validation subset as in training
    val_subset_size = min(500, len(full_val_df))
    eval_df = full_val_df.sample(n=val_subset_size, random_state=42)

    print(f"\nEvaluation set size: {len(eval_df)}")
    print("Evaluation set label distribution:\n", eval_df[zero_shot_label_column_name].value_counts())

    # Get ground truth labels
    true_labels = eval_df[zero_shot_label_column_name].tolist()

    # Dictionary to store all results
    all_results = []

    # Models to benchmark
    models_to_test = [
        (LLAMA_PATH, "Llama-3-8B-Instruct (Base)", "base"),
        (MISTRAL_PATH, "Mistral-7B-Instruct-v0.3 (Base)", "base"),
        (QWEN_PATH, "Qwen2.5-7B-Instruct (Base)", "base"),
        (GEMMA_PATH, "Gemma-2-9B-IT (Base)", "base"),
        (LLAMA_PATH, "Llama-3-8B-Instruct (Finetuned)", "finetuned"),
    ]

    for model_path, model_name, model_type in models_to_test:
        try:
            # Load model
            if model_type == "base":
                if not os.path.exists(model_path):
                    print(f"\nSkipping {model_name} - path not found: {model_path}")
                    continue
                model, tokenizer = load_base_model(model_path, model_name)
            else:  # finetuned
                model, tokenizer = load_finetuned_model(model_path, FINETUNED_MODEL_PATH, model_name)

            # Run inference
            # Only use chat template for finetuned model (which was trained with it)
            use_chat_template = (model_type == "finetuned")
            predictions = run_inference_on_eval_set(model, tokenizer, eval_df, model_name, use_chat_template)

            # Compute metrics
            metrics = compute_metrics_from_predictions(true_labels, predictions, model_name)
            all_results.append(metrics)

            # Save individual predictions
            results_df = eval_df.copy()
            results_df['prediction'] = predictions
            results_df['correct'] = results_df[zero_shot_label_column_name] == results_df['prediction']
            output_path = os.path.join(benchmark_results_path, f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_predictions.csv")
            results_df.to_csv(output_path, index=False)
            print(f"Predictions saved to: {output_path}")

            # Clean up
            cleanup_model(model)
            del tokenizer

        except Exception as e:
            print(f"\nError processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create comparison table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('f1', ascending=False)

    print("\n" + results_df.to_string(index=False))

    # Save results
    results_path = os.path.join(benchmark_results_path, "benchmark_summary.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nBenchmark summary saved to: {results_path}")

    # Print best model
    best_model = results_df.iloc[0]
    print("\n" + "="*80)
    print(f"BEST MODEL: {best_model['model']}")
    print(f"F1 Score: {best_model['f1']:.4f}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Precision: {best_model['precision']:.4f}")
    print(f"Recall: {best_model['recall']:.4f}")
    print("="*80)


if __name__ == "__main__":
    import sys
    from datetime import datetime

    log_file = "benchmark_all_models.log"

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

        print(f"=== Benchmark started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        print(f"All output is being logged to {os.path.abspath(log_file)}")

        try:
            main()
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"=== Benchmark finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            sys.stdout = original_stdout

    print(f"Benchmark finished. Log saved to {log_file}")
