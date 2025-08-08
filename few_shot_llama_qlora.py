#!/usr/bin/env python3
"""
Enhanced Few-shot learning script for LLaMA QLoRA
Implements advanced few-shot learning techniques with semantic example selection,
chain-of-thought prompting, and optimized hyperparameters for better accuracy.
"""

import os
import sys
import json
import pickle
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import random

# Set CUDA_VISIBLE_DEVICES before any other imports
import torch

# Set PyTorch memory management to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

N = torch.cuda.device_count()
if N == 0:
    print("No CUDA devices available!")
    sys.exit(1)

import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import bitsandbytes as bnb
from filelock import FileLock

# Try to import sentence transformers for semantic similarity (optional)
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic similarity features will be disabled.")

# Configure logging - will be handled by our TeeOutput in main()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global debug flag
DEBUG = True

def debug_print(message, important=False):
    """Debug print function - only print important messages by default"""
    if DEBUG and important:
        print(f"[DEBUG] {message}", flush=True)

def debug_gpu_memory(prefix="", important=False):
    """Debug function to print GPU memory usage - only when important"""
    if DEBUG and important and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        debug_print(f"{prefix}GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free, {total:.2f}GB total", important)

def get_semantic_embeddings(texts, model_name='all-mpnet-base-v2'):
    """Get semantic embeddings for texts using sentence transformers"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        debug_print("Sentence transformers not available, using fallback", important=True)
        return None
    
    try:
        embedder = SentenceTransformer(model_name)
        embeddings = embedder.encode(texts, convert_to_tensor=True)
        return embeddings
    except Exception as e:
        debug_print(f"Error getting semantic embeddings: {e}", important=True)
        return None

def select_semantic_examples(target_text, train_df, k=6, balance_classes=True):
    """Select examples using semantic similarity to target text"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        debug_print("Semantic selection not available, falling back to random", important=True)
        return select_random_examples(train_df, k, balance_classes)
    
    try:
        # Get embeddings for all training texts
        all_texts = train_df['DM'].tolist()
        embeddings = get_semantic_embeddings(all_texts)
        
        if embeddings is None:
            return select_random_examples(train_df, k, balance_classes)
        
        # Get embedding for target text
        target_embedding = get_semantic_embeddings([target_text])
        if target_embedding is None:
            return select_random_examples(train_df, k, balance_classes)
        
        # Find most similar examples
        similarities = util.pytorch_cos_sim(target_embedding, embeddings)[0]
        top_indices = torch.topk(similarities, k=min(k*2, len(similarities))).indices.cpu().numpy()
        
        # Balance classes if requested
        if balance_classes:
            selected_examples = []
            lonely_count = 0
            not_lonely_count = 0
            max_per_class = k // 2
            
            for idx in top_indices:
                example = train_df.iloc[idx]
                if example['label'] == 1 and lonely_count < max_per_class:
                    selected_examples.append(example)
                    lonely_count += 1
                elif example['label'] == 0 and not_lonely_count < max_per_class:
                    selected_examples.append(example)
                    not_lonely_count += 1
                
                if len(selected_examples) >= k:
                    break
            
            # Fill remaining slots if needed
            if len(selected_examples) < k:
                remaining = k - len(selected_examples)
                remaining_examples = train_df.iloc[top_indices[len(selected_examples):len(selected_examples)+remaining]]
                selected_examples.extend(remaining_examples.to_dict('records'))
            
            return pd.DataFrame(selected_examples)
        else:
            return train_df.iloc[top_indices[:k]]
    
    except Exception as e:
        debug_print(f"Error in semantic selection: {e}", important=True)
        return select_random_examples(train_df, k, balance_classes)

def controlled_random_subsampling(embeddings, texts, k, oversample_factor=5, random_state=42):
    """
    Implement controlled random subsampling for more diverse example selection.
    
    Args:
        embeddings: numpy array of embeddings
        texts: list of corresponding texts
        k: number of examples to select
        oversample_factor: factor to oversample (e.g., 5 means select top 5*k examples)
        random_state: random seed for reproducibility
    
    Returns:
        list: selected indices
    """
    if len(embeddings) <= k:
        return list(range(len(embeddings)))
    
    # Calculate distances to centroid
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    
    # Get top M examples (oversample_factor * k)
    top_m = np.argsort(distances)[:min(oversample_factor * k, len(distances))]
    
    # Randomly draw k examples from the top M
    np.random.seed(random_state)
    chosen_idx = np.random.choice(top_m, size=min(k, len(top_m)), replace=False)
    
    # Return the selected indices
    selected_indices = chosen_idx.tolist()
    
    debug_print(f"Controlled random subsampling: selected {len(selected_indices)} from top {len(top_m)} examples (oversample_factor={oversample_factor})", important=True)
    
    return selected_indices

def select_random_examples(train_df, k=6, balance_classes=True):
    """Select random examples with optional class balancing"""
    if balance_classes:
        lonely_data = train_df[train_df['label'] == 1]
        not_lonely_data = train_df[train_df['label'] == 0]
        
        k_per_class = k // 2
        selected_lonely = lonely_data.sample(n=min(k_per_class, len(lonely_data)), random_state=42)
        selected_not_lonely = not_lonely_data.sample(n=min(k_per_class, len(not_lonely_data)), random_state=42)
        
        selected = pd.concat([selected_lonely, selected_not_lonely], ignore_index=True)
        
        # Fill remaining slots if needed
        if len(selected) < k:
            remaining = k - len(selected)
            remaining_data = train_df[~train_df.index.isin(selected.index)]
            if len(remaining_data) > 0:
                additional = remaining_data.sample(n=min(remaining, len(remaining_data)), random_state=42)
                selected = pd.concat([selected, additional], ignore_index=True)
        
        return selected.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        return train_df.sample(n=min(k, len(train_df)), random_state=42).reset_index(drop=True)

def cluster_examples(train_df, n_clusters=4, use_semantic=True):
    """Cluster examples to find diverse prototypes"""
    try:
        if use_semantic and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use semantic embeddings for clustering
            texts = train_df['DM'].tolist()
            embeddings = get_semantic_embeddings(texts)
            
            if embeddings is not None:
                # Reduce dimensionality for clustering
                pca = PCA(n_components=min(50, len(embeddings[0])))
                reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())
                
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(reduced_embeddings)
                
                # Add cluster labels to dataframe
                clustered_df = train_df.copy()
                clustered_df['cluster'] = cluster_labels
                
                return clustered_df
        else:
            # Fallback to simple clustering based on text length and label
            clustered_df = train_df.copy()
            clustered_df['text_length'] = clustered_df['DM'].str.len()
            
            # Create simple clusters based on length and label
            clustered_df['cluster'] = (
                (clustered_df['text_length'] > clustered_df['text_length'].median()).astype(int) * 2 +
                clustered_df['label']
            )
            
            return clustered_df
    
    except Exception as e:
        debug_print(f"Error in clustering: {e}", important=True)
        # Fallback: add random cluster labels
        clustered_df = train_df.copy()
        clustered_df['cluster'] = np.random.randint(0, n_clusters, size=len(clustered_df))
        return clustered_df

def augment_example(text, method='paraphrase'):
    """Augment a single example using various methods"""
    if method == 'paraphrase':
        # Simple paraphrasing by replacing common words
        paraphrases = {
            'lonely': ['isolated', 'alone', 'solitary'],
            'sad': ['unhappy', 'depressed', 'miserable'],
            'miss': ['long for', 'yearn for', 'want'],
            'friend': ['companion', 'buddy', 'pal'],
            'talk': ['chat', 'converse', 'speak'],
            'weekend': ['weekend', 'days off', 'free time'],
            'home': ['house', 'place', 'apartment']
        }
        
        augmented = text
        for original, alternatives in paraphrases.items():
            if original in augmented.lower():
                replacement = random.choice(alternatives)
                augmented = augmented.replace(original, replacement)
                break  # Only replace one word per augmentation
        
        return augmented
    else:
        return text

def augment_examples(examples_df, augmentation_factor=2):
    """Augment examples by creating variations"""
    augmented_examples = []
    
    for _, row in examples_df.iterrows():
        # Keep original
        augmented_examples.append(row.to_dict())
        
        # Add augmented versions
        for i in range(augmentation_factor - 1):
            augmented_text = augment_example(row['DM'])
            augmented_row = row.copy()
            augmented_row['DM'] = augmented_text
            augmented_examples.append(augmented_row)
    
    return pd.DataFrame(augmented_examples)

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
                "debug": True,
                "num_trials": 10
            }
        }

def parse_args():
    debug_print("parse_args() called")
    parser = argparse.ArgumentParser(description="Simplified Few-shot LLaMA QLoRA Learning")
    parser.add_argument("--cluster_labeling", action="store_true", help="Run cluster labeling optimization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config_file", type=str, default="gpu_config.yaml", help="Configuration file path")
    parser.add_argument("--examples_per_category", type=int, default=3, help="Number of examples per category (for binary: per label, for cluster: per confusion matrix quadrant)")
    parser.add_argument("--exception_ratio", type=int, default=2, help="Ratio for exception examples (exceptions = floor(examples_per_category/exception_ratio)). Default: 2")
    parser.add_argument("--no_exceptions", action="store_true", help="Set exception examples to 0 (only use pattern examples)")
    args = parser.parse_args()
    
    # Load configuration from file
    config = load_config()
    
    # Override with command line arguments if provided
    if args.cluster_labeling:
        config["training"]["cluster_labeling"] = True
    
    debug_print(f"parse_args() returned: {args}")
    debug_print(f"Loaded config: {config}")
    return args, config

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
            print("Using K-means clustering for cluster-based labeling:")
            print(f"  Lonely  : {len(lonely_df)}")
            print(f"  Serious : {len(not_lonely_df1)}")
            print(f"  Casual  : {len(not_lonely_df2)}")
            
            # Get embeddings for all DMs (like in the notebook - no cleaning first)
            debug_print("Getting embeddings for K-means clustering...", important=True)
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    # Use sentence transformers for embeddings (exactly like notebook)
                    embedder = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Get embeddings for each dataset separately (exactly like notebook)
                    debug_print("Getting embeddings for lonely data...", important=True)
                    lonely_texts = lonely_df['DM'].tolist()
                    lonely_embeddings = embedder.encode(lonely_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
                    
                    debug_print("Getting embeddings for serious data...", important=True)
                    serious_texts = not_lonely_df1['DM'].tolist()
                    serious_embeddings = embedder.encode(serious_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
                    
                    debug_print("Getting embeddings for casual data...", important=True)
                    casual_texts = not_lonely_df2['DM'].tolist()
                    casual_embeddings = embedder.encode(casual_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
                    
                    debug_print(f"Embeddings shapes: lonely={lonely_embeddings.shape}, serious={serious_embeddings.shape}, casual={casual_embeddings.shape}", important=True)
                    
                    # Get centroid initializations using original CSV labels (exactly like notebook)
                    debug_print("Computing centroid initializations using original CSV labels...", important=True)
                    
                    centroid_lonely = np.mean(lonely_embeddings, axis=0)
                    centroid_nonlonely = np.mean(np.vstack([serious_embeddings, casual_embeddings]), axis=0)
                    custom_init = np.vstack([centroid_lonely, centroid_nonlonely])
                    
                    debug_print(f"Custom initialization centroids shape: {custom_init.shape}", important=True)
                    debug_print(f"Centroid 0 (lonely) shape: {centroid_lonely.shape}", important=True)
                    debug_print(f"Centroid 1 (nonlonely) shape: {centroid_nonlonely.shape}", important=True)
                    
                    # Combine all embeddings for clustering (exactly like notebook)
                    debug_print("Combining all embeddings for clustering...", important=True)
                    emb_all = np.vstack([lonely_embeddings, serious_embeddings, casual_embeddings])
                    labels_all = ['lonely'] * len(lonely_embeddings) + ['nonlonely'] * (len(serious_embeddings) + len(casual_embeddings))
                    
                    debug_print(f"Combined embeddings shape: {emb_all.shape}", important=True)
                    debug_print(f"Labels: {len(labels_all)} total, {labels_all.count('lonely')} lonely, {labels_all.count('nonlonely')} nonlonely", important=True)
                    
                    # Run K-means clustering (exactly like notebook)
                    debug_print("Running K-means clustering with custom initialization...", important=True)
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=2, init=custom_init, n_init=1, random_state=42)
                    cluster_ids = kmeans.fit_predict(emb_all)
                    
                    debug_print(f"K-means clustering completed. Cluster distribution: {np.bincount(cluster_ids)}", important=True)
                    
                    # Create DataFrame for analysis (exactly like notebook)
                    df_analysis = pd.DataFrame({
                        'cluster': cluster_ids,
                        'label': labels_all
                    })
                    
                    # Cluster/label agreement (exactly like notebook)
                    confusion_matrix = df_analysis.groupby(['cluster', 'label']).size().unstack(fill_value=0)
                    print("\nCluster/label agreement:")
                    print(confusion_matrix)
                    
                    # Track lengths and split cluster_ids accordingly (exactly like notebook)
                    len_lonely = len(lonely_embeddings)
                    len_serious = len(serious_embeddings)
                    len_casual = len(casual_embeddings)
                    
                    cluster_ids_lonely = cluster_ids[:len_lonely]
                    cluster_ids_serious = cluster_ids[len_lonely:len_lonely + len_serious]
                    cluster_ids_casual = cluster_ids[len_lonely + len_serious:]
                    
                    # Add cluster column to each original dataframe (exactly like notebook)
                    lonely_df_with_clusters = lonely_df.copy()
                    serious_df_with_clusters = not_lonely_df1.copy()
                    casual_df_with_clusters = not_lonely_df2.copy()
                    
                    lonely_df_with_clusters['cluster'] = cluster_ids_lonely
                    serious_df_with_clusters['cluster'] = cluster_ids_serious
                    casual_df_with_clusters['cluster'] = cluster_ids_casual
                    
                    # Save to CSV files (like notebook)
                    lonely_df_with_clusters.to_csv("lonely_with_clusters.csv", index=False)
                    serious_df_with_clusters.to_csv("serious_with_clusters.csv", index=False)
                    casual_df_with_clusters.to_csv("casual_with_clusters.csv", index=False)
                    
                    debug_print("Cluster assignments saved to CSV files", important=True)
                    
                    # Print cluster distributions (like notebook)
                    print("\nLonely:")
                    print(lonely_df_with_clusters['cluster'].value_counts().sort_index())
                    print("\nSerious:")
                    print(serious_df_with_clusters['cluster'].value_counts().sort_index())
                    print("\nCasual:")
                    print(casual_df_with_clusters['cluster'].value_counts().sort_index())
                    
                    # Determine which cluster is the "lonely" cluster based on the confusion matrix
                    # From the notebook output, we can see that cluster 0 has more lonely examples
                    lonely_in_cluster_0 = confusion_matrix.loc[0, 'lonely'] if 'lonely' in confusion_matrix.columns else 0
                    lonely_in_cluster_1 = confusion_matrix.loc[1, 'lonely'] if 'lonely' in confusion_matrix.columns else 0
                    
                    if lonely_in_cluster_0 > lonely_in_cluster_1:
                        lonely_cluster = 0
                        not_lonely_cluster = 1
                        debug_print("Cluster 0 is the lonely cluster (label=1)", important=True)
                    else:
                        lonely_cluster = 1
                        not_lonely_cluster = 0
                        debug_print("Cluster 1 is the lonely cluster (label=1)", important=True)
                    
                    # Create the final dataset with cluster-based labels
                    debug_print("Creating final dataset with cluster-based labels...", important=True)
                    
                    # Create the combined dataset in the same order as the embeddings
                    # IMPORTANT: Preserve the original labels (which correspond to keyword presence)
                    # and add cluster assignments separately
                    lonely_df_with_labels = lonely_df_with_clusters.copy()
                    lonely_df_with_labels['original_label'] = 1  # All lonely examples have keywords
                    lonely_df_with_labels['label'] = (cluster_ids_lonely == lonely_cluster).astype(np.int64)
                    
                    serious_df_with_labels = serious_df_with_clusters.copy()
                    serious_df_with_labels['original_label'] = 0  # All serious examples don't have keywords
                    serious_df_with_labels['label'] = (cluster_ids_serious == lonely_cluster).astype(np.int64)
                    
                    casual_df_with_labels = casual_df_with_clusters.copy()
                    casual_df_with_labels['original_label'] = 0  # All casual examples don't have keywords
                    casual_df_with_labels['label'] = (cluster_ids_casual == lonely_cluster).astype(np.int64)
                    
                    # Combine all datasets
                    all_dms = pd.concat([lonely_df_with_labels, serious_df_with_labels, casual_df_with_labels], ignore_index=True)
                    
                    debug_print(f"Final label distribution: {all_dms['label'].value_counts().to_dict()}", important=True)
                    debug_print(f"Original label distribution (keyword presence): {all_dms['original_label'].value_counts().to_dict()}", important=True)
                    
                    debug_print("Cluster-based labeling completed successfully", important=True)
                    
                except Exception as e:
                    debug_print(f"Error in K-means clustering: {e}", important=True)
                    print(f"Falling back to simple binary labeling due to clustering error: {e}")
                    # Fallback to simple binary labeling
                    lonely_df['label'] = 1
                    not_lonely_df1['label'] = 0
                    not_lonely_df2['label'] = 0
                    all_dms = pd.concat([lonely_df, not_lonely_df1, not_lonely_df2], ignore_index=True)
            else:
                debug_print("Sentence transformers not available, falling back to simple binary labeling", important=True)
                # Fallback to simple binary labeling
                lonely_df['label'] = 1
                not_lonely_df1['label'] = 0
                not_lonely_df2['label'] = 0
                all_dms = pd.concat([lonely_df, not_lonely_df1, not_lonely_df2], ignore_index=True)
        else:
            print("Using simple binary labeling with full datasets:")
            print("  Lonely samples -> label = 1")
            print("  Non-lonely samples -> label = 0")
            lonely_df['label'] = 1
            not_lonely_df1['label'] = 0
            not_lonely_df2['label'] = 0
            all_dms = pd.concat([lonely_df, not_lonely_df1, not_lonely_df2], ignore_index=True)
            print(f"\nDataset sizes:")
            print(f"  Lonely  : {len(lonely_df)}")
            print(f"  Serious : {len(not_lonely_df1)}")
            print(f"  Casual  : {len(not_lonely_df2)}")

        debug_print(f"Full dataset size: {len(all_dms)}", important=True)
        
        # Clean the data if not already done for clustering
        if not cluster_labeling:
            print("Cleaning dataset...")
            initial_size = len(all_dms)
            
            # Remove rows where DM is NaN, empty, or not a string
            all_dms = all_dms.dropna(subset=['DM'])
            all_dms = all_dms[all_dms['DM'].astype(str).str.strip() != '']
            all_dms = all_dms[all_dms['DM'].astype(str).str.len() > 0]
            
            # Remove rows where DM is just whitespace or very short
            all_dms = all_dms[all_dms['DM'].astype(str).str.strip().str.len() > 5]
            
            final_size = len(all_dms)
            print(f"Data cleaning: {initial_size} -> {final_size} examples (removed {initial_size - final_size} invalid rows)")
        
        all_dms = all_dms.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df, val_df = train_test_split(
            all_dms,
            test_size=0.05,  # Use 5% for validation
            stratify=all_dms['label'],
            random_state=42
        )
        debug_print(f"Train/val split: {len(train_df)}/{len(val_df)}", important=True)
        
        print("\nFinal dataset sizes:")
        print(f"  Full    : {len(all_dms)}")
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

def create_few_shot_prompt(examples_df, target_text, cluster_labeling=False, use_chain_of_thought=True, use_semantic_selection=False, train_df=None):
    """Create an enhanced few-shot prompt for LLaMA with examples and target text"""
    
    # Select examples using semantic similarity if requested
    if use_semantic_selection and train_df is not None:
        selected_examples = select_semantic_examples(target_text, train_df, k=len(examples_df), balance_classes=True)
    else:
        selected_examples = examples_df
    
    if cluster_labeling:
        # Create sophisticated prompt for cluster labeling
        keywords = ["loneliness", "lonely", "social isolation", "solitude", "miserable"]
        
        # Separate examples by category using the same logic as create_cluster_few_shot_dataset
        lonely_with_keywords = []      # Cluster 0, original_label 1 (PATTERN)
        not_lonely_without_keywords = []  # Cluster 1, original_label 0 (PATTERN)
        lonely_without_keywords = []   # Cluster 0, original_label 0 (EXCEPTION)
        not_lonely_with_keywords = []  # Cluster 1, original_label 1 (EXCEPTION)
        
        for idx, row in selected_examples.iterrows():
            # Get cluster assignment and original label (keyword presence)
            cluster = row.get('cluster_id', row.get('cluster', 0))  # Try both possible column names
            original_label = row.get('original_label', row['label'])  # Use original_label if available, otherwise fall back to label
            
            if cluster == 0 and original_label == 1:
                lonely_with_keywords.append(row)  # PATTERN: lonely cluster, lonely label
            elif cluster == 1 and original_label == 0:
                not_lonely_without_keywords.append(row)  # PATTERN: not lonely cluster, not lonely label
            elif cluster == 0 and original_label == 0:
                lonely_without_keywords.append(row)  # EXCEPTION: lonely cluster, not lonely label
            elif cluster == 1 and original_label == 1:
                not_lonely_with_keywords.append(row)  # EXCEPTION: not lonely cluster, lonely label
        
        # Create the enhanced system prompt
        system_prompt = f"""You are a binary classifier for loneliness detection. Your task is to determine if a person is feeling lonely based on their text messages.

IMPORTANT: Reply with ONLY "1" or "0", nothing else.

Classification rules:
- 1: The person expresses feelings of loneliness, isolation, sadness, or a desire for connection
- 0: The person expresses positive emotions, social connections, or neutral content without loneliness indicators

Key loneliness indicators: {', '.join(keywords)}

IMPORTANT PATTERN: 
- MOST messages with these keywords are lonely (about 79% of the time)
- MOST messages without these keywords are not lonely (about 80% of the time)
- BUT there are exceptions! Some people express loneliness without using these words
- AND some people use these words in non-lonely contexts

Always consider the full context and meaning, not just keyword presence."""

        # Build the enhanced few-shot examples with chain-of-thought
        examples_text = ""
        
        if use_chain_of_thought:
            # Add a chain-of-thought example
            examples_text += "\n\n=== EXAMPLE WITH REASONING ==="
            if lonely_with_keywords:
                example = lonely_with_keywords[0]
                # Analyze keywords, tone, and sentiment
                keywords_found = [kw for kw in keywords if kw in example['DM'].lower()]
                tone_analysis = "negative" if any(word in example['DM'].lower() for word in ['😔', '😞', '😭', '💔', 'sad', 'miserable', 'stuck', 'done']) else "neutral"
                examples_text += f"\nMessage: \"{example['DM']}\"\nReasoning: Keywords present: {', '.join(keywords_found)}. Tone: {tone_analysis}. This follows the common pattern (79% of keyword messages are lonely) AND the context clearly indicates loneliness through direct expression and negative emotions.\nClassification: 1"
            elif lonely_without_keywords:
                example = lonely_without_keywords[0]
                # Analyze context and implicit loneliness indicators
                context_indicators = []
                if any(word in example['DM'].lower() for word in ['no friends', 'no family', 'alone', 'isolated', 'missing']):
                    context_indicators.append("social isolation")
                if any(word in example['DM'].lower() for word in ['want', 'need', 'wish', 'hope']):
                    context_indicators.append("desire for connection")
                if any(word in example['DM'].lower() for word in ['😔', '😞', '😭', '💔', 'sad', 'depressed']):
                    context_indicators.append("negative emotions")
                
                examples_text += f"\nMessage: \"{example['DM']}\"\nReasoning: No explicit loneliness keywords found. Context indicators: {', '.join(context_indicators) if context_indicators else 'none'}. This is an EXCEPTION to the pattern (20% of no-keyword messages are lonely). Despite no keywords, the context and emotional tone clearly indicate social isolation and desire for connection.\nClassification: 1"
        
            # Group examples by category
            if lonely_with_keywords:
                examples_text += "\n\n=== PATTERN EXAMPLES (Keywords = Lonely) ==="
                debug_print(f"Selected {len(lonely_with_keywords)} lonely_with_keywords examples", important=True)
                for i, row in enumerate(lonely_with_keywords):
                    keywords_found = [kw for kw in keywords if kw in str(row['DM']).lower()]
                    examples_text += f"\nMessage: \"{row['DM']}\"\nReasoning: Keywords present: {', '.join(keywords_found)}. This follows the common pattern (79% of keyword messages are lonely).\nClassification: 1"
                    debug_print(f"Lonely with keywords example {i+1}: {row['DM'][:100]}...", important=True)
        
            if not_lonely_without_keywords:
                examples_text += "\n\n=== PATTERN EXAMPLES (No Keywords = Not Lonely) ==="
                debug_print(f"Selected {len(not_lonely_without_keywords)} not_lonely_without_keywords examples", important=True)
                for i, row in enumerate(not_lonely_without_keywords):
                    examples_text += f"\nMessage: \"{row['DM']}\"\nReasoning: No loneliness keywords present. This follows the common pattern (80% of no-keyword messages are not lonely).\nClassification: 0"
                    debug_print(f"Not lonely without keywords example {i+1}: {row['DM'][:100]}...", important=True)
        
            if lonely_without_keywords:
                examples_text += "\n\n=== EXCEPTION EXAMPLES (No Keywords but Lonely) ==="
                debug_print(f"Selected {len(lonely_without_keywords)} lonely_without_keywords examples", important=True)
                for i, row in enumerate(lonely_without_keywords):
                    # Analyze why this is lonely despite no keywords
                    text_lower = str(row['DM']).lower()
                    context_indicators = []
                    if any(word in text_lower for word in ['no friends', 'no family', 'alone', 'isolated', 'missing']):
                        context_indicators.append("social isolation")
                    if any(word in text_lower for word in ['want', 'need', 'wish', 'hope']):
                        context_indicators.append("desire for connection")
                    if any(word in text_lower for word in ['😔', '😞', '😭', '💔', 'sad', 'depressed']):
                        context_indicators.append("negative emotions")
                    
                    examples_text += f"\nMessage: \"{row['DM']}\"\nReasoning: No loneliness keywords present. This is an EXCEPTION to the pattern (20% of no-keyword messages are lonely). Context indicators: {', '.join(context_indicators) if context_indicators else 'none'}. Despite no keywords, the context clearly indicates loneliness.\nClassification: 1"
                    debug_print(f"Lonely without keywords example {i+1}: {row['DM'][:100]}...", important=True)
        
            if not_lonely_with_keywords:
                examples_text += "\n\n=== EXCEPTION EXAMPLES (Keywords but NOT Lonely) ==="
                debug_print(f"Selected {len(not_lonely_with_keywords)} not_lonely_with_keywords examples", important=True)
                for i, row in enumerate(not_lonely_with_keywords):
                    # Analyze why this is an exception
                    text_lower = str(row['DM']).lower()
                    keywords_found = [kw for kw in keywords if kw in text_lower]
                    exception_reason = "unknown"
                    if any(phrase in text_lower for phrase in ['not lonely', 'never lonely', 'not miserable']):
                        exception_reason = "explicit negation"
                    elif any(phrase in text_lower for phrase in ['song', 'music', 'lyrics', 'quote']):
                        exception_reason = "referencing media/art"
                    elif any(phrase in text_lower for phrase in ['joke', 'funny', 'lol', 'haha']):
                        exception_reason = "humorous context"
                    else:
                        exception_reason = "neutral/positive context"
                    
                    examples_text += f"\nMessage: \"{row['DM']}\"\nReasoning: Keywords present: {', '.join(keywords_found)}. This is an EXCEPTION to the pattern (21% of keyword messages are not lonely). Reason: {exception_reason}. Despite keywords, the context indicates this is not about personal loneliness.\nClassification: 0"
                    debug_print(f"Not lonely with keywords example {i+1}: {row['DM'][:100]}...", important=True)
        
    else:
        # Create enhanced simple prompt for binary labeling
        system_prompt = """You are a binary classifier for loneliness detection. Your task is to determine if a person is feeling lonely based on their text messages.

IMPORTANT: Reply with ONLY "1" or "0", nothing else.

Classification rules:
- 1: The person expresses feelings of loneliness, isolation, sadness, or a desire for connection
- 0: The person expresses positive emotions, social connections, or neutral content without loneliness indicators

Consider both explicit statements and implicit meanings."""

        # Build the enhanced few-shot examples
        examples_text = ""
        
        if use_chain_of_thought:
            # Add a chain-of-thought example
            examples_text += "\n\n=== EXAMPLE WITH REASONING ==="
            lonely_example = selected_examples[selected_examples['label'] == 1].iloc[0] if len(selected_examples[selected_examples['label'] == 1]) > 0 else None
            if lonely_example is not None:
                # Analyze keywords, tone, and sentiment for lonely example
                keywords = ["loneliness", "lonely", "social isolation", "solitude", "miserable"]
                keywords_found = [kw for kw in keywords if kw in lonely_example['DM'].lower()]
                tone_analysis = "negative" if any(word in lonely_example['DM'].lower() for word in ['😔', '😞', '😭', '💔', 'sad', 'miserable', 'stuck', 'done', 'anxious', 'scared']) else "neutral"
                sentiment_indicators = []
                if any(word in lonely_example['DM'].lower() for word in ['no friends', 'no family', 'alone', 'isolated', 'missing']):
                    sentiment_indicators.append("social isolation")
                if any(word in lonely_example['DM'].lower() for word in ['want', 'need', 'wish', 'hope']):
                    sentiment_indicators.append("desire for connection")
                if any(word in lonely_example['DM'].lower() for word in ['😔', '😞', '😭', '💔', 'sad', 'depressed']):
                    sentiment_indicators.append("negative emotions")
                
                examples_text += f"\nMessage: \"{lonely_example['DM']}\"\nReasoning: Keywords present: {', '.join(keywords_found) if keywords_found else 'none'}. Tone: {tone_analysis}. Sentiment indicators: {', '.join(sentiment_indicators) if sentiment_indicators else 'none'}. This person expresses loneliness through explicit keywords, negative emotional tone, and context indicating social isolation.\nClassification: 1"
        
        examples_text += "\n\n=== TRAINING EXAMPLES ==="
        for idx, row in selected_examples.iterrows():
            label_text = "1" if row['label'] == 1 else "0"
            examples_text += f"\nMessage: \"{row['DM']}\"\nClassification: {label_text}"
    
    # Create the full prompt with target included for training
    full_prompt = f"{system_prompt}{examples_text}\n\n=== TARGET MESSAGE ===\nMessage: \"{target_text}\"\nClassification:"
    
    return full_prompt

def create_binary_few_shot_dataset(train_df, val_df, examples_per_category=5, seed=42, use_clustering=True, use_augmentation=False):
    """Create few-shot dataset for binary labeling"""
    debug_print(f"create_binary_few_shot_dataset() called with examples_per_category={examples_per_category}", important=True)
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Separate data by class
    lonely_data = train_df[train_df['label'] == 1]
    not_lonely_data = train_df[train_df['label'] == 0]
    
    print(f"\nAvailable examples for binary labeling:")
    print(f"  Lonely (label=1): {len(lonely_data)}")
    print(f"  Not lonely (label=0): {len(not_lonely_data)}")
    
    # Select examples_per_category from each class
    selected_lonely = lonely_data.sample(n=min(examples_per_category, len(lonely_data)), random_state=seed)
    selected_not_lonely = not_lonely_data.sample(n=min(examples_per_category, len(not_lonely_data)), random_state=seed)
    
    # Combine and shuffle
    few_shot_examples_df = pd.concat([selected_lonely, selected_not_lonely], ignore_index=True)
    few_shot_examples_df = few_shot_examples_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"\nBinary few-shot examples selected:")
    print(f"  Total examples: {len(few_shot_examples_df)}")
    print(f"  Lonely examples: {len(few_shot_examples_df[few_shot_examples_df['label'] == 1])}")
    print(f"  Not lonely examples: {len(few_shot_examples_df[few_shot_examples_df['label'] == 0])}")
    
    # For validation, use a smaller subset
    val_subset_size = min(500, len(val_df))
    few_shot_val_df = val_df.sample(n=val_subset_size, random_state=seed)
    
    debug_print("create_binary_few_shot_dataset() completed successfully", important=True)
    return few_shot_examples_df, few_shot_val_df

def create_cluster_few_shot_dataset(train_df, val_df, examples_per_category=5, seed=42, use_clustering=True, use_augmentation=False, exception_ratio=2, no_exceptions=False):
    """Create few-shot dataset for cluster labeling using confusion matrix quadrants with data augmentation"""
    debug_print(f"create_cluster_few_shot_dataset() called with examples_per_category={examples_per_category}, exception_ratio={exception_ratio}", important=True)
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Categorize examples based on cluster assignment vs original label (which corresponds to keyword presence)
    print(f"\nAnalyzing cluster assignment vs original label (keyword presence)...")
    
    # Debug: Check what columns are available
    print(f"Available columns in train_df: {list(train_df.columns)}")
    
    # Debug: Show sample of cluster assignments if available
    if 'cluster' in train_df.columns:
        print(f"Sample cluster assignments:")
        if 'original_label' in train_df.columns:
            print(train_df[['cluster', 'original_label', 'label']].head(10))
            print(f"Cluster distribution:")
            print(train_df['cluster'].value_counts().sort_index())
            print(f"Original label distribution (keyword presence):")
            print(train_df['original_label'].value_counts().sort_index())
            print(f"Cluster-based label distribution:")
            print(train_df['label'].value_counts().sort_index())
        else:
            print(train_df[['cluster', 'label']].head(10))
            print(f"Cluster distribution:")
            print(train_df['cluster'].value_counts().sort_index())
            print(f"Label distribution:")
            print(train_df['label'].value_counts().sort_index())
    elif 'cluster_id' in train_df.columns:
        print(f"Sample cluster assignments (cluster_id):")
        if 'original_label' in train_df.columns:
            print(train_df[['cluster_id', 'original_label', 'label']].head(10))
            print(f"Cluster distribution:")
            print(train_df['cluster_id'].value_counts().sort_index())
            print(f"Original label distribution (keyword presence):")
            print(train_df['original_label'].value_counts().sort_index())
            print(f"Cluster-based label distribution:")
            print(train_df['label'].value_counts().sort_index())
        else:
            print(train_df[['cluster_id', 'label']].head(10))
            print(f"Cluster distribution:")
            print(train_df['cluster_id'].value_counts().sort_index())
            print(f"Label distribution:")
            print(train_df['label'].value_counts().sort_index())
    else:
        print("WARNING: No cluster column found!")
        print("Available columns:", list(train_df.columns))
    
    # Based on the confusion matrix: cluster 0 (lonely) vs cluster 1 (not lonely)
    # and original labels: 1 (lonely/keywords) vs 0 (not lonely/no keywords)
    
    cluster_0_label_1 = []    # Cluster 0, original_label 1 (lonely cluster, lonely label) - PATTERN
    cluster_1_label_0 = []    # Cluster 1, original_label 0 (not lonely cluster, not lonely label) - PATTERN
    cluster_0_label_0 = []    # Cluster 0, original_label 0 (lonely cluster, not lonely label) - EXCEPTION
    cluster_1_label_1 = []    # Cluster 1, original_label 1 (not lonely cluster, lonely label) - EXCEPTION
    
    for idx, row in train_df.iterrows():
        # Get cluster assignment and original label (keyword presence)
        cluster = row.get('cluster_id', row.get('cluster', 0))  # Try both possible column names
        original_label = row.get('original_label', row['label'])  # Use original_label if available, otherwise fall back to label
        
        if cluster == 0 and original_label == 1:
            cluster_0_label_1.append(row)  # PATTERN: lonely cluster, lonely label
        elif cluster == 1 and original_label == 0:
            cluster_1_label_0.append(row)  # PATTERN: not lonely cluster, not lonely label
        elif cluster == 0 and original_label == 0:
            cluster_0_label_0.append(row)  # EXCEPTION: lonely cluster, not lonely label
        elif cluster == 1 and original_label == 1:
            cluster_1_label_1.append(row)  # EXCEPTION: not lonely cluster, lonely label
    
    print(f"\nConfusion matrix analysis:")
    print(f"  Cluster 0, Label 1 (PATTERN - lonely cluster, lonely label): {len(cluster_0_label_1)}")
    print(f"  Cluster 1, Label 0 (PATTERN - not lonely cluster, not lonely label): {len(cluster_1_label_0)}")
    print(f"  Cluster 0, Label 0 (EXCEPTION - lonely cluster, not lonely label): {len(cluster_0_label_0)}")
    print(f"  Cluster 1, Label 1 (EXCEPTION - not lonely cluster, lonely label): {len(cluster_1_label_1)}")
    
    # Select examples from each quadrant of the confusion matrix
    selected_examples = []
    
    # Calculate exception count: floor(examples_per_category/exception_ratio) or 0 if no_exceptions flag is set
    if no_exceptions:
        exception_count = 0
        debug_print("No exceptions flag set - using 0 exception examples", important=True)
    else:
        exception_count = max(1, examples_per_category // exception_ratio)  # Use floor division with exception_ratio
    
    print(f"\nTarget selection:")
    print(f"  Pattern examples (Cluster 0, Label 1): {examples_per_category}")
    print(f"  Pattern examples (Cluster 1, Label 0): {examples_per_category}")
    print(f"  Exception examples (Cluster 0, Label 0): {exception_count} (ratio: {exception_ratio})")
    print(f"  Exception examples (Cluster 1, Label 1): {exception_count} (ratio: {exception_ratio})")
    print(f"  Total expected: {2 * examples_per_category + 2 * exception_count}")
    
    # Always use centroid-based selection for all quadrants
    print(f"\nUsing centroid-based selection for confusion matrix quadrants...")
    
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 1. Cluster 0, Label 1 (PATTERN) - use examples_per_category
        print(f"Selecting {examples_per_category} cluster 0, label 1 examples (PATTERN)...")
        if len(cluster_0_label_1) > 0:
            texts = [row['DM'] for row in cluster_0_label_1]
            embeddings = embedder.encode(texts, batch_size=64, convert_to_numpy=True)
            # Use controlled random subsampling for more diversity
            selected_indices = controlled_random_subsampling(embeddings, texts, examples_per_category, oversample_factor=1)
            # Convert back to row objects using indices
            selected_examples.extend([cluster_0_label_1[i] for i in selected_indices])
            print(f"Selected {len(selected_indices)} cluster 0, label 1 examples using controlled random subsampling")
        else:
            print(f"WARNING: No examples found for cluster 0, label 1")
        
        # 2. Cluster 1, Label 0 (PATTERN) - use examples_per_category
        print(f"Selecting {examples_per_category} cluster 1, label 0 examples (PATTERN)...")
        if len(cluster_1_label_0) > 0:
            texts = [row['DM'] for row in cluster_1_label_0]
            embeddings = embedder.encode(texts, batch_size=64, convert_to_numpy=True)
            # Use controlled random subsampling for more diversity
            selected_indices = controlled_random_subsampling(embeddings, texts, examples_per_category, oversample_factor=5)
            # Convert back to row objects using indices
            selected_examples.extend([cluster_1_label_0[i] for i in selected_indices])
            print(f"Selected {len(selected_indices)} cluster 1, label 0 examples using controlled random subsampling")
        else:
            print(f"WARNING: No examples found for cluster 1, label 0")
        
        # 3. Cluster 0, Label 0 (EXCEPTION) - use floor(examples_per_category/exception_ratio) or skip if no_exceptions
        if exception_count > 0:
            print(f"Selecting {exception_count} cluster 0, label 0 examples (EXCEPTION)...")
            if len(cluster_0_label_0) > 0:
                texts = [row['DM'] for row in cluster_0_label_0]
                embeddings = embedder.encode(texts, batch_size=64, convert_to_numpy=True)
                # Use controlled random subsampling for more diversity
                selected_indices = controlled_random_subsampling(embeddings, texts, exception_count, oversample_factor=5)
                # Convert back to row objects using indices
                selected_examples.extend([cluster_0_label_0[i] for i in selected_indices])
                print(f"Selected {len(selected_indices)} cluster 0, label 0 examples using controlled random subsampling")
            else:
                print(f"WARNING: No examples found for cluster 0, label 0")
        else:
            print("Skipping cluster 0, label 0 examples (no exceptions flag set)")
        
        # 4. Cluster 1, Label 1 (EXCEPTION) - use floor(examples_per_category/exception_ratio) or skip if no_exceptions
        if exception_count > 0:
            print(f"Selecting {exception_count} cluster 1, label 1 examples (EXCEPTION)...")
            if len(cluster_1_label_1) > 0:
                texts = [row['DM'] for row in cluster_1_label_1]
                embeddings = embedder.encode(texts, batch_size=64, convert_to_numpy=True)
                # Use controlled random subsampling for more diversity
                selected_indices = controlled_random_subsampling(embeddings, texts, exception_count, oversample_factor=5)
                # Convert back to row objects using indices
                selected_examples.extend([cluster_1_label_1[i] for i in selected_indices])
                print(f"Selected {len(selected_indices)} cluster 1, label 1 examples using controlled random subsampling")
            else:
                print(f"WARNING: No examples found for cluster 1, label 1")
        else:
            print("Skipping cluster 1, label 1 examples (no exceptions flag set)")
    
    except Exception as e:
        print(f"Error with sentence transformers, falling back to random selection: {e}")
        # Fallback to random selection
        print(f"\nUsing random selection for confusion matrix quadrants...")
        
        # 1. Cluster 0, Label 1 (PATTERN) - use examples_per_category
        print(f"Selecting {examples_per_category} cluster 0, label 1 examples (PATTERN)...")
        if len(cluster_0_label_1) >= examples_per_category:
            selected_examples.extend(random.sample(cluster_0_label_1, examples_per_category))
            print(f"Selected {examples_per_category} cluster 0, label 1 examples")
        else:
            selected_examples.extend(cluster_0_label_1)
            print(f"Selected {len(cluster_0_label_1)} cluster 0, label 1 examples (all available)")
        
        # 2. Cluster 1, Label 0 (PATTERN) - use examples_per_category
        print(f"Selecting {examples_per_category} cluster 1, label 0 examples (PATTERN)...")
        if len(cluster_1_label_0) >= examples_per_category:
            selected_examples.extend(random.sample(cluster_1_label_0, examples_per_category))
            print(f"Selected {examples_per_category} cluster 1, label 0 examples")
        else:
            selected_examples.extend(cluster_1_label_0)
            print(f"Selected {len(cluster_1_label_0)} cluster 1, label 0 examples (all available)")
        
        # 3. Cluster 0, Label 0 (EXCEPTION) - use floor(examples_per_category/exception_ratio) or skip if no_exceptions
        if exception_count > 0:
            print(f"Selecting {exception_count} cluster 0, label 0 examples (EXCEPTION)...")
            if len(cluster_0_label_0) >= exception_count:
                selected_examples.extend(random.sample(cluster_0_label_0, exception_count))
                print(f"Selected {exception_count} cluster 0, label 0 examples")
            else:
                selected_examples.extend(cluster_0_label_0)
                print(f"Selected {len(cluster_0_label_0)} cluster 0, label 0 examples (all available)")
        else:
            print("Skipping cluster 0, label 0 examples (no exceptions flag set)")
        
        # 4. Cluster 1, Label 1 (EXCEPTION) - use floor(examples_per_category/exception_ratio) or skip if no_exceptions
        if exception_count > 0:
            print(f"Selecting {exception_count} cluster 1, label 1 examples (EXCEPTION)...")
            if len(cluster_1_label_1) >= exception_count:
                selected_examples.extend(random.sample(cluster_1_label_1, exception_count))
                print(f"Selected {exception_count} cluster 1, label 1 examples")
            else:
                selected_examples.extend(cluster_1_label_1)
                print(f"Selected {len(cluster_1_label_1)} cluster 1, label 1 examples (all available)")
        else:
            print("Skipping cluster 1, label 1 examples (no exceptions flag set)")
    
    # Apply data augmentation if requested
    if use_augmentation and len(selected_examples) > 0:
        print(f"\nApplying data augmentation...")
        augmented_examples = []
        for example in selected_examples:
            # Add original example
            augmented_examples.append(example)
            
            # Add augmented version
            try:
                augmented_text = augment_example(example['DM'], method='paraphrase')
                if augmented_text and len(augmented_text) > 10:
                    augmented_example = example.copy()
                    augmented_example['DM'] = augmented_text
                    augmented_examples.append(augmented_example)
                    print(f"Added augmented example: {augmented_text[:100]}...")
            except Exception as e:
                print(f"Error augmenting example: {e}")
        
        selected_examples = augmented_examples
        print(f"Data augmentation completed. Total examples: {len(selected_examples)}")
    
    # Convert to DataFrame and shuffle
    few_shot_examples_df = pd.DataFrame(selected_examples)
    few_shot_examples_df = few_shot_examples_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Summary of selected examples by quadrant
    print(f"\n=== FINAL SELECTION SUMMARY ===")
    print(f"Total examples selected: {len(few_shot_examples_df)}")
    
    # Count examples by cluster and original label
    cluster_0_label_1_count = len(few_shot_examples_df[(few_shot_examples_df.get('cluster', 0) == 0) & (few_shot_examples_df.get('original_label', few_shot_examples_df['label']) == 1)])
    cluster_1_label_0_count = len(few_shot_examples_df[(few_shot_examples_df.get('cluster', 0) == 1) & (few_shot_examples_df.get('original_label', few_shot_examples_df['label']) == 0)])
    cluster_0_label_0_count = len(few_shot_examples_df[(few_shot_examples_df.get('cluster', 0) == 0) & (few_shot_examples_df.get('original_label', few_shot_examples_df['label']) == 0)])
    cluster_1_label_1_count = len(few_shot_examples_df[(few_shot_examples_df.get('cluster', 0) == 1) & (few_shot_examples_df.get('original_label', few_shot_examples_df['label']) == 1)])
    
    print(f"  Cluster 0, Label 1 (Keywords = Lonely): {cluster_0_label_1_count} examples")
    print(f"  Cluster 1, Label 0 (No Keywords = Not Lonely): {cluster_1_label_0_count} examples")
    print(f"  Cluster 0, Label 0 (No Keywords but Lonely): {cluster_0_label_0_count} examples")
    print(f"  Cluster 1, Label 1 (Keywords but Not Lonely): {cluster_1_label_1_count} examples")
    
    # Verify we have all quadrants represented
    expected_total = examples_per_category + examples_per_category + exception_count + exception_count
    if len(few_shot_examples_df) != expected_total:
        print(f"WARNING: Expected {expected_total} examples but got {len(few_shot_examples_df)}")
    
    if cluster_0_label_1_count == 0:
        print(f"WARNING: No examples selected for Cluster 0, Label 1 (Keywords = Lonely)")
    if cluster_1_label_0_count == 0:
        print(f"WARNING: No examples selected for Cluster 1, Label 0 (No Keywords = Not Lonely)")
    if cluster_0_label_0_count == 0:
        print(f"WARNING: No examples selected for Cluster 0, Label 0 (No Keywords but Lonely)")
    if cluster_1_label_1_count == 0:
        print(f"WARNING: No examples selected for Cluster 1, Label 1 (Keywords but Not Lonely)")
    
    print(f"\nCluster few-shot examples selected:")
    print(f"  Total examples: {len(few_shot_examples_df)}")
    print(f"  Lonely examples (original_label=1): {len(few_shot_examples_df[few_shot_examples_df.get('original_label', few_shot_examples_df['label']) == 1])}")
    print(f"  Not lonely examples (original_label=0): {len(few_shot_examples_df[few_shot_examples_df.get('original_label', few_shot_examples_df['label']) == 0])}")
    
    # For validation, use a smaller subset
    few_shot_val_df = val_df.sample(n=min(500, len(val_df)), random_state=seed).reset_index(drop=True)
    
    return few_shot_examples_df, few_shot_val_df

def create_few_shot_dataset(train_df, val_df, examples_per_category=5, seed=42, cluster_labeling=False, use_clustering=True, use_augmentation=False, exception_ratio=2, no_exceptions=False):
    """Main function to create few-shot dataset based on labeling type"""
    if cluster_labeling:
        return create_cluster_few_shot_dataset(train_df, val_df, examples_per_category, seed, use_clustering, use_augmentation, exception_ratio, no_exceptions)
    else:
        return create_binary_few_shot_dataset(train_df, val_df, examples_per_category, seed, use_clustering, use_augmentation)

def tokenize_function(examples, tokenizer, max_length=512, cluster_labeling=False, few_shot_examples_df=None, use_chain_of_thought=True, use_semantic_selection=False, train_df=None, is_training=True):
    """Tokenize the text data with enhanced few-shot prompts"""
    # For few-shot learning, we need to create prompts for each example
    prompts = []
    labels = []
    
    for i, text in enumerate(examples["DM"]):
        if few_shot_examples_df is not None:
            # For training: use fixed examples (no semantic selection)
            # For inference: use semantic selection if enabled
            if is_training or not use_semantic_selection:
                # Use the fixed examples from dataset creation
                prompt = create_few_shot_prompt(
                    few_shot_examples_df, 
                    text, 
                    cluster_labeling, 
                    use_chain_of_thought=use_chain_of_thought,
                    use_semantic_selection=False,  # Never use semantic selection during training
                    train_df=None
                )
            else:
                # Use semantic selection for inference
                prompt = create_few_shot_prompt(
                    few_shot_examples_df, 
                    text, 
                    cluster_labeling, 
                    use_chain_of_thought=use_chain_of_thought,
                    use_semantic_selection=True,
                    train_df=train_df
                )
        else:
            # Fallback to simple prompts if no few-shot examples provided
            if cluster_labeling:
                keywords = ["loneliness", "lonely", "social isolation", "solitude", "miserable"]
                prompt = f"You are a binary classifier for loneliness detection. Reply with ONLY '1' or '0'.\nKeywords to watch for: {', '.join(keywords)}\nMessage: \"{text}\"\nClassification:"
            else:
                prompt = f"You are a binary classifier for loneliness detection. Reply with ONLY '1' or '0'.\nMessage: \"{text}\"\nClassification:"
        
        prompts.append(prompt)
        labels.append(examples["label"][i])
        
        # Print the first few prompts for debugging
        if i < 3:  # Only print first 3 prompts to avoid spam
            debug_print(f"=== PROMPT {i+1} ===", important=True)
            debug_print(f"Text: {text[:100]}...", important=True)
            debug_print(f"Label: {examples['label'][i]}", important=True)
            debug_print(f"Full prompt:\n{prompt}", important=True)
            debug_print(f"=== END PROMPT {i+1} ===", important=True)
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )
    
    # Debug: Check for truncation
    for i, (prompt, input_ids) in enumerate(zip(prompts, tokenized["input_ids"])):
        if len(input_ids) >= max_length:
            debug_print(f"WARNING: Prompt {i+1} was TRUNCATED! Original length: {len(tokenizer.encode(prompt))}, Max length: {max_length}, Truncated length: {len(input_ids)}", important=True)
            debug_print(f"Truncated prompt preview: {prompt[:500]}...", important=True)
        else:
            debug_print(f"Prompt {i+1} length: {len(input_ids)} tokens (max: {max_length})", important=True)
    
    # Add labels back
    tokenized["labels"] = labels
    
    return tokenized

def get_model_memory_footprint(model, return_bytes=False):
    """Helper function to get the memory footprint of a PyTorch model."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    if return_bytes:
        return total_bytes
    return total_bytes / (1024**3)

MODEL_PATH = "path/to/your/model"  # Update with your model path


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    debug_print(f"compute_metrics called with logits shape: {logits.shape}, labels shape: {labels.shape}", important=True)
    
    # Ensure inputs are numpy arrays and handle device placement
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    debug_print(f"After conversion - logits shape: {logits.shape}, labels shape: {labels.shape}", important=True)
    debug_print(f"Labels unique values: {np.unique(labels)}", important=True)
    
    # For sequence classification, logits shape is (batch_size, num_classes)
    # Get predictions using argmax
    preds = np.argmax(logits, axis=-1)
    
    debug_print(f"Predictions shape: {preds.shape}, unique values: {np.unique(preds)}", important=True)
    
    # Calculate metrics with focus on F1 score
    acc = (preds == labels).mean()
    
    # Calculate precision, recall, F1 with better handling of edge cases
    try:
        precision = precision_score(labels, preds, average='binary', zero_division=0)
        recall = recall_score(labels, preds, average='binary', zero_division=0)
        f1 = f1_score(labels, preds, average='binary', zero_division=0)
        
        # Additional metrics for better understanding
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            specificity = 0
            sensitivity = 0
        
        debug_print(f"Metrics computed successfully: acc={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, specificity={specificity:.4f}, sensitivity={sensitivity:.4f}", important=True)
    except Exception as e:
        debug_print(f"Error computing metrics: {e}", important=True)
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        specificity = 0.0
        sensitivity = 0.0
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "sensitivity": sensitivity
    }
    
    debug_print(f"Returning metrics: {metrics}", important=True)
    return metrics

def calibrate_decision_threshold(model, val_dataset, tokenizer, few_shot_examples_df=None, cluster_labeling=False, use_chain_of_thought=True, use_semantic_selection=False, train_df=None):
    """Calibrate decision threshold for better few-shot performance"""
    debug_print("calibrate_decision_threshold() called", important=True)
    
    try:
        # Get predictions on validation set
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(min(100, len(val_dataset))):  # Use subset for calibration
                sample = val_dataset[i]
                
                # Create prompt for this sample
                if few_shot_examples_df is not None:
                    prompt = create_few_shot_prompt(
                        few_shot_examples_df, 
                        sample['DM'], 
                        cluster_labeling, 
                        use_chain_of_thought=use_chain_of_thought,
                        use_semantic_selection=use_semantic_selection,
                        train_df=train_df
                    )
                else:
                    if cluster_labeling:
                        keywords = ["loneliness", "lonely", "social isolation", "solitude", "miserable"]
                        prompt = f"You are a binary classifier for loneliness detection. Reply with ONLY '1' or '0'.\nKeywords to watch for: {', '.join(keywords)}\nMessage: \"{sample['DM']}\"\nClassification:"
                    else:
                        prompt = f"You are a binary classifier for loneliness detection. Reply with ONLY '1' or '0'.\nMessage: \"{sample['DM']}\"\nClassification:"
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Get logits
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs[0, 1].cpu().item())  # Probability of class 1 (lonely)
                all_labels.append(sample['label'])
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Find optimal threshold by maximizing F1 score
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs > threshold).astype(int)
            f1 = f1_score(all_labels, preds, average='binary', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        debug_print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})", important=True)
        
        return best_threshold
        
    except Exception as e:
        debug_print(f"Error in threshold calibration: {e}", important=True)
        return 0.5  # Default threshold

def classify_text_with_semantic_few_shot(text, model, tokenizer, train_df, cluster_labeling=False, use_chain_of_thought=True, max_examples=8):
    """Classify a single text using semantic few-shot learning with dynamic example selection"""
    debug_print(f"classify_text_with_semantic_few_shot() called for text: {text[:100]}...", important=True)
    
    try:
        # 1. Retrieve similar examples using semantic similarity
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            similar_examples = select_semantic_examples(text, train_df, k=max_examples, balance_classes=True)
            debug_print(f"Retrieved {len(similar_examples)} similar examples using semantic selection", important=True)
        else:
            # Fallback to random selection if sentence transformers not available
            similar_examples = select_random_examples(train_df, k=max_examples, balance_classes=True)
            debug_print(f"Retrieved {len(similar_examples)} random examples (semantic selection not available)", important=True)
        
        # 2. Build custom prompt with retrieved examples
        prompt = create_few_shot_prompt(
            similar_examples, 
            text, 
            cluster_labeling, 
            use_chain_of_thought=use_chain_of_thought,
            use_semantic_selection=False,  # We already did the selection above
            train_df=None
        )
        
        # 3. Tokenize the prompt
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=3000
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 4. Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0, prediction].item()
        
        debug_print(f"Prediction: {prediction} (confidence: {confidence:.3f})", important=True)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'examples_used': len(similar_examples),
            'prompt_length': len(inputs['input_ids'][0])
        }
        
    except Exception as e:
        debug_print(f"Error in semantic few-shot classification: {e}", important=True)
        return {
            'prediction': 0,
            'confidence': 0.0,
            'probabilities': [0.5, 0.5],
            'examples_used': 0,
            'prompt_length': 0,
            'error': str(e)
        }

def batch_classify_with_semantic_few_shot(texts, model, tokenizer, train_df, cluster_labeling=False, use_chain_of_thought=True, max_examples=8, batch_size=4):
    """Classify multiple texts using semantic few-shot learning"""
    debug_print(f"batch_classify_with_semantic_few_shot() called for {len(texts)} texts", important=True)
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_results = []
        
        for text in batch_texts:
            result = classify_text_with_semantic_few_shot(
                text, model, tokenizer, train_df, 
                cluster_labeling, use_chain_of_thought, max_examples
            )
            batch_results.append(result)
        
        results.extend(batch_results)
        
        # Clear GPU cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    debug_print(f"Completed classification of {len(texts)} texts", important=True)
    return results

def get_optimized_hyperparams():
    """Get optimized hyperparameters for enhanced few-shot learning"""
    return {
        "learning_rate": 5e-5,  # Moderate learning rate for stability
        "weight_decay": 0.01,   # Slightly higher regularization to prevent overfitting
        "per_device_train_batch_size": 1,  # Smaller batch size for better gradient updates
        "lora_r": 64,           # Higher LoRA rank for better expressiveness
        "gradient_accumulation_steps": 8,  # Effective batch size = 8
        "num_train_epochs": 5,  # More epochs for better convergence
        "warmup_ratio": 0.0,    # No warmup for small datasets
        "early_stopping_patience": 3,  # More patience for early stopping
        "probability_decision_boundary": 0.5,
        "fp16": True,           # Enable fp16 for faster training
        "max_length": 3000,     # Shorter context for faster training
        "eval_steps": 1,        # Evaluate every step for better monitoring
        "logging_steps": 1,     # Log every step for better monitoring
        "save_steps": 5,        # Save every 5 steps
        "gradient_checkpointing": True,  # Enable for memory efficiency
    }

def sample_incorrect_predictions(model, val_dataset, tokenizer, few_shot_examples_df, cluster_labeling, use_chain_of_thought, train_df, sample_size=20, seed=42):
    """Sample and log incorrectly classified messages from validation set"""
    debug_print(f"sample_incorrect_predictions() called with sample_size={sample_size}", important=True)
    
    try:
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        model.eval()
        incorrect_predictions = []
        
        # Get predictions on validation set
        debug_print(f"Starting prediction loop for {len(val_dataset)} validation samples", important=True)
        with torch.no_grad():
            for i in range(len(val_dataset)):
                if i % 100 == 0:  # Progress indicator
                    debug_print(f"Processing validation sample {i}/{len(val_dataset)}", important=True)
                sample = val_dataset[i]
                
                # Create prompt for this sample
                if few_shot_examples_df is not None:
                    prompt = create_few_shot_prompt(
                        few_shot_examples_df, 
                        sample['DM'], 
                        cluster_labeling, 
                        use_chain_of_thought=use_chain_of_thought,
                        use_semantic_selection=False,
                        train_df=None
                    )
                else:
                    if cluster_labeling:
                        keywords = ["loneliness", "lonely", "social isolation", "solitude", "miserable"]
                        prompt = f"You are a binary classifier for loneliness detection. Reply with ONLY '1' or '0'.\nKeywords to watch for: {', '.join(keywords)}\nMessage: \"{sample['DM']}\"\nClassification:"
                    else:
                        prompt = f"You are a binary classifier for loneliness detection. Reply with ONLY '1' or '0'.\nMessage: \"{sample['DM']}\"\nClassification:"
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Get prediction
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = torch.softmax(logits, dim=-1)[0, prediction].item()
                
                # Check if prediction is incorrect
                true_label = sample['label']
                if i < 3:  # Debug first few samples
                    debug_print(f"Sample {i}: text='{sample['DM'][:100]}...', true_label={true_label}, prediction={prediction}, confidence={confidence:.3f}", important=True)
                if prediction != true_label:
                    incorrect_predictions.append({
                        'text': sample['DM'],
                        'true_label': true_label,
                        'predicted_label': prediction,
                        'confidence': confidence,
                        'cluster': sample.get('cluster', sample.get('cluster_id', 'N/A')),
                        'original_label': sample.get('original_label', 'N/A'),
                        'index': i
                    })
        
        # Randomly sample from incorrect predictions
        if len(incorrect_predictions) > sample_size:
            sampled_incorrect = random.sample(incorrect_predictions, sample_size)
        else:
            sampled_incorrect = incorrect_predictions
        
        # Log the results
        print(f"\n{'='*80}")
        print(f"SAMPLED INCORRECT PREDICTIONS ({len(sampled_incorrect)} out of {len(incorrect_predictions)} total incorrect)")
        print(f"{'='*80}")
        
        for i, pred in enumerate(sampled_incorrect, 1):
            print(f"\n{i:2d}. INCORRECT PREDICTION")
            print(f"    Text: \"{pred['text']}\"")
            print(f"    True Label: {pred['true_label']} ({'LONELY' if pred['true_label'] == 1 else 'NOT LONELY'})")
            print(f"    Predicted: {pred['predicted_label']} ({'LONELY' if pred['predicted_label'] == 1 else 'NOT LONELY'})")
            print(f"    Confidence: {pred['confidence']:.3f}")
            print(f"    Cluster: {pred['cluster']}")
            print(f"    Original Label: {pred['original_label']}")
            print(f"    Index: {pred['index']}")
            print(f"    {'-'*60}")
        
        # Summary statistics
        print(f"\nSUMMARY:")
        print(f"  Total validation samples: {len(val_dataset)}")
        print(f"  Total incorrect predictions: {len(incorrect_predictions)}")
        print(f"  Error rate: {len(incorrect_predictions)/len(val_dataset)*100:.2f}%")
        print(f"  Sampled incorrect predictions: {len(sampled_incorrect)}")
        
        # Analyze by cluster and label combinations
        if cluster_labeling:
            print(f"\nBREAKDOWN BY CLUSTER/LABEL COMBINATIONS:")
            cluster_label_counts = {}
            for pred in incorrect_predictions:
                key = f"Cluster_{pred['cluster']}_TrueLabel_{pred['true_label']}"
                cluster_label_counts[key] = cluster_label_counts.get(key, 0) + 1
            
            for key, count in sorted(cluster_label_counts.items()):
                print(f"  {key}: {count} incorrect predictions")
        
        print(f"{'='*80}\n")
        
        # Also save to file for later analysis
        with open("incorrect_predictions_sample.txt", "w") as f:
            f.write(f"SAMPLED INCORRECT PREDICTIONS\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n")
            f.write(f"Sample size: {len(sampled_incorrect)}\n")
            f.write(f"Total incorrect: {len(incorrect_predictions)}\n")
            f.write(f"Error rate: {len(incorrect_predictions)/len(val_dataset)*100:.2f}%\n")
            f.write(f"{'='*80}\n\n")
            
            for i, pred in enumerate(sampled_incorrect, 1):
                f.write(f"{i:2d}. INCORRECT PREDICTION\n")
                f.write(f"    Text: \"{pred['text']}\"\n")
                f.write(f"    True Label: {pred['true_label']} ({'LONELY' if pred['true_label'] == 1 else 'NOT LONELY'})\n")
                f.write(f"    Predicted: {pred['predicted_label']} ({'LONELY' if pred['predicted_label'] == 1 else 'NOT LONELY'})\n")
                f.write(f"    Confidence: {pred['confidence']:.3f}\n")
                f.write(f"    Cluster: {pred['cluster']}\n")
                f.write(f"    Original Label: {pred['original_label']}\n")
                f.write(f"    Index: {pred['index']}\n")
                f.write(f"    {'-'*60}\n\n")
        
        debug_print(f"Saved incorrect predictions sample to incorrect_predictions_sample.txt", important=True)
        
    except Exception as e:
        debug_print(f"Error sampling incorrect predictions: {e}", important=True)
        print(f"Error sampling incorrect predictions: {e}")

def run_few_shot_training(tokenizer, train_dataset, val_dataset, examples_per_category, cluster_labeling, use_chain_of_thought=True, use_semantic_selection=False, train_df=None, few_shot_examples_df=None):
    """Run enhanced few-shot training with optimized hyperparameters"""
    debug_print("run_few_shot_training() called", important=True)
    
    fresh_model = None
    
    try:
        # Get optimized hyperparameters
        hyperparams = get_optimized_hyperparams()
        debug_print(f"Using enhanced hyperparameters: {hyperparams}", important=True)
        
        # Select GPU
        N = torch.cuda.device_count()
        device_idx = 0  # Use first GPU for simplicity
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
        torch.cuda.set_device(0)
        debug_print(f"Using GPU {device_idx} (visible as cuda:0 in this process)", important=True)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            debug_gpu_memory("After proactive cache clear - ", important=True)
        
        # Save training data
        training_data = {
            "examples_per_class": examples_per_category,
            "hyperparams": hyperparams,
            "cluster_labeling": cluster_labeling,
            "start_time": datetime.now().isoformat(),
            "trial_type": "few_shot_simple"
        }
        
        with open(f"few_shot_simple_training_data.pkl", "wb") as f:
            pickle.dump(training_data, f)
        debug_print("Training data saved")
        
        # Load model
        try:
            debug_print("Loading model with 8-bit quantization...", important=True)
            
            # Use 8-bit quantization to reduce memory usage
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            with FileLock("model_load.lock"):
                fresh_model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_PATH,
                    quantization_config=bnb_config,
                    device_map={'': device_idx},
                    torch_dtype=torch.bfloat16,
                    num_labels=2,  # Binary classification: 0 and 1
                    problem_type="single_label_classification",
                    trust_remote_code=True,
                )
            
            # Set label mapping for binary classification
            fresh_model.config.label2id = {"not_lonely": 0, "lonely": 1}
            fresh_model.config.id2label = {0: "not_lonely", 1: "lonely"}
            
            debug_print("Loaded sequence classification model for binary classification", important=True)

            if tokenizer.pad_token is not None:
                fresh_model.config.pad_token_id = tokenizer.pad_token_id
                debug_print(f"Set pad_token_id to {tokenizer.pad_token_id}", important=True)
            else:
                debug_print("WARNING: tokenizer.pad_token is None!", important=True)

            debug_print("Model loaded successfully!", important=True)
            
            fresh_model.config.use_cache = False
            
            if torch.cuda.is_available():
                debug_gpu_memory("After model loading - ", important=True)
                
        except Exception as e:
            debug_print(f"Error loading model: {e}", important=True)
            raise e
        
        # Apply LoRA with enhanced configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=hyperparams["lora_r"],
            lora_alpha=64,  # Increased alpha for better expressiveness
            lora_dropout=0.05,  # Lower dropout for better stability
            target_modules=["q_proj","v_proj","k_proj","o_proj"],
            bias="none",  # Don't train bias for stability
        )
        fresh_model = get_peft_model(fresh_model, lora_config)
        fresh_model.config.use_cache = False
        
        # Prepare model for k-bit training
        fresh_model = prepare_model_for_kbit_training(fresh_model)

        # Ensure LoRA layers are trainable
        for name, param in fresh_model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
                debug_print(f"Set {name} to trainable", important=True)
        
        # Unfreeze the classification head for better few-shot learning
        for name, param in fresh_model.named_parameters():
            if name.startswith("score."):
                param.requires_grad = True
                debug_print(f"Unfroze classification head: {name}", important=True)
        
        # Print trainable parameters to verify
        fresh_model.print_trainable_parameters()
        
        # Ensure the model is in training mode and gradients are enabled
        fresh_model.train()
        fresh_model.enable_input_require_grads()
        
        # Force gradient computation to be enabled
        for param in fresh_model.parameters():
            if param.requires_grad:
                param.retain_grad()

        debug_print("LoRA applied successfully and caching disabled", important=True)

        # Training arguments optimized for enhanced few-shot learning
        training_args = TrainingArguments(
            output_dir="./results/few_shot_enhanced",
            # Enhanced hyperparameters
            num_train_epochs=hyperparams["num_train_epochs"],
            per_device_train_batch_size=hyperparams["per_device_train_batch_size"],
            gradient_accumulation_steps=hyperparams["gradient_accumulation_steps"],
            learning_rate=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
            warmup_ratio=hyperparams["warmup_ratio"],
            # Enhanced few-shot specific settings
            fp16=hyperparams["fp16"],
            gradient_checkpointing=hyperparams["gradient_checkpointing"],
            torch_compile=False,  # Disable torch.compile to avoid optimizer warnings with quantization
            optim="adamw_torch",  # Use standard AdamW optimizer for stability
            eval_strategy="steps",
            eval_steps=hyperparams["eval_steps"],  # More frequent evaluation
            logging_strategy="steps",
            logging_steps=hyperparams["logging_steps"],  # More frequent logging
            save_strategy="steps",
            save_steps=hyperparams["save_steps"],  # More frequent saving
            load_best_model_at_end=True,
            # Reporting
            do_eval=True,
            report_to="none",
            save_total_limit=5,  # Keep more checkpoints
            metric_for_best_model="f1",
            greater_is_better=True,
            # Explicitly tell Trainer which column contains labels
            label_names=["labels"],
            # Additional stability settings for enhanced few-shot learning
            dataloader_pin_memory=False,  # Disable pin memory to avoid CUDA issues
            remove_unused_columns=False,  # Keep all columns to avoid data issues
            # Enhanced settings for better few-shot performance
            lr_scheduler_type="constant",  # Constant learning rate for small datasets
            warmup_steps=0,  # Use warmup_ratio instead
            max_grad_norm=1.0,  # Stronger gradient clipping for stability
            # Additional stability settings
            dataloader_num_workers=0,  # Disable multiprocessing for stability
            group_by_length=False,  # Disable length grouping for stability
            # Better optimization settings
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            # Enhanced logging
            logging_first_step=True,
            logging_dir="./logs",
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Debug dataset info
        debug_print(f"Train dataset columns: {train_dataset.column_names}", important=True)
        debug_print(f"Val dataset columns: {val_dataset.column_names}", important=True)
        debug_print(f"Train dataset size: {len(train_dataset)}", important=True)
        debug_print(f"Val dataset size: {len(val_dataset)}", important=True)
        
        # Check a few samples
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            debug_print(f"Train sample {i}: {sample}", important=True)
        
        for i in range(min(3, len(val_dataset))):
            sample = val_dataset[i]
            debug_print(f"Val sample {i}: {sample}", important=True)
        
        # Create trainer with custom loss function
        trainer = Trainer(
            model=fresh_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=hyperparams.get("early_stopping_patience", 3),
                    early_stopping_threshold=0.005  # Lower threshold for better F1
                )
            ]
        )
        
        # Override the compute_loss method to use weighted loss
        original_compute_loss = trainer.compute_loss
        
        def weighted_compute_loss(model, inputs, return_outputs=False, **kwargs):
            """Custom loss function that weights positive class more for better F1"""
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Use weighted cross-entropy loss for better F1 score
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 2.0]).to(logits.device),  # Weight positive class more
                reduction='mean'
            )
            
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
        
        trainer.compute_loss = weighted_compute_loss
        
        # Train the model
        debug_print("Starting training...", important=True)
        if cluster_labeling:
            print(f"Starting few-shot training with {examples_per_category} examples per category (4 categories = {examples_per_category * 4} total)...")
        else:
            print(f"Starting few-shot training with {examples_per_category} examples per class ({examples_per_category * 2} total)...")
        
        if torch.cuda.is_available():
            debug_print(f"Model device before training: {next(fresh_model.parameters()).device}", important=True)
        
        train_result = trainer.train()
        debug_print("Training completed", important=True)
        
        # Evaluate
        debug_print("Evaluating...", important=True)
        print("Evaluating...")
        eval_result = trainer.evaluate()
        debug_print("Evaluation completed", important=True)
        
        # Get the best metric
        best_metric = eval_result.get("eval_f1", 0.0)
        
        debug_print(f"Training completed. Best metric: {best_metric}", important=True)
        print(f"Training completed. Best metric: {best_metric}")
        
        # Sample and log incorrectly classified messages
        debug_print("Sampling incorrectly classified messages...", important=True)
        try:
            sample_incorrect_predictions(
                fresh_model, val_dataset, tokenizer, few_shot_examples_df, 
                cluster_labeling, use_chain_of_thought, train_df, 
                sample_size=20, seed=42
            )
        except Exception as e:
            debug_print(f"Error in sample_incorrect_predictions: {e}", important=True)
            print(f"Error in sample_incorrect_predictions: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            debug_print("GPU cache cleared and synchronized after training", important=True)
        
        # Save metrics
        metrics_data = {
            "hyperparams": hyperparams,
            "examples_per_class": examples_per_category,
            "cluster_labeling": cluster_labeling,
            "train_loss": train_result.training_loss,
            "eval_metrics": eval_result,
            "best_metric": best_metric,
            "end_time": datetime.now().isoformat(),
            "trial_type": "few_shot_simple"
        }
        
        # Save to file
        with open("few_shot_simple_results.pkl", "wb") as f:
            pickle.dump(metrics_data, f)
        
        # Also save as text
        with open("few_shot_simple_results.txt", "w") as f:
            f.write(f"Few-shot Learning Results\n")
            f.write(f"========================\n")
            f.write(f"Examples per class: {examples_per_category}\n")
            f.write(f"Cluster labeling: {cluster_labeling}\n")
            f.write(f"Best F1 Score: {best_metric:.4f}\n")
            f.write(f"Training loss: {train_result.training_loss:.4f}\n")
            f.write(f"Evaluation metrics:\n")
            for key, value in eval_result.items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write(f"\nHyperparameters:\n")
            for key, value in hyperparams.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"Results saved to few_shot_simple_results.pkl and few_shot_simple_results.txt")
        
        return best_metric
        
    except Exception as e:
        debug_print(f"Training failed with error: {e}", important=True)
        print(f"Training failed with error: {e}")
        return 0.0
    
    finally:
        # Clean up
        try:
            if fresh_model is not None:
                del fresh_model
        except:
            pass
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def run_few_shot_learning(cluster_labeling=False, examples_per_category=5, use_chain_of_thought=True, use_semantic_selection=False, use_clustering=True, use_augmentation=False, exception_ratio=2, no_exceptions=False):
    """Run few-shot learning with optimized hyperparameters"""
    debug_print(f"run_few_shot_learning() called with cluster_labeling={cluster_labeling}, examples_per_category={examples_per_category}", important=True)
    print("Starting Few-shot LLaMA QLoRA Learning...")
    
    try:
        # Load dataset
        debug_print("Loading dataset...", important=True)
        train_df, val_df = load_dataset(cluster_labeling)
        debug_print(f"Dataset loaded: train={len(train_df)}, val={len(val_df)}", important=True)
        
        # Create few-shot dataset
        debug_print("Creating few-shot dataset...", important=True)
        few_shot_examples_df, few_shot_val_df = create_few_shot_dataset(
            train_df, val_df, examples_per_category=examples_per_category, 
            seed=42, cluster_labeling=cluster_labeling, use_clustering=use_clustering, use_augmentation=use_augmentation, exception_ratio=exception_ratio, no_exceptions=no_exceptions
        )
        debug_print(f"Few-shot dataset created: examples={len(few_shot_examples_df)}, val={len(few_shot_val_df)}", important=True)
        
        # Create tokenizer
        debug_print("Creating tokenizer...", important=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        debug_print(f"Tokenizer created, vocab size: {len(tokenizer)}", important=True)
        
        # Ensure proper padding configuration
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        debug_print(f"Pad token: {tokenizer.pad_token}  ID: {tokenizer.pad_token_id}")
        
        # Get optimized hyperparameters
        hyperparams = get_optimized_hyperparams()
        
        # Create and tokenize datasets
        debug_print("Creating and tokenizing few-shot datasets...", important=True)
        train_dataset = Dataset.from_pandas(few_shot_examples_df)
        val_dataset = Dataset.from_pandas(few_shot_val_df)
        debug_print(f"Datasets created: train={len(train_dataset)}, val={len(val_dataset)}", important=True)
        
        # Tokenize datasets
        debug_print("Starting tokenization of training dataset...", important=True)
        # Get max_length from hyperparams
        max_length = hyperparams.get("max_length", 4096)
        
        train_dataset = train_dataset.map(
            lambda x: tokenize_function(
                x, tokenizer, max_length=max_length, cluster_labeling=cluster_labeling, 
                few_shot_examples_df=few_shot_examples_df, use_chain_of_thought=use_chain_of_thought,
                use_semantic_selection=use_semantic_selection, train_df=train_df, is_training=True
            ),
            batched=True,
            remove_columns=[c for c in train_dataset.column_names if c not in ("input_ids", "attention_mask", "labels")]
        )
        debug_print("Training dataset tokenization completed", important=True)
        
        val_dataset = val_dataset.map(
            lambda x: tokenize_function(
                x, tokenizer, max_length=max_length, cluster_labeling=cluster_labeling, 
                few_shot_examples_df=few_shot_examples_df, use_chain_of_thought=use_chain_of_thought,
                use_semantic_selection=use_semantic_selection, train_df=train_df, is_training=True
            ),
            batched=True,
            remove_columns=[c for c in val_dataset.column_names if c not in ("input_ids", "attention_mask", "labels")]
        )
        debug_print("Validation dataset tokenization completed", important=True)
        
        # Remove the original columns since we'll create new ones in tokenization
        debug_print("Removing original columns for tokenization...", important=True)
        
        # Set dataset format to PyTorch tensors
        debug_print("Setting dataset format to PyTorch tensors...", important=True)
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        debug_print("Dataset format set to PyTorch tensors", important=True)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            debug_print("GPU cache cleared", important=True)
        
        # Run the few-shot training
        debug_print("Starting few-shot training...", important=True)
        if cluster_labeling:
            print(f"Starting few-shot learning with {examples_per_category} examples per category (4 categories = {examples_per_category * 4} total)")
        else:
            print(f"Starting few-shot learning with {examples_per_category} examples per class ({examples_per_category * 2} total)")
        
        best_value = run_few_shot_training(
            tokenizer, train_dataset, val_dataset, examples_per_category, cluster_labeling,
            use_chain_of_thought=use_chain_of_thought, use_semantic_selection=use_semantic_selection, train_df=train_df, few_shot_examples_df=few_shot_examples_df
        )
        
        debug_print("Few-shot learning completed", important=True)
        print("Few-shot learning completed!")
        
        return best_value
            
    except Exception as e:
        debug_print(f"ERROR in run_few_shot_learning(): {e}", important=True)
        raise e

def main():
    """Main function for few-shot learning"""
    debug_print("main() called", important=True)
    
    try:
        args, config = parse_args()
        debug_print(f"main() - args: {args}", important=True)
        debug_print(f"main() - config: {config}", important=True)
        
        # Get examples per category from args or use enhanced defaults
        examples_per_category = getattr(args, 'examples_per_category', 5)
        exception_ratio = getattr(args, 'exception_ratio', 2)  # Default ratio of 2
        no_exceptions = getattr(args, 'no_exceptions', False)  # Default to False
        
        print("Starting Few-shot LLaMA QLoRA Learning...")
        print(f"Cluster labeling: {config['training']['cluster_labeling']}")
        print(f"Physical GPUs: {config['physical_gpus']}")
        print(f"Number of GPUs: {config['num_gpus']}")
        print(f"Examples per category: {examples_per_category}")
        if no_exceptions:
            print(f"No exceptions flag set - using 0 exception examples")
        else:
            print(f"Exception ratio: {exception_ratio} (exceptions = floor({examples_per_category}/{exception_ratio}) = {examples_per_category // exception_ratio})")
        print(f"Chain-of-thought reasoning: Enabled")
        print(f"Semantic example selection: {'Available for inference' if SENTENCE_TRANSFORMERS_AVAILABLE else 'Disabled'}")
        print(f"Clustering for diversity: Enabled")
        print(f"Data augmentation: Disabled (using 180k real examples)")
        print(f"Training mode: Fixed examples (semantic selection disabled)")
        
        # Check CUDA availability
        debug_print(f"CUDA available at startup: {torch.cuda.is_available()}", important=True)
        if torch.cuda.is_available():
            debug_print(f"Available GPUs: {torch.cuda.device_count()}", important=True)
            for i in range(torch.cuda.device_count()):
                debug_print(f"GPU {i}: {torch.cuda.get_device_name(i)}", important=True)
            debug_gpu_memory("main() startup - ", important=True)
        
        # Run the few-shot learning
        debug_print("Running few-shot learning...", important=True)
        print("Running few-shot learning...")
        
        best_value = run_few_shot_learning(
            cluster_labeling=args.cluster_labeling,
            examples_per_category=args.examples_per_category,
            use_chain_of_thought=True,  # Enable chain-of-thought reasoning
            use_semantic_selection=False,  # Disable semantic selection to avoid overfitting
            use_clustering=False,  # Use improved filtering instead of expensive clustering
            use_augmentation=False,
            exception_ratio=exception_ratio,
            no_exceptions=no_exceptions
        )
        
        debug_print("main() completed successfully", important=True)
        print("Few-shot learning completed!")
        print("Few-shot learning completed for cluster labeling!" if args.cluster_labeling else "Few-shot learning completed for binary labeling!")
        
    except Exception as e:
        debug_print(f"ERROR in main(): {e}", important=True)
        raise e

if __name__ == "__main__":
    # Redirect all output to log file
    import sys
    import os
    from datetime import datetime
    
    log_file = "few_shot_llama_qlora.log"
    
    # Create a custom file-like object that writes to both file and original stdout
    class TeeOutput:
        def __init__(self, file, original_stdout):
            self.file = file
            self.original_stdout = original_stdout
        
        def write(self, text):
            try:
                # Write to file first
                self.file.write(text)
                self.file.flush()  # Force immediate write
                # Then write to console
                self.original_stdout.write(text)
                self.original_stdout.flush()
            except Exception as e:
                # If there's an error, at least try to write to original stdout
                try:
                    self.original_stdout.write(f"Error writing to log: {e}\n")
                    self.original_stdout.write(text)
                    self.original_stdout.flush()
                except:
                    pass
        
        def flush(self):
            try:
                self.file.flush()
                self.original_stdout.flush()
            except:
                pass
    
    # Open log file and redirect stdout and stderr
    with open(log_file, 'w', encoding='utf-8') as f:  # Default buffering
        # Write header to log
        f.write(f"=== Few-shot LLaMA QLoRA Learning Log ===\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log file: {os.path.abspath(log_file)}\n")
        f.write("=" * 50 + "\n\n")
        f.flush()
        
        # Redirect stdout and stderr to the log file
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Create tee objects that write to both file and console
        tee_stdout = TeeOutput(f, original_stdout)
        tee_stderr = TeeOutput(f, original_stderr)
        
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        
        # Clear any existing logging handlers and reconfigure
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Reconfigure logging to use our redirected stdout
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),  # This will now use our TeeOutput
            ],
            force=True  # Force reconfiguration
        )
        
        # Also configure the root logger to use our stdout
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(logging.StreamHandler(sys.stdout))
        root_logger.setLevel(logging.INFO)
        
        # Test that our redirection is working
        print("=== LOGGING TEST ===")
        print("This should appear in both console and log file")
        logging.info("This is a test logging message")
        print("=== END LOGGING TEST ===")
        
        try:
            debug_print("Script started - __main__ block", important=True)
            main()
            debug_print("Script completed", important=True)
        except Exception as e:
            debug_print(f"Script failed with error: {e}", important=True)
            import traceback
            traceback.print_exc()
        finally:
            # Restore original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            # Write footer to log
            f.write(f"\n" + "=" * 50 + "\n")
            f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.flush()  # Ensure footer is written
    
    print(f"Script completed. All output saved to {log_file}")