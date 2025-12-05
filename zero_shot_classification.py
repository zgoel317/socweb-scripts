import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import sys
from datetime import datetime

lonely_dms = "/home/zgoel3/synthetic_dms/llama/training_data/lonely_dms_0.5_0.7_noemojis.csv"
serious_dms = "/home/zgoel3/synthetic_dms/llama/training_data/serious_dms_0.7_0.7_noemojis.csv"
casual_dms = "/home/zgoel3/synthetic_dms/llama/training_data/casual_dms_0.7_0.7_noemojis.csv"
sampled_messages = "/home/zgoel3/synthetic_dms/llama/sampled_messages.csv"

MODEL_PATH = "/home/shared_models/base_models/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"

log_filename = f"zero_shot_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_filename, 'w', buffering=1)

class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = TeeOutput(sys.stdout, log_file)
sys.stderr = TeeOutput(sys.stderr, log_file)

print(f"Log file: {log_filename}")
print(f"Started at: {datetime.now()}")

sampled_df = pd.read_csv(sampled_messages)

# Allow running the script on only the sampled messages by passing the flag `--sampled_only`.
only_sampled = "--sampled_only" in sys.argv

lonely_df = serious_df = casual_df = None
if not only_sampled:
    lonely_df = pd.read_csv(lonely_dms)
    serious_df = pd.read_csv(serious_dms)
    casual_df = pd.read_csv(casual_dms)

# Ensure DM column is string and fill NaNs to avoid tokenizer errors
for _df in (sampled_df, lonely_df, serious_df, casual_df):
    if _df is None:
        continue
    if 'DM' in _df.columns:
        _df['DM'] = _df['DM'].astype(str).fillna('')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

PROMPT= """You are a binary classifier for loneliness detection.                                     
Return **exactly TWO** lines:
  1. A concise explanation (≤15 words) of your decision. **Do NOT include any numbers.**
  2. The label only, either 1 (lonely) or 0 (not lonely).
The second line must contain **only** that single digit and NOTHING else.

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
"""

def classify_message(msg: str):
    """Return (label, explanation) where label is 1/0 or None on failure."""
    # Build chat-style prompt
    chat = [
        {"role": "system", "content": PROMPT.strip()},
        {"role": "user", "content": f"Message to classify:\n{msg}\n\nReturn exactly two lines:\n1. Explanation (no numbers).\n2. Single digit label (1 or 0)."}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
    except Exception as e:
        print(f"Generation error: {e}")
        return None, None

    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    lines = [l.strip() for l in generated.splitlines() if l.strip()]
    if not lines:
        print(f"Could not parse label/explanation from: {generated}")
        return None, None

    label_line = lines[-1]
    explanation = " ".join(lines[:-1]).strip()

    if label_line in {"0", "1"}:
        label = int(label_line)
        return label, explanation

    # fallback: maybe label appeared elsewhere
    m = re.search(r"\b([01])\b", generated)
    if m:
        label = int(m.group(1))
        return label, explanation or generated.replace(m.group(0), '').strip()

    print(f"Could not parse label from: {generated}")
    return None, None

def process_dataset(df, dataset_name, output_path):
    print(f"Processing {dataset_name} dataset...")
    predictions, explanations = [], []
    
    for i, message in enumerate(df['DM']):
        label, explanation = classify_message(message)
        predictions.append(label)
        explanations.append(explanation)
        
        if (i + 1) % 100 == 0:
            print(f"\n=== Progress Update: {i + 1}/{len(df)} messages processed ===")
            
            # Summary stats for all processed so far
            total = len(predictions)
            lonely_count = sum(1 for p in predictions if p == 1)
            not_lonely_count = sum(1 for p in predictions if p == 0)
            none_count = sum(1 for p in predictions if p is None)
            
            print(f"Summary stats (all processed so far):")
            print(f"  Total messages: {total}")
            print(f"  Lonely (1): {lonely_count} ({lonely_count/total*100:.1f}%)")
            print(f"  Not lonely (0): {not_lonely_count} ({not_lonely_count/total*100:.1f}%)")
            print(f"  Failed/None: {none_count} ({none_count/total*100:.1f}%)")
            
            # Get just the last batch (last 100 messages)
            batch_start = max(0, i + 1 - 100)
            batch_df = df.iloc[batch_start:i + 1].copy()
            batch_predictions = predictions[batch_start:i + 1]
            batch_df['zero_shot_label'] = batch_predictions
            batch_df['zero_shot_explanation'] = explanations[batch_start:i + 1]
            
            # Examples from this batch only
            label_0_batch = batch_df[batch_df['zero_shot_label'] == 0]
            if len(label_0_batch) > 0:
                print(f"\nExamples from this batch labeled as NOT LONELY (0):")
                for idx, (_, row) in enumerate(label_0_batch.head(3).iterrows(), 1):
                    msg = row['DM']
                    print(f"  {idx}. {msg[:100]}{'...' if len(msg) > 100 else ''}")
            
            label_1_batch = batch_df[batch_df['zero_shot_label'] == 1]
            if len(label_1_batch) > 0:
                print(f"\nExamples from this batch labeled as LONELY (1):")
                for idx, (_, row) in enumerate(label_1_batch.head(3).iterrows(), 1):
                    msg = row['DM']
                    print(f"  {idx}. {msg[:100]}{'...' if len(msg) > 100 else ''}")
            
            # Save checkpoint after every 100 messages
            temp_df = df.iloc[:i + 1].copy()
            temp_df['zero_shot_label'] = predictions
            temp_df['zero_shot_explanation'] = explanations
            temp_df.to_csv(output_path, index=False)
            print(f"Checkpoint saved to: {output_path}")
            print("=" * 60)
    
    df['zero_shot_label'] = predictions
    df['zero_shot_explanation'] = explanations
    return df

sampled_output = sampled_messages.replace('.csv', '_zero_shot.csv')
sampled_df_labeled = process_dataset(sampled_df, "sampled_messages", sampled_output)
sampled_df_labeled.to_csv(sampled_output, index=False)
print(f"Final save - sampled messages to: {sampled_output}")

if lonely_df is not None:
    lonely_output = lonely_dms.replace('.csv', '_zero_shot_clusters.csv')
    lonely_df_labeled = process_dataset(lonely_df, "lonely", lonely_output)
    lonely_df_labeled.to_csv(lonely_output, index=False)
    print(f"Final save - lonely dataset to: {lonely_output}")

if serious_df is not None:
    serious_output = serious_dms.replace('.csv', '_zero_shot_clusters.csv')
    serious_df_labeled = process_dataset(serious_df, "serious", serious_output)
    serious_df_labeled.to_csv(serious_output, index=False)
    print(f"Final save - serious dataset to: {serious_output}")

if casual_df is not None:
    casual_output = casual_dms.replace('.csv', '_zero_shot_clusters.csv')
    casual_df_labeled = process_dataset(casual_df, "casual", casual_output)
    casual_df_labeled.to_csv(casual_output, index=False)
    print(f"Final save - casual dataset to: {casual_output}")

print(f"\nCompleted at: {datetime.now()}")
print("Zero-shot classification completed and saved!")

log_file.close()