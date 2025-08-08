import pandas as pd
import os
import time
from datetime import datetime
import json
from openai import AzureOpenAI

# Load environment variables and initialize client
endpoint = os.getenv("ENDPOINT_URL", "https://your-azure-openai-endpoint.com")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "put api key here")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

# Prompt for transforming Reddit posts to Instagram DMs
cot_user_prompt ='''
Instagram DMs are a great way to interact with people. They tend to be shorter with an average of 150 -200 characters and are casual in nature while still adopting socially relevant slang and terminology along with emojis. Moreover, younger generations are the primary users of these platforms. They like to interact with each other in private DMs and discuss topics that they find relevant and relatable. Reddit, on the other hand, is more descriptive and open in nature, often taking a more personal air. Along with that, reddit posts vary greatly in length.

I would like you to adopt the persona of a youth between ages 15-21 using Instagram DMs and translate some Reddit posts to simulate realistic Instagram DMs.

Here is an example of what I would like done to transform the following Reddit post to mimic an Instagram DM:

Title: 'Is college really worth it anymore?'
Body: "I've been thinking a lot about whether college is actually worth the cost. With student loans piling up and so many people finding jobs through networking or alternative paths like coding bootcamps, I’m starting to question if getting a degree is the best option. Would love to hear from people who either went to college or skipped it—what’s your experience been like?"

1. Identifying the Core Idea, Tone, and Sentiment of the Reddit Post
- Main Idea: The author is questioning whether college is worth the cost.
- Tone: Thoughtful, skeptical, but open to discussion.
- Sentiment: Mixed—concern about debt and job struggles but not outright negative.
- Structure: Formal, structured, and designed to encourage discussion.

2. Adjusting for Instagram DM Style
Instagram DMs—especially among youths (15-21)—tend to be:
- Conversational rather than formal
- Expressive (using emojis, exaggeration, slang) in relation to the gravity of the content, so use less slang or emojis when the sentiment is more negative
- Condensed (shorter, more direct)
- Relatable/personal (sharing a real or exaggerated experience)

3. Transforming the Content
- Simplify the Message
- Adjust for Casual & Expressive Tone
- Maintain the Original Sentiment

4. Final Output
- Instagram DM: "Lowkey starting to think college might not even be worth it 💀 like ppl graduate w degrees & still struggle to find jobs… idk what to do fr"

We get the final Reddit, Instagram pairing:
"Reddit: \Title: 'Is college really worth it anymore?', Body: "I've been thinking a lot about whether college is actually worth the cost. With student loans piling up and so many people finding jobs through networking or alternative paths like coding bootcamps, I’m starting to question if getting a degree is the best option. Would love to hear from people who either went to college or skipped it—what’s your experience been like?\",
Instagram: \"Bro I literally cannot text first. Like I will sit here and overthink every word for 30 min then just not send anything 😭\""


The input will be in the following format.

Reddit Post Title: {title}
Reddit Post Content: {content}

Please transform the input Reddit post to mimic an Instagram DM and analyze the sentiment of the DM you generated. Respond in a strictly JSON format with the following keys:

DM: The generated Instagram DM.
Sentiment: The perceived sentiment of the generated DM.
'''
# Helper function to call ChatGPT with basic error handling/backoff
def prompt_chatgpt(user_prompt, system_prompt="You are a helpful assistant.", model="gpt-4o", temperature=1.0, top_p=1.0, max_retries=5):
    backoff = 1
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=top_p
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}. Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2
    return "Error: max retries reached."

def clean_gpt_response(res):
    res = res.strip("```json").strip("```").strip()
    return json.loads(res)

def prepare_reddit_csv(input_csv):
    df = pd.read_csv(input_csv)
    df.dropna(subset=['Title', 'Content'], inplace=True)
    df['post'] = df['Title'].fillna('') + "\n" + df['Content'].fillna('')
    if 'retrieved_on' in df.columns:
        df = df.drop_duplicates(subset='retrieved_on')
    return df


def format_to_reddit_submission(dataset):
    return [
        {
            'title': row.get('Title', 'No title'),
            'content': row.get('Content', 'No content')
        }
        for _, row in dataset.iterrows()
    ]


def get_dms(dataset, checkpoint_file="dm_checkpoint.csv", checkpoint_interval=25,
            time_checkpoint_interval=3600, temperature=1.0, top_p=1.0):
    data = []
    count = 0
    start_time = time.time()
    last_time_checkpoint = start_time

    # If checkpoint exists, load it
    if os.path.exists(checkpoint_file):
        df_checkpoint = pd.read_csv(checkpoint_file)
        completed_indices = set(df_checkpoint["Index"])
        data = df_checkpoint.to_dict(orient="records")
        print(f"[{datetime.now()}]: Resuming from checkpoint. {len(completed_indices)} posts loaded.")
    else:
        completed_indices = set()

    for idx, reddit_post in enumerate(dataset):
        if idx in completed_indices:
            continue

        count += 1
        row = {"Index": idx, "Reddit Post": f"Title: {reddit_post['title']}\nContent: {reddit_post['content']}"}

        # Single call with the user-specified temperature and top_p
        gpt_response = prompt_chatgpt(
            cot_user_prompt.format(title=reddit_post['title'], content=reddit_post['content']),
            temperature=temperature,
            top_p=top_p
        )
        try:
            clean_response = clean_gpt_response(gpt_response)
            row["DM"] = clean_response.get("DM", "")
            row["Sentiment"] = clean_response.get("Sentiment", "")
        except Exception as e:
            row["DM"] = f"Error: {str(e)}"
            row["Sentiment"] = "Unknown"

        data.append(row)

        # Optional brief sleep to avoid rate-limit issues
        time.sleep(0.2)

        # Log progress every 10
        if count % 10 == 0:
            print(f"[{datetime.now()}]: Processed {count} posts...")

        # Save based on count
        if count % checkpoint_interval == 0:
            pd.DataFrame(data).to_csv(checkpoint_file, index=False, encoding="UTF-8-SIG")
            print(f"[{datetime.now()}]: Checkpoint saved after {count} posts.")

        # Save based on time
        current_time = time.time()
        if current_time - last_time_checkpoint >= time_checkpoint_interval:
            pd.DataFrame(data).to_csv(checkpoint_file, index=False, encoding="UTF-8-SIG")
            last_time_checkpoint = current_time
            print(f"[{datetime.now()}]: Hourly checkpoint saved after {count} posts.")

    return data

def main(input_csv, output_csv, checkpoint_file, sample_size=None, temperature=1.0, top_p=1.0):
    """
    Main entry point:
    - input_csv: your input file
    - output_csv: base name for the final result
    - checkpoint_file: base name for the checkpoint file
    - sample_size: optional integer to limit how many rows to process
    - temperature, top_p: the single combination for the DM generation
    """
    # Append temp/top_p to file names
    # Example: "my_checkpoint_file_0.7_1.csv", "my_output_file_0.7_1.csv"
    checkpoint_with_params = f"{checkpoint_file.rsplit('.csv', 1)[0]}_{temperature}_{top_p}.csv"
    output_with_params = f"{output_csv.rsplit('.csv', 1)[0]}_{temperature}_{top_p}.csv"

    print("Preparing dataset...")
    df_cleaned = prepare_reddit_csv(input_csv)
    print(f"Cleaned dataset has {len(df_cleaned)} posts.")

    if sample_size:
        df_cleaned = df_cleaned.head(sample_size)
        print(f"Sampled {sample_size} posts for processing.")

    formatted_dataset = format_to_reddit_submission(df_cleaned)

    print(f"Generating Instagram DMs from Reddit posts (temp={temperature}, top_p={top_p})...")
    dms = get_dms(
        formatted_dataset,
        checkpoint_file=checkpoint_with_params,
        temperature=temperature,
        top_p=top_p
    )

    print("Saving results...")
    pd.DataFrame(dms).to_csv(output_with_params, index=False, encoding="UTF-8-SIG")
    print(f"Done! Final DM file saved as: {output_with_params}")
    print(f"Total posts processed: {len(dms)}")

# Example usage (if you run this file directly):
if __name__ == "__main__":
    input_csv_path = "lonely_dedupe.csv"
    output_csv_path = "lonely_dms.csv"
    checkpoint_csv_path = "lonely_ckpt.csv"

    main(
        input_csv_path,
        output_csv_path,
        checkpoint_csv_path,
        sample_size=10_000,  # or None for all
        temperature=0.5,
        top_p=0.7
    )
