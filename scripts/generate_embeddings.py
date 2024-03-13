import os
import openai
import tiktoken
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import datasets
from pathlib import Path
import typer
from typing import Optional

max_tokens = 500

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")


# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens=max_tokens):
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

def extract_and_get_token_counts(txt_file_paths):
    token_counts = []

    for path in txt_file_paths:
        text = open(path, 'r').read()
        token_count = len(text.split())  # Count tokens by splitting on whitespaces
        token_counts.append(token_count)

    return token_counts

def chunkify_and_write(in_dir):
    text_file_paths = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]
    # print(extract_and_get_token_counts(text_file_paths))
    texts = [open(path, 'r').read() for path in text_file_paths]
    data = []
    chunk_id = 0
    for i, text in enumerate(texts):
        chunks = split_into_many(text)
        for j, chunk in enumerate(chunks):
            data.append(
                {
                    "chunk_id": chunk_id,
                    "textfilepath": text_file_paths[i],
                    "chunk_position": j,
                    "chunk": chunk,
                }
            )
            chunk_id += 1

    df = pd.DataFrame(data)
    return df

def generate_embeddings(chunk):
    return openai.Embedding.create(input=chunk, engine='text-embedding-3-small')['data'][0]['embedding']


def add_embeddings(df):
    with ThreadPoolExecutor(max_workers=100) as executor:
        df['embeddings'] = list(tqdm(executor.map(generate_embeddings, df['chunk']), total=len(df)))
    return df


def add_n_tokens(df):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df['n_tokens'] = df['chunk'].apply(lambda x: len(tokenizer.encode(x)))
    return df


def upload_to_hf(df, repo_id):
    dataset = datasets.Dataset.from_pandas(df)
    dataset.push_to_hub(repo_id, private=True)


def main(
    in_dir: Path = typer.Option(Path.cwd().parent / 'data/txts', help="Directory containing input text files"),
    out_dir: Path = typer.Option(Path.cwd().parent / 'data/embeddings', help="Directory to save the output files"),
    repo_id: Optional[str] = typer.Option('autoshaun-embeddings', help="Hugging Face repository ID for uploading embeddings")
):
    chunks_df = chunkify_and_write(in_dir)
    chunks_df = add_embeddings(chunks_df)
    chunks_df = add_n_tokens(chunks_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(out_dir / 'embeddings.csv', index=False)

    if repo_id:
        upload_to_hf(chunks_df, repo_id)
    print('Done')


if __name__ == "__main__":
    typer.run(main)
