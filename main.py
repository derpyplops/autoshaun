import os
import PyPDF2
import numpy as np
import openai
import pandas as pd
import tiktoken
from openai.embeddings_utils import distances_from_embeddings
from rich import print
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import datasets

openai.api_key = os.environ["OPENAI_API_KEY"]
HF_TOKEN = os.environ['HF_TOKEN']


def textify_pdf(filepath):
    # Convert pdf to text with pypdf2
    text = ''
    with open(filepath, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        for page in pdf.pages:
            text += page.extract_text()
    return text


# # Create a dataframe from the list of texts
# df = pd.DataFrame(texts, columns=['fname', 'text'])
#
# # Set the text column to be the raw text with the newlines removed
# df['text'] = df.fname + ". " + remove_newlines(df.text)
# df.to_csv('processed/scraped.csv')
# df.head()

# Press the green button in the gutter to run the script.

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


def extract_and_write_all_pdfs_to_txt():
    pdf_file_paths = [os.path.join('data/pdfs', f) for f in os.listdir('data/pdfs')]

    for file in pdf_file_paths:
        text = textify_pdf(file)
        # write text to same filename but .txt
        new_filename = file.replace('.pdf', '.txt').replace('pdfs', 'txts')
        # create if not exists
        os.makedirs(os.path.dirname(new_filename), exist_ok=True)
        with open(new_filename, 'w') as f:
            f.write(text)


def chunkify_and_write():
    text_file_paths = [os.path.join('data/txts', f) for f in os.listdir('data/txts')]
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
    df.to_csv('chunks.csv')


def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["chunk"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def _answer_question(
    df,
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    prompt = f"Context: {context}\n\nPlease help me with my question, but it is very important that you " \
             f"do not make answers up. If you do not know the answer, say 'I don\'t know.'\n\nQuestion:" \
             f" {question}"
    # If debug, print the raw model response
    if debug:
        print(prompt)
        print("\n\n")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant assisting a customer with their "
                                              "finances."},
                # {"role": "user", "content": "Who won the world series in 2020?"},
                # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""

def generate_embeddings(chunk):
    return openai.Embedding.create(input=chunk, engine='text-embedding-ada-002')['data'][0]['embedding']


def chunks_to_embeddings():
    # load df
    df = pd.read_csv('data/chunks.csv')
    with ThreadPoolExecutor(max_workers=100) as executor:
        df['embeddings'] = list(tqdm(executor.map(generate_embeddings, df['chunk']), total=len(df)))
    df.to_csv('embeddings.csv')


def answer_question(question):
    ds = datasets.load_dataset("derpyplops/autoshaun-embeddings", use_auth_token=HF_TOKEN)
    df = ds['train'].to_pandas()
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    ans = _answer_question(df, question=question)
    return ans


def answer_all_questions(questions):
    for question in questions:
        print(f"[blue]Q: {question}[/blue]\n[magenta]A: {answer_question(question)}[/magenta]\n\n")


def add_n_tokens():
    df = pd.read_csv('embeddings.csv', index_col=0)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df['n_tokens'] = df['chunk'].apply(lambda x: len(tokenizer.encode(x)))
    df.to_csv('embeddings.csv')


def upload_to_hf():
    csv = pd.read_csv('data/embeddings.csv', index_col=0)
    dataset = datasets.Dataset.from_pandas(csv)
    dataset.push_to_hub('autoshaun-embeddings', private=True)

if __name__ == '__main__':
    answer_question("Can I do a partial withdrawal on Great Flexi Cashback?")