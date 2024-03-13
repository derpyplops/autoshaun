import os
import PyPDF2
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings
from rich import print
import datasets
from time import time

openai.api_key = os.environ["OPENAI_API_KEY"]
HF_TOKEN = os.environ['HF_TOKEN']


def time_fn(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


@time_fn
def load_embeddings():
    ds = datasets.load_dataset("derpyplops/autoshaun-embeddings", use_auth_token=HF_TOKEN)
    df = ds['train'].to_pandas()
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    return df


def textify_pdf(filepath):
    # Convert pdf to text with pypdf2
    text = ''
    with open(filepath, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        for page in pdf.pages:
            text += page.extract_text()
    return text


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
    if debug:
        print(prompt)
        print("\n\n")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant assisting a customer with their "
                                              "finances."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""


def answer_all_questions(questions):
    for question in questions:
        print(f"[blue]Q: {question}[/blue]\n[magenta]A: {answer_question(question)}[/magenta]\n\n")


embeddings_df = load_embeddings()


@time_fn
def answer_question(question):
    ans = _answer_question(embeddings_df, question=question)
    return ans


if __name__ == '__main__':
    qn_text = "Can I do a partial withdrawal on Great Flexi Cashback?"
    answer_question(qn_text)
