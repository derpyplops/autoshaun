# Embedding Generation Script

This script generates embeddings from text files, saves them as a CSV file, and optionally uploads them to a Hugging Face repository.

## Prerequisites

- Python 3.6 or higher
- OpenAI API key (set in your environment as OPENAI_API_KEY)

> [!NOTE]
> Take note that use of the OpenAI API incurs costs. For more information, see the 
> [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings/embedding-models).

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/embedding-generation.git

# Navigate to the project directory
cd embedding-generation

# Install the required dependencies
pip install -r requirements.txt

# Generate embeddings
python generate_embeddings.py [OPTIONS]

Options
--in-dir PATH: Directory containing input text files (default: data/txts in the parent directory)
--out-dir PATH: Directory to save the output files (default: data/embeddings in the parent directory)
--repo-id TEXT: Hugging Face repository ID for uploading embeddings (default: 'autoshaun-embeddings')
--help: Show the help message and exit
```

### Example
```
python generate_embeddings.py --in-dir /path/to/input/txts --out-dir /path/to/output/embeddings --repo-id my-embeddings-repo
```

This command will:

- Read text files from the specified input directory (/path/to/input/txts).
- Generate embeddings for each chunk of text.
- Save the embeddings and metadata as a CSV file in the specified output directory (/path/to/output/embeddings).
- Upload the embeddings to the specified Hugging Face repository (my-embeddings-repo).

### Output
The script generates the following output:

- embeddings.csv: A CSV file containing the embeddings and metadata for each chunk of text.
The embeddings are also uploaded to the specified Hugging Face repository if a repository ID is provided.