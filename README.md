# AutoShaun - Financial Assistant

AutoShaun is an intelligent financial assistant that provides answers to financial questions using advanced AI technology. The application features a modern web interface built with Streamlit and leverages OpenAI's GPT models for generating accurate financial responses.

## Features

- 🤖 AI-powered financial assistance
- 💬 Interactive chat interface
- 🔒 Secure login system
- 📱 Modern, responsive UI
- ⚡ Fast response times
- 📚 Context-aware answers based on financial documentation

## Prerequisites

- Python 3.11
- Poetry (Python package manager)
- OpenAI API key
- Hugging Face token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autoshaun.git
cd autoshaun
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export HF_TOKEN="your-huggingface-token"
```

## Usage

1. Activate the Poetry virtual environment:
```bash
poetry shell
```

2. Start the Streamlit server:
```bash
streamlit run server.py
```

3. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

4. Log in using the provided credentials

5. Start chatting with AutoShaun about your financial questions!

## Project Structure

- `server.py` - Main Streamlit application and chat interface
- `answer_question.py` - Core logic for processing questions and generating answers
- `definitions.py` - Project constants and configurations
- `assets/` - Static assets including avatars and images
- `scripts/` - Utility scripts and tools

## Dependencies

The project uses several key dependencies:
- Streamlit for the web interface
- OpenAI API for AI-powered responses
- PyPDF2 for PDF processing
- Hugging Face datasets for embeddings
- Various data processing libraries (pandas, numpy, etc.)

For a complete list of dependencies, see `pyproject.toml`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

- Jonathan N. (jonathan.n@u.nus.edu)
