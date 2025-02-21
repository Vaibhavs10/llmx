# llmx

A CLI tool for managing MLX-LM models and servers. LLMX provides a convenient interface for downloading, managing, and serving MLX-LM models with OpenAI-compatible API endpoints.

## Installation

### Using uv (Recommended)

First, install `uv` if you haven't already:
```bash
pip install uv
```

Then install llmx using uv:
```bash
# Create and activate a new virtual environment
uv venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt

# Install the package in editable mode
uv pip install -e .
```

### Using pip (Alternative)

```bash
pip install .
```

## Usage

LLMX provides several commands for managing MLX-LM models:

### Available Commands

```bash
# Start a model server
llmx serve <model_id> [--port PORT]

# Start an interactive chat session
llmx chat <model_id> [--port PORT] [--temperature TEMP]

# Stop a running model
llmx stop <port>

# Pull a model from Hugging Face
llmx pull <model_id>

# List downloaded models
llmx list

# List running models
llmx ps

# Get help
llmx help
```

### Example Usage

1. Start an interactive chat session:
```bash
llmx chat mlx-community/Mistral-7B-Instruct-v0.3-4bit --temperature 0.7
```

2. Start a model server (for API access):
```bash
llmx serve mlx-community/Mistral-7B-Instruct-v0.3-4bit --port 8080
```

3. Make a request to the server:
```bash
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

4. List running models:
```bash
llmx ps
```

5. Stop the server:
```bash
llmx stop 8080
```

## Interactive Chat Mode

The `chat` command provides an interactive chat interface where you can have a conversation with the model. Features include:

- Automatic server management (starts/stops as needed)
- Markdown rendering of responses
- Conversation history tracking
- Type 'exit' or press Ctrl+C to end the chat
- Option to keep the server running after chat ends

## Model Storage

Models are stored in the Hugging Face Hub's default cache location (`~/.cache/huggingface/hub` on Unix systems). This allows for better integration with other tools and avoids duplicating storage. The `llmx` tool manages only the running state of models in `~/.llmx/running.json`.

## Requirements

- Python 3.8 or higher
- MLX-LM
- Rich
- Requests
- Hugging Face Hub 