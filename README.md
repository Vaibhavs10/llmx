# LLMX

A CLI tool for managing MLX-LM models and servers. LLMX provides a convenient interface for downloading, managing, and serving MLX-LM models with OpenAI-compatible API endpoints.

## Installation

```bash
pip install .
```

## Usage

LLMX provides several commands for managing MLX-LM models:

### Basic Commands

```bash
# Start a model server
llmx serve <model_id> [--port PORT]

# Start an interactive chat session
llmx chat <model_id> [--port PORT] [--temperature TEMP]

# Show model information
llmx show <model_id>

# Run a model server (alias for serve)
llmx run <model_id> [--port PORT]

# Stop a running model
llmx stop <port>

# Pull a model from Hugging Face
llmx pull <model_id>

# Push a model to Hugging Face
llmx push <model_id>

# List downloaded models
llmx list

# List running models
llmx ps

# Remove a model
llmx rm <model_id>

# Get help
llmx help
```

### Example Usage

1. Pull a model from Hugging Face:
```bash
llmx pull mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

2. Start an interactive chat session:
```bash
llmx chat mlx-community/Mistral-7B-Instruct-v0.3-4bit --temperature 0.7
```

3. Start the model server (for API access):
```bash
llmx serve mlx-community/Mistral-7B-Instruct-v0.3-4bit --port 8080
```

4. Make a request to the server:
```bash
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

5. List running models:
```bash
llmx ps
```

6. Stop the server:
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

Example:
```bash
llmx chat mistralai/Mistral-7B-Instruct-v0.2 --temperature 0.7
```

## Model Storage

Models are stored in `~/.llmx/models/` by default. Each model is stored in its own directory named after the model ID.

## Requirements

- Python 3.8 or higher
- MLX-LM
- Click
- Rich
- Requests
- Hugging Face Hub 