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

2. Start the model server:
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

## Model Storage

Models are stored in `~/.llmx/models/` by default. Each model is stored in its own directory named after the model ID.

## Requirements

- Python 3.8 or higher
- MLX-LM
- Click
- Rich
- Requests
- Hugging Face Hub 