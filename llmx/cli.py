"""
LLMX CLI implementation
"""
import os
import json
import argparse
from pathlib import Path
import subprocess
import signal
import requests
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

console = Console()
LLMX_HOME = os.path.expanduser("~/.llmx/running.json")
os.makedirs(os.path.dirname(LLMX_HOME), exist_ok=True)

def load_running_models():
    """Load and clean up running models state"""
    if not os.path.exists(LLMX_HOME):
        return {}
        
    try:
        with open(LLMX_HOME, 'r') as f:
            running_models = json.load(f)
        
        # Clean up stale entries
        cleaned_models = {}
        for port, info in running_models.items():
            try:
                # Check if process is still running
                os.kill(info["pid"], 0)
                cleaned_models[port] = info
            except ProcessLookupError:
                continue
        
        # Save cleaned state
        if cleaned_models != running_models:
            save_running_models(cleaned_models)
            
        return cleaned_models
    except (json.JSONDecodeError, IOError):
        # If file is corrupted or can't be read, start fresh
        return {}

def save_running_models(running_models):
    with open(LLMX_HOME, 'w') as f:
        json.dump(running_models, f)

def start_server(model_id, port=8080):
    """Start a model server and return process info"""
    cmd = f"mlx_lm.server --model {model_id} --port {port}"
    process = subprocess.Popen(cmd.split(), start_new_session=True)
    return {
        "model_id": model_id,
        "pid": process.pid,
        "port": port
    }

def stop_server(port, running_models):
    """Stop a server and update running models"""
    try:
        os.killpg(os.getpgid(running_models[port]["pid"]), signal.SIGTERM)
        del running_models[port]
        save_running_models(running_models)
        return True
    except (KeyError, ProcessLookupError):
        return False

def ensure_model(model_id):
    """Ensure model is downloaded, return model path"""
    try:
        return snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.safetensors", "*.bin"]
        )
    except Exception as e:
        console.print(f"[red]Error pulling model: {str(e)}[/red]")
        return None

def chat_session(port, temperature=0.7):
    """Run an interactive chat session"""
    messages = []
    try:
        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
                if user_input.lower() == 'exit':
                    break
                    
                messages.append({"role": "user", "content": user_input})
                response = requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={"messages": messages, "temperature": temperature},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    assistant_message = response.json()["choices"][0]["message"]["content"]
                    messages.append({"role": "assistant", "content": assistant_message})
                    console.print(f"\n[bold purple]Assistant[/bold purple]")
                    console.print(Markdown(assistant_message))
                    console.print()
                else:
                    console.print(f"[red]Error: Server returned status code {response.status_code}[/red]")
                    
            except (KeyboardInterrupt, requests.exceptions.RequestException) as e:
                break
    finally:
        return Confirm.ask("\nKeep server running?", default=False)

def show_help():
    help_text = """
[bold]LLMX Commands:[/bold]

  [green]serve[/green] <model_id> [--port PORT]      Start a model server
  [green]chat[/green] <model_id> [--port PORT]       Start an interactive chat session
  [green]stop[/green] <port>                         Stop a running model
  [green]pull[/green] <model_id>                     Pull a model from Hugging Face
  [green]list[/green]                                List downloaded models
  [green]ps[/green]                                  List running models
  [green]help[/green]                                Show this help message

[bold]Example:[/bold] llmx chat mlx-community/Mistral-7B-Instruct-v0.3-4bit --temperature 0.7
"""
    console.print(Panel(help_text, title="LLMX Help", border_style="blue"))

def main():
    parser = argparse.ArgumentParser(description="LLMX - MLX-LM model management tool", add_help=False)
    subparsers = parser.add_subparsers(dest='command')

    # Add command parsers
    serve_parser = subparsers.add_parser('serve', add_help=False)
    serve_parser.add_argument('model_id')
    serve_parser.add_argument('--port', type=int, default=8080)

    chat_parser = subparsers.add_parser('chat', add_help=False)
    chat_parser.add_argument('model_id')
    chat_parser.add_argument('--port', type=int, default=8080)
    chat_parser.add_argument('--temperature', type=float, default=0.7)

    stop_parser = subparsers.add_parser('stop', add_help=False)
    stop_parser.add_argument('port')

    pull_parser = subparsers.add_parser('pull', add_help=False)
    pull_parser.add_argument('model_id')

    for cmd in ['list', 'ps', 'help']:
        subparsers.add_parser(cmd, add_help=False)

    args = parser.parse_args()
    running_models = load_running_models()

    if args.command in ['serve', 'chat']:
        port = str(getattr(args, 'port', 8080))
        if port in running_models:
            console.print(f"[red]Port {port} is already in use[/red]")
            return

        if not ensure_model(args.model_id):
            return

        server_info = start_server(args.model_id, int(port))
        running_models[port] = server_info
        save_running_models(running_models)
        console.print(f"[green]Started model server for {args.model_id} on port {port}[/green]")

        if args.command == 'chat':
            import time
            time.sleep(2)  # Give server time to start
            keep_running = chat_session(int(port), getattr(args, 'temperature', 0.7))
            if not keep_running:
                stop_server(port, running_models)
                console.print(f"[green]Stopped server on port {port}[/green]")

    elif args.command == 'stop':
        port = str(args.port)
        if port not in running_models:
            console.print(f"[red]No server running on port {port}[/red]")
            return
        if stop_server(port, running_models):
            console.print(f"[green]Stopped server on port {port}[/green]")

    elif args.command == 'pull':
        ensure_model(args.model_id)

    elif args.command == 'list':
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model ID", style="green")
        table.add_column("Size", justify="right")
        
        for model_dir in Path(HUGGINGFACE_HUB_CACHE).glob("models--*"):
            try:
                parts = model_dir.name.split('--')
                model_id = f"{parts[1]}/{parts[2]}" if len(parts) == 3 else parts[1]
                size = sum(f.stat().st_size for f in model_dir.glob('**/*') if f.is_file())
                table.add_row(model_id, f"{size / 1024 / 1024:.1f} MB")
            except Exception:
                continue
        console.print(table)

    elif args.command == 'ps':
        if not running_models:
            console.print("[yellow]No running models[/yellow]")
            return
            
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Port")
        table.add_column("Model ID")
        table.add_column("Status")

        for port, info in running_models.items():
            try:
                os.kill(info["pid"], 0)
                status = "[green]Running[/green]"
            except ProcessLookupError:
                status = "[red]Stopped[/red]"
            table.add_row(str(port), info["model_id"], status)
        console.print(table)

    else:
        show_help()

if __name__ == '__main__':
    main() 