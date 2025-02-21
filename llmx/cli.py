"""
LLMX CLI implementation
"""
import os
import json
import argparse
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.columns import Columns
from pathlib import Path
import subprocess
import signal
import requests
from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

console = Console()

# Constants
LLMX_HOME = os.path.expanduser("~/.llmx")
RUNNING_FILE = os.path.join(LLMX_HOME, "running.json")

# Ensure directories exist
os.makedirs(LLMX_HOME, exist_ok=True)

def load_running_models():
    if os.path.exists(RUNNING_FILE):
        with open(RUNNING_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_running_models(running_models):
    with open(RUNNING_FILE, 'w') as f:
        json.dump(running_models, f)

def get_model_path(model_id):
    """Get the path to a model in the Hugging Face cache"""
    # First try to find it in the cache
    cached_files = try_to_load_from_cache(model_id)
    if cached_files:
        # Return the directory containing the model
        return str(Path(cached_files).parent)
    return None

def show_help():
    """Show help information"""
    help_text = """
[bold]LLMX Commands:[/bold]

  [green]serve[/green] <model_id> [--port PORT]      Start a model server
  [green]chat[/green] <model_id> [--port PORT]       Start an interactive chat session
  [green]show[/green] <model_id>                     Show model information
  [green]stop[/green] <port>                         Stop a running model
  [green]pull[/green] <model_id>                     Pull a model from Hugging Face
  [green]push[/green] <model_id>                     Push a model to Hugging Face
  [green]list[/green]                                List downloaded models
  [green]ps[/green]                                  List running models
  [green]rm[/green] <model_id>                       Remove a model
  [green]help[/green]                                Show this help message

[bold]Examples:[/bold]

  llmx pull mlx-community/Mistral-7B-Instruct-v0.3-4bit
  llmx chat mlx-community/Mistral-7B-Instruct-v0.3-4bit --temperature 0.7
  llmx serve mlx-community/Mistral-7B-Instruct-v0.3-4bit --port 8080
"""
    console.print(Panel(help_text, title="LLMX Help", border_style="blue"))

def serve_model(args):
    """Start a model server"""
    running_models = load_running_models()
    port = getattr(args, 'port', 8080)
    
    if str(port) in running_models:
        console.print(f"[red]Port {port} is already in use by another model server[/red]")
        return

    model_path = get_model_path(args.model_id)
    if not model_path:
        console.print(f"[yellow]Model {args.model_id} not found locally. Attempting to pull...[/yellow]")
        try:
            model_path = snapshot_download(
                repo_id=args.model_id,
                ignore_patterns=["*.safetensors", "*.bin"]
            )
        except Exception as e:
            console.print(f"[red]Error pulling model: {str(e)}[/red]")
            return

    cmd = f"mlx_lm.server --model {args.model_id} --port {port}"
    process = subprocess.Popen(cmd.split(), start_new_session=True)
    
    running_models[str(port)] = {
        "model_id": args.model_id,
        "pid": process.pid,
        "port": port
    }
    save_running_models(running_models)
    
    console.print(f"[green]Started model server for {args.model_id} on port {port}[/green]")

def chat_model(args):
    """Start an interactive chat session"""
    running_models = load_running_models()
    port = getattr(args, 'port', 8080)
    temperature = getattr(args, 'temperature', 0.7)
    server_started = False
    
    if str(port) not in running_models:
        model_path = get_model_path(args.model_id)
        if not model_path:
            console.print(f"[yellow]Model {args.model_id} not found locally. Attempting to pull...[/yellow]")
            try:
                model_path = snapshot_download(
                    repo_id=args.model_id,
                    ignore_patterns=["*.safetensors", "*.bin"]
                )
            except Exception as e:
                console.print(f"[red]Error pulling model: {str(e)}[/red]")
                return

        cmd = f"mlx_lm.server --model {args.model_id} --port {port}"
        process = subprocess.Popen(cmd.split(), start_new_session=True)
        
        running_models[str(port)] = {
            "model_id": args.model_id,
            "pid": process.pid,
            "port": port
        }
        save_running_models(running_models)
        server_started = True
        console.print(f"[green]Started model server for {args.model_id} on port {port}[/green]")
        # Give the server a moment to start
        import time
        time.sleep(2)
    
    console.print("\n[bold blue]Starting chat session. Type 'exit' or press Ctrl+C to end the chat.[/bold blue]\n")
    
    messages = []
    try:
        while True:
            user_input = Prompt.ask("[bold green]You[/bold green]")
            
            if user_input.lower() == 'exit':
                break
                
            messages.append({"role": "user", "content": user_input})
            
            try:
                response = requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={
                        "messages": messages,
                        "temperature": temperature
                    },
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
                    
            except requests.exceptions.RequestException as e:
                console.print(f"[red]Error communicating with the server: {str(e)}[/red]")
                break
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Chat session ended.[/yellow]")
    
    if server_started:
        keep_running = Confirm.ask(
            "\nDo you want to keep the server running?",
            default=False
        )
        
        if not keep_running:
            try:
                os.killpg(os.getpgid(running_models[str(port)]["pid"]), signal.SIGTERM)
                del running_models[str(port)]
                save_running_models(running_models)
                console.print("[green]Stopped model server.[/green]")
            except (KeyError, ProcessLookupError):
                pass

def show_model(args):
    """Show model information"""
    model_path = get_model_path(args.model_id)
    if not model_path:
        console.print(f"[red]Model {args.model_id} not found locally[/red]")
        return

    # Show model info
    console.print(Panel.fit(
        f"[bold]Model ID:[/bold] {args.model_id}\n"
        f"[bold]Cache Path:[/bold] {model_path}",
        title="Model Information",
        border_style="green"
    ))

def stop_model(args):
    """Stop a running model"""
    running_models = load_running_models()
    port = str(args.port)
    
    if port not in running_models:
        console.print(f"[red]No model server running on port {port}[/red]")
        return

    try:
        os.killpg(os.getpgid(running_models[port]["pid"]), signal.SIGTERM)
        del running_models[port]
        save_running_models(running_models)
        console.print(f"[green]Stopped model server on port {port}[/green]")
    except ProcessLookupError:
        console.print(f"[yellow]Process already stopped[/yellow]")
        del running_models[port]
        save_running_models(running_models)

def pull_model(args):
    """Pull a model from Hugging Face"""
    if get_model_path(args.model_id):
        console.print(f"[yellow]Model {args.model_id} already exists in cache[/yellow]")
        return

    try:
        with console.status(f"[bold blue]Pulling model {args.model_id}...[/bold blue]"):
            model_path = snapshot_download(
                repo_id=args.model_id,
                ignore_patterns=["*.safetensors", "*.bin"]
            )
        console.print(f"[green]Successfully pulled model {args.model_id}[/green]")
        console.print(f"Cached at: {model_path}")
    except Exception as e:
        console.print(f"[red]Error pulling model: {str(e)}[/red]")

def push_model(args):
    """Push a model to Hugging Face"""
    model_path = get_model_path(args.model_id)
    if not model_path:
        console.print(f"[red]Model {args.model_id} not found locally[/red]")
        return

    try:
        with console.status(f"[bold blue]Pushing model {args.model_id}...[/bold blue]"):
            api = HfApi()
            api.upload_folder(
                folder_path=model_path,
                repo_id=args.model_id,
                repo_type="model"
            )
        console.print(f"[green]Successfully pushed model {args.model_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error pushing model: {str(e)}[/red]")

def list_models(args):
    """List downloaded models"""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model ID")
    table.add_column("Cache Location")
    table.add_column("Size")

    cache_dir = HUGGINGFACE_HUB_CACHE
    found_models = False
    
    with console.status("[bold blue]Scanning cache for models...[/bold blue]"):
        for model_dir in Path(cache_dir).glob("**/model.json"):
            try:
                model_path = model_dir.parent
                size = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file())
                size_str = f"{size / 1024 / 1024:.1f} MB"
                model_id = str(model_path.relative_to(cache_dir)).split('/')[0]
                table.add_row(model_id, str(model_path), size_str)
                found_models = True
            except Exception:
                continue

    if found_models:
        console.print(table)
    else:
        console.print("[yellow]No models found in cache[/yellow]")

def list_running(args):
    """List running models"""
    running_models = load_running_models()
    
    if not running_models:
        console.print("[yellow]No running models found[/yellow]")
        return
        
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Port")
    table.add_column("Model ID")
    table.add_column("PID")
    table.add_column("Status")

    for port, info in running_models.items():
        try:
            os.kill(info["pid"], 0)
            status = "[green]Running[/green]"
        except ProcessLookupError:
            status = "[red]Stopped[/red]"
        
        table.add_row(
            str(port),
            info["model_id"],
            str(info["pid"]),
            status
        )

    console.print(table)

def remove_model(args):
    """Remove a downloaded model"""
    model_path = get_model_path(args.model_id)
    if not model_path:
        console.print(f"[red]Model {args.model_id} not found locally[/red]")
        return

    # Check if model is running
    running_models = load_running_models()
    for port, info in running_models.items():
        if info["model_id"] == args.model_id:
            console.print(f"[red]Cannot remove model {args.model_id} - it is currently running on port {port}[/red]")
            return

    if Confirm.ask(f"Are you sure you want to remove {args.model_id}?", default=False):
        import shutil
        shutil.rmtree(model_path)
        console.print(f"[green]Removed model {args.model_id}[/green]")

def main():
    parser = argparse.ArgumentParser(description="LLMX - MLX-LM model management tool", add_help=False)
    subparsers = parser.add_subparsers(dest='command')

    # Serve command
    serve_parser = subparsers.add_parser('serve', add_help=False)
    serve_parser.add_argument('model_id')
    serve_parser.add_argument('--port', type=int, default=8080)

    # Chat command
    chat_parser = subparsers.add_parser('chat', add_help=False)
    chat_parser.add_argument('model_id')
    chat_parser.add_argument('--port', type=int, default=8080)
    chat_parser.add_argument('--temperature', type=float, default=0.7)

    # Show command
    show_parser = subparsers.add_parser('show', add_help=False)
    show_parser.add_argument('model_id')

    # Stop command
    stop_parser = subparsers.add_parser('stop', add_help=False)
    stop_parser.add_argument('port')

    # Pull command
    pull_parser = subparsers.add_parser('pull', add_help=False)
    pull_parser.add_argument('model_id')

    # Push command
    push_parser = subparsers.add_parser('push', add_help=False)
    push_parser.add_argument('model_id')

    # List command
    subparsers.add_parser('list', add_help=False)

    # PS command
    subparsers.add_parser('ps', add_help=False)

    # Remove command
    rm_parser = subparsers.add_parser('rm', add_help=False)
    rm_parser.add_argument('model_id')

    # Help command
    subparsers.add_parser('help', add_help=False)

    args = parser.parse_args()

    if args.command == 'serve':
        serve_model(args)
    elif args.command == 'chat':
        chat_model(args)
    elif args.command == 'show':
        show_model(args)
    elif args.command == 'stop':
        stop_model(args)
    elif args.command == 'pull':
        pull_model(args)
    elif args.command == 'push':
        push_model(args)
    elif args.command == 'list':
        list_models(args)
    elif args.command == 'ps':
        list_running(args)
    elif args.command == 'rm':
        remove_model(args)
    elif args.command == 'help' or not args.command:
        show_help()
    else:
        console.print("[red]Unknown command. Use 'llmx help' to see available commands.[/red]")

if __name__ == '__main__':
    main() 