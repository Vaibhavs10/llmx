"""
LLMX CLI implementation
"""
import os
import json
import click
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from pathlib import Path
import subprocess
import signal
import requests
from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache, get_cache_dir

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

@click.group()
def cli():
    """LLMX - CLI tool for managing MLX-LM models and servers"""
    pass

@cli.command()
@click.argument('model_id')
@click.option('--port', default=8080, help='Port to run the server on')
def serve(model_id, port):
    """Start a model server"""
    running_models = load_running_models()
    
    if str(port) in running_models:
        console.print(f"[red]Port {port} is already in use by another model server[/red]")
        return

    model_path = get_model_path(model_id)
    if not model_path:
        console.print(f"[yellow]Model {model_id} not found locally. Attempting to pull...[/yellow]")
        try:
            model_path = snapshot_download(
                repo_id=model_id,
                ignore_patterns=["*.safetensors", "*.bin"]  # Only download MLX format
            )
        except Exception as e:
            console.print(f"[red]Error pulling model: {str(e)}[/red]")
            return

    cmd = f"mlx_lm.server --model {model_id} --port {port}"
    process = subprocess.Popen(cmd.split(), start_new_session=True)
    
    running_models[str(port)] = {
        "model_id": model_id,
        "pid": process.pid,
        "port": port
    }
    save_running_models(running_models)
    
    console.print(f"[green]Started model server for {model_id} on port {port}[/green]")

@cli.command()
@click.argument('model_id')
def show(model_id):
    """Show information for a model"""
    model_path = get_model_path(model_id)
    if not model_path:
        console.print(f"[red]Model {model_id} not found locally[/red]")
        return

    # Show model info
    console.print(f"[green]Model: {model_id}[/green]")
    console.print(f"Path: {model_path}")
    
    # TODO: Add more model info display (config, size, etc)

@cli.command()
@click.argument('model_id')
@click.option('--port', default=8080, help='Port to run the server on')
def run(model_id, port):
    """Run a model server (alias for serve)"""
    ctx = click.get_current_context()
    ctx.forward(serve)

@cli.command()
@click.argument('port')
def stop(port):
    """Stop a running model server"""
    running_models = load_running_models()
    
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

@cli.command()
@click.argument('model_id')
def pull(model_id):
    """Pull a model from Hugging Face"""
    if get_model_path(model_id):
        console.print(f"[yellow]Model {model_id} already exists in cache[/yellow]")
        return

    try:
        model_path = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.safetensors", "*.bin"]  # Only download MLX format
        )
        console.print(f"[green]Successfully pulled model {model_id}[/green]")
        console.print(f"Cached at: {model_path}")
    except Exception as e:
        console.print(f"[red]Error pulling model: {str(e)}[/red]")

@cli.command()
@click.argument('model_id')
def push(model_id):
    """Push a model to Hugging Face"""
    model_path = get_model_path(model_id)
    if not model_path:
        console.print(f"[red]Model {model_id} not found locally[/red]")
        return

    try:
        api = HfApi()
        api.upload_folder(
            folder_path=model_path,
            repo_id=model_id,
            repo_type="model"
        )
        console.print(f"[green]Successfully pushed model {model_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error pushing model: {str(e)}[/red]")

@cli.command()
def list():
    """List downloaded models"""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model ID")
    table.add_column("Cache Location")
    table.add_column("Size")

    cache_dir = get_cache_dir()
    # This is a simplified approach - in practice you'd want to parse the cache more carefully
    for model_dir in Path(cache_dir).glob("**/model.json"):
        try:
            model_path = model_dir.parent
            size = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file())
            size_str = f"{size / 1024 / 1024:.1f} MB"
            # Extract model ID from path - this is a simplified approach
            model_id = str(model_path.relative_to(cache_dir)).split('/')[0]
            table.add_row(model_id, str(model_path), size_str)
        except Exception:
            continue

    console.print(table)

@cli.command()
def ps():
    """List running models"""
    running_models = load_running_models()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Port")
    table.add_column("Model ID")
    table.add_column("PID")
    table.add_column("Status")

    for port, info in running_models.items():
        try:
            os.kill(info["pid"], 0)
            status = "Running"
        except ProcessLookupError:
            status = "Stopped"
        
        table.add_row(
            str(port),
            info["model_id"],
            str(info["pid"]),
            status
        )

    console.print(table)

@cli.command()
@click.argument('model_id')
def rm(model_id):
    """Remove a downloaded model"""
    model_path = get_model_path(model_id)
    if not model_path:
        console.print(f"[red]Model {model_id} not found locally[/red]")
        return

    # Check if model is running
    running_models = load_running_models()
    for port, info in running_models.items():
        if info["model_id"] == model_id:
            console.print(f"[red]Cannot remove model {model_id} - it is currently running on port {port}[/red]")
            return

    import shutil
    shutil.rmtree(model_path)
    console.print(f"[green]Removed model {model_id}[/green]")

@cli.command()
@click.argument('model_id')
@click.option('--port', default=8080, help='Port to run the server on')
@click.option('--temperature', default=0.7, help='Sampling temperature')
def chat(model_id, port, temperature):
    """Start an interactive chat session with a model"""
    running_models = load_running_models()
    server_started = False
    
    # Check if we need to start the server
    if str(port) not in running_models:
        model_path = get_model_path(model_id)
        if not model_path:
            console.print(f"[yellow]Model {model_id} not found locally. Attempting to pull...[/yellow]")
            try:
                model_path = snapshot_download(
                    repo_id=model_id,
                    ignore_patterns=["*.safetensors", "*.bin"]
                )
            except Exception as e:
                console.print(f"[red]Error pulling model: {str(e)}[/red]")
                return

        cmd = f"mlx_lm.server --model {model_id} --port {port}"
        process = subprocess.Popen(cmd.split(), start_new_session=True)
        
        running_models[str(port)] = {
            "model_id": model_id,
            "pid": process.pid,
            "port": port
        }
        save_running_models(running_models)
        server_started = True
        console.print(f"[green]Started model server for {model_id} on port {port}[/green]")
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
    
    # If we started the server, ask if user wants to keep it running
    if server_started:
        keep_running = Prompt.ask(
            "\nDo you want to keep the server running?",
            choices=["y", "n"],
            default="n"
        )
        
        if keep_running.lower() != 'y':
            try:
                os.killpg(os.getpgid(running_models[str(port)]["pid"]), signal.SIGTERM)
                del running_models[str(port)]
                save_running_models(running_models)
                console.print("[green]Stopped model server.[/green]")
            except (KeyError, ProcessLookupError):
                pass

if __name__ == '__main__':
    cli() 