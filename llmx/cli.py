"""
LLMX CLI implementation
"""
import os
import json
import click
from rich.console import Console
from rich.table import Table
from pathlib import Path
import subprocess
import signal
import requests
from huggingface_hub import HfApi

console = Console()

# Constants
LLMX_HOME = os.path.expanduser("~/.llmx")
MODELS_DIR = os.path.join(LLMX_HOME, "models")
RUNNING_FILE = os.path.join(LLMX_HOME, "running.json")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)

def load_running_models():
    if os.path.exists(RUNNING_FILE):
        with open(RUNNING_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_running_models(running_models):
    with open(RUNNING_FILE, 'w') as f:
        json.dump(running_models, f)

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

    model_path = os.path.join(MODELS_DIR, model_id)
    if not os.path.exists(model_path):
        console.print(f"[yellow]Model {model_id} not found locally. Attempting to pull...[/yellow]")
        subprocess.run(['llmx', 'pull', model_id])

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
    model_path = os.path.join(MODELS_DIR, model_id)
    if not os.path.exists(model_path):
        console.print(f"[red]Model {model_id} not found locally[/red]")
        return

    # TODO: Add model info display
    console.print(f"[green]Model: {model_id}[/green]")
    console.print(f"Path: {model_path}")

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
    model_path = os.path.join(MODELS_DIR, model_id)
    if os.path.exists(model_path):
        console.print(f"[yellow]Model {model_id} already exists locally[/yellow]")
        return

    try:
        api = HfApi()
        api.snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            ignore_patterns=["*.safetensors", "*.bin"]  # Only download MLX format
        )
        console.print(f"[green]Successfully pulled model {model_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error pulling model: {str(e)}[/red]")

@cli.command()
@click.argument('model_id')
def push(model_id):
    """Push a model to Hugging Face"""
    model_path = os.path.join(MODELS_DIR, model_id)
    if not os.path.exists(model_path):
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
    table.add_column("Status")
    table.add_column("Size")

    for model in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model)
        size = sum(f.stat().st_size for f in Path(model_path).glob('**/*') if f.is_file())
        size_str = f"{size / 1024 / 1024:.1f} MB"
        table.add_row(model, "Downloaded", size_str)

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
    model_path = os.path.join(MODELS_DIR, model_id)
    if not os.path.exists(model_path):
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

if __name__ == '__main__':
    cli() 