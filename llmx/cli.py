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
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

console = Console()
LLMX_HOME = os.path.expanduser("~/.llmx/running.json")
os.makedirs(os.path.dirname(LLMX_HOME), exist_ok=True)

def manage_running_models(action='load', port=None, server_info=None):
    """Unified function to manage running models state"""
    try:
        running_models = {}
        if os.path.exists(LLMX_HOME):
            with open(LLMX_HOME, 'r') as f:
                running_models = json.load(f)
            
            # Clean up stale entries
            running_models = {
                p: info for p, info in running_models.items()
                if is_process_running(info.get("pid"))
            }
        
        if action == 'load':
            return running_models
        elif action == 'save':
            if server_info:
                running_models[port] = server_info
            elif port in running_models:
                del running_models[port]
            with open(LLMX_HOME, 'w') as f:
                json.dump(running_models, f)
            return running_models
    except (json.JSONDecodeError, IOError):
        return {}

def is_process_running(pid):
    """Check if a process is running"""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, TypeError):
        return False

def manage_server(action, model_id=None, port=None):
    """Unified function to manage model servers"""
    running_models = manage_running_models('load')
    
    if action == 'start':
        if str(port) in running_models:
            console.print(f"[red]Port {port} is already in use[/red]")
            return None
            
        try:
            cmd = f"mlx_lm.server --model {model_id} --port {port}"
            process = subprocess.Popen(cmd.split(), start_new_session=True)
            server_info = {"model_id": model_id, "pid": process.pid, "port": port}
            manage_running_models('save', str(port), server_info)
            console.print(f"[green]Started model server for {model_id} on port {port}[/green]")
            return server_info
        except Exception as e:
            console.print(f"[red]Error starting server: {str(e)}[/red]")
            return None
            
    elif action == 'stop':
        try:
            os.killpg(os.getpgid(running_models[str(port)]["pid"]), signal.SIGTERM)
            manage_running_models('save', str(port))
            console.print(f"[green]Stopped server on port {port}[/green]")
            return True
        except (KeyError, ProcessLookupError):
            console.print(f"[red]No server running on port {port}[/red]")
            return False

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
                console.print(f"\n[bold purple]Assistant[/bold purple]")
                
                # Stream the response
                current_message = []
                with requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={"messages": messages, "temperature": temperature, "stream": True},
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                    stream=True
                ) as response:
                    if response.status_code != 200:
                        console.print(f"[red]Error: Server returned status code {response.status_code}[/red]")
                        continue
                        
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line.decode('utf-8').removeprefix('data: '))
                            if data.get("choices"):
                                chunk = data["choices"][0].get("delta", {}).get("content", "")
                                if chunk:
                                    current_message.append(chunk)
                                    console.print(chunk, end="")
                        except json.JSONDecodeError:
                            continue
                
                console.print()  # New line after response
                assistant_message = "".join(current_message)
                messages.append({"role": "assistant", "content": assistant_message})
                    
            except (KeyboardInterrupt, requests.exceptions.RequestException) as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                break
    finally:
        return Confirm.ask("\nKeep server running?", default=False)

def main():
    parser = argparse.ArgumentParser(description="LLMX - MLX-LM model management tool", add_help=False)
    subparsers = parser.add_subparsers(dest='command')

    # Add command parsers
    for cmd, help_text in {
        'serve': 'Start a model server',
        'chat': 'Start an interactive chat session',
        'stop': 'Stop a running model',
        'pull': 'Pull a model from Hugging Face',
        'list': 'List downloaded models',
        'ps': 'List running models',
        'help': 'Show help message'
    }.items():
        cmd_parser = subparsers.add_parser(cmd, add_help=False, help=help_text)
        if cmd in ['serve', 'chat']:
            cmd_parser.add_argument('model_id')
            cmd_parser.add_argument('--port', type=int, default=8080)
            if cmd == 'chat':
                cmd_parser.add_argument('--temperature', type=float, default=0.7)
        elif cmd == 'stop':
            cmd_parser.add_argument('port')
        elif cmd == 'pull':
            cmd_parser.add_argument('model_id')

    args = parser.parse_args()
    running_models = manage_running_models('load')

    if args.command in ['serve', 'chat']:
        try:
            model_path = snapshot_download(
                repo_id=args.model_id,
                ignore_patterns=["*.safetensors", "*.bin"]
            )
            if not model_path:
                return
                
            port = int(getattr(args, 'port', 8080))
            if server_info := manage_server('start', args.model_id, port):
                if args.command == 'chat':
                    import time
                    time.sleep(2)  # Give server time to start
                    if not chat_session(port, getattr(args, 'temperature', 0.7)):
                        manage_server('stop', port=port)
                        
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")

    elif args.command == 'stop':
        manage_server('stop', port=args.port)

    elif args.command == 'pull':
        try:
            snapshot_download(
                repo_id=args.model_id,
                ignore_patterns=["*.safetensors", "*.bin"]
            )
        except Exception as e:
            console.print(f"[red]Error pulling model: {str(e)}[/red]")

    elif args.command == 'list':
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model ID", style="green")
        table.add_column("Size", justify="right")
        
        for model_dir in Path(HUGGINGFACE_HUB_CACHE).glob("models--*"):
            try:
                parts = model_dir.name.split('--')
                model_id = f"{parts[1]}/{parts[2]}" if len(parts) == 3 else parts[1]
                size = sum(f.stat(follow_symlinks=False).st_size for f in model_dir.glob('**/*') if f.is_file())
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
            status = "[green]Running[/green]" if is_process_running(info["pid"]) else "[red]Stopped[/red]"
            table.add_row(str(port), info["model_id"], status)
        console.print(table)

    else:
        help_text = """
[bold]llmx Commands:[/bold]

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

if __name__ == '__main__':
    main() 