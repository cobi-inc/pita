"""
PITA CLI - Command Line Interface for the PITA library.

This module provides the main entry point for the `pita` command,
enabling subcommands like `pita serve` to start the API server.
"""

import click
import os


@click.group()
@click.version_option(version="0.0.1", prog_name="pita")
def cli():
    """PITA - Probabilistic Inference Time Algorithms.
    
    A library to enable probabilistic inference time algorithms from any model.
    """
    pass


@cli.command()
@click.option(
    '--model', '-m',
    default=None,
    help='Model name or path (default: PITA_MODEL env or Qwen/Qwen2.5-0.5B-Instruct)'
)
@click.option(
    '--engine', '-e',
    default=None,
    type=click.Choice(['vllm', 'llama_cpp']),
    help='Inference engine (default: PITA_ENGINE env or vllm)'
)
@click.option(
    '--tokenizer', '-t',
    default=None,
    help='Tokenizer path (optional, defaults to model path)'
)
@click.option(
    '--port', '-p',
    default=None,
    type=int,
    help='Port number (default: PITA_PORT env or 8001)'
)
@click.option(
    '--host', '-h',
    default=None,
    help='Host address (default: PITA_HOST env or 0.0.0.0)'
)
def serve(model, engine, tokenizer, port, host):
    """Start the PITA API server.
    
    Launches an OpenAI-compatible API server for inference.
    Configuration can be set via CLI options or environment variables:
    
    \b
    - PITA_MODEL: Model name or path
    - PITA_ENGINE: Inference engine (vllm or llama_cpp)
    - PITA_TOKENIZER: Tokenizer path
    - PITA_PORT: Server port
    - PITA_HOST: Server host
    
    Examples:
    
    \b
        pita serve
        pita serve --model Qwen/Qwen2.5-0.5B-Instruct --engine vllm
        pita serve -m ./my-model.gguf -e llama_cpp -p 8080
    """
    from pita.api.serve import start_server
    start_server(model=model, engine=engine, tokenizer=tokenizer, port=port, host=host)


if __name__ == "__main__":
    cli()
