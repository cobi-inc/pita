# Custom Software
from pita.api.api_template import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatMessageRole,
    Usage,
)
from pita.inference.LLM_backend import AutoregressiveSampler
from pita.api.test_time_coding import decode

# Standard Libraries
import time
import uuid
import argparse
import os
import contextlib
from typing import Optional, List, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Configuration ---
def get_default_config():
    tokenizer_env = os.environ.get("PITA_TOKENIZER")
    if tokenizer_env is not None and tokenizer_env.strip().lower() == "none":
        tokenizer_env = None
    return {
        "engine": os.environ.get("PITA_ENGINE", "vllm"),
        "model": os.environ.get("PITA_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        "tokenizer": tokenizer_env,
        "port": int(os.environ.get("PITA_PORT", 8001)),
        "host": os.environ.get("PITA_HOST", "0.0.0.0")
    }

SERVER_CONFIG = get_default_config()

# --- Application Factory ---
def create_app(config: Dict[str, Any] = None) -> FastAPI:
    if config is None:
        config = SERVER_CONFIG

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        ENGINE = config["engine"]
        MODEL_NAME = config["model"]
        TOKENIZER_PATH = config["tokenizer"]
        
        # Defaults
        DTYPE = "auto"
        GPU_MEMORY_UTILIZATION = 0.85
        CONTEXT_LENGTH = 1024 
        MAX_PROBS = 1000  # Unified max_probs for both logits and logprobs

        if ENGINE == "llama_cpp":
            if TOKENIZER_PATH is None:
                TOKENIZER_PATH = MODEL_NAME 

        print(f"Loading model {MODEL_NAME} using {ENGINE}...")

        # Instantiate Autoregressive Sampler directly
        # Note: max_logprobs and logits_per_token are unified into max_probs in the base class
        sampler = AutoregressiveSampler(
            engine=ENGINE,
            model=MODEL_NAME,
            dtype=DTYPE,
            tokenizer_path=TOKENIZER_PATH,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=CONTEXT_LENGTH,
            max_probs=MAX_PROBS, 
            logits_processor=False, # Defaulting to False unless needed
            trust_remote_code=False,
            sampling_params=None
        )
        
        sampler.sampling_params.max_tokens = 1000
        app.state.sampler = sampler
        print("PITA Server Ready.")
        
        yield
        
        # Shutdown
        print("Shutting down PITA Server...")
        sampler = getattr(app.state, "sampler", None)
        if sampler is not None:
            try:
                if sampler.engine == "vllm":
                    sampler.llm.close()
                elif sampler.engine == "llama_cpp":
                    sampler.llm.close()
            except Exception as e:
                # Ensure shutdown continues even if closing the LLM fails
                print(f"Warning: error while closing sampler LLM: {e}")
        app.state.sampler = None
        print("Shutdown complete.")

    app = FastAPI(title="PITA API", lifespan=lifespan)

    # --- CORS Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- API Endpoint ---
    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_completion(request: ChatCompletionRequest, raw_request: Request):
        print(f"New chat completion request: {time.time()}")

        sampler = getattr(raw_request.app.state, "sampler", None)
        if sampler is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        max_tokens = request.max_tokens if request.max_tokens is not None else sampler.sampling_params.max_tokens
        if max_tokens > sampler.tokenizer.model_max_length:
            raise HTTPException(
                status_code=400,
                detail=f"Requested {max_tokens} tokens, but the model can only generate up to {sampler.tokenizer.model_max_length} tokens.",
            )
        sampler.sampling_params.max_tokens = max_tokens 

        # Check for ITS scaling parameters in system prompt
        chain_sampling, token_sampling = None, None
        if len(request.messages) > 0 and request.messages[0].role == ChatMessageRole.System:
            system_content = request.messages[0].content
            if system_content.startswith("ITS"):
                chain_sampling, token_sampling = decode(system_content)
                request.messages[0].content = " ".join(system_content.split(" ")[1:])

        prompt = sampler.tokenizer.apply_chat_template(
            request.messages,
            tokenize = False,
            add_generation_prompt = True,
            enable_thinking = sampler.sampling_params.enable_thinking
        )
        
        if chain_sampling is not None:
            if token_sampling is not None:
                if getattr(sampler, "token_sample_name", None) != "Power Sampling":
                    print("Enabling Power Sampling automatically.")
                    sampler.enable_power_sampling(
                        block_size=token_sampling.block_size,
                        MCMC_steps=token_sampling.MCMC_steps,
                        token_metric=token_sampling.token_metric 
                    )
                
                if hasattr(chain_sampling, "token_sampling_method"):
                    chain_sampling.token_sampling_method = "token_sample"
            else:
                chain_sampling.token_sampling_method = "standard"
            output = chain_sampling.sample(sampler, prompt)
        elif token_sampling is not None:
            output = token_sampling.sample(sampler, prompt)
        else:
            output = sampler.sample(prompt)
            
        generated_text = sampler.tokenizer.decode(output.tokens, skip_special_tokens=True)

        chat_completion_choice = ChatCompletionChoice(
            index=0,
            message=ChatCompletionMessage(
                role=ChatMessageRole.Assistant,
                content=generated_text
            ),
            finish_reason="stop"
        )

        prompt_encoded = sampler.tokenizer.encode(prompt)
        prompt_token_count = len(prompt_encoded)
        completion_token_count = len(output.tokens)
        usage = Usage(
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
            total_tokens=prompt_token_count + completion_token_count
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[chat_completion_choice],
            usage=usage
        )

    @app.get("/v1/models")
    async def list_models(raw_request: Request):
        sampler = getattr(raw_request.app.state, "sampler", None)
        if sampler is None:
            # Mirror the initialization check behavior used in create_completion
            raise HTTPException(status_code=503, detail="Model is not initialized")
        return {
            "object": "list",
            "data": [
                {
                    "id": sampler.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "custom",
                }
            ],
        }

    return app

# Create a default app instance for backward compatibility or direct imports
# Note: This app instance uses default environment variables. 
# To use custom config, call create_app(config) instead.
app = create_app()

def start_server(model=None, engine=None, tokenizer=None, port=None, host=None):
    """
    Start the PITA API server with the specified configuration.
    
    This function is the main entry point for starting the server programmatically
    or via the CLI. Parameters default to environment variables if not specified.
    
    Args:
        model: Model name or path (default: PITA_MODEL env or 'Qwen/Qwen2.5-0.5B-Instruct')
        engine: Inference engine - 'vllm' or 'llama_cpp' (default: PITA_ENGINE env or 'vllm')
        tokenizer: Tokenizer path (default: PITA_TOKENIZER env or None)
        port: Port number (default: PITA_PORT env or 8001)
        host: Host address (default: PITA_HOST env or '0.0.0.0')
    """
    default_config = get_default_config()
    
    # Use provided values or fall back to defaults from environment
    config = {
        "model": model if model is not None else default_config["model"],
        "engine": engine if engine is not None else default_config["engine"],
        "tokenizer": tokenizer if tokenizer is not None else default_config["tokenizer"],
        "port": port if port is not None else default_config["port"],
        "host": host if host is not None else default_config["host"],
    }
    
    # Create a dedicated app instance with the specified config
    server_app = create_app(config)
    
    uvicorn.run(server_app, host=config["host"], port=config["port"])


def run_server():
    """
    Run the PITA API server using command-line arguments.
    
    This function provides backward compatibility for running the server via:
        python -m pita.api.serve
    
    For programmatic use, prefer start_server() instead.
    """
    parser = argparse.ArgumentParser(description="PITA API Server")
    default_config = get_default_config()
    parser.add_argument("--model", type=str, default=default_config["model"], help="Model name or path")
    parser.add_argument("--engine", type=str, default=default_config["engine"], choices=["vllm", "llama_cpp"], help="Inference engine")
    parser.add_argument("--tokenizer", type=str, default=default_config["tokenizer"], help="Tokenizer path (optional)")
    parser.add_argument("--port", type=int, default=default_config["port"], help="Port number")
    parser.add_argument("--host", type=str, default=default_config["host"], help="Host address")
    
    args = parser.parse_args()
    
    start_server(
        model=args.model,
        engine=args.engine,
        tokenizer=args.tokenizer,
        port=args.port,
        host=args.host
    )


if __name__ == "__main__":
    run_server()

