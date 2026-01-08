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
import argparse
import os
from typing import Optional, List, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Configuration ---
SERVER_CONFIG = {
    "engine": os.environ.get("PITA_ENGINE", "vllm"),
    "model": os.environ.get("PITA_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    "tokenizer": os.environ.get("PITA_TOKENIZER", None),
    "port": int(os.environ.get("PITA_PORT", 8001)),
    "host": os.environ.get("PITA_HOST", "0.0.0.0")
}

# --- FastAPI Application ---
app = FastAPI(title="PITA API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Sampler State ---
SERVER_STATE = {"sampler": None}

# --- Initialization on Startup ---
@app.on_event("startup")
async def startup_event():
    ENGINE = SERVER_CONFIG["engine"]
    MODEL_NAME = SERVER_CONFIG["model"]
    TOKENIZER_PATH = SERVER_CONFIG["tokenizer"]
    
    # Defaults
    DTYPE = "auto"
    GPU_MEMORY_UTILIZATION = 0.85
    CONTEXT_LENGTH = 1024 
    MAX_PROBS = 1000  # Unified max_probs for both logits and logprobs

    if ENGINE == "llama_cpp":
        DTYPE = "Q5_K_M"
        if TOKENIZER_PATH is None:
            TOKENIZER_PATH = MODEL_NAME 
        MAX_PROBS = 0 # llama.cpp handling might differ, matching previous logic roughly

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
    SERVER_STATE["sampler"] = sampler
    print("PITA Server Ready.")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down PITA Server...")
    sampler = SERVER_STATE["sampler"]
    if sampler is not None:
        if sampler.engine == "vllm":
            sampler.llm.close()
        elif sampler.engine == "llama_cpp":
            sampler.llm.close()
            
    SERVER_STATE["sampler"] = None
    print("Shutdown complete.")

# --- API Endpoint ---
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_completion(request: ChatCompletionRequest):
    print(f"New chat completion request: {time.time()}")

    if SERVER_STATE["sampler"] is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    sampler = SERVER_STATE["sampler"]

    max_tokens = request.max_tokens if request.max_tokens is not None else sampler.sampling_params.max_tokens
    if max_tokens > sampler.tokenizer.model_max_length:
        raise HTTPException(status_code=400, detail=f"Requested {max_tokens} tokens. {sampler.model} can only provide {sampler.tokenizer.model_max_length} tokens.")
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
        id="chatcmpl-0",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[chat_completion_choice],
        usage=usage
    )

@app.get("/v1/models")
async def list_models():
    sampler = SERVER_STATE.get("sampler")
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

def run_server():
    parser = argparse.ArgumentParser(description="PITA API Server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name or path")
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "llama_cpp"], help="Inference engine")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (optional)")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    
    args = parser.parse_args()
    
    # Update configuration
    SERVER_CONFIG["model"] = args.model
    SERVER_CONFIG["engine"] = args.engine
    SERVER_CONFIG["tokenizer"] = args.tokenizer
    SERVER_CONFIG["port"] = args.port
    SERVER_CONFIG["host"] = args.host
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    run_server()
