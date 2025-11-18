# Custom Software
from src.api.api_template import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatMessageRole,
    Usage,
)
from src.sampling.power_sample import AutoregressiveSampler, sliding_window_power_sample, power_sampling

import time
import uuid
from typing import Optional, List, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import torch
from transformers import AutoTokenizer
from vllm import LLM

# Import from your existing power_sample.py
# Adjust the import path if necessary based on where you place this script

app = FastAPI(title="Power Sampling API")

# --- Global Sampler State ---
SERVER_STATE = {"sampler": None}

# --- Initialization on Startup ---
@app.on_event("startup")
async def startup_event():
    # Configuration hardcoded; ideally load from config/env vars
    MODEL_NAME = "Qwen/Qwen3-4B-AWQ" # Example model from paper
    TOKEN_COUNT = 18000 # Default max buffer for generation
    BLOCK_SIZE = 400 # tokens per block. Number of blocks = token_count / block_size
    MCMC_STEPS = 10 

    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Initialize VLLM locally for performance (as done in power_sample.py main)
    llm = LLM(model=MODEL_NAME,
              dtype="auto",
              gpu_memory_utilization=0.85,
              max_model_len=TOKEN_COUNT,
              max_logprobs=100, # needed for MCMC
              logprobs_mode='raw_logits',
              trust_remote_code=True)

    # Initialize the sampler with defaults; these might be overridden per request
    sampler = AutoregressiveSampler(
        api=False,
        llm=llm,
        tokenizer=tokenizer,
        enable_thinking=False,
        power_sampling_temperature=0.25, # Default, can be overridden by request
        top_k=100,
        token_count=TOKEN_COUNT,
        block_size=BLOCK_SIZE,
        MCMC_steps=MCMC_STEPS
    )

    SERVER_STATE["sampler"] = sampler
    print("Power Sampling Server Ready.")

# --- API Endpoint ---
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_completion(request: ChatCompletionRequest):
    # New Request
    print(f"New chat completion request: {time.time()}")

    # Check if the server is online
    if SERVER_STATE["sampler"] is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Grab the Sampler from the Server State
    sampler = SERVER_STATE["sampler"]


    # Check the max tokens the user want to generate
    # Handle optional max_tokens (default to sampler's token_count if None)
    max_tokens = request.max_tokens if request.max_tokens is not None else sampler.token_count
    if max_tokens > sampler.tokenizer.model_max_length:
        raise HTTPException(status_code=400, detail=f"Requested {request.max_tokens}. {sampler.llm.model} can only provide {sampler.tokenizer.model_max_length} tokens.")
    sampler.token_count = max_tokens 

    # Format the message into the chat template
    prompt = sampler.tokenizer.apply_chat_template(
        request.messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = sampler.enable_thinking
    )
    
    temperature = request.temperature if request.temperature is not None else 1
    print("Power Sampling with the following parameters:")
    print(f" - Tokens: {max_tokens}")
    print(f" - MCMC Steps: {request.MCMC_steps}")
    print(f" - Block Size: {request.block_size}")

    # Call the power sampling function
    # Note: sliding_window_power_sample returns (text, acceptances, block_acceptances, total_tokens)
    if(request.MCMC_steps != None and request.block_size != None):
        # Set the power sampling parameters
        sampler.MCMC_steps = request.MCMC_steps
        sampler.block_size = request.block_size

        # Call the power sampling function
        generated_text, _, _, _, total_generated = power_sampling(
            sampler=sampler,
            prompt=prompt
        )
    else:
        generated_text, _, _ = sampler.sample(prompt, sampler.token_count)
        generated_text = sampler.tokenizer.decode(generated_text, skip_special_tokens=True)

    # TO DO
    # Create message ID
    message_id = "0"

    # TO DO:Indicate the reason for finishing
    finish_reason = "NA"

    # Create ChatCompletionChoice   
    chat_completion_choice = ChatCompletionChoice(
        index=0,
        message=ChatCompletionMessage(
            role=ChatMessageRole.Assistant,
            content=generated_text
        ),
        finish_reason=finish_reason
    )

    # Create Usage object
    prompt_token_count = len(sampler.tokenizer.encode(prompt))
    completion_token_count = len(sampler.tokenizer.encode(generated_text))
    total_token_count = prompt_token_count + completion_token_count
    usage = Usage(
        prompt_tokens=prompt_token_count,
        completion_tokens=completion_token_count,
        total_tokens=total_token_count
    )

    # Return just the generated text to the AI Scientist client
    return ChatCompletionResponse(
        id=message_id,
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[chat_completion_choice],
        usage=usage
    )

@app.get("/v1/models")
async def list_models():
    sampler = SERVER_STATE["sampler"]
    return {"object": "list", "data": [{"id": "Qwen/Qwen3-4B-AWQ", "object": "model", "created": 0, "owned_by": "custom"}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)