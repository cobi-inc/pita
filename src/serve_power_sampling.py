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
from power_sample import AutoregressiveSampler, sliding_window_power_sample, power_sampling

app = FastAPI(title="Power Sampling API")

# --- Global Sampler State ---
SERVER_STATE = {"sampler": None}

# --- AI Scientist API ---
class CompletionRequest(BaseModel):
    model: str
    messages: Union[str, List[str], List[Dict[str, Any]]]  # Accept chat format too
    temperature: float = 1.0
    max_tokens: int = 1024
    n: int = 1

class CompletionResponse(BaseModel):
    content: str

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
@app.post("/power_sample", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    if SERVER_STATE["sampler"] is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Grab the Sampler from the Server State
    sampler = SERVER_STATE["sampler"]
    
    # Ensure token_count matches max_tokens if possible, or clamp it
    sampler.token_count = request.max_tokens
    
    # Format the message into the chat template
    prompt = sampler.tokenizer.apply_chat_template(
        request.messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = sampler.enable_thinking
    )

    # Call the power sampling function
    # Note: sliding_window_power_sample returns (text, acceptances, block_acceptances, total_tokens)
    if(float(request.temperature) != 1):
        generated_text, _, _, total_generated = power_sampling(
            sampler=sampler,
            prompt=prompt
        )
    else:
        generated_text, _, _ = sampler.sample(prompt, sampler.token_count)
        generated_text = sampler.tokenizer.decode(generated_text, skip_special_tokens=False)

    # Return just the generated text to the AI Scientist client
    return CompletionResponse(
        content=generated_text
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)