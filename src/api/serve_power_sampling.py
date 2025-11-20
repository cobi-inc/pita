# Custom Software
from src.api.api_template import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatMessageRole,
    Usage,
)
from src.sampling.power_sample import power_sampling
from src.inference.autoregressive_sampler_backend import create_autoregressive_sampler
from src.api.test_time_coding import decode

# Standard Libraries
import time
from typing import Optional, List, Union, Dict, Any
from fastapi import FastAPI, HTTPException
import uvicorn

# Import from your existing power_sample.py
# Adjust the import path if necessary based on where you place this script

app = FastAPI(title="Power Sampling API")

# --- Global Sampler State ---
SERVER_STATE = {"sampler": None}

# --- Initialization on Startup ---
@app.on_event("startup")
async def startup_event():
    # Configuration hardcoded; ideally load from config/env vars
    ENGINE = "vllm"
    MODEL_NAME = "Qwen/Qwen3-4B-AWQ" # Example model from paper
    DTYPE = "auto" # Let the engine decide
    GPU_MEMORY_UTILIZATION = 0.85
    CONTEXT_LENGTH = 8192 # Default max buffer for generation
    MAX_LOGPROBS = 100

    print(f"Loading model {MODEL_NAME}...")

    sampler = create_autoregressive_sampler(ENGINE, 
                                MODEL_NAME, 
                                dtype=DTYPE, 
                                gpu_memory_utilization=GPU_MEMORY_UTILIZATION, 
                                max_model_len=CONTEXT_LENGTH, 
                                max_logprobs = MAX_LOGPROBS)

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

    # Check the message in the system prompt for inference scaling sampling parameters
    # Example system prompt: "PS_1000_250_3 SMC_ You are a personal..." Means perform PS with 1000 total tokens, block size 250, 3 MCMC steps
    ps_params, smc_params, best_of_params = None, None, None
    if len(request.messages) > 0 and request.messages[0].role == ChatMessageRole.System:
        system_content = request.messages[0].content
        if system_content.startswith("ITS"):
            # Decode the parameters from the system prompt
            ps_params, smc_params, best_of_params = decode(system_content)

            # Remove the parameter encoding from the system prompt
            request.messages[0].content = " ".join(system_content.split(" ")[1:])


    # Format the message into the chat template
    prompt = sampler.tokenizer.apply_chat_template(
        request.messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = sampler.sampling_params.enable_thinking
    )
    
    # Call the power sampling function
    # Note: power_sampling returns (text, acceptances, block_acceptances, index_proposals, total_tokens)
    if(ps_params is not None):
        # Set the power sampling parameters
        sampler.power_sampling_params = ps_params

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