# Custom Software
from pita.api.api_template import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatMessageRole,
    Usage,
)
from pita.inference.autoregressive_sampler_backend import create_autoregressive_sampler
from pita.api.test_time_coding import decode

# Standard Libraries
import time
from typing import Optional, List, Union, Dict, Any
from fastapi import FastAPI, HTTPException
import uvicorn

# --- FastAPI Application ---
app = FastAPI(title="PITA API")

# --- Global Sampler State ---
SERVER_STATE = {"sampler": None}

# --- Initialization on Startup ---
@app.on_event("startup")
async def startup_event():
    # Configuration hardcoded; ideally load from config/env vars
    ENGINE = "vllm"

    # LLM parameters
    if(ENGINE == "vllm"):
        MODEL_NAME = "Qwen/Qwen3-4B-AWQ"
        DTYPE = "auto"
        GPU_MEMORY_UTILIZATION = 0.85
        TOKENIZER_PATH = None
        CONTEXT_LENGTH = 1024 # Default max buffer for generation
        MAX_LOGPROBS = 1000
        LOGITS_PER_TOKEN = 1000

    elif(ENGINE == "llama_cpp"):
        MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
        DTYPE = "Q5_K_M"
        GPU_MEMORY_UTILIZATION = 0.85
        TOKENIZER_PATH = "Qwen/Qwen3-4B-AWQ"
        CONTEXT_LENGTH = 1024 # Default max buffer for generation
        MAX_LOGPROBS = None
        LOGITS_PER_TOKEN = 1000

    print(f"Loading model {MODEL_NAME} using {ENGINE}...")

    #Initialize Autoregressive Sampler
    sampler = create_autoregressive_sampler(
        engine = ENGINE, 
        model = MODEL_NAME, 
        dtype = DTYPE,
        tokenizer_path = TOKENIZER_PATH, 
        gpu_memory_utilization = GPU_MEMORY_UTILIZATION, 
        max_model_len = CONTEXT_LENGTH, 
        max_logprobs = MAX_LOGPROBS,
        logits_per_token = LOGITS_PER_TOKEN
    ) 
    
    sampler.sampling_params.max_tokens = 1000

    SERVER_STATE["sampler"] = sampler
    print("PITA Server Ready.")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down PITA Server...")
    sampler = SERVER_STATE["sampler"]
    if sampler is not None:
        # Properly close/free the model based on engine
        if sampler.engine == "vllm":
            sampler.llm.close()
        elif sampler.engine == "llama_cpp":
            # Close the Llama object directly
            sampler.llm.close()  # or del sampler.llm if close() is not available

    SERVER_STATE["sampler"] = None
    print("Shutdown complete.")

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
    # Handle optional max_tokens (default to sampler's max_tokens if None)
    max_tokens = request.max_tokens if request.max_tokens is not None else sampler.sampling_params.max_tokens
    if max_tokens > sampler.tokenizer.model_max_length:
        raise HTTPException(status_code=400, detail=f"Requested {max_tokens} tokens. {sampler.model} can only provide {sampler.tokenizer.model_max_length} tokens.")
    sampler.sampling_params.max_tokens = max_tokens 

    # Check the message in the system prompt for inference scaling sampling parameters
    # Format: ITS_<chain_sampling>_<params>_<token_sampling>_<params>
    chain_sampling, token_sampling = None, None
    if len(request.messages) > 0 and request.messages[0].role == ChatMessageRole.System:
        system_content = request.messages[0].content
        if system_content.startswith("ITS"):
            # Decode the parameters from the system prompt
            chain_sampling, token_sampling = decode(system_content)

            # Remove the parameter encoding from the system prompt
            request.messages[0].content = " ".join(system_content.split(" ")[1:])


    # Format the message into the chat template
    prompt = sampler.tokenizer.apply_chat_template(
        request.messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = sampler.sampling_params.enable_thinking
    )
    
    # Call the appropriate sampling method based on decoded parameters
    # Chain sampling (SMC, Best-of-N) operates on full sequences
    # Token sampling (Power Sampling) can be used standalone or combined with chain sampling
    
    if chain_sampling is not None:
        # Use chain sampling method (SMC or Best-of-N)
        # If token_sampling is also specified, configure chain sampler to use power sampling
        if token_sampling is not None:
            # Verify Power Sampling is enabled on the sampler before proceeding
            if getattr(sampler, "token_sample_name", None) != "Power Sampling":
                # Automatically enable Power Sampling using parameters from the token_sampling object
                print("Power Sampling not enabled on sampler. Enabling automatically based on request parameters.")
                sampler.enable_power_sampling(
                    block_size=token_sampling.block_size,
                    MCMC_steps=token_sampling.MCMC_steps,
                    token_metric=token_sampling.token_metric 
                )
            
            # Configure SMC to use token_sample method (Power Sampling)
            # Use hasattr check as Best-of-N does not support token_sampling_method yet
            if hasattr(chain_sampling, "token_sampling_method"):
                chain_sampling.token_sampling_method = "token_sample"
        else:
            chain_sampling.token_sampling_method = "standard"
        output = chain_sampling.sample(sampler, prompt)
        generated_text = sampler.tokenizer.decode(output.tokens, skip_special_tokens=True)
        
    elif token_sampling is not None:
        # Use token sampling method (Power Sampling) standalone
        output = token_sampling.sample(sampler, prompt)
        generated_text = sampler.tokenizer.decode(output.tokens, skip_special_tokens=True)
        
    else:
        # Default: Standard autoregressive sampling
        output = sampler.sample(prompt)
        generated_text = sampler.tokenizer.decode(output.tokens, skip_special_tokens=True)

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
    return {"object": "list", "data": [{"id": sampler.model, "object": "model", "created": 0, "owned_by": "custom"}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
