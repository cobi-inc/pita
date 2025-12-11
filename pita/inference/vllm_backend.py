# vLLM Libraries
from vllm import LLM, SamplingParams

# vLLM Custom Logit Processor Library
from pita.inference.vllm_logits_processor import LogitsLoggingProcessor

# Memory Libraries
import redis

# Utilities
import numpy as np
from pita.utils.constants import REDIS_HOST, REDIS_PORT

# Take in the context (string) and max_new_tokens (int)
# Returns the generated tokens. the chosen token logprobs, and all the logprobs as lists to the user
def sample(
        self, 
        context: str | list[str], # The input context string to generate from
        max_new_tokens: int | list[int], # The maximum number of new tokens to generate
        **kwargs # Additional keyword arguments passed to the vLLM generate function
    ) -> tuple[
            list[int] | list[list[int]], 
            list[float] | list[list[float]], 
            list[float] | list[list[float]], 
            list[float] | list[list[float]], 
            list[float] | list[list[float]]
        ]:
        
    # Update the max tokens if needed
    if(self.sampling_params.engine_params.max_tokens != max_new_tokens):
        self.sampling_params.engine_params.max_tokens = max_new_tokens
    
    # Generate a new response from the LLM
    llm_output = self.llm.generate(
        context, 
        sampling_params=self.sampling_params.engine_params, 
        **kwargs
    )
    
    print(llm_output)

    # Get the generated tokens
    tokens = llm_output[0].outputs[0].token_ids

    # Extract all logits/logprobs for each position (list of lists to handle variable lengths)
    logits_logprobs = [[obj.logprob for obj in position_dict.values()] for position_dict in llm_output[0].outputs[0].logprobs]
    
    if(self.sampling_params.logits_per_token is not None):
        top_k_logits = logits_logprobs
        top_k_logprobs = None
    else:
        top_k_logprobs = logits_logprobs
        top_k_logits = None

    # Get the Normalization Constants from Redis
    unprocessed_normalization_constant = []
    temp_processed_normalization_constant = []
    if (hasattr(self.sampling_params.engine_params, 'extra_args') and hasattr(self.sampling_params.engine_params.extra_args, 'req_id')  ):
        # Set the req_id used to store the normalization constants in Redis
        req_id = self.sampling_params.engine_params.extra_args["req_id"]

        # Create a local Redis client to retrieve the normalization constants
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        
        # Retrieve the normalization constants from Redis using the req_id
        normalization_terms = redis_client.lrange(req_id, 0, -1)
        
        # Clean up the Redis key after retrieval
        redis_client.delete(req_id)

        # Parse the normalization terms (format: "norm_val,norm_temp_val,max_val")
        for term in normalization_terms:
            parts = term.split(',')
            unprocessed_normalization_constant.append(float(parts[0]))
            temp_processed_normalization_constant.append(float(parts[1]))

    # Returns the generated token_ids, the chosen token logit/logprob, the top_k logits/logprobs, and the normalization constants
    return tokens, top_k_logits, top_k_logprobs, unprocessed_normalization_constant, temp_processed_normalization_constant


# Create the LLM object given the model name and engine parameters
def create_LLM_object(
        model_name,
        model_type=None, # TO DO: Add support for both safetensors (default) and GGUF in vLLM
        dtype="auto", 
        gpu_memory_utilization=0.85, 
        max_model_len=2048, 
        max_logprobs=1000, # Controls how many logprobs or logits are stored for each token
        logits=True, 
        **kwargs
    ):

    if(logits):
        # User wants unprocessed logits output
        logprobs_mode = 'raw_logits'
        # Enable the Redis logging logits processor by adding it to the kwargs
        kwargs["logits_processors"] = [LogitsLoggingProcessor]

    else:
        # Default to processed logprobs if the user does not want logits
        logprobs_mode = 'processed_logprobs'
        
    # Initialize VLLM locally for performance (as done in power_sample.py main)
    llm = LLM(model=model_name,
              dtype=dtype,
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len,
              max_logprobs=max_logprobs, # Controls how many logprobs or logits are stored for each token
              logprobs_mode=logprobs_mode,
              **kwargs)

    return llm

def create_vllm_engine_params():
    # Create the vLLM SamplingParams object from the common Sampling_Params
    vllm_params = SamplingParams()
    return vllm_params

# Check that the vLLM engine can support power sampling with the given configuration
def check_vllm_power_sampling_compatibility(sampler):
    # Make sure the user has actually set logits_per_token
    if(sampler.sampling_params.logits_per_token is None):
        raise ValueError("LLM engine logits_per_token must be set to enable power sampling.")
    
    # Make sure top_k matches logits_per_token to make sure that the inference engine is actually using only the logits requested
    if(sampler.sampling_params.top_k != sampler.sampling_params.logits_per_token):
        print("Warning: The sampler top_k does not match the LLM engine logits_per_token setting. This may lead to unexpected behavior during power sampling.")
        print("Automatically setting the LLM engine top_k to match the logits_per_token.")
        sampler.sampling_params.top_k = sampler.sampling_params.logits_per_token

    # For vLLM, make sure that logprobs_mode is set to 'raw_logits' to get unprocessed logits
    if(sampler.llm.llm_engine.model_config.logprobs_mode != 'raw_logits'):
        raise ValueError(
            f"vLLM engine logprobs_mode must be set to 'raw_logits' to enable power sampling."
            f"\nvLLM engine logprobs_mode is set to {sampler.llm.llm_engine.model_config.logprobs_mode}." 
            f"\nThis is done by setting logits=True when creating the LLM object."
                        )
