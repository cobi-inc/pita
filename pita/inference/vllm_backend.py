# vLLM Libraries
from vllm import LLM, SamplingParams

# vLLM Custom Logit Processor Library
from pita.inference.vllm_logits_processor import LogitsLoggingProcessor

# Memory Libraries
import redis

# Utilities
import numpy as np
from pita.utils.constants import REDIS_HOST, REDIS_PORT
from pita.utils.redis_manager import RedisManager

def sample(
        self, 
        context: str | list[str], 
        max_new_tokens: int | list[int], 
        **kwargs 
    ) -> tuple[
            list[int] | list[list[int]], 
            list[float] | list[list[float]], 
            list[float] | list[list[float]], 
            list[float] | list[list[float]], 
            list[float] | list[list[float]]  
        ]:
    """
    Generate text from the given context using the vLLM engine.

    Args:
        context (str | list[str]): The input context string to generate from.
        max_new_tokens (int | list[int]): The maximum number of new tokens to generate.
        **kwargs: Additional keyword arguments passed to the vLLM generate function.

    Returns:
        tokens: list[int] | list[list[int]]: The generated token IDs.
        top_k_logits: list[float] | list[list[float]] | None: The top_k logits (if logits_per_token is set).
        top_k_logprobs: list[float] | list[list[float]] | None: The top_k logprobs (if logprobs is set).
        unprocessed_log_normalization_constant: list[float] | list[list[float]]: The log(Normalization Constants - Unprocessed) for each token.
        temp_processed_log_normalization_constant: list[float] | list[list[float]]: The log(Normalization Constants - Temperature Processed) for each token.
        entropy: list[float] | list[list[float]]: The entropy for each token.
    """
        
    # Update the max tokens if needed
    if(self.sampling_params.engine_params.max_tokens != max_new_tokens):
        self.sampling_params.engine_params.max_tokens = max_new_tokens
    
    # Set the sampling_params logprobs to logits_per_token if needed. Always assume the users when using vLLM wants logits if logits_per_token is sets
    if(self.sampling_params.logits_per_token is not None and self.sampling_params.logprobs is not None):
        if(self.sampling_params.logits_per_token > self.sampling_params.logprobs ):
            self.sampling_params.logprobs = self.sampling_params.logits_per_token

    # Generate a new response from the LLM
    llm_output = self.llm.generate(
        context, 
        sampling_params=self.sampling_params.engine_params, 
        **kwargs
    )
    
    # Get the generated tokens
    tokens = llm_output[0].outputs[0].token_ids

    # Extract all logits/logprobs for each position (list of lists to handle variable lengths)
    # Create a 2D array of Nones to hold the logits/logprobs
    logits_logprobs = np.full((len(llm_output[0].outputs[0].logprobs), 1 + (self.sampling_params.logits_per_token if self.sampling_params.logits_per_token is not None else self.sampling_params.logprobs)), None)    
    for token_idx in range(len(tokens)):
        for logit_idx, values in enumerate(llm_output[0].outputs[0].logprobs[token_idx].values()):
            logits_logprobs[token_idx][logit_idx] = values.logprob

    if(self.sampling_params.logits_per_token is not None):
        top_k_logits = logits_logprobs
        top_k_logprobs = None
    else:
        top_k_logprobs = logits_logprobs
        top_k_logits = None

    # Get the Normalization Constants from Redis
    unprocessed_log_normalization_constant = []
    temp_processed_log_normalization_constant = []
    entropy = []
    if (hasattr(self.sampling_params.engine_params, 'extra_args') and 'req_id' in self.sampling_params.engine_params.extra_args):        
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
            unprocessed_log_normalization_constant.append(float(parts[0]))
            temp_processed_log_normalization_constant.append(float(parts[1]))
            entropy.append(float(parts[2]))

    # Returns the generated token_ids, the chosen token logit/logprob, the top_k logits/logprobs, and the normalization constants
    return tokens, top_k_logits, top_k_logprobs, unprocessed_log_normalization_constant, temp_processed_log_normalization_constant, entropy

def create_LLM_object(
        model_name,
        model_type=None, 
        dtype="auto", 
        gpu_memory_utilization=0.85, 
        max_model_len=2048, 
        max_logprobs=1000, 
        logits=True, 
        logits_processor=False,
        **kwargs
    ):
    """
    Create the LLM object given the model name and engine parameters.

    Args:
        model_name (str): The name of the model to load.
        model_type (str, optional): The type of model (e.g., 'safetensors', 'gguf'). Defaults to None.
        dtype (str, optional): The data type to use. Defaults to "auto".
        gpu_memory_utilization (float, optional): The fraction of GPU memory to use. Defaults to 0.85.
        max_model_len (int, optional): The maximum length of the model context. Defaults to 2048.
        max_logprobs (int, optional): Controls how many logprobs or logits are stored for each token. Defaults to 1000.
        logits (bool, optional): Whether to output unprocessed logits. Defaults to True.
        logits_processor (bool, optional): Whether to enable the Redis logging logits processor. Defaults to False.
        **kwargs: Additional keyword arguments passed to the LLM constructor.

    Returns:
        LLM: The initialized vLLM LLM object.
    """

    if(logits):
        # User wants unprocessed logits output
        logprobs_mode = 'raw_logits'
    else:
        # Default to processed logprobs if the user does not want logits
        logprobs_mode = 'processed_logprobs'

    if(logits_processor):
        # Enable the Redis logging logits processor by adding it to the kwargs
        kwargs["logits_processors"] = [LogitsLoggingProcessor]
        RedisManager.start()
        print("Added LogitsLoggingProcessor to the vLLM LLM object for logging logits.")
    else:
        print("LogitsLoggingProcessor not enabled. Logits will not be logged.")

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
    """
    Create the vLLM SamplingParams object from the common Sampling_Params.

    Returns:
        SamplingParams: A new instance of vLLM SamplingParams.
    """
    # Create the vLLM SamplingParams object from the common Sampling_Params
    vllm_params = SamplingParams()
    return vllm_params

def check_vllm_power_sampling_compatibility(sampler):
    """
    Check that the vLLM engine can support power sampling with the given configuration.

    Args:
        sampler: The sampler object containing sampling parameters and the LLM engine.

    Raises:
        ValueError: If logits_per_token is not set.
        ValueError: If vLLM engine logprobs_mode is not 'raw_logits'.
        ValueError: If 'req_id' is not in extra_args.
    """
    # Make sure the user has actually set logits_per_token
    if(sampler.sampling_params.logits_per_token is None):
        raise ValueError("LLM engine logits_per_token must be set to enable power sampling.")
    
    # For vLLM, make sure that logprobs_mode is set to 'raw_logits' to get unprocessed logits
    if(sampler.llm.llm_engine.model_config.logprobs_mode != 'raw_logits'):
        raise ValueError(
            f"vLLM engine logprobs_mode must be set to 'raw_logits' to enable power sampling."
            f"\nvLLM engine logprobs_mode is set to {sampler.llm.llm_engine.model_config.logprobs_mode}." 
            f"\nThis is done by setting logits=True when creating the LLM object."
                        )
    # Print all the extra_args of the vLLM SamplingParams
    print("vLLM SamplingParams extra_args:", sampler.sampling_params.engine_params.extra_args)  

    # Make sure the user has enabled the logits processor
    if('req_id' not in sampler.sampling_params.engine_params.extra_args):
        raise ValueError("req_id must be set to use power sampling with vLLM.")
    
    # Set the normalization constant in the extra_args of the vLLM SamplingParams to True 
    sampler.sampling_params.engine_params.extra_args["normalization_constants"] = True
    print("Enabled normalization constants in vLLM SamplingParams for power sampling.")