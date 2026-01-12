"""
TensorRT-LLM Backend for PITA inference framework.

This module provides functions to create and use TensorRT-LLM for text generation,
following the same pattern as vllm_backend.py and llama_cpp_backend.py.
"""

from tensorrt_llm import LLM, SamplingParams
from typing import Any

# Custom Libraries
from pita.inference.LLM_backend import Output
from pita.inference.tensorRT_logits_processor import create_logits_processor

# Standard Libraries
import uuid
import warnings
import redis

from pita.utils.constants import REDIS_HOST, REDIS_PORT


def sample(
        self,
        context: str | list[str],
        **kwargs: Any
    ) -> Output:
    """
    Generate text from the given context using the TensorRT-LLM engine.

    Args:
        context (str | list[str]): The input context string to generate from.
        **kwargs: Additional keyword arguments passed to the TensorRT-LLM generate function.

    Returns:
        Output: An Output object containing:
            - tokens: The generated token IDs.
            - top_k_logits: The top_k logits (if logits_per_token is set).
            - top_k_logprobs: The top_k logprobs (if logprobs_per_token is set).
            - unprocessed_log_normalization_constant: The log normalization constants for each token.
            - temp_processed_log_normalization_constant: The temperature-scaled log normalization constants.
            - entropy: The entropy for each token.

        See the :class:`pita.inference.LLM_backend.Output` class documentation
        for a complete description of the fields and their semantics.
    """
    # Determine if we need normalization constants or entropy
    calculate_normalization = getattr(self.sampling_params, 'enable_normalization_constants', False)
    calculate_entropy = getattr(self.sampling_params, 'enable_entropy', False)
    
    # Check if context is a list of strings or a single string
    if isinstance(context, list):
        context_list_len = len(context)
    else:
        context_list_len = 1
        context = [context]  # Normalize to list for uniform handling
    
    all_outputs = []
    
    for context_input in context:
        # Generate unique request ID for Redis IPC
        req_id = f"tensorrt_{uuid.uuid4().hex}"
        
        # Create logits processor if normalization or entropy is needed
        if calculate_normalization or calculate_entropy:
            logits_processor = create_logits_processor(
                req_id=req_id,
                temperature=self.sampling_params.temperature,
                calculate_normalization=calculate_normalization,
                calculate_entropy=calculate_entropy
            )
            self.sampling_params.engine_params.logits_processor = logits_processor
        else:
            self.sampling_params.engine_params.logits_processor = None
        
        # Check if logprobs_per_token/logits_per_token is greater than 1. If so, raise an error for unsupported configuration
        if self.sampling_params.logprobs_per_token and self.sampling_params.logprobs_per_token > 1:
            raise ValueError(
                "logprobs_per_token > 1 is not supported for the TensorRT-LLM backend. "
                "Please set logprobs_per_token to 1 or disable it."
            )
        if self.sampling_params.logits_per_token and self.sampling_params.logits_per_token > 1:
            raise ValueError(
                "logits_per_token > 1 is not supported for the TensorRT-LLM backend. "
                "Please set logits_per_token to 1 or disable it."
            )

        # Generate
        llm_output = self.llm.generate(
            context_input, 
            sampling_params=self.sampling_params.engine_params,
            **kwargs
        )
        
        # Extract tokens from output
        tokens = list(llm_output.outputs[0].token_ids)
        n_completion = len(tokens)
        
        # Retrieve normalization constants and entropy from Redis
        unprocessed_log_normalization_constant = []
        temp_processed_log_normalization_constant = []
        entropy = []
        
        if calculate_normalization or calculate_entropy:
            redis_client = None
            try:
                redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
                
                # Set a TTL as a fallback in case cleanup fails (e.g., process crash)
                redis_client.expire(req_id, 60)
                
                # Retrieve all values from Redis using the request ID
                normalization_terms = redis_client.lrange(req_id, 0, -1)
                
                # Parse the normalization terms (format: "norm_val,norm_temp_val,entropy_val")
                for term in normalization_terms:
                    parts = term.split(',')
                    unprocessed_log_normalization_constant.append(float(parts[0]))
                    temp_processed_log_normalization_constant.append(float(parts[1]))
                    entropy.append(float(parts[2]))
            except Exception as e:
                print(f"Warning: Failed to retrieve results from Redis: {e}")
            finally:
                # Always clean up the Redis key, even if an exception occurred
                if redis_client is not None:
                    try:
                        redis_client.delete(req_id)
                    except Exception:
                        pass  # Ignore cleanup errors; TTL will handle expiration

        # Extract logprobs if available
        logprobs_per_token = self.sampling_params.logprobs_per_token 
        logits_per_token = self.sampling_params.logits_per_token 
        
        top_k_logits = []
        top_k_logprobs = []
        
        if hasattr(llm_output.outputs[0], 'logprobs') and llm_output.outputs[0].logprobs:
            # Extract logprobs from output
            for token_logprobs in llm_output.outputs[0].logprobs:
                if token_logprobs:
                    # Get top-k logprobs
                    # Sort by logprob value - token_logprobs is dict of token_id -> Logprob object
                    sorted_logprobs = sorted(
                        token_logprobs.items(), 
                        key=lambda x: getattr(x[1], 'logprob', x[1]) if hasattr(x[1], 'logprob') else x[1], 
                        reverse=True
                    )
                    # Extract the float value from Logprob objects
                    token_top_logprobs = [
                        getattr(lp, 'logprob', lp) if hasattr(lp, 'logprob') else lp 
                        for _, lp in sorted_logprobs[:logprobs_per_token]
                    ]
                    top_k_logprobs.append(token_top_logprobs)
                    
                    # Calculate logits from logprobs (logit = logprob + log_norm_constant)
                    if temp_processed_log_normalization_constant and len(temp_processed_log_normalization_constant) > len(top_k_logits):
                        idx = len(top_k_logits)
                        token_top_logits = [
                            (lp + temp_processed_log_normalization_constant[idx]) * self.sampling_params.temperature 
                            for lp in token_top_logprobs[:logits_per_token]
                        ]
                        top_k_logits.append(token_top_logits)
                    else:
                        top_k_logits.append([])
                else:
                    top_k_logprobs.append([])
                    top_k_logits.append([])
        else:
            # No logprobs available, fill with empty lists
            top_k_logits = [[] for _ in range(n_completion)]
            top_k_logprobs = [[] for _ in range(n_completion)]
        
        # Ensure arrays have consistent length
        while len(unprocessed_log_normalization_constant) < n_completion:
            unprocessed_log_normalization_constant.append(0.0)
        while len(temp_processed_log_normalization_constant) < n_completion:
            temp_processed_log_normalization_constant.append(0.0)
        while len(entropy) < n_completion:
            entropy.append(0.0)
        
        # Trim to n_completion if needed
        unprocessed_log_normalization_constant = unprocessed_log_normalization_constant[:n_completion]
        temp_processed_log_normalization_constant = temp_processed_log_normalization_constant[:n_completion]
        entropy = entropy[:n_completion]
        
        output = Output(
            tokens=tokens,
            top_k_logits=top_k_logits,
            top_k_logprobs=top_k_logprobs,
            unprocessed_log_normalization_constant=unprocessed_log_normalization_constant,
            temp_processed_log_normalization_constant=temp_processed_log_normalization_constant,
            entropy=entropy
        )
        all_outputs.append(output)
    
    # If only one context was provided, return single Output
    if context_list_len == 1:
        return all_outputs[0]
    
    # For multiple contexts, combine into a single Output with lists of lists
    combined = Output(
        tokens=[o.tokens for o in all_outputs],
        top_k_logits=[o.top_k_logits for o in all_outputs],
        top_k_logprobs=[o.top_k_logprobs for o in all_outputs],
        unprocessed_log_normalization_constant=[o.unprocessed_log_normalization_constant for o in all_outputs],
        temp_processed_log_normalization_constant=[o.temp_processed_log_normalization_constant for o in all_outputs],
        entropy=[o.entropy for o in all_outputs]
    )
    return combined


def create_LLM_object(
        model_name: str,
        model_type: str = None,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        max_logprobs: int = None,
        logits_processor: bool = False,
        **kwargs: Any
    ) -> LLM:
    """
    Create the LLM object given the model name and engine parameters.

    Args:
        model_name (str): The name of the model to load (HuggingFace model name or path).
        model_type (str, optional): The type of model. Defaults to None.
        dtype (str, optional): The data type to use. Defaults to "auto".
        gpu_memory_utilization (float, optional): Kept for API compatibility with other backends; ignored by TensorRT-LLM and not passed to the LLM constructor.
        max_model_len (int, optional): The maximum context length. Defaults to 2048.
        max_logprobs (int, optional): Kept for API compatibility with other backends; ignored by TensorRT-LLM and not passed to the LLM constructor.
        logits_processor (bool, optional): Whether logits processing is enabled. Defaults to False.
        **kwargs: Additional keyword arguments passed to the LLM constructor.

    Returns:
        LLM: The initialized TensorRT-LLM LLM object.
    """
    # TensorRT-LLM LLM class handles model loading and optimization
    llm = LLM(
        model=model_name,
        dtype=dtype,
        max_num_tokens=max_model_len,
        trust_remote_code=True,
        **kwargs
    )

    if logits_processor:
        print("TensorRT-LLM LogitsProcessor enabled. Normalization constants and entropy will be calculated per-request.")
    
    print("--- TensorRT-LLM Model Initialization Complete. ---")
    
    return llm


def create_tensorrt_engine_params() -> SamplingParams:
    """
    Create the TensorRT-LLM SamplingParams object.

    Returns:
        SamplingParams: A new instance of TensorRT-LLM SamplingParams.
    """
    return SamplingParams()


def check_token_metric_compatibility(sampler: Any, token_metric: str) -> None:
    """
    Check that the TensorRT-LLM engine can support the given token metric with the given configuration.

    Args:
        sampler: The sampler object containing sampling parameters and the LLM engine.
        token_metric: The token metric to check compatibility for.

    Raises:
        ValueError: If the configuration doesn't support the requested token metric.
    """
    if token_metric == "logprobs":
        # logprobs requires logits_per_token to be set
        if sampler.sampling_params.logprobs_per_token is None or sampler.sampling_params.logprobs_per_token < 1:
            raise ValueError(
                "logprobs_per_token must be set to at least 1 to use 'logprobs' token metric with TensorRT-LLM backend."
            )
        # Enable normalization constants for logprobs calculation
        sampler.sampling_params.enable_normalization_constants = True
        print("Enabled normalization constants in sampling params for logprobs metric.")
        
    elif token_metric == "power_distribution":
        # power_distribution requires normalization constants
        if sampler.sampling_params.logits_per_token is None or sampler.sampling_params.logits_per_token < 1 or sampler.sampling_params.logprobs_per_token is None or sampler.sampling_params.logprobs_per_token < 1:
            raise ValueError(
                "logits_per_token (and logprobs_per_token, which logits_per_token depends on) must be set to at least 1 to use 'power_distribution' token metric with TensorRT-LLM backend."
            )
        # Enable normalization constants
        sampler.sampling_params.enable_normalization_constants = True
        print("Enabled normalization constants in sampling params for power_distribution metric.")
        
    elif token_metric == "entropy":
        # Enable entropy calculation
        sampler.sampling_params.enable_entropy = True
        print("Enabled entropy calculation in sampling params for entropy metric.")
        
    elif token_metric == "likelihood_confidence":
        # likelihood_confidence requires logprobs and entropy
        if sampler.sampling_params.logprobs_per_token is None or sampler.sampling_params.logprobs_per_token < 1:
            raise ValueError(
                "logprobs_per_token must be set to at least 1 to use 'likelihood_confidence' token metric with TensorRT-LLM backend."
            )
        sampler.sampling_params.enable_normalization_constants = True
        sampler.sampling_params.enable_entropy = True
        print("Enabled normalization constants and entropy in sampling params for likelihood_confidence metric.")
    else:
        raise ValueError(f"Unknown token metric: {token_metric}")