"""
TensorRT-LLM Backend for PITA inference framework.

This module provides functions to create and use TensorRT-LLM for text generation,
following the same pattern as vllm_backend.py and llama_cpp_backend.py.
"""

from tensorrt_llm import LLM, SamplingParams

# Custom Libraries
from pita.inference.LLM_backend import Output
from pita.inference.tensorRT_logits_processor import create_logits_processor


def sample(
        self, 
        context: str | list[str], 
        **kwargs 
    ) -> Output:
    """
    Generate text from the given context using the TensorRT-LLM engine.

    Args:
        self: The AutoregressiveSampler instance.
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
    """
    # Determine if we need normalization constants or entropy
    calculate_normalization = getattr(self.sampling_params, 'enable_normalization_constants', False)
    calculate_entropy = getattr(self.sampling_params, 'enable_entropy', False)
    
    # Create a fresh logits processor for this sample call
    logits_processor = create_logits_processor(
        temperature=self.sampling_params.temperature,
        calculate_normalization=calculate_normalization,
        calculate_entropy=calculate_entropy
    )

    # Check if context is a list of strings or a single string
    if isinstance(context, list):
        context_list_len = len(context)
    else:
        context_list_len = 1
        context = [context]  # Normalize to list for uniform handling
    
    all_outputs = []
    
    for context_input in context:
        # Reset the logits processor for each context
        logits_processor.reset()
        
        # Create TensorRT-LLM SamplingParams
        sampling_params = SamplingParams(
            max_tokens=self.sampling_params.max_tokens,
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            top_k=self.sampling_params.top_k if self.sampling_params.top_k > 0 else None,
            seed=self.sampling_params.seed,
        )
        
        # Add logprobs if requested
        if self.sampling_params.logprobs_per_token and self.sampling_params.logprobs_per_token > 0:
            sampling_params.logprobs = self.sampling_params.logprobs_per_token
        
        # Generate with the logits processor
        llm_output = self.llm.generate(
            context_input, 
            sampling_params=sampling_params,
            logits_processor=logits_processor,
            **kwargs
        )
        
        # Extract tokens from output
        tokens = list(llm_output.outputs[0].token_ids)
        n_completion = len(tokens)
        
        # Get normalization constants from the logits processor
        unprocessed_log_normalization_constant = logits_processor.log_norm_constants
        temp_processed_log_normalization_constant = logits_processor.log_norm_constants_temp_scaled
        entropy = logits_processor.entropy
        
        # Extract logprobs if available
        logprobs_per_token = self.sampling_params.logprobs_per_token or 0
        logits_per_token = self.sampling_params.logits_per_token or 0
        
        top_k_logits = []
        top_k_logprobs = []
        
        if hasattr(llm_output.outputs[0], 'logprobs') and llm_output.outputs[0].logprobs:
            # Extract logprobs from output
            for token_logprobs in llm_output.outputs[0].logprobs:
                if token_logprobs:
                    # Get top-k logprobs
                    sorted_logprobs = sorted(token_logprobs.items(), key=lambda x: x[1], reverse=True)
                    token_top_logprobs = [lp for _, lp in sorted_logprobs[:logprobs_per_token]]
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
        model_name,
        model_type=None, 
        dtype="auto", 
        gpu_memory_utilization=0.85, 
        max_model_len=2048, 
        max_logprobs=None,
        logits_processor=False,
        **kwargs
    ):
    """
    Create the LLM object given the model name and engine parameters.

    Args:
        model_name (str): The name of the model to load (HuggingFace model name or path).
        model_type (str, optional): The type of model. Defaults to None.
        dtype (str, optional): The data type to use. Defaults to "auto".
        gpu_memory_utilization (float, optional): The fraction of GPU memory to use. Defaults to 0.85.
        max_model_len (int, optional): The maximum context length. Defaults to 2048.
        max_logprobs (int, optional): Controls how many logprobs are stored for each token. Defaults to None.
        logits_processor (bool, optional): Whether logits processing is enabled. Defaults to False.
        **kwargs: Additional keyword arguments passed to the LLM constructor.

    Returns:
        LLM: The initialized TensorRT-LLM LLM object.
    """
    # TensorRT-LLM LLM class handles model loading and optimization
    llm = LLM(
        model=model_name,
        **kwargs
    )
    
    if logits_processor:
        print("TensorRT-LLM LogitsProcessor enabled. Normalization constants and entropy will be calculated per-request.")
    
    print("--- TensorRT-LLM Model Initialization Complete. ---")
    
    return llm


def create_tensorrt_engine_params():
    """
    Create the TensorRT-LLM SamplingParams object.

    Returns:
        SamplingParams: A new instance of TensorRT-LLM SamplingParams.
    """
    return SamplingParams()


def check_token_metric_compatibility(sampler, token_metric: str):
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
        if sampler.sampling_params.logits_per_token is None or sampler.sampling_params.logits_per_token < 1:
            raise ValueError(
                "logits_per_token must be set to at least 1 to use 'logprobs' token metric with TensorRT-LLM backend."
            )
        # Enable normalization constants for logprobs calculation
        sampler.sampling_params.enable_normalization_constants = True
        print("Enabled normalization constants in sampling params for logprobs metric.")
        
    elif token_metric == "power_distribution":
        # power_distribution requires normalization constants
        if sampler.sampling_params.logits_per_token is None or sampler.sampling_params.logits_per_token < 1:
            raise ValueError(
                "logits_per_token must be set to at least 1 to use 'power_distribution' token metric with TensorRT-LLM backend."
            )
        # Enable normalization constants
        sampler.sampling_params.enable_normalization_constants = True
        print("Enabled normalization constants in sampling params for power_distribution metric.")
        
    elif token_metric == "entropy":
        # entropy requires the entropy calculation to be enabled
        if sampler.sampling_params.logits_per_token is None or sampler.sampling_params.logits_per_token < 1:
            raise ValueError(
                "logits_per_token must be set to at least 1 to use 'entropy' token metric with TensorRT-LLM backend."
            )
        # Enable entropy calculation
        sampler.sampling_params.enable_entropy = True
        print("Enabled entropy calculation in sampling params for entropy metric.")
        
    elif token_metric == "likelihood_confidence":
        # likelihood_confidence requires logprobs and entropy
        if sampler.sampling_params.logits_per_token is None or sampler.sampling_params.logits_per_token < 1:
            raise ValueError(
                "logits_per_token must be set to at least 1 to use 'likelihood_confidence' token metric with TensorRT-LLM backend."
            )
        sampler.sampling_params.enable_normalization_constants = True
        sampler.sampling_params.enable_entropy = True
        print("Enabled normalization constants and entropy in sampling params for likelihood_confidence metric.")
    else:
        raise ValueError(f"Unknown token metric: {token_metric}")