from llama_cpp import Llama
import numpy as np

# Custom Libraries
from pita.utils.system_utils import get_total_vram, get_gpu_vram_usage_mb
from pita.inference.LLM_backend import Output

# Logits Processor
from pita.inference.llama_cpp_logits_processor import create_logits_processor_list


def sample(
        self, 
        context: str | list[str], 
        **kwargs 
    ) -> Output:
    """
    Generate text from the given context using the llama.cpp backend.

    Args:
        context (str | list[str]): The input context string to generate from.
        **kwargs: Additional keyword arguments passed to the underlying llama.cpp generation function.

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
    logits_processor_list, logits_processor = create_logits_processor_list(
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
    
    # For batch processing, we'd need to handle multiple contexts
    # Currently llama.cpp doesn't support true batching, so we process sequentially
    all_outputs = []
    
    for context_input in context:
        # Reset the logits processor for each context
        logits_processor.reset()
        
        # Generate a new response from the LLM
        llm_output = self.llm.create_completion(
            prompt=context_input,
            max_tokens=self.sampling_params.max_tokens,
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            min_p=self.sampling_params.min_p,
            stop=self.sampling_params.stop,
            frequency_penalty=self.sampling_params.frequency_penalty,
            presence_penalty=self.sampling_params.presence_penalty,
            repeat_penalty=self.sampling_params.repetition_penalty,
            top_k=self.sampling_params.top_k,
            seed=self.sampling_params.seed,
            logprobs=self.sampling_params.logprobs_per_token,
            logits_processor=logits_processor_list,
            **kwargs
        )
        
        # We need to know where the prompt ends and the generation begins.
        n_prompt = llm_output['usage']['prompt_tokens']
        n_total = llm_output['usage']['total_tokens']
        n_completion = n_total - n_prompt

        # Reconstruct an array of all generated tokens
        # Note: Re-encoding may produce a different token count than n_completion
        # due to tokenizer differences (e.g., BOS tokens). We use n_completion
        # as the authoritative length since the logits processor was called
        # exactly n_completion times.
        tokens = list(self.tokenizer.encode(llm_output['choices'][0]['text']))
        
        # Validate and adjust token length to match n_completion
        if len(tokens) != n_completion:
            # Adjust tokens to match n_completion length
            if len(tokens) < n_completion:
                # Pad with zeros if too short
                tokens = tokens + [0] * (n_completion - len(tokens))
            else:
                # Truncate if too long
                tokens = tokens[:n_completion]
        
        # Get logits from self.llm.scores if logits_per_token is set
        # scores logits are stored in self.llm.scores
        # The previous index's scores correspond to the next token prediction
        # token[i] is predicted by scores[i-1]
        logits_per_token = self.sampling_params.logits_per_token or 0
        logprobs_per_token = self.sampling_params.logprobs_per_token or 0
        
        if logits_per_token > 0 and hasattr(self.llm, 'scores') and self.llm.scores is not None:
            # Use partition to find top logits_per_token indices
            scores_slice = self.llm.scores[n_prompt-1:n_total-1]
            if len(scores_slice) > 0:
                top_k_logits = -np.partition(-scores_slice, logits_per_token, axis=1)[:, :logits_per_token]
                top_k_logits = top_k_logits.tolist()
            else:
                top_k_logits = [[]] * n_completion
        else:
            top_k_logits = [[]] * n_completion
        
        # Get normalization constants from the logits processor
        # These arrays have exactly n_completion entries
        unprocessed_log_normalization_constant = logits_processor.log_norm_constants
        temp_processed_log_normalization_constant = logits_processor.log_norm_constants_temp_scaled
        entropy = logits_processor.entropy
        
        # Calculate logprobs from logits and normalization constants
        if logprobs_per_token > 0 and top_k_logits and temp_processed_log_normalization_constant:
            top_k_logits_array = np.array(top_k_logits)
            temp_norm_array = np.array(temp_processed_log_normalization_constant)[:, np.newaxis]
            top_k_logprobs = (top_k_logits_array / self.sampling_params.temperature) - temp_norm_array
            top_k_logprobs = top_k_logprobs[:, :logprobs_per_token].tolist()
        else:
            top_k_logprobs = [[]] * n_completion
        
        # Validate that all arrays have consistent length
        assert len(tokens) == n_completion, f"tokens length {len(tokens)} != n_completion {n_completion}"
        assert len(top_k_logits) == n_completion, f"top_k_logits length {len(top_k_logits)} != n_completion {n_completion}"
        assert len(top_k_logprobs) == n_completion, f"top_k_logprobs length {len(top_k_logprobs)} != n_completion {n_completion}"
        assert len(unprocessed_log_normalization_constant) == n_completion, f"unprocessed_log_normalization_constant length {len(unprocessed_log_normalization_constant)} != n_completion {n_completion}"
        assert len(temp_processed_log_normalization_constant) == n_completion, f"temp_processed_log_normalization_constant length {len(temp_processed_log_normalization_constant)} != n_completion {n_completion}"
        assert len(entropy) == n_completion, f"entropy length {len(entropy)} != n_completion {n_completion}"
        
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
    # This matches the vLLM batch behavior
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
        model_name (str): The name of the model to load (Hugging Face repo ID for GGUF models).
        model_type (str, optional): The type of model. Inferred from model_name if not provided.
            Currently only 'gguf' is supported.
        dtype (str, optional): The data type/quantization to use. Defaults to "auto" (f16).
        gpu_memory_utilization (float, optional): The fraction of GPU memory to use. Defaults to 0.85.
        max_model_len (int, optional): The maximum context length. Defaults to 2048.
        max_logprobs (int, optional): Unused for llama.cpp, kept for API compatibility.
        logits_processor (bool, optional): Whether logits processing is enabled. 
            When True, scores are available via llm.scores. Defaults to False.
        **kwargs: Additional keyword arguments passed to the Llama constructor.

    Returns:
        Llama: The initialized llama.cpp Llama object.
        
    Raises:
        ValueError: If model_type is 'safetensors' (not supported) or unsupported.
    """
    # Infer model_type from model_name if not provided
    if model_type is None:
        # Check if model name contains common GGUF indicators
        if 'gguf' in model_name.lower() or model_name.endswith('.gguf'):
            model_type = "gguf"
        else:
            # Default to gguf for llama.cpp
            model_type = "gguf"
            print(f"Warning: model_type not specified, defaulting to 'gguf' for llama.cpp backend.")
    
    if model_type == "gguf":
        # Find the correct dtype in the GGUF Hugging Face Repo
        if dtype == "auto":
            kwargs['filename'] = "*f16*"
        else:
            kwargs['filename'] = f"*{dtype}*"
    elif model_type == "safetensors":
        raise ValueError("safetensors model type is not currently supported in llama.cpp backend. Please use gguf model type.")
    else:
        raise ValueError(f"{model_type} is an unsupported model type. Supported types are 'gguf'.")

    # Check to see if the user wants to use the GPU
    if gpu_memory_utilization > 0 and 'n_gpu_layers' not in kwargs:
        kwargs['n_gpu_layers'] = -1  # Use as many GPU layers as possible

    # Get the System VRAM
    total_vram_mb = get_total_vram()

    # Get the VRAM usage before loading the model
    vram_before = get_gpu_vram_usage_mb() or 0

    # Determine if we need logits_all
    # With logits processor, we don't need logits_all=True as the processor captures what we need
    # However, for compatibility with direct scores access, we may still want it
    logits_all = logits_processor

    # Initialize LLaMA.cpp locally
    llm = Llama.from_pretrained(
        repo_id=model_name,
        n_ctx=max_model_len,
        logits_all=logits_all,
        **kwargs
    )

    # Get the VRAM used to load the model
    vram_after = get_gpu_vram_usage_mb() or 0
    vram_mb = vram_after - vram_before

    if vram_mb < 1:
        print("Warning: Could not extract VRAM usage from llama.cpp logs. Model may be loaded into CPU RAM. Proceeding without VRAM check.")    
    else:
        try:
            total_vram_int = int(total_vram_mb)
            vram_mb_int = int(vram_mb)
        except (ValueError, TypeError):
            print(f"Warning: Could not extract total VRAM value ('{total_vram_mb}'). Skipping VRAM utilization check.")
        else:
            if vram_mb_int / total_vram_int > gpu_memory_utilization:
                raise ValueError(
                    "VRAM usage exceeds the specified GPU memory utilization threshold.\n"
                    "Options to Reduce VRAM:\n"
                    "1. Reduce the context size (n_ctx parameter)\n"
                    "2. Turn off GPU KV-caching with kwarg: offload_kqv = True\n"
                    "3. Load only 'N' layers to the GPU kwarg: n_gpu_layers = N\n"
                )
            else:
                print(f"VRAM Usage for Model Load: {vram_mb_int} MiB / {total_vram_int} MiB ({(vram_mb_int/total_vram_int)*100:.2f} %)")
    
    print("--- Model Initialization Complete. ---")

    # Return created LLM object
    return llm


def check_token_metric_compatibility(sampler, token_metric: str):
    """
    Check that the llama.cpp engine can support the given token metric with the given configuration.

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
                "logits_per_token must be set to at least 1 to use 'logprobs' token metric with llama.cpp backend."
            )
        # Enable normalization constants for logprobs calculation
        sampler.sampling_params.enable_normalization_constants = True
        print("Enabled normalization constants in sampling params for logprobs metric.")
        
    elif token_metric == "power_distribution":
        # power_distribution requires normalization constants
        if sampler.sampling_params.logits_per_token is None or sampler.sampling_params.logits_per_token < 1:
            raise ValueError(
                "logits_per_token must be set to at least 1 to use 'power_distribution' token metric with llama.cpp backend."
            )
        # Enable normalization constants
        sampler.sampling_params.enable_normalization_constants = True
        print("Enabled normalization constants in sampling params for power_distribution metric.")
        
    elif token_metric == "entropy":
        # entropy requires the entropy calculation to be enabled
        if sampler.sampling_params.logits_per_token is None or sampler.sampling_params.logits_per_token < 1:
            raise ValueError(
                "logits_per_token must be set to at least 1 to use 'entropy' token metric with llama.cpp backend."
            )
        # Enable entropy calculation
        sampler.sampling_params.enable_entropy = True
        print("Enabled entropy calculation in sampling params for entropy metric.")
        
    elif token_metric == "likelihood_confidence":
        # likelihood_confidence requires logprobs
        if sampler.sampling_params.logits_per_token is None or sampler.sampling_params.logits_per_token < 1:
            raise ValueError(
                "logits_per_token must be set to at least 1 to use 'likelihood_confidence' token metric with llama.cpp backend."
            )
        sampler.sampling_params.enable_normalization_constants = True
        print("Enabled normalization constants in sampling params for likelihood_confidence metric.")
    else:
        raise ValueError(f"Unknown token metric: {token_metric}")