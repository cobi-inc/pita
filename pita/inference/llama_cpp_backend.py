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
        # Reset the LLM state for a fresh start with each context
        self.llm.reset()
        logits_processor.reset()
        
        # Use generate() to extract the token_ids instead of create_completion
        if isinstance(context_input, str):
            prompt_tokens = self.llm.tokenize(context_input.encode('utf-8'))
        else:
            prompt_tokens = context_input

        tokens = []
        top_k_logits = []
        logits_per_token = self.sampling_params.logits_per_token or 0
        
        if len(prompt_tokens) > 0:
            self.llm.eval([prompt_tokens[0]])
        
        # Generation loop
        generator = self.llm.generate(
            prompt_tokens[1:] if len(prompt_tokens) > 0 else [],
            top_k=self.sampling_params.top_k,
            top_p=self.sampling_params.top_p,
            min_p=self.sampling_params.min_p,
            temp=self.sampling_params.temperature,
            repeat_penalty=self.sampling_params.repetition_penalty,
            frequency_penalty=self.sampling_params.frequency_penalty,
            presence_penalty=self.sampling_params.presence_penalty,
            logits_processor=logits_processor_list,
            **kwargs
        )

        for token in generator:
            # For each token, self.llm.scores[self.llm.n_tokens - 1] contains the logits
            # that were used to sample it.
            current_logits = self.llm.scores[self.llm.n_tokens - 1, :]
            
            if logits_per_token > 0:
                # Extract logits for the current step.
                # We always place the chosen token's logit first, then fill with the
                # highest remaining logits until we reach logits_per_token elements
                # or run out of logits.
                sorted_indices = np.argsort(current_logits)[::-1]

                # Ensure the chosen token logit is first as requested
                step_logits = [float(current_logits[token])]
                for idx in sorted_indices:
                    if idx == token:
                        continue
                    if len(step_logits) >= logits_per_token:
                        break
                    step_logits.append(float(current_logits[idx]))
                top_k_logits.append(step_logits)
            
            tokens.append(int(token))
            
            # Check stopping criteria
            if len(tokens) >= self.sampling_params.max_tokens:
                break
            if token == self.llm.token_eos():
                break
            if self.sampling_params.stop_token_ids and token in self.sampling_params.stop_token_ids:
                break

        # Find the token count from the token_ids
        token_count = len(tokens)

        # We only trim data from the logits processor as it is the only source that is guaranteed to have the wrong length
        unprocessed_log_normalization_constant = logits_processor.log_norm_constants[:token_count]
        temp_processed_log_normalization_constant = logits_processor.log_norm_constants_temp_scaled[:token_count]
        entropy = logits_processor.entropy[:token_count]

        # Use the temp_processed_log_normalization_constant to calculate the logprobs
        top_k_logprobs = []
        logprobs_per_token = self.sampling_params.logprobs_per_token or 0
        if logprobs_per_token > 0 and top_k_logits:
            for i in range(token_count):
                logits_row = np.array(top_k_logits[i])
                temp_norm = temp_processed_log_normalization_constant[i]
                # logprob = (logit / temp) - logsumexp(logits / temp)
                row_logprobs = (logits_row / self.sampling_params.temperature) - temp_norm
                # Slice to the requested logprobs amount
                top_k_logprobs.append(row_logprobs[:logprobs_per_token].tolist())
        else:
            top_k_logprobs = [[]] * token_count

        if not top_k_logits:
            top_k_logits = [[]] * token_count
        
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