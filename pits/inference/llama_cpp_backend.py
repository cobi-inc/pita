from llama_cpp import Llama
import numpy as np

# Custom Libraries
from pits.utils.system_utils import get_total_vram, get_gpu_vram_usage_mb

# Take in the context (string) and max_new_tokens (int)
# Returns arrays of the generated token_ids, the chosen token logprobs, and all the logprobs as lists to the user
def sample(
        self, 
        context, # The input context string to generate from
        max_new_tokens, # The maximum number of new tokens to generate
        **kwargs # Additional keyword arguments passed to the backend Llama create_completion function
    ):

    # Update the max tokens if needed
    if(self.sampling_params.max_tokens != max_new_tokens):
        self.sampling_params.max_tokens = max_new_tokens

    # Generate a new response from the LLM
    llm_output = self.llm.create_completion(
        prompt = context,
        max_tokens = self.sampling_params.max_tokens,
        temperature = self.sampling_params.temperature,
        top_p = self.sampling_params.top_p,
        min_p = self.sampling_params.min_p,
        stop = self.sampling_params.stop,
        frequency_penalty = self.sampling_params.frequency_penalty,
        presence_penalty = self.sampling_params.presence_penalty,
        repeat_penalty = self.sampling_params.repetition_penalty,
        top_k = self.sampling_params.top_k,
        seed = self.sampling_params.seed,
        logprobs = self.sampling_params.logprobs,
        **kwargs
    )
    
    # We need to know where the prompt ends and the generation begins.
    n_prompt = llm_output['usage']['prompt_tokens']
    n_total = llm_output['usage']['total_tokens']

    # Reconstruct an array of all generated tokens
    # self.llm.input_ids doesn't store the last generated token, so we need to get it from llm_output
    tokens = np.array(self.tokenizer.encode(llm_output['choices'][0]['text']), dtype=np.int32)

    # Use partition to find top self.sampling_params.logits_per_token indices
    # scores logits are stored in self.llm.scores. The previous index's scores correspond to the next token prediction token[i] is predicted by scores[i-1]
    top_k_logits = -np.partition(-self.llm.scores[n_prompt-1:n_total-1], self.sampling_params.logits_per_token, axis=1)[:, :self.sampling_params.logits_per_token]
    
    # Use advanced indexing to extract the actual logit values for these indices
    # shape: (n_completion, TOP_K)
    chosen_token_logit = self.llm.scores[np.arange(n_prompt-1, n_total-1), tokens]

    # Returns the generated array token_ids, the chosen token logit/logprob, and the top_k logits/logprobs where k = self.sampling_params.logits_per_token
    return tokens, chosen_token_logit, top_k_logits

# Create the LLM object given the model name and engine parameters
def create_LLM_object(
        model_name,  # Model name that will be used to load the LLM (Hugging Face only currently)
        model_type=None, # GGUF models only supported currently by LLaMA.cpp. Will throw error if safetensors is selected
        dtype="auto", 
        gpu_memory_utilization=0.85, 
        max_model_len=2048, 
        max_logprobs=None, 
        logits=True, 
        **kwargs):
    
    if(model_type == "gguf"):
        # File the correct dtype in the GGUF Hugging Face Repo
        if dtype == "auto":
            kwargs['filename'] = "*f16*"
        else:
            kwargs['filename'] = f"*{dtype}*"
    elif(model_type == "safetensors"):
        raise ValueError("safetensors model type is not currently supported in llama.cpp backend. Please use gguf model type.")

    # Check to see if the user wants to use the GPU
    if(gpu_memory_utilization > 0 and 'n_gpu_layers' not in kwargs):
        kwargs['n_gpu_layers'] = -1  # Use as many GPU layers as possible

    # Get the System VRAM
    total_vram_mb = get_total_vram()

    # Get the VRAM usage before loading the model
    vram_before = get_gpu_vram_usage_mb() or 0

    # Initialize LLaMA.cpp locally for performance (as done in power_sample.py main)
    if(model_type == "gguf"):
        # Loading the GGUF model with from_pretrained
        llm = Llama.from_pretrained(
            repo_id=model_name,
            n_ctx=max_model_len, # Text Context length
            logits_all=logits, # Whether to output logits
            **kwargs
        )
    else:
        raise ValueError(f"{model_type} is an unsupported model type. Supported types are 'safetensors' and 'gguf'.")
    
    # Get the VRAM used to load the model
    vram_after = get_gpu_vram_usage_mb() or 0
    vram_mb = vram_after - vram_before

    if(vram_mb < 1):
        print("Warning: Could not extract VRAM usage from llama.cpp logs. Model loaded into CPU RAM. Proceeding without VRAM check.")    
    else:
        if(int(vram_mb)/int(total_vram_mb) > gpu_memory_utilization):
            raise ValueError(
                "VRAM usage exceeds the specified GPU memory utilization threshold.\n"
                "Options to Reduce VRAM:\n"
                "1. Reduce the context size (n_ctx parameter)\n"
                "2. Turn off GPU KV-caching with kwarg: offload_kqv = True\n"
                "3. Load only 'N' layers to the GPU kwarg: n_gpu_layers = N\n"
            )
        try:
            total_vram_int = int(total_vram_mb)
            vram_mb_int = int(vram_mb)
        except (ValueError, TypeError):
            print(f"Warning: Could not extract total VRAM value ('{total_vram_mb}'). Skipping VRAM utilization check.")
        else:
            if(vram_mb_int / total_vram_int > gpu_memory_utilization):
                raise ValueError("VRAM usage exceeds the specified GPU memory utilization threshold. \n Options to Reduce VRAM:\n 1. Reduce Context Size:\n 2. turn off GPU KV-caching with kwarg: offload_kqv = True. \n 3. Load only 'N' layers to the GPU kwarg: n_gpu_layers = N \n")
            else:
                print(f"VRAM Usage for Model Load: {vram_mb_int} MiB / {total_vram_int} MiB ({(vram_mb_int/total_vram_int)*100:.2f} %)")
    print("--- Model Initialization Complete. ---")

    # Return created LLM object
    return llm

def check_llama_cpp_power_sampling_compatibility(sampler):
    # Check to make sure the llama engine can output all of the logits needed for power sampling
    if(sampler.llm._logits_all != True):
        raise ValueError("LLM engine logits_all must be set to 'True' to enable power sampling. This is done by setting logits=True when creating the LLM object.")
    
    # Make sure top_k matches logits_per_token to make sure that the inference engine is actually using only the logits requested
    if(sampler.sampling_params.top_k != sampler.sampling_params.logits_per_token):
        print("Warning: The sampler top_k does not match the LLM engine logits_per_token setting. This may lead to unexpected behavior during power sampling.")
        print("Automatically setting the LLM engine logits_per_token to match the sampler top_k.")
        sampler.sampling_params.top_k = sampler.sampling_params.logits_per_token

    # Give a warning that logprobs gives log-probabilities to the user dramatically slowing down inference
    if(sampler.sampling_params.logprobs is not None):
        print("Warning: llama.cpp backend logprobs parameter is set to output log-probabilities which may dramatically slow down inference. It is recommended to set logprobs=None when using power sampling with llama.cpp backend.")