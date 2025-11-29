# vLLM Libraries
from vllm import LLM, SamplingParams

# Utilities
import numpy as np

# Take in the context (string) and max_new_tokens (int)
# Returns the generated tokens. the chosen token logprobs, and all the logprobs as lists to the user
def sample(
        self, 
        context, # The input context string to generate from
        max_new_tokens, # The maximum number of new tokens to generate
        **kwargs # Additional keyword arguments passed to the vLLM generate function
    ):

    # Update the max tokens if needed
    if(self.sampling_params.engine_params.max_tokens != max_new_tokens):
        self.sampling_params.engine_params.max_tokens = max_new_tokens

    # Generate a new response from the LLM
    llm_output = self.llm.generate(
        context, 
        sampling_params=self.sampling_params.engine_params, 
        **kwargs
    )

    # Handle both single and batched inputs
    if isinstance(context, str):
        # Single prompt - original behavior
        tokens = llm_output[0].outputs[0].token_ids
        top_k_logits = np.array([[obj.logprob for obj in position_dict.values()] for position_dict in llm_output[0].outputs[0].logprobs])
        chosen_token_logit = np.array([llm_output[0].outputs[0].logprobs[i][tokens[i]].logprob for i in range(len(tokens))])
    else:
        # Batched prompts - return lists of results per sequence
        tokens = [output.outputs[0].token_ids for output in llm_output]
        top_k_logits = [
            np.array([[obj.logprob for obj in position_dict.values()] for position_dict in output.outputs[0].logprobs])
            for output in llm_output
        ]
        chosen_token_logit = [
            np.array([output.outputs[0].logprobs[i][output.outputs[0].token_ids[i]].logprob for i in range(len(output.outputs[0].token_ids))])
            for output in llm_output
        ]
    # Returns the generated token_ids, the chosen token logit/logprob, and the top_k logits/logprobs
    return tokens, chosen_token_logit, top_k_logits

# Create the LLM object given the model name and engine parameters
def create_LLM_object(
        model_name,
        model_type=None, # TO DO: Add support for both safetensors (default) and GGUF in vLLM
        dtype="auto", 
        gpu_memory_utilization=0.85, 
        max_model_len=2048, 
        max_logprobs=100, # Controls how many logprobs or logits are stored for each token
        logits=True, 
        **kwargs
    ):

    if(logits):
        # User wants unprocessed logits output
        logprobs_mode = 'raw_logits'
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
