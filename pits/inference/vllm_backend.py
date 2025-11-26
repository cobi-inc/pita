# vLLM Libraries
from vllm import LLM, SamplingParams, TokensPrompt

# Take in the context (string) and max_new_tokens (int)
# Returns the generated tokens. the chosen token logprobs, and all the logprobs as lists to the user
def sample(
        self, 
        context, 
        max_new_tokens, 
        **kwargs
    ):

    # Prepare the context as a TokensPrompt if it's a list of token IDs
    if isinstance(context, list):
        context = TokensPrompt(prompt_token_ids=context)
    
    # Update the max tokens if needed
    if(self.sampling_params.engine_params.max_tokens != max_new_tokens):
        self.sampling_params.engine_params.max_tokens = max_new_tokens

    # Generate a new response from the LLM
    llm_output = self.llm.generate(context, sampling_params=self.sampling_params.engine_params, **kwargs)
    tokens = llm_output[0].outputs[0].token_ids
    
    # Extract the top top_k logits/logprobs for each generated token
    # Logits vs Logprobs depend on the logprobs_mode set during the LLM Class initialization
    top_k_logits = [[obj.logprob for obj in position_dict.values()] for position_dict in llm_output[0].outputs[0].logprobs]
    # Create a list of the logit/logprob for the chosen token at each position
    chosen_token_logit = [llm_output[0].outputs[0].logprobs[i][tokens[i]].logprob for i in range(len(tokens))]
    
    # Returns the generated token_ids, the chosen token logit/logprob, and the top_k logits/logprobs
    return tokens, chosen_token_logit, top_k_logits

# Create the LLM object given the model name and engine parameters
def create_LLM_object(
        model_name,
        model_type=None, # TO DO: Add support for both safetensors (default) and GGUF in vLLM
        dtype="auto", 
        gpu_memory_utilization=0.85, 
        max_model_len=2048, 
        max_logprobs=100, 
        logits=True, 
        **kwargs
    ):

    if(logits):
        logprobs_mode = 'raw_logits'
        
    # Initialize VLLM locally for performance (as done in power_sample.py main)
    llm = LLM(model=model_name,
              dtype=dtype,
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len,
              max_logprobs=max_logprobs, # needed for MCMC
              logprobs_mode=logprobs_mode,
              **kwargs)

    return llm

def create_vllm_engine_params():
    # Create the vLLM SamplingParams object from the common Sampling_Params
    vllm_params = SamplingParams()
    return vllm_params

# Check that the vLLM engine can suppport power sampling with the given configuration
def check_vllm_power_sampling_compatibility(sampler):
    # Check to make sure the vLLM engine is outputing logits/logprobs
    if(sampler.sampling_params.top_k <= 0):
        raise ValueError("LLM engine top_k must be set to a positive integer to enable power sampling.")
    
    if(sampler.llm.max_logprobs is None or sampler.llm.max_logprobs < sampler.sampling_params.top_k):
        raise ValueError("LLM engine max_logprobs must be set to at least top_k to enable power sampling.")
    
    if(sampler.llm.logprobs_mode != 'raw_logits'):
        raise ValueError("LLM engine logprobs_mode must be set to 'raw_logits' to enable power sampling. This is done by setting logits=True when creating the LLM object.")
