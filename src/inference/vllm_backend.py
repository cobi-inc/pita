# vLLM Libraries
from vllm import LLM, SamplingParams, TokensPrompt

# Take in the context (string) and max_new_tokens (int)
# Returns the generated tokens. the chosen token logprobs, and all the logprobs as lists to the user
def sample(self, context, max_new_tokens):
    # Prepare the context as a TokensPrompt if it's a list of token IDs
    if isinstance(context, list):
        context = TokensPrompt(prompt_token_ids=context)
    
    # Set the sampling parameters of the LLM
    if(self.sampling_params.engine_params is None):
        self.sampling_params.engine_params = SamplingParams( temperature=self.sampling_params.temperature, 
                                    top_k=self.sampling_params.top_k, 
                                    max_tokens=max_new_tokens, 
                                    logprobs=self.sampling_params.top_k, 
                                    stop_token_ids =[self.tokenizer.eos_token_id])

    if(self.sampling_params.engine_params.max_tokens != max_new_tokens):
        self.sampling_params.engine_params.max_tokens = max_new_tokens

    # Generate a new response from the LLM
    llm_output = self.llm.generate(context, sampling_params=self.sampling_params.engine_params)
    tokens = llm_output[0].outputs[0].token_ids
    
    # Extract the top top_k logits/logprobs for each generated token
    # Logits vs Logprobs depend on the logprobs_mode set during the LLM Class initialization
    top_k_logits = [[obj.logprob for obj in position_dict.values()] for position_dict in llm_output[0].outputs[0].logprobs]
    # Create a list of the logit/logprob for the chosen token at each position
    chosen_token_logit = [llm_output[0].outputs[0].logprobs[i][tokens[i]].logprob for i in range(len(tokens))]

    return tokens, chosen_token_logit, top_k_logits

# Create the LLM object given the model name and engine parameters
def create_LLM_object(model_name, dtype="auto", gpu_memory_utilization=0.85, max_model_len=2048, max_logprobs=100):
    # Initialize VLLM locally for performance (as done in power_sample.py main)
    llm = LLM(model=model_name,
              dtype=dtype,
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len,
              max_logprobs=max_logprobs, # needed for MCMC
              logprobs_mode='raw_logits',
              trust_remote_code=True)

    return llm