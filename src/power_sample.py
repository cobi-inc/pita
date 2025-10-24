import random
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer
import torch
from torch.nn import functional as F

class AutoregressiveSampler:
    def __init__(self, llm, tokenizer, power_sampling_temperature=1.0, logprobs=100, detokenize=False, token_count=1000, block_size=50, MCMC_steps=5):
        self.llm = llm
        self.tokenizer = tokenizer
        self.power_sampling_temperature = power_sampling_temperature
        self.detokenize = detokenize
        self.token_count = token_count
        self.block_size = block_size
        self.MCMC_steps = MCMC_steps
        self.block_num = token_count // block_size

    def sample(self, context, max_new_tokens):
        if isinstance(context, list):
            context = TokensPrompt(prompt_token_ids=context)
        # Set the sampling parameters of the LLM
        sample_params = SamplingParams(temperature=self.power_sampling_temperature, max_tokens=max_new_tokens, logprobs=-1, detokenize=self.detokenize)
        # Generate a new response from the LLM
        llm_output = self.llm.generate(context, sampling_params=sample_params)
        return llm_output

# Find the output log probabilities of the token sequences for both the p_temp and p_power distributions
def logprobs(output, sampler):
    # Get a list of the logprobs dictionaries from the output
    logprobs_list = output[0].outputs[0].logprobs
    token_ids = output[0].outputs[0].token_ids

    # Initialize tensor with -inf (shape: num_tokens x vocab_size)
    logits = torch.full((len(token_ids), len(logprobs_list[0])), float('-inf'))
    
    # Fill the tensor with logprobs from the dicts (access .logprob from Logprob objects)
    for i, logprob_dict in enumerate(logprobs_list):
        for token_id, logprob_obj in logprob_dict.items():
            logits[i, token_id] = logprob_obj.logprob  # Extract the float logprob
    
    
    # Scale the raw logits by the temperature
    scaled_logits = logits / sampler.power_sampling_temperature

    # Compute logsumexp (normalization constant) for each position
    log_Z = torch.logsumexp(logits, dim=-1)  # Shape: (num_tokens,)
    log_Z_scaled = torch.logsumexp(scaled_logits, dim=-1)  # Shape: (num_tokens,)
    
    # Extract log probs for only the generated tokens
    indices = torch.arange(len(token_ids))
    log_probs = (1/sampler.power_sampling_temperature) * (logits[indices, token_ids] - log_Z)
    log_probs_temp_scaled = scaled_logits[indices, token_ids] - log_Z_scaled

    # Print out the token

    return log_probs, log_probs_temp_scaled

def sliding_window_power_sample(sampler: AutoregressiveSampler, prompt, temperature, power, token_count, seed):
    # Set random seed
    random.seed(seed)

    # Log Probabilites
    logprob = [] # Current list of unscaled log probabilites of the new sample. Length of block_size
    logprob_temp_scaled = [] # Current list of tokens probabilites individually scaled by temperature. Length of block_size
    proposed_logprob = [] # Proposed list of unscaled log probabilites of the new sample. Length of max_new_tokens
    proposed_logprob_temp_scaled = [] # Proposed list of tokens probabilites individually scaled by temperature. Length of max_new_tokens

    # Tokenized prompt + output token sequence
    context = sampler.tokenizer.encode(prompt)
    prompt_length = len(context)
    
    print(context)

    acceptances = 0
    block_acceptances = []

    # Iterate over the number of blocks to be generated
    for block_idx in tqdm(range(sampler.block_num)):
        # Block Acceptances Ratio 
        block_acceptance = 0
        # Generate next block of tokens as baseline
        output = sampler.sample(context, sampler.block_size)
        
        # Calculate the inital logprobabilities for the generated block
        logprob, logprob_temp_scaled = logprobs(output, sampler)
        context.extend(output[0].outputs[0].token_ids)
        #Iterate over the number of MCMC steps
        for _ in tqdm(range(sampler.MCMC_steps)):
            #Find a new point to start a proposal from. Generate idx tokens for the step
            idx = random.randint(0,sampler.block_size -1)

            #Set the new context for the proposed block
            context_proposed = context[:-(sampler.block_size-idx)]
            #Generate proposed block of tokens
            proposed_output = sampler.sample(context_proposed, sampler.block_size-idx)

            # Find the log probabilities of the generated tokens
            proposed_logprob, proposed_logprob_temp_scaled = logprobs(proposed_output, sampler)
            # Calculate the Metro-Hastings acceptance ratio
            # Power Scaled Sequence Log Probability + Temperature Scaled Sequence Log Probability - Current Power Scaled Sequence Log Probability - Current Temperature Scaled Sequence Log Probability
            log_acceptance_ratio = sum(proposed_logprob) + sum(proposed_logprob_temp_scaled) - sum(logprob[-idx:]) - sum(logprob_temp_scaled[-idx:])
            # Accept or reject the proposed block based on the acceptance ratio
            if np.random.rand() < np.exp(log_acceptance_ratio):
                # Ensure the context is updated with the accepted proposal
                context[-(sampler.block_size - idx):] = proposed_output[0].outputs[0].token_ids

                # Update the logprob lists with the accepted proposal's log probabilities
                logprob[-(sampler.block_size - idx):] = proposed_logprob
                logprob_temp_scaled[-(sampler.block_size - idx):] = proposed_logprob_temp_scaled

                # Collected data about the acceptance ratio for overall run and block
                acceptances += 1
                block_acceptance += 1

        block_acceptances.append(block_acceptance)

        # Check if an EOS token has been generated and end the process if so
        if(sampler.tokenizer.eos_token_id in context[-1:]):
            return sampler.tokenizer.decode(context, skip_special_tokens=True), acceptances, block_acceptances


    # EOS never found, just return the full generated context
    return sampler.tokenizer.decode(context, skip_special_tokens=True), acceptances, block_acceptances



def vllm_test(sampler, prompt = "Once upon a time"):
    # User Initial Prompt
    prompt_obj = TokensPrompt(prompt_token_ids=sampler.tokenizer.encode(prompt))
    output = sampler.sample(prompt_obj, max_new_tokens=50)
    print(sampler.tokenizer.decode(output[0].outputs[0].token_ids, skip_special_tokens=True))
    #print(output[0].outputs[0].logprobs)
    #print the dimensions of the logprobs
    token_ids = output[0].outputs[0].token_ids

    # Initialize tensor with -inf (shape: num_tokens x vocab_size)
    logprobs_list = output[0].outputs[0].logprobs
    logits = torch.full((len(logprobs_list), len(logprobs_list[0])), float('-inf'))

    # Fill the tensor with logprobs from the dicts (access .logprob from Logprob objects)
    for i, logprob_dict in enumerate(logprobs_list):
        for token_id, logprob_obj in logprob_dict.items():
            logits[i, token_id] = logprob_obj.logprob  # Extract the float logprob

    # Scale the raw logits by the temperature
    scaled_logits = logits / sampler.power_sampling_temperature

    # Compute logsumexp (normalization constant) for each position
    log_Z = torch.logsumexp(logits, dim=-1)  # Shape: (num_tokens,)
    log_Z_scaled = torch.logsumexp(scaled_logits, dim=-1)  # Shape: (num_tokens,)
    
    # Extract log probs for only the generated tokens
    indices = torch.arange(len(token_ids))
    log_probs = (1/sampler.power_sampling_temperature) * (logits[indices, token_ids] - log_Z)
    log_probs_temp_scaled = scaled_logits[indices, token_ids] - log_Z_scaled

    print("Log Probs for generated tokens:", log_probs, log_probs_temp_scaled)
    print("Logprobs tensor size:", logits.size())

if __name__ == "__main__":
    # Initialize the random number generator
    seed = 42
    random.seed(seed)

    # Power Sampling Hyperparameters
    token_count = 200
    block_size = 50
    MCMC_steps = 5

    # LLM parameters
    model = "Qwen/Qwen3-4B-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code = True)
    skip_tokenizer_init = True
    dtype = "auto"
    quantization = None
    gpu_memory_utilization = 0.9

    # Initalize model
    llm = LLM(model=model, 
              skip_tokenizer_init=skip_tokenizer_init, 
              dtype=dtype, 
              quantization=quantization, 
              gpu_memory_utilization=gpu_memory_utilization,
              max_logprobs= tokenizer.vocab_size + token_count + 1000,
              logprobs_mode='raw_logits')

    #Sampling parameters for vLLM
    temperature = 0.25
    detokenize = False

    #Initalize Autogressive Sampler
    sampler = AutoregressiveSampler(llm, 
                                    tokenizer,
                                    power_sampling_temperature=temperature,
                                    detokenize=detokenize,
                                    token_count=token_count,
                                    block_size=block_size,
                                    MCMC_steps=MCMC_steps
                                    )


    # Test vLLM sampling
    #vllm_test(sampler)

    # Call the power_sample function
    prompt_response, total_acceptances, block_acceptances = sliding_window_power_sample(sampler, prompt="Once upon a time", temperature=temperature, power=1.0, token_count=token_count, seed=seed)
    print("Generated Text:", prompt_response)
    print("Total Acceptances:", total_acceptances)
    print("Block Acceptances:", block_acceptances)