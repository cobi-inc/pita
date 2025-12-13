# Standard Libraries
import os
import random

# Math Libraries
import numpy as np
import torch

# Helper Library
from tqdm import tqdm

# Custom Libraries
from pita.inference.LLM_backend import AutoregressiveSampler

# Lazy imports for backends - will be imported when needed
vllm_backend = None
llama_cpp_backend = None

def _get_vllm_backend():
    global vllm_backend
    if vllm_backend is None:
        import pita.inference.vllm_backend as _vllm_backend
        vllm_backend = _vllm_backend
    return vllm_backend

def _get_llama_cpp_backend():
    global llama_cpp_backend
    if llama_cpp_backend is None:
        import pita.inference.llama_cpp_backend as _llama_cpp_backend
        llama_cpp_backend = _llama_cpp_backend
    return llama_cpp_backend

# Power Sampling Parameters
class Power_Sampling_Params:
    def __init__(
        self, 
        block_size=50, # How many blocks to divide the total output tokens into for power sampling. Smaller block sizes = better quality but slower
        MCMC_steps=5 # Number of MCMC steps to perform per block. More steps = better quality but slower
    ):
        self.block_size = block_size
        self.MCMC_steps = MCMC_steps

# Checks that the LLM and parameters are compatible with power sampling and enables power sampling
def enable_power_sampling(sampler, block_size, MCMC_steps):
    # Check if the sampler is initialized
    if(sampler is None):
        raise ValueError("Sampler must be initialized before enabling power sampling.")
    
    # Check that you have a logit space to operate in
    if sampler.sampling_params.logits_per_token is None or sampler.sampling_params.logits_per_token <= 0:
        raise ValueError("Sampler must be initialized with logits_per_token to enable power sampling.")

    # Check the individual engine compatibility for power sampling
    if sampler.engine == "vllm":
        backend = _get_vllm_backend()
        backend.check_vllm_power_sampling_compatibility(sampler)

    elif sampler.engine == "llama_cpp":
        backend = _get_llama_cpp_backend()
        backend.check_llama_cpp_power_sampling_compatibility(sampler)
        
    else:
        raise ValueError(f"Engine {sampler.engine} not supported for Power Sampling.")

    # Set the power sampling parameters
    sampler.power_sampling_params = Power_Sampling_Params(
        block_size=block_size,
        MCMC_steps=MCMC_steps
    )
    
    print(f"Power Sampling Enabled: Logits Consider = {sampler.sampling_params.logits_per_token}, Total Output Tokens = {sampler.sampling_params.max_tokens}, Block Size = {block_size}, MCMC Steps = {MCMC_steps}, Temperature (1/alpha) = {sampler.sampling_params.temperature}")

# Find the output log probabilities of the token sequences for both the p_temp and p_power distributions
# token_ids is a list of the chosen tokens
# logprobs_list is a list of lists of the logprobs of each possible token for a given position in the token sequence from vLLM
def logprobs(tokens_list, top_k_logits, unprocessed_normalization_constant, temp_processed_normalization_constant, power_sampling_temperature):
    # Initialize normalization tensors with -inf (shape: num_tokens)
    # Find the chosen token logits from the top_k_logits and scale them by the max logit
    chosen_token_logit_list = np.zeros(len(tokens_list))
    for i in range(len(tokens_list)):
        if(len(top_k_logits[i]) > 1):
            chosen_token_logit_list[i] = top_k_logits[i][0] - np.max(top_k_logits[i][:2])
            
    # log_probs is the power sampled version =
    # log(softmax(logit_selected)^(1/T)) = 
    # (1/T) * log((e^(logit_selected) / sum(e^all_logits)) = 
    # (1/T)(logit_selected - log(sum(e^all_logits))
    # The scaled version uses already temperature scaled logits
    # Scaled = log(softmax(logit_selected / T)) =
    # 1/T * logit_selected - log(sum(e^(all_logits / T)))
    print("Shape of chosen_token_logit_list:", chosen_token_logit_list.shape)  
    print("Shape of unprocessed_normalization_constant:", np.array(unprocessed_normalization_constant).shape)
    logprob_initial = (1/power_sampling_temperature) * (chosen_token_logit_list - unprocessed_normalization_constant)
    logprob_temp_scaled_initial = (1/power_sampling_temperature) * chosen_token_logit_list - temp_processed_normalization_constant

    return logprob_initial, logprob_temp_scaled_initial

# Performs sliding window power sampling on the given prompt
# Sliding window only performs power sampling on a specific block size of tokens at a time instead of the whole prompt
# Increases the speed of getting a prompt response
# Input is the sampler object, prompt string, temperature, power, total token count to generate, and random seed
# Output is the generated string, total acceptances, and block acceptances
def sliding_window_power_sample(sampler: AutoregressiveSampler, prompt):
    # Log Probabilites
    logprob = [] # Current list of unscaled log probabilites of the new sample. Length of block_size
    logprob_temp_scaled = [] # Current list of tokens probabilites individually scaled by temperature. Length of block_size
    proposed_logprob = [] # Proposed list of unscaled log probabilites of the new sample. Length of max_new_tokens
    proposed_logprob_temp_scaled = [] # Proposed list of tokens probabilites individually scaled by temperature. Length of max_new_tokens
    
    # Statistic Collection
    # Acceptance parameters
    total_tokens_generated = 0
    acceptances = 0
    block_acceptances = []

    # New Context Window to be changed and token history to keep track of all accepted tokens
    context = []
    token_history = ""

    # Iterate over the number of blocks to be generated
    block_count = sampler.sampling_params.max_tokens // sampler.power_sampling_params.block_size
    for block_idx in tqdm(range(block_count), disable=True):
        # Block Acceptances Ratio
        block_acceptance = 0

        # Generate next block of tokens as baseline
        # If the programmatical LLM is being used
        tokens_list, token_logprob_list, logprobs_list = sampler.sample(prompt + token_history + sampler.tokenizer.decode(context, skip_special_tokens=False), sampler.block_size)
        
        # Record how many tokens have been generated
        total_tokens_generated += len(tokens_list)

        # Calculate the initial logprobabilities for the generated block
        logprob, logprob_temp_scaled = logprobs(tokens_list, token_logprob_list, logprobs_list, sampler.sampling_params.temperature)
        # Extend the context with the newly generated tokens
        context = tokens_list

        #Iterate over the number of MCMC steps
        for _ in tqdm(range(sampler.MCMC_steps), disable=True):
            #Find a new point to start a proposal from. Generate idx tokens for the step
            idx = random.randint(0,len(context) -1)
            
            #Set the new context for the proposed block
            context_proposed = context[:-(len(context)-idx)]

            #Generate proposed block of tokens
            proposed_tokens_list, proposed_token_logprob_list, proposed_logprobs_list = sampler.sample(prompt + token_history + sampler.tokenizer.decode(context_proposed, skip_special_tokens=False), len(context) - idx)
            

            # Record how many tokens have been generated
            total_tokens_generated += len(proposed_tokens_list)

            # Find the log probabilities of the generated tokens
            proposed_logprob, proposed_logprob_temp_scaled = logprobs(proposed_tokens_list, proposed_token_logprob_list, proposed_logprobs_list, sampler.power_sampling_temperature)
           
            # Calculate the Metro-Hastings acceptance ratio
            # Power Scaled Sequence Log Probability + Temperature Scaled Sequence Log Probability - Current Power Scaled Sequence Log Probability - Current Temperature Scaled Sequence Log Probability
            log_acceptance_ratio = sum(proposed_logprob) + sum(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) - sum(logprob[idx:idx+len(proposed_tokens_list)]) - sum(proposed_logprob_temp_scaled)

            # Check to make sure we are comparing the correct number of elements
            assert(len(proposed_logprob) == len(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) == len(logprob[idx:idx+len(proposed_tokens_list)]) == len(proposed_logprob_temp_scaled))

            # Accept or reject the proposed block based on the acceptance ratio
            if np.random.rand() < np.exp(log_acceptance_ratio):
                # print("Accepted Proposal at index", idx)
                # Ensure the context is updated with the accepted proposal
                context = context_proposed + proposed_tokens_list

                # Update the logprob lists with the accepted proposal's log probabilities
                logprob = torch.cat([logprob[:idx], proposed_logprob], dim=0)
                logprob_temp_scaled = torch.cat([logprob_temp_scaled[:idx], proposed_logprob_temp_scaled], dim=0)

                # Collected data about the acceptance ratio for overall run and block
                acceptances += 1
                block_acceptance += 1
        
        #record block acceptances
        block_acceptances.append(block_acceptance)

        # Update the prompt with the newly generated/accepted context
        token_history = token_history + sampler.tokenizer.decode(context, skip_special_tokens=False)

        # Check if an EOS token has been generated and end the process if so
        if(sampler.tokenizer.eos_token_id in context):
            return token_history, acceptances, block_acceptances, total_tokens_generated


    # EOS never found, just return the full generated context
    return token_history, acceptances, block_acceptances, total_tokens_generated

# Performs power sampling on the given prompt
def power_sampling(
    sampler: AutoregressiveSampler, 
    prompt,
    logging=False
):  

    # Statistic Logging in a CSV
    if(logging):
        total_tokens_generated = 0
        acceptances = 0
        block_acceptances = []
        index_proposals = []
        acceptance_ratios = []

    # Log Probabilites
    logprob = [] # Current list of unscaled log probabilites of the new sample. Length of block_size
    logprob_temp_scaled = [] # Current list of tokens probabilites individually scaled by temperature. Length of block_size
    proposed_logprob = [] # Proposed list of unscaled log probabilites of the new sample. Length of max_new_tokens
    proposed_logprob_temp_scaled = [] # Proposed list of tokens probabilites individually scaled by temperature. Length of max_new_tokens

    # New Context Window to be changed
    context = []

    block_count = sampler.sampling_params.max_tokens // sampler.power_sampling_params.block_size
    # Iterate over the number of blocks to be generated
    for block_idx in tqdm(range(block_count), disable=True):
        
        # Block Acceptances Ratio
        if(logging):
            block_acceptance = 0
            index_proposal_block = []

        # Generate next block of tokens as baseline
        # If the programmatical LLM is being used
        tokens_list, top_k_logits, _, unprocessed_normalization_constant, temp_processed_normalization_constant = sampler.sample(prompt +  sampler.tokenizer.decode(context, skip_special_tokens=False), sampler.power_sampling_params.block_size)

        # Record how many tokens have been generated
        if(logging):
            total_tokens_generated += len(tokens_list)
        
        # Calculate the initial power sampling and low-temperature logprobabilities for the generated block
        logprob_initial, logprob_temp_scaled_initial = logprobs(tokens_list, top_k_logits, unprocessed_normalization_constant, temp_processed_normalization_constant, sampler.sampling_params.temperature)

        # Extend the initial log probabilities
        logprob = [*logprob, *logprob_initial.tolist()]
        logprob_temp_scaled = [*logprob_temp_scaled, *logprob_temp_scaled_initial.tolist()]

        # Extend the context with the newly generated tokens
        context.extend(tokens_list)

        #Iterate over the number of MCMC steps
        for _ in tqdm(range(sampler.power_sampling_params.MCMC_steps), disable=True):
            #Find a new point to start a proposal from. Generate idx tokens for the step
            idx = random.randint(1, len(context) - 1)
            
            # Logging the proposal index
            if(logging):
                index_proposal_block.append(idx)

            #Set the new context for the proposed block
            context_proposed = context[:idx]

            #Generate proposed block of tokens
            proposed_tokens_list, proposed_top_k_logits_list, _, unprocessed_normalization_constant, temp_processed_normalization_constant = sampler.sample(prompt +  sampler.tokenizer.decode(context_proposed, skip_special_tokens=False), len(context) - idx)

            # Record how many tokens have been generated
            if(logging):
                total_tokens_generated += len(proposed_tokens_list)

            # Find the log probabilities of the generated tokens
            proposed_logprob, proposed_logprob_temp_scaled = logprobs(proposed_tokens_list, proposed_top_k_logits_list, unprocessed_normalization_constant, temp_processed_normalization_constant, sampler.sampling_params.temperature)

            # Extend the initial log probabilities
            # Calculate the Metro-Hastings acceptance ratio
            # Power Scaled Sequence Log Probability + Temperature Scaled Sequence Log Probability - Current Power Scaled Sequence Log Probability - Current Temperature Scaled Sequence Log Probability
            log_acceptance_ratio = sum(proposed_logprob) + sum(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) - sum(logprob[idx:idx+len(proposed_tokens_list)]) - sum(proposed_logprob_temp_scaled)

            # Check to make sure we are comparing the correct number of elements
            assert(len(proposed_logprob) == len(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) == len(logprob[idx:idx+len(proposed_tokens_list)]) == len(proposed_logprob_temp_scaled))

            # Log Acceptance Ratio
            if(logging):
                acceptance_ratios.append(log_acceptance_ratio)
            
            # Accept or reject the proposed block based on the acceptance ratio
            if np.random.rand() < np.exp(log_acceptance_ratio):
                # print("Accepted Proposal at index", idx)
                # Ensure the context is updated with the accepted proposal
                context[idx:] = proposed_tokens_list
                
                # Update the logprob lists with the accepted proposal's log probabilities
                logprob = [*logprob[:idx], *proposed_logprob]
                logprob_temp_scaled = [*logprob_temp_scaled[:idx], *proposed_logprob_temp_scaled]
                
                # Collected data about the acceptance ratio for overall run and block
                acceptances += 1
                block_acceptance += 1



        #record block acceptances
        block_acceptances.append(block_acceptance)
        index_proposals.append(index_proposal_block)

        # Check if an EOS token has been generated and end the process if so
        if(sampler.tokenizer.eos_token_id in context):
            return sampler.tokenizer.decode(context, skip_special_tokens=False), acceptances, block_acceptances, index_proposals, total_tokens_generated


    # EOS never found, just return the full generated context
    return sampler.tokenizer.decode(context, skip_special_tokens=False), acceptances, block_acceptances, index_proposals, total_tokens_generated
