# Standard Libraries
import os
import random
import time

# Math Libraries
import numpy as np
import torch

# Helper Library
from tqdm import tqdm
import json

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
        block_size=192, # How many blocks to divide the total output tokens into for power sampling. Smaller block sizes = better quality but slower
        MCMC_steps=8 # Number of MCMC steps to perform per block. More steps = better quality but slower
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
def logprobs(chosen_logit_list, unprocessed_normalization_constant, temp_processed_normalization_constant, power_sampling_temperature):

    # log_probs is the power sampled version =
    # log(softmax(logit_selected)^(1/T)) = 
    # (1/T) * log((e^(logit_selected) / sum(e^all_logits)) = 
    # (1/T)(logit_selected - log(sum(e^all_logits))
    # The scaled version uses already temperature scaled logits
    # Scaled = log(softmax(logit_selected / T)) =
    # 1/T * logit_selected - log(sum(e^(all_logits / T)))
    logprob_initial = (1/power_sampling_temperature) * (chosen_logit_list - unprocessed_normalization_constant)
    logprob_temp_scaled_initial = (1/power_sampling_temperature) * chosen_logit_list - temp_processed_normalization_constant

    return logprob_initial, logprob_temp_scaled_initial

# Performs power sampling on the given prompt
def power_sampling(
    sampler: AutoregressiveSampler, 
    prompt,
    logging=False,
    log_file_path=None
):  
    # Set the random seed for reproducibility
    if sampler.sampling_params.seed is not None:
        np.random.seed(sampler.sampling_params.seed)
        random.seed(sampler.sampling_params.seed)

    # Statistic Logging in a CSV
    if(logging):
        #create or overwrite log file
        power_sampling_log_path = log_file_path if log_file_path is not None else f"power_sampling_log_{time.strftime('%H%M%S_%d_%m_%Y')}.csv"
        with open(power_sampling_log_path, "w") as log_file:
            log_file.write(f'"{json.dumps(vars(sampler), default=str).replace("\"", "\"\"")}"\n')
            log_file.write(f'"{prompt.replace("\"", "\"\"")}"\n')
            log_file.write("proposed_power_sampling_logprob_norm,proposed_low_temp_logprob_norm,compared_power_sampling_logprob_norm,compared_low_temp_logprob_norm,new_power_sampling_logprob_norm,new_low_temp_logprob_norm,acceptance_ratio,accepted,starting_index,tokens_generated,\n")
    
    logprob = [] # Current list of unscaled log probabilites of the new sample. Length of block_size
    logprob_temp_scaled = [] # Current list of tokens probabilites individually scaled by temperature. Length of block_size
    proposed_logprob = [] # Proposed list of unscaled log probabilites of the new sample. Length of max_new_tokens
    proposed_logprob_temp_scaled = [] # Proposed list of tokens probabilites individually scaled by temperature. Length of max_new_tokens

    # New Context Window to be changed
    context = []

    block_count = sampler.sampling_params.max_tokens // sampler.power_sampling_params.block_size
    # Iterate over the number of blocks to be generated
    for block_idx in tqdm(range(block_count), disable=True):
        
        # Generate next block of tokens as baseline
        # If the programmatical LLM is being used
        tokens_list, top_k_logits, _, unprocessed_log_normalization_constant, temp_processed_log_normalization_constant, entropy = sampler.sample(prompt +  sampler.tokenizer.decode(context, skip_special_tokens=False), sampler.power_sampling_params.block_size)
        
        # Calculate the initial power sampling and low-temperature logprobabilities for the generated block
        logprob_initial, logprob_temp_scaled_initial = logprobs(top_k_logits[:, 0], unprocessed_log_normalization_constant, temp_processed_log_normalization_constant, sampler.sampling_params.temperature)

        # Extend the initial log probabilities
        logprob = [*logprob, *logprob_initial.tolist()]
        logprob_temp_scaled = [*logprob_temp_scaled, *logprob_temp_scaled_initial.tolist()]

        # Extend the context with the newly generated tokens
        context.extend(tokens_list)

        if(logging):
            proposed_power_sampling_logprob_norm = "None"
            proposed_low_temp_logprob_norm = "None "
            compared_power_sampling_logprob_norm = "None"
            compared_low_temp_logprob_norm = "None"
            new_power_sampling_logprob_norm = sum(logprob_initial)/len(logprob_initial)
            new_low_temp_logprob_norm = sum(logprob_temp_scaled_initial)/len(logprob_temp_scaled_initial)
            acceptance_ratio = "None"
            accepted = "None"
            tokens_generated = len(tokens_list)
            starting_index = len(context)-tokens_generated
            # Write initial generated block data to log
            with open(power_sampling_log_path, "a") as log_file:
                log_file.write(f"{proposed_power_sampling_logprob_norm},{proposed_low_temp_logprob_norm},{compared_power_sampling_logprob_norm},{compared_low_temp_logprob_norm},{new_power_sampling_logprob_norm},{new_low_temp_logprob_norm},{acceptance_ratio},{accepted},{starting_index},{tokens_generated}\n")

        #Iterate over the number of MCMC steps
        for _ in tqdm(range(sampler.power_sampling_params.MCMC_steps), disable=True):
            #Find a new point to start a proposal from. Generate idx tokens for the step.
            idx = random.randint(0, len(context) - 1)
            
            #Set the new context for the proposed block
            context_proposed = context[:idx]

            #Generate proposed block of tokens
            proposed_tokens_list, proposed_top_k_logits_list, _, unprocessed_log_normalization_constant, temp_processed_log_normalization_constant, entropy = sampler.sample(prompt +  sampler.tokenizer.decode(context_proposed, skip_special_tokens=False), len(context) - idx)

            # Find the log probabilities of the generated tokens
            proposed_logprob, proposed_logprob_temp_scaled = logprobs(proposed_top_k_logits_list[:, 0], unprocessed_log_normalization_constant, temp_processed_log_normalization_constant, sampler.sampling_params.temperature)

            # Extend the initial log probabilities
            # Calculate the Metro-Hastings acceptance ratio
            # Power Scaled Sequence Log Probability + Temperature Scaled Sequence Log Probability - Current Power Scaled Sequence Log Probability - Current Temperature Scaled Sequence Log Probability
            log_acceptance_ratio = sum(proposed_logprob) + sum(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) - sum(logprob[idx:idx+len(proposed_tokens_list)]) - sum(proposed_logprob_temp_scaled)

            # Check to make sure we are comparing the correct number of elements
            assert(len(proposed_logprob) == len(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) == len(logprob[idx:idx+len(proposed_tokens_list)]) == len(proposed_logprob_temp_scaled))

            # Log the logprobs and acceptance ratio
            if(logging):
                proposed_power_sampling_logprob_norm = sum(proposed_logprob)
                proposed_low_temp_logprob_norm = sum(proposed_logprob_temp_scaled)
                compared_power_sampling_logprob_norm = sum(logprob[idx:idx+len(proposed_tokens_list)])
                compared_low_temp_logprob_norm = sum(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)])
    
                # Write initial generated block data to log
                with open(power_sampling_log_path, "a") as log_file:
                    log_file.write(f"{proposed_power_sampling_logprob_norm},{proposed_low_temp_logprob_norm},{compared_power_sampling_logprob_norm},{compared_low_temp_logprob_norm},")
                
            acceptance = False
            # Accept or reject the proposed block based on the acceptance ratio
            if np.random.rand() < np.exp(log_acceptance_ratio):
                # print("Accepted Proposal at index", idx)
                # Ensure the context is updated with the accepted proposal
                context[idx:] = proposed_tokens_list
                
                # Update the logprob lists with the accepted proposal's log probabilities
                logprob = [*logprob[:idx], *proposed_logprob]
                logprob_temp_scaled = [*logprob_temp_scaled[:idx], *proposed_logprob_temp_scaled]
                
                # Flag acceptance
                acceptance = True

            if(logging):
                new_power_sampling_logprob_norm = sum(logprob)/len(logprob)
                new_low_temp_logprob_norm = sum(logprob_temp_scaled)/len(logprob_temp_scaled)
                acceptance_ratio = np.exp(log_acceptance_ratio)
                accepted = acceptance
                tokens_generated = len(proposed_tokens_list)
                starting_index = idx
                with open(power_sampling_log_path, "a") as log_file:
                    log_file.write(f"{new_power_sampling_logprob_norm},{new_low_temp_logprob_norm},{acceptance_ratio},{accepted},{starting_index},{tokens_generated}\n")
        
        # Check if an EOS token has been generated and end the process if so
        if(sampler.tokenizer.eos_token_id in context):
            decoded_text = sampler.tokenizer.decode(context, skip_special_tokens=False)
            if logging:
                with open(power_sampling_log_path, "a") as log_file:
                    log_file.write(f'"{decoded_text.replace("\"", "\"\"")}"\n')
            return decoded_text


    # EOS never found, just return the full generated context
    decoded_text = sampler.tokenizer.decode(context, skip_special_tokens=False)
    if logging:
        with open(power_sampling_log_path, "a") as log_file:
            log_file.write(f'"{decoded_text.replace("\"", "\"\"")}"\n')
    return decoded_text