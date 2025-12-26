# Standard Libraries
import os
import random
import time

# Math Libraries
import numpy as np
from scipy.special import logsumexp

# Helper Library
from tqdm import tqdm
import json

# Custom Libraries
from pita.inference.LLM_backend import AutoregressiveSampler, Output
from pita.sampling.token_metrics import calc_token_metric

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
class Power_Sampling:
    """
    Power Sampling Class that stores the parameters and methods used for power sampling.
    
    Attributes:
        block_size (int): How many blocks to divide the total output tokens into for power sampling. Smaller block sizes = better quality but slower
        MCMC_steps (int): Number of MCMC steps to perform per block. More steps = better quality but slower
        token_metric (str): Metric to use for token selection. Can be "logprobs", "power_distribution", or "entropy"
    """
    def __init__(
        self, 
        block_size: int = 192, # How many blocks to divide the total output tokens into for power sampling. Smaller block sizes = better quality but slower
        MCMC_steps: int = 8, # Number of MCMC steps to perform per block. More steps = better quality but slower
        token_metric: str = "power_distribution"
    ):
        self.block_size = block_size
        self.MCMC_steps = MCMC_steps
        self.token_metric = token_metric

    # TODO Implement the power sampling method in the Power_Sampling class
    # Power Sampling method 
    def sample(
        self, 
        sampler: AutoregressiveSampler, 
        prompt: str,
        logging: bool = False,
        log_file_path: str = None
    )-> Output:
        """
        Sample using power sampling.

        Args:
            sampler (AutoregressiveSampler): The sampler object containing sampling parameters and the LLM engine.
            prompt (str): The prompt to sample from.
            logging (bool, optional): Whether to log the sampling process. Defaults to False.
            log_file_path (str, optional): The path to the log file. Defaults to None.
        Returns:
            Output (Output): The output of the sampling process.
        """
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
                log_file.write("proposed_target_distribution_sum,proposed_sampling_distribution_sum,current_target_distribution_sum,current_sampling_distribution_sum,new_target_distribution_normalized,new_sampling_distribution_normalized,acceptance_ratio,accepted,starting_index,tokens_generated,\n")

        # Intialize arrays to store the probabilities of the current tokens
        current_target_distribution = [] # Current list of unscaled log probabilities of the new sample. Length of block_size
        current_sampling_distribution = [] # Current list of tokens probabilities individually scaled by temperature. Length of block_size

        # New Context Window to be changed
        context = []
        logits = []
        logprobs = []
        unprocessed_log_normalization_constant = []
        temp_processed_log_normalization_constant = []
        entropy = []

        # Number of blocks to be sampled
        block_count = sampler.sampling_params.max_tokens // self.block_size
        for block_idx in range(block_count):
            # Sample the initial new tokens for the block
            output = sampler.sample(prompt +  sampler.tokenizer.decode(context, skip_special_tokens=False), self.block_size)
            
            # Calculate the log probabilities of the initial new tokens for the block
            target_distribution = calc_token_metric(output, sampler, self.token_metric)
            sampling_distribution = calc_token_metric(output, sampler, "logprobs")

            # Extend the distributions
            current_target_distribution = [*current_target_distribution, *target_distribution.tolist()]
            current_sampling_distribution = [*current_sampling_distribution, *sampling_distribution.tolist()]

            # Extend the context with the newly generated tokens
            context.extend(output.tokens)
            # Extend the other Output attributes along
            logits.extend(output.logits)
            logprobs.extend(output.logprobs)
            unprocessed_log_normalization_constant.extend(output.unprocessed_log_normalization_constant)
            temp_processed_log_normalization_constant.extend(output.temp_processed_log_normalization_constant)
            entropy.extend(output.entropy)

            # Log Results
            if(logging):
                proposed_target_distribution_sum = "None"
                proposed_sampling_distribution_sum = "None "
                current_target_distribution_sum = "None"
                current_sampling_distribution_sum = "None"
                new_target_distribution_normalized = sum(target_distribution)/len(target_distribution)
                new_sampling_distribution_normalized = sum(sampling_distribution)/len(sampling_distribution)
                acceptance_ratio = "None"
                accepted = "None"
                tokens_generated = len(output.tokens)
                starting_index = len(context)-tokens_generated
                # Write initial generated block data to log
                with open(power_sampling_log_path, "a") as log_file:
                    log_file.write(f"{proposed_target_distribution_sum},{proposed_sampling_distribution_sum},{current_target_distribution_sum},{current_sampling_distribution_sum},{new_target_distribution_normalized},{new_sampling_distribution_normalized},{acceptance_ratio},{accepted},{starting_index},{tokens_generated}\n")

            # Perform the MCMC Steps to hone in on the target distribution
            for _ in range(self.MCMC_steps):
                #Find a new point to start a proposal from. Generate idx tokens for the step.
                idx = random.randint(0, len(context) - 1)

                #Set the new context for the proposed block
                context_proposed = context[:idx]
                
            #Generate proposed block of tokens
            output = sampler.sample(prompt +  sampler.tokenizer.decode(context_proposed, skip_special_tokens=False), len(context) - idx)

            #Find the proposed probability distributions 
            proposed_target_distribution = calc_token_metric(output, sampler, self.token_metric)
            proposed_sampling_distribution = calc_token_metric(output, sampler, "logprobs")

            # Calculate the Metro-Hastings acceptance ratio
            # Power Scaled Sequence Log Probability + Temperature Scaled Sequence Log Probability - Current Power Scaled Sequence Log Probability - Current Temperature Scaled Sequence Log Probability
            log_acceptance_ratio = sum(proposed_target_distribution) + sum(current_sampling_distribution[idx:idx+len(output.tokens)]) - sum(current_target_distribution[idx:idx+len(output.tokens)]) - sum(proposed_sampling_distribution)
            
            # Check to make sure we are comparing the correct number of elements
            assert(len(proposed_target_distribution) == len(current_sampling_distribution[idx:idx+len(output.tokens)]) == len(current_target_distribution[idx:idx+len(output.tokens)]) == len(proposed_sampling_distribution))

            # Log the logprobs and acceptance ratio
            if(logging):
                proposed_target_distribution_sum = sum(proposed_target_distribution)
                proposed_sampling_distribution_sum = sum(proposed_sampling_distribution)
                current_target_distribution_sum = sum(current_target_distribution[idx:idx+len(output.tokens)])
                current_sampling_distribution_sum = sum(current_sampling_distribution[idx:idx+len(output.tokens)])
    
                # Write initial generated block data to log
                with open(power_sampling_log_path, "a") as log_file:
                    log_file.write(f"{proposed_target_distribution_sum},{proposed_sampling_distribution_sum},{current_target_distribution_sum},{current_sampling_distribution_sum},")
                
            acceptance = False
            # Accept or reject the proposed block based on the acceptance ratio
            if np.random.rand() < np.exp(log_acceptance_ratio):
                # Ensure the context is updated with the accepted proposal
                context[idx:] = output.tokens
                # Replace the tail of the other Output attributes along with the context
                logits = np.concatenate([logits[:idx], output.logits], axis=0)
                logprobs = np.concatenate([logprobs[:idx], output.logprobs], axis=0)
                unprocessed_log_normalization_constant = np.concatenate(
                    [unprocessed_log_normalization_constant[:idx], output.unprocessed_log_normalization_constant],
                    axis=0,
                )
                temp_processed_log_normalization_constant = np.concatenate(
                    [temp_processed_log_normalization_constant[:idx], output.temp_processed_log_normalization_constant],
                    axis=0,
                )
                entropy = np.concatenate([entropy[:idx], output.entropy], axis=0)

                # Update the logprob lists with the accepted proposal's log probabilities
                current_target_distribution = [*current_target_distribution[:idx], *proposed_target_distribution]
                current_sampling_distribution = [*current_sampling_distribution[:idx], *proposed_sampling_distribution]
                
                # Flag acceptance
                acceptance = True

            # Log the new distributions and acceptance ratio
            if(logging):
                current_target_distribution_norm = sum(current_target_distribution)/len(current_target_distribution)
                current_sampling_distribution_norm = sum(current_sampling_distribution)/len(current_sampling_distribution)
                acceptance_ratio = np.exp(log_acceptance_ratio)
                accepted = acceptance
                tokens_generated = len(output.tokens)
                starting_index = idx
                with open(power_sampling_log_path, "a") as log_file:
                    log_file.write(f"{current_target_distribution_norm},{current_sampling_distribution_norm},{acceptance_ratio},{accepted},{starting_index},{tokens_generated}\n")
        
        # Check if an EOS token has been generated and end the process if so
        if(sampler.tokenizer.eos_token_id in context):
            decoded_text = sampler.tokenizer.decode(context, skip_special_tokens=False)
            if logging:
                with open(power_sampling_log_path, "a") as log_file:
                    log_file.write(f'"{decoded_text.replace("\"", "\"\"")}"\n')
            return Output(tokens=context,top_k_logits=logits,top_k_logprobs=logprobs,unprocessed_log_normalization_constant=unprocessed_log_normalization_constant,temp_processed_log_normalization_constant=temp_processed_log_normalization_constant,entropy=entropy)


        # EOS never found, just return the full generated context
        decoded_text = sampler.tokenizer.decode(context, skip_special_tokens=False)
        if logging:
            with open(power_sampling_log_path, "a") as log_file:
                log_file.write(f'"{decoded_text.replace("\"", "\"\"")}"\n')
        
        return Output(tokens=context,top_k_logits=logits,top_k_logprobs=logprobs,unprocessed_log_normalization_constant=unprocessed_log_normalization_constant,temp_processed_log_normalization_constant=temp_processed_log_normalization_constant,entropy=entropy)
