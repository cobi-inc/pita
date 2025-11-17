# Standard Libraries
import os
import random
import time

# Math Libraries
import numpy as np
import pandas as pd

# Helper Library
from tqdm import tqdm
from src.utils.parse_utils import parse_answer
# Inference Library
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput, CompletionOutput
from transformers import AutoTokenizer
import requests 

# Pytorch Library
import torch
# Benchmarking library
import datasets
# Parsing Library
import regex

# User Files
from src.utils.benchmarking_utils import benchmark_sampling

# Autoregressive Sampler Class
# Stores parameters concerning the LLM, autoregressive sampling, and power sampling
# Includes Functions:
# sample() - Samples from the LLM given a context and max new tokens using either the API or programmatical LLM
class AutoregressiveSampler:
    def __init__(self, api, llm, tokenizer, enable_thinking=False, power_sampling_temperature=1.0, top_k = -1, logprobs=100, token_count=1000, block_size=50, MCMC_steps=5):
        self.api = api
        self.llm = llm
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.power_sampling_temperature = power_sampling_temperature
        self.top_k = top_k
        self.token_count = token_count
        self.block_size = block_size
        self.MCMC_steps = MCMC_steps
        self.block_num = token_count // block_size
        self.api_url = "http://localhost:8000/v1/completions"

    # Take in the context (string) and max_new_tokens (int)
    # Returns the generated tokens. the chosen token logprobs, and all the logprobs as lists to the user
    def sample(self, context, max_new_tokens):
        if(self.api == True):
            # Use the vLLM API to generate a response
            # Create payload
            payload = {
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "prompt": context,
                "max_tokens": max_new_tokens,
                "temperature": self.power_sampling_temperature,
                "logprobs": 5,  #  size of self.tokenizer.vocab_size crashes server due to memory leak  # Number of logprobs to return per token
                "stop": None
            }

            # Send the request to the API server
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # Raise an error for bad status codes
            result = response.json()

            # Extract text and logprobs from the OpenAI-compatible response
            output = result["choices"][0].get("logprobs")
            tokens = output["tokens"]
            token_logprobs = output["token_logprobs"] # could be used later to speed up computations
            logprobs = [[list(logprobs[i].values())] for i in range(len(output["top_logprobs"]))]

            return tokens, token_logprobs, logprobs

        else:
            # Prepare the context as a TokensPrompt if it's a list of token IDs
            if isinstance(context, list):
                context = TokensPrompt(prompt_token_ids=context)
            
            # Set the sampling parameters of the LLM
            sample_params = SamplingParams( temperature=self.power_sampling_temperature, 
                                            top_k=self.top_k, 
                                            max_tokens=max_new_tokens, 
                                            logprobs=self.top_k, 
                                            stop_token_ids =[self.tokenizer.eos_token_id])

            # Generate a new response from the LLM
            llm_output = self.llm.generate(context, sampling_params=sample_params)
            tokens = llm_output[0].outputs[0].token_ids

            logprobs = [[obj.logprob for obj in position_dict.values()] for position_dict in llm_output[0].outputs[0].logprobs]
            token_logprobs = [llm_output[0].outputs[0].logprobs[i][tokens[i]].logprob for i in range(len(tokens))]

            return tokens, token_logprobs, logprobs

# Find the output log probabilities of the token sequences for both the p_temp and p_power distributions
# token_ids is a list of the chosen tokens
# lopprobs_list is a list of lists of the logprobs of each possible token for a given position in the token sequence from vLLM
def logprobs(token_ids, token_logprob, logprobs_list, sampler):
    # Initialize normalization tensors with -inf (shape: num_tokens)
    log_Z = torch.empty(len(token_ids))
    log_Z_scaled = torch.empty(len(token_ids))
    token_logprob_tensor = torch.FloatTensor(token_logprob)

    # Calculate the normalization terms from the logprob dictionaries
    for i in range(len(logprobs_list)):
        current_token_logits = torch.FloatTensor(logprobs_list[i])
        log_Z[i] = torch.logsumexp(current_token_logits, dim=0)
        log_Z_scaled[i] = torch.logsumexp(current_token_logits / sampler.power_sampling_temperature, dim=0)

    # log_probs is the power sampled version =
    # log(softmax(logit_selected)^(1/T)) = 
    # (1/T) * log((e^(logit_selected) / sum(e^all_logits)) = 
    # (1/T)(logit_selected - log(sum(e^all_logits))
    # The scaled version uses already temperature scaled logits
    # Scaled = log(softmax(logit_selected / T)) =
    # 1/T * logit_selected - log(sum(e^(all_logits / T)))
    log_probs = (1/sampler.power_sampling_temperature) * (token_logprob_tensor - log_Z)
    log_probs_temp_scaled = (1/sampler.power_sampling_temperature) * token_logprob_tensor - log_Z_scaled

    return log_probs, log_probs_temp_scaled

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
    for block_idx in tqdm(range(sampler.block_num), disable=True):
        # Block Acceptances Ratio 
        block_acceptance = 0

        # Generate next block of tokens as baseline
        # If the programmatical LLM is being used
        tokens_list, token_logprob_list, logprobs_list = sampler.sample(prompt + token_history + sampler.tokenizer.decode(context, skip_special_tokens=False), sampler.block_size)
        
        # Record how many tokens have been generated
        total_tokens_generated += len(tokens_list)

        # Calculate the initial logprobabilities for the generated block
        logprob, logprob_temp_scaled = logprobs(tokens_list, token_logprob_list, logprobs_list, sampler)
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
            proposed_logprob, proposed_logprob_temp_scaled = logprobs(proposed_tokens_list, proposed_token_logprob_list, proposed_logprobs_list, sampler)
           
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

def power_sampling(
                sampler: AutoregressiveSampler, 
                prompt
                ):
    # Statistic Collection
    total_tokens_generated = 0
    acceptances = 0
    block_acceptances = []

    # Log Probabilites
    logprob = [] # Current list of unscaled log probabilites of the new sample. Length of block_size
    logprob_temp_scaled = [] # Current list of tokens probabilites individually scaled by temperature. Length of block_size
    proposed_logprob = [] # Proposed list of unscaled log probabilites of the new sample. Length of max_new_tokens
    proposed_logprob_temp_scaled = [] # Proposed list of tokens probabilites individually scaled by temperature. Length of max_new_tokens

    # New Context Window to be changed
    context = []

    block_count = sampler.token_count // sampler.block_size
    # Iterate over the number of blocks to be generated
    for block_idx in tqdm(range(block_count), disable=True):
        # Block Acceptances Ratio 
        block_acceptance = 0

        # Generate next block of tokens as baseline
        # If the programmatical LLM is being used
        tokens_list, token_logprob_list, logprobs_list = sampler.sample(prompt +  sampler.tokenizer.decode(context, skip_special_tokens=False), sampler.block_size)

        # Record how many tokens have been generated
        total_tokens_generated += len(tokens_list)

        # Calculate the initial logprobabilities for the generated block
        logprob_initial, logprob_temp_scaled_initial = logprobs(tokens_list, token_logprob_list, logprobs_list, sampler)

        # Extend the initial log probabilities
        logprob = [*logprob, *logprob_initial.tolist()]
        logprob_temp_scaled = [*logprob_temp_scaled, *logprob_temp_scaled_initial.tolist()]

        # Extend the context with the newly generated tokens
        context.extend(tokens_list)

        #Iterate over the number of MCMC steps
        for _ in tqdm(range(sampler.MCMC_steps), disable=True):
            # print("\n\nCurrent Block and MCMC Step:", block_idx, _)

            #Find a new point to start a proposal from. Generate idx tokens for the step
            idx = random.randint(1, len(context) - 1)
            
            #Set the new context for the proposed block
            context_proposed = context[:idx]

            print("Context Token Length for Proposal:", len(context_proposed))
            print("Tokens to generate:", len(context) - idx)
            #print("Input Prompt:\n", prompt + sampler.tokenizer.decode(context_proposed, skip_special_tokens=True) + '\n')

            #Generate proposed block of tokens
            proposed_tokens_list, proposed_token_logprob_list, proposed_logprobs_list = sampler.sample(prompt + sampler.tokenizer.decode(context_proposed, skip_special_tokens=False), len(context) - idx)
            
            # Record how many tokens have been generated
            total_tokens_generated += len(proposed_tokens_list)

            # Find the log probabilities of the generated tokens
            proposed_logprob, proposed_logprob_temp_scaled = logprobs(proposed_tokens_list, proposed_token_logprob_list, proposed_logprobs_list, sampler)
            
            # Calculate the Metro-Hastings acceptance ratio
            # Power Scaled Sequence Log Probability + Temperature Scaled Sequence Log Probability - Current Power Scaled Sequence Log Probability - Current Temperature Scaled Sequence Log Probability
            log_acceptance_ratio = sum(proposed_logprob) + sum(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) - sum(logprob[idx:idx+len(proposed_tokens_list)]) - sum(proposed_logprob_temp_scaled)

            # Check to make sure we are comparing the correct number of elements
            assert(len(proposed_logprob) == len(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) == len(logprob[idx:idx+len(proposed_tokens_list)]) == len(proposed_logprob_temp_scaled))

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

        # Check if an EOS token has been generated and end the process if so
        if(sampler.tokenizer.eos_token_id in context):
            return sampler.tokenizer.decode(context, skip_special_tokens=False), acceptances, block_acceptances, total_tokens_generated


    # EOS never found, just return the full generated context
    return sampler.tokenizer.decode(context, skip_special_tokens=False), acceptances, block_acceptances, total_tokens_generated

