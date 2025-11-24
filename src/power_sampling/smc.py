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

class SequentialMonteCarlo:
    def __init__(self, sampler: AutoregressiveSampler):
        self.sampler = sampler

    def generate(self, prompt: str, num_particles: int = None, tokens_per_step: int = None):

        # Initialize particles to be the prompt 
        particles = [self.sampler.tokenizer.encode(prompt) for _ in range(num_particles)]
        
        # cumulative log weights for each particle
        cum_log_weights = np.zeros(num_particles, dtype=float)

        total_tokens_generated = 0
        max_tokens = self.sampler.token_count

        while total_tokens_generated < max_tokens:
            # determine how many tokens to generate in this step
            max_new_tokens = min(tokens_per_step, max_tokens - total_tokens_generated)

            for i in range(num_particles):
                context = particles[i]
                tokens, token_logprobs, _ = self.sampler.sample(context, max_new_tokens) # get tokens and their logprobs

                # append generated tokens 
                particles[i] = context + tokens 

                # for each particle generated block find cumulative log probability 
                block_logprob = float(np.sum(token_logprobs)) if len(token_logprobs) > 0 else 0.0
                cum_log_weights[i] += block_logprob # accumulate from previous block generation 

            max_cum_log_weight = np.max(cum_log_weights)
            weight = np.exp(cum_log_weights - max_cum_log_weight)
            normalized_weights = weight / np.sum(weight)

            # Resample indices according to normalized weights (multinomial resampling)
            indices = np.random.choice(num_particles, size=num_particles, p=normalized_weights)
            particles = [particles[idx] for idx in indices]
            cum_log_weights = cum_log_weights[indices]

            total_tokens_generated += max_new_tokens

        # Select the best particle by cumulative log-weight and return decoded text
        best_idx = int(np.argmax(cum_log_weights))
        best_particle_ids = particles[best_idx]
        return self.sampler.tokenizer.decode(best_particle_ids, skip_special_tokens=True)


       

