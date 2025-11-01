# Standard Libraries
import os
import random
import time
import re

# Math Libraries
import numpy as np
import pandas as pd

# Helper Library
from tqdm import tqdm
# Inference Library
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput, CompletionOutput
from transformers import AutoTokenizer
import requests 

# Pytorch Library
import torch
from torch.nn import functional as F
# Benchmarking library
import datasets
# Parsing Library
import regex

# Prompting constants and templates
PROMPT = "Can you solve the following math problem? "
BASE = " Put your final answer within \\boxed{{}}."
COT = " Please reason step by step, and put your final answer within \\boxed{{}}."
COT_ALT = " Please explain your reasoning with a detailed, step-by-step solution, and present your final answer within \\boxed{{}}."


class AutoregressiveSampler:
    def __init__(self, api, llm, tokenizer, power_sampling_temperature=1.0, logprobs=100, detokenize=False, token_count=1000, block_size=50, MCMC_steps=5):
        self.api = api
        self.llm = llm
        self.tokenizer = tokenizer
        self.power_sampling_temperature = power_sampling_temperature
        self.detokenize = detokenize
        self.token_count = token_count
        self.block_size = block_size
        self.MCMC_steps = MCMC_steps
        self.block_num = token_count // block_size
        self.api_url = "http://localhost:8000/v1/completions"

    def sample(self, context, max_new_tokens):
        if(self.api == True):
            # Use the vLLM API to generate a response
            # Create payload
            payload = {
                "model": "Qwen/Qwen3-4B-AWQ",
                "prompt": context,
                "max_tokens": max_new_tokens,
                "temperature": self.power_sampling_temperature,
                "logprobs": 5,  # Number of logprobs to return per token
                "stop": None
            }

            # Send the request to the API server
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # Raise an error for bad status codes
            result = response.json()

            # Extract text and logprobs from the OpenAI-compatible response
            choice = result["choices"][0]
            logprobs = choice.get("logprobs") # Use .get for safety
            tokens = logprobs["tokens"]

            print("Tokens:", tokens)
            print("Logprobs:", logprobs)

            # This part needs to be properly implemented to handle the API response
            # and make it compatible with the rest of your script.
            # Returning a placeholder to avoid an immediate crash.
            return tokens, logprobs


        else:
            # Prepare the context as a TokensPrompt if it's a list of token IDs
            if isinstance(context, list):
                context = TokensPrompt(prompt_token_ids=context)
            
            # Set the sampling parameters of the LLM
            sample_params = SamplingParams(temperature=self.power_sampling_temperature, max_tokens=max_new_tokens, logprobs=-1, detokenize=self.detokenize)
            
            # Generate a new response from the LLM
            llm_output = self.llm.generate(context, sampling_params=sample_params)
            tokens = llm_output[0].outputs[0].token_ids
            logprobs = llm_output[0].outputs[0].logprobs
            return tokens, logprobs

def parse_answer(text):
    """
    Parse the final answer from generated text.
    Looks for answers in \\boxed{} format and extracts the content.
    
    Args:
        text (str): The generated text containing the answer
        
    Returns:
        str: The extracted answer, or "No answer found" if no boxed answer is found
    """
    
    # Look for \boxed{...} pattern
    boxed_pattern = r'\\boxed\{((?:[^{}]+|(?R))*)\}'
    matches = regex.findall(boxed_pattern, text)

    if matches:
        # Return the last boxed answer found (in case there are multiple)
        return matches[-1].strip()
    
    # If no boxed answer found, look for other common answer patterns
    # Look for "The answer is ..." or "Answer: ..." patterns
    answer_patterns = [
        r'(?:The answer is|Answer:)\s*([^\n.]+)',
        r'(?:Final answer|Final result):\s*([^\n.]+)',
        r'(?:Therefore|So),?\s*([^\n.]+)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    
    # If still no answer found, try to extract the last line that looks like an answer
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith(('Step', 'First', 'Next', 'Then', 'Finally', 'Therefore', 'So')):
            # Check if it looks like a mathematical answer
            if any(char.isdigit() or char in '+-*/=()[]{}' for char in line):
                return line
    
    return "No answer found"

def format_prompt(question, cot=True):
    format_str = PROMPT + question
    if cot:
        format_str+=COT
    else:
        format_str+=BASE
    return format_str

# Find the output log probabilities of the token sequences for both the p_temp and p_power distributions
def logprobs(token_ids, logprobs_list, sampler):
    # Initialize tensor with -inf (shape: num_tokens x vocab_size)
    logits = torch.full((len(token_ids), len(logprobs_list[0])), float('-inf'))
    
    # Fill the tensor with logprobs from the dicts (access .logprob from Logprob objects)
    for i, logprob_dict in enumerate(logprobs_list):
        for token_id, logprob_obj in logprob_dict.items():
            logits[i, token_id] = logprob_obj.logprob  # Extract the float logprob
    
    
    # Scale the raw logits by the temperature
    scaled_logits = logits / sampler.power_sampling_temperature

    # Compute logsumexp (normalization constant) for each position sum over vocab of exponetiated logits
    log_Z = torch.logsumexp(logits, dim=-1)  # Shape: (num_tokens,)
    log_Z_scaled = torch.logsumexp(scaled_logits, dim=-1)  # Shape: (num_tokens,)
    

    # Extract log probs for only the generated tokens
    indices = torch.arange(len(token_ids))
    # log_probs is the power sampled version = log(softmax(logit_selected)^(1/T)) = (1/T)(logits - log_Z)
    log_probs = (1/sampler.power_sampling_temperature) * (logits[indices, token_ids] - log_Z)
    log_probs_temp_scaled = scaled_logits[indices, token_ids] - log_Z_scaled

    #Print out the selected token probabilites
    # print("Selected Token IDs:", token_ids)
    # print("Logits for selected tokens:", logits[indices, token_ids])
    # print("Scaled Logits for selected tokens:", scaled_logits[indices, token_ids])

    # Print out the log_z and log_z_scaled
    # print("Log Z values:", log_Z)
    # print("Log Z Scaled values:", log_Z_scaled)

    # Print out the token
    # print("Log Probs for generated tokens:", log_probs, "\nLog Probs for generated tokens (Temp Scaled):", log_probs_temp_scaled)

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
    if(sampler.detokenize == False):
        context = sampler.tokenizer.encode(prompt)
    else:
        context = prompt
    prompt_length = len(context)
    
    print(context)

    acceptances = 0
    block_acceptances = []

    # Iterate over the number of blocks to be generated
    for block_idx in tqdm(range(sampler.block_num)):
        # Block Acceptances Ratio 
        block_acceptance = 0

        # Generate next block of tokens as baseline
        # If the programmatical LLM is being used
        tokens_list, logprobs_list = sampler.sample(context, sampler.block_size)
        
        # Calculate the inital logprobabilities for the generated block
        logprob, logprob_temp_scaled = logprobs(tokens_list, logprobs_list, sampler)
        context.extend(tokens_list)
        #Iterate over the number of MCMC steps
        for _ in tqdm(range(sampler.MCMC_steps)):
            #Find a new point to start a proposal from. Generate idx tokens for the step
            idx = random.randint(0,sampler.block_size -1)
            
            #Set the new context for the proposed block
            context_proposed = context[:-(sampler.block_size-idx)]
            #Generate proposed block of tokens
            proposed_tokens_list, proposed_logprobs_list = sampler.sample(context_proposed, sampler.block_size-idx)

            # Find the log probabilities of the generated tokens
            proposed_logprob, proposed_logprob_temp_scaled = logprobs(proposed_tokens_list, proposed_logprobs_list, sampler)
            # Calculate the Metro-Hastings acceptance ratio
            # Power Scaled Sequence Log Probability + Temperature Scaled Sequence Log Probability - Current Power Scaled Sequence Log Probability - Current Temperature Scaled Sequence Log Probability
            log_acceptance_ratio = sum(proposed_logprob) + sum(proposed_logprob_temp_scaled) - sum(logprob[idx:]) - sum(logprob_temp_scaled[idx:])

            # Accept or reject the proposed block based on the acceptance ratio
            if np.random.rand() < np.exp(log_acceptance_ratio):
                # Ensure the context is updated with the accepted proposal
                context[-(sampler.block_size - idx):] = proposed_tokens_list

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

def math500_benchmark_sampling(sampler, power_sampling = False, low_temp_sampling = False, naive_sampling = False, question_max = 0, seed = 0, output_file_name = "results/math500_power_sampling_results.csv", verbose = 0):
    # Load the Math500 dataset
    dataset = datasets.load_dataset("HuggingFaceH4/MATH-500")["test"]
    # Convert all keys to lowercase
    dataset = dataset.map(lambda x: {k.lower(): v for k, v in x.items()})
    # convert answers to a string
    dataset = dataset.cast_column('answer', datasets.Value('string'))

    # Store results
    results = []    
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    output_file = open(output_file_name, "w")

    # Track number of questions asked
    question_count = 0

    # Iterate over the dataset
    for question_index, question_data in tqdm(enumerate(dataset), desc = "Benchmark on MATH"):
        
        if(question_max == question_count and question_max != 0):
            break

        #Extract the problem and question from the dataset
        question = question_data["problem"]
        answer = question_data["answer"]
        
        # Format the prompt with prompt engineering
        formatted_prompt = format_prompt(question, cot=True)

        # Store the prompt and answers in the results csv
        result_row = {
            "question": question,
            "correct_answer": answer
        }

        # Generate a response using the sampler
        if(power_sampling):
            # Send the prompt to the sliding window power sampling function
            power_sampling_output, power_sampling_total_acceptances, power_sampling_block_acceptances = sliding_window_power_sample(sampler, prompt=formatted_prompt, temperature=sampler.power_sampling_temperature, power=1.0, token_count=sampler.token_count, seed=random.randint(0, 10000))
            # Parse the answer
            power_sampling_answer = parse_answer(power_sampling_output)
            # Save the results
            result_row["power_sampling_output"] = power_sampling_output
            result_row["power_sampling_answer"] = power_sampling_answer
            result_row["power_sampling_total_acceptances"] = power_sampling_total_acceptances
            result_row["power_sampling_block_acceptances"] = power_sampling_block_acceptances

            if(verbose == 3):
                # Log detailed output for debugging
                # Log the MCMC Block, Index, Proposed Probability, Current Probability, Acceptance Ratio, Random Number Generated, Accepted/Rejected
                pass

        # Generate a response with just low temperature sampling
        if(low_temp_sampling):
            # Prompt the LLM and get the output/answer
            # Use the raw tokens or text depending on if the tokenizer is skipped
            if(sampler.skip_special_tokens == True):
                context = sampler.tokenizer.encode(formatted_prompt)
            else:
                context = formatted_prompt
            low_temp_tokens_list, low_temp_logprobs_list = sampler.sample(context, sampler.token_count)
            # Decode the output if the tokenizer was skipped
            if(sampler.skip_tokenizer_init == True):
                low_temp_tokens_list = sampler.tokenizer.decode(low_temp_tokens_list, skip_special_tokens=True)

            # Parse the answer
            low_temp_sampling_answer = parse_answer(low_temp_tokens_list)

            # Save the results
            result_row["low_temp_sampling_output"] = low_temp_tokens_list
            result_row["low_temp_sampling_answer"] = low_temp_sampling_answer

        if(naive_sampling):
            # Save and change the temperature to 1.0 for naive sampling
            saved_temperature = sampler.power_sampling_temperature
            sampler.power_sampling_temperature = 1.0
            
            # Use the raw tokens or text depending on if the tokenizer is skipped
            if(sampler.skip_special_tokens == True):
                context = sampler.tokenizer.encode(formatted_prompt)
            else:
                context = formatted_prompt
            # Prompt the LLM and get the output/answer
            naive_tokens_list, naive_logprobs_list = sampler.sample(context, sampler.token_count)

            # Decode the output if the tokenizer was skipped
            if(sampler.skip_tokenizer_init == True):
                naive_tokens_list = sampler.tokenizer.decode(naive_tokens_list, skip_special_tokens=True)

            # Parse the answer
            naive_sampling_answer = parse_answer(naive_tokens_list)
            # Save the results
            result_row["naive_sampling_output"] = naive_tokens_list
            result_row["naive_sampling_answer"] = naive_sampling_answer
            # Set the temperature back to original
            sampler.power_sampling_temperature = saved_temperature

        # Write the question and final answer to the output file
        results.append(result_row)
        # Write to CSV after each iteration, only write header for first row
        df = pd.DataFrame([result_row])
        df.to_csv(output_file, index=False, header=(question_index==0))
        output_file.flush()
        os.fsync(output_file.fileno())

        #Increment the question count
        question_count += 1

if __name__ == "__main__":
    # Initialize the random number generator
    seed = 42
    random.seed(seed)

    # Power Sampling Hyperparameters
    token_count = 500 #total tokens for response
    block_size = 50 # tokens per block. Number of blocks = token_count / block_size
    MCMC_steps = 2 

    # Set wether to use the API server or programmatical LLM
    api_condition = True

    #Sampling parameters for the LLM
    temperature = 0.5
    detokenize = False
    
    # LLM parameters
    model = "Qwen/Qwen3-4B-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code = True)
    skip_tokenizer_init = True
    dtype = "auto"
    quantization = None
    gpu_memory_utilization = 0.9

    # If not using an API
    if(api_condition == False):
        # Initalize model
        llm = LLM(model=model, 
                skip_tokenizer_init=skip_tokenizer_init, 
                dtype=dtype, 
                quantization=quantization, 
                gpu_memory_utilization=gpu_memory_utilization,
                max_logprobs=tokenizer.vocab_size + token_count + 1000,
                logprobs_mode='raw_logits')
    # If you are using an API endpoint
    else: 
        llm = None

    #Initalize Autogressive Sampler
    sampler = AutoregressiveSampler(api_condition,
                                    llm, 
                                    tokenizer,
                                    power_sampling_temperature=temperature,
                                    detokenize=detokenize,
                                    token_count=token_count,
                                    block_size=block_size,
                                    MCMC_steps=MCMC_steps
                                    )

    # Test MATH500 Benchmark
    power_sampling_on = True
    low_temp_sampling_on = False
    naive_sampling_on = False
    math500_benchmark_sampling(sampler, power_sampling_on, low_temp_sampling_on, naive_sampling_on, question_max = 1, output_file_name = "results/math500_power_sampling_results.csv", seed=seed)

    # Test vLLM sampling
    #vllm_test(sampler)

    # Call the power_sample function
    # prompt_response, total_acceptances, block_acceptances = sliding_window_power_sample(sampler, prompt="Once upon a time", temperature=temperature, power=1.0, token_count=token_count, seed=seed)
    # print("Generated Text:", prompt_response)
    # print("Total Acceptances:", total_acceptances)
    # print("Block Acceptances:", block_acceptances)