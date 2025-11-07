# Standard Libraries
import os
import random
import time

# Math Libraries
import numpy as np
import pandas as pd

# Helper Library
from tqdm import tqdm
from parse_utils import parse_answer
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

# Prompting constants and templates
PROMPT = "Can you solve the following math problem? "
AIME = "The solution to this math problem is an integer between 0 and 999."
BASE = " Put your last and final answer within \\boxed{{}}."
COT = " Please reason step by step, and put your last and final answer within \\boxed{{}}."
COT_ALT = " Please explain your reasoning with a detailed, step-by-step solution, and present your last and final answer within \\boxed{{}}."

# Autoregressive Sampler Class
# Stores parameters concerning the LLM, autoregressive sampling, and power sampling
# Includes Functions:
# sample() - Samples from the LLM given a context and max new tokens using either the API or programmatical LLM
class AutoregressiveSampler:
    def __init__(self, api, llm, tokenizer, power_sampling_temperature=1.0, top_k = -1, logprobs=100, token_count=1000, block_size=50, MCMC_steps=5):
        self.api = api
        self.llm = llm
        self.tokenizer = tokenizer
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

# Creates a question that can be inserted into the LLM
# It is a prompt template that adds chain of thought prompting if cot=True
def format_prompt(dataset_name, question, cot=True):
    format_str = PROMPT + question

    # Let the LLM know that AIME answers can only be between 0 and 999
    if dataset_name == "AIME":
        format_str += AIME

    # Enable chain of thought prompting
    if cot:
        format_str += COT
    else:
        format_str += BASE

    return format_str

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

# Performs sliding window power sampling on the given prompt
# Sliding window only performs power sampling on a specific block size of tokens at a time instead of the whole prompt
# Increases the speed of getting a prompt response
# Input is the sampler object, prompt string, temperature, power, total token count to generate, and random seed
# Output is the generated string, total acceptances, and block acceptances
def sliding_window_power_sample(sampler: AutoregressiveSampler, prompt, temperature, power, token_count, seed):
    # Set random seed
    random.seed(seed)

    # Log Probabilites
    logprob = [] # Current list of unscaled log probabilites of the new sample. Length of block_size
    logprob_temp_scaled = [] # Current list of tokens probabilites individually scaled by temperature. Length of block_size
    proposed_logprob = [] # Proposed list of unscaled log probabilites of the new sample. Length of max_new_tokens
    proposed_logprob_temp_scaled = [] # Proposed list of tokens probabilites individually scaled by temperature. Length of max_new_tokens

    # Acceptance parameters
    acceptances = 0
    block_acceptances = []

    # New Context Window to be changed
    context = []

    # Iterate over the number of blocks to be generated
    for block_idx in tqdm(range(sampler.block_num), disable=True):
        # Block Acceptances Ratio 
        block_acceptance = 0

        # print("Input Prompt:", prompt + '\n')

        # Generate next block of tokens as baseline
        # If the programmatical LLM is being used
        tokens_list, token_logprob_list, logprobs_list = sampler.sample(prompt +  sampler.tokenizer.decode(context, skip_special_tokens=False), sampler.block_size)

        # Calculate the initial logprobabilities for the generated block
        logprob, logprob_temp_scaled = logprobs(tokens_list, token_logprob_list, logprobs_list, sampler)
        # Extend the context with the newly generated tokens
        context = tokens_list

        # print("Proposed Response:", sampler.tokenizer.decode(context, skip_special_tokens=True))

        #Iterate over the number of MCMC steps
        for _ in tqdm(range(sampler.MCMC_steps), disable=True):
            # print("\n\nCurrent Block and MCMC Step:", block_idx, _)

            #Find a new point to start a proposal from. Generate idx tokens for the step
            idx = random.randint(0,len(context) -1)
            
            #Set the new context for the proposed block
            context_proposed = context[:-(len(context)-idx)]

            # print("Context Token Length for Proposal:", len(context_proposed))
            # print("Tokens to generate:", len(context) - idx)
            # print("Input Prompt:\n", prompt + sampler.tokenizer.decode(context_proposed, skip_special_tokens=True) + '\n')

            #Generate proposed block of tokens
            proposed_tokens_list, proposed_token_logprob_list, proposed_logprobs_list = sampler.sample(prompt + sampler.tokenizer.decode(context_proposed, skip_special_tokens=False), len(context) - idx)
            
            # Print out the current block, MCMC step, proposed context, tokens to generate, and proposed tokens

            # print("Proposed Response:\n", sampler.tokenizer.decode(proposed_tokens_list, skip_special_tokens=True))
            # assert(len(proposed_tokens_list + context_proposed ) == len(context))
            # print("Tokens generated:", len(proposed_tokens_list))

            # Find the log probabilities of the generated tokens
            proposed_logprob, proposed_logprob_temp_scaled = logprobs(proposed_tokens_list, proposed_token_logprob_list, proposed_logprobs_list, sampler)
            # Calculate the Metro-Hastings acceptance ratio
            # Power Scaled Sequence Log Probability + Temperature Scaled Sequence Log Probability - Current Power Scaled Sequence Log Probability - Current Temperature Scaled Sequence Log Probability
            log_acceptance_ratio = sum(proposed_logprob) + sum(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) - sum(logprob[idx:idx+len(proposed_tokens_list)]) - sum(proposed_logprob_temp_scaled)

            # Check to make sure we are comparing the correct number of elements
            assert(len(proposed_logprob) == len(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) == len(logprob[idx:idx+len(proposed_tokens_list)]) == len(proposed_logprob_temp_scaled))

            # print("Proposed Logprob Sum:", sum(proposed_logprob))
            # print("Current Logprob Temp Scaled Sum:", sum(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]))
            # print("Current Logprob Sum:", sum(logprob[idx:idx+len(proposed_tokens_list)]))
            # print("Proposed Logprob Temp Scaled Sum:", sum(proposed_logprob_temp_scaled))
            # print("Log Acceptance Ratio:", np.exp(log_acceptance_ratio))

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

        block_acceptances.append(block_acceptance)

        prompt = prompt + sampler.tokenizer.decode(context, skip_special_tokens=False)
        # print("EOS Token ID:", sampler.tokenizer.eos_token_id)
        # print("Current Context:\n", context)
        # Check if an EOS token has been generated and end the process if so
        if(sampler.tokenizer.eos_token_id in context):
            return prompt, acceptances, block_acceptances


    # EOS never found, just return the full generated context
    return prompt, acceptances, block_acceptances

def power_sampling(sampler: AutoregressiveSampler, prompt, temperature, power, token_count, seed):
    # Set random seed
    random.seed(seed)

    # Log Probabilites
    logprob = [] # Current list of unscaled log probabilites of the new sample. Length of block_size
    logprob_temp_scaled = [] # Current list of tokens probabilites individually scaled by temperature. Length of block_size
    proposed_logprob = [] # Proposed list of unscaled log probabilites of the new sample. Length of max_new_tokens
    proposed_logprob_temp_scaled = [] # Proposed list of tokens probabilites individually scaled by temperature. Length of max_new_tokens

    # Acceptance parameters
    acceptances = 0

    # New Context Window to be changed
    context = []

    # Iterate over the number of blocks to be generated
    for block_idx in tqdm(range(sampler.block_num), disable=True):
        # Block Acceptances Ratio 
        block_acceptance = 0

        # print("Input Prompt:", prompt + '\n')

        # Generate next block of tokens as baseline
        # If the programmatical LLM is being used
        tokens_list, token_logprob_list, logprobs_list = sampler.sample(prompt +  sampler.tokenizer.decode(context, skip_special_tokens=False), sampler.block_size)

        # Calculate the initial logprobabilities for the generated block
        logprob_initial, logprob_temp_scaled_initial = logprobs(tokens_list, token_logprob_list, logprobs_list, sampler)

        # Extend the initial log probabilities
        logprob = [*logprob, *logprob_initial.tolist()]
        logprob_temp_scaled = [*logprob_temp_scaled, *logprob_temp_scaled_initial.tolist()]

        # Extend the context with the newly generated tokens
        context.extend(tokens_list)

        # print("Proposed Response:", sampler.tokenizer.decode(context, skip_special_tokens=True))

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
            
            # Print out the current block, MCMC step, proposed context, tokens to generate, and proposed tokens

            #print("Proposed Response:\n", sampler.tokenizer.decode(proposed_tokens_list, skip_special_tokens=True))
            print("Initial Context Length: ", len(context))
            print("Given Context Length:", len(context_proposed))
            print("Tokens generated:", len(proposed_tokens_list))

            #assert(len(proposed_tokens_list + context_proposed ) == len(context))

            # Find the log probabilities of the generated tokens
            proposed_logprob, proposed_logprob_temp_scaled = logprobs(proposed_tokens_list, proposed_token_logprob_list, proposed_logprobs_list, sampler)
            # Calculate the Metro-Hastings acceptance ratio
            # Power Scaled Sequence Log Probability + Temperature Scaled Sequence Log Probability - Current Power Scaled Sequence Log Probability - Current Temperature Scaled Sequence Log Probability
            log_acceptance_ratio = sum(proposed_logprob) + sum(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) - sum(logprob[idx:idx+len(proposed_tokens_list)]) - sum(proposed_logprob_temp_scaled)

            # Check to make sure we are comparing the correct number of elements
            assert(len(proposed_logprob) == len(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]) == len(logprob[idx:idx+len(proposed_tokens_list)]) == len(proposed_logprob_temp_scaled))

            print("Proposed Logprob Sum:", sum(proposed_logprob))
            print("Current Logprob Temp Scaled Sum:", sum(logprob_temp_scaled[idx:idx+len(proposed_tokens_list)]))
            print("Current Logprob Sum:", sum(logprob[idx:idx+len(proposed_tokens_list)]))
            print("Proposed Logprob Temp Scaled Sum:", sum(proposed_logprob_temp_scaled))
            print("Log Acceptance Ratio:", np.exp(log_acceptance_ratio))

            # Accept or reject the proposed block based on the acceptance ratio
            if np.random.rand() < np.exp(log_acceptance_ratio):
                # print("Accepted Proposal at index", idx)
                # Ensure the context is updated with the accepted proposal
                context[idx:] = proposed_tokens_list
                
                # Update the logprob lists with the accepted proposal's log probabilities
                print("Logprob length before update:", len(logprob))
                print("Proposed Logprob length:", len(proposed_logprob))
                
                logprob = [*logprob[:idx], *proposed_logprob]
                
                print("Logprob length after update:", len(logprob))
                print("Logprob Temp Scaled length before update:", len(logprob_temp_scaled))
                print("Proposed Logprob Temp Scaled length:", len(proposed_logprob_temp_scaled))
                
                logprob_temp_scaled = [*logprob_temp_scaled[:idx], *proposed_logprob_temp_scaled]
                
                print("Logprob Temp Scaled length after update:", len(logprob_temp_scaled))
                # Collected data about the acceptance ratio for overall run and block
                acceptances += 1

        prompt = prompt + sampler.tokenizer.decode(context, skip_special_tokens=False)
        #print("EOS Token ID:", sampler.tokenizer.eos_token_id)
        #print("Current Context:\n", context)
        # Check if an EOS token has been generated and end the process if so
        if(sampler.tokenizer.eos_token_id in context):
            return prompt, acceptances


    # EOS never found, just return the full generated context
    return prompt, acceptances

def benchmark_preprocessing(dataset_name):
    if(dataset_name == "MATH500"):
        # Load the Math500 dataset
        dataset = datasets.load_dataset("HuggingFaceH4/MATH-500")["test"]
        # Convert all keys to lowercase
        dataset = dataset.map(lambda x: {k.lower(): v for k, v in x.items()})
        # convert answers to a string
        dataset = dataset.cast_column('answer', datasets.Value('string'))
        return dataset

    elif(dataset_name == "AIME"):
        #Load both parts of the AIME tests and concatenate them
        dataset = datasets.concatenate_datasets([datasets.load_dataset("opencompass/AIME2025", "AIME2025-I")["test"], 
                                                datasets.load_dataset("opencompass/AIME2025", "AIME2025-II")["test"]])
        # Convert all keys to lowercase
        dataset = dataset.map(lambda x: {k.lower(): v for k, v in x.items()})
        # convert answers to a string
        dataset = dataset.cast_column('answer', datasets.Value('string'))
        # convert the question column name to "problem"
        dataset = dataset.rename_column("question", "problem")
        return dataset

    else: 
        print("Unknown dataset:", dataset_name)
        return None

# Benchmark the Math500 dataset with different sampling methods
def benchmark_sampling(dataset_name, sampler, chain_of_thought, power_sampling_on = False, power_sampling_windowed_on = False, low_temp_sampling_on = False, naive_sampling_on = False, question_max = 0, seed = 0, output_file_name = "results/math500_power_sampling_results.csv", verbose = 0):
    # Load dataset
    dataset = benchmark_preprocessing(dataset_name)

    # Store results
    results = []    
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    output_file = open(output_file_name, "w")

    # Track number of questions asked
    question_count = 0

    # Iterate over the dataset
    for question_index, question_data in tqdm(enumerate(dataset), desc = "Benchmark on " + dataset_name, disable=True):
        
        if(question_max == question_count and question_max != 0):
            break

        #Extract the problem and question from the dataset
        question = question_data["problem"]
        answer = question_data["answer"]
        
        # Format the prompt with prompt engineering
        formatted_prompt = format_prompt(dataset_name, question, cot=chain_of_thought)

        # Store the prompt and answers in the results csv
        result_row = {
            "question": formatted_prompt,
            "correct_answer": answer
        }

        # Generate a response using the sampler
        if(power_sampling):
            #Time how long it takes to get a response
            start_time = time.time()

            # Send the prompt to the sliding window power sampling function
            power_sampling_output, power_sampling_total_acceptances = power_sampling(sampler, prompt=formatted_prompt, temperature=sampler.power_sampling_temperature, power=1.0, token_count=sampler.token_count, seed=random.randint(0, 10000))
            
            # Find the end time of the power sampling
            end_time = time.time()

            # Parse the answer
            power_sampling_answer = parse_answer(power_sampling_output, answer)
            # print("Parsed Sampling Answer:", power_sampling_answer)

            # Save the results
            result_row["power_sampling_output"] = power_sampling_output
            result_row["power_sampling_output_token_count"] = len(sampler.tokenizer.encode(power_sampling_output))
            result_row["power_sampling_time_to_solution"] = end_time - start_time
            result_row["power_sampling_answer"] = power_sampling_answer
            result_row["power_sampling_total_acceptances"] = power_sampling_total_acceptances

            # TODO: Implement more verbose logging
            if(verbose == 3):
                # Log detailed output for debugging
                # Log the MCMC Block, Index, Proposed Probability, Current Probability, Acceptance Ratio, Random Number Generated, Accepted/Rejected
                pass

        # Generate a response using the sampler
        if(power_sampling_windowed_on):
            #Time how long it takes to get a response
            start_time = time.time()

            # Send the prompt to the sliding window power sampling function
            power_sampling_windowed_output, power_sampling_windowed_total_acceptances, power_sampling_windowed_block_acceptances = sliding_window_power_sample(sampler, prompt=formatted_prompt, temperature=sampler.power_sampling_temperature, power=1.0, token_count=sampler.token_count, seed=random.randint(0, 10000))
            
            # Find the end time of the power sampling
            end_time = time.time()

            # Parse the answer
            power_sampling_windowed_answer = parse_answer(power_sampling_windowed_output, answer)
            # print("Parsed Sampling Answer:", power_sampling_windowed_answer)

            # Save the results
            result_row["power_sampling_windowed_output"] = power_sampling_windowed_output
            result_row["power_sampling_windowed_output_token_count"] = len(sampler.tokenizer.encode(power_sampling_windowed_output))
            result_row["power_sampling_windowed_time_to_solution"] = end_time - start_time
            result_row["power_sampling_windowed_answer"] = power_sampling_windowed_answer
            result_row["power_sampling_windowed_total_acceptances"] = power_sampling_windowed_total_acceptances
            result_row["power_sampling_windowed_block_acceptances"] = power_sampling_windowed_block_acceptances

            # TODO: Implement more verbose logging
            if(verbose == 3):
                # Log detailed output for debugging
                # Log the MCMC Block, Index, Proposed Probability, Current Probability, Acceptance Ratio, Random Number Generated, Accepted/Rejected
                pass

        # Generate a response with just low temperature sampling
        if(low_temp_sampling_on):
            #Time how long it takes to get a response
            start_time = time.time()

            # Prompt the LLM and get the output/answer
            low_temp_tokens_list, _, low_temp_logprob_list= sampler.sample(formatted_prompt, sampler.token_count)
            print() 
            # Find the end time of the low temperature sampling
            end_time = time.time()

            # Parse the answer
            sampler.tokenizer.decode(low_temp_tokens_list, skip_special_tokens=False)
            low_temp_sampling_answer = parse_answer(sampler.tokenizer.decode(low_temp_tokens_list, skip_special_tokens=False), answer)

            # Save the results
            result_row["low_temp_sampling_output"] = sampler.tokenizer.decode(low_temp_tokens_list, skip_special_tokens=False)
            result_row["low_temp_sampling_output_token_count"] = len(low_temp_tokens_list)
            result_row["low_temp_sampling_time_to_solution"] = end_time - start_time
            result_row["low_temp_sampling_answer"] = low_temp_sampling_answer

        if(naive_sampling_on):
            # Save and change the temperature to 1.0 for naive sampling
            saved_temperature = sampler.power_sampling_temperature
            sampler.power_sampling_temperature = 1.0
            
            #Time how long it takes to get a response
            start_time = time.time()

            # Prompt the LLM and get the output/answer
            naive_tokens_list, _, _ = sampler.sample(formatted_prompt, sampler.token_count)

            # Find the end time of the naive sampling
            end_time = time.time()

            # Parse the answer
            naive_sampling_answer = parse_answer(sampler.tokenizer.decode(naive_tokens_list, skip_special_tokens=False), answer)
            # Save the results
            result_row["naive_sampling_output"] = sampler.tokenizer.decode(naive_tokens_list, skip_special_tokens=False)
            result_row["naive_sampling_output_token_count"] = len(naive_tokens_list)
            result_row["naive_sampling_time_to_solution"] = end_time - start_time
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

# Main function to test power sampling
if __name__ == "__main__":
    # Tell Pytorch to use the GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize the random number generator
    seed = 42
    random.seed(seed)

    # Power Sampling Hyperparameters
    token_count = 1500 #total tokens for response
    block_size = 500 # tokens per block. Number of blocks = token_count / block_size
    MCMC_steps = 3 

    # Set whether to use the API server or programmatical LLM
    api_condition = False

    #Sampling parameters for the LLM
    temperature = 0.75
    top_k = 100 # Consider all tokens when -1 or N tokens when N > 0


    # LLM parameters
    model = "Qwen/Qwen3-4B-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code = True)
    skip_tokenizer_init = False
    dtype = "auto"
    quantization = "awq"
    gpu_memory_utilization = 0.7
    max_model_len = 8192

    # If not using an API
    if(api_condition == False):
        # Initialize model
        llm = LLM(model=model, 
                skip_tokenizer_init=skip_tokenizer_init, 
                dtype=dtype, 
                quantization=quantization, 
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                #max_logprobs=tokenizer.vocab_size + token_count + 1000,
                max_logprobs = top_k,
                logprobs_mode='raw_logits')
    # If you are using an API endpoint
    else: 
        llm = None

    #Initialize Autoregressive Sampler
    sampler = AutoregressiveSampler(api_condition,
                                    llm, 
                                    tokenizer,
                                    power_sampling_temperature=temperature,
                                    top_k=top_k,
                                    token_count=token_count,
                                    block_size=block_size,
                                    MCMC_steps=MCMC_steps
                                    )

    # Test MATH500 Benchmark
    dataset_name = "AIME"
    power_sampling_on = True
    power_sampling_windowed_on = False
    low_temp_sampling_on = False
    naive_sampling_on = False
    chain_of_thought = False
    #for temp in [0.25, 0.5, 0.75]:
    for temp in [0.5]:
        sampler.power_sampling_temperature = temp
        benchmark_sampling(dataset_name, sampler, chain_of_thought, power_sampling_on, power_sampling_windowed_on, low_temp_sampling_on, naive_sampling_on, question_max = 1, output_file_name = f"results/{dataset_name}_power_sampling_results_temp_{temp}.csv", seed=seed)


    # Call the power_sample function
    # prompt_response, total_acceptances, block_acceptances = sliding_window_power_sample(sampler, prompt="Once upon a time", temperature=temperature, power=1.0, token_count=token_count, seed=seed)
    # print("Generated Text:", prompt_response)
    # print("Total Acceptances:", total_acceptances)
    # print("Block Acceptances:", block_acceptances)