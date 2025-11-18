# System Libraires
import os
import time
import random
# Data Manipulation Libraries
import pandas as pd
# Util Libraries
from tqdm import tqdm
#Tokenizing Libraries
from transformers import AutoTokenizer
# Benchmarking library
import datasets

#Import Custom Libraries
from src.utils.parse_utils import parse_answer
from src.sampling.power_sample import power_sampling, sliding_window_power_sample

# Prompting constants and templates
CONFIDENCE_BOOSTER = "You are very knowledgeable. An expert. Think and respond with confidence. "
PROMPT = "Can you solve the following math problem? "
AIME = "The solution to this math problem is an integer between 0 and 999. "
BASE = "Put your last and final answer within \\boxed{{}}. "
COT = "Please reason step by step, and put your last and final answer within \\boxed{{}}. "
COT_ALT = " Please explain your reasoning with a detailed, step-by-step solution, and present your last and final answer within \\boxed{{}}. "

class Benchmarking:
    def __init__(self, dataset_name, 
                sampler,
                enable_thinking=False,
                chain_of_thought=False, 
                power_sampling_on=False, 
                power_sampling_windowed_on=False, 
                low_temp_sampling_on=False, 
                naive_sampling_on=False,
                question_max=0,
                seed=0,
                output_file_name="results/benchmarking_results.csv",
                verbose=0):

        self.dataset_name = dataset_name
        self.sampler = sampler
        self.chain_of_thought = chain_of_thought
        self.power_sampling_on = power_sampling_on
        self.power_sampling_windowed_on = power_sampling_windowed_on
        self.low_temp_sampling_on = low_temp_sampling_on
        self.naive_sampling_on = naive_sampling_on
        self.question_max = question_max
        self.seed = seed
        self.output_file_name = output_file_name
        self.verbose = verbose

def load_dataset(dataset_name):
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

# Manual prompt creation from a question that can be inserted into the LLM
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
    
    # Enable a confidence boost
    confidence = True
    if confidence:
        format_str = CONFIDENCE_BOOSTER + format_str

    return format_str

# Use the tokenizer.apply_chat_template_prompt to format messages for chat models
def tokenizer_chat_template_prompt(tokenizer, dataset_name, question, boost_confidence, enable_cot, enable_thinking):
    # Create messages for the chat template
    if(dataset_name == "AIME"):
        system_message = PROMPT + AIME 
    else:
        system_message = PROMPT

    # Enable chain of thought prompting
    if(enable_cot):
        system_message = COT
    else:
        system_message = BASE

    # Enable a confidence boost
    if boost_confidence:
        system_message = CONFIDENCE_BOOSTER + system_message

    # Create the message format for apply_chat_template function
    messages = [
        {
            "role": "system",
            # Crucial for benchmarks: explicitly ask for reasoning and boxed format
            "content": system_message
        },
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = enable_thinking
    )

    print(prompt)

    return prompt

# Benchmark the Math500 dataset with different sampling methods
def benchmark_sampling(dataset_name, sampler, chain_of_thought, power_sampling_on = False, power_sampling_windowed_on = False, low_temp_sampling_on = False, naive_sampling_on = False, question_max = 0, seed = 0, output_file_name = "results/math500_power_sampling_results.csv", verbose = 0):
    # Load dataset
    dataset = load_dataset(dataset_name)

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
        # formatted_prompt = format_prompt(dataset_name, question, cot=chain_of_thought)

        # Format the prompt with the tokenizer chat template
        formatted_prompt = tokenizer_chat_template_prompt(sampler.tokenizer, dataset_name, question, False, False, False)

        # Store the prompt and answers in the results csv
        result_row = {
            "question": formatted_prompt,
            "correct_answer": answer
        }

        # Generate a response using the sampler
        if(power_sampling_on):
            #Time how long it takes to get a response
            start_time = time.time()

            # Send the prompt to the sliding window power sampling function
            power_sampling_output, power_sampling_total_acceptances, power_sampling_block_acceptances, power_sampling_index_proposals, power_sampling_total_token_count = power_sampling(sampler, prompt=formatted_prompt)
            
            # Find the end time of the power sampling
            end_time = time.time()

            # Parse the answer
            power_sampling_answer = parse_answer(power_sampling_output, answer)
            # print("Parsed Sampling Answer:", power_sampling_answer)

            # Save the results
            result_row["power_sampling_output"] = power_sampling_output
            result_row["power_sampling_output_token_count"] = len(sampler.tokenizer.encode(power_sampling_output))
            result_row["power_sampling_total_token_count"] = power_sampling_total_token_count
            result_row["power_sampling_time_to_solution"] = end_time - start_time
            result_row["power_sampling_answer"] = power_sampling_answer
            result_row["power_sampling_total_acceptances"] = power_sampling_total_acceptances
            result_row["power_sampling_block_acceptances"] = power_sampling_block_acceptances
            result_row["power_sampling_index_proposals"] = power_sampling_index_proposals

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
            power_sampling_windowed_output, power_sampling_windowed_total_acceptances, power_sampling_windowed_block_acceptances, power_sampling_windowed_total_token_count = sliding_window_power_sample(sampler, prompt=formatted_prompt)
            
            # Find the end time of the power sampling
            end_time = time.time()

            # Parse the answer
            power_sampling_windowed_answer = parse_answer(power_sampling_windowed_output, answer)
            # print("Parsed Sampling Answer:", power_sampling_windowed_answer)

            # Save the results
            result_row["power_sampling_windowed_output"] = power_sampling_windowed_output
            result_row["power_sampling_windowed_output_token_count"] = len(sampler.tokenizer.encode(power_sampling_windowed_output))
            result_row["power_sampling_windowed_total_token_count"] = power_sampling_windowed_total_token_count
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
