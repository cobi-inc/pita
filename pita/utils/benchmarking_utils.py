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
from pita.inference.LLM_backend import AutoregressiveSampler
from pita.utils.parse_utils import parse_answer
from pita.sampling.power_sample import Power_Sampling
from pita.sampling.smc import Sequential_Monte_Carlo
from pita.sampling.best_of import Best_of_N

# Prompting constants and templates
#MATH_SYSTEM_MESSAGE = "You are a very knowledgeable math expert. Think and respond with confidence."
MATH_SYSTEM_MESSAGE = ""
MATH_PRE_QUESTION = "Can you solve the following math problem? "
AIME_PRE_QUESTION = "The solution to the math problem is an integer between 0 and 999. "
MATH_ANSWER_FORMAT = " Put your final answer within \\boxed{{}}."
COT = "Reason step by step, and put your last and final answer within \\boxed{{}}. "
COT_ALT = " Please explain your reasoning with a detailed, step-by-step solution, and present your last and final answer within \\boxed{{}}. "

# Use the tokenizer.apply_chat_template_prompt to format messages for chat models
def tokenizer_chat_template(
    tokenizer: AutoTokenizer,
    enable_thinking: bool,
    system_message: str, 
    user_message: str,
) -> str:

    # Create the message format for apply_chat_template function
    messages = [
        {
            "role": "system",
            # Crucial for benchmarks: explicitly ask for reasoning and boxed format
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    # Apply the chat template to create the final prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = enable_thinking
    )

    return prompt

def format_dataset( 
    dataset: datasets.Dataset,
    pre_question: str,
    post_question: str
) -> {str | list[str], str | list[str]}:
    # Lists to store the questions and answers
    question_list = []
    answer_list = []

    # Iterate through the dataset and format each question
    for(dataset_index, data) in enumerate(dataset):
        # Extract the problem and answer from the dataset
        problem = data["problem"]
        answer = data["answer"]

        # Format the question with pre and post templates
        formatted_question = pre_question + problem + post_question

        # Store back in dataset
        question_list.append(formatted_question)
        answer_list.append(answer)

    return question_list, answer_list

# Load a dataset based on name
def load_benchmark(
    dataset_name: str
) -> {str | list[str], str | list[str]}:
        # Load either the MATH500 or AIME dataset
        if(dataset_name == "MATH500"):
            # Load the Math500 dataset
            dataset = datasets.load_dataset("HuggingFaceH4/MATH-500")["test"]
            # Convert all keys to lowercase
            dataset = dataset.map(lambda x: {k.lower(): v for k, v in x.items()})
            # convert answers to a string
            dataset = dataset.cast_column('answer', datasets.Value('string'))

            # Create the system message, pre, and post question templates
            system_message = MATH_SYSTEM_MESSAGE
            pre_question = MATH_PRE_QUESTION 
            post_question = MATH_ANSWER_FORMAT

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
            
            # Create the system message, pre, and post question templates
            system_message = MATH_SYSTEM_MESSAGE
            pre_question = MATH_PRE_QUESTION
            post_question = MATH_ANSWER_FORMAT

        else: 
            raise ValueError(f"Dataset {dataset_name} not supported for benchmarking.")

        # Format the dataset and return the system message, question list, and answer list
        question_list, answer_list = format_dataset(dataset, pre_question, post_question)
        return system_message, question_list, answer_list

# Benchmark the Math500 dataset with different sampling methods
def benchmark_sampling(
    llm: AutoregressiveSampler,
    system_message: str,
    question_list: list[str],
    answer_list: list[str],
    enable_thinking: bool, 
    chat_template: bool,
    sampling_techniques: list[bool], # greedy sampling, low temp sampling, power sampling, smc, best of n, power sampling and smc
    max_questions: int = 0, 
    output_file_name: str = "math500_power_sampling_results.csv",
    **kwargs
) -> None:
    # Store results
    results = []    
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    output_file = open(output_file_name, "w")

    # Create a template for the log file path
    log_file_path_template = os.path.join(kwargs["log_file_path"], "question_{}")

    # Iterate over the dataset
    for question_index, question in tqdm(enumerate(question_list), disable=True):
        
        # Break if we have reached the max number of questions to ask
        if(max_questions == question_index and max_questions != 0):
            break

        #Retrive the dataset answer
        answer = answer_list[question_index]
        
        # Prepare prompt based on whether LLM has chat template or not
        if chat_template:
            formatted_prompt = tokenizer_chat_template(llm.tokenizer, enable_thinking, system_message, question)
        else:
            formatted_prompt = system_message +  question

        # Store the prompt and answers in the results csv
        result_row = {
            "question": formatted_prompt,
            "correct_answer": answer
        }

        # Generate a response using the sampler
        if(sampling_techniques[2]): # Power Sampling
            #Add the question numbe to the kwargs log_file_path
            kwargs["log_file_path"] = log_file_path_template.format(question_index)

            #Time how long it takes to get a response
            start_time = time.time()

            # Send the prompt to the sliding window power sampling function
            output = llm.token_sample(formatted_prompt, **kwargs)

            # Find the end time of the power sampling
            end_time = time.time()

            # Parse the answer
            power_sampling_answer = parse_answer(llm.tokenizer.decode(output.tokens, skip_special_tokens=False))

            # Save the results
            result_row["mcmc_completion"] = llm.tokenizer.decode(output.tokens, skip_special_tokens=False)
            result_row["mcmc_output_token_count"] = len(output.tokens)
            result_row["mcmc_time_to_solution"] = end_time - start_time
            result_row["mcmc_answer"] = power_sampling_answer

        # Generate a response with just low temperature sampling
        if(sampling_techniques[1]): # Low Temperature Sampling
            #Time how long it takes to get a response
            start_time = time.time()

            # Prompt the LLM and get the output/answer
            output = llm.sample(formatted_prompt)
            
            # Find the end time of the low temperature sampling
            end_time = time.time()

            # Parse the answer
            low_temp_sampling_answer = parse_answer(llm.tokenizer.decode(output.tokens, skip_special_tokens=False))

            # Save the results
            result_row["naive_completion"] = llm.tokenizer.decode(output.tokens, skip_special_tokens=False)
            result_row["naive_sampling_output_token_count"] = len(output.tokens)
            result_row["naive_sampling_time_to_solution"] = end_time - start_time
            result_row["naive_answer"] = low_temp_sampling_answer
        
        if(sampling_techniques[0]): # Naive Sampling
            # Save and change the temperature to 1.0 for naive sampling
            saved_temperature = llm.sampling_params.temperature
            llm.sampling_params.temperature = 1.0
            
            #Time how long it takes to get a response
            start_time = time.time()

            # Prompt the LLM and get the output/answer
            output = llm.sample(formatted_prompt)
            
            # Find the end time of the naive sampling
            end_time = time.time()

            # Parse the answer
            std_sampling_answer = parse_answer(llm.tokenizer.decode(output.tokens, skip_special_tokens=False))
            # Save the results
            result_row["std_completion"] = llm.tokenizer.decode(output.tokens, skip_special_tokens=False)
            result_row["std_sampling_output_token_count"] = len(output.tokens)
            result_row["std_sampling_time_to_solution"] = end_time - start_time
            result_row["std_answer"] = std_sampling_answer
            
            # Set the temperature back to original
            llm.sampling_params.temperature = saved_temperature

        # Write the question and final answer to the output file
        results.append(result_row)
        # Write to CSV after each iteration, only write header for first row
        df = pd.DataFrame([result_row])
        df.to_csv(output_file, index=False, header=(question_index==0))
        output_file.flush()
        os.fsync(output_file.fileno())

