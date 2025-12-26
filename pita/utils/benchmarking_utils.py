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
MATH_SYSTEM_MESSAGE = "You are a very knowledgeable math expert. Think and respond with confidence."
MATH_PRE_QUESTION = "Solve the following math problem. "
AIME_PRE_QUESTION = "The solution to the math problem is an integer between 0 and 999. "
MATH_ANSWER_FORMAT = "Put your final answer within \\boxed{{}}. "
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
            pre_question = MATH_PRE_QUESTION + AIME_PRE_QUESTION + MATH_ANSWER_FORMAT
            post_question = ""

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
            pre_question = MATH_PRE_QUESTION + MATH_ANSWER_FORMAT
            post_question = ""

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
    sampling_techniques: list[bool], # greedy sampling, low temp sampling, power sampling, smc, best of n, power sampling and smc
    max_questions: int = 0, 
    output_file_name: str = "math500_power_sampling_results.csv",
    **kwargs
) -> None:
    # Extract kwargs
    if("power_sampling_logging" in kwargs):
        power_sampling_logging = kwargs["power_sampling_logging"]
    if("power_sampling_logging_path" in kwargs):
        power_sampling_logging_path = kwargs["power_sampling_logging_path"]
    # Store results
    results = []    
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    output_file = open(output_file_name, "w")

    # Iterate over the dataset
    for question_index, question in tqdm(enumerate(question_list), disable=True):
        
        # Break if we have reached the max number of questions to ask
        if(max_questions == question_index and max_questions != 0):
            break

        #Retrive the dataset answer
        answer = answer_list[question_index]
        
        # Prepare prompt based on whether LLM has chat template or not
        if hasattr(llm.tokenizer, "apply_chat_template"):
            formatted_prompt = tokenizer_chat_template(llm.tokenizer, enable_thinking, system_message, question)
        else:
            formatted_prompt = system_message + "\n\n" + question

        # Store the prompt and answers in the results csv
        result_row = {
            "question": formatted_prompt,
            "correct_answer": answer
        }

        # Generate a response using the sampler
        if(sampling_techniques[2]): # Power Sampling
            #Time how long it takes to get a response
            start_time = time.time()

            # Setup Logging file paths
            if(power_sampling_logging != False):
                os.makedirs(power_sampling_logging_path, exist_ok=True)

            # Send the prompt to the sliding window power sampling function
            power_sampling_output = power_sampling(
                llm, # Autoregressive Sampler Object
                formatted_prompt, # Template prompt
                logging = power_sampling_logging, # Enable logging to CSV file
                log_file_path = f"{power_sampling_logging_path}/power_sampling_log_question_{question_index}.csv"
            )

            # Find the end time of the power sampling
            end_time = time.time()

            # Parse the answer
            power_sampling_answer = parse_answer(power_sampling_output, answer)

            # Save the results
            result_row["power_sampling_output"] = power_sampling_output
            result_row["power_sampling_answer"] = power_sampling_answer
            result_row["power_sampling_output_token_count"] = len(llm.tokenizer.encode(power_sampling_output))
            result_row["power_sampling_time_to_solution"] = end_time - start_time

        # Generate a response with just low temperature sampling
        if(sampling_techniques[1]): # Low Temperature Sampling
            #Time how long it takes to get a response
            start_time = time.time()

            # Prompt the LLM and get the output/answer
            low_temp_tokens_list, _, _, _, _ = llm.sample(formatted_prompt, llm.sampling_params.max_tokens)
            
            # Find the end time of the low temperature sampling
            end_time = time.time()

            # Parse the answer
            low_temp_sampling_answer = parse_answer(llm.tokenizer.decode(low_temp_tokens_list, skip_special_tokens=False), answer)

            # Save the results
            result_row["low_temp_sampling_output"] = llm.tokenizer.decode(low_temp_tokens_list, skip_special_tokens=False)
            result_row["low_temp_sampling_output_token_count"] = len(low_temp_tokens_list)
            result_row["low_temp_sampling_time_to_solution"] = end_time - start_time
            result_row["low_temp_sampling_answer"] = low_temp_sampling_answer

        if(sampling_techniques[0]): # Naive Sampling
            # Save and change the temperature to 1.0 for naive sampling
            saved_temperature = llm.sampling_params.temperature
            llm.sampling_params.temperature = 1.0
            
            #Time how long it takes to get a response
            start_time = time.time()

            # Prompt the LLM and get the output/answer
            naive_tokens_list, _, _, _, _ = llm.sample(formatted_prompt, llm.sampling_params.max_tokens)
            
            # Find the end time of the naive sampling
            end_time = time.time()

            # Parse the answer
            naive_sampling_answer = parse_answer(llm.tokenizer.decode(naive_tokens_list, skip_special_tokens=False), answer)
            # Save the results
            result_row["naive_sampling_output"] = llm.tokenizer.decode(naive_tokens_list, skip_special_tokens=False)
            result_row["naive_sampling_output_token_count"] = len(naive_tokens_list)
            result_row["naive_sampling_time_to_solution"] = end_time - start_time
            result_row["naive_sampling_answer"] = naive_sampling_answer
            
            # Set the temperature back to original
            llm.sampling_params.temperature = saved_temperature

        # Write the question and final answer to the output file
        results.append(result_row)
        # Write to CSV after each iteration, only write header for first row
        df = pd.DataFrame([result_row])
        df.to_csv(output_file, index=False, header=(question_index==0))
        output_file.flush()
        os.fsync(output_file.fileno())

