# Training Free Reasoning Libraries
from src.power_sampling.power_sample import AutoregressiveSampler
from src.utils.benchmarking_utils import benchmark_sampling

# Pytorch Library
import torch

# Inference Library
from vllm import LLM
from transformers import AutoTokenizer

#Standard Libraries
import random

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
    token_count = 8192 #total tokens for response
    block_size = 400 # tokens per block. Number of blocks = token_count / block_size
    MCMC_steps = 10 

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
    quantization = None
    gpu_memory_utilization = 0.8
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
                                    enable_thinking=False,
                                    power_sampling_temperature=temperature,
                                    top_k=top_k,
                                    token_count=token_count,
                                    block_size=block_size,
                                    MCMC_steps=MCMC_steps
                                    )
    
    # Test MATH500 Benchmark
    dataset_name = "MATH500"
    power_sampling_on = False
    power_sampling_windowed_on = False
    low_temp_sampling_on = False
    naive_sampling_on = True
    chain_of_thought = False
    #for temp in [0.25, 0.5, 0.75]:
    for temp in [1]:
        sampler.power_sampling_temperature = temp
        benchmark_sampling(dataset_name, sampler, chain_of_thought, power_sampling_on, power_sampling_windowed_on, low_temp_sampling_on, naive_sampling_on, question_max = 10, output_file_name = f"results/{dataset_name}_power_sampling_results_temp_{temp}.csv", seed=seed)
