# Training Free Reasoning Libraries
from pita.utils.benchmarking_utils import benchmark_sampling
from pita.inference.autoregressive_sampler_backend import create_autoregressive_sampler
from pita.sampling.power_sample import enable_power_sampling

# Pytorch Library
import torch

#Standard Libraries
import random
import time

# Main function to test power sampling
if __name__ == "__main__":


    # Initialize the random number generator
    #seed = 42
    seed = time.time_ns() % (2**32 - 1)
    torch.manual_seed(seed)
    random.seed(seed)

    # Power Sampling Hyperparameters
    total_output_tokens = 1000 #total tokens for response
    block_size = 250 # tokens per block. Number of blocks = token_count / block_size
    MCMC_steps = 5 

    engine_name = "vllm"

    # LLM parameters
    if(engine_name == "vllm"):
        model_name = "Qwen/Qwen3-4B-AWQ"
        dtype = "auto"
        tokenizer_path = None
        max_logprobs = 100
        logits_per_token = 100

    elif(engine_name == "llama_cpp"):
        model_name = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
        dtype = "Q5_K_M"
        tokenizer_path="Qwen/Qwen3-4B-AWQ"
        max_logprobs = None
        logits_per_token = 1000
        
        # Tell Pytorch to use the GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    gpu_memory_utilization = 0.8
    max_model_len = 8192

    #Initialize Autoregressive Sampler
    sampler = create_autoregressive_sampler(
        engine=engine_name, 
        model=model_name, 
        dtype=dtype,
        tokenizer_path="Qwen/Qwen3-4B-AWQ", 
        gpu_memory_utilization=gpu_memory_utilization, 
        max_model_len=max_model_len, 
        max_logprobs = max_logprobs,
        logits_per_token = logits_per_token
    )

    # Create the power sampling parameters to use
    enable_power_sampling(sampler, total_output_tokens, block_size, MCMC_steps)

    # Test MATH500 Benchmark
    dataset_name = "AIME"
    power_sampling_on = True
    power_sampling_windowed_on = False
    low_temp_sampling_on = False
    naive_sampling_on = False
    chain_of_thought = False
    #for temp in [0.25, 0.5, 0.75]:
    for temp in [0.5]:
        sampler.sampling_params.temperature = temp
        benchmark_sampling(
            dataset_name, 
            sampler, 
            chain_of_thought, 
            power_sampling_on, 
            power_sampling_windowed_on, 
            low_temp_sampling_on, 
            naive_sampling_on, 
            question_max = 1, 
            output_file_name = f"results/{dataset_name}_power_sampling_results_temp_{temp}.csv", 
            seed=seed
        )

    del sampler