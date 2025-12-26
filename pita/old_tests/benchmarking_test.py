# Training Free Reasoning Libraries
from pita.utils.benchmarking_utils import benchmark_sampling, load_benchmark
from pita.inference.LLM_backend import AutoregressiveSampler

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

    engine_name = "vllm"

    # LLM parameters
    if(engine_name == "vllm"):
        _model_name = "Qwen/Qwen2.5-7B"
        _dtype = "auto"
        _tokenizer_path = None
        _gpu_memory_utilization = 0.85
        _max_model_len = 3072
        _max_probs = 1

    elif(engine_name == "llama_cpp"):
        _model_name = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
        _dtype = "Q5_K_M"
        _tokenizer_path="Qwen/Qwen3-4B-Instruct-2507"
        _gpu_memory_utilization = 0.85
        _max_model_len = 3072
        _max_probs = None
        
        # Tell Pytorch to use the GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    #Initialize LLM 
    llm = AutoregressiveSampler(
        engine="vllm",
        model=_model_name,
        dtype=_dtype,
        tokenizer_path=_tokenizer_path,
        gpu_memory_utilization=_gpu_memory_utilization,
        max_model_len=_max_model_len,
        max_probs=_max_probs,
        logits_processor=True,
        trust_remote_code=True,
        sampling_params=None
    )

    # Set sampling parameters
    llm.sampling_params.temperature = 0.25
    llm.sampling_params.max_tokens = 3072

    # Enable Power Sampling
    # Power Sampling Hyperparameters
    block_size = 192 # tokens per block. Number of blocks = token_count / block_size
    MCMC_steps = 10
    token_metric = "power_distribution"

    # Enable Power Sampling
    llm.enable_power_sampling(
        block_size, # tokens per block. Number of blocks = token_count / block_size
        MCMC_steps, # MCMC steps per block
        token_metric
    )

    # Load dataset to test
    system_message, question_list, answer_list = load_benchmark("MATH500")
    # Define sampling techniques to benchmark
    sampling_techniques = [True, True, False, False, False] # temp=1 sampling, low temp sampling, power sampling, smc, best of n
    # Define the **kwargs for the benchmark
    kwargs = {
    "power_sampling_logging": True,
    "power_sampling_logging_path": "results/power_sampling_logs"
    }
    # Run the benchmark
    benchmark_sampling(
        llm=llm,
        system_message=system_message,
        question_list=question_list,
        answer_list=answer_list,
        enable_thinking=False, 
        chat_template=False,
        sampling_techniques=sampling_techniques, 
        max_questions=500, 
        output_file_name="results/math500_power_sampling_results.csv",
        **kwargs
    )