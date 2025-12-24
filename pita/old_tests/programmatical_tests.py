#PITA Libraries
from pita.inference.LLM_backend import create_autoregressive_sampler
from pita.sampling.power_sample import enable_power_sampling, power_sampling
from pita.sampling.smc import enable_smc_sampling, sequential_monte_carlo
from pita.sampling.best_of import enable_best_of_sampling, best_of_n_logprob

#Other Libraries
import time

def test_pita_lib(
    engine_name,
    model_name: str = None,
    dtype: str = None,
    tokenizer_path: str = None,
    gpu_memory_utilization: float = None,
    max_model_len: int = None,
    max_logprobs: int = None,
    logits_per_token: int = None,
    en_base_test : bool = False, 
    en_power_sampling_test: bool = False,
    en_smc_sampling_test: bool = False,
    en_best_of_sampling_test: bool = False
) -> None:
    
    # Create the LLM engine and sampler
    # LLM parameters
    if(engine_name == "vllm"):
        _engine_name = "vllm"
        if(model_name is None ):
            print("Defaulting to Qwen3-4B-AWQ for vLLM")
            _model_name = "Qwen/Qwen3-4B-AWQ"
            _dtype = "auto"
            _tokenizer_path = None
            _gpu_memory_utilization = 0.85
            _max_model_len = 2048
            _max_logprobs = 2
            _logits_per_token = 2
            _logits_processor = True
        else:
            print(f"Using user provided model {model_name} for vLLM. Make sure all engine parameters are set correctly.")
            _model_name = model_name
            _dtype = dtype
            _tokenizer_path = tokenizer_path
            _gpu_memory_utilization = gpu_memory_utilization
            _max_model_len = max_model_len
            _max_logprobs = max_logprobs
            _logits_per_token = logits_per_token
            _logits_processor = True

    elif(engine_name == "llama_cpp"):
        _engine_name = "llama_cpp"
        if(model_name is None):
            print("Defaulting to Unsloth Qwen3-4B-Instruct-2507-GGUF_Q5_K_M for llama_cpp")
            _model_name = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
            _dtype = "Q5_K_M"
            _tokenizer_path="Qwen/Qwen3-4B-Instruct-2507"
            _gpu_memory_utilization = 0.85
            _max_model_len = 2048
            _max_logprobs = None
            _logits_per_token = 100
            _logits_processor = True

        else:
            print(f"Using user provided model {model_name} for llama_cpp. Make sure all engine parameters are set correctly.")
            _model_name = model_name
            _dtype = dtype
            _tokenizer_path = tokenizer_path
            _gpu_memory_utilization = gpu_memory_utilization
            _max_model_len = max_model_len
            _max_logprobs = max_logprobs
            _logits_per_token = logits_per_token
            _logits_processor = True

    else:
        raise ValueError(f"Engine {engine_name} not supported for testing.")
    #Initialize Autoregressive Sampler
    sampler = create_autoregressive_sampler(
        engine=_engine_name, 
        model=_model_name, 
        dtype=_dtype,
        tokenizer_path=_tokenizer_path, 
        gpu_memory_utilization=_gpu_memory_utilization, 
        max_model_len=_max_model_len, 
        max_logprobs = _max_logprobs,
        logits_per_token = _logits_per_token,
        logits_processor = _logits_processor
    )

    # Message to test model and tokenizer with
    messages = [
        {"role": "system", "content": "You are a personal assistant."},
        {"role": "user", "content": "Hello! What is 1+1+1+1+2? Write a story about it."},
    ]

    # Test the Tokenizer and Chat Template is functioning correctly
    prompt = sampler.tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )

    # Log the results of each test in a file
    with open(f"test_results_{_engine_name}.log", "w") as log_file:
        log_file.write(f"--- Testing PITA Library with engine: {_engine_name} ---\n")
        log_file.write(f"Model Name: {_model_name}\n")
        log_file.write(f"Dtype: {_dtype}\n")
        log_file.write(f"Tokenizer Path: {_tokenizer_path}\n")
        log_file.write(f"GPU Memory Utilization: {_gpu_memory_utilization}\n")
        log_file.write(f"Max Model Length: {_max_model_len}\n")
        log_file.write(f"Max Logprobs: {_max_logprobs}\n")
        log_file.write(f"Logits Per Token: {_logits_per_token}\n")
        log_file.write(f"Enable Base Test: {en_base_test}\n")
        log_file.write(f"Enable Power Sampling Test: {en_power_sampling_test}\n")
        log_file.write(f"Enable Best Of Sampling Test: {en_best_of_sampling_test}\n")
        log_file.write("\n")
        log_file.write(f"Generated Prompt: \n{prompt}\n")
    


    # Test the AutoregressiveSampler.sample() function
    if(en_base_test):
        # Set max tokens for sampling
        sampler.sampling_params.max_tokens = 1000
        sampler.sampling_params.logprobs = 0
        # Test basic sampling
        output = sampler.sample(prompt, sampler.sampling_params.max_tokens)
        output = sampler.tokenizer.decode(output[0], skip_special_tokens=True)
        # Log Results
        with open(f"test_results_{_engine_name}.log", "a") as log_file:
            log_file.write(f"Base Sampling Test Output: \n{output}\n")
            log_file.write("\n")

    # Test Power Sampling
    if(en_power_sampling_test):
        # Set sampling parameters
        sampler.sampling_params.max_tokens = 1000
        sampler.sampling_params.temperature = 0.25

        # Power Sampling Hyperparameters
        block_size = 250 # tokens per block. Number of blocks = token_count / block_size
        MCMC_steps = 5 

        # Enable Power Sampling
        enable_power_sampling(
            sampler,
            block_size, # tokens per block. Number of blocks = token_count / block_size
            MCMC_steps, # MCMC steps per block
        )

        # Log Power Sampling parameters
        with open(f"test_results_{_engine_name}.log", "a") as log_file:
            log_file.write(f"Power Sampling Parameters:\n")
            log_file.write(f"Max Tokens: {sampler.sampling_params.max_tokens}\n")
            log_file.write(f"Temperature: {sampler.sampling_params.temperature}\n")
            log_file.write(f"Block Size: {block_size}\n")
            log_file.write(f"MCMC Steps: {MCMC_steps}\n")

        # Measure Time take for power sampling with logging
        start_time = time.time()

        # Test power sampling
        output = power_sampling(
            sampler, # Autoregressive Sampler Object
            prompt, # Template prompt
            logging = True # Enable logging to CSV file
        )

        # Calculate time taken and user tokens per second
        end_time = time.time()
        time_taken = end_time - start_time
        tokens_generated = len(sampler.tokenizer.tokenize(output))
        user_tokens_per_second = tokens_generated / time_taken

        with open(f"test_results_{_engine_name}.log", "a") as log_file:
            log_file.write(f"Power Sampling Test Output: \n{output}\n")
            log_file.write(f"Power Sampling Log File: power_sampling_log.csv\n")
            log_file.write(f"User Token Per Second Estimate (Includes overhead from logging): {user_tokens_per_second}\n")
            log_file.write("\n")

    # Test Sequential Monte Carlo Sampling
    if(en_smc_sampling_test):
        # Set max tokens for sampling
        sampler.sampling_params.max_tokens = 500

        # Sequential Monte Carlo Sampling Hyperparameters
        num_particles = 5
        tokens_per_step = 32
        stop_on_eos = True

        # Enable Sequential Monte Carlo Sampling
        enable_smc_sampling(
            sampler,
            num_particles=num_particles,
            tokens_per_step=tokens_per_step,
            stop_on_eos=stop_on_eos
        )

        # Test Sequential Monte Carlo Sampling
        output = SequentialMonteCarlo(
            sampler,
            prompt
        )

        with open(f"test_results_{_engine_name}.log", "a") as log_file:
            log_file.write(f"Sequential Monte Carlo Sampling Test Output: \n{output}\n")
            log_file.write(f"Number of Particles: {num_particles}\n")
            log_file.write(f"Tokens Per Step: {tokens_per_step}\n")
            log_file.write(f"Stop on EOS: {stop_on_eos}\n")
            log_file.write("\n")

    # Test Best Of Sampling
    if(en_best_of_sampling_test):
        # Set Best Of Sampling Parameters
        best_of = 5
        seq_top_k = 3

        # Enable Best Of Sampling
        enable_best_of_sampling(
            sampler,
            sequence_n = best_of, # Number of sequences to sample and choose the best from
            sequence_top_k = seq_top_k, # Number of top_k sequences to choose from (top_k <= sequences)
        )

        # Test Best Of Sampling
        output = best_of_n_logprob(
            prompt,
            sampler
        )

        # Log Results
        with open(f"test_results_{_engine_name}.log", "a") as log_file:
            log_file.write(f"Best Of Sampling Test Output: \n{output}\n")
            log_file.write(f"Number of Sequences = {sampler.best_of_sampling_params.sequence_n}\n")
            log_file.write(f"Top K = {sampler.best_of_sampling_params.sequence_top_k}\n")
            log_file.write("\n")

    # Shutdown the sampler to free resources
    if(engine_name == "llama_cpp"):
        sampler.llm.close()
    
if __name__ == "__main__":
    # Test PITA Library with vLLM
    test_pita_lib(
        engine_name = "vllm",
        en_base_test = False,
        en_power_sampling_test = True,
        en_smc_sampling_test = False,
        en_best_of_sampling_test = False
    )

    # Test PITA Library with llama.cpp
    # test_pita_lib(
    #     engine_name = "llama_cpp",
    #     en_base_test = True,
    #     en_power_sampling_test = True,
    #     en_smc_sampling_test = True,
    #     en_best_of_sampling_test = True
    # )