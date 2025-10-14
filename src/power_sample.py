import random
from vllm import LLM, SamplingParams

def power_sample(model, prompt, temperature, power, token_count, block_size, MCMC_steps, seed):
    # Set random seed
    random.seed(seed)

    # Initialize the model

    return


def vllm_test(model, sampling_params, prompt):
    llm = LLM(model=model)
    response = llm.generate(prompt, sampling_params)
    return response

if __name__ == "__main__":
    # Initialize the random number generator
    seed = 42
    random.seed(seed)


    # Hyperparameters
    temperature = 1.0
    power = 4.0
    token_count = 1000
    block_size = 50
    MCMC_steps = 5

    # Model parameters
    model = "Qwen/Qwen2.5-7B"
    sampling_params = SamplingParams(temperature=temperature, max_tokens=token_count)

    # User Initial Prompt
    prompt = "Once upon a time"

    print(vllm_test(model, sampling_params, prompt))


    # Call the power_sample function
    #output = power_sample(model, prompt, temperature, power, token_count, block_size, MCMC_steps, seed)
    # Print the output
    #print("Generated Text:", output)