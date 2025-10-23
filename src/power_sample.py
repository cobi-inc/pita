import random
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

class AutoregressiveSampler:
    def __init__(self, llm, tokenizer, power_sampling_temperature=1.0, logprobs=100, detokenize=False, token_count=1000, block_size=50, MCMC_steps=5):
        self.llm = llm
        self.tokenizer = tokenizer
        self.power_sampling_temperature = power_sampling_temperature
        self.logprobs = logprobs
        self.detokenize = detokenize
        self.token_count = token_count
        self.block_size = block_size
        self.MCMC_steps = MCMC_steps
        self.block_num = token_count // block_size

    def sample(self, context, max_new_tokens):
        # Set the sampling parameters of the LLM
        sample_params = SamplingParams(temperature=self.power_sampling_temperature, max_tokens=max_new_tokens, logprobs=self.logprobs, detokenize=self.detokenize)
        # Generate a new response from the LLM
        llm_output = self.llm.generate(context, sampling_params=sample_params)
        return llm_output

        

def power_sample(sampler: AutoregressiveSampler, prompt, temperature, power, token_count, seed):
    # Set random seed
    random.seed(seed)

    # Log Probabilites
    logprob = [] # Current list of unscaled log probabilites of the new sample. Length of block_size
    logprob_temp_scaled = [] # Current list of tokens probabilites individually scaled by temperature. Length of block_size
    proposed_logprob = [] # Proposed list of unscaled log probabilites of the new sample. Length of max_new_tokens
    proposed_logprob_temp_scaled = [] # Proposed list of tokens probabilites individually scaled by temperature. Length of max_new_tokens

    # Tokenized Prompt
    context = sampler.tokenizer.encode(prompt)

    # Iterate over the number of blocks to be generated
    for block_idx in tqdm(range(sampler.block_num)):
        # Generate next block of tokens as baseline
        output = sampler.sample(context, sampler.block_size)
        

        #Iterate over the number of MCMC steps
        for _ in tqdm(range(sampler.MCMC_steps)):
            #Find a new point to start a proposal from
            idx = random.randint(0,sampler.block_size -1)

            #Set the new context for the proposed block
            context_proposed = context[:-(sampler.block_size - idx)]
            #Generate proposed block of tokens
            proposed_output = sampler.sample(context_proposed)

            # Find the log probabilities of the generated tokens

            # Calculate the Metro-Hastings acceptance ratio
            log_acceptance_ratio = sum()
            # Accept or reject the proposed block based on the acceptance ratio

        # Check if an EOS token has been generated and end the process if so




    return


def vllm_test(sampler, prompt = "Once upon a time"):
    # User Initial Prompt
    prompt_obj = TokensPrompt(prompt_token_ids=sampler.tokenizer.encode(prompt))
    output = sampler.sample(prompt_obj, max_new_tokens=50)
    print(sampler.tokenizer.decode(output[0].outputs[0].token_ids, skip_special_tokens=True))
    #print(output[0].outputs[0].logprobs)


if __name__ == "__main__":
    # Initialize the random number generator
    seed = 42
    random.seed(seed)

    # Power Sampling Hyperparameters
    token_count = 1000
    block_size = 50
    MCMC_steps = 5

    # LLM parameters
    model = "Qwen/Qwen3-4B-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code = True)
    skip_tokenizer_init = True
    dtype = "auto"
    quantization = None
    gpu_memory_utilization = 0.9

    # Initalize model
    llm = LLM(model=model, skip_tokenizer_init=skip_tokenizer_init, dtype=dtype, quantization=quantization, gpu_memory_utilization=gpu_memory_utilization, max_logprobs= tokenizer.vocab_size + token_count)

    #Sampling parameters for vLLM
    temperature = 0.25
    logprobs = -1
    detokenize = False

    #Initalize Autogressive Sampler
    sampler = AutoregressiveSampler(llm, 
                                    tokenizer,
                                    power_sampling_temperature=temperature,
                                    logprobs=logprobs,
                                    detokenize=detokenize,
                                    token_count=token_count,
                                    block_size=block_size,
                                    MCMC_steps=MCMC_steps
                                    )


    # Test vLLM sampling
    vllm_test(sampler)

    # Call the power_sample function
    #output = power_sample(model, tokenizer, prompt, temperature, power, token_count, block_size, MCMC_steps, seed)
    # Print the output
    #print("Generated Text:", output)