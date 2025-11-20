#Tokenizers
from transformers import AutoTokenizer

#Custom Classes and Constructors
import src.inference.vllm_backend as vllm_backend

# Autoregressive Sampler Class
# Stores parameters concerning the LLM, autoregressive sampling, and power sampling
# Includes Functions:
# sample() - Samples from the LLM given a context and max new tokens programmatical LLM
class AutoregressiveSampler:
    def __init__(
        self, 
        model, # LLM Model name
        llm, # LLM object to use for sampling
        tokenizer, # Tokenizer to use for encoding/decoding (HuggingFace AutoTokenizer)
        sample_fn, # Function to use for sampling from the autoregressive model
        sampling_params, # Parameters to use for standard sampling
        power_sampling_params = None, # Parameters to use for power sampling
        smc_sampling_params = None, # Parameters to use for SMC sampling
        best_of_sampling_params = None # Parameters to use for best-of sampling
    ):
        self.model = model
        self.llm = llm
        self.tokenizer = tokenizer
        self.sample_fn = sample_fn
        self.sampling_params = sampling_params
        self.power_sampling_params = power_sampling_params
        self.smc_sampling_params = smc_sampling_params
        self.best_of_sampling_params = best_of_sampling_params
    def sample(self, context, max_new_tokens):
        return self.sample_fn(self, context, max_new_tokens)

#Common Sampling Parameters
class Sampling_Params:
    def __init__(
        self,
        engine_params = None, # Engine specific parameter Class (vLLM: SamplingParams, Add more as needed)
        enable_thinking = False,
        max_tokens = 16, # Max Number of tokens to generate per sequence
        temperature = 1.0, # Controls randomness of sampling. Lower is more deterministic, higher is more random
        top_p = 1.0, # Controls tokens to consider based on cumulative probability. Must be in (0, 1]
        top_k = 0, # Controls number of top tokens to consider. 0 or -1 is considers all tokens
        logprobs = None, # Number of logits/logprobs to return per output token. logprobs+1 token returned (includes chosen token). -1 returns all vocab_size log probabilities
        presence_penalty = 0.0, # Penalizes new tokens based on appearance in generated text so far. > 0 encourages new tokens, < 0 encourages repeats
        frequency_penalty = 0.0, # Penalizes new tokens based on frequency in generated text so far. > 0 encourages new tokens, < 0 encourages repeats
        repetition_penalty = 1.0, # Penalizes new tokens based on appearance in prompt AND generated text so far. > 1 encourages new tokens, < 1 encourages repeats
        min_p = 0.0, # Represents the minimum probability for a token to be considered. 0 disables
        seed = None, # Random seed
        stop = None, # Strings that stop token generation. Returned output excludes stop strings
        stop_token_ids = None, # Token IDs that stop token generation. Returned output excludes stop tokens
        ignore_eos = False, # Continues generating tokens after EOS token is generated.
        min_tokens = 0, # Minimum Number of tokens to generate per sequence before EOS or stop is considered
    ):
        self.engine_params = engine_params
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p= top_p
        self.top_k = top_k
        self.logprobs = logprobs
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p
        self.seed = seed
        self.stop = stop
        self.stop_token_ids = stop_token_ids
        self.ignore_eos = ignore_eos
        self.min_tokens = min_tokens


class Power_Sampling_Params:
    def __init__(
        self, 
        total_output_tokens=1000, # Max sequence lenght in tokens to generate when power sampling
        block_size=50, # How many blocks to divide the total output tokens into for power sampling. Smaller block sizes = better quality but slower
        MCMC_steps=5 # Number of MCMC steps to perform per block. More steps = better quality but slower
    ):
        self.total_output_tokens = total_output_tokens 
        self.block_size = block_size
        self.MCMC_steps = MCMC_steps

# TO DO once SMC is implemented
class SMC_Sampling_Params:
    def __init__(
        self, 
        particles=4, 
        particle_length=50, 
        resample_interval=50
    ):
        self.particles = particles
        self.particle_length = particle_length
        self.resample_interval = resample_interval

class Best_of_Sampling_Params:
    def __init__(
        self,
        n = 1, # Number of outputs to return for the given prompt request
        best_of = 5, # The top `best_of` sequences generated. best_of must be greater than or equal to n
    ):
        self.n = n
        self.best_of = best_of

# Create an AutogressiveSampler object given the engine, engine parameters, and model name
def create_autoregressive_sampler(
    engine, # Engine to use for autoregressive sampling. Currently only "vllm" is supported
    model, # Model to load 
    dtype="auto", # Data type to use when loading the model. "auto" lets the engine decide
    gpu_memory_utilization=0.85, # GPU memory utilization to use 
    max_model_len=2048, # Max model context length
    max_logprobs = 100, # Number of logits/logprobs to store per output token
    sampling_params=None # General sampling parameters to use (Sampling_Params Class)
):
                                
    print(f"Loading model {model} with {engine}...")
    if(engine == "vllm"):
        # Create the LLM object
        llm = vllm_backend.create_LLM_object(
            model_name=model, 
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_logprobs=max_logprobs
        )
        # Set the autoregressive sampler function
        autoregressive_sampler = vllm_backend.sample
    else:
        raise ValueError(f"Engine {engine} not supported for Autoregressive Sampler. Try 'vllm'.")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # Create the Autoregressive Sampler object
    sampler = AutoregressiveSampler(
        model=model,
        llm=llm,
        tokenizer=tokenizer,
        sample_fn=autoregressive_sampler,
        sampling_params= Sampling_Params(logprobs = max_logprobs) if sampling_params is None else sampling_params
    )

    print("Model loaded successfully. Sampling parameters set to default values.")

    return sampler

def enable_power_sampling(sampler, total_output_tokens, block_size, MCMC_steps):
    # Check if the sampler is initialized
    if(sampler is None):
        raise ValueError("Sampler must be initialized before enabling power sampling.")
    
    # Check to make sure the vLLM engine is outputing logits/logprobs
    if(sampler.sampling_params.top_k <= 0):
        raise ValueError("LLM engine top_k must be set to a positive integer to enable power sampling.")
    
    # Set the power sampling parameters
    sampler.power_sampling_params = Power_Sampling_Params(
        total_output_tokens=total_output_tokens,
        block_size=block_size,
        MCMC_steps=MCMC_steps
    )

    print(f"Power Sampling Enabled: Logits Consider = {sampler.sampling_params.top_k}, Total Output Tokens = {total_output_tokens}, Block Size = {block_size}, MCMC Steps = {MCMC_steps}, Temperature (1/alpha) = {sampler.sampling_params.temperature}")

def enable_SMC_sampling(sampler, particles, particle_length, resample_interval):
    # Check if the sampler is initialized
    if(sampler is None):
        raise ValueError("Sampler must be initialized before enabling SMC sampling.")
    
    # Check to make sure the LLM engine is outputing logits/logprobs
    if(sampler.sampling_params.top_k <= 0):
        raise ValueError("LLM engine top_k must be set to a positive integer to enable SMC sampling.")
    
    # Set the SMC sampling parameters
    sampler.smc_sampling_params = SMC_Sampling_Params(
        particles=particles,
        particle_length=particle_length,
        resample_interval=resample_interval
    )

    print(f"SMC Sampling Enabled: Logits Consider = {sampler.sampling_params.top_k}, Particles = {particles}, Particle Length = {particle_length}, Resample Interval = {resample_interval}, Temperature = {sampler.sampling_params.temperature}")