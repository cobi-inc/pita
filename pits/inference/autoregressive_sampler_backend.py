#Tokenizers
from transformers import AutoTokenizer

#Custom Classes and Constructors
import pits.inference.vllm_backend as vllm_backend

# Engine-specific parameter mappings
ENGINE_PARAM_MAPS = {
    'vllm': {
        'max_tokens': 'max_tokens',
        'temperature': 'temperature',
        'top_p': 'top_p',
        'top_k': 'top_k',
        'logprobs': 'logprobs',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'repetition_penalty': 'repetition_penalty',
        'min_p': 'min_p',
        'seed': 'seed',
        'stop': 'stop',
        'stop_token_ids': 'stop_token_ids',
        'ignore_eos': 'ignore_eos',
        'min_tokens': 'min_tokens',
    },
    'transformers': {
        'max_tokens': 'max_new_tokens',
        'temperature': 'temperature',
        'top_p': 'top_p',
        'top_k': 'top_k',
        'repetition_penalty': 'repetition_penalty',
        'stop_token_ids': 'eos_token_id',
        # transformers uses different names/doesn't support all params
    },
    'llama_cpp': {
        'max_tokens': 'max_tokens',
        'temperature': 'temp',
        'top_p': 'top_p',
        'top_k': 'top_k',
        'repetition_penalty': 'repeat_penalty',
        'seed': 'seed',
        'stop': 'stop',
    },
    # Add more engines as needed
}

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
        self.engine = None
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
        engine = None, # Engine name (e.g., "vllm", "transformers", etc.)
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
        object.__setattr__(self, 'engine', engine)
        object.__setattr__(self, 'engine_params', engine_params)
        object.__setattr__(self, 'enable_thinking', enable_thinking)
        object.__setattr__(self, 'max_tokens', max_tokens)
        object.__setattr__(self, 'temperature', temperature)
        object.__setattr__(self, 'top_p', top_p)
        object.__setattr__(self, 'top_k', top_k)
        object.__setattr__(self, 'logprobs', logprobs)
        object.__setattr__(self, 'presence_penalty', presence_penalty)
        object.__setattr__(self, 'frequency_penalty', frequency_penalty)
        object.__setattr__(self, 'repetition_penalty', repetition_penalty)
        object.__setattr__(self, 'min_p', min_p)
        object.__setattr__(self, 'seed', seed)
        object.__setattr__(self, 'stop', stop)
        object.__setattr__(self, 'stop_token_ids', stop_token_ids)
        object.__setattr__(self, 'ignore_eos', ignore_eos)
        object.__setattr__(self, 'min_tokens', min_tokens)

        # Sync all parameters to engine_params after initialization
        if engine is not None and engine_params is not None:
            for param_name in ['max_tokens', 'temperature', 'top_p', 'top_k', 'logprobs', 
                               'presence_penalty', 'frequency_penalty', 'repetition_penalty',
                               'min_p', 'seed', 'stop', 'stop_token_ids', 'ignore_eos', 'min_tokens']:
                self._sync_param_to_engine(param_name, getattr(self, param_name))


    def __setattr__(self, name, value):
        # Also sync to engine_params if it exists
        super().__setattr__(name, value)
        self._sync_param_to_engine(name, value)


    def _sync_param_to_engine(self, param_name, value):
        """Sync a single parameter to engine_params"""
        if not hasattr(self, 'engine') or self.engine is None:
            raise ValueError("Engine must be set in Sampling_Params to sync parameters to engine_params.")

        if self.engine_params is None:
            raise ValueError("engine_params Class must be set in Sampling_Params to sync parameters to engine_params.")
        
        # Sync logic here
        engine_map = ENGINE_PARAM_MAPS.get(self.engine, {})
        engine_param_name = engine_map.get(param_name)
        # If the engine supports this parameter, set it
        if engine_param_name is not None:
            setattr(self.engine_params, engine_param_name, value)


class Power_Sampling_Params:
    def __init__(
        self, 
        total_output_tokens=1000, # Max sequence length in tokens to generate when power sampling
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
    logprobs_mode='raw_logits', # Mode for logprobs: 'raw_logits' or 'normalized'
    trust_remote_code=True, # Whether to trust remote code when loading the model
    sampling_params=None, # General sampling parameters to use (Sampling_Params Class)
    **kwargs
):
                                
    print(f"Loading model {model} with {engine}...")
    if(engine == "vllm"):
        # Create the LLM object
        llm = vllm_backend.create_LLM_object(
            model_name=model, 
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_logprobs=max_logprobs,
            logprobs_mode=logprobs_mode,
            **kwargs
        )
        # Set the autoregressive sampler function
        autoregressive_sampler = vllm_backend.sample
        # Set the engine name
        autoregressive_sampler.engine = "vllm"

        engine_params = vllm_backend.create_vllm_engine_params()

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
        sampling_params= Sampling_Params(engine = engine, engine_params=engine_params, top_k = max_logprobs, logprobs = max_logprobs) if sampling_params is None else sampling_params
    )

    print("Model loaded successfully. Sampling parameters set to default values.")

    return sampler

# Checks that the LLM and parameters are compatible with power sampling and enables power sampling
def enable_power_sampling(sampler, total_output_tokens, block_size, MCMC_steps):
    # Check if the sampler is initialized
    if(sampler is None):
        raise ValueError("Sampler must be initialized before enabling power sampling.")
    
    # Check the individual engine compatibility for power sampling
    if sampler.engine == "vllm":
        vllm_backend.check_vllm_power_sampling_compatibility(sampler)
    else:
        print(f"Warning: Engine {sampler.engine} not supported for Power Sampling.")

    # Set the power sampling parameters
    sampler.power_sampling_params = Power_Sampling_Params(
        total_output_tokens=total_output_tokens,
        block_size=block_size,
        MCMC_steps=MCMC_steps
    )

    print(f"Power Sampling Enabled: Logits Consider = {sampler.sampling_params.top_k}, Total Output Tokens = {total_output_tokens}, Block Size = {block_size}, MCMC Steps = {MCMC_steps}, Temperature (1/alpha) = {sampler.sampling_params.temperature}")

# TO DO once SMC is implemented
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

# TO DO once best-of sampling is implemented
def enable_best_of_sampling(sampler, n, best_of):
    pass