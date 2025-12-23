#Tokenizers
from transformers import AutoTokenizer

# Custom Libraries
# Lazy imports for backends - will be imported when needed
vllm_backend = None
llama_cpp_backend = None

def _get_vllm_backend():
    global vllm_backend
    if vllm_backend is None:
        import pita.inference.vllm_backend as _vllm_backend
        vllm_backend = _vllm_backend
    return vllm_backend

def _get_llama_cpp_backend():
    global llama_cpp_backend
    if llama_cpp_backend is None:
        import pita.inference.llama_cpp_backend as _llama_cpp_backend
        llama_cpp_backend = _llama_cpp_backend
    return llama_cpp_backend

# Utils
from pita.utils.system_utils import detect_model_type
import time
from pita.utils.constants import REDIS_HOST, REDIS_PORT

# Memory Management
import redis

# Engine-specific parameter mappings
# llama_cpp does not have a separate engine_params class, so it is not included here
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
    # Add more engines as needed
}

class AutoregressiveSampler:
    """Stores parameters concerning the LLM, autoregressive sampling, and power sampling.

    Attributes:
        engine (str): The engine used for sampling.
        model (str): The LLM Model name.
        llm (object): LLM object to use for sampling.
        tokenizer (object): Tokenizer to use for encoding/decoding (HuggingFace AutoTokenizer).
        sample_fn (object): Function to use for sampling from the autoregressive model.
        sampling_params (object): Parameters to use for standard sampling.
        power_sampling_params (object): Parameters to use for power sampling.
        smc_sampling_params (object): Parameters to use for SMC sampling.
        best_of_sampling_params (object): Parameters to use for best-of sampling.
    """
    def __init__(
        self,
        engine: str,
        model: str,
        llm: object,
        tokenizer: object,
        sample_fn: object,
        sampling_params: object,
        power_sampling_params: object = None,
        smc_sampling_params: object = None,
        best_of_sampling_params: object = None
    ):  
        self.engine = engine
        self.model = model
        self.llm = llm
        self.tokenizer = tokenizer
        self.sample_fn = sample_fn
        self.sampling_params = sampling_params
        self.power_sampling_params = power_sampling_params
        self.smc_sampling_params = smc_sampling_params
        self.best_of_sampling_params = best_of_sampling_params
        
    def sample(self, 
        context: str, 
        max_new_tokens: int):
        """Samples programmatical from the LLM given a context and max new tokens. Sample function is the engine_backend.sample function.

        Args:
            context (str): The input context.
            max_new_tokens (int): Maximum number of new tokens to generate.
            **kwargs: Additional keyword arguments passed to the chosen LLM Inference Engine.

        Returns:
            tokens: list[int] | list[list[int]]: The generated token IDs.
            top_k_logits: list[float] | list[list[float]] | None: The top_k logits (if logits_per_token is set).
            top_k_logprobs: list[float] | list[list[float]] | None: The top_k logprobs (if logprobs is set).
            unprocessed_log_normalization_constant: list[float] | list[list[float]]: The log(Normalization Constants - Unprocessed) for each token.
            temp_processed_log_normalization_constant: list[float] | list[list[float]]: The log(Normalization Constants - Temperature Processed) for each token.
            entropy: list[float] | list[list[float]]: The entropy for each token.
        """
        return self.sample_fn(self, context, max_new_tokens)

class Sampling_Params:
    """Sampling parameters used for generating results from the LLM. Generalized across all engines. Changes to this class should be reflected in the engine specific parameter classes.

    Args:
        engine: Engine name (e.g., "vllm", "transformers", etc.).
        engine_params: Engine specific parameter Class (vLLM: SamplingParams, llama.cpp: None).
        enable_thinking: Whether to enable thinking.
        max_tokens: Max Number of tokens to generate per sequence.
        temperature: Controls randomness of sampling. Lower is more deterministic, higher is more random.
        top_p: Controls tokens to consider based on cumulative probability. Must be in (0, 1].
        top_k: Controls number of top tokens to consider. 0 or -1 considers all tokens.
        logprobs: Number of logits/logprobs to return per output token. logprobs+1 token returned (includes chosen token). -1 returns all vocab_size log probabilities.
        logits_per_token: Number of descending ranked logits to return per output token.
        presence_penalty: Penalizes new tokens based on appearance in generated text so far. > 0 encourages new tokens, < 0 encourages repeats.
        frequency_penalty: Penalizes new tokens based on frequency in generated text so far. > 0 encourages new tokens, < 0 encourages repeats.
        repetition_penalty: Penalizes new tokens based on appearance in prompt AND generated text so far. > 1 encourages new tokens, < 1 encourages repeats.
        min_p: Represents the minimum probability for a token to be considered. 0 disables.
        seed: Random seed.
        stop: Strings that stop token generation. Returned output excludes stop strings.
        stop_token_ids: Token IDs that stop token generation. Returned output excludes stop tokens.
        ignore_eos: Continues generating tokens after EOS token is generated.
        min_tokens: Minimum Number of tokens to generate per sequence before EOS or stop is considered.
        normalization_constants: Normalization constants.
        entropy: Entropy.
    """
    def __init__(
        self,
        engine = None,
        engine_params = None,
        enable_thinking = False,
        max_tokens = 16,
        temperature = 1.0,
        top_p = 1.0,
        top_k = -1,
        logprobs = None,
        logits_per_token = None,
        presence_penalty = 0.0,
        frequency_penalty = 0.0,
        repetition_penalty = 1.0,
        min_p = 0.0,
        seed = None,
        stop = None,
        stop_token_ids = None,
        ignore_eos = False,
        min_tokens = 0,
        normalization_constants = None,
        entropy = None
    ):  
        object.__setattr__(self, 'engine', engine)
        object.__setattr__(self, 'engine_params', engine_params)
        object.__setattr__(self, 'enable_thinking', enable_thinking)
        object.__setattr__(self, 'max_tokens', max_tokens)
        object.__setattr__(self, 'temperature', temperature)
        object.__setattr__(self, 'top_p', top_p)
        object.__setattr__(self, 'top_k', top_k)
        object.__setattr__(self, 'logprobs', logprobs)
        object.__setattr__(self, 'logits_per_token', logits_per_token)
        object.__setattr__(self, 'presence_penalty', presence_penalty)
        object.__setattr__(self, 'frequency_penalty', frequency_penalty)
        object.__setattr__(self, 'repetition_penalty', repetition_penalty)
        object.__setattr__(self, 'min_p', min_p)
        object.__setattr__(self, 'seed', seed)
        object.__setattr__(self, 'stop', stop)
        object.__setattr__(self, 'stop_token_ids', stop_token_ids)
        object.__setattr__(self, 'ignore_eos', ignore_eos)
        object.__setattr__(self, 'min_tokens', min_tokens)
        object.__setattr__(self, 'normalization_constants', normalization_constants)
        object.__setattr__(self, 'entropy', entropy)

        # Sync all parameters to engine_params after initialization
        if engine is not None and engine_params is not None:
            for param_name in ['max_tokens', 'temperature', 'top_p', 'top_k', 'logprobs', 'logits_per_token',
                               'presence_penalty', 'frequency_penalty', 'repetition_penalty',
                               'min_p', 'seed', 'stop', 'stop_token_ids', 'ignore_eos', 'min_tokens']:
                self._sync_param_to_engine(param_name, getattr(self, param_name))


    def __setattr__(self, name, value):
        # Also sync to engine_params if it exists
        super().__setattr__(name, value)
        self._sync_param_to_engine(name, value)


    def _sync_param_to_engine(self, param_name, value):
        # Skip syncing for llama_cpp as it does not use a separate engine_params class
        if self.engine == "llama_cpp":
            return

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

def create_autoregressive_sampler(
    engine,
    model,
    dtype = "auto",
    tokenizer_path = None,
    gpu_memory_utilization = 0.85,
    max_model_len = 1024,
    max_logprobs = None,
    logits_per_token = None,
    logits_processor = False,
    trust_remote_code = True,
    sampling_params = None,
    **kwargs
):
    """Create an AutoregressiveSampler object given the engine, engine parameters, and model name.

    Args:
        engine: Engine to use for autoregressive sampling. Currently only "vllm" and "llama_cpp" are supported.
        model: Model to load.
        dtype: Data type to use when loading the model. "auto" lets the engine decide.
        tokenizer_path: Path to a model with a tokenizer if the model path doesn't include a tokenizer.
        gpu_memory_utilization: GPU memory utilization to use.
        max_model_len: Max model context length (context window = prompt + generated tokens).
        max_logprobs: Number of logits/logprobs to store per output token.
        logits_per_token: Number of descending ranked logits to return per output token.
        logits_processor: Whether to enable the internal logits processor that allows for normalization constants and entropy to be calculated.
        trust_remote_code: Whether to trust remote code when loading the model.
        sampling_params: General sampling parameters to use (Sampling_Params Class).
        **kwargs: Additional keyword arguments passed to the backend LLM creation function.

    Returns:
        An AutoregressiveSampler object.
        
    Raises:
        ValueError: If the engine is not supported.
    """
                                
    print(f"Loading model {model} with {engine}...")

    # Determine Model Type for Hugging Face Repos
    model_type = detect_model_type(model)

    # Enable the use of logits if logits_per_token is set
    # This is important as the engine cannot be set to return logits after it is initialized
    if(logits_per_token is not None):
        logits = True
        prob_count = logits_per_token
    else:
        logits = False

    if(engine == "vllm"):
        backend = _get_vllm_backend()

        # vLLM uses both logits and logprobs interchangeably depending on the logprobs_mode set during initialization
        # some libraries have distinct modes for logits vs logprobs like llama_cpp
        # As a logit space library first, we set logprobs_mode to 'raw_logits' when logits=True
        # Additionally, we default to preferring logits_per_token over max_logprobs when both are for clarity
        if(logits == True):
            print("vLLM does not output both logits and logprobs separately. Both max_logprobs and logits_per_token are set. Defaulting to using logits_per_token for vLLM max_logprobs engine parameter and in vLLM's logprobs Sampling_Params.")
            prob_count = logits_per_token
            max_logprobs = logits_per_token
        elif(logits == False and max_logprobs is not None):
            print("vLLM does not output both logits and logprobs separately. max_logprobs is set while logits_per_token is not set. Defaulting to using and returning max_logprobs for vLLM.")
            prob_count = max_logprobs
        else:
            prob_count = None
            
        # Create the LLM object
        llm = backend.create_LLM_object(
            model_name = model, 
            dtype = dtype,
            gpu_memory_utilization = gpu_memory_utilization,
            max_model_len = max_model_len,
            max_logprobs = prob_count,
            logits = logits,
            logits_processor = logits_processor,
            **kwargs
        )  
        
        # Set the autoregressive sampler function
        autoregressive_sampler = backend.sample

        # Create the engine parameters used for the completion function in vLLM
        engine_params = backend.create_vllm_engine_params()

        # Set the redis client for the LogitsLoggingProcessor
        # Add the normalization_constants and normalization_constants_temp_scaled lists to extra_args
        if(logits_processor):
            print("Enabling logits processing in engine parameters extra_args.")
            engine_params.extra_args = {}
            engine_params.extra_args["req_id"] = "my_request_" + str(time.time())
        
    elif(engine == "llama_cpp"):
        backend = _get_llama_cpp_backend()
        # Create the LLM object
        llm = backend.create_LLM_object(
            model_name = model, 
            model_type = model_type,
            dtype = dtype, 
            gpu_memory_utilization = gpu_memory_utilization, 
            max_model_len = max_model_len,
            logits = logits,
            **kwargs
        )
        # Set the autoregressive sampler function
        autoregressive_sampler = backend.sample
        # Llama.cpp does not have a separate engine params class
        engine_params = None

    else:
        raise ValueError(f"Engine {engine} not supported for Autoregressive Sampler. Supported engines are: 'vllm', 'llama_cpp'")
    
    # Create tokenizer depending on whether a tokenizer path is provided
    # Needed as some models do not include the tokenizer files in the same repo as the model
    if tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)

    print("Engine Params Extra Args:", engine_params.extra_args if engine_params is not None else "N/A")

    # Create the Autoregressive Sampler object
    sampler = AutoregressiveSampler(
        engine=engine,
        model=model,
        llm=llm,
        tokenizer=tokenizer,
        sample_fn=autoregressive_sampler,
        sampling_params= Sampling_Params(
            engine = engine, 
            engine_params = engine_params, 
            logprobs = max_logprobs,
            logits_per_token = logits_per_token
            
        ) if sampling_params is None else sampling_params
    )

    return sampler
