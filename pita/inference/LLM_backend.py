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
import time

# Engine-specific parameter mappings
# llama_cpp does not have a separate engine_params class, so it is not included here
ENGINE_PARAM_MAPS = {
    'vllm': {
        'max_tokens': 'max_tokens',
        'temperature': 'temperature',
        'top_p': 'top_p',
        'top_k': 'top_k',
        'logprobs_per_token': 'logprobs',
        'logits_per_token': 'logprobs',
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

class Sampling_Params:
    """Sampling parameters used for generating results from the LLM. Generalized across all engines. Changes to this class should be reflected in the engine specific parameter classes.

    Args:
        engine (str): Engine name (e.g., "vllm", "transformers", etc.).
        engine_params (object): Engine specific parameter Class (vLLM: SamplingParams, llama.cpp: None).
        enable_thinking (bool): Whether to enable thinking.
        max_tokens (int): Max Number of tokens to generate per sequence.
        temperature (float): Controls randomness of sampling. Lower is more deterministic, higher is more random.
        top_p (float): Controls tokens to consider based on cumulative probability. Must be in (0, 1].
        top_k (int): Controls number of top tokens to consider. 0 or -1 considers all tokens.
        logprobs_per_token (int): Number of logprobs to return per output token. logprobs+1 token returned (includes chosen token).
        logits_per_token (int): Number of descending ranked logits to return per output token.
        presence_penalty (float): Penalizes new tokens based on appearance in generated text so far. > 0 encourages new tokens, < 0 encourages repeats.
        frequency_penalty (float): Penalizes new tokens based on frequency in generated text so far. > 0 encourages new tokens, < 0 encourages repeats.
        repetition_penalty (float): Penalizes new tokens based on appearance in prompt AND generated text so far. > 1 encourages new tokens, < 1 encourages repeats.
        min_p (float): Represents the minimum probability for a token to be considered. 0 disables.
        seed (int): Random seed.
        stop (list[str]): Strings that stop token generation. Returned output excludes stop strings.
        stop_token_ids (list[int]): Token IDs that stop token generation. Returned output excludes stop tokens.
        ignore_eos (bool): Continues generating tokens after EOS token is generated.
        min_tokens (int): Minimum Number of tokens to generate per sequence before EOS or stop is considered.
        enable_normalization_constants (bool): Whether to enable normalization constants.
        enable_entropy (bool): Whether to enable entropy.
    """
    def __init__(
        self,
        engine: str = None,
        engine_params: object = None,
        enable_thinking: bool = False,
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        logprobs_per_token: int = None,
        logits_per_token: int = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_p: float = 0.0,
        seed: int = None,
        stop: list[str] = None,
        stop_token_ids: list[int] = None,
        ignore_eos: bool = False,
        min_tokens: int = 0,
        enable_normalization_constants: bool = False,
        enable_entropy: bool = False
    ):  
        object.__setattr__(self, 'engine', engine)
        object.__setattr__(self, 'engine_params', engine_params)
        object.__setattr__(self, 'enable_thinking', enable_thinking)
        object.__setattr__(self, 'max_tokens', max_tokens)
        object.__setattr__(self, 'temperature', temperature)
        object.__setattr__(self, 'top_p', top_p)
        object.__setattr__(self, 'top_k', top_k)
        object.__setattr__(self, 'logprobs_per_token', logprobs_per_token)
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
        object.__setattr__(self, 'enable_normalization_constants', enable_normalization_constants)
        object.__setattr__(self, 'enable_entropy', enable_entropy)

        # Sync all parameters to engine_params after initialization
        if engine is not None and engine_params is not None:
            for param_name in ['max_tokens', 'temperature', 'top_p', 'top_k', 'logprobs_per_token', 'logits_per_token',
                               'presence_penalty', 'frequency_penalty', 'repetition_penalty',
                               'min_p', 'seed', 'stop', 'stop_token_ids', 'ignore_eos', 'min_tokens']:
                self._sync_param_to_engine(param_name, getattr(self, param_name))


    def __setattr__(self, name, value):
        # Also sync to engine_params if it exists
        super().__setattr__(name, value)

        # If attribute is dependent on a Logits Processor, makes sure to propogate the change
        if(self.engine == "vllm"):
            if(name == "enable_normalization_constants"):
                self.engine_params.extra_args["normalization_constants"] = value
                return
            elif(name == "enable_entropy"):
                self.engine_params.extra_args["entropy"] = value
                return

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
        
        # Check if engine is vLLM and logprobs/logits are being changed
        if self.engine == "vllm":
            if(param_name == "logprobs_per_token"):
                if(value < self.logits_per_token):
                    # Do not overwrite the vLLM engine parameter "logprobs" as logits_per_token will fail
                    return
            if(param_name == "logits_per_token"):
                if(value < self.logprobs_per_token):
                    # Do not overwrite the vLLM engine parameter "logits_per_token" as logprobs_per_token will fail
                    return
        
        # Sync logic here
        engine_map = ENGINE_PARAM_MAPS.get(self.engine, {})
        engine_param_name = engine_map.get(param_name)
        # If the engine supports this parameter, set it
        if engine_param_name is not None:
            setattr(self.engine_params, engine_param_name, value)

class Output:
    """ Output object for any LLM sampling.
    
    Attributes:
        tokens (list[int] | list[list[int]]): The generated token IDs.
        top_k_logits (list[float] | list[list[float]] | None): The top_k logits (if logits_per_token is set). First value is always the chosen token logit.
        top_k_logprobs (list[float] | list[list[float]] | None): The top_k logprobs (if logprobs is set). First value is always the chosen token logprob.
        unprocessed_log_normalization_constant (list[float] | list[list[float]]): The log(Normalization Constants - Unprocessed) for each token.
        temp_processed_log_normalization_constant (list[float] | list[list[float]]): The log(Normalization Constants - Temperature Processed) for each token.
        entropy (list[float] | list[list[float]]): The entropy for each token.
    """
    def __init__(
        self,
        tokens: list[int] | list[list[int]] = None,
        top_k_logits: list[float] | list[list[float]] | None = None,
        top_k_logprobs: list[float] | list[list[float]] | None = None,
        unprocessed_log_normalization_constant: list[float] | list[list[float]] = None,
        temp_processed_log_normalization_constant: list[float] | list[list[float]] = None,
        entropy: list[float] | list[list[float]] = None,
    ):
        self.tokens = tokens
        self.top_k_logits = top_k_logits
        self.top_k_logprobs = top_k_logprobs
        self.unprocessed_log_normalization_constant = unprocessed_log_normalization_constant
        self.temp_processed_log_normalization_constant = temp_processed_log_normalization_constant
        self.entropy = entropy

    def append(self, other: 'Output'):
        """
        Appends the data from another Output object to this one by extending internal lists.
        
        Args:
            other (Output): The other output object to append.
        """
        if other is None:
            return

        # Helper function to extend list attributes safely
        def _extend_field(field_name):
            self_val = getattr(self, field_name)
            other_val = getattr(other, field_name)
            
            if other_val is not None:
                if self_val is None:
                    # If we don't have the list yet, shallow copy it from the other
                    setattr(self, field_name, other_val.copy() if isinstance(other_val, list) else other_val)
                elif isinstance(self_val, list) and isinstance(other_val, list):
                    self_val.extend(other_val)

        _extend_field('tokens')
        _extend_field('top_k_logits')
        _extend_field('top_k_logprobs')
        _extend_field('unprocessed_log_normalization_constant')
        _extend_field('temp_processed_log_normalization_constant')
        _extend_field('entropy')

class AutoregressiveSampler:
    """Stores parameters concerning the LLM, autoregressive sampling, and power sampling.

    Attributes:
        engine (str): The engine used for sampling.
        model (str): The LLM Model name.
        llm (object): LLM object from engine used for inference/sampling.
        tokenizer (object): Tokenizer to use for encoding/decoding (HuggingFace AutoTokenizer).
        sample_fn (object): Standard Sampling Function to use for sampling from the autoregressive model without test time scaling.
        sampling_params (object): Parameters to use for standard sampling.
        chain_sampling (object): Chain Sampling Object used for chain level test time scaling (i.e Best-of-N, SMC, etc.)
        token_sampling (object): Token Sampling Object used for token level test time scaling (i.e Metropolis-Hastings Sampling)
        chain_sample_fn (object): The chain sampling function to use for chain level test time scaling.
        token_sample_fn (object): The token sampling function to use for token level test time scaling.
    """
    def __init__(
        self,
        engine: str,
        model: str,
        dtype: str,
        tokenizer_path: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_probs: int,
        logits_processor: bool,
        trust_remote_code: bool,
        sampling_params: Sampling_Params,
        **kwargs
    ):      
    
        """Create an AutoregressiveSampler object given the engine, engine parameters, and model name.

        Args:
            engine (str): Engine to use for autoregressive sampling. Currently only "vllm" and "llama_cpp" are supported.
            model (str): Model to load.
            dtype (str): Data type to use when loading the model. "auto" lets the engine decide.
            tokenizer_path (str): Path to a model with a tokenizer if the model path doesn't include a tokenizer.
            gpu_memory_utilization (float): GPU memory utilization to use.
            max_model_len (int): Max model context length (context window = prompt + generated tokens).
            max_probs (int): Number of top ranked probabilites (logits & logprobs) to store per output token.
            logits_processor (bool): Whether to enable the internal logits processor that allows for normalization constants and entropy to be calculated.
            trust_remote_code (bool): Whether to trust remote code when loading the model.
            sampling_params (Sampling_Params): General sampling parameters to use (Sampling_Params Class).
            **kwargs: Additional keyword arguments passed to the backend LLM creation function.

        Returns:
            An AutoregressiveSampler object.
            
        Raises:
            ValueError: If the engine is not supported.
        """
        self.engine = engine
        self.model = model

        print(f"Loading model {model} with {engine}...")
        
        # Seperate Backend Loading for each engine
        if(engine == "vllm"):
            backend = _get_vllm_backend()

            if(max_probs > 0 and logits_processor == False):
                print("max_probs is set but logits_processor is False. Setting logits_processor to True.")
                logits_processor = True

            # Create the LLM object
            self.llm = backend.create_LLM_object(
                model_name = model, 
                dtype = dtype,
                gpu_memory_utilization = gpu_memory_utilization,
                max_model_len = max_model_len,
                max_probs = max_probs,
                logits_processor = logits_processor,
                **kwargs
            )  
            
            # Set the autoregressive sampler function
            self.sample_fn = backend.sample

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
            self.llm = backend.create_LLM_object(
                model_name = model, 
                model_type = model_type,
                dtype = dtype, 
                gpu_memory_utilization = gpu_memory_utilization, 
                max_model_len = max_model_len,
                logits = logits,
                **kwargs
            )
            # Set the autoregressive sampler function
            self.sample_fn = backend.sample
            # Llama.cpp does not have a separate engine params class
            engine_params = None

        else:
            raise ValueError(f"Engine {engine} not supported for Autoregressive Sampler. Supported engines are: 'vllm', 'llama_cpp'")
        
        # Create tokenizer depending on whether a tokenizer path is provided
        # Needed as some models do not include the tokenizer files in the same repo as the model
        if tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)

        print("Engine Params Extra Args:", engine_params.extra_args if engine_params is not None else "N/A")

        # Intialize the Sampling Params
        if(sampling_params is None):
            self.sampling_params = Sampling_Params(
                engine = engine, 
                engine_params = engine_params, 
                logprobs_per_token = max_probs,
                logits_per_token = max_probs
            )
        else:
            self.sampling_params = sampling_params

        # Intialize the other test-time sampling parameters to Nones 
        self.chain_sampling = None
        self.token_sampling = None

    def sample(self, 
        context: str,
        **kwargs
    )-> Output:
        """Samples programmatical from the LLM given a context and max new tokens. Sample function is the engine_backend.sample function.

        Args:
            context (str): The input context.
            max_new_tokens (int): Maximum number of new tokens to generate.
            **kwargs: Additional keyword arguments passed to the chosen LLM Inference Engine.

        Returns:
            Output: The output of the sample function.
        """
        return self.sample_fn(self, context, **kwargs)

    def token_sample(self, 
        context: str,
        **kwargs
    )-> Output:
        """Samples programmatical from the LLM using the token sampling function

        Args:
            context (str): The input context.
            **kwargs: Additional keyword arguments passed to the chosen LLM Inference Engine.

        Returns:
            Output: The output of the sample function.
        """
        if getattr(self, "token_sample_name", None) == "Power Sampling":
            return self.token_sample_fn(self, context, **kwargs)
        else:
            raise ValueError("Token sampling is not enabled for this LLM/Engine.")

    def chain_sample(self, 
        context: str,
        **kwargs
    )-> Output:
        """Samples programmatical from the LLM using the chain sampling function

        Args:
            context (str): The input context.
            **kwargs: Additional keyword arguments passed to the chosen LLM Inference Engine.

        Returns:
            Output: The output of the sample function.
        """
        if getattr(self, "chain_sample_name", None) == "SMC" or getattr(self, "chain_sample_name", None) == "Best-of-N":
            return self.chain_sample_fn(self, context, **kwargs)
        else:
            raise ValueError("Chain sampling is not enabled for this LLM/Engine.")

    # Chain Sampling Methods
    def enable_smc(
        self,
        num_particles: int,
        tokens_per_step: int,
        stop_on_eos: bool,
        token_metric: str,  
        aggregation: str
    )-> None:
        """
        Enables SMC sampling for the chosen LLM/Engine.

        Args:
            num_particles (int): Number of particles to use for SMC.
            tokens_per_step (int): Number of tokens to generate per step.
            stop_on_eos (bool): (WIP)Whether to stop on end of sequence.
            token_metric (str): Token metric to use to grade each particle. Can be logprobs, power_distribution, entropy, or PRM
            aggregation (str): Aggregation method of the scores of each particle. Can be the last, minimum, product, or model_aggregate.
        
        Returns:
            None
        """
        # Check if chain sampling has already been enabled. If so replace it with SMC.
        if(self.chain_sampling is not None):
            print("Warning: Current Chain Sampling Strategy is being replaced with SMC.")
        
        # Check if the engine/LLM is set up for SMC
        if(token_metric == "PRM"):
            raise ValueError("PRM is not supported YET for SMC.")
        elif(token_metric == "logprobs" or token_metric == "power_distribution" or token_metric == "entropy"):
            if(self.engine == "vllm"):
                vllm_backend.check_token_metric_compatibility(self, token_metric)
            elif(self.engine == "llama.cpp"):
                llama_cpp_backend.check_token_metric_compatibility(self,token_metric)
        else:
            raise ValueError(f"{token_metric} not supported for SMC.")

        # Check if the aggregation method is supported
        if(aggregation == "last" or aggregation == "minimum" or aggregation == "product" or aggregation == "model_aggregate"):
            pass
        else:
            raise ValueError(f"{aggregation} not supported for SMC.")

        # Create the SMC Class
        from pita.sampling.smc import Sequential_Monte_Carlo
        self.chain_sampling = Sequential_Monte_Carlo(
            num_particles=num_particles,
            tokens_per_step=tokens_per_step,
            stop_on_eos=stop_on_eos,
            token_metric=token_metric,
            aggregation=aggregation
        )

        # Set the chain sampling function to the SMC sample function
        self.chain_sample_fn = self.chain_sampling.sample
        self.chain_sample_name = "SMC"

    def enable_best_of_n(
        self,
        sequence_n: int,
        sequence_top_k: int,
        token_metric: str
    )-> None:
        """
        Enables Best-of-N sampling for the chosen LLM/Engine.

        Args:
            sequence_n (int): Number of sequences to sample and choose the best from.
            sequence_top_k (int): Number of top_k sequences to choose from (top_k <= sequence_n). If top_k = 1, greedy selection is used.
            token_metric (str): Token metric to use to grade each sequence. Can be logprobs, power_distribution, entropy, or PRM.
        
        Returns:
            None
        """
        # Check if chain sampling has already been enabled. If so replace it with Best-of-N.
        if(self.chain_sampling is not None):
            print("Warning: Current Chain Sampling Strategy is being replaced with Best-of-N.")
        
        # Check if sequence_top_k is valid
        if(sequence_top_k > sequence_n):
            raise ValueError("sequence_top_k must be less than or equal to sequence_n.")
        
        # Check if the engine/LLM is set up for Best-of-N
        if(token_metric == "PRM"):
            raise ValueError("PRM is not supported YET for Best-of-N.")
        elif(token_metric == "logprobs" or token_metric == "power_distribution" or token_metric == "entropy"):
            if(self.engine == "vllm"):
                vllm_backend.check_token_metric_compatibility(self, token_metric)
            elif(self.engine == "llama.cpp"):
                llama_cpp_backend.check_token_metric_compatibility(self, token_metric)
        else:
            raise ValueError(f"{token_metric} not supported for Best-of-N.")

        # Create the Best-of-N Class
        from pita.sampling.best_of import Best_of_N
        self.chain_sampling = Best_of_N(
            sequence_n=sequence_n,
            sequence_top_k=sequence_top_k,
            token_metric=token_metric
        )

        # Set the chain sampling function to the Best-of-N sample function
        self.chain_sample_fn = self.chain_sampling.sample
        self.chain_sample_name = "Best-of-N"

    # Token Sampling Methods
    def enable_power_sampling(
        self,
        block_size: int,
        MCMC_steps: int,
        token_metric: str,
    )-> None:
        """
        Enables Power Sampling for the chosen LLM/Engine. Checks to see if the engine/LLM is compatible with Power Sampling by verifying that the token metric is supported/available to be used

        Args:
            block_size (int): Number of tokens to generate per step.
            MCMC_steps (int): Number of MCMC steps to use for Power Sampling.
            token_metric (str): Token metric to use to grade each particle. Can be logprobs, power_distribution, entropy, or PRM
        
        Returns:
            None
        """
        # Check if chain sampling has already been enabled. If so replace it with Power Sampling.
        if(self.token_sampling is not None):
            print("Warning: Current Token Sampling Strategy is being replaced with Power Sampling.")
        
        # Check if the engine/LLM is set up for Power Sampling
        if(token_metric == "PRM"):
            raise ValueError("PRM is not supported YET for Power Sampling.")
        elif(token_metric == "logprobs" or token_metric == "power_distribution" or token_metric == "entropy"):
            if(self.engine == "vllm"):
                vllm_backend.check_token_metric_compatibility(self, token_metric)
            elif(self.engine == "llama.cpp"):
                llama_cpp_backend.check_token_metric_compatibility(self,token_metric)
        else:
            raise ValueError(f"{token_metric} not supported for Power Sampling.")

        # Create the Power Sampling Class
        from pita.sampling.power_sample import Power_Sampling
        self.token_sampling = Power_Sampling(
            block_size=block_size,
            MCMC_steps=MCMC_steps,
            token_metric=token_metric
        )

        # Set the token sampling function to the Power Sampling sample function
        self.token_sample_fn = self.token_sampling.sample
        self.token_sample_name = "Power Sampling"
