import pytest
from pita.inference.LLM_backend import AutoregressiveSampler
from transformers import AutoTokenizer
import pita.inference.llama_cpp_backend as llama_cpp_backend

# Constants
# Using a small GGUF model from Hugging Face for testing
# This model is the GGUF version of facebook/opt-125m
MODEL = "afrideva/opt-125m-GGUF"
TOKENIZER_MODEL = "facebook/opt-125m"

# Test Utils
def tokenizer_chat_template(
    tokenizer: AutoTokenizer,
    enable_thinking: bool,
    system_message: str, 
    user_message: str,
) -> str:

    # Create the message format for apply_chat_template function
    messages = [
        {
            "role": "system",
            # Crucial for benchmarks: explicitly ask for reasoning and boxed format
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    # Apply the chat template to create the final prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = enable_thinking
    )

    return prompt


# Initalize the AutoregressiveSampler
# Do it once for all the tests in this file
@pytest.fixture(scope="module")
def sampler():
    sampler = AutoregressiveSampler(
        engine="llama_cpp",
        model=MODEL,
        dtype="f16",  # Use f16 quantization for GGUF
        tokenizer_path=TOKENIZER_MODEL,  # Need to specify tokenizer separately for GGUF models
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        max_probs=10,
        logits_processor=True,
        trust_remote_code=True,
        sampling_params=None,
        model_type="gguf"  # Explicitly specify GGUF model type
    )
    yield sampler
    del sampler.llm
    del sampler.tokenizer
    del sampler

# Test the init of the AutoregressiveSampler
def test_sampler_init(sampler):
    assert sampler.engine == "llama_cpp"
    assert sampler.model == MODEL
    assert sampler.llm is not None
    assert sampler.tokenizer is not None
    assert sampler.sample_fn == llama_cpp_backend.sample
    assert sampler.chain_sampling is None
    assert sampler.token_sampling is None

def test_sampling_params_initialized(sampler):
    # Test that the sampling params are not None
    assert sampler.sampling_params is not None

def test_max_tokens(sampler):
    # Test that the max tokens is set to 16
    sampler.sampling_params.max_tokens = 16
    assert sampler.sampling_params.max_tokens == 16
    output = sampler.sample("Hello")
    assert len(output.tokens) == 16

def test_normalization_constants(sampler):
    # Set normalization constants to True
    sampler.sampling_params.enable_normalization_constants = True
    assert sampler.sampling_params.enable_normalization_constants == True
    output = sampler.sample("Hello")
    assert output.unprocessed_log_normalization_constant[0] != 0
    assert output.temp_processed_log_normalization_constant[0] != 0

def test_temperature(sampler):
    # Set temperature to 1 
    sampler.sampling_params.temperature = 1
    assert sampler.sampling_params.temperature == 1
    output = sampler.sample("Hello")
    assert output.unprocessed_log_normalization_constant == output.temp_processed_log_normalization_constant

    # Set temperature to 0.25
    sampler.sampling_params.temperature = 0.25
    assert sampler.sampling_params.temperature == 0.25
    output = sampler.sample("Hello")
    assert output.unprocessed_log_normalization_constant != output.temp_processed_log_normalization_constant
    
def test_prob_outputs(sampler):
    # Set logprobs_per_token to 4
    sampler.sampling_params.logprobs_per_token = 4
    # set logits_per_token to 6
    sampler.sampling_params.logits_per_token = 6
    output = sampler.sample("Hello")
    assert len(output.top_k_logprobs[0]) >= 4
    assert len(output.top_k_logprobs[0]) < 6
    assert len(output.top_k_logits[0]) >= 6
    assert len(output.top_k_logits[0]) < 8

    # Set logprobs_per_token and logits_per_token to 0
    sampler.sampling_params.logprobs_per_token = 0
    sampler.sampling_params.logits_per_token = 0
    output = sampler.sample("Hello")
    assert len(output.top_k_logprobs[0]) == 0
    assert len(output.top_k_logits[0]) == 0

def test_logit_to_logprob_conversion(sampler):
    # Set the temperature to 1
    sampler.sampling_params.temperature = 1
    # Set logprobs_per_token and logits_per_token to 1
    sampler.sampling_params.logprobs_per_token = 1
    sampler.sampling_params.logits_per_token = 1
    
    output = sampler.sample("Hello")
    # Check that the logit to logprob conversion is correct when the temperature is 1
    assert output.unprocessed_log_normalization_constant == output.temp_processed_log_normalization_constant    

    # Check the logit to logprob conversion
    assert output.top_k_logprobs[0] == output.top_k_logits[0] - output.temp_processed_log_normalization_constant[0]

    # Set the temperature to 0.25
    sampler.sampling_params.temperature = 0.25
    output = sampler.sample("Hello")
    # Check that the logit to logprob conversion is correct when the temperature is not 1
    assert output.top_k_logprobs[0] != output.top_k_logits[0] - output.temp_processed_log_normalization_constant[0]
    assert output.top_k_logprobs[0] == output.top_k_logits[0] / sampler.sampling_params.temperature - output.temp_processed_log_normalization_constant[0]

def test_entropy(sampler):
    # Enable entropy calculation
    sampler.sampling_params.enable_entropy = True
    output = sampler.sample("Hello")
    assert output.entropy[0] != 0
    
    # Disable entropy calculation
    sampler.sampling_params.enable_entropy = False
    output = sampler.sample("Hello")
    assert output.entropy[0] == 0
