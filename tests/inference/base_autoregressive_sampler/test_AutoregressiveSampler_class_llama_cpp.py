import pytest

# Skip this entire module if llama_cpp is not installed
llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python is required for these tests")

from pita.inference.LLM_backend import AutoregressiveSampler
from transformers import AutoTokenizer
import pita.inference.llama_cpp_backend as llama_cpp_backend

# Constants
# Using TheBloke's TinyLlama GGUF model which has actual GGUF files
MODEL = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
TOKENIZER_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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


# Initialize the AutoregressiveSampler
# Do it once for all the tests in this file
@pytest.fixture(scope="module")
def sampler():
    sampler = AutoregressiveSampler(
        engine="llama_cpp",
        model=MODEL,
        dtype="Q4_K_M",  # Use Q4_K_M quantization for GGUF
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
    output = sampler.sample("Hello. Can you write a story about a cat?")
    # Note: The tokenizer may add BOS tokens when re-encoding the output text.
    # llama_cpp generates max_tokens completion tokens, but re-encoding may differ.
    # We check that we're in a reasonable range.
    assert len(output.tokens) >= 14 and len(output.tokens) <= 20

def test_normalization_constants(sampler):
    # Preserve original setting to avoid leaking state to other tests
    original_enable_normalization_constants = sampler.sampling_params.enable_normalization_constants
    try:
        # Set normalization constants to True
        sampler.sampling_params.enable_normalization_constants = True
        assert sampler.sampling_params.enable_normalization_constants is True
        output = sampler.sample("Hello")
        assert output.unprocessed_log_normalization_constant[0] != 0
        assert output.temp_processed_log_normalization_constant[0] != 0
    finally:
        # Restore original value to keep tests independent
        sampler.sampling_params.enable_normalization_constants = original_enable_normalization_constants

def test_temperature(sampler):
    # Preserve original settings to avoid leaking state to other tests
    original_enable_normalization_constants = sampler.sampling_params.enable_normalization_constants
    original_temperature = sampler.sampling_params.temperature
    try:
        # Enable normalization constants to test temperature effects
        sampler.sampling_params.enable_normalization_constants = True
        
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
    finally:
        # Restore original values to keep tests independent
        sampler.sampling_params.enable_normalization_constants = original_enable_normalization_constants
        sampler.sampling_params.temperature = original_temperature
    
def test_prob_outputs(sampler):
    # Preserve original settings to avoid leaking state to other tests
    original_logprobs_per_token = sampler.sampling_params.logprobs_per_token
    original_logits_per_token = sampler.sampling_params.logits_per_token
    try:
        # Set logprobs_per_token to 4
        sampler.sampling_params.logprobs_per_token = 4
        # set logits_per_token to 6
        sampler.sampling_params.logits_per_token = 6
        output = sampler.sample("Hello")
        assert len(output.top_k_logprobs[0]) == 4
        assert len(output.top_k_logits[0]) == 6

        # Test that disabling these parameters (setting to 0) works correctly
        sampler.sampling_params.logprobs_per_token = 0
        sampler.sampling_params.logits_per_token = 0
        output = sampler.sample("Hello")
        assert len(output.top_k_logprobs[0]) == 0
        assert len(output.top_k_logits[0]) == 0
    finally:
        # Restore original values to keep tests independent
        sampler.sampling_params.logprobs_per_token = original_logprobs_per_token
        sampler.sampling_params.logits_per_token = original_logits_per_token

def test_logit_to_logprob_conversion(sampler):
    # Set the temperature to 1
    sampler.sampling_params.temperature = 1
    # Reset logprobs_per_token and logits_per_token (may have been set to 0 by previous test)
    sampler.sampling_params.logprobs_per_token = 1
    sampler.sampling_params.logits_per_token = 1
    sampler.sampling_params.max_tokens = 16  # Ensure we get some output
    
    output = sampler.sample("Hello")
    # Check that the logit to logprob conversion is correct when the temperature is 1
    assert output.unprocessed_log_normalization_constant == output.temp_processed_log_normalization_constant    

    # Verify we have logits and logprobs to compare
    assert len(output.top_k_logits) > 0, "No logits returned"
    assert len(output.top_k_logits[0]) > 0, "First token has no logits"
    
    # Check the logit to logprob conversion
    # Note: llama_cpp returns lists for top_k_logits/logprobs, so access [0][0] for first token's first logit
    expected_logprob = output.top_k_logits[0][0] - output.temp_processed_log_normalization_constant[0]
    assert output.top_k_logprobs[0][0] == pytest.approx(expected_logprob)

    # Set the temperature to 0.25
    sampler.sampling_params.temperature = 0.25
    output = sampler.sample("Hello")
    # Check that the logit to logprob conversion is correct when the temperature is not 1
    expected_logprob_temp = output.top_k_logits[0][0] / sampler.sampling_params.temperature - output.temp_processed_log_normalization_constant[0]
    assert output.top_k_logprobs[0][0] == pytest.approx(expected_logprob_temp)

def test_entropy(sampler):
    original_enable_entropy = sampler.sampling_params.enable_entropy
    try:
        # Enable entropy calculation
        sampler.sampling_params.enable_entropy = True
        output = sampler.sample("Hello")
        assert output.entropy[0] != 0

        # Disable entropy calculation
        sampler.sampling_params.enable_entropy = False
        output = sampler.sample("Hello")
        assert output.entropy[0] == 0
    finally:
        sampler.sampling_params.enable_entropy = original_enable_entropy
