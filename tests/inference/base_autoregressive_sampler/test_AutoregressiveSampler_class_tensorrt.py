"""
Test suite for AutoregressiveSampler class with TensorRT backend.

This test file supports parameterized model testing via `conftest.py`.
Default model: TinyLlama-1.1B-Chat (requires TensorRT engine build)
"""
import pytest

# Skip this entire module if tensorrt_llm is not installed
tensorrt_llm = pytest.importorskip("tensorrt_llm", reason="TensorRT-LLM is required for these tests")

from pita.inference.LLM_backend import AutoregressiveSampler
from transformers import AutoTokenizer
import pita.inference.tensorRT_backend as tensorrt_backend

# Test Utils
def tokenizer_chat_template(
    tokenizer: AutoTokenizer,
    enable_thinking: bool,
    system_message: str, 
    user_message: str,
) -> str:
    """Apply chat template to prompt."""
    
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
def sampler(tensorrt_model_config):
    """
    Initialize the AutoregressiveSampler with the configured TensorRT model.
    """
    sampler = AutoregressiveSampler(
        engine="tensorrt",
        model=tensorrt_model_config["model"],
        dtype="auto",
        tokenizer_path=None,
        gpu_memory_utilization=tensorrt_model_config["gpu_memory_utilization"],
        max_model_len=tensorrt_model_config["max_model_len"],
        max_probs=1,
        logits_processor=True,
        trust_remote_code=True,
        sampling_params=None
    )
    try:
        yield sampler
    finally:
        del sampler

# Test the init of the AutoregressiveSampler
def test_sampler_init(sampler, tensorrt_model_config):
    assert sampler.engine == "tensorrt"
    assert sampler.model == tensorrt_model_config["model"]
    assert sampler.llm is not None
    assert sampler.tokenizer is not None
    assert sampler.sample_fn == tensorrt_backend.sample
    assert sampler.chain_sampling is None
    assert sampler.token_sampling is None
    assert sampler.sampling_params.logprobs_per_token == 1
    assert sampler.sampling_params.logits_per_token == 1


def test_sampling_params_initialized(sampler):
    # Test that the sampling params are not None
    assert sampler.sampling_params is not None

def test_max_tokens(sampler):
    # Test that the max tokens is set to 16
    sampler.sampling_params.max_tokens = 16
    assert sampler.sampling_params.max_tokens == 16
    output = sampler.sample("Hello. Write a poem about the color blue.")
    assert len(output.tokens) == 16

def test_normalization_constants(sampler):
    # Preserve original settings to avoid leaking state to other tests
    original_enable_normalization_constants = sampler.sampling_params.enable_normalization_constants
    original_max_tokens = sampler.sampling_params.max_tokens
    try:
        # Set normalization constants to True
        sampler.sampling_params.enable_normalization_constants = True
        # Ensure max_tokens is set (may not persist from previous test)
        sampler.sampling_params.max_tokens = 16
        assert sampler.sampling_params.enable_normalization_constants is True
        output = sampler.sample("Hello")
        # Check normalization constants are computed (non-zero)
        assert output.unprocessed_log_normalization_constant[0] != 0
        assert output.temp_processed_log_normalization_constant[0] != 0
    finally:
        # Restore original values to keep tests independent
        sampler.sampling_params.enable_normalization_constants = original_enable_normalization_constants
        sampler.sampling_params.max_tokens = original_max_tokens

def test_temperature(sampler):
    # Preserve original settings to avoid leaking state to other tests
    original_enable_normalization_constants = sampler.sampling_params.enable_normalization_constants
    original_temperature = sampler.sampling_params.temperature
    try:
        # Enable normalization constants (required for this test)
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
        sampler.sampling_params.logprobs_per_token = 1
        # set logits_per_token to 6
        sampler.sampling_params.logits_per_token = 1
        output = sampler.sample("Hello")
        # Check that we get at least some logprobs (TensorRT-LLM may have different behavior)
        assert len(output.top_k_logprobs[0]) == 1

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
    # Preserve original settings to avoid leaking state to other tests
    original_temperature = sampler.sampling_params.temperature
    original_logprobs_per_token = sampler.sampling_params.logprobs_per_token
    original_logits_per_token = sampler.sampling_params.logits_per_token
    try:
        # Set the temperature to 1
        sampler.sampling_params.temperature = 1
        # Set logprobs_per_token and logits_per_token to 1
        sampler.sampling_params.logprobs_per_token = 1
        sampler.sampling_params.logits_per_token = 1
        
        output = sampler.sample("Hello")
        # Check that the logit to logprob conversion is correct when the temperature is 1
        assert output.unprocessed_log_normalization_constant == output.temp_processed_log_normalization_constant    

        # Verify we have logits and logprobs to compare
        if output.top_k_logits and len(output.top_k_logits) > 0 and output.top_k_logits[0]:
            # Check the logit to logprob conversion
            expected_logprob = output.top_k_logits[0][0] - output.temp_processed_log_normalization_constant[0]
            assert output.top_k_logprobs[0][0] == pytest.approx(expected_logprob, rel=1e-3)
    finally:
        # Restore original values to keep tests independent
        sampler.sampling_params.temperature = original_temperature
        sampler.sampling_params.logprobs_per_token = original_logprobs_per_token
        sampler.sampling_params.logits_per_token = original_logits_per_token

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
