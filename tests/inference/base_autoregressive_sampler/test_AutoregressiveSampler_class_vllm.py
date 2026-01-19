"""
Test suite for AutoregressiveSampler class with vLLM backend.

This test file supports parameterized model testing. Use:
  - `--vllm-model=opt-125m` (default) for fast testing with small model
  - `--vllm-model=gpt-oss-20b` to test with GPT OSS 20B
  - `--all-vllm-models` to run tests against all configured models
"""
import pytest

# Skip this entire module if vllm is not properly installed
# We need to check for both the vllm package AND the LLM class
try:
    from vllm import LLM, SamplingParams
    # Use the imported names in a no-op to satisfy static analysis tools
    _ = (LLM, SamplingParams)
except ImportError:
    pytest.skip("vLLM is not properly installed (LLM class not available)", allow_module_level=True)

from pita.inference.LLM_backend import AutoregressiveSampler
from transformers import AutoTokenizer
import pita.inference.vllm_backend as vllm_backend
import numpy as np


def tokenizer_chat_template(
    tokenizer: AutoTokenizer,
    enable_thinking: bool,
    system_message: str, 
    user_message: str,
) -> str:
    """
    Apply chat template to format messages for the model.
    
    Args:
        tokenizer: The tokenizer to use.
        enable_thinking: Whether to enable thinking mode.
        system_message: The system message content.
        user_message: The user message content.
    
    Returns:
        str: The formatted prompt string.
    """
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


@pytest.fixture(scope="module")
def sampler(vllm_model_config):
    """
    Initialize the AutoregressiveSampler with the configured model.
    
    Uses model configuration from conftest.py based on CLI options.
    The fixture is module-scoped to avoid reloading the model for each test.
    
    Args:
        vllm_model_config: Model configuration dict from conftest.py
    
    Yields:
        AutoregressiveSampler: Configured sampler instance.
    """
    sampler = AutoregressiveSampler(
        engine="vllm",
        model=vllm_model_config["model"],
        dtype="auto",
        tokenizer_path=None,
        gpu_memory_utilization=vllm_model_config["gpu_memory_utilization"],
        max_model_len=vllm_model_config["max_model_len"],
        max_probs=10,
        logits_processor=True,
        trust_remote_code=True,
        sampling_params=None
    )
    yield sampler
    del sampler.llm
    del sampler.tokenizer
    del sampler


def test_sampler_init(sampler, vllm_model_config):
    """Test the initialization of the AutoregressiveSampler."""
    assert sampler.engine == "vllm"
    assert sampler.model == vllm_model_config["model"]
    assert sampler.llm is not None
    assert sampler.tokenizer.name_or_path == vllm_model_config["model"]
    assert sampler.sample_fn == vllm_backend.sample
    assert sampler.chain_sampling is None
    assert sampler.token_sampling is None


def test_sampling_params_initialized(sampler):
    """Test that the sampling params are initialized."""
    assert sampler.sampling_params is not None


def test_max_tokens(sampler):
    """Test that max_tokens parameter controls output length."""
    sampler.sampling_params.max_tokens = 16
    assert sampler.sampling_params.max_tokens == 16
    output = sampler.sample("Hello")
    assert len(output.tokens) == 16


def test_normalization_constants(sampler):
    """Test that normalization constants are computed correctly."""
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
    """Test temperature parameter affects normalization constants."""
    # Set temperature to 1 
    original_temperature = sampler.sampling_params.temperature
    original_enable_normalization_constants = sampler.sampling_params.enable_normalization_constants
    try:
        sampler.sampling_params.temperature = 1
        assert sampler.sampling_params.temperature == 1
        sampler.sampling_params.enable_normalization_constants = True
        output = sampler.sample("Hello")
        assert np.array_equal(output.unprocessed_log_normalization_constant, output.temp_processed_log_normalization_constant)
    finally:
        sampler.sampling_params.temperature = original_temperature
        sampler.sampling_params.enable_normalization_constants = original_enable_normalization_constants

    # Set temperature to 0.25
    try:
        sampler.sampling_params.temperature = 0.25
        assert sampler.sampling_params.temperature == 0.25
        sampler.sampling_params.enable_normalization_constants = True
        output = sampler.sample("Hello")
        assert not np.array_equal(output.unprocessed_log_normalization_constant, output.temp_processed_log_normalization_constant)
    finally:
        sampler.sampling_params.temperature = original_temperature
        sampler.sampling_params.enable_normalization_constants = original_enable_normalization_constants
    

def test_prob_outputs(sampler):
    """Test logprobs and logits output parameters."""
    # Preserve original settings to avoid leaking state to other tests
    original_logprobs_per_token = sampler.sampling_params.logprobs_per_token
    original_logits_per_token = sampler.sampling_params.logits_per_token
    try:
        # Set logprobs_per_token to 4
        sampler.sampling_params.logprobs_per_token = 4
        # set logits_per_token to 6
        sampler.sampling_params.logits_per_token = 6
        output = sampler.sample("Hello")
        assert len(output.top_k_logprobs[0]) >= 4
        assert len(output.top_k_logprobs[0]) < 6
        assert len(output.top_k_logits[0]) >= 6
        assert len(output.top_k_logits[0]) < 8

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
    """Test that logit to logprob conversion is mathematically correct."""
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


# TODO verify if the entropy calculation is actually mathematically correct
def test_entropy(sampler):
    """Test entropy calculation."""
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


def test_decode_tokens(sampler):
    """Test that generated tokens can be decoded back to text."""
    sampler.sampling_params.max_tokens = 20
    sampler.sampling_params.temperature = 1.0
    
    output = sampler.sample("Once upon a time")
    
    # Decode the tokens
    decoded_text = sampler.tokenizer.decode(output.tokens, skip_special_tokens=True)
    
    # Verify we got non-empty text
    assert decoded_text is not None
    assert len(decoded_text) > 0
    assert isinstance(decoded_text, str)


# TODO Test the tokenizer_path parameter