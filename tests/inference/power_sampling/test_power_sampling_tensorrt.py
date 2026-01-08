"""
TensorRT-LLM Power Sampling Tests.

This module tests the power sampling integration with the TensorRT-LLM backend,
following the same pattern as test_power_sampling_vllm.py and 
test_power_sampling_llama_cpp.py.
"""

import pytest

# Skip this entire module if tensorrt_llm is not installed
tensorrt_llm = pytest.importorskip(
    "tensorrt_llm", 
    reason="TensorRT-LLM is required for these tests"
)

# PITA Libraries
from pita.inference.LLM_backend import Output, AutoregressiveSampler
from pita.sampling.power_sample import Power_Sampling

# Huggingface Libraries
from transformers import AutoTokenizer

# Standard Libraries
import os

# Constants
# Using the same model as test_AutoregressiveSampler_class_tensorrt.py
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def tokenizer_chat_template(
    tokenizer: AutoTokenizer,
    enable_thinking: bool,
    system_message: str, 
    user_message: str,
) -> str:
    """
    Apply chat template to format messages for the model.
    
    Args:
        tokenizer: The tokenizer to use for applying the chat template.
        enable_thinking: Whether to enable thinking mode in the template.
        system_message: The system message content.
        user_message: The user message content.
    
    Returns:
        str: The formatted prompt string.
    """
    # Create the message format for apply_chat_template function
    messages = [
        {
            "role": "system",
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
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    return prompt


@pytest.fixture(scope="module")
def sampler():
    """
    Create an AutoregressiveSampler for TensorRT-LLM that persists across all tests.
    
    This fixture is module-scoped to avoid repeated model loading overhead.
    """
    sampler = AutoregressiveSampler(
        engine="tensorrt",
        model=MODEL,
        dtype="auto",
        tokenizer_path=None,
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        max_probs=10,
        logits_processor=True,
        trust_remote_code=True,
        sampling_params=None
    )
    yield sampler
    del sampler.llm
    del sampler.tokenizer
    del sampler


# Token metrics to test
METRICS = ["logprobs", "power_distribution", "entropy"]


@pytest.mark.parametrize("token_metric", METRICS)
def test_power_sampling_enable(sampler, token_metric):
    """
    Test that power sampling can be enabled with correct parameters.
    
    Verifies:
        - block_size is correctly set
        - MCMC_steps is correctly set
        - token_metric is correctly set
        - sampler.token_sampling is a Power_Sampling instance
    """
    # Enable power sampling
    sampler.enable_power_sampling(block_size=192, MCMC_steps=8, token_metric=token_metric)
    
    # Check that power sampling is enabled with correct parameters
    assert sampler.token_sampling.block_size == 192
    assert sampler.token_sampling.MCMC_steps == 8
    assert sampler.token_sampling.token_metric == token_metric

    # Check that sampler.token_sampling is a Power_Sampling object
    assert isinstance(sampler.token_sampling, Power_Sampling)


@pytest.mark.parametrize("token_metric", METRICS)
def test_power_sampling_sample(sampler, token_metric):
    """
    Test that power sampling produces valid Output objects with all expected attributes.
    
    Verifies:
        - Output is a valid Output instance
        - All output lists have consistent lengths
        - All expected attributes are present
        - Logging functionality works correctly
    """
    sampler.enable_power_sampling(block_size=192, MCMC_steps=8, token_metric=token_metric)
    
    # Use a simple prompt for testing
    prompt = "Hello, how are you?"
    output = sampler.token_sample(prompt)
    
    # Check that the output is a valid Output object
    assert isinstance(output, Output)

    # Check that the output lists have consistent lengths
    assert len(output.tokens) == len(output.top_k_logits) == len(output.top_k_logprobs) == len(output.unprocessed_log_normalization_constant) == len(output.temp_processed_log_normalization_constant) == len(output.entropy)

    # Check that the output has all expected attributes
    assert hasattr(output, "tokens")
    assert hasattr(output, "top_k_logits")
    assert hasattr(output, "top_k_logprobs")
    assert hasattr(output, "unprocessed_log_normalization_constant")
    assert hasattr(output, "temp_processed_log_normalization_constant")
    assert hasattr(output, "entropy")

    # Check that logging works
    log_file = "power_sampling_log_tensorrt.csv"
    args = {
        "logging": True,
        "log_file_path": log_file
    }
    sampler.token_sample(prompt, **args)

    try:
        # Verify output attributes again after logging call
        assert len(output.tokens) == len(output.top_k_logits) == len(output.top_k_logprobs) == len(output.unprocessed_log_normalization_constant) == len(output.temp_processed_log_normalization_constant) == len(output.entropy)

        assert hasattr(output, "tokens")
        assert hasattr(output, "top_k_logits")
        assert hasattr(output, "top_k_logprobs")
        assert hasattr(output, "unprocessed_log_normalization_constant")
        assert hasattr(output, "temp_processed_log_normalization_constant")
        assert hasattr(output, "entropy")

        # Check that the log file was created
        assert os.path.exists(log_file)
    finally:
        # Clean up the log file
        if os.path.exists(log_file):
            os.remove(log_file)
