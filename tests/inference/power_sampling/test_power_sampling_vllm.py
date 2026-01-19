"""
Test suite for Power Sampling with vLLM backend.

This test file supports parameterized model testing. Use:
  - `--vllm-model=opt-125m` (default) for fast testing with small model
  - `--vllm-model=gpt-oos-20b` to test with GPT OOS 20B
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

# PITA Libraries
from pita.inference.LLM_backend import Output, AutoregressiveSampler
from pita.sampling.power_sample import Power_Sampling

# Huggingface Libraries
from transformers import AutoTokenizer

# Standard Libraries
import os

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

@pytest.fixture(scope="module")
def sampler(vllm_model_config):
    """
    Initialize the AutoregressiveSampler with the configured model.
    
    Uses model configuration from conftest.py based on CLI options.
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

METRICS = ["logprobs", "power_distribution", "entropy"]

@pytest.mark.parametrize("token_metric", METRICS)
def test_power_sampling_enable(sampler, token_metric):
    # Enable power sampling
    sampler.enable_power_sampling(block_size=192, MCMC_steps=8, token_metric=token_metric)
    # Check to see if power sampling is enabled
    assert sampler.token_sampling.block_size == 192
    assert sampler.token_sampling.MCMC_steps == 8
    assert sampler.token_sampling.token_metric == token_metric

    # Check to see if sampler.token_sampling is a Power_Sampling object
    assert isinstance(sampler.token_sampling, Power_Sampling)

@pytest.mark.parametrize("token_metric", METRICS)
def test_power_sampling_sample(sampler, token_metric):
    sampler.enable_power_sampling(block_size=192, MCMC_steps=8, token_metric=token_metric)
    prompt = tokenizer_chat_template(sampler.tokenizer, False, "You are a helpful assistant.", "Hello, how are you?")
    output = sampler.token_sample(prompt)
    # Check that the output is a valid Output object
    assert isinstance(output, Output)

    # Check that the output has the correct values
    assert len(output.tokens) == len(output.top_k_logits) == len(output.top_k_logprobs) == len(output.unprocessed_log_normalization_constant) == len(output.temp_processed_log_normalization_constant) == len(output.entropy)

    # Check that the output has the correct attributes
    assert hasattr(output, "tokens")
    assert hasattr(output, "top_k_logits")
    assert hasattr(output, "top_k_logprobs")
    assert hasattr(output, "unprocessed_log_normalization_constant")
    assert hasattr(output, "temp_processed_log_normalization_constant")
    assert hasattr(output, "entropy")

    # Check that logging works
    args = {
        "logging": True,
        "log_file_path": "power_sampling_log.csv"
    }
    sampler.token_sample(prompt, **args)

    try:
        # Check that the output has the correct values
        assert len(output.tokens) == len(output.top_k_logits) == len(output.top_k_logprobs) == len(output.unprocessed_log_normalization_constant) == len(output.temp_processed_log_normalization_constant) == len(output.entropy)

        # Check that the output has the correct attributes
        assert hasattr(output, "tokens")
        assert hasattr(output, "top_k_logits")
        assert hasattr(output, "top_k_logprobs")
        assert hasattr(output, "unprocessed_log_normalization_constant")
        assert hasattr(output, "temp_processed_log_normalization_constant")
        assert hasattr(output, "entropy")

        # Check for the log files
        assert os.path.exists("power_sampling_log.csv")
    finally:
        if os.path.exists("power_sampling_log.csv"):
            os.remove("power_sampling_log.csv")
