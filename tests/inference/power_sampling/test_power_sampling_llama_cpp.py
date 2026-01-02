import pytest

# Skip this entire module if llama_cpp is not installed
llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python is required for these tests")

# PITA Libraries
from pita.inference.LLM_backend import Output, AutoregressiveSampler
from pita.sampling.power_sample import Power_Sampling

# Huggingface Libraries
from transformers import AutoTokenizer

# Standard Libraries
import os

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
    # Use a simple prompt since OPT models don't have chat templates
    prompt = "Hello, how are you?"
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
        "log_file_path": "power_sampling_log_llama_cpp.csv"
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
        assert os.path.exists("power_sampling_log_llama_cpp.csv")
    finally:
        if os.path.exists("power_sampling_log_llama_cpp.csv"):
            os.remove("power_sampling_log_llama_cpp.csv")
