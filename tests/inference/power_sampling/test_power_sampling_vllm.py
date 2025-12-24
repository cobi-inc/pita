import pytest
from pita.inference.LLM_backend import AutoregressiveSampler
from transformers import AutoTokenizer
import pita.inference.vllm_backend as vllm_backend
# Constants
MODEL = "Qwen/Qwen3-4B-AWQ"

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
        engine="vllm",
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

def test_power_sampling_enable(sampler):
    # Enable power sampling
    sampler.enable_power_sampling(block_size=192, MCMC_steps=8, token_metric="power_distribution")
    # Check to see if power sampling is enabled
    assert sampler.token_sampling.block_size == 192
    assert sampler.token_sampling.MCMC_steps == 8
    assert sampler.token_sampling.token_metric == "power_distribution"

    # Check to see if sampler.token_sampling is a Power_Sampling object
    assert isinstance(sampler.token_sampling, Power_Sampling)