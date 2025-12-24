import pytest
from pita.inference.LLM_backend import AutoregressiveSampler
from transformers import AutoTokenizer
import pita.inference.vllm_backend as vllm_backend

# Initalize the AutoregressiveSampler
@pytest.fixture()
def autoregressive_sampler():
    sampler = AutoregressiveSampler(
        engine="vllm",
        model="facebook/opt-125m",
        dtype="auto",
        tokenizer_path=None,
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        max_logprobs=10,
        logits_per_token=10,
        logits_processor=True,
        trust_remote_code=True,
        sampling_params=None
    )
    yield sampler

# Test the init of the AutoregressiveSampler
def test_autoregressive_sampler_init(sampler):
    assert sampler.engine == "vllm"
    assert sampler.model == "facebook/opt-125m"
    assert sampler.dtype == "auto"
    assert sampler.tokenizer == AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=True)
    assert sampler.sampler_fn == vllm_backend.sampler
    assert sampler.chain_sampling is None
    assert sampler.token_sampling is None

def test_sampling_params(sampler):
    assert sampler.sampling_params is not None