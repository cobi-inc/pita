import pytest
import numpy as np
from pita.inference.LLM_backend import AutoregressiveSampler, Output
from pita.sampling.token_metrics import calc_token_metric

# Constants
MODEL = "facebook/opt-125m"

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

def test_logprobs(sampler):
    output = Output(
        tokens=[1,2,3], 
        top_k_logits=np.array([[1,2,3],[2,3,4],[3,4,5]]), 
        top_k_logprobs=np.array([[1,2,3],[4,5,6],[7,8,9]]), 
        unprocessed_log_normalization_constant=np.array([4,5,6]), 
        temp_processed_log_normalization_constant=np.array([5,6,7]), 
        entropy=np.array([6,7,8])
    )
    logprobs = calc_token_metric(output,sampler,"logprobs")
    assert logprobs.shape == (3,)
    assert np.allclose(logprobs, output.top_k_logprobs[:, 0])

def test_power_distribution(sampler):
    sampler.sampling_params.temperature = 0.5
    output = Output(
        tokens=[1,2,3], 
        top_k_logits=np.array([[1,2,3],[2,3,4],[3,4,5]]), 
        top_k_logprobs=np.array([[1,2,3],[4,5,6],[7,8,9]]), 
        unprocessed_log_normalization_constant=np.array([4,5,6]), 
        temp_processed_log_normalization_constant=np.array([5,6,7]), 
        entropy=np.array([6,7,8])
    )
    power_distribution = calc_token_metric(output, sampler, "power_distribution")
    assert power_distribution.shape == (3,)
    # Find the power distribution for the first token
    # Divide the logits by the unprocessed_log_normalization_constant
    calc_power_distribution = (1./sampler.sampling_params.temperature) * (output.top_k_logits[:, 0] - output.unprocessed_log_normalization_constant)
    assert np.allclose(power_distribution, calc_power_distribution)

def test_entropy(sampler):
    output = Output(
        tokens=[1,2,3], 
        top_k_logits=np.array([[1,2,3],[2,3,4],[3,4,5]]), 
        top_k_logprobs=np.array([[1,2,3],[4,5,6],[7,8,9]]), 
        unprocessed_log_normalization_constant=np.array([4,5,6]), 
        temp_processed_log_normalization_constant=np.array([5,6,7]), 
        entropy=np.array([6,7,8])
    )
    entropy = calc_token_metric(output,sampler,"entropy")
    assert entropy.shape == (3,)
    assert np.allclose(entropy, output.entropy)
    