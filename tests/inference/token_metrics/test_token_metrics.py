import pytest
import numpy as np
from unittest.mock import MagicMock
from pita.inference.LLM_backend import Output
from pita.sampling.token_metrics import calc_token_metric


@pytest.fixture(scope="module")
def sampler():
    """Create a mock sampler that provides the necessary sampling_params."""
    mock_sampler = MagicMock()
    mock_sampler.sampling_params = MagicMock()
    mock_sampler.sampling_params.temperature = 1.0
    yield mock_sampler

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
    # Subtract the unprocessed_log_normalization_constant from the logits and scale by the inverse temperature
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
    entropy = calc_token_metric(output, sampler, "entropy")
    assert entropy.shape == (3,)
    assert np.allclose(entropy, output.entropy)

def test_likelihood_confidence(sampler):
    output = Output(
        tokens=[1,2,3], 
        top_k_logits=np.array([[1,2,3],[2,3,4],[3,4,5]]), 
        top_k_logprobs=np.array([[1,2,3],[4,5,6],[7,8,9]]), 
        unprocessed_log_normalization_constant=np.array([4,5,6]), 
        temp_processed_log_normalization_constant=np.array([5,6,7]), 
        entropy=np.array([6,7,8])
    )
    likelihood_confidence = calc_token_metric(output, sampler, "likelihood_confidence")
    assert likelihood_confidence.shape == (3,)
    # Manually calculate expected values: logprobs[:, 0] - entropy
    expected = output.top_k_logprobs[:, 0] - output.entropy
    assert np.allclose(likelihood_confidence, expected)