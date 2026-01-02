import pytest
import numpy as np
from pita.inference.LLM_backend import AutoregressiveSampler, Output
from pita.sampling.token_metrics import calc_sequence_prob

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

def test_logprobs_full_sequence(sampler):
    """Test calc_sequence_prob with logprobs metric for full sequence."""
    output = Output(
        tokens=[1, 2, 3, 4, 5], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4], [-0.6, -1.1, -1.6]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6, 7, 8]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7, 8, 9]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3, 1.6])
    )
    
    # Calculate expected: exp(sum of logprobs from index 0 to 5)
    expected = np.exp(np.sum(output.top_k_logprobs[:, 0]))
    result = calc_sequence_prob(output, sampler, 0, 5, "logprobs")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_logprobs_partial_sequence(sampler):
    """Test calc_sequence_prob with logprobs metric for partial sequence."""
    output = Output(
        tokens=[1, 2, 3, 4, 5], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4], [-0.6, -1.1, -1.6]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6, 7, 8]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7, 8, 9]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3, 1.6])
    )
    
    # Calculate expected: exp(sum of logprobs from index 1 to 3)
    expected = np.exp(np.sum(output.top_k_logprobs[:, 0][1:3]))
    result = calc_sequence_prob(output, sampler, 1, 3, "logprobs")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_power_distribution_full_sequence(sampler):
    """Test calc_sequence_prob with power_distribution metric for full sequence."""
    sampler.sampling_params.temperature = 0.5
    output = Output(
        tokens=[1, 2, 3, 4], 
        top_k_logits=np.array([[2.0, 1.5, 1.0], [3.0, 2.5, 2.0], [4.0, 3.5, 3.0], [5.0, 4.5, 4.0]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4]]), 
        unprocessed_log_normalization_constant=np.array([1.5, 2.5, 3.5, 4.5]), 
        temp_processed_log_normalization_constant=np.array([2.0, 3.0, 4.0, 5.0]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3])
    )
    
    # Calculate expected: exp(sum of (1/T) * (logits - unprocessed_log_norm))
    power_dist = (1 / sampler.sampling_params.temperature) * (
        output.top_k_logits[:, 0] - np.asarray(output.unprocessed_log_normalization_constant)
    )
    expected = np.exp(np.sum(power_dist))
    result = calc_sequence_prob(output, sampler, 0, 4, "power_distribution")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_power_distribution_partial_sequence(sampler):
    """Test calc_sequence_prob with power_distribution metric for partial sequence."""
    sampler.sampling_params.temperature = 0.8
    output = Output(
        tokens=[1, 2, 3, 4, 5], 
        top_k_logits=np.array([[2.0, 1.5, 1.0], [3.0, 2.5, 2.0], [4.0, 3.5, 3.0], [5.0, 4.5, 4.0], [6.0, 5.5, 5.0]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4], [-0.6, -1.1, -1.6]]), 
        unprocessed_log_normalization_constant=np.array([1.5, 2.5, 3.5, 4.5, 5.5]), 
        temp_processed_log_normalization_constant=np.array([2.0, 3.0, 4.0, 5.0, 6.0]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3, 1.6])
    )
    
    # Test partial sequence from index 1 to 4
    power_dist = (1 / sampler.sampling_params.temperature) * (
        output.top_k_logits[:, 0][1:4] - np.asarray(output.unprocessed_log_normalization_constant)[1:4]
    )
    expected = np.exp(np.sum(power_dist))
    result = calc_sequence_prob(output, sampler, 1, 4, "power_distribution")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_entropy_full_sequence(sampler):
    """Test calc_sequence_prob with entropy metric for full sequence."""
    output = Output(
        tokens=[1, 2, 3, 4], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6, 7]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7, 8]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3])
    )
    
    # Calculate expected: exp(-mean(entropy))
    expected = np.exp(-np.mean(output.entropy))
    result = calc_sequence_prob(output, sampler, 0, 4, "entropy")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_entropy_partial_sequence(sampler):
    """Test calc_sequence_prob with entropy metric for partial sequence."""
    output = Output(
        tokens=[1, 2, 3, 4, 5, 6], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4], [-0.6, -1.1, -1.6], [-0.7, -1.2, -1.7]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6, 7, 8, 9]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7, 8, 9, 10]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3, 1.6, 1.8])
    )
    
    # Calculate expected: exp(-mean(entropy[1:4]))
    expected = np.exp(-np.mean(output.entropy[1:4]))
    result = calc_sequence_prob(output, sampler, 1, 4, "entropy")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_likelihood_confidence_full_sequence(sampler):
    """Test calc_sequence_prob with likelihood_confidence metric for full sequence."""
    output = Output(
        tokens=[1, 2, 3], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7]), 
        entropy=np.array([1.5, 1.2, 1.0])
    )
    
    # Calculate expected: exp(sum(logprobs)) * exp(-mean(entropy))
    likelihood = np.exp(np.sum(output.top_k_logprobs[:, 0]))
    confidence = np.exp(-np.mean(output.entropy))
    expected = likelihood * confidence
    result = calc_sequence_prob(output, sampler, 0, 3, "likelihood_confidence")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_likelihood_confidence_partial_sequence(sampler):
    """Test calc_sequence_prob with likelihood_confidence metric for partial sequence."""
    output = Output(
        tokens=[1, 2, 3, 4, 5], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4], [-0.6, -1.1, -1.6]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6, 7, 8]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7, 8, 9]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3, 1.6])
    )
    
    # Calculate expected: exp(sum(logprobs[2:4])) * exp(-mean(entropy[2:4]))
    likelihood = np.exp(np.sum(output.top_k_logprobs[:, 0][2:4]))
    confidence = np.exp(-np.mean(output.entropy[2:4]))
    expected = likelihood * confidence
    result = calc_sequence_prob(output, sampler, 2, 4, "likelihood_confidence")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_single_token_sequence(sampler):
    """Test calc_sequence_prob with a single token sequence."""
    output = Output(
        tokens=[1, 2, 3], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7]), 
        entropy=np.array([1.5, 1.2, 1.0])
    )
    
    # Test with logprobs for a single token (index 1 to 2)
    expected = np.exp(output.top_k_logprobs[1, 0])
    result = calc_sequence_prob(output, sampler, 1, 2, "logprobs")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_invalid_metric(sampler):
    """Test calc_sequence_prob with an invalid metric."""
    output = Output(
        tokens=[1, 2, 3], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7]), 
        entropy=np.array([1.5, 1.2, 1.0])
    )
    
    with pytest.raises(ValueError, match="Invalid metric"):
        calc_sequence_prob(output, sampler, 0, 3, "invalid_metric")

def test_empty_sequence(sampler):
    """Test calc_sequence_prob with an empty sequence (starting_index == ending_index)."""
    output = Output(
        tokens=[1, 2, 3], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7]), 
        entropy=np.array([1.5, 1.2, 1.0])
    )
    
    # Empty sequence should return exp(0) = 1.0 for logprobs
    result = calc_sequence_prob(output, sampler, 1, 1, "logprobs")
    assert result == pytest.approx(1.0)
    
    # Empty sequence for entropy should return exp(-mean([])) which is exp(nan)
    # Since mean of empty array is nan, we test entropy with a valid range instead
    result = calc_sequence_prob(output, sampler, 1, 2, "entropy")
    expected = np.exp(-np.mean(output.entropy[1:2]))
    assert result == pytest.approx(expected)

def test_different_temperatures(sampler):
    """Test calc_sequence_prob with power_distribution at different temperatures."""
    output = Output(
        tokens=[1, 2, 3], 
        top_k_logits=np.array([[2.0, 1.5, 1.0], [3.0, 2.5, 2.0], [4.0, 3.5, 3.0]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2]]), 
        unprocessed_log_normalization_constant=np.array([1.5, 2.5, 3.5]), 
        temp_processed_log_normalization_constant=np.array([2.0, 3.0, 4.0]), 
        entropy=np.array([1.5, 1.2, 1.0])
    )
    
    # Test with temperature = 0.5
    sampler.sampling_params.temperature = 0.5
    power_dist_05 = (1 / 0.5) * (
        output.top_k_logits[:, 0] - np.asarray(output.unprocessed_log_normalization_constant)
    )
    expected_05 = np.exp(np.sum(power_dist_05))
    result_05 = calc_sequence_prob(output, sampler, 0, 3, "power_distribution")
    assert result_05 == pytest.approx(expected_05)
    
    # Test with temperature = 2.0
    sampler.sampling_params.temperature = 2.0
    power_dist_20 = (1 / 2.0) * (
        output.top_k_logits[:, 0] - np.asarray(output.unprocessed_log_normalization_constant)
    )
    expected_20 = np.exp(np.sum(power_dist_20))
    result_20 = calc_sequence_prob(output, sampler, 0, 3, "power_distribution")
    assert result_20 == pytest.approx(expected_20)
    
    # Results should be different for different temperatures
    assert result_05 != pytest.approx(result_20)
