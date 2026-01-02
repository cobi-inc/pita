import pytest
import numpy as np
from unittest.mock import MagicMock
from pita.inference.LLM_backend import Output
from pita.sampling.token_metrics import calc_sequence_logprob


@pytest.fixture(scope="module")
def sampler():
    """Create a mock sampler that provides the necessary sampling_params."""
    mock_sampler = MagicMock()
    mock_sampler.sampling_params = MagicMock()
    mock_sampler.sampling_params.temperature = 1.0
    yield mock_sampler

def test_logprobs_full_sequence(sampler):
    """Test calc_sequence_logprob with logprobs metric for full sequence."""
    output = Output(
        tokens=[1, 2, 3, 4, 5], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4], [-0.6, -1.1, -1.6]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6, 7, 8]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7, 8, 9]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3, 1.6])
    )
    
    # Calculate expected: sum of logprobs from index 0 to 5
    expected = np.sum(output.top_k_logprobs[:, 0])
    result = calc_sequence_logprob(output, sampler, 0, 5, "logprobs")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_logprobs_partial_sequence(sampler):
    """Test calc_sequence_logprob with logprobs metric for partial sequence."""
    output = Output(
        tokens=[1, 2, 3, 4, 5], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4], [-0.6, -1.1, -1.6]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6, 7, 8]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7, 8, 9]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3, 1.6])
    )
    
    # Calculate expected: sum of logprobs from index 1 to 3
    expected = np.sum(output.top_k_logprobs[:, 0][1:3])
    result = calc_sequence_logprob(output, sampler, 1, 3, "logprobs")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_power_distribution_full_sequence(sampler):
    """Test calc_sequence_logprob with power_distribution metric for full sequence."""
    sampler.sampling_params.temperature = 0.5
    output = Output(
        tokens=[1, 2, 3, 4], 
        top_k_logits=np.array([[2.0, 1.5, 1.0], [3.0, 2.5, 2.0], [4.0, 3.5, 3.0], [5.0, 4.5, 4.0]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4]]), 
        unprocessed_log_normalization_constant=np.array([1.5, 2.5, 3.5, 4.5]), 
        temp_processed_log_normalization_constant=np.array([2.0, 3.0, 4.0, 5.0]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3])
    )
    
    # Calculate expected: sum of (1/T) * (logits - unprocessed_log_norm)
    power_dist = (1 / sampler.sampling_params.temperature) * (
        output.top_k_logits[:, 0] - np.asarray(output.unprocessed_log_normalization_constant)
    )
    expected = np.sum(power_dist)
    result = calc_sequence_logprob(output, sampler, 0, 4, "power_distribution")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_power_distribution_partial_sequence(sampler):
    """Test calc_sequence_logprob with power_distribution metric for partial sequence."""
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
    expected = np.sum(power_dist)
    result = calc_sequence_logprob(output, sampler, 1, 4, "power_distribution")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_entropy_full_sequence(sampler):
    """Test calc_sequence_logprob with entropy metric for full sequence."""
    output = Output(
        tokens=[1, 2, 3, 4], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6, 7]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7, 8]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3])
    )
    
    # Calculate expected: -mean(entropy)
    expected = -np.mean(output.entropy)
    result = calc_sequence_logprob(output, sampler, 0, 4, "entropy")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_entropy_partial_sequence(sampler):
    """Test calc_sequence_logprob with entropy metric for partial sequence."""
    output = Output(
        tokens=[1, 2, 3, 4, 5, 6], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4], [-0.6, -1.1, -1.6], [-0.7, -1.2, -1.7]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6, 7, 8, 9]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7, 8, 9, 10]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3, 1.6, 1.8])
    )
    
    # Calculate expected: -mean(entropy[1:4])
    expected = -np.mean(output.entropy[1:4])
    result = calc_sequence_logprob(output, sampler, 1, 4, "entropy")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_likelihood_confidence_full_sequence(sampler):
    """Test calc_sequence_logprob with likelihood_confidence metric for full sequence."""
    output = Output(
        tokens=[1, 2, 3], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7]), 
        entropy=np.array([1.5, 1.2, 1.0])
    )
    
    # Calculate expected: sum(logprobs) - mean(entropy)
    # This is the log of: p(x) * exp(-H(x)) = exp(log(p(x)) - H(x))
    expected = np.sum(output.top_k_logprobs[:, 0]) - np.mean(output.entropy)
    result = calc_sequence_logprob(output, sampler, 0, 3, "likelihood_confidence")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_likelihood_confidence_partial_sequence(sampler):
    """Test calc_sequence_logprob with likelihood_confidence metric for partial sequence."""
    output = Output(
        tokens=[1, 2, 3, 4, 5], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4], [-0.6, -1.1, -1.6]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6, 7, 8]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7, 8, 9]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3, 1.6])
    )
    
    # Calculate expected: sum(logprobs[2:4]) - mean(entropy[2:4])
    expected = np.sum(output.top_k_logprobs[:, 0][2:4]) - np.mean(output.entropy[2:4])
    result = calc_sequence_logprob(output, sampler, 2, 4, "likelihood_confidence")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_single_token_sequence(sampler):
    """Test calc_sequence_logprob with a single token sequence."""
    output = Output(
        tokens=[1, 2, 3], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7]), 
        entropy=np.array([1.5, 1.2, 1.0])
    )
    
    # Test with logprobs for a single token (index 1 to 2)
    expected = output.top_k_logprobs[1, 0]
    result = calc_sequence_logprob(output, sampler, 1, 2, "logprobs")
    
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(expected)

def test_invalid_metric(sampler):
    """Test calc_sequence_logprob with an invalid metric."""
    output = Output(
        tokens=[1, 2, 3], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7]), 
        entropy=np.array([1.5, 1.2, 1.0])
    )
    
    with pytest.raises(ValueError, match="Invalid metric"):
        calc_sequence_logprob(output, sampler, 0, 3, "invalid_metric")

def test_empty_sequence(sampler):
    """Test calc_sequence_logprob with an empty sequence (starting_index == ending_index)."""
    output = Output(
        tokens=[1, 2, 3], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7]), 
        entropy=np.array([1.5, 1.2, 1.0])
    )
    
    # Empty sequence should return sum([]) = 0.0 for logprobs
    result = calc_sequence_logprob(output, sampler, 1, 1, "logprobs")
    assert result == pytest.approx(0.0)
    
    # Empty sequence for entropy should return -mean([]) which is nan
    # Since mean of empty array is nan, we test entropy with a valid range instead
    result = calc_sequence_logprob(output, sampler, 1, 2, "entropy")
    expected = -np.mean(output.entropy[1:2])
    assert result == pytest.approx(expected)

def test_different_temperatures(sampler):
    """Test calc_sequence_logprob with power_distribution at different temperatures."""
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
    expected_05 = np.sum(power_dist_05)
    result_05 = calc_sequence_logprob(output, sampler, 0, 3, "power_distribution")
    assert result_05 == pytest.approx(expected_05)
    
    # Test with temperature = 2.0
    sampler.sampling_params.temperature = 2.0
    power_dist_20 = (1 / 2.0) * (
        output.top_k_logits[:, 0] - np.asarray(output.unprocessed_log_normalization_constant)
    )
    expected_20 = np.sum(power_dist_20)
    result_20 = calc_sequence_logprob(output, sampler, 0, 3, "power_distribution")
    assert result_20 == pytest.approx(expected_20)
    
    # Results should be different for different temperatures
    assert result_05 != pytest.approx(result_20)

def test_logprob_vs_prob_relationship(sampler):
    """Test that calc_sequence_logprob returns the log of calc_sequence_prob."""
    from pita.sampling.token_metrics import calc_sequence_prob
    
    output = Output(
        tokens=[1, 2, 3], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7]), 
        entropy=np.array([1.5, 1.2, 1.0])
    )
    
    # Test for logprobs metric
    logprob_result = calc_sequence_logprob(output, sampler, 0, 3, "logprobs")
    prob_result = calc_sequence_prob(output, sampler, 0, 3, "logprobs")
    assert logprob_result == pytest.approx(np.log(prob_result))
    
    # Test for entropy metric
    logprob_result = calc_sequence_logprob(output, sampler, 0, 3, "entropy")
    prob_result = calc_sequence_prob(output, sampler, 0, 3, "entropy")
    assert logprob_result == pytest.approx(np.log(prob_result))
    
    # Test for likelihood_confidence metric
    logprob_result = calc_sequence_logprob(output, sampler, 0, 3, "likelihood_confidence")
    prob_result = calc_sequence_prob(output, sampler, 0, 3, "likelihood_confidence")
    assert logprob_result == pytest.approx(np.log(prob_result))

def test_calc_sequence_logprob_matches_token_metric_sum(sampler):
    """Test that calc_sequence_logprob returns the same value as summing calc_token_metric for logprobs and power_distribution."""
    from pita.sampling.token_metrics import calc_token_metric
    
    sampler.sampling_params.temperature = 0.7
    output = Output(
        tokens=[1, 2, 3, 4, 5], 
        top_k_logits=np.array([[2.0, 1.5, 1.0], [3.0, 2.5, 2.0], [4.0, 3.5, 3.0], [5.0, 4.5, 4.0], [6.0, 5.5, 5.0]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4], [-0.6, -1.1, -1.6]]), 
        unprocessed_log_normalization_constant=np.array([1.5, 2.5, 3.5, 4.5, 5.5]), 
        temp_processed_log_normalization_constant=np.array([2.0, 3.0, 4.0, 5.0, 6.0]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3, 1.6])
    )
    
    # Test logprobs metric: calc_sequence_logprob should equal sum of calc_token_metric
    token_metrics_logprobs = calc_token_metric(output, sampler, "logprobs")
    sequence_logprob = calc_sequence_logprob(output, sampler, 0, 5, "logprobs")
    expected_logprobs = np.sum(token_metrics_logprobs[0:5])
    assert sequence_logprob == pytest.approx(expected_logprobs)
    
    # Test partial sequence for logprobs
    sequence_logprob_partial = calc_sequence_logprob(output, sampler, 1, 4, "logprobs")
    expected_logprobs_partial = np.sum(token_metrics_logprobs[1:4])
    assert sequence_logprob_partial == pytest.approx(expected_logprobs_partial)
    
    # Test power_distribution metric: calc_sequence_logprob should equal sum of calc_token_metric
    token_metrics_power = calc_token_metric(output, sampler, "power_distribution")
    sequence_logprob_power = calc_sequence_logprob(output, sampler, 0, 5, "power_distribution")
    expected_power = np.sum(token_metrics_power[0:5])
    assert sequence_logprob_power == pytest.approx(expected_power)
    
    # Test partial sequence for power_distribution
    sequence_logprob_power_partial = calc_sequence_logprob(output, sampler, 2, 5, "power_distribution")
    expected_power_partial = np.sum(token_metrics_power[2:5])
    assert sequence_logprob_power_partial == pytest.approx(expected_power_partial)

def test_calc_sequence_logprob_likelihood_confidence_matches_token_metrics(sampler):
    """Test that calc_sequence_logprob for likelihood_confidence equals sum(logprobs) - mean(entropy)."""
    from pita.sampling.token_metrics import calc_token_metric
    
    output = Output(
        tokens=[1, 2, 3, 4], 
        top_k_logits=np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]), 
        top_k_logprobs=np.array([[-0.5, -1.0, -1.5], [-0.3, -0.8, -1.3], [-0.2, -0.7, -1.2], [-0.4, -0.9, -1.4]]), 
        unprocessed_log_normalization_constant=np.array([4, 5, 6, 7]), 
        temp_processed_log_normalization_constant=np.array([5, 6, 7, 8]), 
        entropy=np.array([1.5, 1.2, 1.0, 1.3])
    )
    
    # likelihood_confidence = sum(logprobs) - mean(entropy)
    token_metrics_logprobs = calc_token_metric(output, sampler, "logprobs")
    token_metrics_entropy = calc_token_metric(output, sampler, "entropy")
    
    # Full sequence
    sequence_logprob = calc_sequence_logprob(output, sampler, 0, 4, "likelihood_confidence")
    expected = np.sum(token_metrics_logprobs[0:4]) - np.mean(token_metrics_entropy[0:4])
    assert sequence_logprob == pytest.approx(expected)
    
    # Partial sequence
    sequence_logprob_partial = calc_sequence_logprob(output, sampler, 1, 3, "likelihood_confidence")
    expected_partial = np.sum(token_metrics_logprobs[1:3]) - np.mean(token_metrics_entropy[1:3])
    assert sequence_logprob_partial == pytest.approx(expected_partial)