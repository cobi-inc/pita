import pytest
import numpy as np
from pita.sampling.best_of import Best_of_N


class TestSequenceScoring:
    """Tests for the Best_of_N.sequence_scoring method."""

    @pytest.mark.parametrize("metric", ["logprobs", "power_distribution"])
    def test_sequence_scoring_logprobs_power_distribution(self, metric):
        """Test sequence scoring for logprobs and power_distribution metrics (higher is better)."""
        best_of_n = Best_of_N(sequence_n=5, sequence_top_k=1, token_metric=metric)
        
        # Initialize sequence scores array
        sequence_scores = np.zeros(5)
        token_metrics = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])
        
        # Score sequence at index 2
        best_of_n.sequence_scoring(sequence_scores, 2, token_metrics)
        
        # Expected: average of token metrics = (-0.1 + -0.2 + -0.3 + -0.4 + -0.5) / 5 = -0.3
        expected_score = np.sum(token_metrics) / len(token_metrics)
        assert sequence_scores[2] == pytest.approx(expected_score)
        assert sequence_scores[2] == pytest.approx(-0.3)
        
        # Check other indices unchanged
        assert sequence_scores[0] == 0.0
        assert sequence_scores[1] == 0.0
        assert sequence_scores[3] == 0.0
        assert sequence_scores[4] == 0.0

    def test_sequence_scoring_entropy(self):
        """Test sequence scoring for entropy metric (lower is better, so negate)."""
        best_of_n = Best_of_N(sequence_n=5, sequence_top_k=1, token_metric="entropy")
        
        # Initialize sequence scores array
        sequence_scores = np.zeros(5)
        token_metrics = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Score sequence at index 1
        best_of_n.sequence_scoring(sequence_scores, 1, token_metrics)
        
        # Expected: -sum / len = -(1+2+3+4+5) / 5 = -3.0
        expected_score = -np.sum(token_metrics) / len(token_metrics)
        assert sequence_scores[1] == pytest.approx(expected_score)
        assert sequence_scores[1] == pytest.approx(-3.0)

    def test_sequence_scoring_multiple_sequences(self):
        """Test scoring multiple sequences sequentially."""
        best_of_n = Best_of_N(sequence_n=3, sequence_top_k=1, token_metric="logprobs")
        
        sequence_scores = np.zeros(3)
        
        # Score first sequence with high values
        token_metrics_0 = np.array([-0.1, -0.1, -0.1])  # avg = -0.1
        best_of_n.sequence_scoring(sequence_scores, 0, token_metrics_0)
        
        # Score second sequence with medium values
        token_metrics_1 = np.array([-0.5, -0.5, -0.5])  # avg = -0.5
        best_of_n.sequence_scoring(sequence_scores, 1, token_metrics_1)
        
        # Score third sequence with low values
        token_metrics_2 = np.array([-1.0, -1.0, -1.0])  # avg = -1.0
        best_of_n.sequence_scoring(sequence_scores, 2, token_metrics_2)
        
        # Check all scores
        assert sequence_scores[0] == pytest.approx(-0.1)
        assert sequence_scores[1] == pytest.approx(-0.5)
        assert sequence_scores[2] == pytest.approx(-1.0)
        
        # Best sequence for logprobs is the highest score (index 0)
        assert np.argmax(sequence_scores) == 0

    def test_sequence_scoring_entropy_multiple_sequences(self):
        """Test scoring multiple sequences with entropy metric."""
        best_of_n = Best_of_N(sequence_n=3, sequence_top_k=1, token_metric="entropy")
        
        sequence_scores = np.zeros(3)
        
        # Score first sequence with low entropy (good)
        token_metrics_0 = np.array([0.5, 0.5, 0.5])  # avg = 0.5, negated = -0.5
        best_of_n.sequence_scoring(sequence_scores, 0, token_metrics_0)
        
        # Score second sequence with medium entropy
        token_metrics_1 = np.array([1.0, 1.0, 1.0])  # avg = 1.0, negated = -1.0
        best_of_n.sequence_scoring(sequence_scores, 1, token_metrics_1)
        
        # Score third sequence with high entropy (bad)
        token_metrics_2 = np.array([2.0, 2.0, 2.0])  # avg = 2.0, negated = -2.0
        best_of_n.sequence_scoring(sequence_scores, 2, token_metrics_2)
        
        # Check all scores
        assert sequence_scores[0] == pytest.approx(-0.5)
        assert sequence_scores[1] == pytest.approx(-1.0)
        assert sequence_scores[2] == pytest.approx(-2.0)
        
        # Best sequence for entropy is the HIGHEST negated score (index 0, which had lowest entropy)
        assert np.argmax(sequence_scores) == 0

    def test_sequence_scoring_single_token(self):
        """Test scoring with a single token metric."""
        best_of_n = Best_of_N(sequence_n=2, sequence_top_k=1, token_metric="logprobs")
        
        sequence_scores = np.zeros(2)
        token_metrics = np.array([-0.693])  # Single token
        
        best_of_n.sequence_scoring(sequence_scores, 0, token_metrics)
        
        assert sequence_scores[0] == pytest.approx(-0.693)

    def test_sequence_scoring_overwrite_previous(self):
        """Test that scoring overwrites previous value at same index."""
        best_of_n = Best_of_N(sequence_n=2, sequence_top_k=1, token_metric="logprobs")
        
        sequence_scores = np.array([0.5, 0.5])  # Pre-initialized with non-zero values
        token_metrics = np.array([-0.3, -0.3, -0.3])
        
        # Score index 0
        best_of_n.sequence_scoring(sequence_scores, 0, token_metrics)
        
        # Should overwrite the previous value
        assert sequence_scores[0] == pytest.approx(-0.3)
        assert sequence_scores[1] == pytest.approx(0.5)  # Unchanged