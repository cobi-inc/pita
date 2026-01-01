import pytest
import numpy as np
from pita.sampling.best_of import Best_of_N


class TestSelectSequence:
    """Tests for the Best_of_N.select_sequence method."""

    def test_select_sequence_greedy(self):
        """Test greedy selection when sequence_top_k = 1."""
        best_of_n = Best_of_N(sequence_n=5, sequence_top_k=1, token_metric="logprobs")
        
        # Create scores where one is clearly the best
        sequence_scores = np.array([-1.0, -0.5, -0.1, -0.8, -0.9])
        
        # With top_k=1, should always return the best index (highest score = index 2)
        # Run multiple times to verify deterministic behavior in the limit
        selected_indices = [best_of_n.select_sequence(sequence_scores) for _ in range(10)]
        
        # All selections should be index 2 (the highest score)
        assert all(idx == 2 for idx in selected_indices)

    def test_select_sequence_top_k_all(self):
        """Test selection when sequence_top_k = sequence_n (all sequences considered)."""
        best_of_n = Best_of_N(sequence_n=3, sequence_top_k=3, token_metric="logprobs")
        
        # Create equal scores
        sequence_scores = np.array([0.0, 0.0, 0.0])
        
        # With equal scores, selection should be uniform over all indices
        selected_indices = [best_of_n.select_sequence(sequence_scores) for _ in range(100)]
        
        # All indices should appear at least once with high probability
        unique_selected = set(selected_indices)
        assert len(unique_selected) >= 2  # At minimum 2 different indices should be selected

    def test_select_sequence_top_k_subset(self):
        """Test selection with sequence_top_k < sequence_n."""
        best_of_n = Best_of_N(sequence_n=5, sequence_top_k=2, token_metric="logprobs")
        
        # Create scores where top 2 are clearly identifiable
        sequence_scores = np.array([-1.0, -0.1, -0.2, -1.5, -1.8])
        # Top 2 are indices 1 (-0.1) and 2 (-0.2)
        
        # Sample multiple times
        selected_indices = [best_of_n.select_sequence(sequence_scores) for _ in range(100)]
        
        # All selections should be from the top_k indices (1 or 2)
        for idx in selected_indices:
            assert idx in [1, 2], f"Selected index {idx} should be in top_k [1, 2]"

    def test_select_sequence_extreme_score_difference(self):
        """Test that extreme score differences result in nearly deterministic selection."""
        best_of_n = Best_of_N(sequence_n=5, sequence_top_k=2, token_metric="logprobs")
        
        # One score is much higher than others
        sequence_scores = np.array([-100.0, 10.0, -50.0, -80.0, -90.0])
        
        # The highest score (index 1) should dominate selection
        selected_indices = [best_of_n.select_sequence(sequence_scores) for _ in range(50)]
        
        # Should always select index 1 (the dominant score)
        assert all(idx == 1 for idx in selected_indices)

    def test_select_sequence_returns_valid_index(self):
        """Test that returned index is always valid."""
        best_of_n = Best_of_N(sequence_n=10, sequence_top_k=3, token_metric="logprobs")
        
        sequence_scores = np.random.randn(10)
        
        for _ in range(50):
            idx = best_of_n.select_sequence(sequence_scores)
            assert isinstance(idx, (int, np.integer))
            assert 0 <= idx < 10

    def test_select_sequence_two_sequences(self):
        """Test selection with minimum number of sequences (2)."""
        best_of_n = Best_of_N(sequence_n=2, sequence_top_k=1, token_metric="logprobs")
        
        # First sequence is better
        sequence_scores = np.array([0.0, -1.0])
        
        selected_indices = [best_of_n.select_sequence(sequence_scores) for _ in range(10)]
        assert all(idx == 0 for idx in selected_indices)
        
        # Second sequence is better
        sequence_scores = np.array([-1.0, 0.0])
        
        selected_indices = [best_of_n.select_sequence(sequence_scores) for _ in range(10)]
        assert all(idx == 1 for idx in selected_indices)

    def test_select_sequence_probabilistic_selection(self):
        """Test that selection is probabilistic when scores are similar."""
        np.random.seed(42)  # For reproducibility
        
        best_of_n = Best_of_N(sequence_n=3, sequence_top_k=3, token_metric="logprobs")
        
        # Similar scores should lead to varied selections
        sequence_scores = np.array([0.0, 0.1, 0.2])
        
        selected_indices = [best_of_n.select_sequence(sequence_scores) for _ in range(300)]
        
        # Count selections
        counts = {0: 0, 1: 0, 2: 0}
        for idx in selected_indices:
            counts[idx] += 1
        
        # All indices should be selected at least some times
        # Higher score (index 2) should be selected more often
        assert counts[0] > 0
        assert counts[1] > 0
        assert counts[2] > 0
        # Index 2 should be selected more than index 0 on average
        assert counts[2] > counts[0]

    def test_select_sequence_large_negative_scores(self):
        """Test handling of large negative scores (common for log probabilities)."""
        best_of_n = Best_of_N(sequence_n=3, sequence_top_k=1, token_metric="logprobs")
        
        # Typical log probability values
        sequence_scores = np.array([-50.0, -30.0, -45.0])
        
        # Best is index 1 (highest/least negative)
        idx = best_of_n.select_sequence(sequence_scores)
        assert idx == 1

    def test_select_sequence_positive_scores(self):
        """Test handling of positive scores."""
        best_of_n = Best_of_N(sequence_n=4, sequence_top_k=2, token_metric="logprobs")
        
        sequence_scores = np.array([1.0, 5.0, 3.0, 2.0])
        
        # Top 2 should be indices 1 (5.0) and 2 (3.0)
        selected_indices = [best_of_n.select_sequence(sequence_scores) for _ in range(50)]
        
        for idx in selected_indices:
            assert idx in [1, 2]