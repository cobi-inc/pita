"""Tests for the Best_of_N.sample method."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pita.sampling.best_of import Best_of_N
from pita.inference.LLM_backend import Output, AutoregressiveSampler, Sampling_Params


class TestBestOfNSample:
    """Tests for the Best_of_N.sample method."""

    @pytest.fixture
    def mock_sampler(self):
        """Create a mock AutoregressiveSampler."""
        sampler = MagicMock(spec=AutoregressiveSampler)
        sampler.tokenizer = MagicMock()
        sampler.tokenizer.eos_token_id = 100
        sampler.tokenizer.decode.return_value = " decoded"
        sampler.sampling_params = MagicMock(spec=Sampling_Params)
        sampler.sampling_params.max_tokens = 10
        sampler.sampling_params.temperature = 1.0
        return sampler

    @pytest.fixture
    def mock_output_factory(self):
        """Factory fixture to create mock Output objects with different tokens."""
        def _create_output(tokens=None, score_influence=0.0):
            if tokens is None:
                tokens = [1, 2, 3]
            return Output(
                tokens=tokens,
                top_k_logits=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]][:len(tokens)]),
                top_k_logprobs=np.array([[-1.0 + score_influence, -0.5], [-0.1, -0.2], [-0.3, -0.4]][:len(tokens)]),
                unprocessed_log_normalization_constant=[0.0] * len(tokens),
                temp_processed_log_normalization_constant=[0.0] * len(tokens),
                entropy=[0.5] * len(tokens)
            )
        return _create_output

    def test_sample_basic_flow(self, mock_sampler, mock_output_factory):
        """Test basic sampling flow with multiple sequences."""
        best_of_n = Best_of_N(sequence_n=3, sequence_top_k=1, token_metric="logprobs")
        
        # Create different outputs for each sequence
        outputs = [
            mock_output_factory([1, 2, 3], score_influence=-1.0),  # Lower score
            mock_output_factory([4, 5, 6], score_influence=0.0),   # Best score
            mock_output_factory([7, 8, 9], score_influence=-0.5),  # Mid score
        ]
        
        # Set up sampler to return outputs sequentially
        mock_sampler.sample.side_effect = outputs
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            # Return different scores for each sequence
            mock_calc.side_effect = [-2.0, -0.5, -1.0]  # Sequence 1 (index 1) is best
            
            result = best_of_n.sample(mock_sampler, "Test prompt")
        
        # Assertions
        assert isinstance(result, Output)
        assert mock_sampler.sample.call_count == 3
        # With greedy selection (top_k=1), should pick the highest score
        assert result.tokens == [4, 5, 6]

    def test_sample_single_sequence(self, mock_sampler, mock_output_factory):
        """Test Best_of_N with only one sequence (sequence_n=1)."""
        best_of_n = Best_of_N(sequence_n=1, sequence_top_k=1, token_metric="logprobs")
        
        output = mock_output_factory([10, 20, 30])
        mock_sampler.sample.return_value = output
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            mock_calc.return_value = -1.0
            
            result = best_of_n.sample(mock_sampler, "Test prompt")
        
        assert mock_sampler.sample.call_count == 1
        assert result.tokens == [10, 20, 30]

    def test_sample_calls_sampler_with_correct_prompt(self, mock_sampler, mock_output_factory):
        """Test that sample() calls sampler.sample with the correct prompt."""
        best_of_n = Best_of_N(sequence_n=2, sequence_top_k=1, token_metric="logprobs")
        
        output = mock_output_factory()
        mock_sampler.sample.return_value = output
        
        test_prompt = "This is a test prompt"
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            mock_calc.return_value = -1.0
            
            best_of_n.sample(mock_sampler, test_prompt)
        
        # All calls should use the same prompt
        for call in mock_sampler.sample.call_args_list:
            assert call[0][0] == test_prompt

    def test_sample_with_different_token_metrics(self, mock_sampler, mock_output_factory):
        """Test sample() works with different token metrics."""
        token_metrics = ["logprobs", "power_distribution", "entropy"]
        
        for metric in token_metrics:
            best_of_n = Best_of_N(sequence_n=2, sequence_top_k=1, token_metric=metric)
            
            outputs = [mock_output_factory([1, 2]), mock_output_factory([3, 4])]
            mock_sampler.sample.side_effect = outputs
            mock_sampler.sample.reset_mock()
            
            with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
                mock_calc.side_effect = [-1.0, -0.5]
                
                result = best_of_n.sample(mock_sampler, "prompt")
                
                # Verify metric is passed correctly
                for call in mock_calc.call_args_list:
                    assert call[0][4] == metric
            
            assert isinstance(result, Output)

    def test_sample_returns_output_object(self, mock_sampler, mock_output_factory):
        """Test that sample() returns a proper Output object."""
        best_of_n = Best_of_N(sequence_n=2, sequence_top_k=1, token_metric="logprobs")
        
        expected_output = mock_output_factory([100, 200, 300])
        other_output = mock_output_factory([1, 2, 3])
        
        mock_sampler.sample.side_effect = [expected_output, other_output]
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            mock_calc.side_effect = [0.0, -5.0]  # First sequence is best
            
            result = best_of_n.sample(mock_sampler, "prompt")
        
        assert isinstance(result, Output)
        assert result.tokens is not None
        assert result.top_k_logits is not None
        assert result.top_k_logprobs is not None

    def test_sample_large_sequence_n(self, mock_sampler, mock_output_factory):
        """Test sampling with a large number of sequences."""
        sequence_n = 20
        best_of_n = Best_of_N(sequence_n=sequence_n, sequence_top_k=1, token_metric="logprobs")
        
        # Create outputs with varying scores
        outputs = [mock_output_factory([i]) for i in range(sequence_n)]
        mock_sampler.sample.side_effect = outputs
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            # Make sequence 10 the best - start from -1 to avoid tie with index 0
            scores = [-float(i + 1) for i in range(sequence_n)]
            scores[10] = 0.0  # Best score
            mock_calc.side_effect = scores
            
            result = best_of_n.sample(mock_sampler, "prompt")
        
        assert mock_sampler.sample.call_count == sequence_n
        assert result.tokens == [10]  # Best sequence

    def test_sample_with_top_k_greater_than_1(self, mock_sampler, mock_output_factory):
        """Test sampling with top_k > 1 (probabilistic selection)."""
        np.random.seed(42)  # For reproducibility
        
        best_of_n = Best_of_N(sequence_n=5, sequence_top_k=3, token_metric="logprobs")
        
        outputs = [mock_output_factory([i]) for i in range(5)]
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            # Run multiple times to verify probabilistic behavior
            selected_tokens = []
            for _ in range(20):
                mock_sampler.sample.side_effect = outputs.copy()
                mock_sampler.sample.reset_mock()
                mock_calc.side_effect = [-1.0, -0.5, -0.1, -1.5, -2.0]  # Top 3: indices 2, 1, 0
                
                result = best_of_n.sample(mock_sampler, "prompt")
                selected_tokens.append(result.tokens[0])
            
            # Should only select from top 3
            for token in selected_tokens:
                assert token in [0, 1, 2]

    def test_sample_passes_correct_indices_to_calc(self, mock_sampler, mock_output_factory):
        """Test that sample() passes correct start/end indices to calc function."""
        best_of_n = Best_of_N(sequence_n=1, sequence_top_k=1, token_metric="logprobs")
        
        # Create output with 5 tokens
        output = Output(
            tokens=[1, 2, 3, 4, 5],
            top_k_logits=np.array([[0.1, 0.2]] * 5),
            top_k_logprobs=np.array([[-1.0, -0.5]] * 5),
            unprocessed_log_normalization_constant=[0.0] * 5,
            temp_processed_log_normalization_constant=[0.0] * 5,
            entropy=[0.5] * 5
        )
        mock_sampler.sample.return_value = output
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            mock_calc.return_value = -1.0
            
            best_of_n.sample(mock_sampler, "prompt")
            
            # Verify indices passed: start=0, end=len(tokens)
            call_args = mock_calc.call_args
            assert call_args[0][2] == 0  # starting_index
            assert call_args[0][3] == 5  # ending_index (len(tokens))

    def test_sample_with_empty_prompt(self, mock_sampler, mock_output_factory):
        """Test sampling with an empty prompt."""
        best_of_n = Best_of_N(sequence_n=2, sequence_top_k=1, token_metric="logprobs")
        
        output = mock_output_factory([1, 2])
        mock_sampler.sample.return_value = output
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            mock_calc.return_value = -1.0
            
            result = best_of_n.sample(mock_sampler, "")
        
        assert isinstance(result, Output)
        mock_sampler.sample.assert_called_with("")

    def test_sample_preserves_output_structure(self, mock_sampler):
        """Test that sample() returns output with all expected fields."""
        best_of_n = Best_of_N(sequence_n=1, sequence_top_k=1, token_metric="logprobs")
        
        # Create a complete output
        expected_output = Output(
            tokens=[1, 2, 3],
            top_k_logits=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            top_k_logprobs=np.array([[-1.0, -0.5, -0.3], [-0.1, -0.2, -0.4], [-0.6, -0.7, -0.8]]),
            unprocessed_log_normalization_constant=[1.0, 2.0, 3.0],
            temp_processed_log_normalization_constant=[0.5, 1.0, 1.5],
            entropy=[0.1, 0.2, 0.3]
        )
        mock_sampler.sample.return_value = expected_output
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            mock_calc.return_value = -1.0
            
            result = best_of_n.sample(mock_sampler, "prompt")
        
        # Verify all fields match
        assert result.tokens == expected_output.tokens
        np.testing.assert_array_equal(result.top_k_logits, expected_output.top_k_logits)
        np.testing.assert_array_equal(result.top_k_logprobs, expected_output.top_k_logprobs)
        assert result.unprocessed_log_normalization_constant == expected_output.unprocessed_log_normalization_constant
        assert result.temp_processed_log_normalization_constant == expected_output.temp_processed_log_normalization_constant
        assert result.entropy == expected_output.entropy


class TestBestOfNSampleEdgeCases:
    """Edge case tests for Best_of_N.sample method."""

    @pytest.fixture
    def mock_sampler(self):
        """Create a mock AutoregressiveSampler."""
        sampler = MagicMock(spec=AutoregressiveSampler)
        sampler.tokenizer = MagicMock()
        sampler.tokenizer.eos_token_id = 100
        sampler.sampling_params = MagicMock(spec=Sampling_Params)
        sampler.sampling_params.temperature = 1.0
        return sampler

    def test_sample_with_equal_scores(self, mock_sampler):
        """Test behavior when all sequences have equal scores."""
        np.random.seed(42)
        
        best_of_n = Best_of_N(sequence_n=3, sequence_top_k=3, token_metric="logprobs")
        
        outputs = [
            Output(tokens=[1], top_k_logits=np.array([[0.1]]), top_k_logprobs=np.array([[-1.0]]),
                   unprocessed_log_normalization_constant=[0.0], temp_processed_log_normalization_constant=[0.0], entropy=[0.5]),
            Output(tokens=[2], top_k_logits=np.array([[0.1]]), top_k_logprobs=np.array([[-1.0]]),
                   unprocessed_log_normalization_constant=[0.0], temp_processed_log_normalization_constant=[0.0], entropy=[0.5]),
            Output(tokens=[3], top_k_logits=np.array([[0.1]]), top_k_logprobs=np.array([[-1.0]]),
                   unprocessed_log_normalization_constant=[0.0], temp_processed_log_normalization_constant=[0.0], entropy=[0.5]),
        ]
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            # Run multiple times with equal scores
            selected_tokens = []
            for _ in range(30):
                mock_sampler.sample.side_effect = outputs.copy()
                mock_sampler.sample.reset_mock()
                mock_calc.side_effect = [0.0, 0.0, 0.0]
                
                result = best_of_n.sample(mock_sampler, "prompt")
                selected_tokens.append(result.tokens[0])
            
            # With equal scores and top_k=3, selection should be uniform
            unique_tokens = set(selected_tokens)
            assert len(unique_tokens) >= 2  # Should see variation

    def test_sample_with_single_token_sequences(self, mock_sampler):
        """Test sampling when sequences contain only one token."""
        best_of_n = Best_of_N(sequence_n=2, sequence_top_k=1, token_metric="logprobs")
        
        outputs = [
            Output(tokens=[42], top_k_logits=np.array([[0.5]]), top_k_logprobs=np.array([[-0.5]]),
                   unprocessed_log_normalization_constant=[0.0], temp_processed_log_normalization_constant=[0.0], entropy=[0.3]),
            Output(tokens=[99], top_k_logits=np.array([[0.1]]), top_k_logprobs=np.array([[-1.5]]),
                   unprocessed_log_normalization_constant=[0.0], temp_processed_log_normalization_constant=[0.0], entropy=[0.8]),
        ]
        mock_sampler.sample.side_effect = outputs
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            mock_calc.side_effect = [-0.5, -1.5]
            
            result = best_of_n.sample(mock_sampler, "prompt")
        
        assert result.tokens == [42]  # First sequence has better score

    def test_sample_with_long_sequences(self, mock_sampler):
        """Test sampling with long sequences (many tokens)."""
        best_of_n = Best_of_N(sequence_n=2, sequence_top_k=1, token_metric="logprobs")
        
        seq_length = 100
        outputs = [
            Output(
                tokens=list(range(seq_length)),
                top_k_logits=np.array([[0.1, 0.2]] * seq_length),
                top_k_logprobs=np.array([[-0.5, -1.0]] * seq_length),
                unprocessed_log_normalization_constant=[0.0] * seq_length,
                temp_processed_log_normalization_constant=[0.0] * seq_length,
                entropy=[0.5] * seq_length
            ),
            Output(
                tokens=list(range(100, 200)),
                top_k_logits=np.array([[0.3, 0.4]] * seq_length),
                top_k_logprobs=np.array([[-0.3, -0.8]] * seq_length),
                unprocessed_log_normalization_constant=[0.0] * seq_length,
                temp_processed_log_normalization_constant=[0.0] * seq_length,
                entropy=[0.6] * seq_length
            ),
        ]
        mock_sampler.sample.side_effect = outputs
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            mock_calc.side_effect = [-50.0, -30.0]  # Second sequence is better
            
            result = best_of_n.sample(mock_sampler, "prompt")
        
        assert result.tokens == list(range(100, 200))
        assert len(result.tokens) == seq_length

    def test_sample_with_variable_length_sequences(self, mock_sampler):
        """Test sampling when sequences have different lengths."""
        best_of_n = Best_of_N(sequence_n=3, sequence_top_k=1, token_metric="logprobs")
        
        outputs = [
            Output(tokens=[1, 2], top_k_logits=np.array([[0.1, 0.2], [0.3, 0.4]]), 
                   top_k_logprobs=np.array([[-1.0, -0.5], [-0.1, -0.2]]),
                   unprocessed_log_normalization_constant=[0.0, 0.0], 
                   temp_processed_log_normalization_constant=[0.0, 0.0], entropy=[0.5, 0.5]),
            Output(tokens=[3, 4, 5, 6], top_k_logits=np.array([[0.1]] * 4), 
                   top_k_logprobs=np.array([[-0.5]] * 4),
                   unprocessed_log_normalization_constant=[0.0] * 4, 
                   temp_processed_log_normalization_constant=[0.0] * 4, entropy=[0.3] * 4),
            Output(tokens=[7], top_k_logits=np.array([[0.2]]), 
                   top_k_logprobs=np.array([[-0.3]]),
                   unprocessed_log_normalization_constant=[0.0], 
                   temp_processed_log_normalization_constant=[0.0], entropy=[0.1]),
        ]
        mock_sampler.sample.side_effect = outputs
        
        with patch("pita.sampling.best_of.calc_sequence_length_normalized_logprob") as mock_calc:
            mock_calc.side_effect = [-1.0, -0.5, -0.2]  # Third sequence is best
            
            result = best_of_n.sample(mock_sampler, "prompt")
        
        assert result.tokens == [7]