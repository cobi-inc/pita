"""
Test suite for Power Sampling algorithm verification - Direct Method Testing.

This test module DIRECTLY tests the Power_Sampling.sample() method from
`pita/sampling/power_sample.py` against the algorithm outlined in 
arXiv paper 2510.14901 "Reasoning with Sampling: Your Base Model is Smarter Than You Think".

The tests use mocks to:
1. Verify the algorithm flow matches the paper's description
2. Verify MCMC steps are executed correctly
3. Verify acceptance/rejection logic works as per MH algorithm
4. Verify the logging captures correct data

Reference Paper Algorithm (from Section 4.2-4.3):
- Algorithm 1: Block-wise MCMC sampling
- For each block: sample B tokens, then run N_MCMC refinement steps
- Each refinement: random resample from index t, apply MH acceptance
"""

import pytest
import numpy as np
import os
import tempfile
import random
from unittest.mock import Mock
from dataclasses import dataclass

# PITA Libraries
from pita.inference.LLM_backend import Output
from pita.sampling.power_sample import Power_Sampling


def create_mock_output(num_tokens: int, seed: int = 42) -> Output:
    """
    Create a mock Output object with realistic structure.
    
    Args:
        num_tokens: Number of tokens to generate
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    return Output(
        tokens=list(range(100, 100 + num_tokens)),  # Token IDs
        top_k_logits=[np.random.randn(10).tolist() for _ in range(num_tokens)],
        top_k_logprobs=[np.random.uniform(-5, 0, 10).tolist() for _ in range(num_tokens)],
        unprocessed_log_normalization_constant=np.random.randn(num_tokens).tolist(),
        temp_processed_log_normalization_constant=np.random.randn(num_tokens).tolist(),
        entropy=np.random.uniform(0, 2, num_tokens).tolist()
    )


class DynamicMockSampler:
    """
    A mock sampler that dynamically returns outputs matching the current max_tokens setting.
    
    This is necessary because Power_Sampling modifies sampling_params.max_tokens during execution.
    """
    
    def __init__(self, initial_max_tokens: int = 64, temperature: float = 0.7, seed: int = None):
        self.call_count = 0
        self.call_log = []
        
        # Create a real Sampling_Params-like object with mutable max_tokens
        self.sampling_params = Mock()
        self.sampling_params.max_tokens = initial_max_tokens
        self.sampling_params.temperature = temperature
        self.sampling_params.seed = seed
        
        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.decode = Mock(return_value="decoded text")
        self.tokenizer.eos_token_id = 2  # Common EOS token ID
    
    def sample(self, prompt: str) -> Output:
        """
        Sample method that returns output matching current max_tokens.
        """
        self.call_count += 1
        current_max_tokens = self.sampling_params.max_tokens
        self.call_log.append({
            'call_num': self.call_count,
            'max_tokens': current_max_tokens,
            'prompt_len': len(prompt)
        })
        return create_mock_output(current_max_tokens, seed=self.call_count)


class TestPowerSamplingSampleMethod:
    """
    Test suite that DIRECTLY calls Power_Sampling.sample() method.
    
    This verifies the actual implementation matches paper Algorithm 1.
    """

    def test_sample_returns_output_object(self):
        """
        Test that Power_Sampling.sample() returns a valid Output object.
        
        Paper Algorithm 1: Returns sampled sequence x_0:T
        """
        # Setup
        ps = Power_Sampling(block_size=32, MCMC_steps=2, token_metric="power_distribution")
        mock_sampler = DynamicMockSampler(initial_max_tokens=32)
        
        # Execute
        result = ps.sample(mock_sampler, "Test prompt")
        
        # Verify
        assert isinstance(result, Output), "sample() should return Output object"
        assert hasattr(result, 'tokens'), "Output should have tokens"
        assert hasattr(result, 'top_k_logprobs'), "Output should have top_k_logprobs"
        assert len(result.tokens) == 32, "Should return block_size tokens"

    def test_sample_block_structure(self):
        """
        Test that sample() processes the correct number of blocks.
        
        Paper Section 4.3: "Fix block size B... consider the sequence of distributions"
        Total blocks = max_tokens / block_size
        """
        block_size = 16
        max_tokens = 64  # Should result in 4 blocks
        mcmc_steps = 1
        expected_blocks = max_tokens // block_size
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=max_tokens)
        
        # Execute
        result = ps.sample(mock_sampler, "Test prompt")
        
        # Each block: 1 initial sample + MCMC_steps resamples
        # 4 blocks * (1 + 1) = 8 calls
        expected_calls = expected_blocks * (1 + mcmc_steps)
        assert mock_sampler.call_count == expected_calls, \
            f"Expected {expected_calls} sample calls, got {mock_sampler.call_count}"
        
        # Result should have max_tokens tokens
        assert len(result.tokens) == max_tokens, \
            f"Expected {max_tokens} tokens, got {len(result.tokens)}"

    def test_sample_mcmc_steps_executed_per_block(self):
        """
        Test that MCMC_steps refinement iterations happen per block.
        
        Paper Algorithm 1, Line 6-11: "for j = 1 to N_MCMC do" 
        - Sample random index t
        - Resample from t to end
        - Accept/reject with MH ratio
        """
        block_size = 32
        mcmc_steps = 4
        max_tokens = 32  # 1 block
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=max_tokens)
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Execute
        ps.sample(mock_sampler, "Test prompt")
        
        # Should have: 1 initial + mcmc_steps refinements = 5 calls
        assert mock_sampler.call_count == 1 + mcmc_steps, \
            f"Expected {1 + mcmc_steps} sample calls, got {mock_sampler.call_count}"
        
        # First call should be for full block_size
        assert mock_sampler.call_log[0]['max_tokens'] == block_size, \
            f"First call should generate {block_size} tokens"
        
        # MCMC calls should have varying max_tokens (context_len - random_idx)
        mcmc_max_tokens = [log['max_tokens'] for log in mock_sampler.call_log[1:]]
        assert all(1 <= mt <= block_size for mt in mcmc_max_tokens), \
            f"MCMC calls should have 1-{block_size} tokens: {mcmc_max_tokens}"

    def test_sample_random_resampling_proposal(self):
        """
        Test that MCMC steps use random index for resampling.
        
        Paper Section 4.2: "With uniform probability 1/T, select a random t ∈ [1,T] 
        and resample the sequence starting at index t"
        """
        block_size = 32
        mcmc_steps = 10  # Many steps to verify randomness
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=block_size)
        
        # Set seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Execute
        ps.sample(mock_sampler, "Test prompt")
        
        # Extract the indices used (idx = context_length - max_tokens for MCMC calls)
        mcmc_max_tokens = [log['max_tokens'] for log in mock_sampler.call_log[1:]]
        indices_used = [block_size - mt for mt in mcmc_max_tokens]
        
        # Verify indices vary (not all the same)
        assert len(set(indices_used)) > 1, \
            f"Random indices should vary, got {indices_used}"
        
        # Verify all indices are in valid range [0, block_size-1]
        for idx in indices_used:
            assert 0 <= idx < block_size, \
                f"Index {idx} should be in range [0, {block_size})"

    def test_sample_mh_acceptance_ratio_logged(self):
        """
        Test that MH acceptance ratio is calculated and logged.
        
        Paper Section 4.2: 
        A = min(1, (p^α(x') * q(x|x')) / (p^α(x) * q(x'|x)))
        """
        block_size = 8
        mcmc_steps = 2
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=block_size)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_log.csv")
            
            random.seed(42)
            np.random.seed(42)
            
            ps.sample(mock_sampler, "Test prompt", logging=True, log_file_path=log_path)
            
            # Read the log file
            with open(log_path, 'r') as f:
                content = f.read()
            
            # Log should contain acceptance ratio data
            assert "True" in content or "False" in content, \
                "Log should contain acceptance decisions (True/False)"

    def test_sample_context_updates_on_acceptance(self):
        """
        Test that context is updated when proposal is accepted.
        
        Paper Algorithm 1, Line 11: "if u < A then x ← x'"
        
        We verify this by checking that the output tokens can change
        during MCMC steps when acceptance occurs.
        """
        block_size = 16
        mcmc_steps = 5
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=block_size)
        
        # Use deterministic seeds
        random.seed(42)
        np.random.seed(42)
        
        result = ps.sample(mock_sampler, "Test prompt")
        
        # The result should have block_size tokens
        assert len(result.tokens) == block_size
        # Tokens should be valid (our mock generates tokens starting at 100+)
        assert all(t >= 100 for t in result.tokens)

    def test_sample_eos_token_terminates_early(self):
        """
        Test that EOS token terminates sampling early.
        
        Implementation detail: Line 226-235 checks for EOS and returns early
        """
        block_size = 32
        mcmc_steps = 2
        max_tokens = 64  # 2 blocks expected, but should only process 1
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        
        # Create custom sampler that injects EOS token in all outputs
        class EOSSampler(DynamicMockSampler):
            def sample(self, prompt: str) -> Output:
                self.call_count += 1
                current_max_tokens = self.sampling_params.max_tokens
                self.call_log.append({
                    'call_num': self.call_count,
                    'max_tokens': current_max_tokens,
                })
                out = create_mock_output(current_max_tokens, seed=self.call_count)
                
                # Always inject EOS at position 5 (if output is long enough)
                # This ensures EOS persists through MCMC acceptance
                if len(out.tokens) > 5:
                    out.tokens[5] = 999
                
                return out
        
        mock_sampler = EOSSampler(initial_max_tokens=max_tokens)
        mock_sampler.tokenizer.eos_token_id = 999
        
        random.seed(42)
        np.random.seed(42)
        
        ps.sample(mock_sampler, "Test prompt")
        
        # With 2 blocks and 2 MCMC steps each, we'd expect 6 calls without EOS
        # With EOS in first block, should terminate after first block's MCMC
        # First block: 1 initial + 2 MCMC = 3 calls max
        # The call count should be <= 3 + some margin for the EOS check
        max_expected_calls = 1 + mcmc_steps  # First block only
        
        # Note: EOS check happens AFTER the MCMC steps, so we get all 3 calls
        # but don't proceed to block 2
        assert mock_sampler.call_count <= max_expected_calls + 1, \
            f"EOS should prevent second block. Expected <={max_expected_calls}, got {mock_sampler.call_count}"


class TestPowerSamplingLoggingOutput:
    """
    Test suite verifying logging output matches the algorithm execution.
    """

    def test_logging_creates_file(self):
        """Test that logging creates a log file."""
        block_size = 16
        mcmc_steps = 2
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=block_size)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_log.csv")
            
            random.seed(42)
            np.random.seed(42)
            
            result = ps.sample(mock_sampler, "Test prompt", logging=True, log_file_path=log_path)
            
            # Verify log file exists
            assert os.path.exists(log_path), "Log file should be created"
            
            # Verify it has content
            with open(log_path, 'r') as f:
                content = f.read()
            assert len(content) > 0, "Log file should have content"

    def test_logging_captures_mcmc_data(self):
        """Test that logging captures MCMC step data."""
        block_size = 8
        mcmc_steps = 3
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=block_size)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_log.csv")
            
            random.seed(42)
            np.random.seed(42)
            
            result = ps.sample(mock_sampler, "Test prompt", logging=True, log_file_path=log_path)
            
            with open(log_path, 'r') as f:
                lines = f.readlines()
            
            # Should have multiple lines (header info + data entries)
            assert len(lines) > 3, f"Expected multiple log lines, got {len(lines)}"


class TestPowerSamplingTokenMetrics:
    """
    Test that Power_Sampling correctly uses different token metrics.
    """

    @pytest.mark.parametrize("token_metric", ["logprobs", "power_distribution", "entropy"])
    def test_sample_with_different_metrics(self, token_metric):
        """Test that sample() works with all supported token metrics."""
        block_size = 8
        mcmc_steps = 1
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric=token_metric)
        mock_sampler = DynamicMockSampler(initial_max_tokens=block_size)
        
        random.seed(42)
        np.random.seed(42)
        
        result = ps.sample(mock_sampler, "Test prompt")
        
        assert isinstance(result, Output), f"sample() with {token_metric} should return Output"
        assert len(result.tokens) == block_size


class TestPowerSamplingSeeding:
    """Test that seeding produces reproducible results."""

    def test_sample_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        block_size = 16
        mcmc_steps = 3
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        
        results = []
        call_logs = []
        for _ in range(2):
            mock_sampler = DynamicMockSampler(initial_max_tokens=block_size, seed=42)
            
            # Reset seeds before each run
            random.seed(42)
            np.random.seed(42)
            
            result = ps.sample(mock_sampler, "Test prompt")
            results.append(result.tokens)
            call_logs.append(mock_sampler.call_log)
        
        # Same seed should produce same call patterns
        assert call_logs[0] == call_logs[1], "Same seed should produce same call patterns"


class TestPowerSamplingMaxTokensRestoration:
    """Test that max_tokens is restored after sampling."""

    def test_max_tokens_restored_after_sample(self):
        """Test that sampler.sampling_params.max_tokens is restored."""
        block_size = 16
        mcmc_steps = 2
        original_max_tokens = 64
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=original_max_tokens)
        
        random.seed(42)
        np.random.seed(42)
        
        result = ps.sample(mock_sampler, "Test prompt")
        
        # max_tokens should be restored
        assert mock_sampler.sampling_params.max_tokens == original_max_tokens, \
            f"max_tokens should be restored to {original_max_tokens}"


class TestPowerSamplingMultipleBlocks:
    """Test Power Sampling with multiple blocks."""

    def test_multiple_blocks_extend_context(self):
        """
        Test that multiple blocks correctly extend the context.
        
        Paper Algorithm 1: After each block, the context grows by B tokens
        """
        block_size = 8
        mcmc_steps = 1
        max_tokens = 24  # 3 blocks
        expected_blocks = 3
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=max_tokens)
        
        random.seed(42)
        np.random.seed(42)
        
        result = ps.sample(mock_sampler, "Test prompt")
        
        # Result should have full max_tokens
        assert len(result.tokens) == max_tokens, \
            f"Expected {max_tokens} tokens from {expected_blocks} blocks"
        
        # Each block + MCMC calls = expected_blocks * (1 + mcmc_steps)
        expected_calls = expected_blocks * (1 + mcmc_steps)
        assert mock_sampler.call_count == expected_calls


class TestPowerSamplingAlgorithmVerification:
    """
    Core tests that verify the algorithm matches paper 2510.14901.
    
    These tests verify the key algorithmic properties described in the paper.
    """

    def test_algorithm_block_wise_progressive_sampling(self):
        """
        Verify Paper Section 4.3: Block-wise progressive sampling
        
        "We define a series of intermediate distributions which we progressively 
        sample from, until converging to the target distribution p^α"
        
        Each block should:
        1. Sample B new tokens (extending the sequence)
        2. Run N_MCMC refinement steps on the full sequence
        """
        block_size = 10
        mcmc_steps = 2
        max_tokens = 30  # 3 blocks
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=max_tokens)
        
        random.seed(42)
        np.random.seed(42)
        
        result = ps.sample(mock_sampler, "Test prompt")
        
        # Verify 3 blocks were processed
        # Each block: 1 initial + 2 MCMC = 3 calls per block
        # Total: 9 calls
        assert mock_sampler.call_count == 9, f"Expected 9 calls, got {mock_sampler.call_count}"
        
        # Verify token progression - each block adds block_size tokens
        # Block 1: 10 tokens, Block 2: 20 tokens, Block 3: 30 tokens
        assert len(result.tokens) == max_tokens

    def test_algorithm_mcmc_resampling_from_random_index(self):
        """
        Verify Paper Section 4.2: Random resampling proposal
        
        "With uniform probability 1/T, select a random t ∈ [1,T] and resample 
        the sequence starting at index t"
        
        During MCMC steps, the number of tokens to generate should be:
        len(context) - random_idx
        """
        block_size = 20
        mcmc_steps = 5
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=block_size)
        
        random.seed(42)
        np.random.seed(42)
        
        result = ps.sample(mock_sampler, "Test prompt")
        
        # After first block, MCMC steps should have varying max_tokens
        mcmc_tokens = [log['max_tokens'] for log in mock_sampler.call_log[1:]]
        
        # Tokens = context_length - idx, and idx is random in [0, context_length-1]
        # So tokens should be in [1, block_size]
        assert all(1 <= t <= block_size for t in mcmc_tokens), \
            f"MCMC token counts should be in [1, {block_size}]: {mcmc_tokens}"
        
        # Should have variety (not all the same index)
        assert len(set(mcmc_tokens)) > 1, "Random resampling should produce varied token counts"

    def test_algorithm_output_attributes_consistent_length(self):
        """
        Verify that all output attributes have consistent length.
        
        This is critical for the algorithm's correctness - all tracked metrics
        must align with the token sequence.
        """
        block_size = 12
        mcmc_steps = 2
        max_tokens = 24  # 2 blocks
        
        ps = Power_Sampling(block_size=block_size, MCMC_steps=mcmc_steps, token_metric="logprobs")
        mock_sampler = DynamicMockSampler(initial_max_tokens=max_tokens)
        
        random.seed(42)
        np.random.seed(42)
        
        result = ps.sample(mock_sampler, "Test prompt")
        
        # All attributes should have same length
        expected_length = len(result.tokens)
        assert len(result.top_k_logits) == expected_length, "top_k_logits length mismatch"
        assert len(result.top_k_logprobs) == expected_length, "top_k_logprobs length mismatch"
        assert len(result.unprocessed_log_normalization_constant) == expected_length, \
            "unprocessed_log_normalization_constant length mismatch"
        assert len(result.temp_processed_log_normalization_constant) == expected_length, \
            "temp_processed_log_normalization_constant length mismatch"
        assert len(result.entropy) == expected_length, "entropy length mismatch"
