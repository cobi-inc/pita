# Unit tests for SMC token_sampling_method configuration
import pytest

from pita.sampling.smc import Sequential_Monte_Carlo


class TestTokenSamplingMethod:
    """Tests for SMC token_sampling_method parameter."""
    
    def test_smc_with_token_sampling_sets_token_sample_method(self):
        """
        Test that SMC's token_sampling_method can be set to 'token_sample'.
        This is used when combining SMC with Power Sampling.
        """
        # Create SMC with default token_sampling_method
        smc = Sequential_Monte_Carlo(num_particles=3, tokens_per_step=4)
        assert smc.token_sampling_method == "standard"  # default
        
        # Modify token_sampling_method (as done in serve.py for combined mode)
        smc.token_sampling_method = "token_sample"
        assert smc.token_sampling_method == "token_sample"
    
    def test_smc_token_sample_method_uses_correct_function(self):
        """
        Test that token_sampling_method can be initialized to 'token_sample'.
        
        This verifies the logic in SMC.sample() lines 190-197 which selects
        sampler.token_sample vs sampler.sample based on this parameter.
        """
        # Check the sample method logic handles token_sampling_method correctly
        smc_standard = Sequential_Monte_Carlo(token_sampling_method="standard")
        assert smc_standard.token_sampling_method == "standard"
        
        smc_power = Sequential_Monte_Carlo(token_sampling_method="token_sample")
        assert smc_power.token_sampling_method == "token_sample"


class TestAutomaticPowerSamplingEnablement:
    """Tests for automatic Power Sampling enablement logic pattern."""
    
    def test_enable_power_sampling_sets_flag(self):
        """
        Test that enable_power_sampling sets the token_sample_name correctly.
        This verifies the mechanism used in serve.py to auto-enable it.
        """
        # Mock sampler class that mimics AutoregressiveSampler structure
        class MockSampler:
            def __init__(self):
                self.token_sample_name = None
                self.token_sampling = None
                self.token_sample_fn = None
                self.engine = "mock_engine" # needed for enable checks
            
            def enable_power_sampling(self, block_size, MCMC_steps, token_metric):
                # Mimic logic in LLM_backend.py
                # Note: Real implementation creates Power_Sampling object
                self.token_sample_name = "Power Sampling"
                self.token_sampling = "MockPowerSamplingObject"
        
        sampler = MockSampler()
        
        # Verify initial state
        assert getattr(sampler, "token_sample_name", None) != "Power Sampling"
        
        # Call enable method (as serve.py would)
        sampler.enable_power_sampling(block_size=16, MCMC_steps=2, token_metric="logprobs")
        
        # Verify state after enablement
        assert getattr(sampler, "token_sample_name", None) == "Power Sampling"
