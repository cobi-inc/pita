"""
Unit tests for power sampling temperature functionality.
Tests that the power sampling temperature is properly set and used in acceptance calculations.
"""

import sys
import unittest
from unittest.mock import Mock, MagicMock, patch

# Import the classes to test (functions that require torch will be tested differently)
from src.inference.autoregressive_sampler_backend import Power_Sampling_Params


class TestPowerSamplingTemperature(unittest.TestCase):
    """Test suite for power sampling temperature functionality."""

    def test_power_sampling_params_has_temperature(self):
        """Test that Power_Sampling_Params includes power_sampling_temperature."""
        params = Power_Sampling_Params(
            total_output_tokens=1000,
            block_size=50,
            MCMC_steps=5,
            power_sampling_temperature=0.5
        )
        
        self.assertEqual(params.power_sampling_temperature, 0.5)
        self.assertEqual(params.total_output_tokens, 1000)
        self.assertEqual(params.block_size, 50)
        self.assertEqual(params.MCMC_steps, 5)

    def test_power_sampling_params_default_temperature(self):
        """Test that Power_Sampling_Params has a default temperature of 1.0."""
        params = Power_Sampling_Params()
        self.assertEqual(params.power_sampling_temperature, 1.0)

    def test_enable_power_sampling_sets_temperature(self):
        """Test that enable_power_sampling properly sets the power_sampling_temperature."""
        # Import with mocking
        with patch('src.inference.autoregressive_sampler_backend.vllm_backend') as mock_vllm:
            from src.inference.autoregressive_sampler_backend import enable_power_sampling
            
            # Create a mock sampler
            sampler = Mock()
            sampler.engine = "vllm"
            sampler.sampling_params = Mock()
            sampler.sampling_params.top_k = 100
            sampler.sampling_params.temperature = 1.0
            
            # Call enable_power_sampling with a custom temperature
            enable_power_sampling(sampler, 1000, 50, 5, power_sampling_temperature=0.75)
            
            # Verify the temperature was set correctly
            self.assertEqual(sampler.power_sampling_params.power_sampling_temperature, 0.75)
            self.assertEqual(sampler.power_sampling_params.total_output_tokens, 1000)
            self.assertEqual(sampler.power_sampling_params.block_size, 50)
            self.assertEqual(sampler.power_sampling_params.MCMC_steps, 5)

    def test_enable_power_sampling_default_temperature(self):
        """Test that enable_power_sampling uses default temperature of 1.0."""
        # Import with mocking
        with patch('src.inference.autoregressive_sampler_backend.vllm_backend') as mock_vllm:
            from src.inference.autoregressive_sampler_backend import enable_power_sampling
            
            # Create a mock sampler
            sampler = Mock()
            sampler.engine = "vllm"
            sampler.sampling_params = Mock()
            sampler.sampling_params.top_k = 100
            sampler.sampling_params.temperature = 1.0
            
            # Call enable_power_sampling without specifying temperature
            enable_power_sampling(sampler, 1000, 50, 5)
            
            # Verify the default temperature was set
            self.assertEqual(sampler.power_sampling_params.power_sampling_temperature, 1.0)


if __name__ == '__main__':
    unittest.main()
