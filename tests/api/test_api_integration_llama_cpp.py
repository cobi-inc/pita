# Integration tests for the full API system using llama_cpp backend
# These tests verify that test_time_coding works end-to-end with actual LLM sampling

import pytest

# Skip this entire module if llama_cpp is not installed
llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python is required for these tests")

from pita.inference.LLM_backend import AutoregressiveSampler
from pita.api.test_time_coding import encode, decode
from pita.sampling.power_sample import Power_Sampling
from pita.sampling.smc import Sequential_Monte_Carlo
from pita.sampling.best_of import Best_of_N


# Constants - Using a small model for testing
MODEL = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
TOKENIZER_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def sampler():
    """
    Initialize a llama_cpp sampler with TinyLlama for integration testing.
    Scoped to module to avoid repeated model loading.
    """
    sampler = AutoregressiveSampler(
        engine="llama_cpp",
        model=MODEL,
        dtype="Q4_K_M",
        tokenizer_path=TOKENIZER_MODEL,
        gpu_memory_utilization=0.85,
        max_model_len=512,  # Smaller context for faster tests
        max_probs=10,
        logits_processor=True,
        trust_remote_code=True,
        sampling_params=None,
        model_type="gguf"
    )
    # Set reasonable defaults for testing
    sampler.sampling_params.max_tokens = 32
    sampler.sampling_params.temperature = 1.0
    sampler.sampling_params.enable_normalization_constants = True
    
    yield sampler
    
    # Cleanup
    del sampler.llm
    del sampler.tokenizer
    del sampler


def create_prompt(sampler, system_content: str, user_message: str) -> str:
    """Helper to create a formatted prompt using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message}
    ]
    return sampler.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=sampler.sampling_params.enable_thinking
    )


class TestIntegrationStandardSampling:
    """Integration tests for standard (no ITS) sampling."""
    
    def test_standard_sampling_returns_output(self, sampler):
        """Standard sampling should return a valid Output object."""
        prompt = create_prompt(sampler, "You are a helpful assistant.", "Say hello.")
        output = sampler.sample(prompt)
        
        assert output is not None
        assert len(output.tokens) > 0
        assert len(output.top_k_logprobs) == len(output.tokens)


class TestIntegrationPowerSampling:
    """Integration tests for Power Sampling."""
    
    def test_power_sampling_basic(self, sampler):
        """Power Sampling should work end-to-end."""
        ps = Power_Sampling(block_size=16, MCMC_steps=2)
        prompt = create_prompt(sampler, "You are a helpful assistant.", "Say hello.")
        
        # Save and modify max_tokens for this test
        original_max_tokens = sampler.sampling_params.max_tokens
        sampler.sampling_params.max_tokens = 32
        
        try:
            output = ps.sample(sampler, prompt)
            
            assert output is not None
            assert len(output.tokens) > 0
            # Power sampling should fill approximately max_tokens
            # (may be slightly less due to block boundaries)
            assert len(output.tokens) >= 16
        finally:
            sampler.sampling_params.max_tokens = original_max_tokens
    
    def test_power_sampling_encoded_roundtrip(self, sampler):
        """Verify encode/decode works correctly for Power Sampling."""
        ps = Power_Sampling(block_size=16, MCMC_steps=2)
        encoded = encode(chain_sampling=None, token_sampling=ps)
        
        # Verify the encoded string format
        assert encoded == "ITS_NONE_PS_16_2"
        
        # Decode and verify parameters match
        chain, token = decode(encoded)
        assert chain is None
        assert token.block_size == 16
        assert token.MCMC_steps == 2


class TestIntegrationSMC:
    """Integration tests for Sequential Monte Carlo sampling."""
    
    def test_smc_basic(self, sampler):
        """SMC sampling should work end-to-end."""
        smc = Sequential_Monte_Carlo(
            num_particles=3,  # Small for fast testing
            tokens_per_step=4,
            stop_on_eos=True
        )
        prompt = create_prompt(sampler, "You are a helpful assistant.", "Say hello.")
        
        original_max_tokens = sampler.sampling_params.max_tokens
        sampler.sampling_params.max_tokens = 16
        
        try:
            output = smc.sample(sampler, prompt)
            
            assert output is not None
            assert len(output.tokens) > 0
        finally:
            sampler.sampling_params.max_tokens = original_max_tokens
    
    def test_smc_encoded_roundtrip(self, sampler):
        """Verify encode/decode works correctly for SMC."""
        smc = Sequential_Monte_Carlo(num_particles=5, tokens_per_step=3, stop_on_eos=True)
        encoded = encode(chain_sampling=smc, token_sampling=None)
        
        assert encoded == "ITS_SMC_5_3_1_NONE"
        
        chain, token = decode(encoded)
        assert chain.num_particles == 5
        assert chain.tokens_per_step == 3
        assert chain.stop_on_eos == True
        assert token is None


class TestIntegrationBestOfN:
    """Integration tests for Best-of-N sampling."""
    
    def test_best_of_n_basic(self, sampler):
        """Best-of-N sampling should work end-to-end."""
        bon = Best_of_N(
            sequence_n=3,  # Small for fast testing
            sequence_top_k=1
        )
        prompt = create_prompt(sampler, "You are a helpful assistant.", "Say hello.")
        
        original_max_tokens = sampler.sampling_params.max_tokens
        sampler.sampling_params.max_tokens = 16
        
        try:
            output = bon.sample(sampler, prompt)
            
            assert output is not None
            assert len(output.tokens) > 0
        finally:
            sampler.sampling_params.max_tokens = original_max_tokens
    
    def test_best_of_n_encoded_roundtrip(self, sampler):
        """Verify encode/decode works correctly for Best-of-N."""
        bon = Best_of_N(sequence_n=5, sequence_top_k=2)
        encoded = encode(chain_sampling=bon, token_sampling=None)
        
        assert encoded == "ITS_BO_5_2_NONE"
        
        chain, token = decode(encoded)
        assert chain.sequence_n == 5
        assert chain.sequence_top_k == 2
        assert token is None


class TestIntegrationFullAPI:
    """Integration tests simulating the full API flow."""
    
    def test_full_api_flow_with_its_prefix(self, sampler):
        """Test the full flow: system prompt with ITS prefix -> decode -> sample."""
        # Simulate system prompt with ITS encoding
        system_content = "ITS_SMC_3_4_1_NONE You are a helpful AI assistant."
        
        # Decode the ITS parameters
        chain, token = decode(system_content)
        
        # Verify decoding
        assert isinstance(chain, Sequential_Monte_Carlo)
        assert chain.num_particles == 3
        assert chain.tokens_per_step == 4
        assert token is None
        
        # Extract the actual system message (what would remain after ITS removal)
        actual_system = " ".join(system_content.split(" ")[1:])
        assert actual_system == "You are a helpful AI assistant."
        
        # Create prompt with the cleaned system message
        prompt = create_prompt(sampler, actual_system, "Hi!")
        
        # Sample using the decoded chain sampling
        original_max_tokens = sampler.sampling_params.max_tokens
        sampler.sampling_params.max_tokens = 16
        
        try:
            output = chain.sample(sampler, prompt)
            assert output is not None
            assert len(output.tokens) > 0
            
            # Decode the output
            text = sampler.tokenizer.decode(output.tokens, skip_special_tokens=True)
            assert len(text) > 0
        finally:
            sampler.sampling_params.max_tokens = original_max_tokens
    
    def test_combined_chain_and_token_sampling_encoding(self, sampler):
        """Test encoding/decoding combined chain + token sampling."""
        smc = Sequential_Monte_Carlo(num_particles=3, tokens_per_step=4, stop_on_eos=True)
        ps = Power_Sampling(block_size=16, MCMC_steps=2)
        
        encoded = encode(chain_sampling=smc, token_sampling=ps)
        assert encoded == "ITS_SMC_3_4_1_PS_16_2"
        
        chain, token = decode(encoded)
        assert isinstance(chain, Sequential_Monte_Carlo)
        assert isinstance(token, Power_Sampling)
        assert chain.num_particles == 3
        assert token.block_size == 16
