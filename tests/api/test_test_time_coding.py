# Tests for the test_time_coding encode and decode functions
import pytest
from pita.api.test_time_coding import encode, decode
from pita.sampling.power_sample import Power_Sampling
from pita.sampling.smc import Sequential_Monte_Carlo



class TestEncode:
    """Tests for the encode function."""

    def test_encode_none_returns_empty_string(self):
        """Encoding with no parameters returns empty string."""
        result = encode(chain_sampling=None, token_sampling=None)
        assert result == ""

    def test_encode_smc_only(self):
        """Encoding SMC without token sampling."""
        smc = Sequential_Monte_Carlo(num_particles=10, tokens_per_step=5, stop_on_eos=True)
        result = encode(chain_sampling=smc, token_sampling=None)
        assert result == "ITS_SMC_10_5_1_NONE"

    def test_encode_smc_stop_on_eos_false(self):
        """Encoding SMC with stop_on_eos=False."""
        smc = Sequential_Monte_Carlo(num_particles=8, tokens_per_step=3, stop_on_eos=False)
        result = encode(chain_sampling=smc, token_sampling=None)
        assert result == "ITS_SMC_8_3_0_NONE"



    def test_encode_power_sampling_only(self):
        """Encoding Power Sampling without chain sampling."""
        ps = Power_Sampling(block_size=192, MCMC_steps=8)
        result = encode(chain_sampling=None, token_sampling=ps)
        assert result == "ITS_NONE_PS_192_8"

    def test_encode_smc_with_power_sampling(self):
        """Encoding SMC combined with Power Sampling."""
        smc = Sequential_Monte_Carlo(num_particles=10, tokens_per_step=5, stop_on_eos=True)
        ps = Power_Sampling(block_size=192, MCMC_steps=8)
        result = encode(chain_sampling=smc, token_sampling=ps)
        assert result == "ITS_SMC_10_5_1_PS_192_8"




class TestDecode:
    """Tests for the decode function."""

    def test_decode_smc_only(self):
        """Decoding SMC without token sampling."""
        chain, token = decode("ITS_SMC_10_5_1_NONE")
        assert isinstance(chain, Sequential_Monte_Carlo)
        assert chain.num_particles == 10
        assert chain.tokens_per_step == 5
        assert chain.stop_on_eos == True
        assert token is None

    def test_decode_smc_stop_on_eos_false(self):
        """Decoding SMC with stop_on_eos=False."""
        chain, token = decode("ITS_SMC_8_3_0_NONE")
        assert isinstance(chain, Sequential_Monte_Carlo)
        assert chain.num_particles == 8
        assert chain.tokens_per_step == 3
        assert chain.stop_on_eos == False
        assert token is None



    def test_decode_power_sampling_only(self):
        """Decoding Power Sampling without chain sampling."""
        chain, token = decode("ITS_NONE_PS_192_8")
        assert chain is None
        assert isinstance(token, Power_Sampling)
        assert token.block_size == 192
        assert token.MCMC_steps == 8

    def test_decode_smc_with_power_sampling(self):
        """Decoding SMC combined with Power Sampling."""
        chain, token = decode("ITS_SMC_10_5_1_PS_192_8")
        assert isinstance(chain, Sequential_Monte_Carlo)
        assert chain.num_particles == 10
        assert chain.tokens_per_step == 5
        assert chain.stop_on_eos == True
        assert isinstance(token, Power_Sampling)
        assert token.block_size == 192
        assert token.MCMC_steps == 8



    def test_decode_with_trailing_text(self):
        """Decoding with trailing text (system prompt content)."""
        chain, token = decode("ITS_SMC_10_5_1_NONE You are a helpful assistant.")
        assert isinstance(chain, Sequential_Monte_Carlo)
        assert chain.num_particles == 10
        assert token is None

    def test_decode_invalid_prefix(self):
        """Decoding without ITS prefix raises ValueError."""
        with pytest.raises(ValueError, match="Must start with 'ITS'"):
            decode("INVALID_SMC_10_5_1_NONE")

    def test_decode_unknown_chain_method(self):
        """Decoding with unknown chain sampling method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chain sampling method"):
            decode("ITS_UNKNOWN_10_5_1_NONE")

    def test_decode_unknown_token_method(self):
        """Decoding with unknown token sampling method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown token sampling method"):
            decode("ITS_NONE_UNKNOWN_192_8")

    def test_decode_smc_invalid_params(self):
        """Decoding SMC with non-numeric params raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SMC parameters"):
            decode("ITS_SMC_abc_5_1_NONE")



    def test_decode_ps_invalid_params(self):
        """Decoding PS with non-numeric params raises ValueError."""
        with pytest.raises(ValueError, match="Invalid PS parameters"):
            decode("ITS_NONE_PS_abc_8")

    def test_decode_missing_token_sampling(self):
        """Decoding with missing token sampling section raises ValueError."""
        with pytest.raises(ValueError, match="Missing token sampling"):
            decode("ITS_SMC_10_5_1")


class TestRoundTrip:
    """Tests to verify encode/decode roundtrip consistency."""

    def test_roundtrip_smc_only(self):
        """Roundtrip encoding/decoding SMC only."""
        original_chain = Sequential_Monte_Carlo(num_particles=10, tokens_per_step=5, stop_on_eos=True)
        encoded = encode(chain_sampling=original_chain, token_sampling=None)
        decoded_chain, decoded_token = decode(encoded)
        
        assert decoded_chain.num_particles == original_chain.num_particles
        assert decoded_chain.tokens_per_step == original_chain.tokens_per_step
        assert decoded_chain.stop_on_eos == original_chain.stop_on_eos
        assert decoded_token is None



    def test_roundtrip_power_sampling_only(self):
        """Roundtrip encoding/decoding Power Sampling only."""
        original_token = Power_Sampling(block_size=192, MCMC_steps=8)
        encoded = encode(chain_sampling=None, token_sampling=original_token)
        decoded_chain, decoded_token = decode(encoded)
        
        assert decoded_chain is None
        assert decoded_token.block_size == original_token.block_size
        assert decoded_token.MCMC_steps == original_token.MCMC_steps

    def test_roundtrip_combined(self):
        """Roundtrip encoding/decoding combined chain + token sampling."""
        original_chain = Sequential_Monte_Carlo(num_particles=12, tokens_per_step=6, stop_on_eos=False)
        original_token = Power_Sampling(block_size=128, MCMC_steps=4)
        encoded = encode(chain_sampling=original_chain, token_sampling=original_token)
        decoded_chain, decoded_token = decode(encoded)
        
        assert decoded_chain.num_particles == original_chain.num_particles
        assert decoded_chain.tokens_per_step == original_chain.tokens_per_step
        assert decoded_chain.stop_on_eos == original_chain.stop_on_eos
        assert decoded_token.block_size == original_token.block_size
        assert decoded_token.MCMC_steps == original_token.MCMC_steps
