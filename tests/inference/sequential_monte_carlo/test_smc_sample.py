
import pytest
from unittest.mock import MagicMock, patch
from pita.sampling.smc import Sequential_Monte_Carlo
from pita.inference.LLM_backend import Output, AutoregressiveSampler, Sampling_Params

@pytest.fixture
def mock_sampler():
    sampler = MagicMock(spec=AutoregressiveSampler)
    sampler.tokenizer = MagicMock()
    sampler.tokenizer.eos_token_id = 100
    sampler.tokenizer.decode.return_value = " decoded"
    sampler.sampling_params = MagicMock(spec=Sampling_Params)
    sampler.sampling_params.max_tokens = 10
    sampler.sampling_params.temperature = 1.0
    return sampler

@pytest.fixture
def mock_output():
    return Output(
        tokens=[1, 2],
        top_k_logits=[[0.1, 0.2], [0.3, 0.4]],
        top_k_logprobs=[[-1.0, -0.5], [-0.1, -0.2]],
        unprocessed_log_normalization_constant=[0.0, 0.0],
        temp_processed_log_normalization_constant=[0.0, 0.0],
        entropy=[0.5, 0.5]
    )   

def test_smc_sample_basic_flow(mock_sampler, mock_output):
    # Setup SMC
    smc = Sequential_Monte_Carlo(
        num_particles=3,
        tokens_per_step=2,
        stop_on_eos=True,
        token_metric="logprobs",
        aggregation="last"
    )

    # Mock sampler.sample to return a valid Output
    mock_sampler.sample.return_value = mock_output
    
    # Mock calc_token_metric to return dummy scores
    # We patch it where it is used in smc.py
    with patch("pita.sampling.smc.calc_token_metric") as mock_calc, \
         patch.object(smc, 'score_update', wraps=smc.score_update) as mock_score_update, \
         patch.object(smc, 'particle_sampling', wraps=smc.particle_sampling) as mock_particle_sampling:
        
        mock_calc.side_effect = lambda output, sampler, metric: [0.9] # Return a high score
        
        # Run sample
        prompt = "Hello"
        result = smc.sample(mock_sampler, prompt)

    # Assertions
    assert isinstance(result, Output)
    # Check that sample was called (3 particles * 2 tokens_per_step * 5 steps (10 total tokens/2 tokens per step))
    assert mock_sampler.sample.call_count == 15
    # Check the call count of Sequential_Monte_Carlo.score_update
    assert mock_score_update.call_count == 15

    # Check the call count of Sequential_Monte_Carlo.particle_sampling
    assert mock_particle_sampling.call_count == 5

    # Check that max_tokens was modified then restored (or at least used correctly)
    assert mock_sampler.sampling_params.max_tokens == 10

    # Check that the correct Output values were return
    # Check the output tokens
    assert result.tokens == [1, 2] * 5
    # Check the output top_k_logprobs
    assert result.top_k_logprobs == [[-1.0, -0.5], [-0.1, -0.2]] * 5
    # Check the output top_k_logits
    assert result.top_k_logits == [[0.1, 0.2], [0.3, 0.4]] * 5
    # Check the output unprocessed_log_normalization_constant
    assert result.unprocessed_log_normalization_constant == [0.0, 0.0] * 5
    # Check the output temp_processed_log_normalization_constant
    assert result.temp_processed_log_normalization_constant == [0.0, 0.0] * 5
    # Check the output entropy
    assert result.entropy == [0.5, 0.5] * 5

def test_smc_sample_eos_handling(mock_sampler):
    # Setup SMC
    smc = Sequential_Monte_Carlo(
        num_particles=2,
        tokens_per_step=2,
        stop_on_eos=True,
        token_metric="logprobs"
    )

    # Output with EOS token
    eos_output = Output(
        tokens=[100], # matches eos_token_id
        top_k_logits=[[0.1]],
        top_k_logprobs=[[-0.1]],
        unprocessed_log_normalization_constant=[0.0],
        temp_processed_log_normalization_constant=[0.0],
        entropy=[0.0]
    )

    mock_sampler.sample.return_value = eos_output
    
    with patch("pita.sampling.smc.calc_token_metric") as mock_calc, \
         patch.object(smc, 'score_update', wraps=smc.score_update) as mock_score_update, \
         patch.object(smc, 'particle_sampling', wraps=smc.particle_sampling) as mock_particle_sampling:
       
        mock_calc.return_value = [1.0]
        
        # Run sample
        # Since all particles hit EOS immediately, it should finish early
        result = smc.sample(mock_sampler, "Start")
        
        # Assertions
        # Validate that it stopped. 
        # If it stops after 1 step because of EOS, sample call count should be num_particles * 1
        assert mock_sampler.sample.call_count == 2 # 1 step * 2 particles
        
        # Check the call count of Sequential_Monte_Carlo.score_update
        # Called once per particle per step -> 2 * 1 = 2
        assert mock_score_update.call_count == 2

        # Check the call count of Sequential_Monte_Carlo.particle_sampling
        # Called once per step -> 1
        assert mock_particle_sampling.call_count == 1

        # Check that max_tokens was modified then restored (or at least used correctly)
        assert mock_sampler.sampling_params.max_tokens == 10

        # Check that the correct Output values were return
        # Since it's stop_on_eos, the result should contain the tokens generated up to EOS
        # The mock returns [100], and 100 is EOS.
        assert result.tokens == [100]
        assert result.top_k_logprobs == [[-0.1]]
        assert result.top_k_logits == [[0.1]]
        assert result.unprocessed_log_normalization_constant == [0.0]
        assert result.temp_processed_log_normalization_constant == [0.0]
        assert result.entropy == [0.0]

def test_smc_sample_token_sampling_method(mock_sampler, mock_output):
    # Test with custom token sampling method
    smc = Sequential_Monte_Carlo(
        num_particles=1,
        tokens_per_step=2,
        token_sampling_method="token_sample"
    )
    
    mock_sampler.token_sample = MagicMock(return_value=mock_output)
    
    with patch("pita.sampling.smc.calc_token_metric") as mock_calc, \
         patch.object(smc, 'score_update', wraps=smc.score_update) as mock_score_update, \
         patch.object(smc, 'particle_sampling', wraps=smc.particle_sampling) as mock_particle_sampling:
        
        mock_calc.return_value = [0.5]
        result = smc.sample(mock_sampler, "Prompt")
        
    mock_sampler.token_sample.assert_called()
    
    # Assertions
    # Check call counts for 1 particle * 1 token_per_step * 10 steps
    # token_sample called 10 times (1 per step for 1 particle)
    assert mock_sampler.token_sample.call_count == 10
    
    # score_update called 10 times (1 per step for 1 particle)
    assert mock_score_update.call_count == 10
    
    # particle_sampling called 10 times (1 per step)
    assert mock_particle_sampling.call_count == 10

    # Check Output Integrity
    # tokens: [1, 2] * 10
    assert result.tokens == [1, 2] * 10
    # top_k_logprobs: [[-1.0, -0.5], [-0.1, -0.2]] * 10
    assert result.top_k_logprobs == [[-1.0, -0.5], [-0.1, -0.2]] * 10
    # top_k_logits: [[0.1, 0.2], [0.3, 0.4]] * 10
    assert result.top_k_logits == [[0.1, 0.2], [0.3, 0.4]] * 10
    # unprocessed_log_normalization_constant: [0.0, 0.0] * 10
    assert result.unprocessed_log_normalization_constant == [0.0, 0.0] * 10
    # temp_processed_log_normalization_constant: [0.0, 0.0] * 10
    assert result.temp_processed_log_normalization_constant == [0.0, 0.0] * 10
    # entropy: [0.5, 0.5] * 10
    assert result.entropy == [0.5, 0.5] * 10

# Test case for when the number of tokens returned is less than tokens_per_step
def test_smc_sample_fewer_tokens_than_step(mock_sampler):
    # Setup SMC with tokens_per_step=2
    smc = Sequential_Monte_Carlo(
        num_particles=1,
        tokens_per_step=2,
        stop_on_eos=False,
        token_metric="logprobs"
    )

    # Output with only 1 token
    short_output = Output(
        tokens=[1],
        top_k_logits=[[0.1]],
        top_k_logprobs=[[-0.1]],
        unprocessed_log_normalization_constant=[0.0],
        temp_processed_log_normalization_constant=[0.0],
        entropy=[0.0]
    )

    mock_sampler.sample.return_value = short_output
    
    with patch("pita.sampling.smc.calc_token_metric") as mock_calc, \
         patch.object(smc, 'score_update', wraps=smc.score_update) as mock_score_update, \
         patch.object(smc, 'particle_sampling', wraps=smc.particle_sampling) as mock_particle_sampling:
        
        mock_calc.return_value = [1.0] # 1 score for 1 token
        
        # Run sample
        result = smc.sample(mock_sampler, "Prompt")
        
        # Verify score_update received the correct token_count (it passes self.tokens_per_step)
        # Even if we got fewer tokens, the code currently passes tokens_per_step (2)
        
        mock_score_update.assert_called()
        args, _ = mock_score_update.call_args
        # args[0] is token_metric_scores (list of float)
        # args[1] is token_count
        assert args[1] == 2 # tokens_per_step
        
        # Assertions
        # 1 particle, 10 max tokens, 2 tokens per step -> 5 steps
        # sampler.sample called 5 times
        assert mock_sampler.sample.call_count == 5
        
        # score_update called 5 times
        assert mock_score_update.call_count == 5
        
        # particle_sampling called 5 times
        assert mock_particle_sampling.call_count == 5
        
        # Check Output Integrity
        # We generated 1 token per step for 5 steps -> 5 tokens total
        assert result.tokens == [1] * 5
        
        # Check other fields
        assert result.top_k_logprobs == [[-0.1]] * 5
        assert result.top_k_logits == [[0.1]] * 5
        assert result.unprocessed_log_normalization_constant == [0.0] * 5
        assert result.temp_processed_log_normalization_constant == [0.0] * 5
        assert result.entropy == [0.0] * 5


# Validates the SMC sample function using a real AutoregressiveSampler (Integration Test)
# Uses Qwen/Qwen3-4B-AWQ model

@pytest.fixture(scope="module")
def real_sampler():
    # Initialize the sampler with Qwen/Qwen3-4B-AWQ
    sampler = AutoregressiveSampler(
        engine="vllm",
        model="Qwen/Qwen3-4B-AWQ",
        dtype="auto",
        tokenizer_path=None,
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        max_probs=10,
        logits_processor=True,
        trust_remote_code=True,
        sampling_params=None
    )
    sampler.sampling_params.max_tokens = 500
    sampler.enable_smc(
        num_particles=4,
        tokens_per_step=100,
        stop_on_eos=True,
        token_metric="logprobs",
        aggregation="last"
    )
    yield sampler
    # Cleanup if needed
    if hasattr(sampler, 'llm'):
        del sampler.llm

def test_smc_with_real_sampler(real_sampler):
    prompt = "The capital of France is"
    
    # Run sample
    result = real_sampler.chain_sample(prompt)

    # Assertions
    assert isinstance(result, Output)
    assert len(result.tokens) > 0
    # Basic check that we got some valid text or tokens back
    # With this small model, output might be repetitive but should exist.
    decoded_text = real_sampler.tokenizer.decode(result.tokens)
    print(f"SMC Generated Text: {decoded_text}")
    
    # Verify structure
    assert result.top_k_logprobs is not None
    assert result.top_k_logits is not None 

# TODO: Add a test case where SMC uses the token_sample method with power sampling
