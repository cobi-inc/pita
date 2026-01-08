
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from pita.api.serve import create_app, ChatMessageRole

# Mock AutoregressiveSampler to avoid loading actual models during tests
@pytest.fixture
def mock_sampler_class():
    with patch("pita.api.serve.AutoregressiveSampler") as mock:
        yield mock

@pytest.fixture
def mock_sampler_instance(mock_sampler_class):
    instance = MagicMock()
    instance.engine = "vllm"
    instance.model = "test-model"
    instance.tokenizer.model_max_length = 2048
    # Mock decode to return some text
    instance.tokenizer.decode.return_value = "Hello world"
    # Mock encode
    instance.tokenizer.encode.return_value = [1, 2, 3]
    # Mock apply_chat_template to return a string
    instance.tokenizer.apply_chat_template.return_value = "system: hi user: hello"
    
    # Mock sample return value
    mock_output = MagicMock()
    mock_output.tokens = [4, 5, 6]
    instance.sample.return_value = mock_output
    
    mock_sampler_class.return_value = instance
    return instance

def test_its_system_prompt_chain_sampling(mock_sampler_instance):
    """Test that ITS system prompt triggers chain sampling and modifies the prompt."""
    config = {
        "engine": "vllm",
        "model": "test-model",
        "tokenizer": None,
        "port": 8000,
        "host": "0.0.0.0"
    }
    app = create_app(config)
    
    # Mock decode to return a chain_sampling object
    mock_chain_sampling = MagicMock()
    # Ensure sample returns an Output-like object
    mock_output = MagicMock()
    mock_output.tokens = [7, 8, 9]
    mock_chain_sampling.sample.return_value = mock_output
    
    # Patch decode in the serve module
    with patch("pita.api.serve.decode", return_value=(mock_chain_sampling, None)) as mock_decode:
        with TestClient(app) as client:
            payload = {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "ITS_SMC_10_5_1_NONE some parameters"},
                    {"role": "user", "content": "hello"}
                ],
                "max_tokens": 100
            }
            response = client.post("/v1/chat/completions", json=payload)
            
            assert response.status_code == 200
            
            # Verify decode was called
            mock_decode.assert_called_once_with("ITS_SMC_10_5_1_NONE some parameters")
            
            # Verify chain_sampling.sample was called
            mock_chain_sampling.sample.assert_called_once()
            
            # Verify regular sampler.sample was NOT called
            mock_sampler_instance.sample.assert_not_called()
            
            # Verify that the system message content was updated (ITS prefix removed)
            # We access the mock calls to apply_chat_template
            call_args = mock_sampler_instance.tokenizer.apply_chat_template.call_args
            assert call_args is not None
            # The first argument is the messages list
            messages_arg = call_args[0][0]
            # Check the system message content
            # Check the system message content
            assert messages_arg[0].role == ChatMessageRole.System
            # "ITS some parameters".split(" ")[1:] -> ["some", "parameters"] -> "some parameters"
            assert messages_arg[0].content == "some parameters"
