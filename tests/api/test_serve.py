
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

def test_create_app_config(mock_sampler_class):
    """Test that create_app uses the provided configuration."""
    config = {
        "engine": "llama_cpp",
        "model": "test-llama",
        "tokenizer": "test-tokenizer",
        "port": 9000,
        "host": "127.0.0.1"
    }
    
    # Use TestClient as context manager to trigger startup/shutdown events
    app = create_app(config)
    with TestClient(app) as client:
        # Check if AutoregressiveSampler was initialized with correct args
        mock_sampler_class.assert_called_once()
        call_args = mock_sampler_class.call_args[1]
        assert call_args["engine"] == "llama_cpp"
        assert call_args["model"] == "test-llama"
        assert call_args["tokenizer_path"] == "test-tokenizer"

def test_endpoint_access_sampler(mock_sampler_instance):
    """Test that endpoints can access the sampler from app.state."""
    config = {
        "engine": "vllm",
        "model": "test-model",
        "tokenizer": None,
        "port": 8000,
        "host": "0.0.0.0"
    }
    app = create_app(config)
    
    with TestClient(app) as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"

def test_create_completion(mock_sampler_instance):
    """Test create_completion endpoint."""
    config = {
        "engine": "vllm",
        "model": "test-model",
        "tokenizer": None,
        "port": 8000,
        "host": "0.0.0.0"
    }
    app = create_app(config)
    
    with TestClient(app) as client:
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello world"
        
        # Verify sample was called
        mock_sampler_instance.sample.assert_called_once()

def test_create_completion_uninitialized(mock_sampler_class):
    """Test behavior when sampler is not initialized."""
    config = {
        "engine": "vllm",
        "model": "test-model",
        "tokenizer": None,
        "port": 8000,
        "host": "0.0.0.0"
    }
    app = create_app(config)
    
    with TestClient(app) as client:
        # Manually force sampler to None to simulate error or uninitialized state
        app.state.sampler = None
        
        response = client.get("/v1/models")
        assert response.status_code == 503
        
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 503
