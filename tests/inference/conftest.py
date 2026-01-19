"""
Pytest configuration for inference tests across all backends.

Provides parameterized model configurations for testing vLLM, LlamaCPP, and TensorRT engines.
"""
import pytest

# vLLM Model Configurations
VLLM_MODELS = {
    "opt-125m": {
        "model": "facebook/opt-125m",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 1024,
    },
    "gpt-oos-20b": {
        "model": "openai/gpt-oss-20b",
        "gpu_memory_utilization": 0.90,
        "max_model_len": 2048,
    },
    "qwen-4b-awq": {
        "model": "Qwen/Qwen3-4B-AWQ",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 2048,
    },
}
DEFAULT_VLLM_MODEL = "opt-125m"

# LlamaCPP Model Configurations (GGUF format)
LLAMACPP_MODELS = {
    "tinyllama-1.1b-gguf": {
        "model": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dtype": "Q4_K_M",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 1024,
    },
}
DEFAULT_LLAMACPP_MODEL = "tinyllama-1.1b-gguf"

# TensorRT Model Configurations
TENSORRT_MODELS = {
    "tinyllama-1.1b": {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 1024,
    },
}
DEFAULT_TENSORRT_MODEL = "tinyllama-1.1b"


def pytest_addoption(parser):
    """Add command-line options for model selection."""
    # vLLM Options
    parser.addoption(
        "--vllm-model",
        action="store",
        default=DEFAULT_VLLM_MODEL,
        choices=list(VLLM_MODELS.keys()),
        help=f"vLLM model to test. Options: {list(VLLM_MODELS.keys())}. Default: {DEFAULT_VLLM_MODEL}"
    )
    parser.addoption(
        "--all-vllm-models",
        action="store_true",
        default=False,
        help="Run tests against all configured vLLM models"
    )

    # LlamaCPP Options
    parser.addoption(
        "--llamacpp-model",
        action="store",
        default=DEFAULT_LLAMACPP_MODEL,
        choices=list(LLAMACPP_MODELS.keys()),
        help=f"LlamaCPP model to test. Options: {list(LLAMACPP_MODELS.keys())}. Default: {DEFAULT_LLAMACPP_MODEL}"
    )
    parser.addoption(
        "--all-llamacpp-models",
        action="store_true",
        default=False,
        help="Run tests against all configured LlamaCPP models"
    )

    # TensorRT Options
    parser.addoption(
        "--tensorrt-model",
        action="store",
        default=DEFAULT_TENSORRT_MODEL,
        choices=list(TENSORRT_MODELS.keys()),
        help=f"TensorRT model to test. Options: {list(TENSORRT_MODELS.keys())}. Default: {DEFAULT_TENSORRT_MODEL}"
    )
    parser.addoption(
        "--all-tensorrt-models",
        action="store_true",
        default=False,
        help="Run tests against all configured TensorRT models"
    )


def pytest_generate_tests(metafunc):
    """Dynamically parameterize tests based on model selection."""
    
    # vLLM Parameterization
    if "vllm_model_config" in metafunc.fixturenames:
        all_models = metafunc.config.getoption("--all-vllm-models")
        if all_models:
            configs = [
                pytest.param(config, id=name)
                for name, config in VLLM_MODELS.items()
            ]
        else:
            model_key = metafunc.config.getoption("--vllm-model")
            configs = [pytest.param(VLLM_MODELS[model_key], id=model_key)]
        metafunc.parametrize("vllm_model_config", configs, scope="module")

    # LlamaCPP Parameterization
    if "llamacpp_model_config" in metafunc.fixturenames:
        all_models = metafunc.config.getoption("--all-llamacpp-models")
        if all_models:
            configs = [
                pytest.param(config, id=name)
                for name, config in LLAMACPP_MODELS.items()
            ]
        else:
            model_key = metafunc.config.getoption("--llamacpp-model")
            configs = [pytest.param(LLAMACPP_MODELS[model_key], id=model_key)]
        metafunc.parametrize("llamacpp_model_config", configs, scope="module")

    # TensorRT Parameterization
    if "tensorrt_model_config" in metafunc.fixturenames:
        all_models = metafunc.config.getoption("--all-tensorrt-models")
        if all_models:
            configs = [
                pytest.param(config, id=name)
                for name, config in TENSORRT_MODELS.items()
            ]
        else:
            model_key = metafunc.config.getoption("--tensorrt-model")
            configs = [pytest.param(TENSORRT_MODELS[model_key], id=model_key)]
        metafunc.parametrize("tensorrt_model_config", configs, scope="module")
