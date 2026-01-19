# PITA Test Suite

This directory contains the test suite for PITA. Tests are organized by functionality and can be run using `pytest`.

## Parameterized Testing

The test suite supports running tests against multiple model configurations and inference backends simultaneously. Configuration is managed in `tests/inference/conftest.py`.

### vLLM Backend
Run tests using the vLLM backend. Requires `pita_vllm_cuda` environment.

```bash
# Run with default model (opt-125m)
pytest tests/inference/

# Run with a specific model
pytest tests/inference/ --vllm-model=gpt-oos-20b

# Run with all configured vLLM models
pytest tests/inference/ --all-vllm-models
```

**Available vLLM Models:**
- `opt-125m` (Default)
- `gpt-oos-20b`
- `qwen-4b-awq`

### LlamaCPP Backend
Run tests using the LlamaCPP backend. Requires `pita_llamacpp_cuda` environment.

```bash
# Run with default model (tinyllama-1.1b-gguf)
pytest tests/inference/base_autoregressive_sampler/test_AutoregressiveSampler_class_llama_cpp.py

# Run with a specific model
pytest tests/inference/power_sampling/test_power_sampling_llama_cpp.py --llamacpp-model=tinyllama-1.1b-gguf

# Run with all configured LlamaCPP models
pytest tests/inference/power_sampling/test_power_sampling_llama_cpp.py --all-llamacpp-models
```

**Available LlamaCPP Models:**
- `tinyllama-1.1b-gguf` (Default)

### TensorRT Backend
Run tests using the TensorRT backend. Requires `pita_tensorrt_cuda` environment.

```bash
# Run with default model (tinyllama-1.1b)
pytest tests/inference/base_autoregressive_sampler/test_AutoregressiveSampler_class_tensorrt.py

# Run with a specific model
pytest tests/inference/base_autoregressive_sampler/test_AutoregressiveSampler_class_tensorrt.py --tensorrt-model=tinyllama-1.1b

# Run with all configured TensorRT models
pytest tests/inference/base_autoregressive_sampler/test_AutoregressiveSampler_class_tensorrt.py --all-tensorrt-models
```

**Available TensorRT Models:**
- `tinyllama-1.1b` (Default)

## Directory Structure

- `inference/`: Tests for inference backends (vLLM, LlamaCPP, TensorRT).
  - `base_autoregressive_sampler/`: Tests for the base `AutoregressiveSampler` class.
  - `power_sampling/`: Tests for Power Sampling algorithms.
- `conftest.py`: Global pytest configuration and parameterized fixtures.
