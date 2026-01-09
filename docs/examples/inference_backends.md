# Inference Backend Examples

This page provides examples of how to initialize and use different inference backends supported by `pita`.

## vLLM Backend

vLLM is a high-throughput, memory-efficient serving engine for LLMs.

```python
from pita.inference.LLM_backend import AutoregressiveSampler

# Initialize vLLM sampler
sampler = AutoregressiveSampler(
    engine="vllm",
    model="facebook/opt-125m",
    dtype="auto",
    gpu_memory_utilization=0.85,
    max_model_len=1024,
    max_probs=100,  # Return top 100 logits/logprobs
    logits_processor=True  # Enable logits processor for entropy/normalization
)

# Configure sampling parameters
sampler.sampling_params.max_tokens = 10
sampler.sampling_params.temperature = 1.0

# Sample from the model
context = "What is the capital of France?"
output = sampler.sample(context)
generated_text = sampler.tokenizer.decode(output.output_ids)
print(generated_text)
```

## Llama.cpp Backend

Llama.cpp is a general-purpose backend for both CPUs and GPUs, optimized for GGUF models.

```python
from pita.inference.LLM_backend import AutoregressiveSampler

# Initialize Llama.cpp sampler
sampler = AutoregressiveSampler(
    engine="llama_cpp",
    model="path/to/your/model.gguf",
    dtype="auto",  # dtype is handled by GGUF quantization
    max_model_len=1024,
    max_probs=100
)

# Configure sampling parameters
sampler.sampling_params.max_tokens = 50

# Sample from the model
context = "Explain the theory of relativity."
output = sampler.sample(context)
generated_text = sampler.tokenizer.decode(output.output_ids)
print(generated_text)
```

## TensorRT Backend

TensorRT provides optimized inference for NVIDIA GPUs.

```python
from pita.inference.LLM_backend import AutoregressiveSampler

# Initialize TensorRT sampler
sampler = AutoregressiveSampler(
    engine="tensorrt",
    model="path/to/tensorrt/engine",
    max_model_len=1024,
    max_probs=100,
    logits_processor=True
)

# Sample from the model
context = "Describe quantum computing."
output = sampler.sample(context)
generated_text = sampler.tokenizer.decode(output.output_ids)
print(generated_text)
```

## Transformers Backend

The Transformers backend is useful for models not yet supported by vLLM or llama.cpp, or when deep customization is needed.

```python
from pita.inference.LLM_backend import AutoregressiveSampler

# Initialize Transformers sampler
sampler = AutoregressiveSampler(
    engine="transformers",
    model="facebook/opt-125m",
    dtype="auto",
    max_model_len=1024
)

# Sample from the model
context = "Once upon a time"
output = sampler.sample(context)
generated_text = sampler.tokenizer.decode(output.output_ids)
print(generated_text)
```
