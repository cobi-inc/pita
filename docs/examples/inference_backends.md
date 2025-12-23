# Inference Backend Examples

This page provides examples of how to initialize and use different inference backends supported by `pita`.

## vLLM Backend

vLLM is a high-throughput, memory-efficient serving engine for LLMs.

```python
from pita.inference.autoregressive_sampler_backend import create_autoregressive_sampler

# Initialize vLLM sampler
sampler = create_autoregressive_sampler(
    engine="vllm",
    model="facebook/opt-125m",
    dtype="auto",
    gpu_memory_utilization=0.85,
    max_model_len=1024,
    logits_per_token=100  # Return top 100 logits
)

# Sample from the model
context = "What is the capital of France?"
generated_text, _, _ = sampler.sample(context, max_new_tokens=10)
print(generated_text)
```

## Llama.cpp Backend

Llama.cpp is a general-purpose backend for both CPUs and GPUs, optimized for GGUF models.

```python
from pita.inference.autoregressive_sampler_backend import create_autoregressive_sampler

# Initialize Llama.cpp sampler
sampler = create_autoregressive_sampler(
    engine="llama_cpp",
    model="path/to/your/model.gguf",
    dtype="Q5_K_M",
    gpu_memory_utilization=0.85,
    max_model_len=1024
)

# Sample from the model
context = "Explain the theory of relativity."
generated_tokens, _, _ = sampler.sample(context, max_new_tokens=50)
generated_text = sampler.tokenizer.decode(generated_tokens)
print(generated_text)
```

## Transformers Backend

The Transformers backend is useful for models not yet supported by vLLM or llama.cpp, or when deep customization is needed.

```python
# Note: Transformers support is currently under development.
# The usage pattern follows the same create_autoregressive_sampler API.
```
