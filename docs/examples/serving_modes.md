# Serving Mode Examples

`pita` supports two primary modes of operation: Programmatic and API.

## Programmatic Mode

Use `pita` directly in your Python code for maximum control and offline processing.

```python
from pita.inference.LLM_backend import AutoregressiveSampler

# Initialize the sampler
sampler = AutoregressiveSampler(
    engine="vllm",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    logits_processor=True
)

# Basic sampling
prompt = "Write a short story about a robot."
output = sampler.sample(prompt)
generated_text = sampler.tokenizer.decode(output.output_ids)
print(f"Generated text: {generated_text}")

# Power Sampling
sampler.enable_power_sampling(
    block_size=250,
    MCMC_steps=3,
    token_metric="power_distribution"
)

output = sampler.token_sample(prompt)
generated_text = sampler.tokenizer.decode(output.output_ids)
print(f"Generated text (Power Sampling): {generated_text}")
```

## API Mode

Run `pita` as a server with an OpenAI-compatible API endpoint.

### Starting the Server

You can start the server using Uvicorn:

```bash
uvicorn pita.api.serve:app --host 0.0.0.0 --port 8001
```

### Querying the API

Once the server is running, you can use any OpenAI-compatible client.

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ]
)

print(response.choices[0].message.content)
```

### Advanced Sampling via System Prompt

You can trigger advanced sampling strategies by prefixing your system prompt with `ITS` and specific parameters:

```python
# Example: Trigger Power Sampling with 1000 tokens, block size 250
system_prompt = "ITS PS_1000_250_3 You are a helpful assistant."
```
