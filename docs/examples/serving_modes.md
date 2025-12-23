# Serving Mode Examples

`pita` supports two primary modes of operation: Programmatic and API.

## Programmatic Mode

Use `pita` directly in your Python code for maximum control and offline processing.

```python
from pita.inference.autoregressive_sampler_backend import create_autoregressive_sampler
from pita.sampling.power_sample import power_sampling

# Load the sampler
sampler = create_autoregressive_sampler(
    engine="vllm",
    model="Qwen/Qwen3-4B-AWQ"
)

# Perform custom sampling
prompt = "Write a short story about a robot."
generated_text, acceptances, _, _, _ = power_sampling(
    sampler=sampler,
    prompt=prompt
)

print(f"Generated text: {generated_text}")
print(f"Acceptance rate: {sum(acceptances)/len(acceptances):.2f}")
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
    model="Qwen/Qwen3-4B-AWQ",
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
