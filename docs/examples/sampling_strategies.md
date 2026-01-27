# Sampling Strategy Examples

`pita` provides advanced sampling strategies to improve the quality and reasoning capabilities of models.

## Power Sampling

Power Sampling uses Metropolis-Hastings MCMC to iteratively refine generated tokens. It operates at the token level, proposing and accepting/rejecting token replacements based on a decision metric.

```python
from pita.inference.LLM_backend import AutoregressiveSampler

# Initialize sampler
sampler = AutoregressiveSampler(
    engine="vllm",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    logits_processor=True  # Required for power sampling
)

# Enable power sampling
sampler.enable_power_sampling(
    block_size=250,          # Tokens generated per block
    MCMC_steps=3,            # Number of MCMC refinement steps
    token_metric="power_distribution"  # Metric for accept/reject decisions
)

# Use token sampling
prompt = "Solve the equation: 3x + 7 = 22"
output = sampler.token_sample(prompt)
generated_text = sampler.tokenizer.decode(output.output_ids)
print(generated_text)
```

### Available Token Metrics for Power Sampling

- `"logprobs"`: Standard log probability scoring
- `"power_distribution"`: Temperature-scaled power distribution (recommended)
- `"entropy"`: Entropy-based metric
- `"likelihood_confidence"`: Combined probability and confidence

## Sequential Monte Carlo (SMC)

SMC maintains multiple candidate sequences (particles) and selectively prunes/extends them based on quality metrics. It operates at the chain level.

```python
from pita.inference.LLM_backend import AutoregressiveSampler

# Initialize sampler
sampler = AutoregressiveSampler(
    engine="vllm",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    logits_processor=True  # Required for SMC metrics
)

# Enable SMC
sampler.enable_smc(
    num_particles=5,         # Number of candidate sequences to maintain
    tokens_per_step=50,      # Tokens generated per SMC step
    stop_on_eos=True,        # Stop when EOS token is generated
    token_metric="likelihood_confidence",  # Metric for particle scoring
    aggregation="last"       # How to aggregate token scores ("last", "minimum", "product")
)

# Use chain sampling
prompt = "Write a detailed explanation of photosynthesis."
output = sampler.chain_sample(prompt)
generated_text = sampler.tokenizer.decode(output.output_ids)
print(generated_text)
```

### SMC Aggregation Methods

- `"last"`: Use only the last token's metric for scoring
- `"minimum"`: Use the minimum metric across all tokens
- `"product"`: Multiply metrics across all tokens
- `"model_aggregate"`: Custom model-based aggregation (WIP)


## Combining Strategies (Advanced)

You can combine chain-level and token-level strategies for hybrid scaling. 

```python
from pita.inference.LLM_backend import AutoregressiveSampler

# Initialize sampler
sampler = AutoregressiveSampler(
    engine="vllm",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    logits_processor=True
)

# Enable token-level (Power Sampling)
sampler.enable_power_sampling(
    block_size=200,
    MCMC_steps=2,
    token_metric="power_distribution"
)

# Use power sampling
output_power = sampler.token_sample(prompt)
```

## Using Sampling Strategies via API

You can trigger sampling strategies via the API server using special system prompts:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="none"
)

# Power Sampling via API: ITS PS_<max_tokens>_<block_size>_<MCMC_steps>
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {"role": "system", "content": "ITS PS_1000_250_3 You are a helpful assistant."},
        {"role": "user", "content": "Solve: 5x - 3 = 17"}
    ]
)

print(response.choices[0].message.content)
```

## Disabling Sampling Strategies

To revert to standard sampling:

```python
# Disable token sampling (if enabled)
if hasattr(sampler, 'token_sample_name'):
    sampler.token_sample_name = None
    sampler.token_sample_fn = None

# Disable chain sampling (if enabled)
if hasattr(sampler, 'chain_sample_name'):
    sampler.chain_sample_name = None
    sampler.chain_sample_fn = None

# Now sampler.sample() will use standard autoregressive sampling
output = sampler.sample(prompt)
```
