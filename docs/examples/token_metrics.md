# Token Metric Examples

Token metrics are used to evaluate and guide the generation process. The `pita.sampling.token_metrics` module provides utilities for calculating various metrics from model outputs.

## Available Metrics

PITA supports four main token metrics:

- **logprobs**: Log probabilities of the generated tokens (standard model confidence)
- **power_distribution**: Temperature-scaled power distribution using logits and normalization constants
- **entropy**: Model uncertainty at each token position
- **likelihood_confidence**: Combined metric multiplying probability by confidence (exp(-entropy))

## Calculating Token Metrics

```python
from pita.inference.LLM_backend import AutoregressiveSampler
from pita.sampling.token_metrics import calc_token_metric

# Initialize sampler with logits processor enabled for entropy/power metrics
sampler = AutoregressiveSampler(
    engine="vllm",
    model="facebook/opt-125m",
    logits_processor=True  # Required for entropy and power_distribution
)

# Generate output
context = "The capital of France is"
output = sampler.sample(context)

# Calculate different token metrics
logprobs = calc_token_metric(output, sampler, metric="logprobs")
power_dist = calc_token_metric(output, sampler, metric="power_distribution")
entropy = calc_token_metric(output, sampler, metric="entropy")

print(f"Log probabilities shape: {logprobs.shape}")
print(f"Power distribution shape: {power_dist.shape}")
print(f"Entropy shape: {entropy.shape}")
```

## Calculating Sequence Probabilities

```python
from pita.sampling.token_metrics import calc_sequence_prob, calc_sequence_logprob

# Calculate probability of a sequence (tokens 0-5)
seq_prob = calc_sequence_prob(
    output=output,
    sampler=sampler,
    starting_index=0,
    ending_index=5,
    metric="logprobs"
)

# Calculate log probability of a sequence
seq_logprob = calc_sequence_logprob(
    output=output,
    sampler=sampler,
    starting_index=0,
    ending_index=5,
    metric="likelihood_confidence"  # Combines logprobs with entropy
)

print(f"Sequence probability: {seq_prob}")
print(f"Sequence log probability: {seq_logprob}")
```

## Length-Normalized Metrics

For comparing sequences of different lengths, use length-normalized versions:

```python
from pita.sampling.token_metrics import (
    calc_sequence_length_normalized_prob,
    calc_sequence_length_normalized_logprob
)

# Length-normalized probability
norm_prob = calc_sequence_length_normalized_prob(
    output=output,
    sampler=sampler,
    starting_index=0,
    ending_index=5,
    metric="logprobs"
)

# Length-normalized log probability
norm_logprob = calc_sequence_length_normalized_logprob(
    output=output,
    sampler=sampler,
    starting_index=0,
    ending_index=5,
    metric="power_distribution"
)
```

## Accessing Raw Metrics from Output

The `Output` object contains raw metrics:

```python
output = sampler.sample(context)

# Access raw metrics
print(f"Output IDs: {output.output_ids}")
print(f"Top-k logprobs: {output.top_k_logprobs}")
print(f"Top-k logits: {output.top_k_logits}")
print(f"Entropy: {output.entropy}")
print(f"Normalization constants: {output.unprocessed_log_normalization_constant}")
```

## Using Metrics in Sampling Strategies

Token metrics are used internally by sampling strategies for decision-making:

```python
# Enable power sampling with a specific token metric
sampler.enable_power_sampling(
    block_size=250,
    MCMC_steps=3,
    token_metric="power_distribution"  # or "logprobs", "entropy", "likelihood_confidence"
)

output = sampler.token_sample(context)

# Enable Best-of-N with a token metric
sampler.enable_best_of_n(
    N=5,
    token_metric="likelihood_confidence"
)

output = sampler.chain_sample(context)
```
