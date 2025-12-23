# Sampling Strategy Examples

`pita` provides advanced sampling strategies to improve the quality and reasoning capabilities of models.

## Power Sampling

Leverages Metropolis-Hastings MCMC Sampling for diverse and high-quality outputs.

```python
from pita.sampling.power_sample import power_sampling

# Assuming 'sampler' is already initialized
text, _, _, _, _ = power_sampling(
    sampler=sampler,
    prompt="Solving complex physics problem...",
    # Custom power sampling params can be set in sampler.power_sampling_params
)
```

## Sequential Monte Carlo (SMC)

SMC/Particle Filtering generates diverse sequences by maintaining multiple candidate paths.

```python
# SMC sampling example
# sampler.smc_sampling_params = ...
# generated_text = smc_sampling(sampler, prompt)
```

## Best-of-N

Generates N sequences and selects the best one based on a metric.

```python
# Best-of-N sampling example
# sampler.best_of_sampling_params = ...
# best_sequence = best_of_n_sampling(sampler, prompt)
```

## Combining Strategies

You can layer strategies to achieve higher reasoning performance.

```python
# Configuration for combined strategy
# sampler.power_sampling_params = ...
# sampler.smc_sampling_params = ...
```
