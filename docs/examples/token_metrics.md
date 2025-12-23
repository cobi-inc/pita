# Token Metric Examples

Token metrics are used to evaluate and guide the generation process.

## Log Probabilities

Standard metric for token likelihood.

```python
# Metrics are often accessed via the sampler's output or internal state
# when using the LogitsLoggingProcessor.
```

## Entropy

Measures the uncertainty of the model at each step.

```python
# Enable entropy calculation in create_autoregressive_sampler
sampler = create_autoregressive_sampler(
    ...,
    logits_processor=True 
)

# During sampling, entropy values are tracked if enabled in Sampling_Params
sampler.sampling_params.entropy = True
```

## Custom Metrics

You can define custom metrics by extending the base metric classes in `pita.sampling.token_metrics`.

```python
from pita.sampling.token_metrics import TokenMetric

class MyCustomMetric(TokenMetric):
    def calculate(self, logits):
        # Your custom logic here
        pass
```
