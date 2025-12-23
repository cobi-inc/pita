# Usage Guide

This guide provides a step-by-step guide to using the `pita` library.

### [Step 1: Choose an Inference Backend](examples/inference_backends.md)

The core functionality of `pita` revolves around inference engines and sampling strategies. First choose an inference backend that you would like to use:

- [vLLM](https://github.com/vllm-project/vllm): A GPU/Accelerated focused platform that leverages pyTorch for inference. Can be used with multi-GPU setups. Limited under-the-hood customization.
- [llama.cpp](https://github.com/ggerganov/llama.cpp): A general purpose backed for both CPUs and GPUs. Written in C++.
- [transformers](https://github.com/huggingface/transformers): A general purpose backed for both CPUs and GPUs. Not optimized for inference, but exposes many under-the-hood customization options.
- [TensorRT](https://github.com/NVIDIA/TensorRT): Nvidia specific platform that is optimized for inference. Can only be used with select GPUs.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): TODO

### [Step 2: Choose Programmatic or API Serving Modes](examples/serving_modes.md)

`pita` can be used in two different modes:

- Programmatic: Use `pita` as a library to run offline inference and sampling strategies.
- API: Use `pita` as a server with limited customization options, but an openAI API compatible endpoint.

### [Step 3.A: Choose a Sampling Strategy](examples/sampling_strategies.md)

`pita` provides several sampling strategies to generate diverse and high-quality outputs. Choose the strategy that best suits your needs:

- Power Sampling: Leverage Metropolis-Hastings MCMC Sampling to generate diverse and high-quality outputs.
- Sequential Monte Carlo/Particle Filtering: Sequential Monte Carlo/Particle Filtering generates diverse and high-quality token sequences, parsing and extending sequences.
- Best-of-N: Select the best N outputs from a set of candidate sequences.
- Beam Search: 
- Combination of Strategies: Combine multiple strategies together to increase the reasoning capabilites of a model.

### [Step 3.B: Choose a Token Metric](examples/token_metrics.md)

`pita` provides several token metrics to evaluate the quality of generated outputs. Choose the metric that best suits your needs:

- Log Probability: Decide based on the log probability of the generated tokens with regular 
- Power Sampling: 
- Entropy: 

### Running from Command Line
TODO
An API endpoint can be created from the command line using:


### Using in Python Code

You can also import and use `pita` components directly in your Python scripts.

```python
import pita
from pita.inference.vllm_backend import create_LLM_object

# Example: setting up an LLM object
llm = create_LLM_object(
    model_name="facebook/opt-125m",
    logits_processor=True
)

# Further usage depends on specific modules (see API Reference)
```


Refer to the [API Reference](api/api_template.md) for detailed documentation on each module.
