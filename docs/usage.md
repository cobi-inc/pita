# Usage

This guide provides basic usage examples for the `pita` library.

## Basic Inference

The core functionality of `pita` revolves around inference engines and sampling strategies. Here is a basic example of how to run inference using the library.

### Running from Command Line

You can run modules directly from the command line using the `python -m` syntax.

```bash
python -m pita.sampling.power_sample --model "your-model-name"
```

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

## detailed Components

The library is structured into several key components:

- **Inference**: Backends for different inference engines (e.g., vLLM, DeepSpeed).
- **Sampling**: Various sampling strategies (e.g., Power Sampling, Best Of N).
- **Utils**: Helper functions for benchmarking, parsing, and system management.

Refer to the [API Reference](api/api_template.md) for detailed documentation on each module.
