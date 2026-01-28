<div align="center">
  <img src="docs/assets/pita_logo.png" alt="PITA Logo" width="300"/>
</div>

# PITA - Probabilistic Inference Time Algorithms (Community Edition)

PITA is a unified Python library for probabilistic inference-time scaling methods with Large Language Models (LLMs). It provides a consistent interface across multiple inference backends and sampling strategies to improve output quality through test-time computation. PITA is managed by [COBI](https://cobi.tech/). Get in touch with the COBI team at info@cobi.tech.

Visualize the PITA library using the [TOKEN SPACE EXPLORER](https://explore.cobi.tech/).

## Features

### Sampling Methodologies
- **Power Sampling**: Metropolis-Hastings MCMC-based token refinement
- **Sequential Monte Carlo (SMC)**: Particle filtering for iterative sequence refinement
- **Hybrid Strategies**: Combine chain and token-level scaling methods

### Decision Metrics
- **Log Probabilities**: Standard model confidence scoring
- **Power Distribution**: Temperature-scaled confidence metrics
- **Entropy**: Model uncertainty quantification
- **Likelihood Confidence**: Combined probability and entropy metrics

### Inference Backends
- **vLLM**: High-throughput GPU inference (primary backend)
- **llama.cpp**: CPU/GPU inference with GGUF model support
- **Transformers (WIP)**: HuggingFace integration for flexibility
- **TensorRT**: NVIDIA-optimized inference
- **DeepSpeed (WIP)**: Distributed inference support

## Documentation

**📚 Full documentation available at: [cobi-inc.github.io/pita](https://cobi-inc.github.io/pita/)**

- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api/api_template.md)
- [Examples](docs/examples/)

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Conda (recommended)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/cobi-inc-MC/pita.git
cd pita

# Install the library
pip install -e .
```

### Backend-Specific Installation

For **vLLM** (GPU recommended):
```bash
pip install -e ".[vllm]"
```

For **llama.cpp** (CPU/GPU):
```bash
pip install -e ".[llama-cpp]"
```

For **TensorRT** (NVIDIA GPUs):
```bash
pip install -e ".[tensorrt]"
```

See [Installation Guide](docs/installation.md) for detailed instructions.

## Quick Example

```python
from pita.inference.LLM_backend import AutoregressiveSampler

# Initialize sampler
sampler = AutoregressiveSampler(
    engine="vllm",
    model="facebook/opt-125m",
    logits_processor=True
)

# Basic sampling
output = sampler.sample("What is the capital of France?")
print(sampler.tokenizer.decode(output.output_ids))

# Advanced sampling with Power Sampling
sampler.enable_power_sampling(
    block_size=250,
    MCMC_steps=3,
    token_metric="power_distribution"
)
output = sampler.token_sample("Solve: 2x + 5 = 13")
print(sampler.tokenizer.decode(output.output_ids))
```

## API Server

Run PITA as an OpenAI-compatible API server:

```bash
# Start server with defaults
pita serve

# Or customize with options
pita serve --model Qwen/Qwen2.5-0.5B-Instruct --engine vllm --port 8001

# Environment variables are also supported
export PITA_ENGINE=vllm
export PITA_MODEL=Qwen/Qwen2.5-0.5B-Instruct
pita serve
```

Query the API:
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-used"  # Local server does not require authentication
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

## Project Structure

```
pita/                              # Repository root
├── pita/                          # Main Python package
│   ├── inference/                 # Backend abstraction layer
│   │   ├── LLM_backend.py         # Core AutoregressiveSampler
│   │   ├── vllm_backend.py        # vLLM engine
│   │   ├── llama_cpp_backend.py   # llama.cpp engine
│   │   ├── tensorRT_backend.py    # TensorRT engine
│   │   └── *_logits_processor.py  # Logits processors
│   ├── sampling/                  # Inference-time algorithms
│   │   ├── power_sample.py        # Power Sampling
│   │   ├── smc.py                 # Sequential Monte Carlo
│   │   └── token_metrics.py       # Metric utilities
│   ├── api/                       # REST API
│   │   ├── serve.py               # FastAPI server
│   │   └── api_template.py        # Request/response models
│   └── utils/                     # Helper functions
│       ├── benchmarking_utils.py
│       ├── parse_utils.py
│       ├── redis_manager.py
│       └── grading_utils/         # Math grading
├── tests/                         # Test suite
├── docs/                          # MkDocs documentation
└── pyproject.toml                 # Project configuration
```

## Contributing

We welcome contributions! Please see the [repository](https://github.com/cobi-inc-MC/pita) for more details.

## License

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

PITA is **dual-licensed** to accommodate both open source and commercial use:

### Open Source License (AGPLv3+)

This software is free and open source under the **GNU Affero General Public License v3.0 or later (AGPLv3+)**.

You can freely use, modify, and distribute PITA under the terms of the AGPLv3+. Key requirements:
- Source code must be made available to users (including network users)
- Derivative works must also be licensed under AGPLv3+
- See the [LICENSE](LICENSE) file for complete terms

### Commercial License

If you need to use PITA in a proprietary application without the source code disclosure requirements of AGPLv3, we offer **commercial licensing**.

Commercial licenses allow you to:
- Use PITA in closed-source software
- Avoid AGPLv3 network use obligations
- Receive custom licensing terms

**Contact:** sales@cobi-inc.com for commercial licensing information.

### Important Licensing Notes

**Third-Party Dependencies:**
All direct dependencies use permissive licenses (MIT, BSD, Apache 2.0). See [NOTICE](NOTICE) for attributions.

**TensorRT Backend (Optional):**
The optional TensorRT backend requires NVIDIA's proprietary TensorRT library with separate licensing. Users must obtain TensorRT from NVIDIA and accept their terms. See [docs/TENSORRT-LICENSE-NOTICE.md](docs/TENSORRT-LICENSE-NOTICE.md) for details.

**HuggingFace Models:**
Individual models from HuggingFace Hub have their own licenses (including GPL, non-commercial, or proprietary licenses). **Users are responsible for verifying model licenses** before use.

For complete licensing information, see [docs/LICENSING-GUIDE.md](docs/LICENSING-GUIDE.md).

## Citation

If you use PITA in your research, please cite:

```bibtex
@misc{pita2025,
  author = {COBI, Inc. Engineering Team},
  title = {PITA: Probabilistic Inference Time Algorithms},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cobi-inc-MC/pita}}
}
```
