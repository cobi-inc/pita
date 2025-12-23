# PITA

**PITA** (Probabilistic Inference Time Algorithms) is a library designed to consolidate and simplfy the usage of probabilistic inference time algorithms with LLMs. It is built on top of existing inference frameworks and provides a unified interface for different inference backends.

## Introduction

`pita` splits probabilistic inference time scaling methods into two categories:

- **Chain Scaling**: Methods that curate the multiple responses to a prompt. For example, Best-of-N creates N sequences and returns the best one based on a decision metric.
- **Token Scaling**: Methods that curate the tokens generated for a prompt. For example, Power Sampling generates iteratively improves the prompt through Metropolis-Hastings Sampling combined with a decision metric.

Chain scaling methods can be combined with token scaling methods to create a hybrid scaling method. For example, Power Best-of-N creates N chains with each chains being generated with Power Sampling. (WIP) See this flow chart for specifics on how to create custom hybrid scaling methods.

Both chain and token scaling methods have shared decision metrics. Decision metrics can be based on token probabiliites, or external graders/process reward models. 

This library can also be used to generate non-probabilistic, non-test-time scaled outputs while taking advantage of the unified interface for different inference backends. Different models run better on different engines and hardware. Develop on your CPU before deploying on your GPU. Swap between ROCm, CUDA, and CPU. `pita` provides a unified interface for the most popular inference backends while your source code remains the same.

## Key Features

### Sampling Methodologies
- **Power Sampling**: Leverage Metropolis-Hastings MCMC Sampling to generate diverse and high-quality outputs.
- **Sequential Monte Carlo/Particle Filtering**: Sequential Monte Carlo/Particle Filtering generates diverse and high-quality token sequences, parsing and extending sequences.
- **Best-of-N**
- **(WIP) Beam Search**
### Decision Metrics
- **Log Probability**: 
- **Power Scaling**: 
- **Entropy**:  
- **(WIP) Entrop Minimization Inference**: 
- **(WIP) Verifiers**: 
### Inference Backends
- **vLLM V1**
- **(WIP) Llama.cpp**
- **(WIP) Transformers**
- **(WIP) TensorRT**
- **(WIP) deepspeed**

## Getting Started

- [Installation](installation.md): Set up your environment and install the library.
- [Usage](usage.md): Learn the basics of running inference and using the library.
- [API Reference](api/api_template.md): Dive into the technical details of modules and classes.

## Contributing

We welcome contributions! Please see the [repository](https://github.com/cobi-inc-MC/pita) for more details on how to contribute.

## License
(WIP)
## Citation
(WIP)
If you use PITA in your research, please cite it as follows:

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

