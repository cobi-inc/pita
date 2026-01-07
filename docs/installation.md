# Installation Guide

This guide explains how to install and set up PITA for different hardware configurations and inference engines.

## Prerequisites

- **Conda** (Miniconda or Anaconda) for environment management
- **Git** for cloning the repository

```bash
git clone https://github.com/your-org/pita.git
cd pita
```

---

## CPU Installation

For development, testing, or systems without GPU acceleration.

### Option 1: llama.cpp (Recommended for CPU)

```bash
# Create environment
conda env create -f environment_files/pita_llama_cpp.yml
conda activate pita_llama_cpp

# Install pita in editable mode
pip install -e .
```

### Option 2: Manual Setup

```bash
conda create -n pita_cpu python=3.12 -y
conda activate pita_cpu

pip install llama-cpp-python
pip install -e .
pip install pytest  # For testing
```

### Verify Installation

```bash
python -c "import pita; import llama_cpp; print('CPU installation successful!')"
```

---

## NVIDIA CUDA Installation

For systems with NVIDIA GPUs. Choose your preferred inference engine:

### Option A: llama.cpp with CUDA (Recommended)

Best for: Smaller models, lower memory usage, flexible quantization options.

#### Quick Setup (Automated)

```bash
cd environment_files
./setup_llamacpp_cuda.sh
```

This script creates the environment, builds llama-cpp-python from source with CUDA, and verifies the installation.

#### Manual Setup

```bash
# Create conda environment
conda env create -f environment_files/power_sampling_llamacpp_cuda.yml
conda activate pita_llamacpp_cuda

# Set up CUDA build environment
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Build llama-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler" \
  pip install llama-cpp-python --no-cache-dir

# Install pita
pip install -e .
```

#### Verify CUDA Backend

```bash
python -c "
import os
lib_dir = os.path.dirname(__import__('llama_cpp').__file__) + '/lib'
libs = os.listdir(lib_dir)
assert 'libggml-cuda.so' in libs, 'CUDA backend not found!'
print('CUDA backend installed:', [l for l in libs if 'cuda' in l])
"
```

---

### Option B: vLLM with CUDA

Best for: Large models, high throughput, production deployments.

```bash
# Create environment with vLLM and CUDA 12.8
conda env create -f environment_files/power_sampling_vllm_cuda.yml
conda activate power_sampling

# Install pita
pip install -e .
```

#### vLLM Requirements

- NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- CUDA 12.x driver installed on host system
- Sufficient GPU memory for your target model

#### Verify vLLM Installation

```bash
python -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')"
```

---

## Choosing an Inference Engine

| Feature | llama.cpp | vLLM |
|---------|-----------|------|
| **Best for** | Experimentation, quantized models | Production, high throughput |
| **Memory usage** | Lower (supports aggressive quantization) | Higher |
| **Model formats** | GGUF | HuggingFace, GPTQ, AWQ |
| **Batch processing** | Limited | Excellent |
| **Setup complexity** | Simple | Moderate |

---

## Running Tests

After installation, verify everything works:

```bash
# Run the test suite
pytest tests/ -v

# Run specific backend tests
pytest tests/inference/ -v -k "llama"  # llama.cpp tests
pytest tests/inference/ -v -k "vllm"   # vLLM tests
```

---

## Troubleshooting

### llama.cpp CUDA Build Errors

**"unsupported GNU version"**: Add the compiler compatibility flag:
```bash
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler" pip install llama-cpp-python
```

**"cuda_runtime.h: No such file"**: Set the CUDA headers path:
```bash
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
```

### vLLM Import Errors

**GPU not detected**: Ensure NVIDIA drivers are installed:
```bash
nvidia-smi  # Should show your GPU
```

### General Issues

**Environment conflicts**: Create a fresh environment:
```bash
conda env remove -n <env_name> -y
conda env create -f <environment_file.yml>
```

---

## Platform Support Matrix

| Platform | llama.cpp | vLLM | Status |
|----------|-----------|------|--------|
| Linux + NVIDIA CUDA | ✅ | ✅ | Fully supported |
| Linux + CPU | ✅ | ❌ | llama.cpp only |
| macOS + Apple Silicon | 🔄 | ❌ | In development |
| Linux + AMD ROCm | 🔄 | 🔄 | In development |

✅ = Supported | 🔄 = In development | ❌ = Not supported
