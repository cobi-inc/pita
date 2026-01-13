# Installation Guide

This guide explains how to install and set up PITA for different hardware configurations and inference engines.

## Prerequisites

- **Conda** (Miniconda or Anaconda) for environment management
- **Git** for cloning the repository

```bash
git clone https://github.com/cobi-inc-MC/pita
cd pita
```

---

## CPU Installation

For development, testing, or systems without GPU acceleration.

### Option 1: llama.cpp (Recommended for CPU)

Save the following as `pita_llama_cpp.yml`:
```yaml
name: pita_llama_cpp
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pip
  - llama-cpp-python
  - pytest
```

Then run:
```bash
conda env create -f pita_llama_cpp.yml
conda activate pita_llama_cpp

# Install pita in editable mode
pip install -e .
```

### Option 2: Windows CPU (llama.cpp)

For Windows users without a dedicated GPU.

Save the following as `llamacpp_windows_cpu.yml`:
```yaml
name: pita_llamacpp_windows_cpu
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - llama-cpp-python
  - pytest
  - pytest-asyncio
```

Then run:
```powershell
conda env create -f llamacpp_windows_cpu.yml
conda activate pita_llamacpp_windows_cpu

# Install pita (ensure you are in the root directory)
pip install -e ".[llama_cpp]"
```

### Option 3: Manual Setup

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

### Option A.1: llama.cpp with CUDA using scripts (Recommended)

For a streamlined installation on Linux, we provide an automated setup script that handles environment creation and CUDA compilation for you.

See the [Automated Installation Guide](automated_installation.md) for detailed instructions on using the `setup_llamacpp_cuda.sh` script.

### Option A.2: llama.cpp with CUDA (Manual)

Best for: Smaller models, lower memory usage, flexible quantization options.

Save the following as `llamacpp_cuda.yml`:
```yaml
name: pita_llamacpp_cuda
channels:
  - defaults
  - nvidia
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - cuda-cudart=12.4.127
  - cuda-toolkit=12.4.1
  - cmake
```

Then run the setup:

```bash
# 1. Create environment
conda env create -f llamacpp_cuda.yml
conda activate pita_llamacpp_cuda

# 2. Set up CUDA build environment
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 3. Build llama-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler" \
  pip install llama-cpp-python --no-cache-dir

# 4. Install pita
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

Save the following as `vllm_cuda.yml`:
```yaml
name: pita_vllm_cuda
channels:
  - defaults
  - nvidia
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - cuda-toolkit=12.8
  - cxx-compiler
  - redis-server
  - pip:
    - vllm==0.11.0
    - pandas==2.3.3
    - datasets==4.3.0
    - regex==2025.9.18
```

Then run:

```bash
# Create environment with vLLM and CUDA 12.8
conda env create -f vllm_cuda.yml
conda activate pita_vllm_cuda

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

### Option C: TensorRT-LLM with CUDA

Best for: Maximum performance on NVIDIA hardware, production deployments.

Save the following as `tensorrt_cuda.yml`:
```yaml
name: pita_tensorrt_cuda
channels:
  - defaults
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - cxx-compiler
  - onnx<1.16.0
  - mpi4py
  - openmpi
  - pytest
  - pip:
    - --extra-index-url https://pypi.nvidia.com/
    - tensorrt_llm
    - torch
    - transformers
    - numpy
    - scipy
    - pandas
    - regex
    - pydantic
    - fastapi
    - redis>=4.0.0
    - redis-server
    - uvicorn
```

Then run:

```bash
# 1. Create environment
conda env create -f tensorrt_cuda.yml
conda activate pita_tensorrt_cuda

# 2. Install pita in editable mode
pip install -e .
```

#### Verify TensorRT-LLM Installation

```bash
python -c "import tensorrt_llm; print(f'TensorRT-LLM {tensorrt_llm.__version__} installed successfully')"
```

---

## Choosing an Inference Engine

| Feature | llama.cpp | vLLM | TensorRT-LLM |
|---------|-----------|------|--------------|
| **Best for** | Experimentation, quantized models | Production, high throughput | Maximum performance, production |
| **Memory usage** | Lower (supports aggressive quantization) | Higher | High (optimized for performance) |
| **Model formats** | GGUF | HuggingFace, GPTQ, AWQ | TensorRT engines (built from HF/ONNX) |
| **Batch processing** | Limited | Excellent | Excellent |
| **Setup complexity** | Simple | Moderate | Moderate/High |

---

## Running Tests

After installation, verify everything works:

```bash
# Run the test suite
pytest tests/ -v

# Run specific backend tests
pytest tests/inference/ -v -k "llama"     # llama.cpp tests
pytest tests/inference/ -v -k "vllm"      # vLLM tests
pytest tests/inference/ -v -k "tensorrt"  # TensorRT-LLM tests
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

### TensorRT-LLM Issues

**"AttributeError: module 'onnx.helper' has no attribute 'float32_to_bfloat16'"**: Ensure you are using `onnx<1.16.0`.

**"ImportError: libmpi.so.40"**: Ensure `openmpi` is installed via conda (`conda list openmpi`).

### General Issues

**Environment conflicts**: Create a fresh environment:
```bash
conda env remove -n <env_name> -y
conda env create -f <environment_file.yml>
```

---

## Platform Support Matrix

| Platform | llama.cpp | vLLM | TensorRT-LLM | Status |
|----------|-----------|------|--------------|--------|
| Linux + NVIDIA CUDA | ✅ | ✅ | ✅ | Fully supported |
| Linux + CPU | ✅ | ❌ | ❌ | llama.cpp only |
| macOS + Apple Silicon | 🔄 | 🔄 | ❌ | In development |
| Linux + AMD ROCm | 🔄 | 🔄 | ❌ | In development |

✅ = Supported | 🔄 = In development | ❌ = Not supported
