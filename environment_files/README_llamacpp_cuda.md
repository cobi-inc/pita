# LLaMA.cpp CUDA Environment Setup

This guide explains how to set up llama-cpp-python with CUDA support for GPU acceleration.

## Quick Setup (Recommended)

Run the automated setup script:

```bash
cd environment_files
./setup_llamacpp_cuda.sh
```

This script will:
1. Create the `pita_llamacpp_cuda` conda environment
2. Build llama-cpp-python from source with CUDA support
3. Verify the CUDA backend is installed

## Manual Installation

If you prefer manual setup:

### Step 1: Create the Conda Environment

```bash
conda env create -f environment_files/power_sampling_llamacpp_cuda.yml
conda activate pita_llamacpp_cuda
```

### Step 2: Build llama-cpp-python with CUDA

```bash
# Set up environment variables for CUDA build
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Build and install with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler" \
  pip install llama-cpp-python --no-cache-dir
```

### Step 3: Verify Installation

```bash
python -c "
import os
lib_dir = os.path.dirname(__import__('llama_cpp').__file__) + '/lib'
libs = [l for l in os.listdir(lib_dir) if 'ggml' in l]
print('Available backends:', libs)
assert 'libggml-cuda.so' in libs, 'CUDA backend not found!'
print('SUCCESS: CUDA backend installed')
"
```

## Why Build from Source?

The pre-built CUDA wheels from llama-cpp-python's GitHub Pages index can be unreliable. Building from source ensures:

- **Reliability** - No dependency on external wheel hosting
- **Optimization** - Compiled for your specific GPU architecture
- **Compatibility** - Works with your exact CUDA driver/toolkit versions

## Environment Details

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| CUDA Toolkit | 12.4.1 |
| llama-cpp-python | 0.3.16+ |

## Run Tests

```bash
conda activate pita_llamacpp_cuda

# Unit tests
pytest tests/api/test_test_time_coding.py -v

# Integration tests with llama_cpp
pytest tests/api/test_api_integration_llama_cpp.py -v
```

## Troubleshooting

### "unsupported GNU version" Error
The `-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler` flag handles GCC 13+ compatibility with CUDA 12.4.

### "cuda_runtime.h: No such file" Error
Ensure `CPATH` includes the CUDA headers:
```bash
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
```

### Wheel Download Issues
If pre-built wheels fail, always fall back to building from source (the default approach in this setup).