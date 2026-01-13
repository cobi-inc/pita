# Automated llama.cpp CUDA Installation

This guide explains how to use the automated setup script to install `pita` with llama.cpp and NVIDIA CUDA support on Linux systems.

## Prerequisites

- **Linux OS**: This script is designed for Linux bash environments.
- **NVIDIA GPU**: A CUDA-capable GPU with driver installed.
- **Conda**: Miniconda or Anaconda installed and initialized.

## Usage

You can create the script locally by saving the following content as `setup_llamacpp_cuda.sh`:

```bash
#!/bin/bash
# Script to set up pita_llamacpp_cuda environment with CUDA support
# This builds llama-cpp-python from source to ensure CUDA compatibility

set -e  # Exit on any error

ENV_NAME="pita_llamacpp_cuda"

# 1. Create a temporary YAML for the base environment
cat <<EOF > llamacpp_cuda.yml
name: \$ENV_NAME
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
EOF

echo "=========================================="
echo "Setting up \$ENV_NAME environment"
echo "=========================================="

# Check if environment already exists
if conda info --envs | grep -q "^\$ENV_NAME "; then
    echo "Environment \$ENV_NAME already exists."
    read -p "Do you want to remove and recreate it? (y/n): " choice
    if [[ "\$choice" == "y" || "\$choice" == "Y" ]]; then
        echo "Removing existing environment..."
        conda env remove -n "\$ENV_NAME" -y
    else
        echo "Keeping existing environment. Will attempt to install llama-cpp-python..."
    fi
fi

# Create environment if it doesn't exist
if ! conda info --envs | grep -q "^\$ENV_NAME "; then
    echo "Creating conda environment from llamacpp_cuda.yml..."
    conda env create -f llamacpp_cuda.yml
fi

echo ""
echo "=========================================="
echo "Installing llama-cpp-python with CUDA"
echo "=========================================="

# Get the conda prefix for this environment
CONDA_PREFIX_PATH=\$(conda info --envs | grep "^\$ENV_NAME " | awk '{print \$NF}')

if [[ -z "\$CONDA_PREFIX_PATH" ]]; then
    CONDA_PREFIX_PATH=\$(conda info --envs | grep "^\$ENV_NAME$" | awk '{print \$NF}')
fi

echo "Environment path: \$CONDA_PREFIX_PATH"

# Set up environment variables for CUDA build
export CUDACXX="\$CONDA_PREFIX_PATH/bin/nvcc"
export CPATH="\$CONDA_PREFIX_PATH/targets/x86_64-linux/include:\$CPATH"
export LD_LIBRARY_PATH="\$CONDA_PREFIX_PATH/lib:\$LD_LIBRARY_PATH"
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"

echo "CUDACXX: \$CUDACXX"
echo "CMAKE_ARGS: \$CMAKE_ARGS"
echo ""

# Install llama-cpp-python with CUDA support
echo "Building and installing llama-cpp-python (this may take a few minutes)..."
conda run -n "\$ENV_NAME" bash -c "export CUDACXX='\$CONDA_PREFIX_PATH/bin/nvcc'; export CPATH='\$CONDA_PREFIX_PATH/targets/x86_64-linux/include:\$CPATH'; export LD_LIBRARY_PATH='\$CONDA_PREFIX_PATH/lib:\$LD_LIBRARY_PATH'; export CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler'; pip install llama-cpp-python --no-cache-dir"

# Verify installation
echo ""
echo "=========================================="
echo "Verifying CUDA backend installation"
echo "=========================================="

CUDA_CHECK=\$(conda run -n "\$ENV_NAME" python -c "
import os
lib_dir = os.path.dirname(__import__('llama_cpp').__file__) + '/lib'
libs = os.listdir(lib_dir)
cuda_libs = [l for l in libs if 'cuda' in l.lower()]
if cuda_libs:
    print('SUCCESS: CUDA backend found:', cuda_libs)
else:
    print('WARNING: No CUDA backend found. Available libs:', libs)
")

echo "\$CUDA_CHECK"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate \$ENV_NAME"
echo ""
```

### Execution Steps

1. **Save the script** as `setup_llamacpp_cuda.sh`.

2. **Make the script executable**:
   ```bash
   chmod +x setup_llamacpp_cuda.sh
   ```

3. **Run the script**:
   ```bash
   ./setup_llamacpp_cuda.sh
   ```

4. **Follow the prompts**:
   The script will check if an environment named `pita_llamacpp_cuda` already exists and ask if you want to recreate it.

## What the Script Does

The script performs the following steps automatically:

- **Environment Creation**: Creates a Conda environment named `pita_llamacpp_cuda` using the correct Python and dependency versions.
- **CUDA Build Setup**: Configures essential environment variables (`CUDACXX`, `CPATH`, `LD_LIBRARY_PATH`) to ensure `llama-cpp-python` can find your CUDA toolkit.
- **Compilation**: Builds `llama-cpp-python` from source with `-DGGML_CUDA=on` to enable GPU acceleration.
- **Verification**: Runs a small Python check to confirm that the CUDA backend (`libggml-cuda`) was correctly included in the build.

## Activating the Environment

Once the script completes successfully, activate your new environment:

```bash
conda activate pita_llamacpp_cuda
```

Then install `pita` in editable mode:

```bash
pip install -e .
```
