#!/bin/bash
# Script to set up pita_llamacpp_cuda environment with CUDA support
# This builds llama-cpp-python from source to ensure CUDA compatibility

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="pita_llamacpp_cuda"
ENV_FILE="$SCRIPT_DIR/power_sampling_llamacpp_cuda.yml"

echo "=========================================="
echo "Setting up $ENV_NAME environment"
echo "=========================================="

# Check if environment already exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists."
    read -p "Do you want to remove and recreate it? (y/n): " choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Keeping existing environment. Will attempt to install llama-cpp-python..."
    fi
fi

# Create environment if it doesn't exist
if ! conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Creating conda environment from $ENV_FILE..."
    conda env create -f "$ENV_FILE"
fi

echo ""
echo "=========================================="
echo "Installing llama-cpp-python with CUDA"
echo "=========================================="

# Get the conda prefix for this environment
CONDA_PREFIX_PATH=$(conda info --envs | grep "^$ENV_NAME " | awk '{print $NF}')

if [[ -z "$CONDA_PREFIX_PATH" ]]; then
    CONDA_PREFIX_PATH=$(conda info --envs | grep "^$ENV_NAME$" | awk '{print $NF}')
fi

echo "Environment path: $CONDA_PREFIX_PATH"

# Set up environment variables for CUDA build
export CUDACXX="$CONDA_PREFIX_PATH/bin/nvcc"
export CPATH="$CONDA_PREFIX_PATH/targets/x86_64-linux/include:$CPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX_PATH/lib:$LD_LIBRARY_PATH"
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"

echo "CUDACXX: $CUDACXX"
echo "CMAKE_ARGS: $CMAKE_ARGS"
echo ""

# Install llama-cpp-python with CUDA support
echo "Building and installing llama-cpp-python (this may take a few minutes)..."
conda run -n "$ENV_NAME" pip install llama-cpp-python --no-cache-dir

# Verify installation
echo ""
echo "=========================================="
echo "Verifying CUDA backend installation"
echo "=========================================="

CUDA_CHECK=$(conda run -n "$ENV_NAME" python -c "
import os
lib_dir = os.path.dirname(__import__('llama_cpp').__file__) + '/lib'
libs = os.listdir(lib_dir)
cuda_libs = [l for l in libs if 'cuda' in l.lower()]
if cuda_libs:
    print('SUCCESS: CUDA backend found:', cuda_libs)
else:
    print('WARNING: No CUDA backend found. Available libs:', libs)
")

echo "$CUDA_CHECK"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
