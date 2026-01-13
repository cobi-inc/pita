# LLaMA.cpp Windows CPU Environment Setup

This guide explains how to set up `pita` with `llama-cpp-python` on Windows using CPU (no GPU required).

## Prerequisites
- **Miniconda** or **Anaconda** installed on Windows.

## Installation Steps

### 1. Create the Conda Environment
Open **Anaconda Powershell Prompt** and run following commands from the project root:

```powershell
# Create environment from YAML file
conda env create -f environment_files/llamacpp_windows_cpu.yml

# Activate the environment
conda activate pita_llamacpp_windows_cpu
```

### 2. Install `pita` 
Install the `pita` library in editable mode with the required dependencies.

```powershell
# Ensure you are in the root 'pita' directory (where pyproject.toml is)
pip install -e ".[llama_cpp]"
```

## Verification

### 1. Verify Imports
Run the following Python one-liner to check if both `pita` and `llama_cpp` can be imported:

```powershell
python -c "import pita; import llama_cpp; print('Success: Imported pita and llama_cpp')"
```

### 2. Run Tests
Run the test suite to ensure everything is working correctly.

**Optional:** Install development dependencies first (required for running tests if you haven't already):
```powershell
pip install ".[dev]"
```

Run the tests:
```powershell
# Run all tests
pytest tests

# Run specific llama_cpp tests
pytest tests/inference/base_autoregressive_sampler/test_AutoregressiveSampler_class_llama_cpp.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'pita'"
Ensure you installed `pita` with `pip install -e .` while the environment was activated.

### Compilation Errors
Depending on your Windows setup, `llama-cpp-python` might try to compile from source if the conda-forge binary is not compatible.
- The `conda-forge` channel usually provides pre-built binaries for Windows CPU.
- If issues persist, verify you have **Visual Studio Build Tools** installed (C++ Desktop Development workload).
