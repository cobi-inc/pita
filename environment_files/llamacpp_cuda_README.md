## LLaMA.cpp CUDA Environment Setup

### Quick Installation

```bash
# Create the environment
conda create -n pita_llama_cpp python=3.12 -y
conda activate pita_llama_cpp

# Install pita package (includes all dependencies: torch, transformers, etc.)
pip install -e .

# Install llama-cpp-python
pip install llama-cpp-python

# Install pytest for testing
pip install pytest
```

### Alternative: Using Environment File

```bash
conda env create -f environment_files/pita_llama_cpp.yml
conda activate pita_llama_cpp
```

### Verify Installation

```bash
python -c "import torch; print(f'Torch {torch.__version__} CUDA: {torch.cuda.is_available()}')"
python -c "import llama_cpp; print(f'llama_cpp version: {llama_cpp.__version__}')"
```

### Run Tests

```bash
# Unit tests for test_time_coding
pytest tests/api/test_test_time_coding.py -v

# Integration tests with llama_cpp
pytest tests/api/test_api_integration_llama_cpp.py -v
```

### Notes

- This environment uses CUDA 12.8 libraries (from pita's dependencies)
- llama-cpp-python 0.3.16+ is recommended for compatibility
- The pita package pins specific NVIDIA library versions to ensure compatibility