# TensorRT-LLM Environment Setup

This guide details how to set up the Conda environment for TensorRT-LLM, ensuring all dependencies (including MPI and ONNX) are correctly installed and linked.

## 1. Create the Conda Environment
Use the provided `tensorrt_cuda.yml` file to create the environment. This file handles Python 3.10 Downgrade, ONNX version pinning, and MPI dependencies.

```bash
conda env create -f environment_files/tensorrt_cuda.yml
conda activate pita_tensorrt_cuda
```

## 2. Install Project in Editable Mode
Install the current project in editable mode so tests can import the `pita` module.

```bash
pip install -e .
```

## 3. Running Tests
You can run the tests directly:

```bash
pytest tests/inference/base_autoregressive_sampler/test_AutoregressiveSampler_class_tensorrt.py
```

## Troubleshooting
- **AttributeError: module 'onnx.helper' has no attribute 'float32_to_bfloat16'**: Ensure you are using `onnx<1.16.0`.
- **ImportError: libmpi.so.40**: Ensure `openmpi` is installed via conda (`conda list openmpi`).
