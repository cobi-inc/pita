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

## 3. Running Tests (Crucial: LD_PRELOAD)
Due to a symbol mismatch between the `torch` package (from pip) and the system `libcudart`, you **must** preload the correct CUDA runtime library when running usage scripts or tests.

Run the following command to export the variable (you can add this to your `.bashrc` or activation script):

```bash
export LD_PRELOAD=$(dirname $(dirname $(which python)))/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
```

Then run your tests:
```bash
pytest tests/inference/base_autoregressive_sampler/test_AutoregressiveSampler_class_tensorrt.py
```

## Troubleshooting
- **AttributeError: module 'onnx.helper' has no attribute 'float32_to_bfloat16'**: Ensure you are using `onnx<1.16.0`.
- **ImportError: libmpi.so.40**: Ensure `openmpi` is installed via conda (`conda list openmpi`).
- **undefined symbol: cudaGetDriverEntryPointByVersion**: You forgot the `LD_PRELOAD` step above.
