# Installation

## Requirements

The `pita` library (Test-Time Training-Free Reasoning) has specific requirements to run effectively, particularly due to its reliance on GPU acceleration and backend inference engines like vLLM.

- **Operating System**: Linux is recommended.
- **Python**: Python 3.8 or higher.
- **Hardware**: GPU support is strongly recommended for inference backends.

The specific Python library requirements are listed in `requirements.txt`.

## Conda Environment Setup

It is highly recommended to use a Conda environment to manage dependencies and avoid conflicts. You can create the environment using the provided `power_sampling.yml` file.

```bash
conda env create -f power_sampling.yml
conda activate power_sampling
```

## Python Package Installation

Once your environment is active, you can install the required Python packages. We recommend using `uv` for faster installation, but standard `pip` works as well.

### Using `uv` (Recommended)

```bash
uv pip install -r requirements.txt
```

### Using `pip`

```bash
pip install -r requirements.txt
```

## Installing `pita`

To install the `pita` library itself, install it in editable mode. This ensures that any changes you make to the code are immediately reflected without needing to reinstall.

```bash
pip install -e .
```

## Verification

To verify that the installation was successful, you can try importing the library in a Python shell:

```python
import pita
print("Pita library installed successfully!")
```
