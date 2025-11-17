# training-free-reasoning
post-training inference without RL

## Requirements
The python library requirements are in requirements.txt
Create a conda environment to manage the packages
```bash
conda env create -f power_sampling.yml
```

## Python Installation
Use the following command to install all the python requirements:
```bash
uv pip install -r requirements.txt
```

Install training-free-reasoning as an editable library to make sure the depnedcies are correct using:
```bash
pip install -e .
```

and

```bash
python -m src.*
```