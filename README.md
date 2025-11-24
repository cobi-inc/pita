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

and run files within the directory with the following format

```bash
python -m pits.*
```

# Software Hierarchy

```
power_sampling.yml 
pyproject.toml
README.md
requirements.txt
grading_utils/
    math/
        eval_math.py
        math_grader.py
        math_normalize.py
        parse_utils.py
src/
    api_test.py
    models.txt
    api/
        api_template.py
        serve_power_sampling.py
    power_sampling/
        power_sample.py
    training_free_reasoning.egg-info/
        dependency_links.txt
        PKG-INFO
        requires.txt
        SOURCES.txt
        top_level.txt
    utils/
        benchmarking_utils.py
        parse_utils.py
```
