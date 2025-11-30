## Installation
conda env create -f environment_files/power_sampling_llamacpp_cuda.yml
conda activate power_sampling_llamacpp_cuda

# Set up CUDA library path (required once per environment)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Reactivate to apply
conda deactivate && conda activate power_sampling_llamacpp_cuda