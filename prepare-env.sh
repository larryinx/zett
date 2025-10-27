#!/bin/bash
# prepare a conda environment for developing hypertok

_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

function prepare_conda_env() {
        # the python version to use
        local python_version=${1:-3.11}; shift
        # the conda env name
        local env_name=${1:-hypertok}; shift

        echo ">>> Preparing conda environment \"${env_name}\", python_version=${python_version}"
        
        # Preparation
        set -e
        eval "$(conda shell.bash hook)"
        conda env remove --name $env_name -y || true
        conda create --name $env_name python=$python_version pip -y
        conda activate $env_name
        pip install --upgrade pip

        # Install libraries
        conda install h5py=3.8.0 -c conda-forge -y
        conda install transformers=4.45.2 -c conda-forge -y
        conda install accelerate=0.23.0 -c conda-forge -y
        conda install wandb=0.15.4 -c conda-forge -y
        conda install optax=0.1.5 -c conda-forge -y
        conda install flax=0.8.0 -c conda-forge -y
        conda install maturin=1.3.0 -c conda-forge -y
        conda install pandas=2.0.3 -c conda-forge -y
        conda install pyahocorasick=2.0.0 -c conda-forge -y
        conda install matplotlib=3.7.2 -c conda-forge -y
        conda install scikit-learn=1.4.2 -c conda-forge -y
        conda install datasets=2.18.0 -c conda-forge -y
        conda install jax=0.4.23 -c conda-forge -y
        CONDA_OVERRIDE_CUDA="12.2" conda install -c conda-forge "jaxlib=0.4.23=cuda120*" -y
        conda install conda-forge::rust -y
        conda install scipy=1.12.0 -c conda-forge -y
        conda install compilers -c conda-forge -y
        conda install patchelf -c conda-forge -y
        cd rust_utils && maturin develop --release

        # pip install -e .[dev]
}


prepare_conda_env "$@"