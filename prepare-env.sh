#!/bin/bash
# prepare a conda environment for developing tokdrift

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
        conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
        conda install nvidia/label/cuda-12.4.0::cuda -y
        conda install transformers evaluate accelerate -c conda-forge -y
        conda install conda-forge::huggingface_hub -y
        conda install datasets=3.6.0 -c conda-forge -y
        conda install maturin -c conda-forge -y
        conda install scikit-learn h5py wandb optax flax pandas pyahocorasick matplotlib -c conda-forge -y
        conda install jax -c conda-forge -y
        conda install conda-forge::rust -y
        # pip install -e .[dev]
}


prepare_conda_env "$@"