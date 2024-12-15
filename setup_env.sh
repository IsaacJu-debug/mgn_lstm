#!/bin/bash

# Setup Script for MGN-LSTM Environment
# This script creates a conda environment with PyTorch, PyTorch Geometric, and other 
# dependencies for reproducing MGN-LSTM
#
# Prerequisites:
# - Anaconda or Miniconda installed
#
# Usage:
# ./setup_env.sh [target_directory]

# Default configurable parameters
VIRTUAL_NAME="mgn_lstm"
PYTORCH_VERSION="2.0.0"
CUDA_VERSION="11.8"
PYTHON_VERSION="3.10"

# Get target directory from command line argument or use current directory
TARGET_ROOT=${1:-$(pwd)}
ENV_PATH="$TARGET_ROOT/$VIRTUAL_NAME"

# Print setup information
echo "Setting up environment with following configuration:"
echo "Environment name: $VIRTUAL_NAME"
echo "Python version: $PYTHON_VERSION"
echo "PyTorch version: $PYTORCH_VERSION"
echo "CUDA version: $CUDA_VERSION"
echo "Installation path: $ENV_PATH"

# Create conda environment
echo "Creating conda environment..."
conda create --prefix $ENV_PATH --yes python=$PYTHON_VERSION

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_PATH

# Install base packages
echo "Installing base packages..."
conda install -y numpy jupyterlab nodejs
pip install --upgrade pip setuptools
pip install dask h5py lxml pandas obspy seaborn pixiedust

# Install PyTorch ecosystem
echo "Installing PyTorch ecosystem..."
conda install -y pytorch==$PYTORCH_VERSION torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
pip install torchsummary

# Install PyTorch Geometric
echo "Installing PyTorch Geometric and dependencies..."
conda install -y pyg -c pyg
pip install torch-geometric-temporal

# Install additional packages
echo "Installing additional packages..."
pip install Pillow wandb

# Create activation script
ACTIVATE_SCRIPT="$TARGET_ROOT/activate_${VIRTUAL_NAME}.sh"
echo "Creating activation script at $ACTIVATE_SCRIPT"
echo '#!/bin/bash' > $ACTIVATE_SCRIPT
echo "conda activate $ENV_PATH" >> $ACTIVATE_SCRIPT
chmod +x $ACTIVATE_SCRIPT

# Print final instructions
echo "
Setup completed successfully!

To activate the environment, run:
source $ACTIVATE_SCRIPT

To verify installation, run:
python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\")'
python -c 'import torch_geometric; print(f\"PyG version: {torch_geometric.__version__}\")'
"