#!/bin/bash

# Configurable parameters
VIRTUAL_NAME=mgn_lstm
TARGET_ROOT=/home/groups/tchelepi/ju1/02_dl_modeling/04_research_projects/01_gnn_flow/03_github_version/env
PYTHON_VERSION=3.10
PYTORCH_VERSION=2.2.0
CUDA_VERSION=11.8
TORCHVISION_VERSION=0.17.0
TORCHAUDIO_VERSION=2.2.0
NUMPY_VERSION=1.24.3  
# Error handling function
handle_error() {
    echo "Error: $1"
    exit 1
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    handle_error "conda is not installed or not in PATH"
fi

# Show environment information
echo "Setting up virtual python environment..."
echo "Using conda from: $(which conda)"
echo "Conda version: $(conda --version)"
echo "Active conda env: $CONDA_DEFAULT_ENV"
echo "-----------------------"

# Check if environment exists
if [ -d "$TARGET_ROOT/$VIRTUAL_NAME" ]; then
    echo "Found existing environment at: $TARGET_ROOT/$VIRTUAL_NAME"
    read -p "Do you want to delete this environment? (y/n): " confirm
    
    # Convert answer to lowercase
    confirm=${confirm,,}
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "yes" ]; then
        echo "Removing existing environment..."
        conda env remove --prefix $TARGET_ROOT/$VIRTUAL_NAME
    else
        echo "Keeping existing environment. Exiting..."
        exit 1
    fi
fi

# Create virtual environment
echo "Creating new environment..."
conda create -y --prefix $TARGET_ROOT/$VIRTUAL_NAME python=$PYTHON_VERSION || handle_error "Failed to create environment"

# Create and update activation script
ACTIVATE_SCRIPT="$TARGET_ROOT/start_${VIRTUAL_NAME}.sh"
echo "Creating activation script: $ACTIVATE_SCRIPT"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
source activate $TARGET_ROOT/$VIRTUAL_NAME

# Set up PYTHONPATH correctly
export PYTHONPATH="$TARGET_ROOT/$VIRTUAL_NAME/site-packages:\$PYTHONPATH"
EOF

chmod +x "$ACTIVATE_SCRIPT"

# Activate the environment
echo "Activating environment..."
source "$ACTIVATE_SCRIPT" || handle_error "Failed to activate environment"

# Install dependencies using conda first
echo "Installing NumPy..."
conda install -y numpy=$NUMPY_VERSION || handle_error "Failed to install NumPy"

echo "Installing conda packages..."
conda install -y -q \
    pandas \
    jupyterlab \
    nodejs \
    h5py \
    seaborn || handle_error "Failed to install conda packages"

# Install PyTorch with specific versions
echo "Installing PyTorch ecosystem..."
conda install -y \
    pytorch=$PYTORCH_VERSION \
    torchvision=$TORCHVISION_VERSION \
    torchaudio=$TORCHAUDIO_VERSION \
    pytorch-cuda=$CUDA_VERSION \
    -c pytorch -c nvidia || handle_error "Failed to install PyTorch"

# Install PyTorch Geometric and related libraries

echo "Installing PyTorch Geometric Libs..."
pip install torch-geometric==2.4.0 --no-deps

# we need to install from wheel that are compiled with your system
pip install --no-deps  \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cu${CUDA_VERSION/./}.html || handle_error "Failed to install PyG core libraries"


pip install -q torch-geometric-temporal

# Verify critical installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"

echo "Environment setup completed successfully!"
echo "To activate the environment, run: source $ACTIVATE_SCRIPT"