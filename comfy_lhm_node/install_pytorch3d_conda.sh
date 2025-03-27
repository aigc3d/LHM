#!/bin/bash

echo "Installing PyTorch3D using conda..."

# Detect conda path in Pinokio
CONDA_PATH=$(find ~/pinokio -name conda -type f 2>/dev/null | head -n 1)

if [ -z "$CONDA_PATH" ]; then
    echo "Error: Could not find conda in Pinokio automatically."
    echo "Please enter the path to your conda executable:"
    read CONDA_PATH
    
    if [ ! -f "$CONDA_PATH" ]; then
        echo "Error: The path $CONDA_PATH does not exist"
        exit 1
    fi
fi

echo "Using Conda at: $CONDA_PATH"

# Use the environment Python is installed in
PYTHON_PATH=$(find ~/pinokio/bin/miniconda/bin -name python3.10 -type f 2>/dev/null | head -n 1)
if [ -z "$PYTHON_PATH" ]; then
    echo "Could not find Python in Pinokio miniconda. Trying to locate Python..."
    PYTHON_PATH=$(find ~/pinokio -name python3.10 -type f 2>/dev/null | head -n 1)
    
    if [ -z "$PYTHON_PATH" ]; then
        echo "Error: Could not find Python. Please install manually."
        exit 1
    fi
fi

# Get the conda environment from Python path
CONDA_ENV_PATH=$(dirname "$PYTHON_PATH")
CONDA_ENV=$(basename $(dirname "$CONDA_ENV_PATH"))

echo "Using Python at: $PYTHON_PATH"
echo "Conda environment: $CONDA_ENV"

# Make a temporary directory for the log files
TEMP_DIR=$(mktemp -d)
LOG_FILE="$TEMP_DIR/conda_install.log"

# Function to run conda commands and handle errors
run_conda_command() {
    echo "Running: $1"
    eval "$1" > "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "Error executing command: $1"
        echo "See log for details:"
        cat "$LOG_FILE"
        echo "Continuing with installation..."
    fi
}

# Check conda-forge channel is in config
run_conda_command "$CONDA_PATH config --show channels"
run_conda_command "$CONDA_PATH config --add channels conda-forge"
run_conda_command "$CONDA_PATH config --set channel_priority flexible"

# Install dependencies first
echo "Installing dependencies..."
run_conda_command "$CONDA_PATH install -y -n base fvcore iopath"

# Try to install PyTorch3D from conda-forge
echo "Installing PyTorch3D..."
run_conda_command "$CONDA_PATH install -y -n base pytorch3d"

# Install PyTorch with MPS support if needed
echo "Updating PyTorch with MPS support..."
run_conda_command "$CONDA_PATH install -y -n base 'pytorch>=2.0.0' 'torchvision>=0.15.0'"

# Install roma 
echo "Installing roma..."
run_conda_command "$CONDA_PATH install -y -n base roma"

# Create our fallback fix for PyTorch3D
echo "Setting up the PyTorch3D compatibility layer..."
LHM_PATH=$(dirname $(realpath "$0"))
FIX_PATH="$LHM_PATH/pytorch3d_lite_fix.py"

cat > "$FIX_PATH" << 'EOL'
# PyTorch3D compatibility layer
import sys
import os

# Try to import the real PyTorch3D
try:
    import pytorch3d
    print("Using conda-installed PyTorch3D")
except ImportError:
    # If real PyTorch3D isn't available, try our custom implementation
    try:
        # First try to import from local module
        from pytorch3d_lite import (
            matrix_to_rotation_6d,
            rotation_6d_to_matrix,
            axis_angle_to_matrix,
            matrix_to_axis_angle,
        )
        
        # Create namespace for pytorch3d
        if 'pytorch3d' not in sys.modules:
            import types
            pytorch3d = types.ModuleType('pytorch3d')
            sys.modules['pytorch3d'] = pytorch3d
            
            # Create submodules
            pytorch3d.transforms = types.ModuleType('pytorch3d.transforms')
            sys.modules['pytorch3d.transforms'] = pytorch3d.transforms
            
            # Map functions to pytorch3d namespace
            pytorch3d.transforms.matrix_to_rotation_6d = matrix_to_rotation_6d
            pytorch3d.transforms.rotation_6d_to_matrix = rotation_6d_to_matrix
            pytorch3d.transforms.axis_angle_to_matrix = axis_angle_to_matrix
            pytorch3d.transforms.matrix_to_axis_angle = matrix_to_axis_angle
            
            print("Using PyTorch3D-Lite as fallback")
    except ImportError:
        print("Warning: Neither PyTorch3D nor PyTorch3D-Lite could be loaded. Some features may not work.")

print("PyTorch3D compatibility layer initialized")
EOL

# Update lhm_import_fix.py to use the compatibility layer
FIX_IMPORT_PATH="$LHM_PATH/lhm_import_fix.py"

cat > "$FIX_IMPORT_PATH" << 'EOL'
# LHM import fix for Pinokio
import sys
import os

# Add this directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add the LHM core to the Python path if needed
LHM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../LHM')
if os.path.exists(LHM_PATH) and LHM_PATH not in sys.path:
    sys.path.append(LHM_PATH)

# Load the PyTorch3D compatibility layer
try:
    from pytorch3d_lite_fix import *
    print("PyTorch3D compatibility layer loaded")
except ImportError:
    print("Warning: PyTorch3D compatibility layer not found. Some features may not work.")
EOL

# Clean up
rm -rf "$TEMP_DIR"

echo "Installation complete!"
echo "Please restart ComfyUI to load PyTorch3D and the full LHM node functionality." 