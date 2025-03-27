#!/bin/bash

echo "Installing PyTorch3D for Apple Silicon..."

# Set required environment variables for build
export MACOSX_DEPLOYMENT_TARGET=10.9
export CC=clang
export CXX=clang++

# Detect Pinokio ComfyUI path
PINOKIO_COMFY_PATH=$(find ~/pinokio -name "comfy.git" -type d 2>/dev/null | head -n 1)

if [ -z "$PINOKIO_COMFY_PATH" ]; then
    echo "Error: Could not find Pinokio ComfyUI path automatically"
    echo "Please enter the path to Pinokio ComfyUI installation (e.g. ~/pinokio/api/comfy.git/app):"
    read PINOKIO_COMFY_PATH
    
    if [ ! -d "$PINOKIO_COMFY_PATH" ]; then
        echo "Error: The path $PINOKIO_COMFY_PATH does not exist"
        exit 1
    fi
fi

# Set path to Python binary
PYTHON_BIN="$PINOKIO_COMFY_PATH/app/env/bin/python"
PIP_BIN="$PINOKIO_COMFY_PATH/app/env/bin/pip"

if [ ! -f "$PYTHON_BIN" ]; then
    # Try alternate location
    PYTHON_BIN="$PINOKIO_COMFY_PATH/env/bin/python"
    PIP_BIN="$PINOKIO_COMFY_PATH/env/bin/pip"
    
    if [ ! -f "$PYTHON_BIN" ]; then
        echo "Error: Python binary not found at expected location"
        echo "Trying to find Python in Pinokio..."
        
        PYTHON_BIN=$(find "$PINOKIO_COMFY_PATH" -name "python" -type f | grep -E "bin/python$" | head -n 1)
        PIP_BIN=$(find "$PINOKIO_COMFY_PATH" -name "pip" -type f | grep -E "bin/pip$" | head -n 1)
        
        if [ -z "$PYTHON_BIN" ]; then
            echo "Error: Could not find Python in Pinokio. Please install manually."
            exit 1
        else
            echo "Found Python at: $PYTHON_BIN"
            echo "Found Pip at: $PIP_BIN"
        fi
    fi
fi

echo "Using Python: $PYTHON_BIN"
echo "Using Pip: $PIP_BIN"

# Activate virtual environment if possible
if [ -f "${PYTHON_BIN%/*}/activate" ]; then
    echo "Activating virtual environment..."
    source "${PYTHON_BIN%/*}/activate"
fi

# Install dependencies first
echo "Installing dependencies..."
$PIP_BIN install --no-cache-dir fvcore iopath

# Install pre-requisites for PyTorch3D
echo "Installing PyTorch3D pre-requisites..."
$PIP_BIN install --no-cache-dir "pytorch3d-lite==0.1.1" ninja

# Install pytorch3d from source (specific version compatible with Apple Silicon)
echo "Installing PyTorch3D from source..."

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
echo "Working in temporary directory: $TEMP_DIR"
cd $TEMP_DIR

# Clone the repo at a specific commit that works well with Apple Silicon
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout 4e46dcfb2dd1c75ab1f6abf79a2e3e52fd8d427a

# Install PyTorch3D
$PIP_BIN install --no-deps -e .

# Install roma which is also needed for LHM
echo "Installing roma..."
$PIP_BIN install roma

echo "Installation complete!"
echo "Please restart ComfyUI to load PyTorch3D and the full LHM node functionality."

# Cleanup
cd ~
rm -rf $TEMP_DIR 