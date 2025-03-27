#!/bin/bash
# Script to install all required dependencies for the LHM node in Pinokio's ComfyUI environment

echo "Installing dependencies for LHM ComfyUI node..."

# Determine Pinokio ComfyUI location
PINOKIO_COMFY_PATH=$(find ~/pinokio -name "comfy.git" -type d 2>/dev/null | head -n 1)

if [ -z "$PINOKIO_COMFY_PATH" ]; then
    echo "Error: Could not find Pinokio ComfyUI path"
    echo "Please enter the path to Pinokio ComfyUI (e.g., ~/pinokio/api/comfy.git):"
    read PINOKIO_COMFY_PATH
fi

if [ ! -d "$PINOKIO_COMFY_PATH" ]; then
    echo "Error: The path $PINOKIO_COMFY_PATH does not exist"
    exit 1
fi

echo "Found Pinokio ComfyUI at: $PINOKIO_COMFY_PATH"

# Check if the virtual environment exists
if [ ! -d "$PINOKIO_COMFY_PATH/app/env" ]; then
    echo "Error: Python virtual environment not found at $PINOKIO_COMFY_PATH/app/env"
    exit 1
fi

# Activate the virtual environment
PYTHON_BIN="$PINOKIO_COMFY_PATH/app/env/bin/python"
PIP_BIN="$PINOKIO_COMFY_PATH/app/env/bin/pip"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python binary not found at $PYTHON_BIN"
    exit 1
fi

echo "Using Python at: $PYTHON_BIN"

# Install basic dependencies
echo "Installing basic dependencies..."
"$PYTHON_BIN" -m pip install omegaconf rembg opencv-python scikit-image matplotlib

# Install onnxruntime (platform-specific)
if [[ $(uname -p) == "arm" ]]; then
    echo "Detected Apple Silicon, installing onnxruntime-silicon..."
    "$PYTHON_BIN" -m pip install onnxruntime-silicon
else
    echo "Installing standard onnxruntime..."
    "$PYTHON_BIN" -m pip install onnxruntime
fi

# Install roma
echo "Installing roma..."
"$PYTHON_BIN" -m pip install roma

# Try to install pytorch3d
echo "Attempting to install pytorch3d (this may fail on some platforms)..."
if [[ $(uname -p) == "arm" ]]; then
    echo "Detected Apple Silicon, using macOS-specific installation..."
    MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ "$PYTHON_BIN" -m pip install --no-deps pytorch3d
    if [ $? -ne 0 ]; then
        echo "Warning: Could not install pytorch3d. Some functionality will be limited."
        echo "You may need to install pytorch3d manually following the instructions at:"
        echo "https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"
    fi
else
    "$PYTHON_BIN" -m pip install --no-deps pytorch3d
    if [ $? -ne 0 ]; then
        echo "Warning: Could not install pytorch3d. Some functionality will be limited."
        echo "You may need to install pytorch3d manually following the instructions at:"
        echo "https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"
    fi
fi

# Set up the LHM node
echo "Setting up LHM node in ComfyUI..."
LHM_PATH="/Users/danny/Desktop/LHM"
CUSTOM_NODES_PATH="$PINOKIO_COMFY_PATH/app/custom_nodes"

# Create custom_nodes directory if it doesn't exist
mkdir -p "$CUSTOM_NODES_PATH"

# Copy LHM node files
echo "Copying LHM node files to ComfyUI..."
mkdir -p "$CUSTOM_NODES_PATH/lhm_node"
cp -r "$LHM_PATH/comfy_lhm_node/"* "$CUSTOM_NODES_PATH/lhm_node/"

# Create symbolic links for LHM core code
echo "Creating symbolic links for LHM core code..."
cd "$PINOKIO_COMFY_PATH/app"
ln -sf "$LHM_PATH/LHM" .
ln -sf "$LHM_PATH/engine" .
ln -sf "$LHM_PATH/configs" .

# Create link for motion data if it exists
if [ -d "$LHM_PATH/train_data/motion_video" ]; then
    echo "Creating symbolic link for motion data..."
    mkdir -p "$PINOKIO_COMFY_PATH/app/train_data"
    ln -sf "$LHM_PATH/train_data/motion_video" "$PINOKIO_COMFY_PATH/app/train_data/"
fi

# Create link for model weights if they exist
if [ -d "$LHM_PATH/checkpoints" ]; then
    echo "Creating symbolic link for model weights..."
    mkdir -p "$PINOKIO_COMFY_PATH/app/models/checkpoints"
    for file in "$LHM_PATH/checkpoints/"*.pth; do
        if [ -f "$file" ]; then
            ln -sf "$file" "$PINOKIO_COMFY_PATH/app/models/checkpoints/$(basename "$file")"
        fi
    done
fi

echo "Installation complete!"
echo "Please restart ComfyUI in Pinokio to load the LHM node."
echo ""
echo "If you haven't downloaded the model weights yet, run:"
echo "cd $LHM_PATH && chmod +x download_weights.sh && ./download_weights.sh" 