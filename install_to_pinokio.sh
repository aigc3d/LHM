#!/bin/bash

# Script to install LHM ComfyUI node to Pinokio's ComfyUI installation
# Usage: ./install_to_pinokio.sh [PINOKIO_DIR]

# Default Pinokio directory
PINOKIO_DIR="${1:-$HOME/pinokio/api/comfy.git/app}"

# Source directory (current project)
SOURCE_DIR="$(pwd)/comfy_lhm_node"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist."
    exit 1
fi

# Check if Pinokio ComfyUI directory exists
if [ ! -d "$PINOKIO_DIR" ]; then
    echo "Error: Pinokio ComfyUI directory $PINOKIO_DIR does not exist."
    echo "Usage: ./install_to_pinokio.sh [path/to/pinokio/comfy/installation]"
    exit 1
fi

# Create custom_nodes directory if it doesn't exist
CUSTOM_NODES_DIR="$PINOKIO_DIR/custom_nodes"
mkdir -p "$CUSTOM_NODES_DIR"

# Create the LHM node directory
TARGET_DIR="$CUSTOM_NODES_DIR/lhm_node"
mkdir -p "$TARGET_DIR"

# Copy all files from comfy_lhm_node to the target directory
echo "Copying files from $SOURCE_DIR to $TARGET_DIR..."
cp -R "$SOURCE_DIR"/* "$TARGET_DIR"

# Install requirements if requirements.txt exists
if [ -f "$SOURCE_DIR/requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r "$SOURCE_DIR/requirements.txt"
fi

# Create a symbolic link or add the main LHM directory to PYTHONPATH
# This is needed because the module imports from the parent directory
echo "Setting up Python path for LHM..."
PYTHON_PATH_FILE="$PINOKIO_DIR/python_path.txt"
LHM_DIR="$(dirname "$(pwd)")"

# Check if we already added this path
if [ -f "$PYTHON_PATH_FILE" ]; then
    if ! grep -q "$LHM_DIR" "$PYTHON_PATH_FILE"; then
        echo "$LHM_DIR" >> "$PYTHON_PATH_FILE"
    fi
else
    echo "$LHM_DIR" > "$PYTHON_PATH_FILE"
fi

# Create a startup script to set PYTHONPATH before ComfyUI starts
STARTUP_SCRIPT="$PINOKIO_DIR/custom_nodes/set_pythonpath.py"
cat > "$STARTUP_SCRIPT" << EOF
import os
import sys

# Add LHM directory to Python path
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "python_path.txt"), "r") as f:
    paths = f.read().splitlines()
    for path in paths:
        if path and path not in sys.path:
            sys.path.append(path)
            print(f"Added {path} to Python path")
EOF

# Ensure the model directory exists in Pinokio
MODEL_DIR="$PINOKIO_DIR/models"
mkdir -p "$MODEL_DIR/checkpoints"

echo ""
echo "==================== INSTALLATION COMPLETED ===================="
echo "LHM node has been installed to Pinokio's ComfyUI at: $TARGET_DIR"
echo ""
echo "IMPORTANT: You need to restart ComfyUI in Pinokio for changes to take effect."
echo ""
echo "If your models are not found, copy or symlink model weights to:"
echo "$MODEL_DIR/checkpoints/"
echo ""
echo "You can also create a symbolic link to your existing model weights:"
echo "ln -s $(pwd)/checkpoints/* $MODEL_DIR/checkpoints/"
echo ""
echo "==============================================================" 