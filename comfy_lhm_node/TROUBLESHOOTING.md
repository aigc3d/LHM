# LHM Node for ComfyUI - Troubleshooting Guide

This guide provides solutions for common issues encountered when installing and using the LHM (Large Animatable Human Model) node in ComfyUI.

## Understanding the Modular Architecture

The LHM node has been designed with a modular architecture that accommodates various installation scenarios:

### Full vs Simplified Implementation

1. **Full Implementation:**
   - Located in `full_implementation.py`
   - Provides complete functionality with 3D reconstruction and animation
   - Requires all dependencies like `pytorch3d`, `roma`, and the full LHM codebase
   - Automatically used when all dependencies are available

2. **Simplified Implementation:**
   - Built into `__init__.py` as a fallback
   - Provides basic functionality without requiring complex dependencies
   - Returns the input image and a simulated animation sequence
   - Automatically activated when dependencies for full implementation are missing

The system automatically detects which dependencies are available and selects the appropriate implementation:
- When you first start ComfyUI, the node attempts to import the full implementation
- If any required dependencies are missing, it gracefully falls back to the simplified implementation
- You can check which implementation is active in the ComfyUI logs

## Installation Guide for Pinokio

### Prerequisites
- Pinokio with ComfyUI installed
- LHM repository cloned to your computer

### Step-by-Step Installation

1. **Use the automated installation scripts**

   The easiest way to install is using one of the provided scripts:
   
   For Python users:
   ```bash
   cd ~/Desktop/LHM/comfy_lhm_node
   chmod +x install_dependencies.py
   ./install_dependencies.py
   ```
   
   For bash users:
   ```bash
   cd ~/Desktop/LHM/comfy_lhm_node
   chmod +x install_dependencies.sh
   ./install_dependencies.sh
   ```

   These scripts will:
   - Find your Pinokio ComfyUI installation
   - Install required dependencies
   - Create symbolic links to LHM code and model weights
   - Set up the necessary directory structure

2. **Manual installation steps (if automated scripts fail)**

   If the automated scripts don't work for your setup, follow these manual steps:

   **Locate your Pinokio ComfyUI installation directory**
   ```bash
   # Typically at one of these locations
   ~/pinokio/api/comfy.git/app
   ```

   **Create the custom_nodes directory if it doesn't exist**
   ```bash
   mkdir -p ~/pinokio/api/comfy.git/app/custom_nodes/lhm_node
   ```

   **Copy the LHM node files**
   ```bash
   cp -r ~/path/to/your/LHM/comfy_lhm_node/* ~/pinokio/api/comfy.git/app/custom_nodes/lhm_node/
   ```

   **Create symbolic links to the core LHM code**
   ```bash
   cd ~/pinokio/api/comfy.git/app
   ln -s ~/path/to/your/LHM/LHM .
   ln -s ~/path/to/your/LHM/engine .
   ln -s ~/path/to/your/LHM/configs .
   ```

3. **Install required Python dependencies**

   ```bash
   # Activate the Pinokio Python environment
   source ~/pinokio/api/comfy.git/app/env/bin/activate
   
   # Or use the full Python path if pip is not in your PATH
   ~/pinokio/api/comfy.git/app/env/bin/python -m pip install omegaconf rembg opencv-python scikit-image matplotlib
   
   # On Apple Silicon Macs, install onnxruntime-silicon
   ~/pinokio/api/comfy.git/app/env/bin/python -m pip install onnxruntime-silicon
   
   # On other systems, use the standard onnxruntime
   ~/pinokio/api/comfy.git/app/env/bin/python -m pip install onnxruntime
   
   # For full functionality, install roma
   ~/pinokio/api/comfy.git/app/env/bin/python -m pip install roma
   
   # pytorch3d is optional but recommended (complex installation)
   # See the pytorch3d-specific instructions below if needed
   ```

4. **Download model weights (if not already downloaded)**

   ```bash
   cd ~/path/to/your/LHM
   chmod +x download_weights.sh
   ./download_weights.sh
   ```

   Note: This will download approximately 18GB of model weights.

5. **Restart ComfyUI in Pinokio**
   - Go to the Pinokio dashboard
   - Click the trash icon to stop ComfyUI
   - Click on ComfyUI to start it again

## How the Modular Implementation Works

The LHM node is designed to work at different capability levels depending on what dependencies are available:

### 1. Import Path Resolution

The `lhm_import_fix.py` module handles Python path issues by:
- Searching for the LHM project in common locations
- Adding the relevant directories to the Python path
- Supporting multiple installation methods (direct installation, symbolic links, etc.)

### 2. Progressive Dependency Loading

When ComfyUI loads the node, this process occurs:
1. Basic dependencies are checked (torch, numpy, etc.)
2. Advanced dependencies are attempted (pytorch3d, roma, etc.)
3. The appropriate implementation is selected:
   - If all dependencies are available: Full implementation is used
   - If any dependencies are missing: Simplified implementation is used

### 3. Node Registration

Two nodes are available based on the dependency situation:
- **LHM Human Reconstruction**: Always available, with functionality level based on dependencies
- **LHM Test Node**: Available in simplified mode, helps verify basic functionality

## Common Issues and Solutions

### Issue: Node doesn't appear in ComfyUI
**Solution:**
- Check ComfyUI logs for import errors
- Verify if node is using simplified implementation
- Install missing dependencies

### Issue: "ModuleNotFoundError: No module named 'pytorch3d'"
**Solution:**
- This complex dependency is optional. Without it, the simplified implementation will be used
- For Apple Silicon Macs:
  ```bash
  MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python -m pip install pytorch3d
  ```
- For other systems, see the [pytorch3d installation documentation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

### Issue: "ModuleNotFoundError: No module named 'roma'"
**Solution:**
- Install roma:
  ```bash
  python -m pip install roma
  ```
- Without this, the simplified implementation will be used

### Issue: "ModuleNotFoundError: No module named 'onnxruntime'"
**Solution:**
- Install the correct onnxruntime for your system:
  ```bash
  # For Apple Silicon Macs (M1/M2/M3)
  python -m pip install onnxruntime-silicon
  
  # For other systems
  python -m pip install onnxruntime
  ```

### Issue: Model weights not found
**Solution:**
- Ensure you've run the download_weights.sh script
- If the script fails, manually download the weights
- Create symbolic links to the weights:
  ```bash
  ln -s ~/path/to/your/LHM/checkpoints/*.pth ~/pinokio/api/comfy.git/app/models/checkpoints/
  ```

### Issue: "pip: command not found" or similar errors
**Solution:**
- Use the full path to the Python interpreter:
  ```bash
  ~/pinokio/api/comfy.git/app/env/bin/python -m pip install package_name
  ```
- Alternatively, activate the virtual environment first:
  ```bash
  source ~/pinokio/api/comfy.git/app/env/bin/activate
  ```

## Testing the Installation

To verify your installation, follow these steps:

1. **Check which implementation is active**
   Open the ComfyUI logs and look for one of these messages:
   - "Successfully loaded full LHM implementation" (full functionality available)
   - "Using simplified implementation - some functionality will be limited" (fallback mode active)

2. **Use the LHM Test Node**
   - Add the "LHM Test Node" to your workflow
   - Connect an image source to it
   - Choose the "Add Border" option to verify processing
   - Run the workflow - a green border should appear around the image

3. **Use the LHM Human Reconstruction Node**
   - Connect an image source to the LHM Human Reconstruction node
   - Run the workflow
   - In simplified mode, you'll get a basic animation output
   - In full mode, you'll get proper 3D reconstruction and animation

## Working Towards Full Functionality

To enable full functionality if the simplified implementation is active:

1. **Check which dependencies are missing**
   Look at the ComfyUI logs for specific import errors

2. **Install all required dependencies**:
   ```bash
   ~/pinokio/api/comfy.git/app/env/bin/python -m pip install omegaconf rembg opencv-python scikit-image matplotlib roma
   ```

3. **Install pytorch3d** (if needed):
   ```bash
   # For macOS with Apple Silicon:
   MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ ~/pinokio/api/comfy.git/app/env/bin/python -m pip install pytorch3d
   ```

4. **Ensure symbolic links are correct**:
   ```bash
   cd ~/pinokio/api/comfy.git/app
   ln -sf ~/path/to/your/LHM/LHM .
   ln -sf ~/path/to/your/LHM/engine .
   ln -sf ~/path/to/your/LHM/configs .
   ```

5. **Restart ComfyUI** to reload the node with full functionality.

## Log File Locations

If you need to check logs for errors:
- ComfyUI logs: `~/pinokio/api/comfy.git/app/user/comfyui.log`
- Pinokio logs: Check the Pinokio dashboard for log options

To check specific errors in the logs:
```bash
cd ~/pinokio/api/comfy.git/app
cat user/comfyui.log | grep -i error
# Or view the last 100 lines
cat user/comfyui.log | tail -n 100
```

## Reporting Issues

If you encounter issues not covered in this guide, please create an issue on the GitHub repository with:
- A clear description of the problem
- Steps to reproduce the issue
- Any relevant log files or error messages 