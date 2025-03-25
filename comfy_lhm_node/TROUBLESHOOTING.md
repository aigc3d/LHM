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

- **Option 1 (Highly Recommended): Direct Installation from Source (Official Method):**
  
  Following the [official PyTorch3D installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), we've had success with:
  ```bash
  # First, ensure PyTorch and torchvision are properly installed with MPS support
  python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
  
  # Verify MPS support
  python -c "import torch; print(f'PyTorch: {torch.__version__}, MPS available: {torch.backends.mps.is_available()}')"
  
  # Install prerequisites
  python -m pip install fvcore iopath
  
  # For macOS with Apple Silicon (M1/M2/M3)
  MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python -m pip install -e "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  
  # Or clone and install from source
  git clone https://github.com/facebookresearch/pytorch3d.git
  cd pytorch3d
  MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python -m pip install -e .
  ```
  The key for Apple Silicon success is setting the environment variables `MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++`.

- **Option 2 (Reliable): Use conda to install PyTorch3D:**
  ```bash
  # Using the bash script
  cd ~/Desktop/LHM/comfy_lhm_node
  chmod +x install_pytorch3d_conda.sh
  ./install_pytorch3d_conda.sh
  
  # Or using the Python script
  cd ~/Desktop/LHM/comfy_lhm_node
  chmod +x install_pytorch3d_conda.py
  ./install_pytorch3d_conda.py
  ```
  This method handles complex dependencies better than pip.

- **Option 3: Use our specially optimized PyTorch MPS installation:**
  ```bash
  cd ~/Desktop/LHM/comfy_lhm_node
  chmod +x install_pytorch_mps.py
  ./install_pytorch_mps.py
  ```

- **Option 4: Use our specially optimized PyTorch3D installation scripts for Apple Silicon:**
  ```bash
  # Using the bash script
  cd ~/Desktop/LHM/comfy_lhm_node
  chmod +x install_pytorch3d_mac.sh
  ./install_pytorch3d_mac.sh
  
  # Or using the Python script
  cd ~/Desktop/LHM/comfy_lhm_node
  chmod +x install_pytorch3d_mac.py
  ./install_pytorch3d_mac.py
  ```
  
- **Option 5: Use PyTorch3D-Lite as an alternative (easier installation):**
  ```bash
  cd ~/Desktop/LHM/comfy_lhm_node
  chmod +x install_pytorch3d_lite.py
  ./install_pytorch3d_lite.py
  ```
  This will install a simplified version of PyTorch3D with fewer features, but it's much easier to install and works on most systems including Apple Silicon.

- **Option 6: Manual installation (advanced):**
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

## Special Instructions for Apple Silicon (M1/M2/M3) Macs

If you're using an Apple Silicon Mac (M1, M2, or M3), you may encounter specific challenges with PyTorch3D. We've developed several solutions to address this:

### 1. Official PyTorch3D Installation (Most Reliable)

The [official PyTorch3D installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) provides specific instructions for Apple Silicon Macs that we've verified work:

```bash
# First ensure you have the appropriate compilers and PyTorch installed
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Install from GitHub with the correct environment variables
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python -m pip install -e "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

The critical factors for successful installation on Apple Silicon are:
- Setting `MACOSX_DEPLOYMENT_TARGET=10.9`
- Using clang as the compiler with `CC=clang CXX=clang++`
- Installing from source (either via git or by cloning the repository)
- Using PyTorch with MPS support enabled

After installation, you can verify it works by running:
```bash
python -c "import pytorch3d; print(f'PyTorch3D version: {pytorch3d.__version__}')"
```

### 2. Conda-Based PyTorch3D Installation (Alternative Approach)

### 3. Optimized PyTorch + MPS + PyTorch3D Installation 

The most reliable solution is to use our combined installation script that:
- Installs PyTorch with proper MPS (Metal Performance Shaders) support
- Installs PyTorch3D from a compatible source build
- Sets up PyTorch3D-Lite as a fallback

```bash
cd ~/Desktop/LHM/comfy_lhm_node
chmod +x install_pytorch_mps.py
./install_pytorch_mps.py
```

This script verifies that MPS is available and correctly configured before proceeding with the PyTorch3D installation, resulting in better performance and compatibility.

### 4. PyTorch3D Full Installation

The `install_pytorch3d_mac.sh` and `install_pytorch3d_mac.py` scripts automate the complex process of installing PyTorch3D on Apple Silicon. These scripts:

- Set the necessary environment variables for compilation
- Find your Pinokio ComfyUI Python installation
- Install prerequisites (fvcore, iopath, ninja)
- Clone the PyTorch3D repository and check out a compatible commit
- Build and install PyTorch3D from source
- Install roma which is also needed for LHM

### 4. PyTorch3D-Lite Alternative

If you encounter difficulties with the full PyTorch3D installation, we provide a lightweight alternative:

- The `install_pytorch3d_lite.py` script installs pytorch3d-lite and creates the necessary compatibility layer
- This version has fewer features but works on most systems without complex compilation
- It provides the core functionality needed for the LHM node

### 5. Solving Animation Format Errors

If you have the error with animation outputs like `TypeError: ... (1, 1, 400, 3), |u1`, you can:

1. **Add a Tensor Reshape node:**
   - Disconnect the animation output from any Preview Image node
   - Add a "Tensor Reshape" node from ComfyUI
   - Connect the LHM animation output to the Tensor Reshape input
   - Set the custom shape in the Tensor Reshape node to `-1, -1, 3`
   - Connect the Tensor Reshape output to your Preview Image node

2. **Update to Full Implementation:**
   - Run one of our PyTorch3D installation scripts
   - Restart ComfyUI
   - The full implementation will handle the animation output correctly

## Checking Installation Success

After running any of the PyTorch3D installation scripts, verify your installation:

1. Restart ComfyUI in Pinokio
2. Check the ComfyUI logs for these messages:
   - "Using conda-installed PyTorch3D" indicates success with the conda method
   - "Successfully loaded full LHM implementation" indicates success with direct installation
   - "PyTorch3D-Lite fix loaded successfully" indicates the lite version is working
   - "Using simplified implementation" indicates installation issues persist

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