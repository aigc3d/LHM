# Changelog

## 2023-06-20
- Initial release of the LHM ComfyUI node
- Basic implementation with simplified fallback

## 2023-06-30
- Added error handling for missing dependencies
- Improved documentation

## 2023-07-10
- Added support for Pinokio installation
- Created installation guide

## 2023-11-15
- Updated to support LHM 1.0
- Added animation output

## 2024-06-22
- Enhanced troubleshooting guide with detailed installation steps
- Added quality of life improvements for error messages

## 2024-06-25
- Added PyTorch3D installation scripts for Apple Silicon
  - Created `install_pytorch3d_mac.sh` - Bash script for installing PyTorch3D on macOS
  - Created `install_pytorch3d_mac.py` - Python version of the installation script
  - Added `install_pytorch3d_lite.py` - Alternative lightweight implementation
- Added PyTorch3D-Lite compatibility layer for easier installation
- Updated TROUBLESHOOTING.md with detailed instructions for dealing with PyTorch3D installation issues
- Added workaround for animation format issues in simplified mode using Tensor Reshape 

## 2024-06-26
- Added optimized PyTorch MPS installation script for Apple Silicon (`install_pytorch_mps.py`)
  - Properly configures PyTorch with Metal Performance Shaders (MPS) support
  - Attempts to install PyTorch3D from source with appropriate environment variables
  - Sets up PyTorch3D-Lite as a fallback in case of installation issues
  - Creates a smarter import fix that tries both regular PyTorch3D and the lite version
- Updated TROUBLESHOOTING.md with the new recommended installation method

## 2024-06-27
- Added conda-based installation scripts for PyTorch3D
  - Created `install_pytorch3d_conda.sh` - Bash script for installing PyTorch3D using conda
  - Created `install_pytorch3d_conda.py` - Python version of the conda installation script
  - These scripts provide the most reliable method for installing PyTorch3D
  - Added conda-forge channel configuration for consistent package availability
  - Enhanced compatibility layer that checks for conda-installed PyTorch3D first
- Updated TROUBLESHOOTING.md to highlight conda as the recommended installation method 

## 2024-06-28 (2)
- Added `create_test_workflow.py` script to automatically generate a sample ComfyUI workflow for testing the LHM node
- Updated `TROUBLESHOOTING.md` with direct references to the official PyTorch3D installation documentation
- Reorganized installation sections to prioritize the official PyTorch3D installation methods
- Added detailed environment variable guidance for Apple Silicon users based on successful installations

## 2024-06-28
- Successfully installed PyTorch3D from source following the official documentation
- Added reference to official PyTorch3D installation guide in `TROUBLESHOOTING.md`
- Created `test_imports.py` to verify all dependencies are properly installed
- Updated `lhm_import_fix.py` to prioritize direct PyTorch3D imports and explicit paths to Pinokio's miniconda Python packages
- Fixed dependency installation guidance for macOS with Apple Silicon and environment variable specifications for macOS compilation 