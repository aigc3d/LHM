#!/usr/bin/env python3
"""
PyTorch3D-Lite Installation Script
This script installs a lightweight version of PyTorch3D that works on most platforms
including Apple Silicon without complex compilation.
"""

import os
import sys
import subprocess
import glob
import argparse
from pathlib import Path

def run_command(cmd, print_output=True):
    """Run a shell command and return the output."""
    print(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    output = []
    for line in process.stdout:
        if print_output:
            print(line.strip())
        output.append(line)
    
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}")
    
    return ''.join(output), process.returncode

def find_pinokio_comfy_path():
    """Find the Pinokio ComfyUI installation path."""
    # Try to find the path automatically
    find_cmd = "find ~/pinokio -name 'comfy.git' -type d 2>/dev/null | head -n 1"
    comfy_path, _ = run_command(find_cmd, print_output=False)
    comfy_path = comfy_path.strip()
    
    if not comfy_path:
        print("Error: Could not find Pinokio ComfyUI path automatically")
        print("Please enter the path to Pinokio ComfyUI installation (e.g. ~/pinokio/api/comfy.git/app):")
        comfy_path = input().strip()
        
        if not os.path.isdir(comfy_path):
            print(f"Error: The path {comfy_path} does not exist")
            sys.exit(1)
    
    return comfy_path

def find_python_and_pip(comfy_path):
    """Find Python and Pip in the Pinokio ComfyUI installation."""
    # Check primary location
    python_bin = os.path.join(comfy_path, "app/env/bin/python")
    pip_bin = os.path.join(comfy_path, "app/env/bin/pip")
    
    if not os.path.isfile(python_bin):
        # Try alternate location
        python_bin = os.path.join(comfy_path, "env/bin/python")
        pip_bin = os.path.join(comfy_path, "env/bin/pip")
        
        if not os.path.isfile(python_bin):
            print("Error: Python binary not found at expected location")
            print("Trying to find Python in Pinokio...")
            
            # Search for Python binary
            find_python_cmd = f"find {comfy_path} -name 'python' -type f | grep -E 'bin/python$' | head -n 1"
            python_result, _ = run_command(find_python_cmd, print_output=False)
            python_bin = python_result.strip()
            
            # Search for pip binary
            find_pip_cmd = f"find {comfy_path} -name 'pip' -type f | grep -E 'bin/pip$' | head -n 1"
            pip_result, _ = run_command(find_pip_cmd, print_output=False)
            pip_bin = pip_result.strip()
            
            if not python_bin:
                print("Error: Could not find Python in Pinokio. Please install manually.")
                sys.exit(1)
            else:
                print(f"Found Python at: {python_bin}")
                print(f"Found Pip at: {pip_bin}")
    
    return python_bin, pip_bin

def parse_args():
    parser = argparse.ArgumentParser(description='Install PyTorch3D-Lite for Apple Silicon.')
    parser.add_argument('--python', dest='python_bin', help='Path to Python executable')
    parser.add_argument('--pip', dest='pip_bin', help='Path to pip executable')
    parser.add_argument('--pinokio', dest='pinokio_path', help='Path to Pinokio ComfyUI installation')
    return parser.parse_args()

def main():
    print("Installing PyTorch3D-Lite...")
    
    # Parse command-line arguments
    args = parse_args()
    
    # Get Python and pip paths
    if args.python_bin and args.pip_bin:
        python_bin = args.python_bin
        pip_bin = args.pip_bin
        
        if not os.path.isfile(python_bin):
            print(f"Error: Python binary not found at specified path: {python_bin}")
            sys.exit(1)
        
        if not os.path.isfile(pip_bin):
            print(f"Error: Pip binary not found at specified path: {pip_bin}")
            sys.exit(1)
    else:
        # Find Pinokio ComfyUI path
        comfy_path = args.pinokio_path if args.pinokio_path else find_pinokio_comfy_path()
        python_bin, pip_bin = find_python_and_pip(comfy_path)
    
    print(f"Using Python: {python_bin}")
    print(f"Using Pip: {pip_bin}")
    
    # Install dependencies first
    print("Installing dependencies...")
    run_command(f"{pip_bin} install --no-cache-dir omegaconf rembg")
    
    # Download the PyTorch3D-Lite package if it's not available in the PyPI
    print("Installing PyTorch3D-Lite (downloading if needed)...")
    
    # Try installing directly first
    install_result, ret_code = run_command(f"{pip_bin} install pytorch3d-lite==0.1.1")
    
    # If it failed, download the package and install locally
    if ret_code != 0:
        print("PyTorch3D-Lite not found in PyPI, downloading directly...")
        package_url = "https://github.com/DenisMedeiros/pytorch3d-lite/archive/refs/tags/v0.1.1.zip"
        run_command(f"curl -L {package_url} -o /tmp/pytorch3d-lite.zip")
        run_command(f"{pip_bin} install /tmp/pytorch3d-lite.zip")
    
    # Install roma which is also needed for LHM
    print("Installing roma...")
    run_command(f"{pip_bin} install roma")
    
    # Create a fix file to help LHM use the lite version
    lhm_path = os.path.dirname(os.path.abspath(__file__))
    lite_fix_path = os.path.join(lhm_path, "pytorch3d_lite_fix.py")
    
    with open(lite_fix_path, 'w') as f:
        f.write("""
# PyTorch3D-Lite fix for LHM
import sys
import os

# This module provides shims for necessary PyTorch3D functions using the lite version
try:
    import pytorch3d_lite
except ImportError:
    # If import fails, add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    try:
        import pytorch3d_lite
    except ImportError:
        # If still failing, try to load from site-packages
        # First get the site-packages directory from the Python path
        import site
        site_packages = site.getsitepackages()
        for site_pkg in site_packages:
            sys.path.append(site_pkg)
            try:
                import pytorch3d_lite
                break
            except ImportError:
                continue
        else:
            print("Error: Could not import pytorch3d_lite from any location")
            sys.exit(1)

# Add this current directory to the path so LHM can find pytorch3d_lite
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Create namespace for pytorch3d
if 'pytorch3d' not in sys.modules:
    import types
    pytorch3d = types.ModuleType('pytorch3d')
    sys.modules['pytorch3d'] = pytorch3d
    
    # Create submodules
    pytorch3d.transforms = types.ModuleType('pytorch3d.transforms')
    sys.modules['pytorch3d.transforms'] = pytorch3d.transforms
    
    # Map lite functions to expected pytorch3d namespace
    from pytorch3d_lite import (
        matrix_to_rotation_6d,
        rotation_6d_to_matrix,
        axis_angle_to_matrix,
        matrix_to_axis_angle,
    )
    
    # Add these to the pytorch3d.transforms namespace
    pytorch3d.transforms.matrix_to_rotation_6d = matrix_to_rotation_6d
    pytorch3d.transforms.rotation_6d_to_matrix = rotation_6d_to_matrix
    pytorch3d.transforms.axis_angle_to_matrix = axis_angle_to_matrix
    pytorch3d.transforms.matrix_to_axis_angle = matrix_to_axis_angle

print("PyTorch3D-Lite fix loaded successfully")
""")
    
    # Create an lhm_import_fix.py if it doesn't exist
    lhm_import_fix_path = os.path.join(lhm_path, "lhm_import_fix.py")
    if not os.path.exists(lhm_import_fix_path):
        with open(lhm_import_fix_path, 'w') as f:
            f.write("""
# LHM import fix for Pinokio
import sys
import os

# Add the LHM core to the Python path if needed
LHM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../LHM')
if os.path.exists(LHM_PATH) and LHM_PATH not in sys.path:
    sys.path.append(LHM_PATH)

# Add this directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Load the PyTorch3D-Lite fix
try:
    from pytorch3d_lite_fix import *
    print("Using PyTorch3D-Lite as replacement for PyTorch3D")
except ImportError:
    print("Warning: PyTorch3D-Lite fix not found. Some features may not work.")
""")
    
    print("Installation complete!")
    print("Please restart ComfyUI to load PyTorch3D-Lite and the LHM node functionality.")

if __name__ == "__main__":
    main() 