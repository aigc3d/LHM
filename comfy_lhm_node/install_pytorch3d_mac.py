#!/usr/bin/env python3
"""
PyTorch3D Installation Script for Apple Silicon (M1/M2/M3) Macs
This script installs PyTorch3D from source in a way compatible with Apple Silicon.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import glob
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

def main():
    print("Installing PyTorch3D for Apple Silicon...")
    
    # Set required environment variables for build
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
    
    # Find Pinokio ComfyUI path
    comfy_path = find_pinokio_comfy_path()
    python_bin, pip_bin = find_python_and_pip(comfy_path)
    
    print(f"Using Python: {python_bin}")
    print(f"Using Pip: {pip_bin}")
    
    # Install dependencies first
    print("Installing dependencies...")
    run_command(f"{pip_bin} install --no-cache-dir fvcore iopath")
    
    # Install pre-requisites for PyTorch3D
    print("Installing PyTorch3D pre-requisites...")
    run_command(f"{pip_bin} install --no-cache-dir 'pytorch3d-lite==0.1.1' ninja")
    
    # Install pytorch3d from source (specific version compatible with Apple Silicon)
    print("Installing PyTorch3D from source...")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Working in temporary directory: {temp_dir}")
        os.chdir(temp_dir)
        
        # Clone the repo at a specific commit that works well with Apple Silicon
        run_command("git clone https://github.com/facebookresearch/pytorch3d.git")
        os.chdir(os.path.join(temp_dir, "pytorch3d"))
        run_command("git checkout 4e46dcfb2dd1c75ab1f6abf79a2e3e52fd8d427a")
        
        # Install PyTorch3D
        run_command(f"{pip_bin} install --no-deps -e .")
    
    # Install roma which is also needed for LHM
    print("Installing roma...")
    run_command(f"{pip_bin} install roma")
    
    print("Installation complete!")
    print("Please restart ComfyUI to load PyTorch3D and the full LHM node functionality.")

if __name__ == "__main__":
    main() 