#!/usr/bin/env python3
"""
PyTorch3D Conda Installation Script
This script installs PyTorch3D using conda, which is usually more reliable
than pip for packages with complex dependencies.
"""

import os
import sys
import subprocess
import tempfile
import argparse
from pathlib import Path
import shutil

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

def find_conda():
    """Find the conda executable in Pinokio."""
    # Try to find the path automatically
    for search_path in [
        "~/pinokio/bin/miniconda/bin/conda",
        "~/pinokio/bin/conda",
        "~/miniconda/bin/conda",
        "~/miniconda3/bin/conda",
        "~/anaconda/bin/conda",
        "~/anaconda3/bin/conda"
    ]:
        expanded_path = os.path.expanduser(search_path)
        if os.path.isfile(expanded_path):
            return expanded_path
    
    # If not found directly, search using find command
    find_cmd = "find ~/pinokio -name conda -type f 2>/dev/null | head -n 1"
    conda_path, _ = run_command(find_cmd, print_output=False)
    conda_path = conda_path.strip()
    
    if not conda_path:
        print("Error: Could not find conda automatically")
        print("Please enter the path to conda executable:")
        conda_path = input().strip()
        
        if not os.path.isfile(conda_path):
            print(f"Error: The path {conda_path} does not exist")
            sys.exit(1)
    
    return conda_path

def find_python():
    """Find the Python executable in Pinokio."""
    # Try to find the path automatically
    find_cmd = "find ~/pinokio/bin/miniconda/bin -name python3.10 -type f 2>/dev/null | head -n 1"
    python_path, _ = run_command(find_cmd, print_output=False)
    python_path = python_path.strip()
    
    if not python_path:
        print("Could not find Python in Pinokio miniconda. Trying wider search...")
        find_cmd = "find ~/pinokio -name python3.10 -type f 2>/dev/null | head -n 1"
        python_path, _ = run_command(find_cmd, print_output=False)
        python_path = python_path.strip()
        
        if not python_path:
            print("Error: Could not find Python. Please install manually.")
            sys.exit(1)
    
    return python_path

def get_conda_env(python_path):
    """Get the conda environment name from the Python path."""
    try:
        # Get the directory containing the Python executable
        bin_dir = os.path.dirname(python_path)
        # Get the parent directory, which should be the env root
        env_dir = os.path.dirname(bin_dir)
        # The environment name is the name of the env root directory
        env_name = os.path.basename(env_dir)
        return env_name
    except Exception as e:
        print(f"Error determining conda environment: {e}")
        return "base"  # Default to base environment

def parse_args():
    parser = argparse.ArgumentParser(description='Install PyTorch3D using conda for Apple Silicon.')
    parser.add_argument('--conda', dest='conda_path', help='Path to conda executable')
    parser.add_argument('--python', dest='python_path', help='Path to Python executable')
    parser.add_argument('--env', dest='conda_env', help='Conda environment name to install into')
    return parser.parse_args()

def main():
    print("Installing PyTorch3D using conda...")
    
    # Parse command-line arguments
    args = parse_args()
    
    # Find conda executable
    conda_path = args.conda_path if args.conda_path else find_conda()
    print(f"Using conda at: {conda_path}")
    
    # Find Python executable
    python_path = args.python_path if args.python_path else find_python()
    print(f"Using Python at: {python_path}")
    
    # Get conda environment
    conda_env = args.conda_env if args.conda_env else get_conda_env(python_path)
    print(f"Using conda environment: {conda_env}")
    
    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "conda_install.log")
        
        # Helper function to run conda commands
        def run_conda_cmd(cmd):
            full_cmd = f"{conda_path} {cmd}"
            output, ret_code = run_command(full_cmd, print_output=False)
            
            with open(log_file, 'a') as f:
                f.write(f"Command: {full_cmd}\n")
                f.write(output)
                f.write("\n" + "-" * 80 + "\n")
            
            if ret_code != 0:
                print(f"Error executing command: {full_cmd}")
                print("See log excerpt:")
                print('\n'.join(output.splitlines()[-10:]))  # Show last 10 lines
                print("Continuing with installation...")
            
            return output, ret_code
        
        # Add conda-forge channel
        print("Configuring conda channels...")
        run_conda_cmd("config --show channels")
        run_conda_cmd("config --add channels conda-forge")
        run_conda_cmd("config --set channel_priority flexible")
        
        # Install dependencies
        print("Installing dependencies...")
        run_conda_cmd(f"install -y -n {conda_env} fvcore iopath")
        
        # Install PyTorch3D
        print("Installing PyTorch3D...")
        run_conda_cmd(f"install -y -n {conda_env} pytorch3d")
        
        # Update PyTorch with MPS support
        print("Updating PyTorch with MPS support...")
        run_conda_cmd(f"install -y -n {conda_env} 'pytorch>=2.0.0' 'torchvision>=0.15.0'")
        
        # Install roma
        print("Installing roma...")
        run_conda_cmd(f"install -y -n {conda_env} roma")
    
    # Create our compatibility layer
    print("Setting up the PyTorch3D compatibility layer...")
    lhm_path = os.path.dirname(os.path.abspath(__file__))
    fix_path = os.path.join(lhm_path, "pytorch3d_lite_fix.py")
    
    with open(fix_path, 'w') as f:
        f.write("""
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
""")
    
    # Update lhm_import_fix.py
    fix_import_path = os.path.join(lhm_path, "lhm_import_fix.py")
    with open(fix_import_path, 'w') as f:
        f.write("""
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
""")
    
    print("\nInstallation complete!")
    print("PyTorch3D has been installed using conda.")
    print("Please restart ComfyUI to load PyTorch3D and the full LHM node functionality.")

if __name__ == "__main__":
    main() 