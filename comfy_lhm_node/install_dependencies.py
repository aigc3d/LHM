#!/usr/bin/env python3
"""
Python script to install all required dependencies for the LHM node in Pinokio's ComfyUI environment.
"""

import os
import sys
import subprocess
import glob
import platform
from pathlib import Path

def run_command(cmd, print_output=True):
    """Run a shell command and optionally print the output."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if print_output:
            print(result.stdout)
        return result.stdout.strip(), True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        return e.stderr, False

def find_pinokio_comfy_path():
    """Find the Pinokio ComfyUI installation path."""
    print("Looking for Pinokio ComfyUI installation...")
    
    # Try to find the path using find command on Unix systems
    if platform.system() != "Windows":
        out, success = run_command("find ~/pinokio -name 'comfy.git' -type d 2>/dev/null | head -n 1", print_output=False)
        if success and out:
            return out
    
    # Manual entry if auto-detection fails
    print("Could not automatically find Pinokio ComfyUI path.")
    path = input("Please enter the path to Pinokio ComfyUI (e.g., ~/pinokio/api/comfy.git): ")
    path = os.path.expanduser(path)
    
    if not os.path.isdir(path):
        print(f"Error: The path {path} does not exist")
        sys.exit(1)
        
    return path

def main():
    """Main installation function."""
    print("Installing dependencies for LHM ComfyUI node...")
    
    # Find Pinokio ComfyUI path
    pinokio_comfy_path = find_pinokio_comfy_path()
    print(f"Found Pinokio ComfyUI at: {pinokio_comfy_path}")
    
    # Check if the virtual environment exists
    env_path = os.path.join(pinokio_comfy_path, "app", "env")
    if not os.path.isdir(env_path):
        print(f"Error: Python virtual environment not found at {env_path}")
        sys.exit(1)
    
    # Get Python path
    python_bin = os.path.join(env_path, "bin", "python")
    if not os.path.isfile(python_bin):
        print(f"Error: Python binary not found at {python_bin}")
        sys.exit(1)
    
    print(f"Using Python at: {python_bin}")
    
    # Install basic dependencies
    print("Installing basic dependencies...")
    run_command(f'"{python_bin}" -m pip install omegaconf rembg opencv-python scikit-image matplotlib')
    
    # Install onnxruntime (platform-specific)
    if platform.machine() == 'arm64' or platform.machine() == 'aarch64':
        print("Detected Apple Silicon, installing onnxruntime-silicon...")
        run_command(f'"{python_bin}" -m pip install onnxruntime-silicon')
    else:
        print("Installing standard onnxruntime...")
        run_command(f'"{python_bin}" -m pip install onnxruntime')
    
    # Install roma
    print("Installing roma...")
    run_command(f'"{python_bin}" -m pip install roma')
    
    # Try to install pytorch3d
    print("Attempting to install pytorch3d (this may fail on some platforms)...")
    if platform.machine() == 'arm64' or platform.machine() == 'aarch64':
        print("Detected Apple Silicon, using macOS-specific installation...")
        env_vars = "MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++"
        out, success = run_command(f'{env_vars} "{python_bin}" -m pip install --no-deps pytorch3d')
        if not success:
            print("Warning: Could not install pytorch3d. Some functionality will be limited.")
            print("You may need to install pytorch3d manually following the instructions at:")
            print("https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md")
    else:
        out, success = run_command(f'"{python_bin}" -m pip install --no-deps pytorch3d')
        if not success:
            print("Warning: Could not install pytorch3d. Some functionality will be limited.")
            print("You may need to install pytorch3d manually following the instructions at:")
            print("https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md")
    
    # Set up the LHM node
    print("Setting up LHM node in ComfyUI...")
    lhm_path = "/Users/danny/Desktop/LHM"  # Hard-coded path for now
    custom_nodes_path = os.path.join(pinokio_comfy_path, "app", "custom_nodes")
    
    # Create custom_nodes directory if it doesn't exist
    os.makedirs(custom_nodes_path, exist_ok=True)
    
    # Copy LHM node files
    print("Copying LHM node files to ComfyUI...")
    lhm_node_path = os.path.join(custom_nodes_path, "lhm_node")
    os.makedirs(lhm_node_path, exist_ok=True)
    
    # Copy all files from comfy_lhm_node to the destination
    source_dir = os.path.join(lhm_path, "comfy_lhm_node")
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        dest_item = os.path.join(lhm_node_path, item)
        
        if os.path.isdir(source_item):
            # For directories, use recursive copy
            run_command(f'cp -r "{source_item}" "{dest_item}"', print_output=False)
        else:
            # For files, simple copy
            run_command(f'cp "{source_item}" "{dest_item}"', print_output=False)
    
    # Create symbolic links for LHM core code
    print("Creating symbolic links for LHM core code...")
    app_dir = os.path.join(pinokio_comfy_path, "app")
    os.chdir(app_dir)
    
    run_command(f'ln -sf "{os.path.join(lhm_path, "LHM")}" .', print_output=False)
    run_command(f'ln -sf "{os.path.join(lhm_path, "engine")}" .', print_output=False)
    run_command(f'ln -sf "{os.path.join(lhm_path, "configs")}" .', print_output=False)
    
    # Create link for motion data if it exists
    motion_data_path = os.path.join(lhm_path, "train_data", "motion_video")
    if os.path.isdir(motion_data_path):
        print("Creating symbolic link for motion data...")
        train_data_dir = os.path.join(app_dir, "train_data")
        os.makedirs(train_data_dir, exist_ok=True)
        
        run_command(f'ln -sf "{motion_data_path}" "{os.path.join(train_data_dir, "motion_video")}"', print_output=False)
    
    # Create link for model weights if they exist
    checkpoints_path = os.path.join(lhm_path, "checkpoints")
    if os.path.isdir(checkpoints_path):
        print("Creating symbolic link for model weights...")
        models_dir = os.path.join(app_dir, "models", "checkpoints")
        os.makedirs(models_dir, exist_ok=True)
        
        for pth_file in glob.glob(os.path.join(checkpoints_path, "*.pth")):
            basename = os.path.basename(pth_file)
            run_command(f'ln -sf "{pth_file}" "{os.path.join(models_dir, basename)}"', print_output=False)
    
    print("Installation complete!")
    print("Please restart ComfyUI in Pinokio to load the LHM node.")
    print("")
    print("If you haven't downloaded the model weights yet, run:")
    print(f"cd {lhm_path} && chmod +x download_weights.sh && ./download_weights.sh")

if __name__ == "__main__":
    main() 