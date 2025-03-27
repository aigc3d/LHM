#!/usr/bin/env python
import os
import sys
import shutil
import subprocess
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Install LHM ComfyUI node to Pinokio')
    parser.add_argument('pinokio_dir', nargs='?', default=os.path.expanduser('~/pinokio/api/comfy.git/app'),
                        help='Path to Pinokio ComfyUI directory')
    args = parser.parse_args()
    
    pinokio_dir = args.pinokio_dir
    
    # Source directory (current project)
    source_dir = os.path.join(os.getcwd(), 'comfy_lhm_node')
    
    # Check if source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        sys.exit(1)
    
    # Check if Pinokio ComfyUI directory exists
    if not os.path.isdir(pinokio_dir):
        print(f"Error: Pinokio ComfyUI directory {pinokio_dir} does not exist.")
        print(f"Usage: python {sys.argv[0]} [path/to/pinokio/comfy/installation]")
        sys.exit(1)
    
    # Create custom_nodes directory if it doesn't exist
    custom_nodes_dir = os.path.join(pinokio_dir, 'custom_nodes')
    os.makedirs(custom_nodes_dir, exist_ok=True)
    
    # Create the LHM node directory
    target_dir = os.path.join(custom_nodes_dir, 'lhm_node')
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy all files from comfy_lhm_node to the target directory
    print(f"Copying files from {source_dir} to {target_dir}...")
    
    # Remove the target directory if it exists
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
        except Exception as e:
            print(f"Warning: Could not delete existing directory: {e}")
    
    # Copy the directory
    try:
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    except Exception as e:
        print(f"Error copying files: {e}")
        sys.exit(1)
    
    # Install requirements if requirements.txt exists
    requirements_file = os.path.join(source_dir, 'requirements.txt')
    if os.path.isfile(requirements_file):
        print("Installing requirements...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file], check=True)
        except subprocess.CalledProcessError:
            print("Warning: Failed to install requirements.")
    
    # Create a symbolic link or add the main LHM directory to PYTHONPATH
    print("Setting up Python path for LHM...")
    python_path_file = os.path.join(pinokio_dir, 'python_path.txt')
    lhm_dir = os.path.dirname(os.getcwd())
    
    # Check if we already added this path
    if os.path.isfile(python_path_file):
        with open(python_path_file, 'r') as f:
            paths = f.read().splitlines()
        
        if lhm_dir not in paths:
            with open(python_path_file, 'a') as f:
                f.write(f"{lhm_dir}\n")
    else:
        with open(python_path_file, 'w') as f:
            f.write(f"{lhm_dir}\n")
    
    # Create a startup script to set PYTHONPATH before ComfyUI starts
    startup_script = os.path.join(custom_nodes_dir, 'set_pythonpath.py')
    with open(startup_script, 'w') as f:
        f.write("""import os
import sys

# Add LHM directory to Python path
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "python_path.txt"), "r") as f:
    paths = f.read().splitlines()
    for path in paths:
        if path and path not in sys.path:
            sys.path.append(path)
            print(f"Added {path} to Python path")
""")
    
    # Ensure the model directory exists in Pinokio
    model_dir = os.path.join(pinokio_dir, 'models')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    print("\n==================== INSTALLATION COMPLETED ====================")
    print(f"LHM node has been installed to Pinokio's ComfyUI at: {target_dir}")
    print("\nIMPORTANT: You need to restart ComfyUI in Pinokio for changes to take effect.")
    print("\nIf your models are not found, copy or symlink model weights to:")
    print(f"{checkpoints_dir}/")
    print("\nYou can also create a symbolic link to your existing model weights:")
    
    if os.name == 'nt':  # Windows
        print(f"mklink /D {checkpoints_dir}\\lhm {os.getcwd()}\\checkpoints")
    else:  # Unix-like
        print(f"ln -s {os.getcwd()}/checkpoints/* {checkpoints_dir}/")
    
    print("\n==============================================================")

if __name__ == "__main__":
    main() 