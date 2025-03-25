"""
Helper module to fix LHM imports in Pinokio/ComfyUI environment.
This file is imported by the node to ensure the LHM package is in the Python path.
"""
import os
import sys

def fix_imports():
    """Add the LHM parent directory to the Python path if not already there."""
    
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the LHM project root
    lhm_dir = os.path.dirname(os.path.dirname(current_dir))
    
    # Add to Python path if not already there
    if lhm_dir not in sys.path:
        sys.path.insert(0, lhm_dir)
        print(f"Added {lhm_dir} to Python path")
    
    # If this is running in a Pinokio environment, also check for python_path.txt
    pinokio_dir = None
    parts = current_dir.split(os.sep)
    for i in range(len(parts) - 1, 0, -1):
        # Look for a potential Pinokio app directory
        if 'pinokio' in parts[i] or 'comfy' in parts[i]:
            potential_app_dir = os.path.join(os.sep, *parts[:i+1])
            python_path_file = os.path.join(potential_app_dir, 'python_path.txt')
            
            if os.path.isfile(python_path_file):
                pinokio_dir = potential_app_dir
                break
    
    # If Pinokio directory found, read the python_path.txt file
    if pinokio_dir:
        try:
            with open(os.path.join(pinokio_dir, 'python_path.txt'), 'r') as f:
                paths = f.read().splitlines()
                for path in paths:
                    if path and path not in sys.path:
                        sys.path.append(path)
                        print(f"Added {path} to Python path (from python_path.txt)")
        except Exception as e:
            print(f"Warning: Could not read python_path.txt: {e}")

# Fix imports when the module is imported
fix_imports() 