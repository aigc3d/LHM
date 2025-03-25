"""
LHM Import Fix Module

This module helps fix Python import path issues by adding the LHM project root
to the Python path if it's not already there.

The module searches for the LHM project directory in several common locations
and adds it to the Python path.
"""

import os
import sys
import glob
from pathlib import Path

def add_to_path_if_exists(path):
    """Add a directory to Python path if it exists and isn't already in the path."""
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added {path} to Python path")
        return True
    return False

def find_lhm_project():
    """Find the LHM project directory in various locations and add it to path."""
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # List of potential paths to check (in order of preference)
    potential_paths = [
        # 1. Sibling directory called LHM
        os.path.join(parent_dir, "LHM"),
        
        # 2. Path in environment variable if set
        os.environ.get("LHM_PATH"),
        
        # 3. Parent directory if it contains LHM files
        parent_dir if os.path.exists(os.path.join(parent_dir, "LHM")) else None,
        
        # 4. Current directory if it's actually inside LHM project
        current_dir if os.path.basename(parent_dir) == "LHM" else None,
        
        # 5. Check if we're in a ComfyUI custom_nodes directory and look up
        os.path.join(os.path.dirname(os.path.dirname(parent_dir)), "LHM"),
        
        # 6. ~/LHM in user's home directory
        os.path.join(str(Path.home()), "LHM"),
        
        # 7. Look in common directories
        "/opt/LHM",
        "/usr/local/LHM",
    ]
    
    # Filter out None values
    potential_paths = [p for p in potential_paths if p is not None]
    
    # Try all potential paths
    for path in potential_paths:
        if add_to_path_if_exists(path):
            # Also add subdirectories that need to be in path
            for subdir in ["engine", "models", "runners"]:
                subpath = os.path.join(path, subdir)
                add_to_path_if_exists(subpath)
            return path
    
    # If we didn't find it yet, try searching for directories containing key LHM files
    search_paths = [
        parent_dir,
        os.path.dirname(parent_dir),
        str(Path.home()),
        "/opt", 
        "/usr/local"
    ]
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        # Look for directories that might be LHM
        for item in os.listdir(search_path):
            item_path = os.path.join(search_path, item)
            if not os.path.isdir(item_path):
                continue
                
            # Check if this looks like an LHM directory
            if (os.path.exists(os.path.join(item_path, "models", "lhm.py")) or
                os.path.exists(os.path.join(item_path, "LHM", "models", "lhm.py"))):
                
                # If we found LHM/models/lhm.py, use the parent directory
                if os.path.exists(os.path.join(item_path, "LHM", "models", "lhm.py")):
                    item_path = os.path.join(item_path, "LHM")
                    
                if add_to_path_if_exists(item_path):
                    for subdir in ["engine", "models", "runners"]:
                        subpath = os.path.join(item_path, subdir)
                        add_to_path_if_exists(subpath)
                    return item_path
    
    print("Warning: Could not find LHM project directory")
    return None

# Call the function to find and add LHM to path
lhm_path = find_lhm_project()

# Print info about python path for debugging
if lhm_path:
    print(f"Found LHM project at: {lhm_path}")
else:
    print("LHM project not found in common locations. Import errors may occur.")
    print(f"Current Python path: {sys.path}") 