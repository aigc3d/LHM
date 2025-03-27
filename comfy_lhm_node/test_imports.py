#!/usr/bin/env python3
"""
Test script to check if LHM can import PyTorch3D correctly.
"""

import sys
import os

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")

# First, import our path fixer
print("\n--- Importing lhm_import_fix ---")
try:
    import lhm_import_fix
    print("Successfully imported lhm_import_fix")
except ImportError as e:
    print(f"Error importing lhm_import_fix: {e}")

# Try to import PyTorch3D directly
print("\n--- Importing PyTorch3D directly ---")
try:
    import pytorch3d
    print(f"Successfully imported PyTorch3D version: {pytorch3d.__version__}")
    print("PyTorch3D is installed and working correctly!")
except ImportError as e:
    print(f"Error importing PyTorch3D: {e}")

# Try to import other required dependencies
print("\n--- Checking other dependencies ---")
dependencies = [
    "torch", 
    "roma",
    "numpy",
    "PIL",
    "cv2",
    "skimage"
]

for dep in dependencies:
    try:
        if dep == "PIL":
            import PIL
            print(f"Successfully imported {dep} version: {PIL.__version__}")
        elif dep == "cv2":
            import cv2
            print(f"Successfully imported {dep} version: {cv2.__version__}")
        elif dep == "skimage":
            import skimage
            print(f"Successfully imported {dep} version: {skimage.__version__}")
        else:
            module = __import__(dep)
            print(f"Successfully imported {dep} version: {module.__version__}")
    except ImportError as e:
        print(f"Error importing {dep}: {e}")
    except AttributeError:
        print(f"Successfully imported {dep} but couldn't determine version")

print("\nImport test complete!") 