# LHM import fix for Pinokio
import sys
import os

# Add this directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add the miniconda Python path to sys.path if not already there
miniconda_path = "/Users/danny/pinokio/bin/miniconda/lib/python3.10/site-packages"
if os.path.exists(miniconda_path) and miniconda_path not in sys.path:
    sys.path.append(miniconda_path)

# Add the LHM core to the Python path if needed
LHM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../LHM')
if os.path.exists(LHM_PATH) and LHM_PATH not in sys.path:
    sys.path.append(LHM_PATH)

# Try to import PyTorch3D directly
try:
    import pytorch3d
    print(f"Using PyTorch3D version: {pytorch3d.__version__}")
except ImportError:
    print("Warning: PyTorch3D not found. Some features may not work.")
    # Try to use the compatibility layer as a fallback
    try:
        from pytorch3d_lite_fix import *
        print("PyTorch3D compatibility layer loaded")
    except ImportError:
        print("Warning: PyTorch3D compatibility layer not found. Some features may not work.")
