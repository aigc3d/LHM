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
