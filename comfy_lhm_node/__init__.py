"""
ComfyUI node for LHM (Large Animatable Human Model).
This module provides a node for 3D human reconstruction and animation in ComfyUI.
"""

import os
import sys
import torch
import numpy as np
import comfy.model_management as model_management

# Import the helper module to fix Python path issues
try:
    from . import lhm_import_fix
except ImportError:
    # If we can't import the module, add parent directory to path manually
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Manually added {parent_dir} to Python path")

# Create a replacement for the missing comfy.cli.args
class ComfyArgs:
    def __init__(self):
        self.disable_cuda_malloc = False

args = ComfyArgs()

# Try importing optional dependencies
try:
    from .full_implementation import (
        LHMReconstructionNode, 
        setup_routes, 
        register_node_instance, 
        unregister_node_instance
    )
    has_full_implementation = True
    print("Successfully loaded full LHM implementation")
except ImportError as e:
    print(f"Warning: Could not load full LHM implementation: {e}")
    print("Using simplified implementation - some functionality will be limited")
    has_full_implementation = False

    # Create dummy functions if we don't have the full implementation
    def register_node_instance(node_id, instance):
        print(f"Registered LHM node (simplified): {node_id}")

    def unregister_node_instance(node_id):
        print(f"Unregistered LHM node (simplified): {node_id}")

    def setup_routes():
        print("Routes setup not available in simplified implementation")

# Try importing PromptServer for status updates
try:
    from server import PromptServer
    has_prompt_server = True
except ImportError:
    has_prompt_server = False
    
    # Create a dummy PromptServer for compatibility
    class DummyPromptServer:
        instance = None
        
        @staticmethod
        def send_sync(*args, **kwargs):
            pass
    
    PromptServer = DummyPromptServer
    PromptServer.instance = PromptServer

# If we don't have the full implementation, use a simplified version
if not has_full_implementation:
    class LHMTestNode:
        """
        A simple test node for LHM.
        This node just passes through the input image to verify node loading works.
        """
        
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "image": ("IMAGE",),
                    "test_mode": (["Simple", "Add Border"], {"default": "Simple"})
                }
            }
        
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "process_image"
        CATEGORY = "LHM"
        
        def __init__(self):
            self.node_id = None
            
        def onNodeCreated(self, node_id):
            self.node_id = node_id
            register_node_instance(node_id, self)
            print(f"LHM Test Node created: {node_id}")
            
        def onNodeRemoved(self):
            if self.node_id:
                unregister_node_instance(self.node_id)
                print(f"LHM Test Node removed: {self.node_id}")
        
        def process_image(self, image, test_mode):
            """Simply return the input image or add a colored border for testing."""
            print(f"LHM Test Node is processing an image with mode: {test_mode}")
            
            if test_mode == "Simple":
                return (image,)
            elif test_mode == "Add Border":
                # Add a green border to verify processing
                image_with_border = image.clone()
                
                # Get dimensions
                b, h, w, c = image.shape
                
                # Create border (10px wide)
                border_width = 10
                
                # Top border
                image_with_border[:, :border_width, :, 1] = 1.0  # Green channel
                # Bottom border
                image_with_border[:, -border_width:, :, 1] = 1.0
                # Left border
                image_with_border[:, :, :border_width, 1] = 1.0
                # Right border
                image_with_border[:, :, -border_width:, 1] = 1.0
                
                return (image_with_border,)
    
    class SimplifiedLHMReconstructionNode:
        """
        Simplified version of the LHM Reconstruction node when full implementation is not available.
        Returns the input image and a simulated animation made from copies of the input image.
        """
        
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "input_image": ("IMAGE",),
                    "model_version": (["LHM-0.5B", "LHM-1B"], {
                        "default": "LHM-0.5B"
                    }),
                    "export_mesh": ("BOOLEAN", {"default": False}),
                    "remove_background": ("BOOLEAN", {"default": True}),
                    "recenter": ("BOOLEAN", {"default": True})
                },
                "optional": {
                    "preview_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                }
            }
        
        RETURN_TYPES = ("IMAGE", "IMAGE")
        RETURN_NAMES = ("processed_image", "animation")
        FUNCTION = "reconstruct_human"
        CATEGORY = "LHM"
        
        def __init__(self):
            """Initialize the node with empty model and components."""
            self.device = model_management.get_torch_device()
            self.node_id = None  # Will be set in onNodeCreated
            
        # Lifecycle hook when node is created in the graph
        def onNodeCreated(self, node_id):
            """Handle node creation event"""
            self.node_id = node_id
            register_node_instance(node_id, self)
            print(f"LHM node created (simplified): {node_id}")
        
        # Lifecycle hook when node is removed from the graph
        def onNodeRemoved(self):
            """Handle node removal event"""
            if self.node_id:
                unregister_node_instance(self.node_id)
                print(f"LHM node removed (simplified): {self.node_id}")
            
        def reconstruct_human(self, input_image, model_version, export_mesh, remove_background, recenter, preview_scale=1.0):
            """
            Simplified method that returns the input image and a mock animation.
            In the full implementation, this would perform human reconstruction.
            """
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 0, "text": "Starting simple reconstruction..."})
            
            try:
                # For this simplified version, just return the input image
                if isinstance(input_image, torch.Tensor):
                    print("SimplifiedLHMReconstructionNode: Processing image")
                    
                    # Apply simple processing
                    if has_prompt_server:
                        PromptServer.instance.send_sync("lhm.progress", {"value": 50, "text": "Creating animation frames..."})
                    
                    # Just reshape the input image to simulate animation frames
                    b, h, w, c = input_image.shape
                    animation = input_image.unsqueeze(1)  # Add a time dimension
                    # Repeat the frame 5 times to simulate animation
                    animation = animation.repeat(1, 5, 1, 1, 1)
                    
                    # Send completion notification
                    if has_prompt_server:
                        PromptServer.instance.send_sync("lhm.progress", {"value": 100, "text": "Simple reconstruction complete"})
                    
                    return input_image, animation
                else:
                    print("SimplifiedLHMReconstructionNode: Invalid input format")
                    return torch.zeros((1, 512, 512, 3)), torch.zeros((1, 5, 512, 512, 3))
                
            except Exception as e:
                # Send error notification
                error_msg = f"Error in simplified LHM reconstruction: {str(e)}"
                if has_prompt_server:
                    PromptServer.instance.send_sync("lhm.progress", {"value": 0, "text": error_msg})
                print(error_msg)
                # Return empty results
                return (
                    torch.zeros((1, 512, 512, 3)), 
                    torch.zeros((1, 5, 512, 512, 3))
                )
    
    # Use the simplified version as our implementation
    LHMReconstructionNode = SimplifiedLHMReconstructionNode

# Register nodes for ComfyUI
NODE_CLASS_MAPPINGS = {}

# Always register the test node
if not has_full_implementation:
    NODE_CLASS_MAPPINGS["LHMTestNode"] = LHMTestNode

# Always register the reconstruction node (either full or simplified)
NODE_CLASS_MAPPINGS["LHMReconstructionNode"] = LHMReconstructionNode

# Display names for nodes
NODE_DISPLAY_NAME_MAPPINGS = {}

if not has_full_implementation:
    NODE_DISPLAY_NAME_MAPPINGS["LHMTestNode"] = "LHM Test Node"

NODE_DISPLAY_NAME_MAPPINGS["LHMReconstructionNode"] = "LHM Human Reconstruction"

# Web directory for client-side extensions
WEB_DIRECTORY = "./web/js"

# Initialize routes
setup_routes()

# Export symbols
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY'] 