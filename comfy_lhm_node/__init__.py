import os
import sys

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

import torch
import numpy as np
from PIL import Image
import cv2
import comfy.model_management as model_management
from comfy.cli import args
from rembg import remove
from omegaconf import OmegaConf
from server import PromptServer

# Add LHM project to Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Removed redundant path addition as it's handled by lhm_import_fix

try:
    from engine.pose_estimation.pose_estimator import PoseEstimator
    from engine.SegmentAPI.base import Bbox
    from LHM.runners.infer.utils import (
        calc_new_tgt_size_by_aspect,
        center_crop_according_to_mask,
        prepare_motion_seqs,
    )
except ImportError as e:
    print(f"Error importing LHM modules: {e}")
    print("Please make sure the LHM project is in your Python path.")
    print(f"Current Python path: {sys.path}")
    raise

# Import resource management routes
from .routes import register_node_instance, unregister_node_instance, setup_routes

class LHMReconstructionNode:
    """
    ComfyUI node for LHM (Large Animatable Human Model) reconstruction.
    
    This node takes an input image and generates:
    1. A processed image with background removal and recentering
    2. An animation sequence based on provided motion data
    3. A 3D mesh of the reconstructed human (optional)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "model_version": (["LHM-0.5B", "LHM-1B"], {
                    "default": "LHM-0.5B"
                }),
                "motion_path": ("STRING", {
                    "default": "./train_data/motion_video/mimo1/smplx_params"
                }),
                "export_mesh": ("BOOLEAN", {"default": False}),
                "remove_background": ("BOOLEAN", {"default": True}),
                "recenter": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "preview_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "VIDEO", "MESH")
    RETURN_NAMES = ("processed_image", "animation", "3d_mesh")
    FUNCTION = "reconstruct_human"
    CATEGORY = "LHM"
    
    # For lazy evaluation support
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Tell ComfyUI when the node needs to be re-evaluated.
        Returns a hash based on the input image and settings.
        """
        # Hash the input image if available
        if "input_image" in kwargs and kwargs["input_image"] is not None:
            if isinstance(kwargs["input_image"], torch.Tensor):
                # Simple hash based on a subset of pixels
                image = kwargs["input_image"]
                if len(image.shape) == 4 and image.shape[0] > 0:
                    sample = image[0, ::10, ::10, 0]
                    return hash(str(sample.cpu().numpy().tobytes()))
        
        # Otherwise return a hash of the settings
        return hash(str(kwargs.get("model_version", "")) + 
                   str(kwargs.get("export_mesh", False)) + 
                   str(kwargs.get("remove_background", True)) + 
                   str(kwargs.get("recenter", True)))
    
    def __init__(self):
        """Initialize the node with empty model and components."""
        self.model = None
        self.device = model_management.get_torch_device()
        self.dtype = model_management.unet_dtype()
        self.pose_estimator = None
        self.face_detector = None
        self.parsing_net = None
        self.cfg = None
        self.last_model_version = None
        self.node_id = None  # Will be set in onNodeCreated
        
    # Lifecycle hook when node is created in the graph
    def onNodeCreated(self, node_id):
        """Handle node creation event"""
        self.node_id = node_id
        # Register this instance for resource management
        register_node_instance(node_id, self)
        print(f"LHM node created: {node_id}")
    
    # Lifecycle hook when node is removed from the graph
    def onNodeRemoved(self):
        """Handle node removal event"""
        if self.node_id:
            # Unregister this instance
            unregister_node_instance(self.node_id)
            print(f"LHM node removed: {self.node_id}")
            
            # Clean up resources
            self.model = None
            self.pose_estimator = None
            self.face_detector = None
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    def update_value(self, value, prev_value=None):
        """
        Helper method for lazy evaluation.
        Returns the previous value if the new value is None.
        """
        if value is not None:
            return value
        return prev_value
        
    def reconstruct_human(self, input_image, model_version, motion_path, export_mesh, remove_background, recenter, preview_scale=1.0):
        """
        Main method to process an input image and generate human reconstruction outputs.
        
        Args:
            input_image: Input image tensor from ComfyUI
            model_version: Which LHM model version to use
            motion_path: Path to the motion sequence data
            export_mesh: Whether to export a 3D mesh
            remove_background: Whether to remove the image background
            recenter: Whether to recenter the human in the image
            preview_scale: Scale factor for preview images
            
        Returns:
            Tuple of (processed_image, animation_sequence, mesh_data)
        """
        try:
            # Send initial progress update
            PromptServer.instance.send_sync("lhm.progress", {"value": 0, "text": "Starting reconstruction..."})
            
            # Convert input_image to numpy array
            if isinstance(input_image, torch.Tensor):
                input_image = input_image.cpu().numpy()
            
            # Convert to PIL Image for preprocessing
            input_image = Image.fromarray((input_image[0] * 255).astype(np.uint8))
            
            # Initialize components if not already loaded or if model version changed
            if self.model is None or self.last_model_version != model_version:
                PromptServer.instance.send_sync("lhm.progress", {"value": 10, "text": "Initializing components..."})
                self.initialize_components(model_version)
                self.last_model_version = model_version
            
            # Preprocess image
            PromptServer.instance.send_sync("lhm.progress", {"value": 30, "text": "Preprocessing image..."})
            processed_image = self.preprocess_image(input_image, remove_background, recenter)
            
            # Run inference
            PromptServer.instance.send_sync("lhm.progress", {"value": 50, "text": "Running inference..."})
            results = self.run_inference(processed_image, motion_path, export_mesh)
            
            # Apply preview scaling if needed
            if preview_scale != 1.0:
                # Scale the processed image and animation for preview
                results = self.apply_preview_scaling(results, preview_scale)
            
            # Complete
            PromptServer.instance.send_sync("lhm.progress", {"value": 100, "text": "Reconstruction complete!"})
            return results
            
        except Exception as e:
            # Send error notification
            error_msg = f"Error in LHM reconstruction: {str(e)}"
            PromptServer.instance.send_sync("lhm.progress", {"value": 0, "text": error_msg})
            print(error_msg)
            # Return empty results
            return (
                torch.zeros((1, 3, 512, 512)), 
                torch.zeros((1, 30, 3, 512, 512)),
                None
            )

    def initialize_components(self, model_version):
        """Initialize the LHM model and related components."""
        try:
            # Load configuration
            PromptServer.instance.send_sync("lhm.progress", {"value": 12, "text": "Loading configuration..."})
            
            # Try multiple locations for the config file
            config_paths = [
                # Regular path assuming our node is directly in ComfyUI/custom_nodes
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "configs", f"{model_version.lower()}.yaml"),
                
                # Pinokio potential path
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                           "configs", f"{model_version.lower()}.yaml"),
                
                # Try a relative path based on the current working directory
                os.path.join(os.getcwd(), "configs", f"{model_version.lower()}.yaml"),
            ]
            
            config_path = None
            for path in config_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                # Look for config file in other potential locations
                lhm_locations = []
                for path in sys.path:
                    potential_config = os.path.join(path, "configs", f"{model_version.lower()}.yaml")
                    if os.path.exists(potential_config):
                        config_path = potential_config
                        break
                    if "LHM" in path or "lhm" in path.lower():
                        lhm_locations.append(path)
                
                # Try LHM-specific locations
                if config_path is None and lhm_locations:
                    for lhm_path in lhm_locations:
                        potential_config = os.path.join(lhm_path, "configs", f"{model_version.lower()}.yaml")
                        if os.path.exists(potential_config):
                            config_path = potential_config
                            break
            
            if config_path is None:
                raise FileNotFoundError(f"Config file for {model_version} not found. Searched in: {config_paths}")
            
            self.cfg = OmegaConf.load(config_path)
            
            # Initialize pose estimator
            PromptServer.instance.send_sync("lhm.progress", {"value": 15, "text": "Initializing pose estimator..."})
            self.pose_estimator = PoseEstimator()
            
            # Initialize face detector and parsing network
            PromptServer.instance.send_sync("lhm.progress", {"value": 18, "text": "Setting up background removal..."})
            try:
                from engine.SegmentAPI.SAM import SAM2Seg
                self.face_detector = SAM2Seg()
            except ImportError:
                print("Warning: SAM2 not found, using rembg for background removal")
                self.face_detector = None
            
            # Load LHM model
            PromptServer.instance.send_sync("lhm.progress", {"value": 20, "text": "Loading LHM model..."})
            self.model = self.load_lhm_model(model_version)
        except Exception as e:
            PromptServer.instance.send_sync("lhm.progress", {"value": 0, "text": f"Initialization error: {str(e)}"})
            raise

    def preprocess_image(self, image, remove_background, recenter):
        """Preprocess the input image with background removal and recentering."""
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Remove background if requested
        if remove_background:
            PromptServer.instance.send_sync("lhm.progress", {"value": 32, "text": "Removing background..."})
            if self.face_detector is not None:
                # Use SAM2 for background removal
                mask = self.face_detector.get_mask(image_np)
            else:
                # Use rembg as fallback
                output = remove(image_np)
                mask = output[:, :, 3] > 0
        else:
            mask = np.ones(image_np.shape[:2], dtype=bool)
        
        # Recenter if requested
        if recenter:
            PromptServer.instance.send_sync("lhm.progress", {"value": 35, "text": "Recentering image..."})
            image_np = center_crop_according_to_mask(image_np, mask)
        
        # Convert back to PIL Image
        return Image.fromarray(image_np)

    def load_lhm_model(self, model_version):
        """Load the LHM model weights and architecture."""
        # Look for the model weights in various locations
        model_paths = [
            # Regular path
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "checkpoints", f"{model_version.lower()}.pth"),
            
            # Pinokio potential path - custom_nodes parent dir
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "checkpoints", f"{model_version.lower()}.pth"),
            
            # Pinokio models directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "models", "checkpoints", f"{model_version.lower()}.pth"),
            
            # Try a relative path based on current working directory
            os.path.join(os.getcwd(), "checkpoints", f"{model_version.lower()}.pth"),
            
            # ComfyUI models/checkpoints directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "models", "checkpoints", f"{model_version.lower()}.pth"),
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            # Look for weights file in other potential locations
            lhm_locations = []
            for path in sys.path:
                potential_weights = os.path.join(path, "checkpoints", f"{model_version.lower()}.pth")
                if os.path.exists(potential_weights):
                    model_path = potential_weights
                    break
                if "LHM" in path or "lhm" in path.lower():
                    lhm_locations.append(path)
            
            # Try LHM-specific locations
            if model_path is None and lhm_locations:
                for lhm_path in lhm_locations:
                    potential_weights = os.path.join(lhm_path, "checkpoints", f"{model_version.lower()}.pth")
                    if os.path.exists(potential_weights):
                        model_path = potential_weights
                        break
                        
        if model_path is None:
            PromptServer.instance.send_sync("lhm.progress", {"value": 0, "text": "Error: Model weights not found!"})
            error_msg = f"Model weights not found. Searched in: {model_paths}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load model using the configuration
        PromptServer.instance.send_sync("lhm.progress", {"value": 22, "text": "Building model architecture..."})
        model = self._build_model(self.cfg)
        
        PromptServer.instance.send_sync("lhm.progress", {"value": 25, "text": f"Loading model weights from {model_path}..."})
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        PromptServer.instance.send_sync("lhm.progress", {"value": 28, "text": "Moving model to device..."})
        model.to(self.device)
        model.eval()
        
        return model

    def _build_model(self, cfg):
        """Build the LHM model architecture based on the configuration."""
        # Import the model class from LHM
        try:
            from LHM.models.lhm import LHM
        except ImportError as e:
            print(f"Error importing LHM model: {e}")
            print(f"Python path: {sys.path}")
            raise
        
        # Create model instance based on the configuration
        model = LHM(
            img_size=cfg.MODEL.IMAGE_SIZE,
            feature_scale=cfg.MODEL.FEATURE_SCALE,
            use_dropout=cfg.MODEL.USE_DROPOUT,
            drop_path=cfg.MODEL.DROP_PATH,
            use_checkpoint=cfg.TRAIN.USE_CHECKPOINT,
            checkpoint_num=cfg.TRAIN.CHECKPOINT_NUM,
        )
        
        return model

    def run_inference(self, processed_image, motion_path, export_mesh):
        """Run inference with the LHM model and post-process results."""
        # Convert processed image to tensor
        PromptServer.instance.send_sync("lhm.progress", {"value": 55, "text": "Preparing tensors..."})
        image_tensor = torch.from_numpy(np.array(processed_image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Prepare motion sequence
        PromptServer.instance.send_sync("lhm.progress", {"value": 60, "text": "Loading motion sequence..."})
        
        # Try to locate motion_path if it doesn't exist as-is
        if not os.path.exists(motion_path):
            # Try a few common locations
            potential_paths = [
                # Relative to ComfyUI
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), motion_path),
                # Relative to LHM project root
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), motion_path),
                # Relative to current working directory
                os.path.join(os.getcwd(), motion_path),
                # Try built-in motion paths in the LHM project
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "train_data", "motion_video", "mimo1", "smplx_params"),
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    motion_path = path
                    print(f"Found motion path at: {motion_path}")
                    break
        
        try:
            motion_seqs = prepare_motion_seqs(motion_path)
        except Exception as e:
            error_msg = f"Error loading motion sequence: {str(e)}"
            print(error_msg)
            PromptServer.instance.send_sync("lhm.progress", {"value": 60, "text": error_msg})
            # Try to use a default motion sequence
            try:
                default_motion_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                               "train_data", "motion_video", "mimo1", "smplx_params")
                motion_seqs = prepare_motion_seqs(default_motion_path)
                print(f"Using default motion path: {default_motion_path}")
            except Exception as e2:
                error_msg = f"Error loading default motion sequence: {str(e2)}"
                print(error_msg)
                PromptServer.instance.send_sync("lhm.progress", {"value": 60, "text": error_msg})
                # Create a dummy motion sequence
                motion_seqs = {'pred_vertices': torch.zeros((1, 30, 10475, 3), device=self.device)}
        
        # Run inference
        PromptServer.instance.send_sync("lhm.progress", {"value": 70, "text": "Running model inference..."})
        with torch.no_grad():
            results = self.model(image_tensor, motion_seqs)
        
        # Process results
        PromptServer.instance.send_sync("lhm.progress", {"value": 90, "text": "Processing results..."})
        processed_image = results['processed_image']
        animation = results['animation']
        mesh = None
        
        # Generate mesh if requested
        if export_mesh:
            PromptServer.instance.send_sync("lhm.progress", {"value": 95, "text": "Generating 3D mesh..."})
            mesh = self.generate_mesh(results)
        
        return processed_image, animation, mesh
    
    def generate_mesh(self, results):
        """Generate a 3D mesh from the model results."""
        try:
            # Extract mesh data from the results
            if 'mesh' in results:
                return results['mesh']
            
            # If not directly available, generate from vertices and faces
            if 'vertices' in results and 'faces' in results:
                try:
                    from LHM.utils.mesh_utils import generate_mesh
                    vertices = results['vertices']
                    faces = results['faces']
                    return generate_mesh(vertices, faces)
                except ImportError:
                    print("Warning: Could not import mesh_utils, using fallback mesh generation")
                    return {'vertices': results['vertices'], 'faces': results['faces']}
            
            # If we can't generate a mesh, return None
            return None
        except Exception as e:
            print(f"Error generating mesh: {str(e)}")
            return None
            
    def apply_preview_scaling(self, results, scale):
        """Scale the results for preview purposes."""
        processed_image, animation, mesh = results
        
        if scale != 1.0 and processed_image is not None:
            # Scale the processed image
            if isinstance(processed_image, torch.Tensor):
                h, w = processed_image.shape[-2:]
                new_h, new_w = int(h * scale), int(w * scale)
                processed_image = torch.nn.functional.interpolate(
                    processed_image, size=(new_h, new_w), mode='bilinear'
                )
            
            # Scale the animation frames
            if animation is not None and isinstance(animation, torch.Tensor):
                b, f, c, h, w = animation.shape
                new_h, new_w = int(h * scale), int(w * scale)
                animation = animation.reshape(b * f, c, h, w)
                animation = torch.nn.functional.interpolate(
                    animation, size=(new_h, new_w), mode='bilinear'
                )
                animation = animation.reshape(b, f, c, new_h, new_w)
        
        return processed_image, animation, mesh

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "LHMReconstructionNode": LHMReconstructionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LHMReconstructionNode": "LHM Human Reconstruction"
}

# Web directory for client-side extensions
WEB_DIRECTORY = "./web/js"

# Initialize routes
setup_routes()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY'] 