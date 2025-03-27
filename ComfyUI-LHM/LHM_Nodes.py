import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import comfy.model_management as model_management
from omegaconf import OmegaConf
import time

from .lib_lhm.engine.pose_estimation.pose_estimator import PoseEstimator

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
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("processed_image", "animation")
    FUNCTION = "execute"
    CATEGORY = "LHM"
    
    def __init__(self):
        """Initialize the node with empty model and components."""

        self.LHM_Model_Dict = {}

    def execute(self, input_image, motion_path):
        """
        Main method to process an input image and generate human reconstruction outputs.
        
        Args:
            input_image: Input image tensor from ComfyUI
            motion_path: Path to the motion sequence data
            
        Returns:
            Tuple of (processed_image, animation_sequence)
        """   
        # if 'pose_estimator' not in self.LHM_Model_Dict:
        pass

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
        # Check if we have the full LHM implementation
        if not has_lhm:
            print("Running LHM node in simplified mode - full implementation not available")
            return self._run_simplified_mode(input_image)
            
        try:
            # Send initial progress update
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 0, "text": "Starting reconstruction..."})
            
            # Convert input_image to numpy array
            if isinstance(input_image, torch.Tensor):
                input_image = input_image.cpu().numpy()
            
            # Convert to PIL Image for preprocessing
            input_image = Image.fromarray((input_image[0] * 255).astype(np.uint8))
            
            # Initialize components if not already loaded or if model version changed
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 10, "text": "Initializing components..."})
                
            if self.model is None or self.last_model_version != model_version:
                self.initialize_components(model_version)
                self.last_model_version = model_version
            
            # Preprocess image
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 30, "text": "Preprocessing image..."})
                
            processed_image = self.preprocess_image(input_image, remove_background, recenter)
            
            # Run inference
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 50, "text": "Running inference..."})
                
            processed_image, animation = self.run_inference(processed_image, motion_path, export_mesh)
            
            # Apply preview scaling if needed
            if preview_scale != 1.0:
                # Scale the processed image and animation for preview
                processed_image, animation = self.apply_preview_scaling(processed_image, animation, preview_scale)
            
            # Complete
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 100, "text": "Reconstruction complete!"})
                
            return processed_image, animation
            
        except Exception as e:
            # Send error notification
            error_msg = f"Error in LHM reconstruction: {str(e)}"
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 0, "text": error_msg})
            print(error_msg)
            # Return empty results
            return self._run_simplified_mode(input_image)

    def _run_simplified_mode(self, input_image):
        """
        Run a simplified version when full functionality is not available.
        Just returns the input image and a simulated animation.
        """
        print("Using simplified mode for LHM node")
        if isinstance(input_image, torch.Tensor):
            # Create animation by repeating the input frame
            animation = input_image.unsqueeze(1)  # Add a time dimension
            animation = animation.repeat(1, 5, 1, 1, 1)  # Repeat 5 frames
            
            return input_image, animation
        else:
            # Handle case where input is not a tensor
            print("Error: Input is not a tensor")
            return torch.zeros((1, 512, 512, 3)), torch.zeros((1, 5, 512, 512, 3))

    def initialize_components(self, model_version):
        """Initialize the LHM model and related components."""
        try:
            # Load configuration
            if has_prompt_server:
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
                raise FileNotFoundError(f"Config file for {model_version} not found.")
            
            self.cfg = OmegaConf.load(config_path)
            
            # Initialize pose estimator
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 15, "text": "Initializing pose estimator..."})
                
            self.pose_estimator = PoseEstimator()
            
            # Initialize face detector and parsing network
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 18, "text": "Setting up background removal..."})
                
            try:
                from engine.SegmentAPI.SAM import SAM2Seg
                self.face_detector = SAM2Seg()
            except ImportError:
                print("Warning: SAM2 not found, using rembg for background removal")
                self.face_detector = None
            
            # Load LHM model
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 20, "text": "Loading LHM model..."})
                
            self.model = self.load_lhm_model(model_version)
            
        except Exception as e:
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 0, "text": f"Initialization error: {str(e)}"})
            raise

    def preprocess_image(self, image, remove_background, recenter):
        """Preprocess the input image with background removal and recentering."""
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Remove background if requested
        if remove_background and has_rembg:
            if has_prompt_server:
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
            if has_prompt_server:
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
            if has_prompt_server:
                PromptServer.instance.send_sync("lhm.progress", {"value": 0, "text": "Error: Model weights not found!"})
            error_msg = f"Model weights not found. Searched in: {model_paths}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load model using the configuration
        if has_prompt_server:
            PromptServer.instance.send_sync("lhm.progress", {"value": 22, "text": "Building model architecture..."})
            
        model = self._build_model(self.cfg)
        
        if has_prompt_server:
            PromptServer.instance.send_sync("lhm.progress", {"value": 25, "text": f"Loading model weights from {model_path}..."})
            
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        if has_prompt_server:
            PromptServer.instance.send_sync("lhm.progress", {"value": 28, "text": "Moving model to device..."})
            
        model.to(self.device)
        model.eval()
        
        return model

    def _build_model(self, cfg):
        """Build the LHM model architecture based on the configuration."""
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
        if has_prompt_server:
            PromptServer.instance.send_sync("lhm.progress", {"value": 55, "text": "Preparing tensors..."})
            
        image_tensor = torch.from_numpy(np.array(processed_image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Prepare motion sequence
        if has_prompt_server:
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
            if has_prompt_server:
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
                if has_prompt_server:
                    PromptServer.instance.send_sync("lhm.progress", {"value": 60, "text": error_msg})
                # Create a dummy motion sequence
                motion_seqs = {'pred_vertices': torch.zeros((1, 30, 10475, 3), device=self.device)}
        
        # Run inference
        if has_prompt_server:
            PromptServer.instance.send_sync("lhm.progress", {"value": 70, "text": "Running model inference..."})
            
        with torch.no_grad():
            results = self.model(image_tensor, motion_seqs)
        
        # Process results
        if has_prompt_server:
            PromptServer.instance.send_sync("lhm.progress", {"value": 90, "text": "Processing results..."})
            
        # Convert to ComfyUI format
        processed_image = results['processed_image'].permute(0, 2, 3, 1)  # [B, H, W, C]
        animation = results['animation'].permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]
        
        return processed_image, animation
            
    def apply_preview_scaling(self, processed_image, animation, scale):
        """Scale the results for preview purposes."""
        if scale != 1.0:
            # Scale the processed image
            if isinstance(processed_image, torch.Tensor):
                b, h, w, c = processed_image.shape
                new_h, new_w = int(h * scale), int(w * scale)
                # Need to convert to channels-first for interpolate
                processed_image = processed_image.permute(0, 3, 1, 2)
                processed_image = torch.nn.functional.interpolate(
                    processed_image, size=(new_h, new_w), mode='bilinear'
                )
                # Convert back to channels-last
                processed_image = processed_image.permute(0, 2, 3, 1)
            
            # Scale the animation frames
            if animation is not None and isinstance(animation, torch.Tensor):
                b, f, h, w, c = animation.shape
                new_h, new_w = int(h * scale), int(w * scale)
                # Reshape to batch of images and convert to channels-first
                animation = animation.reshape(b * f, h, w, c).permute(0, 3, 1, 2)
                animation = torch.nn.functional.interpolate(
                    animation, size=(new_h, new_w), mode='bilinear'
                )
                # Convert back to channels-last and reshape to animation
                animation = animation.permute(0, 2, 3, 1).reshape(b, f, new_h, new_w, c)
        
        return processed_image, animation 


NODE_CLASS_MAPPINGS = {
    "LHM": LHMReconstructionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LHM": "Large Animatable Human Model",
}