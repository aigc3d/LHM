import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import comfy.model_management as model_management
from comfy.cli import args
from rembg import remove
from omegaconf import OmegaConf

# Add LHM project to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    prepare_motion_seqs,
)

class LHMReconstructionNode:
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
            }
        }
    
    RETURN_TYPES = ("IMAGE", "VIDEO", "MESH")
    RETURN_NAMES = ("processed_image", "animation", "3d_mesh")
    FUNCTION = "reconstruct_human"
    CATEGORY = "LHM"
    
    def __init__(self):
        self.model = None
        self.device = model_management.get_torch_device()
        self.dtype = model_management.unet_dtype()
        self.pose_estimator = None
        self.face_detector = None
        self.parsing_net = None
        self.cfg = None
        
    def reconstruct_human(self, input_image, model_version, motion_path, export_mesh, remove_background, recenter):
        # Convert input_image to numpy array
        if isinstance(input_image, torch.Tensor):
            input_image = input_image.cpu().numpy()
        
        # Convert to PIL Image for preprocessing
        input_image = Image.fromarray((input_image * 255).astype(np.uint8))
        
        # Initialize components if not already loaded
        if self.model is None:
            self.initialize_components(model_version)
        
        # Preprocess image
        processed_image = self.preprocess_image(input_image, remove_background, recenter)
        
        # Run inference
        results = self.run_inference(processed_image, motion_path, export_mesh)
        
        return results

    def initialize_components(self, model_version):
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "configs", f"{model_version.lower()}.yaml")
        self.cfg = OmegaConf.load(config_path)
        
        # Initialize pose estimator
        self.pose_estimator = PoseEstimator()
        
        # Initialize face detector and parsing network
        try:
            from engine.SegmentAPI.SAM import SAM2Seg
            self.face_detector = SAM2Seg()
        except ImportError:
            print("Warning: SAM2 not found, using rembg for background removal")
            self.face_detector = None
        
        # Load LHM model
        self.model = self.load_lhm_model(model_version)

    def preprocess_image(self, image, remove_background, recenter):
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Remove background if requested
        if remove_background:
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
            image_np = center_crop_according_to_mask(image_np, mask)
        
        # Convert back to PIL Image
        return Image.fromarray(image_np)

    def load_lhm_model(self, model_version):
        # Load model weights
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "checkpoints", f"{model_version.lower()}.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        # Load model using the configuration
        model = self._build_model(self.cfg)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model

    def _build_model(self, cfg):
        # Implement model building logic based on the configuration
        # This should match the model building in app.py
        pass

    def run_inference(self, processed_image, motion_path, export_mesh):
        # Convert processed image to tensor
        image_tensor = torch.from_numpy(np.array(processed_image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Prepare motion sequence
        motion_seqs = prepare_motion_seqs(motion_path)
        
        # Run inference
        with torch.no_grad():
            results = self.model(image_tensor, motion_seqs)
        
        # Process results
        processed_image = results['processed_image']
        animation = results['animation']
        mesh = results['mesh'] if export_mesh else None
        
        return processed_image, animation, mesh

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "LHMReconstructionNode": LHMReconstructionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LHMReconstructionNode": "LHM Human Reconstruction"
} 