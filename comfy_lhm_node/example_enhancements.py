import os
import sys
import torch
import numpy as np
import importlib.util
import comfy.model_management as model_management

"""
LHM ComfyUI Node - Enhancement Examples and Instructions

This file contains examples and instructions for enhancing the LHM ComfyUI node implementation.
It is based on best practices from the ComfyUI framework and should be used as a reference
when improving the current implementation.
"""

# -------------------------------------------------------------------------
# 1. Enhanced Node Implementation with Proper Docstrings
# -------------------------------------------------------------------------

class EnhancedLHMReconstructionNode:
    """
    LHM Human Reconstruction Node

    This node performs 3D human reconstruction using the LHM (Large Human Model) 
    from a single input image. It supports motion sequence integration and 3D mesh export.

    Class methods
    -------------
    INPUT_TYPES (dict):
        Defines input parameters for the node.
    IS_CHANGED:
        Controls when the node is re-executed.
    check_lazy_status:
        Conditional evaluation of lazy inputs.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The types of each element in the output tuple.
    RETURN_NAMES (`tuple`):
        The names of each output in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method.
    CATEGORY (`str`):
        The category under which the node appears in the UI.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input types for the LHM Reconstruction node.
        
        Returns: `dict`:
            - Key input_fields_group (`string`): Either required, hidden or optional
            - Value input_fields (`dict`): Input fields config with field names and types
        """
        return {
            "required": {
                "input_image": ("IMAGE",),
                "model_version": (["LHM-0.5B", "LHM-1B"], {
                    "default": "LHM-0.5B",
                    "lazy": False  # Model loading is resource-intensive, should happen immediately
                }),
                "motion_path": ("STRING", {
                    "default": "./train_data/motion_video/mimo1/smplx_params",
                    "multiline": False,
                    "lazy": True  # Only load motion data when needed
                }),
                "export_mesh": ("BOOLEAN", {
                    "default": False,
                    "lazy": True  # Only generate mesh when needed
                }),
                "remove_background": ("BOOLEAN", {
                    "default": True,
                    "lazy": True  # Can be lazy as preprocessing depends on this
                }),
                "recenter": ("BOOLEAN", {
                    "default": True,
                    "lazy": True  # Can be lazy as preprocessing depends on this
                })
            },
            "optional": {
                "cache_dir": ("STRING", {
                    "default": "./cache",
                    "multiline": False,
                    "lazy": True
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "COMFY_VIDEO", "MESH_DATA")  # Use custom types for non-standard outputs
    RETURN_NAMES = ("processed_image", "animation", "3d_mesh")
    FUNCTION = "reconstruct_human"
    CATEGORY = "LHM"
    
    def __init__(self):
        """Initialize the LHM Reconstruction node."""
        self.model = None
        self.device = model_management.get_torch_device()
        self.dtype = model_management.unet_dtype()
    
    def check_lazy_status(self, input_image, model_version, motion_path=None, 
                          export_mesh=None, remove_background=None, recenter=None, cache_dir=None):
        """
        Determine which lazy inputs need to be evaluated.
        
        This improves performance by only evaluating necessary inputs based on current state.
        
        Returns:
            list: Names of inputs that need to be evaluated
        """
        needed_inputs = []
        
        # We always need the image
        
        # If we're exporting mesh, we need motion data
        if export_mesh == True and motion_path is None:
            needed_inputs.append("motion_path")
        
        # If doing background removal, we need those parameters
        if remove_background is None:
            needed_inputs.append("remove_background")
            
        # Only need recenter if we're processing the image
        if remove_background == True and recenter is None:
            needed_inputs.append("recenter")
            
        return needed_inputs
    
    def reconstruct_human(self, input_image, model_version, motion_path, 
                         export_mesh, remove_background, recenter, cache_dir=None):
        """
        Perform human reconstruction from the input image.
        
        Args:
            input_image: Input image tensor
            model_version: LHM model version
            motion_path: Path to motion sequence
            export_mesh: Whether to export 3D mesh
            remove_background: Whether to remove background
            recenter: Whether to recenter the image
            cache_dir: Directory for caching results
            
        Returns:
            tuple: (processed_image, animation, 3d_mesh)
        """
        # Example implementation
        processed_image = input_image
        animation = torch.zeros((1, 3, 64, 64))  # Placeholder
        mesh = None if not export_mesh else {"vertices": [], "faces": []}
        
        return processed_image, animation, mesh
    
    @classmethod
    def IS_CHANGED(cls, input_image, model_version, motion_path, 
                  export_mesh, remove_background, recenter, cache_dir=None):
        """
        Control when the node should be re-executed even if inputs haven't changed.
        
        This is useful for nodes that depend on external factors like file changes.
        
        Returns:
            str: A value that when changed causes node re-execution
        """
        # Check if motion files have been modified
        if motion_path and os.path.exists(motion_path):
            try:
                # Get the latest modification time of any file in the motion directory
                latest_mod_time = max(
                    os.path.getmtime(os.path.join(root, file))
                    for root, _, files in os.walk(motion_path)
                    for file in files
                )
                return str(latest_mod_time)
            except Exception:
                pass
        return ""

# -------------------------------------------------------------------------
# 2. Custom Output Types Registration
# -------------------------------------------------------------------------

"""
To handle custom output types like VIDEO and MESH, you should register
custom types with ComfyUI. Here's how:

1. Define your custom types in the global scope:
"""

# Add these to your __init__.py file
class VideoOutput:
    """Custom class to represent video output type."""
    def __init__(self, video_tensor, fps=30):
        self.video_tensor = video_tensor
        self.fps = fps

class MeshOutput:
    """Custom class to represent 3D mesh output type."""
    def __init__(self, vertices, faces, textures=None):
        self.vertices = vertices
        self.faces = faces
        self.textures = textures

# -------------------------------------------------------------------------
# 3. Web Extensions for 3D Visualization
# -------------------------------------------------------------------------

"""
To add 3D visualization for your mesh outputs, create a web extension.
First, add this line to your __init__.py:

```python
WEB_DIRECTORY = "./web"
```

Then, create a ./web directory with your JS files for 3D visualization.
"""

# -------------------------------------------------------------------------
# 4. Error Handling and Validation
# -------------------------------------------------------------------------

def validate_inputs(input_image, model_version, motion_path, export_mesh):
    """
    Validate input parameters to ensure they're correct.
    
    Args:
        input_image: Input image tensor
        model_version: LHM model version
        motion_path: Path to motion sequence
        export_mesh: Whether to export 3D mesh
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Check input image
    if input_image is None or input_image.shape[0] == 0:
        raise ValueError("Input image is empty or invalid")
    
    # Check model version
    valid_models = ["LHM-0.5B", "LHM-1B"]
    if model_version not in valid_models:
        raise ValueError(f"Model version {model_version} not supported. Use one of {valid_models}")
    
    # Check motion path if using
    if export_mesh and (motion_path is None or not os.path.exists(motion_path)):
        raise ValueError(f"Motion path {motion_path} does not exist")
    
    return True

# -------------------------------------------------------------------------
# 5. Caching Implementation
# -------------------------------------------------------------------------

def download_model_weights(model_version, cache_path):
    """Download model weights from the official source."""
    from tqdm import tqdm
    import urllib.request
    
    model_urls = {
        'LHM-0.5B': 'https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/LHM-0.5B.tar',
        'LHM-1B': 'https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/LHM-1B.tar'
    }
    
    if model_version not in model_urls:
        raise ValueError(f"Unknown model version: {model_version}")
    
    url = model_urls[model_version]
    
    def report_progress(block_num, block_size, total_size):
        if total_size > 0:
            progress_bar.update(block_size)
    
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=None, 
              desc=f"Downloading {model_version}") as progress_bar:
        urllib.request.urlretrieve(url, cache_path, reporthook=report_progress)
    
    return cache_path

def implement_caching(model, model_version, cache_dir):
    """
    Implement model weight caching to improve performance.
    
    Args:
        model: Model name
        model_version: Model version
        cache_dir: Cache directory
        
    Returns:
        str: Path to cached model weights
    """
    if cache_dir is None:
        cache_dir = "./cache"
        
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if model is cached
    cache_path = os.path.join(cache_dir, f"{model_version.lower()}.pth")
    if not os.path.exists(cache_path):
        # Download model weights
        download_model_weights(model_version, cache_path)
        
    return cache_path

# -------------------------------------------------------------------------
# 6. Custom API Routes
# -------------------------------------------------------------------------

"""
To add custom API routes for your node, add this to your __init__.py:

```python
from aiohttp import web
from server import PromptServer
import asyncio

# Add API route to get model info
@PromptServer.instance.routes.get("/lhm/models")
async def get_lhm_models(request):
    return web.json_response({
        "models": ["LHM-0.5B", "LHM-1B"],
        "versions": {
            "LHM-0.5B": "1.0.0",
            "LHM-1B": "1.0.0"
        }
    })

# Add API route to download a model
@PromptServer.instance.routes.post("/lhm/download")
async def download_lhm_model(request):
    data = await request.json()
    model_version = data.get("model_version")
    
    if model_version not in ["LHM-0.5B", "LHM-1B"]:
        return web.json_response({"error": "Invalid model version"}, status=400)
    
    # Start download in background
    asyncio.create_task(download_model_task(model_version))
    
    return web.json_response({"status": "download_started"})
```
"""

# -------------------------------------------------------------------------
# 7. Progress Feedback Implementation
# -------------------------------------------------------------------------

"""
To provide progress feedback for long-running operations like model loading,
you can use the ComfyUI progress API. Add this to your methods:

```python
def load_lhm_model(self, model_version):
    from server import PromptServer
    
    # Create a progress callback
    progress_callback = PromptServer.instance.send_sync("progress", {"value": 0, "max": 100})
    
    try:
        # Update progress
        progress_callback({"value": 10, "text": "Loading model weights..."})
        
        # Load model weights
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "checkpoints", f"{model_version.lower()}.pth")
        
        progress_callback({"value": 30, "text": "Building model..."})
        
        # Build model
        model = self._build_model(self.cfg)
        
        progress_callback({"value": 60, "text": "Loading state dict..."})
        
        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        progress_callback({"value": 90, "text": "Moving model to device..."})
        
        # Move to device
        model.to(self.device)
        model.eval()
        
        progress_callback({"value": 100, "text": "Model loaded successfully"})
        
        return model
    except Exception as e:
        progress_callback({"value": 0, "text": f"Error loading model: {str(e)}"})
        raise
```
"""

# -------------------------------------------------------------------------
# 8. Insights from ComfyUI-ReActor Implementation
# -------------------------------------------------------------------------

"""
Based on examining the ComfyUI-ReActor node implementation, here are additional
patterns and features that would be beneficial for our LHM node:
"""

# 8.1 Improved Model Directory Management

def setup_model_directories():
    """
    Set up the model directories in the ComfyUI models directory structure.
    Based on ReActor's approach to directory management.
    """
    # Check if folder_paths is available in ComfyUI
    try:
        import folder_paths
    except ImportError:
        print("folder_paths module not available - running in test mode")
        return None, None
    
    models_dir = folder_paths.models_dir
    LHM_MODELS_PATH = os.path.join(models_dir, "lhm")
    MOTION_MODELS_PATH = os.path.join(LHM_MODELS_PATH, "motion")
    
    # Create directories if they don't exist
    os.makedirs(LHM_MODELS_PATH, exist_ok=True)
    os.makedirs(MOTION_MODELS_PATH, exist_ok=True)
    
    # Register directories with ComfyUI
    folder_paths.folder_names_and_paths["lhm_models"] = ([LHM_MODELS_PATH], folder_paths.supported_pt_extensions)
    folder_paths.folder_names_and_paths["lhm_motion"] = ([MOTION_MODELS_PATH], folder_paths.supported_pt_extensions)
    
    return LHM_MODELS_PATH, MOTION_MODELS_PATH

# 8.2 Advanced Tensor/Image Conversion Utilities

def tensor_to_video(video_tensor, fps=30):
    """
    Convert a tensor of shape [frames, channels, height, width] to a video file.
    Based on ReActor's tensor handling.
    
    Args:
        video_tensor: Tensor containing video frames
        fps: Frames per second
        
    Returns:
        str: Path to saved video file
    """
    import uuid
    import tempfile
    
    # Check if imageio is available
    try:
        import imageio
    except ImportError:
        print("imageio module not available - install with pip install imageio imageio-ffmpeg")
        return None
    
    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"lhm_video_{uuid.uuid4()}.mp4")
    
    # Convert tensor to numpy array
    if isinstance(video_tensor, torch.Tensor):
        video_np = video_tensor.cpu().numpy()
        video_np = (video_np * 255).astype(np.uint8)
    else:
        video_np = video_tensor
    
    # Write video
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in video_np:
            writer.append_data(frame.transpose(1, 2, 0))
    
    return video_path

# 8.3 Memory Management for Large Models

class ModelManager:
    """
    Manager for loading and unloading models to efficiently use GPU memory.
    Inspired by ReActor's approach to model management.
    """
    def __init__(self):
        self.loaded_models = {}
        self.current_model = None
        
    def load_model(self, model_name, model_path):
        """Load a model if not already loaded."""
        if model_name not in self.loaded_models:
            # Unload current model if memory is limited
            if self.current_model and hasattr(model_management, "get_free_memory"):
                if model_management.get_free_memory() < 2000:
                    self.unload_model(self.current_model)
            
            # Load new model
            model = self._load_model_from_path(model_path)
            self.loaded_models[model_name] = model
            self.current_model = model_name
        
        return self.loaded_models[model_name]
    
    def unload_model(self, model_name):
        """Unload a model to free memory."""
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            del self.loaded_models[model_name]
            
            # Force garbage collection
            import gc
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
            if self.current_model == model_name:
                self.current_model = None
    
    def _load_model_from_path(self, model_path):
        """Load model from path with appropriate handling."""
        # Example implementation
        return {"name": os.path.basename(model_path)}

# 8.4 Improved UI with ON/OFF Switches and Custom Labels

class ImprovedLHMNode:
    """Example node with improved UI elements."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "input_image": ("IMAGE",),
                "model_version": (["LHM-0.5B", "LHM-1B"], {"default": "LHM-0.5B"}),
                "advanced_options": ("BOOLEAN", {"default": False, "label_off": "Simple", "label_on": "Advanced"}),
                # More parameters...
            },
            "optional": {
                # Optional parameters shown when advanced_options is True
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "LHM"
    
    def process(self, enabled, input_image, model_version, advanced_options):
        """Process the input image."""
        if not enabled:
            return (input_image,)
        
        # Process the image...
        return (input_image,)

# 8.5 Download Utilities with Progress Reporting

def download_model_weights_with_progress(model_url, save_path, model_name):
    """
    Download model weights with progress reporting.
    Based on ReActor's download function.
    
    Args:
        model_url: URL to download from
        save_path: Path to save the downloaded file
        model_name: Name of the model for display
    """
    # Check if tqdm is available
    try:
        from tqdm import tqdm
    except ImportError:
        print("tqdm module not available - install with pip install tqdm")
        return download_without_progress(model_url, save_path)
    
    import urllib.request
    
    def report_progress(block_num, block_size, total_size):
        if total_size > 0:
            progress_bar.update(block_size)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download with progress bar
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=None, 
              desc=f"Downloading {model_name}") as progress_bar:
        urllib.request.urlretrieve(model_url, save_path, reporthook=report_progress)
    
    return save_path

def download_without_progress(model_url, save_path):
    """Fallback download function without progress reporting."""
    import urllib.request
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download without progress bar
    urllib.request.urlretrieve(model_url, save_path)
    
    return save_path

# 8.6 Custom Type Handling for Complex Outputs

# Register custom types in ComfyUI
def register_lhm_types():
    """Register custom LHM types with ComfyUI."""
    try:
        import comfy.utils
        
        # Check if type is already registered
        if hasattr(comfy.utils, "VIDEO_TYPE"):
            return
        
        # Register video type
        setattr(comfy.utils, "VIDEO_TYPE", "LHM_VIDEO")
        
        # Register mesh type
        setattr(comfy.utils, "MESH_TYPE", "LHM_MESH")
    except ImportError:
        print("comfy.utils module not available - running in test mode")

# 8.7 Modular Node Design

class LHMModelLoader:
    """Node for loading LHM models separately from processing."""
    RETURN_TYPES = ("LHM_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "LHM"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["LHM-0.5B", "LHM-1B"], {"default": "LHM-0.5B"}),
            }
        }
    
    def load_model(self, model_version):
        """Load the specified model version."""
        # Example implementation
        return ({"version": model_version, "loaded": True},)

class LHMReconstruction:
    """Node for reconstruction using a pre-loaded model."""
    RETURN_TYPES = ("IMAGE", "LHM_VIDEO", "LHM_MESH")
    FUNCTION = "reconstruct"
    CATEGORY = "LHM"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "lhm_model": ("LHM_MODEL",),
                # Other parameters...
            }
        }
    
    def reconstruct(self, input_image, lhm_model):
        """Reconstruct a 3D human from the input image."""
        # Example implementation
        return input_image, torch.zeros((1, 3, 64, 64)), {"vertices": [], "faces": []}

# These additions provide a comprehensive set of enhancements based on the
# patterns observed in the ComfyUI-ReActor implementation. 

# If this file is run directly, perform a simple test
if __name__ == "__main__":
    print("LHM ComfyUI Node - Enhancement Examples")
    print("This file contains examples and instructions for enhancing the LHM ComfyUI node implementation.")
    print("It is meant to be imported, not run directly.") 