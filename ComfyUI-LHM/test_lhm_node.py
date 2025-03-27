import torch

class LHMTestNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "LHM"
    
    def process_image(self, image):
        print("LHM Test Node is working!")
        return (image,)

NODE_CLASS_MAPPINGS = {
    "LHMTestNode": LHMTestNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LHMTestNode": "LHM Test Node"
}