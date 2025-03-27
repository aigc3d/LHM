import os
import torch

from pose_utils.camera import get_focalLength_from_fieldOfView


def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device("cuda")):
    """Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0, 0], K[1, 1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
        K[0, -1], K[1, -1] = p_x * img_size, p_y * img_size
    else:
        K[0, -1], K[1, -1] = img_size // 2, img_size // 2

    # Add batch dimension
    K = K.unsqueeze(0).to(device)
    return K
