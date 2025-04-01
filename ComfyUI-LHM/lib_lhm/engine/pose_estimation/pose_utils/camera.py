# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import numpy as np
import math
import torch

OPENCV_TO_OPENGL_CAMERA_CONVENTION = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def perspective_projection(x, K):
    """
    This function computes the perspective projection of a set of points assuming the extrinsinc params have already been applied
    Args:
        - x [bs,N,3]: 3D points
        - K [bs,3,3]: Camera instrincs params
    """
    # Apply perspective distortion
    y = x / x[:, :, -1].unsqueeze(-1)  # (bs, N, 3)

    # Apply camera intrinsics
    y = torch.einsum("bij,bkj->bki", K, y)  # (bs, N, 3)

    return y[:, :, :2]


def inverse_perspective_projection(points, K, distance):
    """
    This function computes the inverse perspective projection of a set of points given an estimated distance.
    Input:
        points (bs, N, 2): 2D points
        K (bs,3,3): camera intrinsics params
        distance (bs, N, 1): distance in the 3D world
    Similar to:
        - pts_l_norm = cv2.undistortPoints(np.expand_dims(pts_l, axis=1), cameraMatrix=K_l, distCoeffs=None)
    """
    # Apply camera intrinsics
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    points = torch.einsum("bij,bkj->bki", torch.inverse(K), points)

    # Apply perspective distortion
    if distance == None:
        return points
    points = points * distance
    return points


def get_focalLength_from_fieldOfView(fov=60, img_size=512):
    """
    Compute the focal length of the camera lens by assuming a certain FOV for the entire image
    Args:
        - fov: float, expressed in degree
        - img_size: int
    Return:
        focal: float
    """
    focal = img_size / (2 * np.tan(np.radians(fov) / 2))
    return focal


def focal_length_normalization(x, f, fovn=60, img_size=448):
    """
    Section 3.1 of https://arxiv.org/pdf/1904.02028.pdf
    E = (fn/f) * E' where E is 1/d
    """
    fn = get_focalLength_from_fieldOfView(fov=fovn, img_size=img_size)
    y = x * (fn / f)
    return y


def undo_focal_length_normalization(y, f, fovn=60, img_size=448):
    """
    Undo focal_length_normalization()
    """
    fn = get_focalLength_from_fieldOfView(fov=fovn, img_size=img_size)
    x = y * (f / fn)
    return x


EPS_LOG = 1e-10


def log_depth(x, eps=EPS_LOG):
    """
    Move depth to log space
    """
    return torch.log(x + eps)


def undo_log_depth(y, eps=EPS_LOG):
    """
    Undo log_depth()
    """
    return torch.exp(y) - eps
