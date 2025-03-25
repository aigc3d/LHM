"""
PyTorch3D-Lite

A minimal implementation of the essential functions from PyTorch3D
needed for the LHM node to work.
"""

import torch
import math
import numpy as np

def matrix_to_rotation_6d(matrix):
    """
    Convert rotation matrices to 6D rotation representation.
    
    Args:
        matrix: (..., 3, 3) rotation matrices
        
    Returns:
        (..., 6) 6D rotation representation
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def rotation_6d_to_matrix(d6):
    """
    Convert 6D rotation representation to rotation matrix.
    
    Args:
        d6: (..., 6) 6D rotation representation
        
    Returns:
        (..., 3, 3) rotation matrices
    """
    batch_dim = d6.shape[:-1]
    d6 = d6.reshape(batch_dim + (2, 3))
    
    x_raw = d6[..., 0, :]
    y_raw = d6[..., 1, :]
    
    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    
    matrix = torch.stack([x, y, z], dim=-2)
    return matrix


def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix.
    
    Args:
        axis_angle: (..., 3) axis-angle representation
        
    Returns:
        (..., 3, 3) rotation matrices
    """
    batch_dims = axis_angle.shape[:-1]
    
    theta = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (theta + 1e-8)
    
    cos = torch.cos(theta)[..., None]
    sin = torch.sin(theta)[..., None]
    
    K = _skew_symmetric_matrix(axis)
    rotation_matrix = (
        torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).view(
            *[1 for _ in range(len(batch_dims))], 3, 3
        )
        + sin * K
        + (1 - cos) * torch.bmm(K, K)
    )
    
    return rotation_matrix


def matrix_to_axis_angle(matrix):
    """
    Convert rotation matrix to axis-angle representation.
    
    Args:
        matrix: (..., 3, 3) rotation matrices
        
    Returns:
        (..., 3) axis-angle representation
    """
    batch_dims = matrix.shape[:-2]
    
    # Ensure the matrix is a valid rotation matrix
    matrix = _normalize_rotation_matrix(matrix)
    
    cos_angle = (torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1) - 1) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    
    # For angles close to 0 or π, we need special handling
    near_zero = torch.abs(angle) < 1e-6
    near_pi = torch.abs(angle - math.pi) < 1e-6
    
    # For near-zero angles, the axis doesn't matter, return small values
    axis_zero = torch.zeros_like(matrix[..., 0])
    
    # For angles near π, we need to find the eigenvector for eigenvalue 1
    axis_pi = _get_axis_for_near_pi_rotation(matrix)
    
    # For general case, use standard formula
    sin_angle = torch.sin(angle.unsqueeze(-1))
    mask = (torch.abs(sin_angle) > 1e-6).squeeze(-1)
    axis_general = torch.empty_like(matrix[..., 0])
    
    if mask.any():
        # (matrix - matrix.transpose(-1, -2)) / (2 * sin_angle)
        axis_general[mask] = torch.stack([
            matrix[mask, 2, 1] - matrix[mask, 1, 2],
            matrix[mask, 0, 2] - matrix[mask, 2, 0],
            matrix[mask, 1, 0] - matrix[mask, 0, 1]
        ], dim=-1) / (2 * sin_angle[mask])
    
    # Combine the results based on conditions
    axis = torch.where(near_zero.unsqueeze(-1), axis_zero, 
                      torch.where(near_pi.unsqueeze(-1), axis_pi, axis_general))
    
    return angle.unsqueeze(-1) * axis


def _skew_symmetric_matrix(vector):
    """
    Create a skew-symmetric matrix from a 3D vector.
    
    Args:
        vector: (..., 3) vector
        
    Returns:
        (..., 3, 3) skew-symmetric matrices
    """
    batch_dims = vector.shape[:-1]
    
    v0 = vector[..., 0]
    v1 = vector[..., 1]
    v2 = vector[..., 2]
    
    zero = torch.zeros_like(v0)
    
    matrix = torch.stack([
        torch.stack([zero, -v2, v1], dim=-1),
        torch.stack([v2, zero, -v0], dim=-1),
        torch.stack([-v1, v0, zero], dim=-1),
    ], dim=-2)
    
    return matrix


def _normalize_rotation_matrix(matrix):
    """
    Ensure the matrix is a valid rotation matrix by normalizing.
    
    Args:
        matrix: (..., 3, 3) matrix
        
    Returns:
        (..., 3, 3) normalized rotation matrix
    """
    u, _, v = torch.svd(matrix)
    rotation = torch.matmul(u, v.transpose(-1, -2))
    
    # Handle reflection case (det = -1)
    det = torch.linalg.det(rotation)
    correction = torch.ones_like(det)
    correction[det < 0] = -1
    
    # Apply correction to the last column
    v_prime = v.clone()
    v_prime[..., :, 2] = v[..., :, 2] * correction.unsqueeze(-1)
    rotation = torch.matmul(u, v_prime.transpose(-1, -2))
    
    return rotation


def _get_axis_for_near_pi_rotation(matrix):
    """
    Find rotation axis for rotations with angles near π.
    
    Args:
        matrix: (..., 3, 3) rotation matrices
        
    Returns:
        (..., 3) axis vectors
    """
    batch_dims = matrix.shape[:-2]
    
    # The rotation axis is the eigenvector of the rotation matrix with eigenvalue 1
    # For a π rotation, the matrix is symmetric and M + I has the rotation axis in its null space
    M_plus_I = matrix + torch.eye(3, dtype=matrix.dtype, device=matrix.device).view(
        *[1 for _ in range(len(batch_dims))], 3, 3
    )
    
    # Find the column with the largest norm (least likely to be in the null space)
    col_norms = torch.norm(M_plus_I, dim=-2)
    _, max_idx = col_norms.max(dim=-1)
    
    # Create a mask to select the batch elements
    batch_size = torch.prod(torch.tensor(batch_dims)) if batch_dims else 1
    batch_indices = torch.arange(batch_size, device=matrix.device)
    
    # Reshape the matrix for easier indexing if needed
    if batch_dims:
        M_plus_I_flat = M_plus_I.reshape(-1, 3, 3)
        max_idx_flat = max_idx.reshape(-1)
    else:
        M_plus_I_flat = M_plus_I
        max_idx_flat = max_idx
    
    # Use the column with largest norm for cross product to find a vector in the null space
    axis = torch.empty(batch_size, 3, device=matrix.device)
    
    for i in range(batch_size):
        if max_idx_flat[i] == 0:
            v1 = M_plus_I_flat[i, :, 1]
            v2 = M_plus_I_flat[i, :, 2]
        elif max_idx_flat[i] == 1:
            v1 = M_plus_I_flat[i, :, 0]
            v2 = M_plus_I_flat[i, :, 2]
        else:
            v1 = M_plus_I_flat[i, :, 0]
            v2 = M_plus_I_flat[i, :, 1]
        
        # Cross product will be in the null space
        null_vec = torch.cross(v1, v2)
        norm = torch.norm(null_vec)
        
        # Normalize if possible, otherwise use a default axis
        if norm > 1e-6:
            axis[i] = null_vec / norm
        else:
            # Fallback to a default axis if cross product is too small
            axis[i] = torch.tensor([1.0, 0.0, 0.0], device=matrix.device)
    
    # Reshape back to original batch dimensions
    if batch_dims:
        axis = axis.reshape(*batch_dims, 3)
    
    return axis 