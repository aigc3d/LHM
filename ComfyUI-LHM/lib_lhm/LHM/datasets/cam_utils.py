# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import torch

"""
R: (N, 3, 3)
T: (N, 3)
E: (N, 4, 4)
vector: (N, 3)
"""


def compose_extrinsic_R_T(R: torch.Tensor, T: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from R and T.
    Batched I/O.
    """
    RT = torch.cat((R, T.unsqueeze(-1)), dim=-1)
    return compose_extrinsic_RT(RT)


def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([
        RT,
        torch.tensor([[[0, 0, 0, 1]]], dtype=RT.dtype, device=RT.device).repeat(RT.shape[0], 1, 1)
        ], dim=1)


def decompose_extrinsic_R_T(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into R and T.
    Batched I/O.
    """
    RT = decompose_extrinsic_RT(E)
    return RT[:, :, :3], RT[:, :, 3]


def decompose_extrinsic_RT(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into RT.
    Batched I/O.
    """
    return E[:, :3, :]


def camera_normalization_objaverse(normed_dist_to_center, poses: torch.Tensor, ret_transform: bool = False):
    assert normed_dist_to_center is not None
    pivotal_pose = compose_extrinsic_RT(poses[:1])
    dist_to_center = pivotal_pose[:, :3, 3].norm(dim=-1, keepdim=True).item() \
        if normed_dist_to_center == 'auto' else normed_dist_to_center

    # compute camera norm (new version)
    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 0, -1, -dist_to_center],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]], dtype=torch.float32)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    # normalize all views
    poses = compose_extrinsic_RT(poses)
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)
    poses = decompose_extrinsic_RT(poses)

    if ret_transform:
        return poses, camera_norm_matrix.squeeze(dim=0)
    return poses


def get_normalized_camera_intrinsics(intrinsics: torch.Tensor):
    """
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    Return batched fx, fy, cx, cy
    """
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]
    width, height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]
    fx, fy = fx / width, fy / height
    cx, cy = cx / width, cy / height
    return fx, fy, cx, cy


def build_camera_principle(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    return torch.cat([
        RT.reshape(-1, 12),
        fx.unsqueeze(-1), fy.unsqueeze(-1), cx.unsqueeze(-1), cy.unsqueeze(-1),
    ], dim=-1)


def build_camera_standard(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    E = compose_extrinsic_RT(RT)
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    I = torch.stack([
        torch.stack([fx, torch.zeros_like(fx), cx], dim=-1),
        torch.stack([torch.zeros_like(fy), fy, cy], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=torch.float32, device=RT.device).repeat(RT.shape[0], 1),
    ], dim=1)
    return torch.cat([
        E.reshape(-1, 16),
        I.reshape(-1, 9),
    ], dim=-1)


def center_looking_at_camera_pose(
    camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None,
    device: torch.device = torch.device('cpu'),
    ):
    """
    camera_position: (M, 3)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4)
    """
    # by default, looking at the origin and world up is pos-z
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
    up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    z_axis = camera_position - look_at
    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)
    x_axis = torch.cross(up_world, z_axis)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True)
    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    return extrinsics


def surrounding_views_linspace(n_views: int, radius: float = 2.0, height: float = 0.8, device: torch.device = torch.device('cpu')):
    """
    n_views: number of surrounding views
    radius: camera dist to center
    height: height of the camera
    return: (M, 3, 4)
    """
    assert n_views > 0
    assert radius > 0

    theta = torch.linspace(-torch.pi / 2, 3 * torch.pi / 2, n_views, device=device)
    projected_radius = math.sqrt(radius ** 2 - height ** 2)
    x = torch.cos(theta) * projected_radius
    y = torch.sin(theta) * projected_radius
    z = torch.full((n_views,), height, device=device)

    camera_positions = torch.stack([x, y, z], dim=1)
    extrinsics = center_looking_at_camera_pose(camera_positions, device=device)

    return extrinsics


def create_intrinsics(
    f: float,
    c: float = None, cx: float = None, cy: float = None,
    w: float = 1., h: float = 1.,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    ):
    """
    return: (3, 2)
    """
    fx = fy = f
    if c is not None:
        assert cx is None and cy is None, "c and cx/cy cannot be used together"
        cx = cy = c
    else:
        assert cx is not None and cy is not None, "cx/cy must be provided when c is not provided"
    fx, fy, cx, cy, w, h = fx/w, fy/h, cx/w, cy/h, 1., 1.
    intrinsics = torch.tensor([
        [fx, fy],
        [cx, cy],
        [w, h],
    ], dtype=dtype, device=device)
    return intrinsics
