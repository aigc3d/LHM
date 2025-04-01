# ORIGINAL LICENSE
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Modified by Zexin He in 2023-2024.
# The modifications are subject to the same license as the original.


import itertools
import torch
import torch.nn as nn

from .utils.renderer import ImportanceRenderer
from .utils.ray_sampler import RaySampler


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.softplus(x - 1)


class OSGDecoder(nn.Module):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """
    def __init__(self, n_features: int,
                 hidden_dim: int = 64, num_layers: int = 4, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 1 + 3),
        )
        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    @torch.compile
    def forward(self, sampled_features, ray_directions):
        # Aggregate features by mean
        # sampled_features = sampled_features.mean(1)
        # Aggregate features by concatenation
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)
        x = sampled_features

        N, M, C = x.shape
        x = x.contiguous().view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]

        return {'rgb': rgb, 'sigma': sigma}


class TriplaneSynthesizer(nn.Module):
    """
    Synthesizer that renders a triplane volume with planes and a camera.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L19
    """

    DEFAULT_RENDERING_KWARGS = {
        'ray_start': 'auto',
        'ray_end': 'auto',
        'box_warp': 2.,
        'white_back': False,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'sampler_bbox_min': -1.,
        'sampler_bbox_max': 1.,
    }

    def __init__(self, triplane_dim: int, samples_per_ray: int):
        super().__init__()

        # attributes
        self.triplane_dim = triplane_dim
        self.rendering_kwargs = {
            **self.DEFAULT_RENDERING_KWARGS,
            'depth_resolution': samples_per_ray // 2,
            'depth_resolution_importance': samples_per_ray // 2,
        }

        # renderings
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()

        # modules
        self.decoder = OSGDecoder(n_features=triplane_dim)

    def forward(self, planes, cameras, anchors, resolutions, bg_colors, region_size: int):
        # planes: (N, 3, D', H', W')
        # cameras: (N, M, D_cam)
        # anchors: (N, M, 2)
        # resolutions: (N, M, 1)
        # bg_colors: (N, M, 1)
        # region_size: int
        assert planes.shape[0] == cameras.shape[0], "Batch size mismatch for planes and cameras"
        assert planes.shape[0] == anchors.shape[0], "Batch size mismatch for planes and anchors"
        assert cameras.shape[1] == anchors.shape[1], "Number of views mismatch for cameras and anchors"
        N, M = cameras.shape[:2]
        
        cam2world_matrix = cameras[..., :16].view(N, M, 4, 4)
        intrinsics = cameras[..., 16:25].view(N, M, 3, 3)

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(
            cam2world_matrix=cam2world_matrix.reshape(-1, 4, 4),
            intrinsics=intrinsics.reshape(-1, 3, 3),
            resolutions=resolutions.reshape(-1, 1),
            anchors=anchors.reshape(-1, 2),
            region_size=region_size,
        )
        assert N*M == ray_origins.shape[0], "Batch size mismatch for ray_origins"
        assert ray_origins.dim() == 3, "ray_origins should be 3-dimensional"

        # Perform volume rendering
        rgb_samples, depth_samples, weights_samples = self.renderer(
            planes.repeat_interleave(M, dim=0), self.decoder, ray_origins, ray_directions, self.rendering_kwargs,
            bg_colors=bg_colors.reshape(-1, 1),
        )

        # Reshape into 'raw' neural-rendered image
        Himg = Wimg = region_size
        rgb_images = rgb_samples.permute(0, 2, 1).reshape(N, M, rgb_samples.shape[-1], Himg, Wimg).contiguous()
        depth_images = depth_samples.permute(0, 2, 1).reshape(N, M, 1, Himg, Wimg)
        weight_images = weights_samples.permute(0, 2, 1).reshape(N, M, 1, Himg, Wimg)

        return {
            'images_rgb': rgb_images,
            'images_depth': depth_images,
            'images_weight': weight_images,
        }

    def forward_grid(self, planes, grid_size: int, aabb: torch.Tensor = None):
        # planes: (N, 3, D', H', W')
        # grid_size: int
        # aabb: (N, 2, 3)
        if aabb is None:
            aabb = torch.tensor([
                [self.rendering_kwargs['sampler_bbox_min']] * 3,
                [self.rendering_kwargs['sampler_bbox_max']] * 3,
            ], device=planes.device, dtype=planes.dtype).unsqueeze(0).repeat(planes.shape[0], 1, 1)
        assert planes.shape[0] == aabb.shape[0], "Batch size mismatch for planes and aabb"
        N = planes.shape[0]

        # create grid points for triplane query
        grid_points = []
        for i in range(N):
            grid_points.append(torch.stack(torch.meshgrid(
                torch.linspace(aabb[i, 0, 0], aabb[i, 1, 0], grid_size, device=planes.device),
                torch.linspace(aabb[i, 0, 1], aabb[i, 1, 1], grid_size, device=planes.device),
                torch.linspace(aabb[i, 0, 2], aabb[i, 1, 2], grid_size, device=planes.device),
                indexing='ij',
            ), dim=-1).reshape(-1, 3))
        cube_grid = torch.stack(grid_points, dim=0).to(planes.device)

        features = self.forward_points(planes, cube_grid)

        # reshape into grid
        features = {
            k: v.reshape(N, grid_size, grid_size, grid_size, -1)
            for k, v in features.items()
        }
        return features

    def forward_points(self, planes, points: torch.Tensor, chunk_size: int = 2**20):
        # planes: (N, 3, D', H', W')
        # points: (N, P, 3)
        N, P = points.shape[:2]

        # query triplane in chunks
        outs = []
        for i in range(0, points.shape[1], chunk_size):
            chunk_points = points[:, i:i+chunk_size]

            # query triplane
            chunk_out = self.renderer.run_model_activated(
                planes=planes,
                decoder=self.decoder,
                sample_coordinates=chunk_points,
                sample_directions=torch.zeros_like(chunk_points),
                options=self.rendering_kwargs,
            )
            outs.append(chunk_out)

        # concatenate the outputs
        point_features = {
            k: torch.cat([out[k] for out in outs], dim=1)
            for k in outs[0].keys()
        }
        return point_features
