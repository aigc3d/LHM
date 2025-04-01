import math
import pdb

import torch

try:
    from gsplat.rendering import rasterization

    gsplat_enable = True
except:
    gsplat_enable = False

from LHM.models.rendering.gs_renderer import Camera, GaussianModel, GS3DRenderer
from LHM.models.rendering.utils.sh_utils import eval_sh
from LHM.models.rendering.utils.typing import *

# self.xyz: Tensor = xyz
# self.opacity: Tensor = opacity
# self.rotation: Tensor = rotation
# self.scaling: Tensor = scaling
# self.shs: Tensor = shs  # [B, SH_Coeff, 3]


class GSPlatRenderer(GS3DRenderer):
    """Backed from GS3D, support batch-wise rendering of Gaussian splats."""

    def __init__(self, **params):
        if gsplat_enable is False:
            raise ImportError("GSPlat is not installed, please install it first.")
        else:
            super(GSPlatRenderer, self).__init__(**params)

    def get_gaussians_properties(self, viewpoint_camera, gaussian_model):

        xyz = gaussian_model.xyz
        opacity = gaussian_model.opacity
        scales = gaussian_model.scaling
        rotations = gaussian_model.rotation
        cov3D_precomp = None
        shs = None
        if gaussian_model.use_rgb:
            colors_precomp = gaussian_model.shs
        else:
            raise NotImplementedError
            # shs = gaussian_model.shs

            # shs_view = gaussian_model.get_features.transpose(1, 2).view(
            #     -1, 3, (gaussian_model.max_sh_degree + 1) ** 2
            # )
            # dir_pp = gaussian_model.get_attribute(
            #     "xyz"
            # ) - viewpoint_camera.camera_center.repeat(
            #     gaussian_model.get_features.shape[0], 1
            # )
            # dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(
            #     gaussian_model.active_sh_degree, shs_view, dir_pp_normalized
            # )
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp

    def forward_single_view(
        self,
        gaussian_model: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
    ):

        xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp = (
            self.get_gaussians_properties(viewpoint_camera, gaussian_model)
        )

        intrinsics = viewpoint_camera.intrinsic
        extrinsics = viewpoint_camera.world_view_transform.transpose(
            0, 1
        ).contiguous()  # c2w -> w2c

        img_height = int(viewpoint_camera.height)
        img_width = int(viewpoint_camera.width)

        colors_precomp = colors_precomp.squeeze(1)
        opacity = opacity.squeeze(1)

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            render_rgbd, render_alphas, meta = rasterization(
                means=xyz.float(),
                quats=rotations.float(),
                scales=scales.float(),
                opacities=opacity.float(),
                colors=colors_precomp.float(),
                viewmats=extrinsics.unsqueeze(0).float(),
                Ks=intrinsics.float().unsqueeze(0)[:, :3, :3],
                width=img_width,
                height=img_height,
                near_plane=viewpoint_camera.znear,
                far_plane=viewpoint_camera.zfar,
                # radius_clip=3.0,
                eps2d=0.3,  # 3 pixel
                render_mode="RGB+D",
                backgrounds=background_color.unsqueeze(0).float(),
                camera_model="pinhole",
            )

        render_rgbd = render_rgbd.squeeze(0)
        render_alphas = render_alphas.squeeze(0)

        rendered_image = render_rgbd[:, :, :3]
        rendered_depth = render_rgbd[:, :, 3:]

        ret = {
            "comp_rgb": rendered_image,  # [H, W, 3]
            "comp_rgb_bg": background_color,
            "comp_mask": render_alphas,
            "comp_depth": rendered_depth,
        }

        # if ret_mask:
        #     mask_bg_color = torch.zeros(3, dtype=torch.float32, device=self.device)
        #     raster_settings = GaussianRasterizationSettings(
        #         image_height=int(viewpoint_camera.height),
        #         image_width=int(viewpoint_camera.width),
        #         tanfovx=tanfovx,
        #         tanfovy=tanfovy,
        #         bg=mask_bg_color,
        #         scale_modifier=self.scaling_modifier,
        #         viewmatrix=viewpoint_camera.world_view_transform,
        #         projmatrix=viewpoint_camera.full_proj_transform.float(),
        #         sh_degree=0,
        #         campos=viewpoint_camera.camera_center,
        #         prefiltered=False,
        #         debug=False
        #     )
        #     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        #     with torch.autocast(device_type=self.device.type, dtype=torch.float32):
        #         rendered_mask, radii = rasterizer(
        #             means3D = means3D,
        #             means2D = means2D,
        #             # shs = ,
        #             colors_precomp = torch.ones_like(means3D),
        #             opacities = opacity,
        #             scales = scales,
        #             rotations = rotations,
        #             cov3D_precomp = cov3D_precomp)
        #         ret["comp_mask"] = rendered_mask.permute(1, 2, 0)

        return ret
