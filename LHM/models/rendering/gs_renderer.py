import copy
import math
import os
import pdb
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms.rotation_conversions import quaternion_multiply

from LHM.models.rendering.smpl_x import SMPLXModel, read_smplx_param
from LHM.models.rendering.smpl_x_voxel_dense_sampling import SMPLXVoxelMeshModel
from LHM.models.rendering.utils.sh_utils import RGB2SH, SH2RGB
from LHM.models.rendering.utils.typing import *
from LHM.models.rendering.utils.utils import MLP, trunc_exp
from LHM.models.utils import LinerParameterTuner, StaticParameterTuner
from LHM.outputs.output import GaussianAppOutput


def auto_repeat_size(tensor, repeat_num, axis=0):
    repeat_size = [1] * tensor.dim()
    repeat_size[axis] = repeat_num
    return repeat_size


def aabb(xyz):
    return torch.min(xyz, dim=0).values, torch.max(xyz, dim=0).values


def inverse_sigmoid(x):

    if isinstance(x, float):
        x = torch.tensor(x).float()

    return torch.log(x / (1 - x))


def generate_rotation_matrix_y(degrees):
    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    R = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]

    return np.asarray(R, dtype=np.float32)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y


class Camera:
    def __init__(
        self,
        w2c,
        intrinsic,
        FoVx,
        FoVy,
        height,
        width,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width
        self.world_view_transform = w2c.transpose(0, 1)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(w2c.device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.intrinsic = intrinsic

    @staticmethod
    def from_c2w(c2w, intrinsic, height, width):
        w2c = torch.inverse(c2w)
        FoVx, FoVy = intrinsic_to_fov(
            intrinsic,
            w=torch.tensor(width, device=w2c.device),
            h=torch.tensor(height, device=w2c.device),
        )
        return Camera(
            w2c=w2c,
            intrinsic=intrinsic,
            FoVx=FoVx,
            FoVy=FoVy,
            height=height,
            width=width,
        )


class GaussianModel:

    def setup_functions(self):

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # rgb activation function
        self.rgb_activation = torch.sigmoid

    def __init__(self, xyz, opacity, rotation, scaling, shs, use_rgb=False) -> None:
        """
        Initializes the GSRenderer object.
        Args:
            xyz (Tensor): The xyz coordinates.
            opacity (Tensor): The opacity values.
            rotation (Tensor): The rotation values.
            scaling (Tensor): The scaling values.
            before_activate: if True, the output appearance is needed to process by activation function.
            shs (Tensor): The spherical harmonics coefficients.
            use_rgb (bool, optional): Indicates whether shs represents RGB values. Defaults to False.
        """

        self.setup_functions()

        self.xyz: Tensor = xyz
        self.opacity: Tensor = opacity
        self.rotation: Tensor = rotation
        self.scaling: Tensor = scaling
        self.shs: Tensor = shs  # [B, SH_Coeff, 3]

        self.use_rgb = use_rgb  # shs indicates rgb?

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]

        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        if self.use_rgb:
            shs = RGB2SH(self.shs)
        else:
            shs = self.shs

        features_dc = shs[:, :1]
        features_rest = shs[:, 1:]

        f_dc = (
            features_dc.float().detach().flatten(start_dim=1).contiguous().cpu().numpy()
        )
        f_rest = (
            features_rest.float()
            .detach()
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = (
            inverse_sigmoid(torch.clamp(self.opacity, 1e-3, 1 - 1e-3))
            .detach()
            .cpu()
            .numpy()
        )

        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):

        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]

        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        sh_degree = int(math.sqrt((len(extra_f_names) + 3) / 3)) - 1

        print("load sh degree: ", sh_degree)

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # 0, 3, 8, 15
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = torch.from_numpy(xyz).to(self.xyz)
        opacities = torch.from_numpy(opacities).to(self.opacity)
        rotation = torch.from_numpy(rots).to(self.rotation)
        scales = torch.from_numpy(scales).to(self.scaling)
        features_dc = torch.from_numpy(features_dc).to(self.shs)
        features_rest = torch.from_numpy(features_extra).to(self.shs)

        shs = torch.cat([features_dc, features_rest], dim=2)

        if self.use_rgb:
            shs = SH2RGB(shs)
        else:
            shs = shs

        self.xyz: Tensor = xyz
        self.opacity: Tensor = self.opacity_activation(opacities)
        self.rotation: Tensor = self.rotation_activation(rotation)
        self.scaling: Tensor = self.scaling_activation(scales)
        self.shs: Tensor = shs.permute(0, 2, 1)

        self.active_sh_degree = sh_degree

    def clone(self):
        xyz = self.xyz.clone()
        opacity = self.opacity.clone()
        rotation = self.rotation.clone()
        scaling = self.scaling.clone()
        shs = self.shs.clone()
        use_rgb = self.use_rgb
        return GaussianModel(xyz, opacity, rotation, scaling, shs, use_rgb)


class GSLayer(nn.Module):
    """W/O Activation Function"""

    def setup_functions(self):

        self.scaling_activation = trunc_exp  # proposed by torch-ngp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.rgb_activation = torch.sigmoid

    def __init__(
        self,
        in_channels,
        use_rgb,
        clip_scaling=0.2,
        init_scaling=-5.0,
        init_density=0.1,
        sh_degree=None,
        xyz_offset=True,
        restrict_offset=True,
        xyz_offset_max_step=None,
        fix_opacity=False,
        fix_rotation=False,
        use_fine_feat=False,
    ):
        super().__init__()
        self.setup_functions()

        if isinstance(clip_scaling, omegaconf.listconfig.ListConfig) or isinstance(
            clip_scaling, list
        ):
            self.clip_scaling_pruner = LinerParameterTuner(*clip_scaling)
        else:
            self.clip_scaling_pruner = StaticParameterTuner(clip_scaling)
        self.clip_scaling = self.clip_scaling_pruner.get_value(0)

        self.use_rgb = use_rgb
        self.restrict_offset = restrict_offset
        self.xyz_offset = xyz_offset
        self.xyz_offset_max_step = xyz_offset_max_step  # 1.2 / 32
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.use_fine_feat = use_fine_feat

        self.attr_dict = {
            "shs": (sh_degree + 1) ** 2 * 3,
            "scaling": 3,
            "xyz": 3,
            "opacity": None,
            "rotation": None,
        }
        if not self.fix_opacity:
            self.attr_dict["opacity"] = 1
        if not self.fix_rotation:
            self.attr_dict["rotation"] = 4

        self.out_layers = nn.ModuleDict()
        for key, out_ch in self.attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == "shs" and use_rgb:
                    out_ch = 3
                if key == "shs":
                    shs_out_ch = out_ch
                layer = nn.Linear(in_channels, out_ch)
            # initialize
            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            self.out_layers[key] = layer

        if self.use_fine_feat:
            fine_shs_layer = nn.Linear(in_channels, shs_out_ch)
            nn.init.constant_(fine_shs_layer.weight, 0)
            nn.init.constant_(fine_shs_layer.bias, 0)
            self.out_layers["fine_shs"] = fine_shs_layer

    def hyper_step(self, step):
        self.clip_scaling = self.clip_scaling_pruner.get_value(step)

    def constrain_forward(self, ret, constrain_dict):

        # body scaling constrain
        # gs_attr.scaling[is_constrain_body] = gs_attr.scaling[is_constrain_body].clamp(max=0.02)  # magic number, which is used to constrain 
        # hand opacity constrain 

        # force the hand's opacity to be 0.95
        # gs_attr.opacity[is_hand] = gs_attr.opacity[is_hand].clamp(min=0.95)

        # body scaling constrain
        is_constrain_body = constrain_dict['is_constrain_body']
        scaling = ret['scaling'] 
        scaling[is_constrain_body] = scaling[is_constrain_body].clamp(max = 0.02)
        ret['scaling'] = scaling

        return ret

    def forward(self, x, pts, x_fine=None, constrain_dict=None):
        assert len(x.shape) == 2
        ret = {}
        for k in self.attr_dict:
            layer = self.out_layers[k]

            v = layer(x)
            if k == "rotation":
                if self.fix_rotation:
                    v = matrix_to_quaternion(
                        torch.eye(3).type_as(x)[None, :, :].repeat(x.shape[0], 1, 1)
                    )  # constant rotation
                else:
                    # v = torch.nn.functional.normalize(v)
                    v = self.rotation_activation(v)
            elif k == "scaling":
                # v = trunc_exp(v)
                v = self.scaling_activation(v)

                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            elif k == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    # v = torch.sigmoid(v)
                    v = self.opacity_activation(v)
            elif k == "shs":
                if self.use_rgb:
                    # v = torch.sigmoid(v)
                    v = self.rgb_activation(v)

                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v_fine = torch.tanh(v_fine)
                        v = v + v_fine
                else:
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v = v + v_fine
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                # TODO check
                if self.restrict_offset:
                    max_step = self.xyz_offset_max_step
                    v = (torch.sigmoid(v) - 0.5) * max_step
                if self.xyz_offset:
                    pass
                else:
                    assert NotImplementedError
                    v = v + pts
                k = "offset_xyz"
            ret[k] = v

        ret["use_rgb"] = self.use_rgb

        if constrain_dict is not None:
            ret = self.constrain_forward(ret, constrain_dict)

        return GaussianAppOutput(**ret)


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )

        self.register_buffer("basis", e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)

        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(
            torch.cat([self.embed(input, self.basis), input], dim=2)
        )  # B x N x C
        embed = self.norm(embed)
        return embed


class CrossAttnBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for SparseLRM architecture.
    """

    # Block contains a cross-attention layer, a self-attention layer, and an MLP
    def __init__(
        self,
        inner_dim: int,
        cond_dim: int,
        num_heads: int,
        eps: float = None,
        attn_drop: float = 0.0,
        attn_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
        feedforward=False,
    ):
        super().__init__()
        # TODO check already apply normalization
        # self.norm_q = nn.LayerNorm(inner_dim, eps=eps)
        # self.norm_k = nn.LayerNorm(cond_dim, eps=eps)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            kdim=cond_dim,
            vdim=cond_dim,
            dropout=attn_drop,
            bias=attn_bias,
            batch_first=True,
        )

        self.mlp = None
        if feedforward:
            self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=inner_dim,
                num_heads=num_heads,
                dropout=attn_drop,
                bias=attn_bias,
                batch_first=True,
            )
            self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
            self.mlp = nn.Sequential(
                nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(mlp_drop),
                nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
                nn.Dropout(mlp_drop),
            )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = self.cross_attn(
            self.norm_q(x), self.norm_k(cond), cond, need_weights=False
        )[0]
        if self.mlp is not None:
            before_sa = self.norm2(x)
            x = (
                x
                + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
            )
            x = x + self.mlp(self.norm3(x))
        return x


class DecoderCrossAttn(nn.Module):
    def __init__(
        self, query_dim, context_dim, num_heads, mlp=False, decode_with_extra_info=None
    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.cross_attn = CrossAttnBlock(
            inner_dim=query_dim,
            cond_dim=context_dim,
            num_heads=num_heads,
            feedforward=mlp,
            eps=1e-5,
        )
        self.decode_with_extra_info = decode_with_extra_info
        if decode_with_extra_info is not None:
            if decode_with_extra_info["type"] == "dinov2p14_feat":
                context_dim = decode_with_extra_info["cond_dim"]
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=context_dim,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )
            elif decode_with_extra_info["type"] == "decoder_dinov2p14_feat":
                from LHM.models.encoders.dinov2_wrapper import Dinov2Wrapper

                self.encoder = Dinov2Wrapper(
                    model_name="dinov2_vits14_reg", freeze=False, encoder_feat_dim=384
                )
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=384,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )
            elif decode_with_extra_info["type"] == "decoder_resnet18_feat":
                from LHM.models.encoders.xunet_wrapper import XnetWrapper

                self.encoder = XnetWrapper(
                    model_name="resnet18", freeze=False, encoder_feat_dim=64
                )
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=64,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )

    def resize_image(self, image, multiply):
        B, _, H, W = image.shape
        new_h, new_w = (
            math.ceil(H / multiply) * multiply,
            math.ceil(W / multiply) * multiply,
        )
        image = F.interpolate(
            image, (new_h, new_w), align_corners=True, mode="bilinear"
        )
        return image

    def forward(self, pcl_query, pcl_latent, extra_info=None):
        out = self.cross_attn(pcl_query, pcl_latent)
        if self.decode_with_extra_info is not None:
            out_dict = {}
            out_dict["coarse"] = out
            if self.decode_with_extra_info["type"] == "dinov2p14_feat":
                out = self.cross_attn_color(out, extra_info["image_feats"])
                out_dict["fine"] = out
                return out_dict
            elif self.decode_with_extra_info["type"] == "decoder_dinov2p14_feat":
                img_feat = self.encoder(extra_info["image"])
                out = self.cross_attn_color(out, img_feat)
                out_dict["fine"] = out
                return out_dict
            elif self.decode_with_extra_info["type"] == "decoder_resnet18_feat":
                image = extra_info["image"]
                image = self.resize_image(image, multiply=32)
                img_feat = self.encoder(image)
                out = self.cross_attn_color(out, img_feat)
                out_dict["fine"] = out
                return out_dict
        return out


class GS3DRenderer(nn.Module):
    def __init__(
        self,
        human_model_path,
        subdivide_num,
        smpl_type,
        feat_dim,
        query_dim,
        use_rgb,
        sh_degree,
        xyz_offset_max_step,
        mlp_network_config,
        expr_param_dim,
        shape_param_dim,
        clip_scaling=0.2,
        cano_pose_type=0,
        decoder_mlp=False,
        skip_decoder=False,
        fix_opacity=False,
        fix_rotation=False,
        decode_with_extra_info=None,
        gradient_checkpointing=False,
        apply_pose_blendshape=False,
        dense_sample_pts=40000,  # only use for dense_smaple_smplx
    ):

        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.skip_decoder = skip_decoder
        self.smpl_type = smpl_type
        assert self.smpl_type in ["smplx", "smplx_0", "smplx_1", "smplx_2"]

        self.scaling_modifier = 1.0
        self.sh_degree = sh_degree

        if self.smpl_type == "smplx_0" or self.smpl_type == "smplx":
            # Using pytorch3d dense sampling
            self.smplx_model = SMPLXModel(
                human_model_path,
                gender="neutral",
                subdivide_num=subdivide_num,
                shape_param_dim=shape_param_dim,
                expr_param_dim=expr_param_dim,
                cano_pose_type=cano_pose_type,
                apply_pose_blendshape=apply_pose_blendshape,
            )
        elif self.smpl_type == "smplx_1":
            raise NotImplementedError("inference version does not support")
        elif self.smpl_type == "smplx_2":
            self.smplx_model = SMPLXVoxelMeshModel(
                human_model_path,
                gender="neutral",
                subdivide_num=subdivide_num,
                shape_param_dim=shape_param_dim,
                expr_param_dim=expr_param_dim,
                cano_pose_type=cano_pose_type,
                dense_sample_points=dense_sample_pts,
                apply_pose_blendshape=apply_pose_blendshape,
            )
        else:
            raise NotImplementedError

        if not self.skip_decoder:
            self.pcl_embed = PointEmbed(dim=query_dim)
            self.decoder_cross_attn = DecoderCrossAttn(
                query_dim=query_dim,
                context_dim=feat_dim,
                num_heads=1,
                mlp=decoder_mlp,
                decode_with_extra_info=decode_with_extra_info,
            )

        self.mlp_network_config = mlp_network_config

        # using to mapping transformer decode feature to regression features. as decode feature is processed by NormLayer.
        if self.mlp_network_config is not None:
            self.mlp_net = MLP(query_dim, query_dim, **self.mlp_network_config)

        self.gs_net = GSLayer(
            in_channels=query_dim,
            use_rgb=use_rgb,
            sh_degree=self.sh_degree,
            clip_scaling=clip_scaling,
            init_scaling=-5.0,
            init_density=0.1,
            xyz_offset=True,
            restrict_offset=True,
            xyz_offset_max_step=xyz_offset_max_step,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            use_fine_feat=(
                True
                if decode_with_extra_info is not None
                and decode_with_extra_info["type"] is not None
                else False
            ),
        )

    def hyper_step(self, step):
        self.gs_net.hyper_step(step)

    def forward_single_view(
        self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        bg_color = background_color
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.gs_net.use_rgb:
            colors_precomp = gs.shs.squeeze(1).float()
            shs = None
        else:
            colors_precomp = None
            shs = gs.shs.float()

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # NOTE that dadong tries to regress rgb not shs
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D=means3D.float(),
                means2D=means2D.float(),
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity.float(),
                scales=scales.float(),
                rotations=rotations.float(),
                cov3D_precomp=cov3D_precomp,
            )

        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),  # [H, W, 3]
            "comp_rgb_bg": bg_color,
            "comp_mask": rendered_alpha.permute(1, 2, 0),
            "comp_depth": rendered_depth.permute(1, 2, 0),
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

    def animate_gs_model(
        self, gs_attr: GaussianAppOutput, query_points, smplx_data, debug=False
    ):
        """
        query_points: [N, 3]
        """

        device = gs_attr.offset_xyz.device

        if debug:
            N = gs_attr.offset_xyz.shape[0]
            gs_attr.xyz = torch.ones_like(gs_attr.offset_xyz) * 0.0

            rotation = matrix_to_quaternion(
                torch.eye(3).float()[None, :, :].repeat(N, 1, 1)
            ).to(
                device
            )  # constant rotation
            opacity = torch.ones((N, 1)).float().to(device)  # constant opacity

            gs_attr.opacity = opacity
            gs_attr.rotation = rotation
            # gs_attr.scaling = torch.ones_like(gs_attr.scaling) * 0.05
            # print(gs_attr.shs.shape)

        # build cano_dependent_pose
        cano_smplx_data_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "expr",
            "trans",
        ]

        merge_smplx_data = dict()
        for cano_smplx_data_key in cano_smplx_data_keys:
            warp_data = smplx_data[cano_smplx_data_key]
            cano_pose = torch.zeros_like(warp_data[:1])

            if cano_smplx_data_key == "body_pose":
                # A-posed
                cano_pose[0, 15, -1] = -math.pi / 6
                cano_pose[0, 16, -1] = +math.pi / 6

            merge_pose = torch.cat([warp_data, cano_pose], dim=0)
            merge_smplx_data[cano_smplx_data_key] = merge_pose

        merge_smplx_data["betas"] = smplx_data["betas"]
        merge_smplx_data["transform_mat_neutral_pose"] = smplx_data[
            "transform_mat_neutral_pose"
        ]

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            mean_3d = (
                query_points + gs_attr.offset_xyz
            )  # [N, 3]  # canonical space offset.

            # matrix to warp predefined pose to zero-pose
            transform_mat_neutral_pose = merge_smplx_data[
                "transform_mat_neutral_pose"
            ]  # [55, 4, 4]
            num_view = merge_smplx_data["body_pose"].shape[0]  # [Nv, 21, 3]
            mean_3d = mean_3d.unsqueeze(0).repeat(num_view, 1, 1)  # [Nv, N, 3]
            query_points = query_points.unsqueeze(0).repeat(num_view, 1, 1)
            transform_mat_neutral_pose = transform_mat_neutral_pose.unsqueeze(0).repeat(
                num_view, 1, 1, 1
            )

            # print(mean_3d.shape, transform_mat_neutral_pose.shape, query_points.shape, smplx_data["body_pose"].shape, smplx_data["betas"].shape)
            mean_3d, transform_matrix = (
                self.smplx_model.transform_to_posed_verts_from_neutral_pose(
                    mean_3d,
                    merge_smplx_data,
                    query_points,
                    transform_mat_neutral_pose=transform_mat_neutral_pose,  # from predefined pose to zero-pose matrix
                    device=device,
                )
            )  # [B, N, 3]

            # rotation appearance from canonical space to view_posed
            num_view, N, _, _ = transform_matrix.shape
            transform_rotation = transform_matrix[:, :, :3, :3]

            rigid_rotation_matrix = torch.nn.functional.normalize(
                matrix_to_quaternion(transform_rotation), dim=-1
            )
            I = matrix_to_quaternion(torch.eye(3)).to(device)

            # inference constrain
            is_constrain_body = self.smplx_model.is_constrain_body
            rigid_rotation_matrix[:, is_constrain_body] = I
            rotation_neutral_pose = gs_attr.rotation.unsqueeze(0).repeat(num_view, 1, 1)

            # TODO do not move underarm gs

            # QUATERNION MULTIPLY
            rotation_pose_verts = quaternion_multiply(
                rigid_rotation_matrix, rotation_neutral_pose
            )
            # rotation_pose_verts = rotation_neutral_pose

        gs_list = []
        cano_gs_list = []
        for i in range(num_view):
            gs_copy = GaussianModel(
                xyz=mean_3d[i],
                opacity=gs_attr.opacity,
                # rotation=gs_attr.rotation,
                rotation=rotation_pose_verts[i],
                scaling=gs_attr.scaling,
                shs=gs_attr.shs,
                use_rgb=self.gs_net.use_rgb,
            )  # [N, 3]

            if i == num_view - 1:
                cano_gs_list.append(gs_copy)
            else:
                gs_list.append(gs_copy)

        return gs_list, cano_gs_list

    def forward_gs_attr(self, x, query_points, smplx_data, debug=False, x_fine=None):
        """
        x: [N, C] Float[Tensor, "Np Cp"],
        query_points: [N, 3] Float[Tensor, "Np 3"]
        """
        device = x.device
        if self.mlp_network_config is not None:
            # x is processed by LayerNorm
            x = self.mlp_net(x)
            if x_fine is not None:
                x_fine = self.mlp_net(x_fine)

        # NOTE that gs_attr contains offset xyz
        is_constrain_body = self.smplx_model.is_constrain_body
        is_hands =  self.smplx_model.is_rhand + self.smplx_model.is_lhand 

        constrain_dict=dict(
            is_constrain_body=is_constrain_body,
            is_hands=is_hands
        )

        gs_attr: GaussianAppOutput = self.gs_net(x, query_points, x_fine, constrain_dict)

        return gs_attr

    def get_query_points(self, smplx_data, device):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                # print(smplx_data["betas"].shape, smplx_data["face_offset"].shape, smplx_data["joint_offset"].shape)
                positions, _, transform_mat_neutral_pose = (
                    self.smplx_model.get_query_points(smplx_data, device=device)
                )  # [B, N, 3]
        smplx_data["transform_mat_neutral_pose"] = (
            transform_mat_neutral_pose  # [B, 55, 4, 4]
        )
        return positions, smplx_data

    def decoder_cross_attn_wrapper(self, pcl_embed, latent_feat, extra_info):
        # if self.training and self.gradient_checkpointing:
        #     def create_custom_forward(module):
        #         def custom_forward(*inputs):
        #             return module(*inputs)
        #         return custom_forward
        #     ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        #     gs_feats = torch.utils.checkpoint.checkpoint(
        #         create_custom_forward(self.decoder_cross_attn),
        #         pcl_embed.to(dtype=latent_feat.dtype),
        #         latent_feat,
        #         extra_info,
        #         **ckpt_kwargs,
        #     )
        # else:
        gs_feats = self.decoder_cross_attn(
            pcl_embed.to(dtype=latent_feat.dtype), latent_feat, extra_info
        )
        return gs_feats

    def query_latent_feat(
        self,
        positions: Float[Tensor, "*B N1 3"],
        smplx_data,
        latent_feat: Float[Tensor, "*B N2 C"],
        extra_info,
    ):
        device = latent_feat.device
        if self.skip_decoder:
            gs_feats = latent_feat
            assert positions is not None
        else:
            assert positions is None
            if positions is None:
                positions, smplx_data = self.get_query_points(smplx_data, device)

            with torch.autocast(device_type=device.type, dtype=torch.float32):
                pcl_embed = self.pcl_embed(positions)

            gs_feats = self.decoder_cross_attn_wrapper(
                pcl_embed, latent_feat, extra_info
            )

        return gs_feats, positions, smplx_data

    def forward_single_batch(
        self,
        gs_list: list[GaussianModel],
        c2ws: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "Nv 3"]],
        debug: bool = False,
    ):
        out_list = []
        self.device = gs_list[0].xyz.device

        for v_idx, (c2w, intrinsic) in enumerate(zip(c2ws, intrinsics)):
            out_list.append(
                self.forward_single_view(
                    gs_list[v_idx],
                    Camera.from_c2w(c2w, intrinsic, height, width),
                    background_color[v_idx],
                )
            )

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = gs_list

        # debug = True
        if debug:
            import cv2

            cv2.imwrite(
                "fuck.png",
                (out["comp_rgb"].detach().cpu().numpy()[0, ..., ::-1] * 255).astype(
                    np.uint8
                ),
            )

        return out

    @torch.no_grad()
    def forward_cano_batch(
        self,
        gs_list: list[GaussianModel],
        c2ws: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        background_color: Optional[Float[Tensor, "Nv 3"]],
        height: int = 512,
        width: int = 512,
        debug: bool = False,
    ):
        """using to visualization."""
        degree_list = [0, 90, 180, 270]
        out_list = []
        self.device = gs_list[0].xyz.device

        gs_list_copy = [gs_list[0].clone() for _ in range(len(degree_list))]

        rotation_gs_list = []

        for rotation_degree, gs in zip(degree_list, gs_list_copy):

            _R = torch.eye(3).to(gs.xyz)
            _R[-1, -1] *= -1
            _R[1, 1] *= -1

            self_R = torch.from_numpy(generate_rotation_matrix_y(rotation_degree)).to(
                _R
            )
            _R = self_R @ _R

            gs.xyz = (_R @ gs.xyz.T).T

            _min, _max = aabb(gs.xyz)
            center = (_min + _max) / 2
            gs.xyz -= center.unsqueeze(0)

            _R_quaternion = matrix_to_quaternion(_R)
            gs.rotation = quaternion_multiply(_R_quaternion, gs.rotation)

            gs.xyz[..., -1] += 2.5  # move to (0, 0, 3)
            rotation_gs_list.append(gs)

        intrinsics = torch.eye(4).to(intrinsics).unsqueeze(0)
        intrinsics[0, 0, 0] = width
        intrinsics[0, 1, 1] = height
        intrinsics[0, 0, 2] = width / 2
        intrinsics[0, 1, 2] = height / 2

        for v_idx, gs in enumerate(rotation_gs_list):
            out_list.append(
                self.forward_single_view(
                    rotation_gs_list[v_idx],
                    Camera.from_c2w(c2ws[0], intrinsics[0], height, width),
                    torch.ones_like(background_color[0]),
                )
            )

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = rotation_gs_list

        if debug:
            import cv2

            for i in range(4):
                cv2.imwrite(
                    f"fuck_{i}.png",
                    (out["comp_rgb"].detach().cpu().numpy()[i, ..., ::-1] * 255).astype(
                        np.uint8
                    ),
                )

        return out

    def get_single_batch_smpl_data(self, smpl_data, bidx):
        smpl_data_single_batch = {}
        for k, v in smpl_data.items():
            smpl_data_single_batch[k] = v[
                bidx
            ]  # e.g. body_pose: [B, N_v, 21, 3] -> [N_v, 21, 3]
            if k == "betas" or (k == "joint_offset") or (k == "face_offset"):
                smpl_data_single_batch[k] = v[
                    bidx : bidx + 1
                ]  # e.g. betas: [B, 100] -> [1, 100]
        return smpl_data_single_batch

    def get_single_view_smpl_data(self, smpl_data, vidx):
        smpl_data_single_view = {}
        for k, v in smpl_data.items():
            assert v.shape[0] == 1
            if (
                k == "betas"
                or (k == "joint_offset")
                or (k == "face_offset")
                or (k == "transform_mat_neutral_pose")
            ):
                smpl_data_single_view[k] = v  # e.g. betas: [1, 100] -> [1, 100]
            else:
                smpl_data_single_view[k] = v[
                    :, vidx : vidx + 1
                ]  # e.g. body_pose: [1, N_v, 21, 3] -> [1, 1, 21, 3]
        return smpl_data_single_view

    def forward_gs(
        self,
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np_q 3"],
        smplx_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        additional_features: Optional[dict] = None,
        debug: bool = False,
        **kwargs,
    ):

        batch_size = gs_hidden_features.shape[0]

        # obtain gs_features embedding, cur points position, and also smplx params
        query_gs_features, query_points, smplx_data = self.query_latent_feat(
            query_points, smplx_data, gs_hidden_features, additional_features
        )

        gs_attr_list = []
        for b in range(batch_size):
            if isinstance(query_gs_features, dict):
                gs_attr = self.forward_gs_attr(
                    query_gs_features["coarse"][b],
                    query_points[b],
                    None,
                    debug,
                    x_fine=query_gs_features["fine"][b],
                )
            else:
                gs_attr = self.forward_gs_attr(
                    query_gs_features[b], query_points[b], None, debug
                )
            gs_attr_list.append(gs_attr)

        return gs_attr_list, query_points, smplx_data

    def forward_animate_gs(
        self,
        gs_attr_list,
        query_points,
        smplx_data,
        c2w,
        intrinsic,
        height,
        width,
        background_color,
        debug=False,
        df_data=None,  # deepfashion-style dataset
    ):
        batch_size = len(gs_attr_list)
        out_list = []
        cano_out_list = []  # inference DO NOT use

        N_view = smplx_data["root_pose"].shape[1]

        for b in range(batch_size):
            gs_attr = gs_attr_list[b]
            query_pt = query_points[b]
            # len(animatable_gs_model_list) = num_view
            merge_animatable_gs_model_list, cano_gs_model_list = self.animate_gs_model(
                gs_attr,
                query_pt,
                self.get_single_batch_smpl_data(smplx_data, b),
                debug=debug,
            )

            animatable_gs_model_list = merge_animatable_gs_model_list[:N_view]

            assert len(animatable_gs_model_list) == c2w.shape[1]

            # gs render animated gs model.
            out_list.append(
                self.forward_single_batch(
                    animatable_gs_model_list,
                    c2w[b],
                    intrinsic[b],
                    height,
                    width,
                    background_color[b] if background_color is not None else None,
                    debug=debug,
                )
            )

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v, dim=0)
            else:
                out[k] = v

        out["comp_rgb"] = out["comp_rgb"].permute(
            0, 1, 4, 2, 3
        )  # [B, NV, H, W, 3] -> [B, NV, 3, H, W]
        out["comp_mask"] = out["comp_mask"].permute(
            0, 1, 4, 2, 3
        )  # [B, NV, H, W, 3] -> [B, NV, 1, H, W]
        out["comp_depth"] = out["comp_depth"].permute(
            0, 1, 4, 2, 3
        )  # [B, NV, H, W, 3] -> [B, NV, 1, H, W]
        return out

    def forward(
        self,
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np 3"],
        smplx_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        additional_features: Optional[Float[Tensor, "B C H W"]] = None,
        background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
        debug: bool = False,
        **kwargs,
    ):

        # need shape_params of smplx_data to get querty points and get "transform_mat_neutral_pose"
        # only forward gs params
        gs_attr_list, query_points, smplx_data = self.forward_gs(
            gs_hidden_features,
            query_points,
            smplx_data=smplx_data,
            additional_features=additional_features,
            debug=debug,
        )

        out = self.forward_animate_gs(
            gs_attr_list,
            query_points,
            smplx_data,
            c2w,
            intrinsic,
            height,
            width,
            background_color,
            debug,
            df_data=kwargs["df_data"],
        )
        out["gs_attr"] = gs_attr_list

        return out


def test():
    import cv2

    human_model_path = "./pretrained_models/human_model_files"
    smplx_data_root = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/gyeongsik/smplx_optimized/smplx_params_smoothed"
    shape_param_file = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/gyeongsik/smplx_optimized/shape_param.json"

    batch_size = 1
    device = "cuda"
    smplx_data, cam_param_list, ori_image_list = read_smplx_param(
        smplx_data_root=smplx_data_root, shape_param_file=shape_param_file, batch_size=2
    )
    smplx_data_tmp = smplx_data
    for k, v in smplx_data.items():
        smplx_data_tmp[k] = v.unsqueeze(0)
        if (k == "betas") or (k == "face_offset") or (k == "joint_offset"):
            smplx_data_tmp[k] = v[0].unsqueeze(0)
    smplx_data = smplx_data_tmp

    gs_render = GS3DRenderer(
        human_model_path=human_model_path,
        subdivide_num=2,
        smpl_type="smplx",
        feat_dim=64,
        query_dim=64,
        use_rgb=False,
        sh_degree=3,
        mlp_network_config=None,
        xyz_offset_max_step=1.8 / 32,
    )

    gs_render.to(device)
    # print(cam_param_list[0])

    c2w_list = []
    intr_list = []
    for cam_param in cam_param_list:
        c2w = torch.eye(4).to(device)
        c2w[:3, :3] = cam_param["R"]
        c2w[:3, 3] = cam_param["t"]
        c2w_list.append(c2w)
        intr = torch.eye(4).to(device)
        intr[0, 0] = cam_param["focal"][0]
        intr[1, 1] = cam_param["focal"][1]
        intr[0, 2] = cam_param["princpt"][0]
        intr[1, 2] = cam_param["princpt"][1]
        intr_list.append(intr)

    c2w = torch.stack(c2w_list).unsqueeze(0)
    intrinsic = torch.stack(intr_list).unsqueeze(0)

    out = gs_render.forward(
        gs_hidden_features=torch.zeros((batch_size, 2048, 64)).float().to(device),
        query_points=None,
        smplx_data=smplx_data,
        c2w=c2w,
        intrinsic=intrinsic,
        height=int(cam_param_list[0]["princpt"][1]) * 2,
        width=int(cam_param_list[0]["princpt"][0]) * 2,
        background_color=torch.tensor([1.0, 1.0, 1.0])
        .float()
        .view(1, 1, 3)
        .repeat(batch_size, 2, 1)
        .to(device),
        debug=False,
    )

    for k, v in out.items():
        if k == "comp_rgb_bg":
            print("comp_rgb_bg", v)
            continue
        for b_idx in range(len(v)):
            if k == "3dgs":
                for v_idx in range(len(v[b_idx])):
                    v[b_idx][v_idx].save_ply(f"./debug_vis/{b_idx}_{v_idx}.ply")
                continue
            for v_idx in range(v.shape[1]):
                save_path = os.path.join("./debug_vis", f"{b_idx}_{v_idx}_{k}.jpg")
                cv2.imwrite(
                    save_path,
                    (v[b_idx, v_idx].detach().cpu().numpy() * 255).astype(np.uint8),
                )


def test1():
    import cv2

    human_model_path = "./pretrained_models/human_model_files"
    device = "cuda"

    # root_dir = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data"
    # meta_path = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/data_list.json"
    # dataset = ExAvatarDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=3,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(224, 224), source_image_res=384)

    # root_dir = "/data1/datasets1/3d_human_data/humman/humman_compressed"
    # meta_path = "/data1/datasets1/3d_human_data/humman/humman_id_debug_list.json"
    # dataset = HuMManDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=3,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(682, 384), source_image_res=384)

    # from openlrm.datasets.static_human import StaticHumanDataset
    # root_dir = "./train_data/static_human_data"
    # meta_path = "./train_data/static_human_data/data_id_list.json"
    # dataset = StaticHumanDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=7,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(682, 384), source_image_res=384,
    #                 debug=False)

    # from openlrm.datasets.singleview_human import SingleViewHumanDataset
    # root_dir = "./train_data/single_view"
    # meta_path = "./train_data/single_view/data_list.json"
    # dataset = SingleViewHumanDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=0,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(682, 384), source_image_res=384,
    #                 debug=False)

    from accelerate.utils import set_seed

    set_seed(1234)
    from LHM.datasets.video_human import VideoHumanDataset

    root_dir = "./train_data/ClothVideo"
    meta_path = "./train_data/ClothVideo/label/valid_id_with_img_list.json"
    dataset = VideoHumanDataset(
        root_dirs=root_dir,
        meta_path=meta_path,
        sample_side_views=7,
        render_image_res_low=384,
        render_image_res_high=384,
        render_region_size=(682, 384),
        source_image_res=384,
        enlarge_ratio=[0.85, 1.2],
        debug=False,
    )

    data = dataset[0]

    def get_smplx_params(data):
        smplx_params = {}
        smplx_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "expr",
            "trans",
            "betas",
        ]
        for k, v in data.items():
            if k in smplx_keys:
                # print(k, v.shape)
                smplx_params[k] = data[k]
        return smplx_params

    smplx_data = get_smplx_params(data)

    smplx_data_tmp = {}
    for k, v in smplx_data.items():
        smplx_data_tmp[k] = v.unsqueeze(0).to(device)
        print(k, v.shape)
    smplx_data = smplx_data_tmp

    c2ws = data["c2ws"].unsqueeze(0).to(device)
    intrs = data["intrs"].unsqueeze(0).to(device)
    render_images = data["render_image"].numpy()
    render_h = data["render_full_resolutions"][0, 0]
    render_w = data["render_full_resolutions"][0, 1]
    render_bg_colors = data["render_bg_colors"].unsqueeze(0).to(device)
    print("c2ws", c2ws.shape, "intrs", intrs.shape, intrs)

    gs_render = GS3DRenderer(
        human_model_path=human_model_path,
        subdivide_num=2,
        smpl_type="smplx",
        feat_dim=64,
        query_dim=64,
        use_rgb=False,
        sh_degree=3,
        mlp_network_config=None,
        xyz_offset_max_step=1.8 / 32,
        expr_param_dim=10,
        shape_param_dim=10,
        fix_opacity=True,
        fix_rotation=True,
    )
    gs_render.to(device)

    out = gs_render.forward(
        gs_hidden_features=torch.zeros((1, 2048, 64)).float().to(device),
        query_points=None,
        smplx_data=smplx_data,
        c2w=c2ws,
        intrinsic=intrs,
        height=render_h,
        width=render_w,
        background_color=render_bg_colors,
        debug=False,
    )
    os.makedirs("./debug_vis/gs_render", exist_ok=True)
    for k, v in out.items():
        if k == "comp_rgb_bg":
            print("comp_rgb_bg", v)
            continue
        for b_idx in range(len(v)):
            if k == "3dgs":
                for v_idx in range(len(v[b_idx])):
                    v[b_idx][v_idx].save_ply(
                        f"./debug_vis/gs_render/{b_idx}_{v_idx}.ply"
                    )
                continue
            for v_idx in range(v.shape[1]):
                save_path = os.path.join(
                    "./debug_vis/gs_render", f"{b_idx}_{v_idx}_{k}.jpg"
                )
                img = (
                    v[b_idx, v_idx].permute(1, 2, 0).detach().cpu().numpy() * 255
                ).astype(np.uint8)
                print(img.shape, save_path)
                if "mask" in k:
                    render_img = render_images[v_idx].transpose(1, 2, 0) * 255
                    cv2.imwrite(
                        save_path,
                        np.hstack(
                            [np.tile(img, (1, 1, 3)), render_img.astype(np.uint8)]
                        ),
                    )
                else:
                    cv2.imwrite(save_path, img)


if __name__ == "__main__":
    # test1()
    test()
    test()
    test()
