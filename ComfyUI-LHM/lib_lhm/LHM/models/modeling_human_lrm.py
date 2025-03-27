# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu  && Xiaodong Gu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:40:57
# @Function      : Main codes for LHM
import os
import pdb
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers.utils import is_torch_version

from LHM.models.arcface_utils import ResNetArcFace
from LHM.models.ESRGANer_utils import ESRGANEasyModel
from LHM.models.rendering.gs_renderer import GS3DRenderer, PointEmbed
from LHM.models.rendering.gsplat_renderer import GSPlatRenderer

# from openlrm.models.stylegan2_utils import EasyStyleGAN_series_model
from LHM.models.utils import linear

from .embedder import CameraEmbedder
from .rendering.synthesizer import TriplaneSynthesizer
from .transformer import TransformerDecoder

logger = get_logger(__name__)


class ModelHumanLRM(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_type="cond",
        tf_grad_ckpt=False,
        encoder_grad_ckpt=False,
        encoder_freeze: bool = True,
        encoder_type: str = "dino",
        encoder_model_name: str = "facebook/dino-vitb16",
        encoder_feat_dim: int = 768,
        num_pcl: int = 2048,
        pcl_dim: int = 512,
        human_model_path=None,
        smplx_subdivide_num=2,
        smplx_type="smplx",
        gs_query_dim=None,
        gs_use_rgb=False,
        gs_sh=3,
        gs_mlp_network_config=None,
        gs_xyz_offset_max_step=1.8 / 32,
        gs_clip_scaling=0.2,
        shape_param_dim=100,
        expr_param_dim=50,
        fix_opacity=False,
        fix_rotation=False,
        use_face_id=False,
        facesr=False,
        use_stylegan2_prior=False,
        **kwargs,
    ):
        super().__init__()

        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt

        # attributes
        self.encoder_feat_dim = encoder_feat_dim

        # modules
        # image encoder  default dino-v2
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            encoder_feat_dim=encoder_feat_dim,
        )

        # learnable points embedding
        skip_decoder = False
        self.latent_query_points_type = kwargs.get(
            "latent_query_points_type", "embedding"
        )
        if self.latent_query_points_type == "embedding":
            self.num_pcl = num_pcl  # 2048
            self.pcl_embeddings = nn.Embedding(num_pcl, pcl_dim)  # 1024
        elif self.latent_query_points_type.startswith("smplx"):
            latent_query_points_file = os.path.join(
                human_model_path, "smplx_points", f"{self.latent_query_points_type}.npy"
            )
            pcl_embeddings = torch.from_numpy(np.load(latent_query_points_file)).float()
            print(
                f"==========load smplx points:{latent_query_points_file}, shape:{pcl_embeddings.shape}"
            )
            self.register_buffer("pcl_embeddings", pcl_embeddings)
            self.pcl_embed = PointEmbed(dim=pcl_dim)
        elif self.latent_query_points_type.startswith("e2e_smplx"):
            skip_decoder = True
            self.pcl_embed = PointEmbed(dim=pcl_dim)  # pcl dim 1024
        else:
            raise NotImplementedError
        print(f"==========skip_decoder:{skip_decoder}")

        # transformer
        self.transformer = self.build_transformer(
            transformer_type,
            transformer_layers,
            transformer_heads,
            transformer_dim,
            encoder_feat_dim,
            **kwargs,
        )

        # renderer
        cano_pose_type = kwargs.get("cano_pose_type", 0)
        dense_sample_pts = kwargs.get("dense_sample_pts", 40000)

        # original 3DGS Raster
        self.renderer = GS3DRenderer(
            human_model_path=human_model_path,
            subdivide_num=smplx_subdivide_num,
            smpl_type=smplx_type,
            feat_dim=transformer_dim,
            query_dim=gs_query_dim,
            use_rgb=gs_use_rgb,
            sh_degree=gs_sh,
            mlp_network_config=gs_mlp_network_config,
            xyz_offset_max_step=gs_xyz_offset_max_step,
            clip_scaling=gs_clip_scaling,
            shape_param_dim=shape_param_dim,
            expr_param_dim=expr_param_dim,
            cano_pose_type=cano_pose_type,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            decoder_mlp=kwargs.get("decoder_mlp", False),
            skip_decoder=skip_decoder,
            decode_with_extra_info=kwargs.get("decode_with_extra_info", None),
            gradient_checkpointing=self.gradient_checkpointing,
            apply_pose_blendshape=kwargs.get("apply_pose_blendshape", False),
            dense_sample_pts=dense_sample_pts,
        )

        # face_id
        self.use_face_id = use_face_id
        self.facesr = facesr
        self.use_stylegan2_prior = use_stylegan2_prior

        if self.use_face_id:
            self.id_face_net = ResNetArcFace()

        if self.facesr:
            self.faceESRGAN = ESRGANEasyModel()
        if self.use_stylegan2_prior:
            self.stylegan2_prior = EasyStyleGAN_series_model()  # harm PSNR.

    def compute_discriminator_loss(self, data):
        return -F.softplus(self.stylegan2_prior(data)).mean()  # StyleGAN2

    def train(self, mode=True):
        super().train(mode)
        if self.use_face_id:
            # setting id_face_net to evaluation
            self.id_face_net.eval()

    def build_transformer(
        self,
        transformer_type,
        transformer_layers,
        transformer_heads,
        transformer_dim,
        encoder_feat_dim,
        **kwargs,
    ):
        return TransformerDecoder(
            block_type=transformer_type,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=encoder_feat_dim,
            mod_dim=None,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def get_last_layer(self):
        return self.renderer.gs_net.out_layers["shs"].weight

    def hyper_step(self, step):
        pass

    @staticmethod
    def _encoder_fn(encoder_type: str):
        encoder_type = encoder_type.lower()
        assert encoder_type in [
            "dino",
            "dinov2",
            "dinov2_unet",
            "resunet",
            "dinov2_featup",
            "dinov2_dpt",
            "dinov2_fusion",
            "sapiens",
        ], "Unsupported encoder type"
        if encoder_type == "dino":
            from .encoders.dino_wrapper import DinoWrapper

            logger.info("Using DINO as the encoder")
            return DinoWrapper
        elif encoder_type == "dinov2":
            from .encoders.dinov2_wrapper import Dinov2Wrapper

            logger.info("Using DINOv2 as the encoder")
            return Dinov2Wrapper
        elif encoder_type == "dinov2_unet":
            from .encoders.dinov2_unet_wrapper import Dinov2UnetWrapper

            logger.info("Using Dinov2Unet as the encoder")
            return Dinov2UnetWrapper
        elif encoder_type == "resunet":
            from .encoders.xunet_wrapper import XnetWrapper

            logger.info("Using XnetWrapper as the encoder")
            return XnetWrapper
        elif encoder_type == "dinov2_featup":
            from .encoders.dinov2_featup_wrapper import Dinov2FeatUpWrapper

            logger.info("Using Dinov2FeatUpWrapper as the encoder")
            return Dinov2FeatUpWrapper
        elif encoder_type == "dinov2_dpt":
            from .encoders.dinov2_dpt_wrapper import Dinov2DPTWrapper

            logger.info("Using Dinov2DPTWrapper as the encoder")
            return Dinov2DPTWrapper
        elif encoder_type == "dinov2_fusion":
            from .encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper

            logger.info("Using Dinov2FusionWrapper as the encoder")
            return Dinov2FusionWrapper
        elif encoder_type == "sapiens":
            from .encoders.sapiens_warpper import SapiensWrapper

            logger.info("Using Sapiens as the encoder")
            return SapiensWrapper

    def forward_transformer(self, image_feats, camera_embeddings, query_points):
        """
        Applies forward transformation to the input features.
        Args:
            image_feats (torch.Tensor): Input image features. Shape [B, C, H, W].
            camera_embeddings (torch.Tensor): Camera embeddings. Shape [B, D].
            query_points (torch.Tensor): Query points. Shape [B, L, D].
        Returns:
            torch.Tensor: Transformed features. Shape [B, L, D].
        """

        B = image_feats.shape[0]

        if self.latent_query_points_type == "embedding":
            range_ = torch.arange(self.num_pcl, device=image_feats.device)
            x = self.pcl_embeddings(range_).unsqueeze(0).repeat((B, 1, 1))  # [B, L, D]

        elif self.latent_query_points_type.startswith("smplx"):
            x = self.pcl_embed(self.pcl_embeddings.unsqueeze(0)).repeat(
                (B, 1, 1)
            )  # [B, L, D]

        elif self.latent_query_points_type.startswith("e2e_smplx"):
            # Linear warp -> MLP + LayerNorm
            x = self.pcl_embed(query_points)  # [B, L, D]

        x = self.transformer(
            x,
            cond=image_feats,
            mod=camera_embeddings,
        )  # [B, L, D]
        return x

    def forward_encode_image(self, image):
        # encode image

        if self.training and self.encoder_gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            image_feats = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.encoder),
                image,
                **ckpt_kwargs,
            )
        else:
            image_feats = self.encoder(image)
        return image_feats

    @torch.compile
    def forward_latent_points(self, image, camera, query_points=None):
        """
        Forward pass of the latent points generation.
        Args:
            image (torch.Tensor): Input image tensor of shape [B, C_img, H_img, W_img].
            camera (torch.Tensor): Camera tensor of shape [B, D_cam_raw].
            query_points (torch.Tensor, optional): Query points tensor. for example, smplx surface points, Defaults to None.
        Returns:
            torch.Tensor: Generated tokens tensor.
            torch.Tensor: Encoded image features tensor.
        """

        B = image.shape[0]

        # encode image
        # image_feats is cond texture
        image_feats = self.forward_encode_image(image)

        assert (
            image_feats.shape[-1] == self.encoder_feat_dim
        ), f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        # # embed camera
        # camera_embeddings = self.camera_embedder(camera)
        # assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
        #     f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"

        # transformer generating latent points
        tokens = self.forward_transformer(
            image_feats, camera_embeddings=None, query_points=query_points
        )

        return tokens, image_feats

    def forward(
        self,
        image,
        source_c2ws,
        source_intrs,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        smplx_params,
        **kwargs,
    ):

        # image: [B, N_ref, C_img, H_img, W_img]
        # source_c2ws: [B, N_ref, 4, 4]
        # source_intrs: [B, N_ref, 4, 4]
        # render_c2ws: [B, N_source, 4, 4]
        # render_intrs: [B, N_source, 4, 4]
        # render_bg_colors: [B, N_source, 3]
        # smplx_params: Dict, e.g., pose_shape: [B, N_source, 21, 3], betas:[B, 100]
        # kwargs: Dict, e.g., src_head_imgs

        assert (
            image.shape[0] == render_c2ws.shape[0]
        ), "Batch size mismatch for image and render_c2ws"
        assert (
            image.shape[0] == render_bg_colors.shape[0]
        ), "Batch size mismatch for image and render_bg_colors"
        assert (
            image.shape[0] == smplx_params["betas"].shape[0]
        ), "Batch size mismatch for image and smplx_params"
        assert (
            image.shape[0] == smplx_params["body_pose"].shape[0]
        ), "Batch size mismatch for image and smplx_params"
        assert len(smplx_params["betas"].shape) == 2

        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
            render_intrs[0, 0, 0, 2] * 2
        )
        query_points = None
        if self.latent_query_points_type.startswith("e2e_smplx"):
            query_points, smplx_params = self.renderer.get_query_points(
                smplx_params, device=image.device
            )

        latent_points, image_feats = self.forward_latent_points(
            image[:, 0], camera=None, query_points=query_points
        )  # [B, N, C]

        # render target views

        render_results = self.renderer(
            gs_hidden_features=latent_points,
            query_points=query_points,
            smplx_data=smplx_params,
            c2w=render_c2ws,
            intrinsic=render_intrs,
            height=render_h,
            width=render_w,
            background_color=render_bg_colors,
            additional_features={"image_feats": image_feats, "image": image[:, 0]},
            df_data=kwargs["df_data"],
        )

        N, M = render_c2ws.shape[:2]
        assert (
            render_results["comp_rgb"].shape[0] == N
        ), "Batch size mismatch for render_results"
        assert (
            render_results["comp_rgb"].shape[1] == M
        ), "Number of rendered views should be consistent with render_cameras"

        gs_attrs_list = render_results.pop("gs_attr")

        offset_list = []
        scaling_list = []
        for gs_attrs in gs_attrs_list:
            offset_list.append(gs_attrs.offset_xyz)
            scaling_list.append(gs_attrs.scaling)
        offset_output = torch.stack(offset_list)
        scaling_output = torch.stack(scaling_list)

        return {
            "latent_points": latent_points,
            "offset_output": offset_output,
            "scaling_output": scaling_output,
            **render_results,
        }

    def hyper_step(self, step):

        self.renderer.hyper_step(step)

    @torch.no_grad()
    def infer_single_view(
        self,
        image,
        source_c2ws,
        source_intrs,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        smplx_params,
    ):
        # image: [B, N_ref, C_img, H_img, W_img]
        # source_c2ws: [B, N_ref, 4, 4]
        # source_intrs: [B, N_ref, 4, 4]
        # render_c2ws: [B, N_source, 4, 4]
        # render_intrs: [B, N_source, 4, 4]
        # render_bg_colors: [B, N_source, 3]
        # smplx_params: Dict, e.g., pose_shape: [B, N_source, 21, 3], betas:[B, 100]
        assert (
            image.shape[0] == render_c2ws.shape[0]
        ), "Batch size mismatch for image and render_c2ws"
        assert (
            image.shape[0] == render_bg_colors.shape[0]
        ), "Batch size mismatch for image and render_bg_colors"
        assert (
            image.shape[0] == smplx_params["betas"].shape[0]
        ), "Batch size mismatch for image and smplx_params"
        assert (
            image.shape[0] == smplx_params["body_pose"].shape[0]
        ), "Batch size mismatch for image and smplx_params"
        assert len(smplx_params["betas"].shape) == 2
        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
            render_intrs[0, 0, 0, 2] * 2
        )
        assert image.shape[0] == 1
        num_views = render_c2ws.shape[1]
        query_points = None

        if self.latent_query_points_type.startswith("e2e_smplx"):
            # obtain subdivide smplx points and transform_matrix from predefined pose to zero-pose (null pose)
            query_points, smplx_params = self.renderer.get_query_points(
                smplx_params, device=image.device
            )

        # using DiT to predict query points features.
        latent_points, image_feats = self.forward_latent_points(
            image[:, 0], camera=None, query_points=query_points
        )  # [B, N, C]

        gs_model_list, query_points, smplx_params = self.renderer.forward_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            smplx_data=smplx_params,
            additional_features={"image_feats": image_feats, "image": image[:, 0]},
        )

        # render target views
        render_res_list = []
        for view_idx in range(num_views):
            render_res = self.renderer.forward_animate_gs(
                gs_model_list,
                query_points,
                self.renderer.get_single_view_smpl_data(smplx_params, view_idx),
                render_c2ws[:, view_idx : view_idx + 1],
                render_intrs[:, view_idx : view_idx + 1],
                render_h,
                render_w,
                render_bg_colors[:, view_idx : view_idx + 1],
            )
            render_res_list.append(render_res)

        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                out[k].append(v)
        for k, v in out.items():
            # print(f"out key:{k}")
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.concat(v, dim=1)
                if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                    out[k] = out[k][0].permute(
                        0, 2, 3, 1
                    )  # [1, Nv, 3, H, W] -> [Nv, 3, H, W] - > [Nv, H, W, 3]
            else:
                out[k] = v
        return out


class ModelHumanLRMSapdinoBodyHeadSD3_5(ModelHumanLRM):
    """Using SD3BodyHeadMMJointTransformerBlock"""

    def __init__(self, **kwargs):
        super(ModelHumanLRMSapdinoBodyHeadSD3_5, self).__init__(**kwargs)

        # fine encoder
        fine_encoder_type = kwargs["fine_encoder_type"]
        fine_encoder_model_name = kwargs["fine_encoder_model_name"]
        fine_encoder_feat_dim = kwargs["fine_encoder_feat_dim"]
        fine_encoder_freeze = kwargs["fine_encoder_freeze"]

        self.fine_encoder_feat_dim = fine_encoder_feat_dim

        self.fine_encoder = self._encoder_fn(fine_encoder_type)(
            model_name=fine_encoder_model_name,
            freeze=fine_encoder_freeze,
            encoder_feat_dim=fine_encoder_feat_dim,
        )

        pcl_dim = kwargs.get("pcl_dim", 1024)

        input_dim = kwargs.get("fine_encoder_feat_dim", pcl_dim)
        mid_dim = input_dim // 2
        self.motion_embed_mlp = nn.Sequential(
            linear(input_dim, mid_dim),
            nn.SiLU(),
            linear(mid_dim, pcl_dim * 2),
        )

    def build_transformer(
        self,
        transformer_type,
        transformer_layers,
        transformer_heads,
        transformer_dim,
        encoder_feat_dim,
        **kwargs,
    ):

        return TransformerDecoder(
            block_type=transformer_type,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=kwargs.get("fine_encoder_feat_dim", 1024),
            mod_dim=None,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def obtain_facesr(self, head_image):
        def tensor_to_image(head_image):
            head_image = head_image.permute(0, 2, 3, 1)
            head_image_numpy = head_image.detach().cpu().numpy()
            head_image_numpy = (head_image_numpy * 255).astype(np.uint8)

            head_image_numpy = head_image_numpy[..., ::-1]  # RGB2BGR

            return head_image_numpy

        def image_to_tensor(head_image_numpy):
            head_image_numpy = head_image_numpy[..., ::-1]  # BGR2RGB
            head_image_tensor = (
                torch.from_numpy(head_image_numpy.copy()).permute(0, 3, 1, 2).float()
            )
            head_image_tensor = head_image_tensor / 255.0
            return head_image_tensor

        device = head_image.device
        B, V, C, H, W = head_image.shape
        head_image = head_image.view(-1, C, H, W)
        head_image_numpy = tensor_to_image(head_image)

        sr_head_image_list = []

        for _i, head_image in enumerate(head_image_numpy):
            sr_head_image = self.faceESRGAN(head_image)
            sr_head_image_list.append(sr_head_image)

        sr_head_image_numpy = np.stack(sr_head_image_list, axis=0)

        head_image_tensor = image_to_tensor(sr_head_image_numpy)
        _, _, new_H, new_W = head_image_tensor.shape

        head_image = head_image_tensor.view(B, V, C, new_H, new_W).to(device)

        return head_image

    def obtain_params(self, cfg):
        # add all bias and LayerNorm params to no_decay_params
        no_decay_params, decay_params = [], []

        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, "bias") and module.bias is not None:
                no_decay_params.append(module.bias)

        # add remaining parameters to decay_params
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in self.parameters() if id(p) not in _no_decay_ids]

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # Optimizer
        opt_groups = [
            {
                "params": decay_params,
                "weight_decay": cfg.train.optim.weight_decay,
                "lr": cfg.train.optim.lr,
                "name": "decay",
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "lr": cfg.train.optim.lr,
                "name": "no_decay",
            },
        ]

        logger.info("======== Weight Decay Parameters ========")
        logger.info(f"Total: {len(decay_params)}")
        logger.info("======== No Weight Decay Parameters ========")
        logger.info(f"Total: {len(no_decay_params)}")

        print(f"Total Params: {len(no_decay_params) + len(decay_params)}")

        return opt_groups


    def forward_moitonembed(self, motion_tokens):

        motion_tokens = motion_tokens.mean(dim=1, keepdim=True)

        motion_tokens = self.motion_embed_mlp(motion_tokens).squeeze(
            1
        )  # [B, 2*D]  # one for head, one for body

        return motion_tokens

    @torch.compile
    def forward_latent_points(self, image, head_image, camera, query_points=None):
        """
        Forward pass of the latent points generation.
        Args:
            image (torch.Tensor): Input image tensor of shape [B, C_img, H_img, W_img].
            head_image (torch.Tensor): Input head image tensor of shape [B, C_img, H_img, W_img].
            camera (torch.Tensor): Camera tensor of shape [B, D_cam_raw].
            query_points (torch.Tensor, optional): Query points tensor. for example, smplx surface points, Defaults to None.
        Returns:
            torch.Tensor: Generated tokens tensor.
            torch.Tensor: Encoded image features tensor.
        """

        B = image.shape[0]

        # encode image
        # image_feats is cond texture
        image_feats, head_feats, body_feats = self.forward_encode_image(
            image, head_image
        )

        motion_tokens = self.forward_moitonembed(body_feats)

        assert (
            image_feats.shape[-1] == self.fine_encoder_feat_dim
        ), f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.fine_encoder_feat_dim}"

        # # embed camera
        # camera_embeddings = self.camera_embedder(camera)
        # assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
        #     f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"

        # transformer generating latent points

        tokens = self.forward_transformer(
            image_feats,
            camera_embeddings=None,
            query_points=query_points,
            motion_embed=motion_tokens,
        )

        return tokens, image_feats

    def forward_fine_encode_image(self, image):
        if self.training and self.encoder_gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            image_feats = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.fine_encoder),
                image,
                **ckpt_kwargs,
            )
        else:
            image_feats = self.fine_encoder(image)
        return image_feats

    def forward_encode_image(self, image, head_image):
        # encode image

        body_embed = self.forward_fine_encode_image(image)  # 4096 tokens
        head_embed = super(ModelHumanLRMSapdinoBodyHeadSD3_5, self).forward_encode_image(
            head_image
        )  # 1024 tokens

        head_embed = F.pad(
            head_embed, (0, body_embed.shape[-1] - head_embed.shape[-1], 0, 0, 0, 0)
        )  # the same as sd3, learnable

        merge_tokens = torch.cat([body_embed, head_embed], dim=1)

        return merge_tokens, head_embed, body_embed

    @torch.no_grad()
    def infer_single_view(
        self,
        image,
        head_image,
        source_c2ws,
        source_intrs,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        smplx_params,
    ):
        assert len(smplx_params["betas"].shape) == 2

        if self.facesr:
            head_image = self.obtain_facesr(head_image)

        assert image.shape[0] == 1

        query_points = None
        if self.latent_query_points_type.startswith("e2e_smplx"):
            query_points, smplx_params = self.renderer.get_query_points(
                smplx_params, device=image.device
            )

        latent_points, image_feats = self.forward_latent_points(
            image[:, 0], head_image[:, 0], camera=None, query_points=query_points
        )  # [B, N, C]

        self.renderer.hyper_step(10000000)  # set to max step

        gs_model_list, query_points, smplx_params = self.renderer.forward_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            smplx_data=smplx_params,
            additional_features={"image_feats": image_feats, "image": image[:, 0]},
        )


        return gs_model_list, query_points, smplx_params['transform_mat_neutral_pose']
    

    def animation_infer(self, gs_model_list, query_points, smplx_params, render_c2ws, render_intrs, render_bg_colors):
        '''Inference code avoid repeat forward.
        '''

        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
            render_intrs[0, 0, 0, 2] * 2
        )
        # render target views
        render_res_list = []
        num_views = render_c2ws.shape[1]

        for view_idx in range(num_views):
            render_res = self.renderer.forward_animate_gs(
                gs_model_list,
                query_points,
                self.renderer.get_single_view_smpl_data(smplx_params, view_idx),
                render_c2ws[:, view_idx : view_idx + 1],
                render_intrs[:, view_idx : view_idx + 1],
                render_h,
                render_w,
                render_bg_colors[:, view_idx : view_idx + 1],
            )
            render_res_list.append(render_res)

        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                if isinstance(v[0], torch.Tensor):
                    out[k].append(v.detach())
                else:
                    out[k].append(v)
        for k, v in out.items():
            # print(f"out key:{k}")
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.concat(v, dim=1)
                if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                    out[k] = out[k][0].permute(
                        0, 2, 3, 1
                    )  # [1, Nv, 3, H, W] -> [Nv, 3, H, W] - > [Nv, H, W, 3]
            else:
                out[k] = v
        return out

    def animation_infer_gs(self, gs_attr_list, query_points, smplx_params):
        '''Inference code to query gs mesh.
        '''
        batch_size = len(gs_attr_list)
        for b in range(batch_size):
            gs_attr = gs_attr_list[b]
            query_pt = query_points[b]


            merge_animatable_gs_model_list, cano_gs_model_list = self.renderer.animate_gs_model(
                gs_attr,
                query_pt,
                self.renderer.get_single_batch_smpl_data(smplx_params, b),
                debug=False,
            )
        
        return merge_animatable_gs_model_list[0]

    def forward_transformer(
        self, image_feats, camera_embeddings, query_points, motion_embed=None
    ):
        """
        Applies forward transformation to the input features.
        Args:
            image_feats (torch.Tensor): Input image features. Shape [B, C, H, W].
            camera_embeddings (torch.Tensor): Camera embeddings. Shape [B, D].
            query_points (torch.Tensor): Query points. Shape [B, L, D].
            motion embed(torch.Tensor): Query points. Shape [B, L, D].
        Returns:
            torch.Tensor: Transformed features. Shape [B, L, D].
        """

        B = image_feats.shape[0]

        if self.latent_query_points_type == "embedding":
            range_ = torch.arange(self.num_pcl, device=image_feats.device)
            x = self.pcl_embeddings(range_).unsqueeze(0).repeat((B, 1, 1))  # [B, L, D]

        elif self.latent_query_points_type.startswith("smplx"):
            x = self.pcl_embed(self.pcl_embeddings.unsqueeze(0)).repeat(
                (B, 1, 1)
            )  # [B, L, D]

        elif self.latent_query_points_type.startswith("e2e_smplx"):
            # Linear warp -> MLP + LayerNorm

            x = self.pcl_embed(query_points)  # [B, L, D]

        x = self.transformer(
            x,
            cond=image_feats,
            mod=camera_embeddings,
            temb=motion_embed,
        )  # [B, L, D]
        return x

    def forward(
        image,
        source_c2ws,
        source_intrs,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        smplx_params,
        **kwargs,
    ):

        # image: [B, N_ref, C_img, H_img, W_img]
        # source_c2ws: [B, N_ref, 4, 4]
        # source_intrs: [B, N_ref, 4, 4]
        # render_c2ws: [B, N_source, 4, 4]
        # render_intrs: [B, N_source, 4, 4]
        # render_bg_colors: [B, N_source, 3]
        # smplx_params: Dict, e.g., pose_shape: [B, N_source, 21, 3], betas:[B, 100]
        # kwargs: Dict, e.g., src_head_imgs
        head_image = kwargs["source_head_rgbs"]
        if self.facesr:
            head_image = self.obtain_facesr(head_image)

        assert (
            image.shape[0] == render_c2ws.shape[0]
        ), "Batch size mismatch for image and render_c2ws"
        assert (
            head_image.shape[0] == render_c2ws.shape[0]
        ), "Batch size mismatch for image and render_c2ws"
        assert (
            image.shape[0] == render_bg_colors.shape[0]
        ), "Batch size mismatch for image and render_bg_colors"
        assert (
            image.shape[0] == smplx_params["betas"].shape[0]
        ), "Batch size mismatch for image and smplx_params"
        assert (
            image.shape[0] == smplx_params["body_pose"].shape[0]
        ), "Batch size mismatch for image and smplx_params"
        assert len(smplx_params["betas"].shape) == 2

        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
            render_intrs[0, 0, 0, 2] * 2
        )

        query_points = None
        if self.latent_query_points_type.startswith("e2e_smplx"):
            query_points, smplx_params = self.renderer.get_query_points(
                smplx_params, device=image.device
            )

        latent_points, image_feats = self.forward_latent_points(
            image[:, 0], head_image[:, 0], camera=None, query_points=query_points
        )  # [B, N, C]

        # render target views
        render_results = self.renderer(
            gs_hidden_features=latent_points,
            query_points=query_points,
            smplx_data=smplx_params,
            c2w=render_c2ws,
            intrinsic=render_intrs,
            height=render_h,
            width=render_w,
            background_color=render_bg_colors,
            additional_features={"image_feats": image_feats, "image": image[:, 0]},
            df_data=kwargs["df_data"],
        )

        N, M = render_c2ws.shape[:2]
        assert (
            render_results["comp_rgb"].shape[0] == N
        ), "Batch size mismatch for render_results"
        assert (
            render_results["comp_rgb"].shape[1] == M
        ), "Number of rendered views should be consistent with render_cameras"

        gs_attrs_list = render_results.pop("gs_attr")

        offset_list = []
        scaling_list = []
        for gs_attrs in gs_attrs_list:
            offset_list.append(gs_attrs.offset_xyz)
            scaling_list.append(gs_attrs.scaling)
        offset_output = torch.stack(offset_list)
        scaling_output = torch.stack(scaling_list)

        return {
            "latent_points": latent_points,
            "offset_output": offset_output,
            "scaling_output": scaling_output,
            **render_results,
        }
