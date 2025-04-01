# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os
import sys

current_dir_path = os.path.dirname(__file__)
sys.path.append(current_dir_path + "/../pose_estimation")
import copy

import einops
import numpy as np
import roma
import torch
import torch.nn as nn

from blocks import (
    Dinov2Backbone,
    FourierPositionEncoding,
    SMPL_Layer,
    TransformerDecoder,
)

from pose_utils import (
    inverse_perspective_projection,
    pad_to_max,
    rebatch,
    rot6d_to_rotmat,
    undo_focal_length_normalization,
    undo_log_depth,
    unpatch,
)




from torch import nn


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def load_model(ckpt_path, model_path, device=torch.device("cuda")):
    """Open a checkpoint, build Multi-HMR using saved arguments, load the model weigths."""
    # Model

    assert os.path.isfile(ckpt_path), f"{ckpt_path} not found"

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k, v in vars(ckpt["args"]).items():
        kwargs[k] = v
    print(ckpt["args"].img_size)
    # Build the model.
    if isinstance(ckpt["args"].img_size, list):
        kwargs["img_size"] = ckpt["args"].img_size[0]
    else:
        kwargs["img_size"] = ckpt["args"].img_size
    kwargs["smplx_dir"] = model_path
    print("Loading model...")
    model = Model(**kwargs).to(device)
    print("Model loaded")
    # Load weights into model.
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.output_mesh = True
    model.eval()
    return model


def forward_model(
    model,
    input_image,
    camera_parameters,
    det_thresh=0.3,
    nms_kernel_size=1,
    pseudo_idx=None,
    max_dist=None,
):
    """Make a forward pass on an input image and camera parameters."""

    # Forward the model.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            humans = model(
                input_image,
                is_training=False,
                nms_kernel_size=int(nms_kernel_size),
                det_thresh=det_thresh,
                K=camera_parameters,
                idx=pseudo_idx,
                max_dist=max_dist,
            )

    return humans


class Model(nn.Module):
    """A ViT backbone followed by a "HPH" head (stack of cross attention layers with queries corresponding to detected humans.)"""

    def __init__(
        self,
        backbone="dinov2_vitb14",
        pretrained_backbone=False,
        img_size=896,
        camera_embedding="geometric",  # geometric encodes viewing directions with fourrier encoding
        camera_embedding_num_bands=16,  # increase the size of the camera embedding
        camera_embedding_max_resolution=64,  # does not increase the size of the camera embedding
        nearness=True,  # regress log(1/z)
        xat_depth=2,  # number of cross attention block (SA, CA, MLP) in the HPH head.
        xat_num_heads=8,  # Number of attention heads
        dict_smpl_layer=None,
        person_center="head",
        clip_dist=True,
        num_betas=10,
        smplx_dir=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        # Save options
        self.img_size = img_size
        self.nearness = nearness
        self.clip_dist = (clip_dist,)
        self.xat_depth = xat_depth
        self.xat_num_heads = xat_num_heads
        self.num_betas = num_betas
        self.output_mesh = True

        # Setup backbone
        self.backbone = Dinov2Backbone(backbone, pretrained=pretrained_backbone)
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        assert self.img_size % self.patch_size == 0, "Invalid img size"

        # Camera instrinsics
        self.fovn = 60
        self.camera_embedding = camera_embedding
        self.camera_embed_dim = 0
        if self.camera_embedding is not None:
            if not self.camera_embedding == "geometric":
                raise NotImplementedError(
                    "Only geometric camera embedding is implemented"
                )
            self.camera = FourierPositionEncoding(
                n=3,
                num_bands=camera_embedding_num_bands,
                max_resolution=camera_embedding_max_resolution,
            )
            # import pdb
            # pdb.set_trace()
            self.camera_embed_dim = self.camera.channels

        # Heads - Detection
        self.mlp_classif = regression_mlp(
            [self.embed_dim, self.embed_dim, 1]
        )  # bg or human

        # Heads - Human properties
        self.mlp_offset = regression_mlp([self.embed_dim, self.embed_dim, 2])  # offset

        # SMPL Layers
        self.nrot = 53
        dict_smpl_layer = {
            "neutral": {
                10: SMPL_Layer(
                    smplx_dir,
                    type="smplx",
                    gender="neutral",
                    num_betas=10,
                    kid=False,
                    person_center=person_center,
                ),
                11: SMPL_Layer(
                    smplx_dir,
                    type="smplx",
                    gender="neutral",
                    num_betas=11,
                    kid=False,
                    person_center=person_center,
                ),
            }
        }
        _moduleDict = []
        for k, _smpl_layer in dict_smpl_layer.items():
            for x, y in _smpl_layer.items():
                _moduleDict.append([f"{k}_{x}", copy.deepcopy(y)])
        self.smpl_layer = nn.ModuleDict(_moduleDict)

        self.x_attention_head = HPH(
            num_body_joints=self.nrot - 1,  # 23,
            context_dim=self.embed_dim + self.camera_embed_dim,
            dim=1024,
            depth=self.xat_depth,
            heads=self.xat_num_heads,
            mlp_dim=1024,
            dim_head=32,
            dropout=0.0,
            emb_dropout=0.0,
            at_token_res=self.img_size // self.patch_size,
            num_betas=self.num_betas,
            smplx_dir=smplx_dir,
        )

        print(f"person center is {person_center}")

    # set whether do filter
    def set_filter(self, apply_filter):
        self.apply_filter = apply_filter

    def detection(
        self,
        z,
        nms_kernel_size,
        det_thresh,
        N,
        idx=None,
        max_dist=None,
        is_training=False,
    ):
        """Detection score on the entire low res image"""
        scores = _sigmoid(self.mlp_classif(z))  # per token detection score.
        # Restore Height and Width dimensions.
        scores = unpatch(
            scores, patch_size=1, c=scores.shape[2], img_size=int(np.sqrt(N))
        )
        pseudo_idx = idx
        if not is_training:
            if (
                nms_kernel_size > 1
            ):  # Easy nms: supress adjacent high scores with max pooling.
                scores = _nms(scores, kernel=nms_kernel_size)
            _scores = torch.permute(scores, (0, 2, 3, 1))

            # Binary decision (keep confident detections)
            idx = apply_threshold(det_thresh, _scores)
            if pseudo_idx is not None:
                max_dist = 4 if max_dist is None else max_dist
                mask = (torch.abs(idx[1] - pseudo_idx[1]) <= max_dist) & (
                    torch.abs(idx[2] - pseudo_idx[2]) <= max_dist
                )
                idx_num = torch.sum(mask)
                if idx_num < 1:
                    top = torch.clamp(
                        pseudo_idx[1] - max_dist, min=0, max=_scores.shape[1] - 1
                    )
                    bottom = torch.clamp(
                        pseudo_idx[1] + max_dist, min=0, max=_scores.shape[1]
                    )
                    left = torch.clamp(
                        pseudo_idx[2] - max_dist, min=0, max=_scores.shape[2] - 1
                    )
                    right = torch.clamp(
                        pseudo_idx[2] + max_dist, min=0, max=_scores.shape[2]
                    )

                    neigborhoods = _scores[:, top:bottom, left:right, :]

                    idx = torch.argmax(neigborhoods)
                    try:
                        idx = unravel_index(idx, neigborhoods.shape)
                    except Exception as e:
                        print(pseudo_idx)
                        raise e
                    idx = (
                        pseudo_idx[0],
                        idx[1] + pseudo_idx[1] - max_dist,
                        idx[2] + pseudo_idx[2] - max_dist,
                        pseudo_idx[3],
                    )

                elif idx_num > 1:  # TODO

                    idx = (idx[0][mask], idx[1][mask], idx[2][mask], idx[3][mask])
                else:
                    idx = (idx[0][mask], idx[1][mask], idx[2][mask], idx[3][mask])
        else:
            assert idx is not None  # training time
        # Scores
        scores_detected = scores[
            idx[0], idx[3], idx[1], idx[2]
        ]  # scores of the detected humans only

        scores = torch.permute(scores, (0, 2, 3, 1))
        return scores, scores_detected, idx

    def embedd_camera(self, K, z):
        """Embed viewing directions using fourrier encoding."""
        bs = z.shape[0]
        _h, _w = list(z.shape[-2:])
        points = (
            torch.stack(
                [
                    torch.arange(0, _h, 1).reshape(-1, 1).repeat(1, _w),
                    torch.arange(0, _w, 1).reshape(1, -1).repeat(_h, 1),
                ],
                -1,
            )
            .to(z.device)
            .float()
        )  # [h,w,2]
        points = (
            points * self.patch_size + self.patch_size // 2
        )  # move to pixel space - we give the pixel center of each token
        points = points.reshape(1, -1, 2).repeat(bs, 1, 1)  # (bs, N, 2): 2D points
        distance = torch.ones(bs, points.shape[1], 1).to(
            K.device
        )  # (bs, N, 1): distance in the 3D world
        rays = inverse_perspective_projection(points, K, distance)  # (bs, N, 3)
        rays_embeddings = self.camera(pos=rays)

        # Repeat for each element of the batch
        z_K = rays_embeddings.reshape(bs, _h, _w, self.camera_embed_dim)  # [bs,h,w,D]
        return z_K

    def to_euclidean_dist(self, x, dist, _K):
        # Focal length normalization
        focal = _K[:, [0], [0]]
        dist = undo_focal_length_normalization(
            dist, focal, fovn=self.fovn, img_size=x.shape[-1]
        )
        # log space
        if self.nearness:
            dist = undo_log_depth(dist)

        # Clamping
        if self.clip_dist:
            dist = torch.clamp(dist, 0, 50)

        return dist

    def get_smpl(self):
        return self.smpl_layer[f"neutral_{self.num_betas}"]

    def generate_meshes(self, out):
        """
        Generates meshes for each person detected in the image.

        This function processes the output of the detection model, which includes rotation vectors,
        shapes, locations, distances, expressions, and other information related to SMPL-X parameters.

        Parameters:
        out (dict): A dictionary containing detection results and SMPL-X related parameters.

        Returns:
        list: A list of dictionaries, each containing information about a detected person's mesh.
        """
        # Neutral
        persons = []
        rotvec, shape, loc, dist, expression, K_det = (
            out["rotvec"],
            out["shape"],
            out["loc"],
            out["dist"],
            out["expression"],
            out["K_det"],
        )
        scores_det = out["scores_det"]
        idx = out["idx"]
        smpl_out = self.smpl_layer[f"neutral_{self.num_betas}"](
            rotvec, shape, loc, dist, None, K=K_det, expression=expression
        )
        out.update(smpl_out)

        for i in range(idx[0].shape[0]):
            person = {
                # Detection
                "scores": scores_det[i],  # detection scores
                "loc": out["loc"][i],  # 2d pixel location of the primary keypoints
                # SMPL-X params
                "transl": out["transl"][i],  # from the primary keypoint i.e. the head
                "transl_pelvis": out["transl_pelvis"][i],  # of the pelvis joint
                "rotvec": out["rotvec"][i],
                "expression": out["expression"][i],
                "shape": out["shape"][i],
                # SMPL-X meshs
                "v3d": out["v3d"][i],
                "j3d": out["j3d"][i],
                "j2d": out["j2d"][i],
            }
            persons.append(person)

        return persons

    def forward(
        self,
        x,
        idx=None,
        max_dist=None,
        det_thresh=0.3,
        nms_kernel_size=3,
        K=None,
        is_training=False,
        *args,
        **kwargs,
    ):
        """
        Forward pass of the model and compute the loss according to the groundtruth
        Args:
            - x: RGB image - [bs,3,224,224]
            - idx: GT location of persons - tuple of 3 tensor of shape [p]
            - idx_j2d: GT location of 2d-kpts for each detected humans - tensor of shape [bs',14,2] - location in pixel space
        Return:
            - y: [bs,D,16,16]
        """
        persons = []
        out = {}

        # Feature extraction
        z = self.backbone(x)
        B, N, C = z.size()  # [bs,256,768]

        # Detection
        scores, scores_det, idx = self.detection(
            z,
            nms_kernel_size=nms_kernel_size,
            det_thresh=det_thresh,
            N=N,
            idx=idx,
            max_dist=max_dist,
            is_training=is_training,
        )
        if torch.any(scores_det < 0.1):
            return persons
        if len(idx[1]) == 0 and not is_training:
            # no humans detected in the frame
            return persons

        # Map of Dense Feature
        z = unpatch(
            z, patch_size=1, c=z.shape[2], img_size=int(np.sqrt(N))
        )  # [bs,D,16,16]
        z_all = z

        # Extract the 'central' features
        z = torch.reshape(
            z, (z.shape[0], 1, z.shape[1] // 1, z.shape[2], z.shape[3])
        )  # [bs,stack_K,D,16,16]
        z_central = z[idx[0], idx[3], :, idx[1], idx[2]]  # dense vectors

        # 2D offset regression
        offset = self.mlp_offset(z_central)

        # Camera instrincs
        K_det = K[idx[0]]  # cameras for detected person
        z_K = self.embedd_camera(K, z)  # Embed viewing directions.
        z_central = torch.cat(
            [z_central, z_K[idx[0], idx[1], idx[2]]], 1
        )  # Add to query tokens.
        z_all = torch.cat(
            [z_all, z_K.permute(0, 3, 1, 2)], 1
        )  # for the cross-attention only
        z = torch.cat([z, z_K.permute(0, 3, 1, 2).unsqueeze(1)], 2)

        # Distance for estimating the 3D location in 3D space
        loc = torch.stack([idx[2], idx[1]]).permute(
            1, 0
        )  # Moving from higher resolution the location of the pelvis
        loc = (loc + 0.5 + offset) * self.patch_size

        # SMPL parameter regression
        kv = z_all[
            idx[0]
        ]  # retrieving dense features associated to each central vector
        pred_smpl_params, pred_cam = self.x_attention_head(
            z_central, kv, idx_0=idx[0], idx_det=idx
        )

        # Get outputs from the SMPL layer.
        shape = pred_smpl_params["betas"]
        rotmat = torch.cat(
            [pred_smpl_params["global_orient"], pred_smpl_params["body_pose"]], 1
        )
        expression = pred_smpl_params["expression"]
        rotvec = roma.rotmat_to_rotvec(rotmat)

        # Distance
        dist = pred_cam[:, 0][:, None]
        out["dist_postprocessed"] = (
            dist  # before applying any post-processing such as focal length normalization, inverse or log
        )
        dist = self.to_euclidean_dist(x, dist, K_det)

        # Populate output dictionnary
        out.update(
            {
                "scores": scores,
                "offset": offset,
                "dist": dist,
                "expression": expression,
                "rotmat": rotmat,
                "shape": shape,
                "rotvec": rotvec,
                "loc": loc,
            }
        )

        assert (
            rotvec.shape[0] == shape.shape[0] == loc.shape[0] == dist.shape[0]
        ), "Incoherent shapes"

        if not self.output_mesh:
            out.update(
                {
                    "K_det": K_det,
                    "scores_det": scores_det,
                    "idx": idx,
                }
            )
            return out

        # Neutral
        smpl_out = self.smpl_layer[f"neutral_{self.num_betas}"](
            rotvec, shape, loc, dist, None, K=K_det, expression=expression
        )
        out.update(smpl_out)

        # Return
        if is_training:
            return out
        else:
            # Populate a dictionnary for each person
            for i in range(idx[0].shape[0]):
                person = {
                    # Detection
                    "scores": scores_det[i],  # detection scores
                    "loc": out["loc"][i],  # 2d pixel location of the primary keypoints
                    # SMPL-X params
                    "transl": out["transl"][
                        i
                    ],  # from the primary keypoint i.e. the head
                    "transl_pelvis": out["transl_pelvis"][i],  # of the pelvis joint
                    "rotvec": out["rotvec"][i],
                    "expression": out["expression"][i],
                    "shape": out["shape"][i],
                    # SMPL-X meshs
                    "v3d": out["v3d"][i],
                    "j3d": out["j3d"][i],
                    "j2d": out["j2d"][i],
                    "dist": out["dist"][i],
                    "offset": out["offset"][i],
                }
                persons.append(person)

            return persons


class HPH(nn.Module):
    """Cross-attention based SMPL Transformer decoder

    Code modified from:
    https://github.com/shubham-goel/4D-Humans/blob/a0def798c7eac811a63c8220fcc22d983b39785e/hmr2/models/heads/smpl_head.py#L17
    https://github.com/shubham-goel/4D-Humans/blob/a0def798c7eac811a63c8220fcc22d983b39785e/hmr2/models/components/pose_transformer.py#L301
    """

    def __init__(
        self,
        num_body_joints=52,
        context_dim=1280,
        dim=1024,
        depth=2,
        heads=8,
        mlp_dim=1024,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        at_token_res=32,
        num_betas=10,
        smplx_dir=None,
    ):
        super().__init__()

        self.joint_rep_type, self.joint_rep_dim = "6d", 6
        self.num_body_joints = num_body_joints
        self.nrot = self.num_body_joints + 1

        npose = self.joint_rep_dim * (self.num_body_joints + 1)
        self.npose = npose

        self.depth = (depth,)
        self.heads = (heads,)
        self.res = at_token_res
        self.input_is_mean_shape = True
        _context_dim = context_dim  # for the central features
        self.num_betas = num_betas
        assert num_betas in [10, 11]

        # Transformer Decoder setup.
        # Based on https://github.com/shubham-goel/4D-Humans/blob/8830bb330558eea2395b7f57088ef0aae7f8fa22/hmr2/configs_hydra/experiment/hmr_vit_transformer.yaml#L35
        transformer_args = dict(
            num_tokens=1,
            token_dim=(
                (npose + self.num_betas + 3 + _context_dim)
                if self.input_is_mean_shape
                else 1
            ),
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            context_dim=context_dim,
        )
        self.transformer = TransformerDecoder(**transformer_args)

        dim = transformer_args["dim"]

        # Final decoders to regress targets
        self.decpose, self.decshape, self.deccam, self.decexpression = [
            nn.Linear(dim, od) for od in [npose, num_betas, 3, 10]
        ]

        # Register bufffers for the smpl layer.
        self.set_smpl_init(smplx_dir)

        # Init learned embeddings for the cross attention queries
        self.init_learned_queries(context_dim)

    def init_learned_queries(self, context_dim, std=0.2):
        """Init learned embeddings for queries"""
        self.cross_queries_x = nn.Parameter(torch.zeros(self.res, context_dim))
        torch.nn.init.normal_(self.cross_queries_x, std=std)

        self.cross_queries_y = nn.Parameter(torch.zeros(self.res, context_dim))
        torch.nn.init.normal_(self.cross_queries_y, std=std)

        self.cross_values_x = nn.Parameter(torch.zeros(self.res, context_dim))
        torch.nn.init.normal_(self.cross_values_x, std=std)

        self.cross_values_y = nn.Parameter(
            nn.Parameter(torch.zeros(self.res, context_dim))
        )
        torch.nn.init.normal_(self.cross_values_y, std=std)

    def set_smpl_init(self, smplx_dir):
        """Fetch saved SMPL parameters and register buffers."""
        mean_params = np.load(os.path.join(smplx_dir, "smpl_mean_params.npz"))
        if self.nrot == 53:
            init_body_pose = (
                torch.eye(3)
                .reshape(1, 3, 3)
                .repeat(self.nrot, 1, 1)[:, :, :2]
                .flatten(1)
                .reshape(1, -1)
            )
            init_body_pose[:, : 24 * 6] = torch.from_numpy(
                mean_params["pose"][:]
            ).float()  # global_orient+body_pose from SMPL
        else:
            init_body_pose = torch.from_numpy(
                mean_params["pose"].astype(np.float32)
            ).unsqueeze(0)

        init_betas = torch.from_numpy(mean_params["shape"].astype("float32")).unsqueeze(
            0
        )
        init_cam = torch.from_numpy(mean_params["cam"].astype(np.float32)).unsqueeze(0)
        init_betas_kid = torch.cat(
            [init_betas, torch.zeros_like(init_betas[:, [0]])], 1
        )
        init_expression = 0.0 * torch.from_numpy(
            mean_params["shape"].astype("float32")
        ).unsqueeze(0)

        if self.num_betas == 11:
            init_betas = torch.cat([init_betas, torch.zeros_like(init_betas[:, :1])], 1)

        self.register_buffer("init_body_pose", init_body_pose)
        self.register_buffer("init_betas", init_betas)
        self.register_buffer("init_betas_kid", init_betas_kid)
        self.register_buffer("init_cam", init_cam)
        self.register_buffer("init_expression", init_expression)

    def cross_attn_inputs(self, x, x_central, idx_0, idx_det):
        """Reshape and pad x_central to have the right shape for Cross-attention processing.
        Inject learned embeddings to query and key inputs at the location of detected people.
        """

        h, w = x.shape[2], x.shape[3]
        x = einops.rearrange(x, "b c h w -> b (h w) c")

        assert idx_0 is not None, "Learned cross queries only work with multicross"

        if idx_0.shape[0] > 0:
            # reconstruct the batch/nb_people dimensions: pad for images with fewer people than max.
            counts, idx_det_0 = rebatch(idx_0, idx_det)
            old_shape = x_central.shape

            # Legacy check for old versions
            assert idx_det is not None, "idx_det needed for learned_attention"

            # xx is the tensor with all features
            xx = einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            # Get learned embeddings for queries, at positions with detected people.
            queries_xy = (
                self.cross_queries_x[idx_det[1]] + self.cross_queries_y[idx_det[2]]
            )
            # Add the embedding to the central features.
            x_central = x_central + queries_xy
            assert x_central.shape == old_shape, "Problem with shape"

            # Make it a tensor of dim. [batch, max_ppl_along_batch, ...]
            x_central, mask = pad_to_max(x_central, counts)

            # xx = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            xx = xx[torch.cumsum(counts, dim=0) - 1]

            # Inject leared embeddings for key/values at detected locations.
            values_xy = (
                self.cross_values_x[idx_det[1]] + self.cross_values_y[idx_det[2]]
            )
            xx[idx_det_0, :, idx_det[1], idx_det[2]] += values_xy

            x = einops.rearrange(xx, "b c h w -> b (h w) c")
            num_ppl = x_central.shape[1]
        else:
            mask = None
            num_ppl = 1
            counts = None
        return x, x_central, mask, num_ppl, counts

    def forward(self, x_central, x, idx_0=None, idx_det=None, **kwargs):
        """ "
        Forward the HPH module.
        """
        batch_size = x.shape[0]

        # Reshape inputs for cross attention and inject learned embeddings for queries and values.
        x, x_central, mask, num_ppl, counts = self.cross_attn_inputs(
            x, x_central, idx_0, idx_det
        )

        # Add init (mean smpl params) to the query for each quantity being regressed.
        bs = x_central.shape[0] if idx_0.shape[0] else batch_size
        expand = lambda x: x.expand(bs, num_ppl, -1)
        pred_body_pose, pred_betas, pred_cam, pred_expression = [
            expand(x)
            for x in [
                self.init_body_pose,
                self.init_betas,
                self.init_cam,
                self.init_expression,
            ]
        ]
        token = torch.cat([x_central, pred_body_pose, pred_betas, pred_cam], dim=-1)
        if len(token.shape) == 2:
            token = token[:, None, :]

        # Process query and inputs with the cross-attention module.
        token_out = self.transformer(token, context=x, mask=mask)

        # Reshape outputs from [batch_size, nmax_ppl, ...] to [total_ppl, ...]
        if mask is not None:
            # Stack along batch axis.
            token_out_list = [token_out[i, :c, ...] for i, c in enumerate(counts)]
            token_out = torch.concat(token_out_list, dim=0)
        else:
            token_out = token_out.squeeze(1)  # (B, C)

        # Decoded output token and add to init for each quantity to regress.
        reshape = (
            (lambda x: x)
            if idx_0.shape[0] == 0
            else (lambda x: x[0, 0, ...][None, ...])
        )
        decoders = [self.decpose, self.decshape, self.deccam, self.decexpression]
        inits = [pred_body_pose, pred_betas, pred_cam, pred_expression]
        pred_body_pose, pred_betas, pred_cam, pred_expression = [
            d(token_out) + reshape(i) for d, i in zip(decoders, inits)
        ]

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = rot6d_to_rotmat

        # conversion
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(
            batch_size, self.num_body_joints + 1, 3, 3
        )

        # Build the output dict
        pred_smpl_params = {
            "global_orient": pred_body_pose[:, [0]],
            "body_pose": pred_body_pose[:, 1:],
            "betas": pred_betas,
            #'betas_kid': pred_betas_kid,
            "expression": pred_expression,
        }
        return pred_smpl_params, pred_cam  # , pred_smpl_params_list


def regression_mlp(layers_sizes):
    """
    Return a fully connected network.
    """
    assert len(layers_sizes) >= 2
    in_features = layers_sizes[0]
    layers = []
    for i in range(1, len(layers_sizes) - 1):
        out_features = layers_sizes[i]
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, layers_sizes[-1]))
    return torch.nn.Sequential(*layers)


def apply_threshold(det_thresh, _scores):
    """Apply thresholding to detection scores; if stack_K is used and det_thresh is a list, apply to each channel separately"""
    if isinstance(det_thresh, list):
        det_thresh = det_thresh[0]
    idx = torch.where(_scores >= det_thresh)
    return idx


def _nms(heat, kernel=3):
    """easy non maximal supression (as in CenterNet)"""

    if kernel not in [2, 4]:
        pad = (kernel - 1) // 2
    else:
        if kernel == 2:
            pad = 1
        else:
            pad = 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)

    if hmax.shape[2] > heat.shape[2]:
        hmax = hmax[:, :, : heat.shape[2], : heat.shape[3]]

    keep = (hmax == heat).float()

    return heat * keep


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


if __name__ == "__main__":
    Model()
