# -*- coding: utf-8 -*-# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-01-08 21:42:24, Version 0.0, SMPLX + FLAME2019 + Voxel-Based Queries.
# @Function      : SMPLX-related functions
# @Description   : 1.canonical query, 2.offset, 3.blendshape -> 4.posed-view

import copy
import math
import os
import os.path as osp
import pdb
import pickle
import sys

sys.path.append("./")
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import trimesh
from pytorch3d.io import load_ply, save_ply
from pytorch3d.ops import SubdivideMeshes, knn_points
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from smplx.lbs import batch_rigid_transform
from torch.nn import functional as F

from LHM.models.rendering.mesh_utils import Mesh
from LHM.models.rendering.smplx import smplx
from LHM.models.rendering.smplx.smplx.lbs import blend_shapes
from LHM.models.rendering.smplx.vis_utils import render_mesh

"""
Subdivide a triangle mesh by adding a new vertex at the center of each edge and dividing each face into four new faces.
Vectors of vertex attributes can also be subdivided by averaging the values of the attributes at the two vertices which form each edge. 
This implementation preserves face orientation - if the vertices of a face are all ordered counter-clockwise, 
then the faces in the subdivided meshes will also have their vertices ordered counter-clockwise.
If meshes is provided as an input, the initializer performs the relatively expensive computation of determining the new face indices. 
This one-time computation can be reused for all meshes with the same face topology but different vertex positions.
"""


def avaliable_device():

    import torch

    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device


class SMPLX_Mesh(object):
    def __init__(
        self,
        human_model_path,
        shape_param_dim=100,
        expr_param_dim=50,
        subdivide_num=2,
        cano_pose_type=0,
    ):
        """SMPLX using dense sampling"""
        super().__init__()
        self.human_model_path = human_model_path
        self.shape_param_dim = shape_param_dim
        self.expr_param_dim = expr_param_dim
        if shape_param_dim == 10 and expr_param_dim == 10:
            self.layer_arg = {
                "create_global_orient": False,
                "create_body_pose": False,
                "create_left_hand_pose": False,
                "create_right_hand_pose": False,
                "create_jaw_pose": False,
                "create_leye_pose": False,
                "create_reye_pose": False,
                "create_betas": False,
                "create_expression": False,
                "create_transl": False,
            }
            self.layer = {
                gender: smplx.create(
                    human_model_path,
                    "smplx",
                    gender=gender,
                    num_betas=self.shape_param_dim,
                    num_expression_coeffs=self.expr_param_dim,
                    use_pca=False,
                    use_face_contour=False,
                    flat_hand_mean=True,
                    **self.layer_arg,
                )
                for gender in ["neutral", "male", "female"]
            }
        else:
            self.layer_arg = {
                "create_global_orient": False,
                "create_body_pose": False,
                "create_left_hand_pose": False,
                "create_right_hand_pose": False,
                "create_jaw_pose": False,
                "create_leye_pose": False,
                "create_reye_pose": False,
                "create_betas": False,
                "create_expression": False,
                "create_transl": False,
            }
            self.layer = {
                gender: smplx.create(
                    human_model_path,
                    "smplx",
                    gender=gender,
                    num_betas=self.shape_param_dim,
                    num_expression_coeffs=self.expr_param_dim,
                    use_pca=False,
                    use_face_contour=True,
                    flat_hand_mean=True,
                    **self.layer_arg,
                )
                for gender in ["neutral", "male", "female"]
            }

        self.face_vertex_idx = np.load(
            osp.join(human_model_path, "smplx", "SMPL-X__FLAME_vertex_ids.npy")
        )
        if shape_param_dim == 10 and expr_param_dim == 10:
            print("not using flame expr")
        else:
            self.layer = {
                gender: self.get_expr_from_flame(self.layer[gender])
                for gender in ["neutral", "male", "female"]
            }
        self.vertex_num = 10475
        self.face_orig = self.layer["neutral"].faces.astype(np.int64)
        self.is_cavity, self.face = self.add_cavity()
        with open(
            osp.join(human_model_path, "smplx", "MANO_SMPLX_vertex_ids.pkl"), "rb"
        ) as f:
            hand_vertex_idx = pickle.load(f, encoding="latin1")
        self.rhand_vertex_idx = hand_vertex_idx["right_hand"]
        self.lhand_vertex_idx = hand_vertex_idx["left_hand"]
        self.expr_vertex_idx = self.get_expr_vertex_idx()

        # SMPLX joint set
        self.joint_num = (
            55  # 22 (body joints: 21 + 1) + 3 (face joints) + 30 (hand joints)
        )
        self.joints_name = (
            "Pelvis",
            "L_Hip",
            "R_Hip",
            "Spine_1",
            "L_Knee",
            "R_Knee",
            "Spine_2",
            "L_Ankle",
            "R_Ankle",
            "Spine_3",
            "L_Foot",
            "R_Foot",
            "Neck",
            "L_Collar",
            "R_Collar",
            "Head",
            "L_Shoulder",  # 16
            "R_Shoulder",  # 17
            "L_Elbow",
            "R_Elbow",
            "L_Wrist",
            "R_Wrist",  # body joints
            "Jaw",
            "L_Eye",
            "R_Eye",  # face joints
            "L_Index_1",
            "L_Index_2",
            "L_Index_3",
            "L_Middle_1",
            "L_Middle_2",
            "L_Middle_3",
            "L_Pinky_1",
            "L_Pinky_2",
            "L_Pinky_3",
            "L_Ring_1",
            "L_Ring_2",
            "L_Ring_3",
            "L_Thumb_1",
            "L_Thumb_2",
            "L_Thumb_3",  # left hand joints
            "R_Index_1",
            "R_Index_2",
            "R_Index_3",
            "R_Middle_1",
            "R_Middle_2",
            "R_Middle_3",
            "R_Pinky_1",
            "R_Pinky_2",
            "R_Pinky_3",
            "R_Ring_1",
            "R_Ring_2",
            "R_Ring_3",
            "R_Thumb_1",
            "R_Thumb_2",
            "R_Thumb_3",  # right hand joints
        )
        self.root_joint_idx = self.joints_name.index("Pelvis")
        self.joint_part = {
            "body": range(
                self.joints_name.index("Pelvis"), self.joints_name.index("R_Wrist") + 1
            ),
            "face": range(
                self.joints_name.index("Jaw"), self.joints_name.index("R_Eye") + 1
            ),
            "lhand": range(
                self.joints_name.index("L_Index_1"),
                self.joints_name.index("L_Thumb_3") + 1,
            ),
            "rhand": range(
                self.joints_name.index("R_Index_1"),
                self.joints_name.index("R_Thumb_3") + 1,
            ),
            "lower_body": [
                self.joints_name.index("Pelvis"),
                self.joints_name.index("R_Hip"),
                self.joints_name.index("L_Hip"),
                self.joints_name.index("R_Knee"),
                self.joints_name.index("L_Knee"),
                self.joints_name.index("R_Ankle"),
                self.joints_name.index("L_Ankle"),
                self.joints_name.index("R_Foot"),
                self.joints_name.index("L_Foot"),
            ],
            "upper_body": [
                self.joints_name.index("Spine_1"),
                self.joints_name.index("Spine_2"),
                self.joints_name.index("Spine_3"),
                self.joints_name.index("L_Collar"),
                self.joints_name.index("R_Collar"),
                self.joints_name.index("L_Shoulder"),
                self.joints_name.index("R_Shoulder"),
                self.joints_name.index("L_Elbow"),
                self.joints_name.index("R_Elbow"),
                self.joints_name.index("L_Wrist"),
                self.joints_name.index("R_Wrist"),
            ],
        }

        self.lower_body_vertex_idx = self.get_body("lower_body")
        self.upper_body_vertex_idx = self.get_body("upper_body")

        self.neutral_body_pose = torch.zeros(
            (len(self.joint_part["body"]) - 1, 3)
        )  # 大 pose in axis-angle representation (body pose without root joint)
        if cano_pose_type == 0:  # exavatar-cano-pose
            self.neutral_body_pose[0] = torch.FloatTensor([0, 0, 1])
            self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -1])
        else:  #
            self.neutral_body_pose[0] = torch.FloatTensor([0, 0, math.pi / 9])
            self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -math.pi / 9])

        self.neutral_jaw_pose = torch.FloatTensor([1 / 3, 0, 0])

        # subdivider

        self.body_head_mapping = self.get_body_face_mapping()

        self.register_constrain_prior()

    def register_constrain_prior(self):
        """As video cannot provide insufficient supervision for the canonical space, we add some human prior to constrain the rotation. Although it is a trick, it is very effective."""
        constrain_body = np.load(
            "./pretrained_models/voxel_grid/human_prior_constrain.npz"
        )["masks"]

        self.constrain_body_vertex_idx = np.where(constrain_body > 0)[0]

    def get_body(self, name):
        """using skinning to find lower body vertices."""
        lower_body_skinning_index = set(self.joint_part[name])
        skinning_weight = self.layer["neutral"].lbs_weights.float()
        skinning_part = skinning_weight.argmax(1)
        skinning_part = skinning_part.cpu().numpy()
        lower_body_vertice_idx = []
        for v_id, v_s in enumerate(skinning_part):
            if v_s in lower_body_skinning_index:
                lower_body_vertice_idx.append(v_id)

        lower_body_vertice_idx = np.asarray(lower_body_vertice_idx)

        return lower_body_vertice_idx

    def get_expr_from_flame(self, smplx_layer):
        flame_layer = smplx.create(
            self.human_model_path,
            "flame",
            gender="neutral",
            num_betas=self.shape_param_dim,
            num_expression_coeffs=self.expr_param_dim,
        )
        smplx_layer.expr_dirs[self.face_vertex_idx, :, :] = flame_layer.expr_dirs
        return smplx_layer

    def set_id_info(self, shape_param, face_offset, joint_offset, locator_offset):
        self.shape_param = shape_param
        self.face_offset = face_offset
        self.joint_offset = joint_offset
        self.locator_offset = locator_offset

    def get_joint_offset(self, joint_offset):
        device = joint_offset.device
        batch_size = joint_offset.shape[0]
        weight = torch.ones((batch_size, self.joint_num, 1)).float().to(device)
        weight[:, self.root_joint_idx, :] = 0
        joint_offset = joint_offset * weight
        return joint_offset

    def get_subdivider(self, subdivide_num):
        vert = self.layer["neutral"].v_template.float().cuda()
        face = torch.LongTensor(self.face).cuda()
        mesh = Meshes(vert[None, :, :], face[None, :, :])

        if subdivide_num > 0:
            subdivider_list = [SubdivideMeshes(mesh)]
            for i in range(subdivide_num - 1):
                mesh = subdivider_list[-1](mesh)
                subdivider_list.append(SubdivideMeshes(mesh))
        else:
            subdivider_list = [mesh]
        return subdivider_list

    def get_body_face_mapping(self):
        face_vertex_idx = self.face_vertex_idx
        face_vertex_set = set(face_vertex_idx)
        face = self.face.reshape(-1).tolist()
        face_label = [f in face_vertex_set for f in face]
        face_label = np.asarray(face_label).reshape(-1, 3)
        face_label = face_label.sum(-1)
        face_id = np.where(face_label == 3)[0]

        head_face = self.face[face_id]

        body_set = set(np.arange(self.vertex_num))
        body_v_id = body_set - face_vertex_set
        body_v_id = np.array(list(body_v_id))

        body_face_id = np.where(face_label == 0)[0]
        body_face = self.face[body_face_id]

        ret_dict = dict(
            head=dict(face=head_face, vert=face_vertex_idx),
            body=dict(face=body_face, vert=body_v_id),
        )

        return ret_dict

    def add_cavity(self):
        lip_vertex_idx = [2844, 2855, 8977, 1740, 1730, 1789, 8953, 2892]
        is_cavity = np.zeros((self.vertex_num), dtype=np.float32)
        is_cavity[lip_vertex_idx] = 1.0

        cavity_face = [[0, 1, 7], [1, 2, 7], [2, 3, 5], [3, 4, 5], [2, 5, 6], [2, 6, 7]]
        face_new = list(self.face_orig)
        for face in cavity_face:
            v1, v2, v3 = face
            face_new.append(
                [lip_vertex_idx[v1], lip_vertex_idx[v2], lip_vertex_idx[v3]]
            )
        face_new = np.array(face_new, dtype=np.int64)
        return is_cavity, face_new

    def get_expr_vertex_idx(self):
        # FLAME 2020 has all vertices of expr_vertex_idx. use FLAME 2019
        """
        SMPLX + FLAME2019 Version
        according to LBS weights to search related vertices ID
        """

        with open(
            osp.join(self.human_model_path, "flame", "2019", "generic_model.pkl"), "rb"
        ) as f:
            flame_2019 = pickle.load(f, encoding="latin1")
        vertex_idxs = np.where(
            (flame_2019["shapedirs"][:, :, 300 : 300 + self.expr_param_dim] != 0).sum(
                (1, 2)
            )
            > 0
        )[
            0
        ]  # FLAME.SHAPE_SPACE_DIM == 300

        # exclude neck and eyeball regions
        flame_joints_name = ("Neck", "Head", "Jaw", "L_Eye", "R_Eye")
        expr_vertex_idx = []
        flame_vertex_num = flame_2019["v_template"].shape[0]
        is_neck_eye = torch.zeros((flame_vertex_num)).float()
        is_neck_eye[
            flame_2019["weights"].argmax(1) == flame_joints_name.index("Neck")
        ] = 1
        is_neck_eye[
            flame_2019["weights"].argmax(1) == flame_joints_name.index("L_Eye")
        ] = 1
        is_neck_eye[
            flame_2019["weights"].argmax(1) == flame_joints_name.index("R_Eye")
        ] = 1
        for idx in vertex_idxs:
            if is_neck_eye[idx]:
                continue
            expr_vertex_idx.append(idx)

        expr_vertex_idx = np.array(expr_vertex_idx)
        expr_vertex_idx = self.face_vertex_idx[expr_vertex_idx]

        return expr_vertex_idx

    def get_arm(self, mesh_neutral_pose, skinning_weight):
        normal = (
            Meshes(
                verts=mesh_neutral_pose[None, :, :],
                faces=torch.LongTensor(self.face_upsampled).cuda()[None, :, :],
            )
            .verts_normals_packed()
            .reshape(self.vertex_num_upsampled, 3)
            .detach()
        )
        part_label = skinning_weight.argmax(1)
        is_arm = 0
        for name in ("R_Shoulder", "R_Elbow", "L_Shoulder", "L_Elbow"):
            is_arm = is_arm + (part_label == self.joints_name.index(name))
        is_arm = is_arm > 0
        is_upper_arm = is_arm * (normal[:, 1] > math.cos(math.pi / 3))
        is_lower_arm = is_arm * (normal[:, 1] <= math.cos(math.pi / 3))
        return is_upper_arm, is_lower_arm


class SMPLXVoxelMeshModel(nn.Module):
    def __init__(
        self,
        human_model_path,
        gender,
        subdivide_num,
        expr_param_dim=50,
        shape_param_dim=100,
        cano_pose_type=0,
        body_face_ratio=3,
        dense_sample_points=40000,
        apply_pose_blendshape=False,
    ) -> None:
        super().__init__()

        self.smpl_x = SMPLX_Mesh(
            human_model_path=human_model_path,
            shape_param_dim=shape_param_dim,
            expr_param_dim=expr_param_dim,
            subdivide_num=subdivide_num,
            cano_pose_type=cano_pose_type,
        )
        self.smplx_layer = copy.deepcopy(self.smpl_x.layer[gender])

        # register
        self.apply_pose_blendshape = apply_pose_blendshape
        self.cano_pose_type = cano_pose_type
        self.dense_sample(body_face_ratio, dense_sample_points)
        self.smplx_init()

    def rebuild_mesh(self, v, vertices_id, faces_id, num_dense_samples):
        choice_vertices = v[vertices_id]

        new_mapping = dict()

        for new_id, vertice_id in enumerate(vertices_id):
            new_mapping[vertice_id] = new_id

        faces_id_list = faces_id.reshape(-1).tolist()

        new_faces_id = []
        for face_id in faces_id_list:
            new_faces_id.append(new_mapping[face_id])
        new_faces_id = torch.from_numpy(np.array(new_faces_id).reshape(faces_id.shape))

        mymesh = Mesh(v=choice_vertices, f=new_faces_id)

        dense_sample_pts = mymesh.sample_surface(num_dense_samples).detach().cpu()

        return dense_sample_pts

    def dense_sample(self, body_face_ratio, dense_sample_points):

        buff_path = f"./pretrained_models/dense_sample_points/{self.cano_pose_type}_{dense_sample_points}.ply"

        if os.path.exists(buff_path):
            dense_sample_pts, _ = load_ply(buff_path)

            _bin = dense_sample_points // (body_face_ratio + 1)
            body_pts = int(_bin * body_face_ratio)
            self.is_body = torch.arange(dense_sample_pts.shape[0])
            self.is_body[:body_pts] = 1
            self.is_body[body_pts:] = 0
            self.dense_pts = dense_sample_pts
        else:
            smpl_x = self.smpl_x
            body_face_mapping = smpl_x.get_body_face_mapping()
            face = smpl_x.face
            template_verts = self.smplx_layer.v_template

            _bin = dense_sample_points // (body_face_ratio + 1)

            # build body mesh
            body_pts = int(_bin * body_face_ratio)
            body_dict = body_face_mapping["body"]
            face = body_dict["face"]
            verts = body_dict["vert"]

            dense_body_pts = self.rebuild_mesh(template_verts, verts, face, body_pts)

            # build face mesh
            head_pts = int(_bin)
            head_dict = body_face_mapping["head"]
            head_face = head_dict["face"]
            head_verts = head_dict["vert"]
            dense_head_pts = self.rebuild_mesh(
                template_verts, head_verts, head_face, head_pts
            )

            self.dense_pts = torch.cat([dense_body_pts, dense_head_pts], dim=0)
            self.is_body = torch.arange(self.dense_pts.shape[0])
            self.is_body[:body_pts] = 1
            self.is_body[body_pts:] = 0

            save_ply(buff_path, self.dense_pts)

    @torch.no_grad()
    def voxel_smooth_register(
        self, voxel_v, template_v, lbs_weights, k=3, smooth_k=30, smooth_n=3000
    ):
        """Smooth KNN to handle skirt deformation."""

        lbs_weights = lbs_weights.cuda()

        dist = knn_points(
            voxel_v.unsqueeze(0).cuda(),
            template_v.unsqueeze(0).cuda(),
            K=1,
            return_nn=True,
        )
        mesh_dis = torch.sqrt(dist.dists)
        mesh_indices = dist.idx.squeeze(0, -1)
        knn_lbs_weights = lbs_weights[mesh_indices]

        mesh_dis = mesh_dis.squeeze()

        print(f"Using k = {smooth_k}, N={smooth_n} for LBS smoothing")
        # Smooth Skinning

        knn_dis = knn_points(
            voxel_v.unsqueeze(0).cuda(),
            voxel_v.unsqueeze(0).cuda(),
            K=smooth_k + 1,
            return_nn=True,
        )
        voxel_dis = torch.sqrt(knn_dis.dists)
        voxel_indices = knn_dis.idx
        voxel_indices = voxel_indices.squeeze()[:, 1:]
        voxel_dis = voxel_dis.squeeze()[:, 1:]

        knn_weights = 1.0 / (mesh_dis[voxel_indices] * voxel_dis)
        knn_weights = knn_weights / knn_weights.sum(-1, keepdim=True)  # [N, K]

        def dists_to_weights(
            dists: torch.Tensor, low: float = None, high: float = None
        ):
            if low is None:
                low = high
            if high is None:
                high = low
            assert high >= low
            weights = dists.clone()
            weights[dists <= low] = 0.0
            weights[dists >= high] = 1.0
            indices = (dists > low) & (dists < high)
            weights[indices] = (dists[indices] - low) / (high - low)
            return weights

        update_weights = dists_to_weights(mesh_dis, low=0.01).unsqueeze(-1)  # [N, 1]

        from tqdm import tqdm

        for _ in tqdm(range(smooth_n)):
            N, _ = update_weights.shape
            new_lbs_weights_chunk_list = []
            for chunk_i in range(0, N, 1000000):

                knn_weights_chunk = knn_weights[chunk_i : chunk_i + 1000000]
                voxel_indices_chunk = voxel_indices[chunk_i : chunk_i + 1000000]

                new_lbs_weights_chunk = torch.einsum(
                    "nk,nkj->nj",
                    knn_weights_chunk,
                    knn_lbs_weights[voxel_indices_chunk],
                )
                new_lbs_weights_chunk_list.append(new_lbs_weights_chunk)
            new_lbs_weights = torch.cat(new_lbs_weights_chunk_list, dim=0)
            if update_weights is None:
                knn_lbs_weights = new_lbs_weights
            else:
                knn_lbs_weights = (
                    1.0 - update_weights
                ) * knn_lbs_weights + update_weights * new_lbs_weights

        return knn_lbs_weights

    def voxel_skinning_init(self, scale_ratio=1.05, voxel_size=256):

        skinning_weight = self.smplx_layer.lbs_weights.float()

        smplx_data = {"betas": torch.zeros(1, self.smpl_x.shape_param_dim)}
        device = skinning_weight.device

        _, mesh_neutral_pose_wo_upsample, _ = self.get_neutral_pose_human(
            jaw_zero_pose=True,
            use_id_info=True,
            shape_param=smplx_data["betas"],
            device=device,
            face_offset=smplx_data.get("face_offset", None),
            joint_offset=smplx_data.get("joint_offset", None),
        )

        template_verts = mesh_neutral_pose_wo_upsample.squeeze(0)

        def scale_voxel_size(template_verts, scale_ratio=1.0):
            min_values, _ = torch.min(template_verts, dim=0)
            max_values, _ = torch.max(template_verts, dim=0)

            center = (min_values + max_values) / 2
            size = max_values - min_values

            scale_size = size * scale_ratio

            upper = center + scale_size / 2
            bottom = center - scale_size / 2

            return torch.cat([bottom[:, None], upper[:, None]], dim=1)

        mini_size_bbox = scale_voxel_size(template_verts, scale_ratio)
        z_voxel_size = voxel_size // 2

        # build coordinate
        x_range = np.linspace(0, voxel_size - 1, voxel_size) / (
            voxel_size - 1
        )  # from 0 to 255，
        y_range = np.linspace(0, voxel_size - 1, voxel_size) / (voxel_size - 1)
        z_range = np.linspace(0, z_voxel_size - 1, z_voxel_size) / (z_voxel_size - 1)

        x, y, z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        coordinates = torch.from_numpy(np.stack([x, y, z], axis=-1))

        coordinates[..., 0] = mini_size_bbox[0, 0] + coordinates[..., 0] * (
            mini_size_bbox[0, 1] - mini_size_bbox[0, 0]
        )
        coordinates[..., 1] = mini_size_bbox[1, 0] + coordinates[..., 1] * (
            mini_size_bbox[1, 1] - mini_size_bbox[1, 0]
        )
        coordinates[..., 2] = mini_size_bbox[2, 0] + coordinates[..., 2] * (
            mini_size_bbox[2, 1] - mini_size_bbox[2, 0]
        )

        coordinates = coordinates.view(-1, 3).float()
        coordinates = coordinates.cuda()

        if os.path.exists(f"./pretrained_models/voxel_grid/voxel_{voxel_size}.pth"):
            print(f"load voxel_grid voxel_{voxel_size}.pth")
            voxel_flat = torch.load(
                os.path.join(f"pretrained_models/voxel_grid/voxel_{voxel_size}.pth"),
                map_location=avaliable_device(),
            )
        else:
            voxel_flat = self.voxel_smooth_register(
                coordinates, template_verts, skinning_weight, k=1, smooth_n=3000
            )

            torch.save(
                voxel_flat,
                os.path.join(f"pretrained_models/voxel_grid/voxel_{voxel_size}.pth"),
            )

        N, LBS_F = voxel_flat.shape

        # x, y, z, C
        voxel_grid_original = voxel_flat.view(
            voxel_size, voxel_size, z_voxel_size, LBS_F
        )

        # [W H D 55]->[55, D, H, W]
        voxel_grid = voxel_grid_original.permute(3, 2, 1, 0)

        return voxel_grid, mini_size_bbox

    def smplx_init(self):
        """
        Initialize the sub-devided smplx model by registering buffers for various attributes
        This method performs the following steps:
        1. Upsamples the mesh and other assets.
        2. Computes skinning weights, pose directions, expression directions, and various flags for different body parts.
        3. Reshapes and permutes the pose and expression directions.
        4. Converts the flags to boolean values.
        5. Registers buffers for the computed attributes.
        Args:
            self: The object instance.
        Returns:
            None
        """

        def _query(weights, indx):

            weights = weights.squeeze(0)
            assert weights.dim() == 2

            return weights[indx]

        smpl_x = self.smpl_x

        # using KNN to query subdivided mesh
        dense_pts = self.dense_pts.cuda()
        template_verts = self.smplx_layer.v_template

        nn_vertex_idxs = knn_points(
            dense_pts.unsqueeze(0).cuda(),
            template_verts.unsqueeze(0).cuda(),
            K=1,
            return_nn=True,
        ).idx
        query_indx = nn_vertex_idxs.squeeze(0, -1).detach().cpu()

        skinning_weight = self.smplx_layer.lbs_weights.float()

        """ PCA regression function w.r.t vertices offset
        """
        pose_dirs = self.smplx_layer.posedirs.permute(1, 0).reshape(
            smpl_x.vertex_num, 3 * (smpl_x.joint_num - 1) * 9
        )
        expr_dirs = self.smplx_layer.expr_dirs.view(
            smpl_x.vertex_num, 3 * smpl_x.expr_param_dim
        )
        shape_dirs = self.smplx_layer.shapedirs.view(
            smpl_x.vertex_num, 3 * smpl_x.shape_param_dim
        )

        (
            is_rhand,
            is_lhand,
            is_face,
            is_face_expr,
            is_lower_body,
            is_upper_body,
            is_constrain_body,
        ) = (
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
            torch.zeros((smpl_x.vertex_num, 1)).float(),
        )
        (
            is_rhand[smpl_x.rhand_vertex_idx],
            is_lhand[smpl_x.lhand_vertex_idx],
            is_face[smpl_x.face_vertex_idx],
            is_face_expr[smpl_x.expr_vertex_idx],
            is_lower_body[smpl_x.lower_body_vertex_idx],
            is_upper_body[smpl_x.upper_body_vertex_idx],
            is_constrain_body[smpl_x.constrain_body_vertex_idx],
        ) = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        is_cavity = torch.FloatTensor(smpl_x.is_cavity)[:, None]

        skinning_weight = _query(skinning_weight, query_indx)
        pose_dirs = _query(pose_dirs, query_indx)
        shape_dirs = _query(shape_dirs, query_indx)
        expr_dirs = _query(expr_dirs, query_indx)
        is_rhand = _query(is_rhand, query_indx)
        is_lhand = _query(is_lhand, query_indx)
        is_face = _query(is_face, query_indx)
        is_face_expr = _query(is_face_expr, query_indx)
        is_lower_body = _query(is_lower_body, query_indx)
        is_upper_body = _query(is_upper_body, query_indx)
        is_constrain_body = _query(is_constrain_body, query_indx)
        is_cavity = _query(is_cavity, query_indx)

        vertex_num_upsampled = self.dense_pts.shape[0]

        pose_dirs = pose_dirs.reshape(
            vertex_num_upsampled * 3, (smpl_x.joint_num - 1) * 9
        ).permute(1, 0)
        expr_dirs = expr_dirs.view(vertex_num_upsampled, 3, smpl_x.expr_param_dim)
        shape_dirs = shape_dirs.view(vertex_num_upsampled, 3, smpl_x.shape_param_dim)

        (
            is_rhand,
            is_lhand,
            is_face,
            is_face_expr,
            is_lower_body,
            is_upper_body,
            is_constrain_body,
        ) = (
            is_rhand[:, 0] > 0,
            is_lhand[:, 0] > 0,
            is_face[:, 0] > 0,
            is_face_expr[:, 0] > 0,
            is_lower_body[:, 0] > 0,
            is_upper_body[:, 0] > 0,
            is_constrain_body[:, 0] > 0,
        )
        is_cavity = is_cavity[:, 0] > 0

        # self.register_buffer('pos_enc_mesh', xyz)
        self.register_buffer("skinning_weight", skinning_weight.contiguous())
        self.register_buffer("pose_dirs", pose_dirs.contiguous())
        self.register_buffer("expr_dirs", expr_dirs.contiguous())
        self.register_buffer("shape_dirs", shape_dirs.contiguous())
        self.register_buffer("is_rhand", is_rhand.contiguous())
        self.register_buffer("is_lhand", is_lhand.contiguous())
        self.register_buffer("is_face", is_face.contiguous())
        self.register_buffer("is_face_expr", is_face_expr.contiguous())
        self.register_buffer("is_lower_body", is_lower_body.contiguous())
        self.register_buffer("is_upper_body", is_upper_body.contiguous())
        self.register_buffer("is_constrain_body", is_constrain_body.contiguous())
        self.register_buffer("is_cavity", is_cavity.contiguous())

        self.vertex_num_upsampled = vertex_num_upsampled
        self.smpl_x.vertex_num_upsampled = vertex_num_upsampled  # compatible with SMPLX

        voxel_skinning_weight, voxel_bbox = self.voxel_skinning_init(voxel_size=192)
        self.register_buffer("voxel_ws", voxel_skinning_weight)
        self.register_buffer("voxel_bbox", voxel_bbox)

        # self.query_voxel_debug()

    def get_body_infos(self):

        head_id = torch.where(self.is_face == True)[0]
        body_id = torch.where(self.is_face == False)[0]

        is_lower_body = torch.where(self.is_lower_body == True)[0]
        is_upper_body = torch.where(self.is_upper_body == True)[0]
        is_rhand = torch.where(self.is_rhand == True)[0]
        is_lhand = torch.where(self.is_lhand == True)[0]

        is_hand = torch.cat([is_rhand, is_lhand])

        return dict(
            head=head_id,
            body=body_id,
            lower_body=is_lower_body,
            upper_body=is_upper_body,
            hands=is_hand,
        )

    def query_voxel_debug(self):

        skinning_weight = self.smplx_layer.lbs_weights.float()
        smplx_data = {"betas": torch.zeros(1, self.smpl_x.shape_param_dim)}
        device = skinning_weight.device

        _, mesh_neutral_pose_wo_upsample, _ = self.get_neutral_pose_human(
            jaw_zero_pose=True,
            use_id_info=True,
            shape_param=smplx_data["betas"],
            device=device,
            face_offset=smplx_data.get("face_offset", None),
            joint_offset=smplx_data.get("joint_offset", None),
        )

        template_verts = mesh_neutral_pose_wo_upsample

        query_skinning = (
            self.query_voxel_skinning_weights(template_verts).squeeze(0).detach().cpu()
        )
        skinning_weight = self.smplx_layer.lbs_weights.float()

        diff = torch.abs(query_skinning - skinning_weight)

        print(diff.sum())

    def query_voxel_skinning_weights(self, vs):
        """using voxel-based skinning method
        vs: [B n c]
        """
        voxel_bbox = self.voxel_bbox

        scale = voxel_bbox[..., 1] - voxel_bbox[..., 0]
        center = voxel_bbox.mean(dim=1)
        normalized_vs = (vs - center[None, None, :]) / scale[None, None]
        # mapping to [-1, 1] **3
        normalized_vs = normalized_vs * 2
        normalized_vs.to(self.voxel_ws)

        B, N, _ = normalized_vs.shape

        query_ws = F.grid_sample(
            self.voxel_ws.unsqueeze(0),  # 1 C D H W
            normalized_vs.reshape(1, 1, 1, -1, 3).to(self.voxel_ws),
            align_corners=True,
            padding_mode="border",
        )
        query_ws = query_ws.view(B, -1, N)
        query_ws = query_ws.permute(0, 2, 1)

        return query_ws  # [B N C]

    def get_zero_pose_human(
        self, shape_param, device, face_offset, joint_offset, return_mesh=False
    ):
        smpl_x = self.smpl_x
        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        zero_body_pose = (
            torch.zeros((batch_size, (len(smpl_x.joint_part["body"]) - 1) * 3))
            .float()
            .to(device)
        )
        zero_hand_pose = (
            torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )
        zero_expr = torch.zeros((batch_size, smpl_x.expr_param_dim)).float().to(device)

        face_offset = face_offset
        joint_offset = (
            smpl_x.get_joint_offset(joint_offset) if joint_offset is not None else None
        )
        output = self.smplx_layer(
            global_orient=zero_pose,
            body_pose=zero_body_pose,
            left_hand_pose=zero_hand_pose,
            right_hand_pose=zero_hand_pose,
            jaw_pose=zero_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )
        joint_zero_pose = output.joints[:, : smpl_x.joint_num, :]  # zero pose human

        if not return_mesh:
            return joint_zero_pose
        else:
            raise NotImplementedError

    def get_transform_mat_joint(
        self, transform_mat_neutral_pose, joint_zero_pose, smplx_param
    ):
        """_summary_
        Args:
            transform_mat_neutral_pose (_type_): [B, 55, 4, 4]
            joint_zero_pose (_type_): [B, 55, 3]
            smplx_param (_type_): dict
        Returns:
            _type_: _description_
        """

        # 1. 大 pose -> zero pose
        transform_mat_joint_1 = transform_mat_neutral_pose

        # 2. zero pose -> image pose
        root_pose = smplx_param["root_pose"]
        body_pose = smplx_param["body_pose"]
        jaw_pose = smplx_param["jaw_pose"]
        leye_pose = smplx_param["leye_pose"]
        reye_pose = smplx_param["reye_pose"]
        lhand_pose = smplx_param["lhand_pose"]
        rhand_pose = smplx_param["rhand_pose"]
        # trans = smplx_param['trans']

        # forward kinematics

        pose = torch.cat(
            (
                root_pose.unsqueeze(1),
                body_pose,
                jaw_pose.unsqueeze(1),
                leye_pose.unsqueeze(1),
                reye_pose.unsqueeze(1),
                lhand_pose,
                rhand_pose,
            ),
            dim=1,
        )  # [B, 55, 3]
        pose = axis_angle_to_matrix(pose)  # [B, 55, 3, 3]
        posed_joints, transform_mat_joint_2 = batch_rigid_transform(
            pose[:, :, :, :], joint_zero_pose[:, :, :], self.smplx_layer.parents
        )
        transform_mat_joint_2 = transform_mat_joint_2  # [B, 55, 4, 4]

        # 3. combine 1. 大 pose -> zero pose and 2. zero pose -> image pose
        if transform_mat_joint_1 is not None:
            transform_mat_joint = torch.matmul(
                transform_mat_joint_2, transform_mat_joint_1
            )  # [B, 55, 4, 4]
        else:
            transform_mat_joint = transform_mat_joint_2

        return transform_mat_joint, posed_joints

    def get_transform_mat_vertex(self, transform_mat_joint, query_points, fix_mask):
        batch_size = transform_mat_joint.shape[0]

        query_skinning = self.query_voxel_skinning_weights(query_points)
        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_skinning[fix_mask] = skinning_weight[fix_mask]

        transform_mat_vertex = torch.matmul(
            skinning_weight,
            transform_mat_joint.view(batch_size, self.smpl_x.joint_num, 16),
        ).view(batch_size, self.smpl_x.vertex_num_upsampled, 4, 4)
        return transform_mat_vertex

    def get_posed_blendshape(self, smplx_param):
        # posed_blendshape is only applied on hand and face, which parts are closed to smplx model
        root_pose = smplx_param["root_pose"]
        body_pose = smplx_param["body_pose"]
        jaw_pose = smplx_param["jaw_pose"]
        leye_pose = smplx_param["leye_pose"]
        reye_pose = smplx_param["reye_pose"]
        lhand_pose = smplx_param["lhand_pose"]
        rhand_pose = smplx_param["rhand_pose"]
        batch_size = root_pose.shape[0]

        pose = torch.cat(
            (
                body_pose,
                jaw_pose.unsqueeze(1),
                leye_pose.unsqueeze(1),
                reye_pose.unsqueeze(1),
                lhand_pose,
                rhand_pose,
            ),
            dim=1,
        )  # [B, 54, 3]
        # smplx pose-dependent vertex offset
        pose = (
            axis_angle_to_matrix(pose) - torch.eye(3)[None, None, :, :].float().cuda()
        ).view(batch_size, (self.smpl_x.joint_num - 1) * 9)
        # (B, 54 * 9) x (54*9, V)

        smplx_pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(
            batch_size, self.smpl_x.vertex_num_upsampled, 3
        )
        return smplx_pose_offset

    def lbs(self, xyz, transform_mat_vertex, trans):
        batch_size = xyz.shape[0]
        xyz = torch.cat(
            (xyz, torch.ones_like(xyz[:, :, :1])), dim=-1
        )  # 大 pose. xyz1 [B, N, 4]
        xyz = torch.matmul(transform_mat_vertex, xyz[:, :, :, None]).view(
            batch_size, self.vertex_num_upsampled, 4
        )[
            :, :, :3
        ]  # [B, N, 3]
        if trans is not None:
            xyz = xyz + trans.unsqueeze(1)
        return xyz

    def lr_idx_to_hr_idx(self, idx):
        # follow 'subdivide_homogeneous' function of https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes
        # the low-res part takes first N_lr vertices out of N_hr vertices
        return idx

    def transform_to_posed_verts_from_neutral_pose(
        self, mean_3d, smplx_data, mesh_neutral_pose, transform_mat_neutral_pose, device
    ):
        """
        Transform the mean 3D vertices to posed vertices from the neutral pose.

            mean_3d (torch.Tensor): Mean 3D vertices with shape [B*Nv, N, 3] + offset.
            smplx_data (dict): SMPL-X data containing body_pose with shape [B*Nv, 21, 3] and betas with shape [B, 100].
            mesh_neutral_pose (torch.Tensor): Mesh vertices in the neutral pose with shape [B*Nv, N, 3].
            transform_mat_neutral_pose (torch.Tensor): Transformation matrix of the neutral pose with shape [B*Nv, 4, 4].
            device (torch.device): Device to perform the computation.

        Returns:
           torch.Tensor: Posed vertices with shape [B*Nv, N, 3] + offset.
        """

        batch_size = mean_3d.shape[0]
        shape_param = smplx_data["betas"]
        face_offset = smplx_data.get("face_offset", None)
        joint_offset = smplx_data.get("joint_offset", None)

        if shape_param.shape[0] != batch_size:
            num_views = batch_size // shape_param.shape[0]
            # print(shape_param.shape, batch_size)
            shape_param = (
                shape_param.unsqueeze(1)
                .repeat(1, num_views, 1)
                .view(-1, shape_param.shape[1])
            )
            if face_offset is not None:
                face_offset = (
                    face_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *face_offset.shape[1:])
                )
            if joint_offset is not None:
                joint_offset = (
                    joint_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *joint_offset.shape[1:])
                )

        # smplx facial expression offset

        try:
            smplx_expr_offset = (
                smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.expr_dirs
            ).sum(
                -1
            )  # [B, 1, 1, 50] x [N_V, 3, 50] -> [B, N_v, 3]
        except:
            print("no use flame params")
            smplx_expr_offset = 0.0

        mean_3d = mean_3d + smplx_expr_offset  # 大 pose

        # get nearest vertex

        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        mask = (
            ((self.is_rhand + self.is_lhand + self.is_face) > 0)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        # compute vertices-LBS function
        transform_mat_null_vertex = self.get_transform_mat_vertex(
            transform_mat_neutral_pose, mean_3d, mask
        )

        null_mean_3d = self.lbs(
            mean_3d, transform_mat_null_vertex, torch.zeros_like(smplx_data["trans"])
        )  # posed with smplx_param

        # blend_shape offset
        blend_shape_offset = blend_shapes(shape_param, self.shape_dirs)
        null_mean3d_blendshape = null_mean_3d + blend_shape_offset

        # get transformation matrix of the nearest vertex and perform lbs
        joint_null_pose = self.get_zero_pose_human(
            shape_param=shape_param,  # target shape
            device=device,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        # NOTE that the question "joint_zero_pose" is different with (transform_mat_neutral_pose)'s joints.
        transform_mat_joint, j3d = self.get_transform_mat_joint(
            None, joint_null_pose, smplx_data
        )

        # compute vertices-LBS function
        transform_mat_vertex = self.get_transform_mat_vertex(
            transform_mat_joint, mean_3d, mask
        )

        posed_mean_3d = self.lbs(
            null_mean3d_blendshape, transform_mat_vertex, smplx_data["trans"]
        )  # posed with smplx_param

        # as we do not use transform port [...,:,3],so we simply compute chain matrix
        neutral_to_posed_vertex = torch.matmul(
            transform_mat_vertex, transform_mat_null_vertex
        )  # [B, N, 4, 4]

        return posed_mean_3d, neutral_to_posed_vertex

    def get_query_points(self, smplx_data, device):
        """transform_mat_neutral_pose is function to warp pre-defined posed to zero-pose"""

        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
            self.get_neutral_pose_human(
                jaw_zero_pose=True,
                use_id_info=False,  # we blendshape at zero-pose
                shape_param=smplx_data["betas"],
                device=device,
                face_offset=smplx_data.get("face_offset", None),
                joint_offset=smplx_data.get("joint_offset", None),
            )
        )

        return (
            mesh_neutral_pose,
            mesh_neutral_pose_wo_upsample,
            transform_mat_neutral_pose,
        )

    def transform_to_posed_verts(self, smplx_data, device):
        """_summary_
        Args:
            smplx_data (_type_): e.g., body_pose:[B*Nv, 21, 3], betas:[B*Nv, 100]
        """

        # neutral posed verts
        mesh_neutral_pose, _, transform_mat_neutral_pose = self.get_query_points(
            smplx_data, device
        )

        # print(mesh_neutral_pose.shape, transform_mat_neutral_pose.shape, mesh_neutral_pose.shape, smplx_data["body_pose"].shape)
        mean_3d, transform_matrix = self.transform_to_posed_verts_from_neutral_pose(
            mesh_neutral_pose,
            smplx_data,
            mesh_neutral_pose,
            transform_mat_neutral_pose,
            device,
        )

        return mean_3d, transform_matrix

    def upsample_mesh_batch(
        self,
        smpl_x,
        shape_param,
        neutral_body_pose,
        jaw_pose,
        expression,
        betas,
        face_offset=None,
        joint_offset=None,
        device=None,
    ):
        """using blendshape to offset pts"""

        device = device if device is not None else avaliable_device()

        batch_size = shape_param.shape[0]
        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        zero_hand_pose = (
            torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )

        dense_pts = self.dense_pts.to(device)
        dense_pts = dense_pts.unsqueeze(0).repeat(expression.shape[0], 1, 1)

        blend_shape_offset = blend_shapes(betas, self.shape_dirs)

        dense_pts = dense_pts + blend_shape_offset

        joint_zero_pose = self.get_zero_pose_human(
            shape_param=shape_param,
            device=device,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        neutral_pose = torch.cat(
            (
                zero_pose,
                neutral_body_pose,
                jaw_pose,
                zero_pose,
                zero_pose,
                zero_hand_pose,
                zero_hand_pose,
            ),
            dim=1,
        )  # [B, 55, 3]

        neutral_pose = axis_angle_to_matrix(
            neutral_pose.view(-1, 55, 3)
        )  # [B, 55, 3, 3]
        posed_joints, transform_mat_joint = batch_rigid_transform(
            neutral_pose[:, :, :, :], joint_zero_pose[:, :, :], self.smplx_layer.parents
        )

        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # B 55 4,4, B N 55 -> B N 4 4
        transform_mat_vertex = torch.einsum(
            "blij,bnl->bnij", transform_mat_joint, skinning_weight
        )
        mesh_neutral_pose_upsampled = self.lbs(dense_pts, transform_mat_vertex, None)

        return mesh_neutral_pose_upsampled

    def transform_to_neutral_pose(
        self, mean_3d, smplx_data, mesh_neutral_pose, transform_mat_neutral_pose, device
    ):
        """
        Transform the mean 3D vertices to posed vertices from the neutral pose.

            mean_3d (torch.Tensor): Mean 3D vertices with shape [B*Nv, N, 3] + offset.
            smplx_data (dict): SMPL-X data containing body_pose with shape [B*Nv, 21, 3] and betas with shape [B, 100].
            mesh_neutral_pose (torch.Tensor): Mesh vertices in the neutral pose with shape [B*Nv, N, 3].
            transform_mat_neutral_pose (torch.Tensor): Transformation matrix of the neutral pose with shape [B*Nv, 4, 4].
            device (torch.device): Device to perform the computation.

        Returns:
           torch.Tensor: Posed vertices with shape [B*Nv, N, 3] + offset.
        """

        batch_size = mean_3d.shape[0]
        shape_param = smplx_data["betas"]
        face_offset = smplx_data.get("face_offset", None)
        joint_offset = smplx_data.get("joint_offset", None)
        if shape_param.shape[0] != batch_size:
            num_views = batch_size // shape_param.shape[0]
            # print(shape_param.shape, batch_size)
            shape_param = (
                shape_param.unsqueeze(1)
                .repeat(1, num_views, 1)
                .view(-1, shape_param.shape[1])
            )
            if face_offset is not None:
                face_offset = (
                    face_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *face_offset.shape[1:])
                )
            if joint_offset is not None:
                joint_offset = (
                    joint_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *joint_offset.shape[1:])
                )

        # smplx facial expression offset
        smplx_expr_offset = (
            smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.expr_dirs
        ).sum(
            -1
        )  # [B, 1, 1, 50] x [N_V, 3, 50] -> [B, N_v, 3]
        mean_3d = mean_3d + smplx_expr_offset  # 大 pose

    def get_neutral_pose_human(
        self, jaw_zero_pose, use_id_info, shape_param, device, face_offset, joint_offset
    ):

        smpl_x = self.smpl_x
        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        neutral_body_pose = (
            smpl_x.neutral_body_pose.view(1, -1).repeat(batch_size, 1).to(device)
        )  # 大 pose
        zero_hand_pose = (
            torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )
        zero_expr = torch.zeros((batch_size, smpl_x.expr_param_dim)).float().to(device)

        if jaw_zero_pose:
            jaw_pose = torch.zeros((batch_size, 3)).float().to(device)
        else:
            jaw_pose = (
                smpl_x.neutral_jaw_pose.view(1, 3).repeat(batch_size, 1).to(device)
            )  # open mouth

        if use_id_info:
            shape_param = shape_param
            # face_offset = smpl_x.face_offset[None,:,:].float().to(device)
            # joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
            face_offset = face_offset
            joint_offset = (
                smpl_x.get_joint_offset(joint_offset)
                if joint_offset is not None
                else None
            )

        else:
            shape_param = (
                torch.zeros((batch_size, smpl_x.shape_param_dim)).float().to(device)
            )
            face_offset = None
            joint_offset = None

        output = self.smplx_layer(
            global_orient=zero_pose,
            body_pose=neutral_body_pose,
            left_hand_pose=zero_hand_pose,
            right_hand_pose=zero_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        # using dense sample strategy, and warp to neutral pose
        mesh_neutral_pose_upsampled = self.upsample_mesh_batch(
            smpl_x,
            shape_param=shape_param,
            neutral_body_pose=neutral_body_pose,
            jaw_pose=jaw_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
            device=device,
        )

        mesh_neutral_pose = output.vertices
        joint_neutral_pose = output.joints[
            :, : smpl_x.joint_num, :
        ]  # 大 pose human  [B, 55, 3]

        # compute transformation matrix for making 大 pose to zero pose
        neutral_body_pose = neutral_body_pose.view(
            batch_size, len(smpl_x.joint_part["body"]) - 1, 3
        )
        zero_hand_pose = zero_hand_pose.view(
            batch_size, len(smpl_x.joint_part["lhand"]), 3
        )

        neutral_body_pose_inv = matrix_to_axis_angle(
            torch.inverse(axis_angle_to_matrix(neutral_body_pose))
        )
        jaw_pose_inv = matrix_to_axis_angle(
            torch.inverse(axis_angle_to_matrix(jaw_pose))
        )

        zero_pose = zero_pose.unsqueeze(1)
        jaw_pose_inv = jaw_pose_inv.unsqueeze(1)

        pose = torch.cat(
            (
                zero_pose,
                neutral_body_pose_inv,
                jaw_pose_inv,
                zero_pose,
                zero_pose,
                zero_hand_pose,
                zero_hand_pose,
            ),
            dim=1,
        )

        pose = axis_angle_to_matrix(pose)  # [B, 55, 3, 3]

        _, transform_mat_neutral_pose = batch_rigid_transform(
            pose[:, :, :, :], joint_neutral_pose[:, :, :], self.smplx_layer.parents
        )  # [B, 55, 4, 4]

        return (
            mesh_neutral_pose_upsampled,
            mesh_neutral_pose,
            transform_mat_neutral_pose,
        )


def read_smplx_param(smplx_data_root, shape_param_file, batch_size=1, device="cuda"):
    import json
    from glob import glob

    import cv2

    data_root_path = osp.dirname(osp.dirname(smplx_data_root))

    # load smplx parameters
    smplx_param_path_list = sorted(glob(osp.join(smplx_data_root, "*.json")))
    print(smplx_param_path_list[:3])

    smplx_params_all_frames = {}
    for smplx_param_path in smplx_param_path_list:
        frame_idx = int(smplx_param_path.split("/")[-1][:-5])
        with open(smplx_param_path) as f:
            smplx_params_all_frames[frame_idx] = {
                k: torch.FloatTensor(v) for k, v in json.load(f).items()
            }

    with open(shape_param_file) as f:
        shape_param = torch.FloatTensor(json.load(f))

    smplx_params = {}
    smplx_params["betas"] = shape_param.unsqueeze(0).repeat(batch_size, 1)

    select_frame_idx = [200, 400, 600]
    smplx_params_tmp = defaultdict(list)
    cam_param_list = []
    ori_image_list = []
    for b_idx in range(batch_size):
        frame_idx = select_frame_idx[b_idx]

        for k, v in smplx_params_all_frames[frame_idx].items():
            smplx_params_tmp[k].append(v)

        with open(
            osp.join(data_root_path, "cam_params", str(frame_idx) + ".json")
        ) as f:
            cam_param = {
                k: torch.FloatTensor(v).cuda() for k, v in json.load(f).items()
            }
            cam_param_list.append(cam_param)

        img = cv2.imread(osp.join(data_root_path, "frames", str(frame_idx) + ".png"))
        ori_image_list.append(img)

    for k, v in smplx_params_tmp.items():
        smplx_params[k] = torch.stack(smplx_params_tmp[k])

    root_path = osp.dirname(smplx_data_root)
    with open(osp.join(root_path, "face_offset.json")) as f:
        face_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, "joint_offset.json")) as f:
        joint_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, "locator_offset.json")) as f:
        locator_offset = torch.FloatTensor(json.load(f))

    smplx_params["locator_offset"] = locator_offset.unsqueeze(0).repeat(
        batch_size, 1, 1
    )
    smplx_params["joint_offset"] = joint_offset.unsqueeze(0).repeat(batch_size, 1, 1)
    smplx_params["face_offset"] = face_offset.unsqueeze(0).repeat(batch_size, 1, 1)

    for k, v in smplx_params.items():
        print(k, v.shape)
        smplx_params[k] = v.to(device)

    return smplx_params, cam_param_list, ori_image_list


def test():
    import cv2

    human_model_path = "./pretrained_models/human_model_files"
    gender = "male"
    # gender = "neutral"

    smplx_model = SMPLXMesh_Model(human_model_path, gender, subdivide_num=2)
    smplx_model.to("cuda")

    smplx_data_root = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/gyeongsik/smplx_optimized/smplx_params_smoothed"
    shape_param_file = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/gyeongsik/smplx_optimized/shape_param.json"
    smplx_data, cam_param_list, ori_image_list = read_smplx_param(
        smplx_data_root=smplx_data_root, shape_param_file=shape_param_file, batch_size=2
    )
    posed_verts = smplx_model.transform_to_posed_verts(
        smplx_data=smplx_data, device="cuda"
    )

    smplx_face = smplx_model.smpl_x.face_upsampled
    trimesh.Trimesh(
        vertices=posed_verts[0].detach().cpu().numpy(), faces=smplx_face
    ).export("./posed_obj1.obj")
    trimesh.Trimesh(
        vertices=posed_verts[1].detach().cpu().numpy(), faces=smplx_face
    ).export("./posed_obj2.obj")

    neutral_posed_verts, _, _ = smplx_model.get_query_points(
        smplx_data=smplx_data, device="cuda"
    )
    smplx_face = smplx_model.smpl_x.face
    trimesh.Trimesh(
        vertices=neutral_posed_verts[0].detach().cpu().numpy(), faces=smplx_face
    ).export("./neutral_posed_obj1.obj")
    trimesh.Trimesh(
        vertices=neutral_posed_verts[1].detach().cpu().numpy(), faces=smplx_face
    ).export("./neutral_posed_obj2.obj")

    for idx, (cam_param, img) in enumerate(zip(cam_param_list, ori_image_list)):
        render_shape = img.shape[:2]
        mesh_render, is_bkg = render_mesh(
            posed_verts[idx],
            smplx_face,
            cam_param,
            np.ones((render_shape[0], render_shape[1], 3), dtype=np.float32) * 255,
            return_bg_mask=True,
        )
        mesh_render = mesh_render.astype(np.uint8)
        cv2.imwrite(
            f"./debug_render_{idx}.jpg",
            np.clip(
                (0.9 * mesh_render + 0.1 * img) * (1 - is_bkg) + is_bkg * img, 0, 255
            ).astype(np.uint8),
        )
        # cv2.imwrite(f"./debug_render_{idx}_img.jpg", np.clip(img, 0, 255).astype(np.uint8))
        # cv2.imwrite(f"./debug_render_{idx}_mesh.jpg", np.clip(mesh_render, 0, 255).astype(np.uint8))


def read_smplx_param_humman(
    imgs_root, smplx_params_root, img_size=896, batch_size=1, device="cuda"
):
    import json
    import os
    from glob import glob

    import cv2
    from PIL import Image, ImageOps

    # Input images
    suffixes = (".jpg", ".jpeg", ".png", ".webp")
    img_path_list = [
        os.path.join(imgs_root, file)
        for file in os.listdir(imgs_root)
        if file.endswith(suffixes) and file[0] != "."
    ]

    ori_image_list = []
    smplx_params_tmp = defaultdict(list)

    for img_path in img_path_list:
        smplx_path = os.path.join(
            smplx_params_root, os.path.splitext(os.path.basename(img_path))[0] + ".json"
        )

        # Open and reshape
        img_pil = Image.open(img_path).convert("RGB")
        img_pil = ImageOps.contain(
            img_pil, (img_size, img_size)
        )  # keep the same aspect ratio
        # ori_w, ori_h = img_pil.size
        # img_pil_pad = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side
        # offset_w, offset_h = (img_size - ori_w) // 2, (img_size - ori_h) // 2

        # img = np.array(img_pil_pad)[:, :, (2, 1, 0)]
        img = np.array(img_pil)[:, :, (2, 1, 0)]
        ori_image_list.append(img)

        with open(smplx_path) as f:
            smplx_param = {k: torch.FloatTensor(v) for k, v in json.load(f).items()}

        for k, v in smplx_param.items():
            smplx_params_tmp[k].append(v)

    smplx_params = {}
    for k, v in smplx_params_tmp.items():
        smplx_params[k] = torch.stack(smplx_params_tmp[k])

    for k, v in smplx_params.items():
        print(k, v.shape)
        smplx_params[k] = v.to(device)

    cam_param_list = []
    for i in range(smplx_params["focal"].shape[0]):
        princpt = smplx_params["princpt"][i]
        cam_param = {"focal": smplx_params["focal"][i], "princpt": princpt}
        cam_param_list.append(cam_param)
    return smplx_params, cam_param_list, ori_image_list


def generate_smplx_point():

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
                smplx_params[k] = data[k].unsqueeze(0).cuda()
        return smplx_params

    def sample_one(data):
        smplx_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "trans",
        ]
        for k, v in data.items():
            if k in smplx_keys:
                # print(k, v.shape)
                data[k] = data[k][:, 0]
        return data

    human_model_path = "./pretrained_models/human_model_files"
    gender = "neutral"
    subdivide_num = 1
    smplx_model = SMPLXVoxelMeshModel(
        human_model_path,
        gender,
        shape_param_dim=10,
        expr_param_dim=100,
        subdivide_num=subdivide_num,
        dense_sample_points=40000,
        cano_pose_type=1,
    )
    smplx_model.to("cuda")

    # save_file = f"pretrained_models/human_model_files/smplx_points/smplx_subdivide{subdivide_num}.npy"
    save_file = f"debug/smplx_points/smplx_subdivide{subdivide_num}.npy"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    smplx_data = {}
    smplx_data["betas"] = torch.zeros((1, 10)).to(device="cuda")
    mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
        smplx_model.get_query_points(smplx_data=smplx_data, device="cuda")
    )

    debug_pose = torch.load("./debug/pose_example.pth")
    debug_pose["expr"] = torch.FloatTensor([0.0] * 100)

    smplx_data = get_smplx_params(debug_pose)
    smplx_data = sample_one(smplx_data)
    smplx_data["betas"] = torch.ones_like(smplx_data["betas"])

    warp_posed, _ = smplx_model.transform_to_posed_verts_from_neutral_pose(
        mesh_neutral_pose,
        smplx_data,
        mesh_neutral_pose,
        transform_mat_neutral_pose,
        "cuda",
    )

    # save_ply("warp_posed.ply", warp_posed[0].detach().cpu())
    save_ply(
        "body_constrain_posed.ply",
        warp_posed[0, smplx_model.is_constrain_body].detach().cpu(),
    )


if __name__ == "__main__":
    generate_smplx_point()
