# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Xiaodong Gu, Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-01-08 21:42:24, Version 0.0, SMPLX + FLAME2019
# @Function      : SMPLX-related functions

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
from pytorch3d.ops import SubdivideMeshes, knn_points
from pytorch3d.structures import Meshes
from smplx.lbs import batch_rigid_transform
from torch.nn import functional as F

from LHM.models.rendering.smplx import smplx
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


class SMPLX(object):
    def __init__(
        self,
        human_model_path,
        shape_param_dim=100,
        expr_param_dim=50,
        subdivide_num=2,
        cano_pose_type=0,
    ):
        """SMPLX using pytorch3d subdivsion"""
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
            "L_Shoulder",
            "R_Shoulder",
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
        }

        self.lower_body_vertex_idx = self.get_lower_body()

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
        self.subdivide_num = subdivide_num
        self.subdivider_list = self.get_subdivider(subdivide_num)
        self.subdivider_cpu_list = self.get_subdivider_cpu(subdivide_num)
        self.face_upsampled = (
            self.subdivider_list[-1]._subdivided_faces.cpu().numpy()
            if self.subdivide_num > 0
            else self.face
        )
        print("face_upsampled:", self.face_upsampled.shape)
        self.vertex_num_upsampled = int(np.max(self.face_upsampled) + 1)

    def get_lower_body(self):
        """using skinning to find lower body vertices."""
        lower_body_skinning_index = set(self.joint_part["lower_body"])
        skinning_weight = self.layer["neutral"].lbs_weights.float()
        skinning_part = skinning_weight.argmax(1)
        skinning_part = skinning_part.cpu().numpy()
        lower_body_vertice_idx = []
        for v_id, v_s in enumerate(skinning_part):
            if v_s in lower_body_skinning_index:
                lower_body_vertice_idx.append(v_id)

        lower_body_vertice_idx = np.asarray(lower_body_vertice_idx)

        # debug
        # template_v = self.layer["neutral"].v_template
        # lower_body_v = template_v[lower_body_vertice_idx]
        # save_ply("lower_body_v.ply", lower_body_v)
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

    def get_subdivider_cpu(self, subdivide_num):
        vert = self.layer["neutral"].v_template.float()
        face = torch.LongTensor(self.face)
        mesh = Meshes(vert[None, :, :], face[None, :, :])

        if subdivide_num > 0:
            subdivider_list = [SubdivideMeshes(mesh)]
            for i in range(subdivide_num - 1):
                mesh = subdivider_list[-1](mesh)
                subdivider_list.append(SubdivideMeshes(mesh))
        else:
            subdivider_list = [mesh]
        return subdivider_list

    def upsample_mesh_cpu(self, vert, feat_list=None):
        face = torch.LongTensor(self.face)
        mesh = Meshes(vert[None, :, :], face[None, :, :])
        if self.subdivide_num > 0:
            if feat_list is None:
                for subdivider in self.subdivider_cpu_list:
                    mesh = subdivider(mesh)
                vert = mesh.verts_list()[0]
                return vert
            else:
                feat_dims = [x.shape[1] for x in feat_list]
                feats = torch.cat(feat_list, 1)
                for subdivider in self.subdivider_cpu_list:
                    mesh, feats = subdivider(mesh, feats)
                vert = mesh.verts_list()[0]
                feats = feats[0]
                feat_list = torch.split(feats, feat_dims, dim=1)
                return vert, *feat_list
        else:
            if feat_list is None:
                # for subdivider in self.subdivider_cpu_list:
                #     mesh = subdivider(mesh)
                # vert = mesh.verts_list()[0]
                return vert
            else:
                return vert, *feat_list

    def upsample_mesh(self, vert, feat_list=None, device="cuda"):
        face = torch.LongTensor(self.face).to(device)
        mesh = Meshes(vert[None, :, :], face[None, :, :])
        if self.subdivide_num > 0:
            if feat_list is None:
                for subdivider in self.subdivider_list:
                    mesh = subdivider(mesh)
                vert = mesh.verts_list()[0]
                return vert
            else:
                feat_dims = [x.shape[1] for x in feat_list]
                feats = torch.cat(feat_list, 1)
                for subdivider in self.subdivider_list:
                    mesh, feats = subdivider(mesh, feats)
                vert = mesh.verts_list()[0]
                feats = feats[0]
                feat_list = torch.split(feats, feat_dims, dim=1)
                return vert, *feat_list
        else:
            if feat_list is None:
                # for subdivider in self.subdivider_list:
                #     mesh = subdivider(mesh)
                # vert = mesh.verts_list()[0]
                return vert
            else:
                # feat_dims = [x.shape[1] for x in feat_list]
                # feats = torch.cat(feat_list,1)
                # for subdivider in self.subdivider_list:
                #     mesh, feats = subdivider(mesh, feats)
                # vert = mesh.verts_list()[0]
                # feats = feats[0]
                # feat_list = torch.split(feats, feat_dims, dim=1)
                return vert, *feat_list

    def upsample_mesh_batch(self, vert, device="cuda"):
        if self.subdivide_num > 0:
            face = (
                torch.LongTensor(self.face)
                .to(device)
                .unsqueeze(0)
                .repeat(vert.shape[0], 1, 1)
            )
            mesh = Meshes(vert, face)
            for subdivider in self.subdivider_list:
                mesh = subdivider(mesh)
            vert = torch.stack(mesh.verts_list(), dim=0)
        else:
            pass
        return vert

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


class SMPLXModel(nn.Module):
    def __init__(
        self,
        human_model_path,
        gender,
        subdivide_num,
        expr_param_dim=50,
        shape_param_dim=100,
        cano_pose_type=0,
        apply_pose_blendshape=False,
    ) -> None:
        super().__init__()

        # self.smpl_x = SMPLX(
        #     human_model_path=human_model_path,
        #     shape_param_dim=shape_param_dim,
        #     expr_param_dim=expr_param_dim,
        #     subdivide_num=subdivide_num,
        #     cano_pose_type=cano_pose_type,
        # )

        self.smpl_x = SMPLX(
            human_model_path=human_model_path,
            shape_param_dim=shape_param_dim,
            expr_param_dim=expr_param_dim,
            subdivide_num=subdivide_num,
            cano_pose_type=cano_pose_type,
        )
        self.smplx_layer = copy.deepcopy(self.smpl_x.layer[gender])

        self.apply_pose_blendshape = apply_pose_blendshape
        # register
        self.smplx_init()

    def get_body_infos(self):

        head_id = torch.where(self.is_face == True)[0]
        body_id = torch.where(self.is_face == False)[0]
        return dict(head=head_id, body=body_id)

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

        smpl_x = self.smpl_x

        # # upsample mesh and other assets
        # xyz, _, _, _ = self.get_neutral_pose_human(jaw_zero_pose=False, use_id_info=False, device=device)

        skinning_weight = self.smplx_layer.lbs_weights.float()

        """ PCA regression function w.r.t vertices offset
        """
        pose_dirs = self.smplx_layer.posedirs.permute(1, 0).reshape(
            smpl_x.vertex_num, 3 * (smpl_x.joint_num - 1) * 9
        )
        expr_dirs = self.smplx_layer.expr_dirs.view(
            smpl_x.vertex_num, 3 * smpl_x.expr_param_dim
        )

        is_rhand, is_lhand, is_face, is_face_expr, is_lower_body = (
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
        ) = (1.0, 1.0, 1.0, 1.0, 1.0)
        is_cavity = torch.FloatTensor(smpl_x.is_cavity)[:, None]

        # obtain subvided apperance
        (
            _,
            skinning_weight,
            pose_dirs,
            expr_dirs,
            is_rhand,
            is_lhand,
            is_face,
            is_face_expr,
            is_lower_body,
            is_cavity,
        ) = smpl_x.upsample_mesh_cpu(
            torch.ones((smpl_x.vertex_num, 3)).float(),
            [
                skinning_weight,
                pose_dirs,
                expr_dirs,
                is_rhand,
                is_lhand,
                is_face,
                is_face_expr,
                is_lower_body,
                is_cavity,
            ],
        )  # upsample with dummy vertex

        pose_dirs = pose_dirs.reshape(
            smpl_x.vertex_num_upsampled * 3, (smpl_x.joint_num - 1) * 9
        ).permute(
            1, 0
        )  # (J * 9, V * 3)
        expr_dirs = expr_dirs.view(
            smpl_x.vertex_num_upsampled, 3, smpl_x.expr_param_dim
        )
        is_rhand, is_lhand, is_face, is_face_expr, is_lower_body = (
            is_rhand[:, 0] > 0,
            is_lhand[:, 0] > 0,
            is_face[:, 0] > 0,
            is_face_expr[:, 0] > 0,
            is_lower_body[:, 0] > 0,
        )
        is_cavity = is_cavity[:, 0] > 0

        # self.register_buffer('pos_enc_mesh', xyz)
        # is legs

        self.register_buffer("skinning_weight", skinning_weight.contiguous())
        self.register_buffer("pose_dirs", pose_dirs.contiguous())
        self.register_buffer("expr_dirs", expr_dirs.contiguous())
        self.register_buffer("is_rhand", is_rhand.contiguous())
        self.register_buffer("is_lhand", is_lhand.contiguous())
        self.register_buffer("is_face", is_face.contiguous())
        self.register_buffer("is_lower_body", is_lower_body.contiguous())
        self.register_buffer("is_face_expr", is_face_expr.contiguous())
        self.register_buffer("is_cavity", is_cavity.contiguous())

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

        # smplx layer is smplx model
        # ['vertices', 'joints', 'full_pose', 'global_orient', 'transl', 'v_shaped', 'betas', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'expression', 'jaw_pose']

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

        mesh_neutral_pose_upsampled = smpl_x.upsample_mesh_batch(
            output.vertices, device=device
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

        # transform_mat_neutral_pose is a function to warp neutral pose to zero pose  (neutral pose is *-pose)
        _, transform_mat_neutral_pose = batch_rigid_transform(
            pose[:, :, :, :], joint_neutral_pose[:, :, :], self.smplx_layer.parents
        )  # [B, 55, 4, 4]

        return (
            mesh_neutral_pose_upsampled,
            mesh_neutral_pose,
            transform_mat_neutral_pose,
        )

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
            mesh_zero_pose = output.vertices[0]  # zero pose human
            mesh_zero_pose_upsampled = smpl_x.upsample_mesh(
                mesh_zero_pose
            )  # zero pose human
            return mesh_zero_pose_upsampled, mesh_zero_pose, joint_zero_pose

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
        transform_mat_joint = torch.matmul(
            transform_mat_joint_2, transform_mat_joint_1
        )  # [B, 55, 4, 4]

        return transform_mat_joint, posed_joints

    def get_transform_mat_vertex(self, transform_mat_joint, nn_vertex_idxs):
        batch_size = transform_mat_joint.shape[0]
        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)
        skinning_weight = skinning_weight.view(-1, skinning_weight.shape[-1])[
            nn_vertex_idxs.view(-1)
        ].view(
            nn_vertex_idxs.shape[0], nn_vertex_idxs.shape[1], skinning_weight.shape[-1]
        )
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
            batch_size, self.smpl_x.vertex_num_upsampled, 4
        )[
            :, :, :3
        ]  # [B, N, 3]
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
            smplx_expr_offset = 0.0

        mean_3d = mean_3d + smplx_expr_offset  # 大 pose

        if self.apply_pose_blendshape:
            smplx_pose_offset = self.get_posed_blendshape(smplx_data)
            mask = (
                ((self.is_rhand + self.is_lhand + self.is_face_expr) > 0)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            mean_3d[mask] += smplx_pose_offset[mask]

        # get nearest vertex

        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        nn_vertex_idxs = knn_points(
            mean_3d[:, :, :], mesh_neutral_pose[:, :, :], K=1, return_nn=True
        ).idx[
            :, :, 0
        ]  # dimension: smpl_x.vertex_num_upsampled
        # nn_vertex_idxs = self.lr_idx_to_hr_idx(nn_vertex_idxs)
        mask = (
            ((self.is_rhand + self.is_lhand + self.is_face) > 0)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        nn_vertex_idxs[mask] = (
            torch.arange(self.smpl_x.vertex_num_upsampled)
            .to(device)
            .unsqueeze(0)
            .repeat(batch_size, 1)[mask]
        )

        # get transformation matrix of the nearest vertex and perform lbs
        joint_zero_pose = self.get_zero_pose_human(
            shape_param=shape_param,
            device=device,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        # NOTE that the question "joint_zero_pose" is different with (transform_mat_neutral_pose)'s joints.
        transform_mat_joint, j3d = self.get_transform_mat_joint(
            transform_mat_neutral_pose, joint_zero_pose, smplx_data
        )

        # compute vertices-LBS function
        transform_mat_vertex = self.get_transform_mat_vertex(
            transform_mat_joint, nn_vertex_idxs
        )

        mean_3d = self.lbs(
            mean_3d, transform_mat_vertex, smplx_data["trans"]
        )  # posed with smplx_param

        return mean_3d, transform_mat_vertex

    def get_query_points(self, smplx_data, device):
        """transform_mat_neutral_pose is function to warp pre-defined posed to zero-pose"""
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
            self.get_neutral_pose_human(
                jaw_zero_pose=True,
                use_id_info=True,
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
    # smplx_params["betas"][0] = torch.zeros_like(smplx_params["betas"][0])
    # smplx_params["betas"] = torch.zeros_like(smplx_params["betas"])

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

    # batch_size = smplx_data['root_pose'].shape[0]
    # root_pose = smplx_data['root_pose']
    # body_pose = smplx_data['body_pose']
    # jaw_pose = smplx_data['jaw_pose']
    # leye_pose = smplx_data['leye_pose']
    # reye_pose = smplx_data['reye_pose']
    # lhand_pose = smplx_data['lhand_pose'].view(batch_size, len(smplx.smpl_x.joint_part['lhand'])*3)
    # rhand_pose = smplx_data['rhand_pose'].view(batch_size, len(smplx.smpl_x.joint_part['rhand'])*3)
    # expr = smplx_data['expr'].view(batch_size, smplx.smpl_x.expr_param_dim)
    # trans = smplx_data['trans'].view(batch_size, 3)
    # shape = smplx_data["betas"]
    # face_offset = smplx_data["face_offset"]
    # joint_offset = smplx_data["joint_offset"]

    # smplx_layer = smplx.smplx_layer
    # smplx_face = smplx.smpl_x.face
    # output = smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose,
    #                            leye_pose=leye_pose, reye_pose=reye_pose,
    #                            left_hand_pose=lhand_pose, right_hand_pose=rhand_pose,
    #                            expression=expr, betas=shape,
    #                            transl=trans,
    #                            face_offset=face_offset, joint_offset=joint_offset)
    # posed_verts = [e for e in output.vertices]
    # trimesh.Trimesh(vertices=posed_verts[0].detach().cpu().numpy(), faces=smplx_face).export("./posed_obj1_from_zeropose.obj")
    # trimesh.Trimesh(vertices=posed_verts[1].detach().cpu().numpy(), faces=smplx_face).export("./posed_obj2_from_zeropose.obj")

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


def test_humman():
    import cv2

    human_model_path = "./pretrained_models/human_model_files"
    # gender = "male"
    gender = "neutral"

    smplx_model = SMPLXModel(
        human_model_path, gender, shape_param_dim=10, expr_param_dim=10, subdivide_num=2
    )
    smplx_model.to("cuda")

    # root_dir = "./train_data/humman/humman_compressed"
    # meta_path = "./train_data/humman/humman_id_list.json"
    # dataset = HuMManDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=3,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(682, 384), source_image_res=384)

    # root_dir = "./train_data/static_human_data"
    # meta_path = "./train_data/static_human_data/data_id_list.json"
    # dataset = StaticHumanDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=7,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(682, 384), source_image_res=384,
    #                 debug=False)

    #     from openlrm.datasets.singleview_human import SingleViewHumanDataset
    #     root_dir = "./train_data/single_view"
    #     meta_path = "./train_data/single_view/data_SHHQ.json"
    #     dataset = SingleViewHumanDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=0,
    #                     render_image_res_low=384, render_image_res_high=384,
    #                     render_region_size=(682, 384), source_image_res=384,
    #                     debug=False)

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
    # for idx, data in enumerate(dataset):
    #     if idx == 2:
    #         break

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

    for k, v in data.items():
        if hasattr(v, "shape"):
            print(k, v.shape)

    smplx_data = get_smplx_params(data)
    smplx_data["betas"] = (
        smplx_data["betas"].unsqueeze(0).repeat(smplx_data["body_pose"].shape[0], 1)
    )

    smplx_data_tmp = {}
    for k, v in smplx_data.items():
        smplx_data_tmp[k] = v.to("cuda")
        print(k, v.shape)
    smplx_data = smplx_data_tmp

    intrs = data["intrs"].to("cuda")
    cam_param_list = [
        {
            "focal": torch.tensor([e[0, 0], e[1, 1]]),
            "princpt": torch.tensor([e[0, 2], e[1, 2]]),
        }
        for e in intrs
    ]
    print(cam_param_list[0])
    ori_image_list = [
        (e.permute(1, 2, 0)[:, :, (2, 1, 0)].numpy() * 255).astype(np.uint8)
        for e in data["render_image"]
    ]

    posed_verts = smplx_model.transform_to_posed_verts(
        smplx_data=smplx_data, device="cuda"
    )

    os.makedirs("./debug_vis/smplx", exist_ok=True)
    smplx_face = smplx_model.smpl_x.face_upsampled
    trimesh.Trimesh(
        vertices=posed_verts[0].detach().cpu().numpy(), faces=smplx_face
    ).export("./debug_vis/smplx/posed_obj1.obj")
    if len(posed_verts) > 1:
        trimesh.Trimesh(
            vertices=posed_verts[1].detach().cpu().numpy(), faces=smplx_face
        ).export("./debug_vis/smplx/posed_obj2.obj")

    neutral_posed_verts, _, _ = smplx_model.get_query_points(
        smplx_data=smplx_data, device="cuda"
    )
    smplx_face = smplx_model.smpl_x.face
    trimesh.Trimesh(
        vertices=neutral_posed_verts[0].detach().cpu().numpy(), faces=smplx_face
    ).export("./debug_vis/smplx/neutral_posed_obj1.obj")
    if len(neutral_posed_verts) > 1:
        trimesh.Trimesh(
            vertices=neutral_posed_verts[1].detach().cpu().numpy(), faces=smplx_face
        ).export("./debug_vis/smplx/neutral_posed_obj2.obj")

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
            f"./debug_vis/smplx/debug_render_{idx}.jpg",
            np.clip(
                (0.9 * mesh_render + 0.1 * img) * (1 - is_bkg) + is_bkg * img, 0, 255
            ).astype(np.uint8),
        )
        # cv2.imwrite(f"./debug_render_{idx}_img.jpg", np.clip(img, 0, 255).astype(np.uint8))
        # cv2.imwrite(f"./debug_render_{idx}_mesh.jpg", np.clip(mesh_render, 0, 255).astype(np.uint8))
        # if idx == 1:
        #     break


def generate_smplx_point():
    human_model_path = "./pretrained_models/human_model_files"
    gender = "neutral"
    subdivide_num = 1
    smplx_model = SMPLXModel(
        human_model_path,
        gender,
        shape_param_dim=10,
        expr_param_dim=10,
        subdivide_num=subdivide_num,
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

    pdb.set_trace()

    smplx_face = smplx_model.smpl_x.face_upsampled

    # trimesh.Trimesh(
    #     vertices=mesh_neutral_pose[0].detach().cpu().numpy(), faces=smplx_face
    # ).export(
    #     f"pretrained_models/human_model_files/smplx_points/smplx_subdivide{subdivide_num}.obj"
    # )

    trimesh.Trimesh(
        vertices=mesh_neutral_pose[0].detach().cpu().numpy(), faces=smplx_face
    ).export(f"debug/smplx_points/smplx_subdivide{subdivide_num}.obj")

    np.save(save_file, mesh_neutral_pose[0].detach().cpu().numpy())

    smplx_face = smplx_model.smpl_x.face
    # save_file = f"pretrained_models/human_model_files/smplx_points/smplx.npy"
    save_file = f"debug/smplx_points/smplx.npy"

    trimesh.Trimesh(
        vertices=mesh_neutral_pose_wo_upsample[0].detach().cpu().numpy(),
        faces=smplx_face,
        process=False,
    ).export(f"debug/smplx_points/smplx.obj")
    np.save(save_file, mesh_neutral_pose_wo_upsample[0].detach().cpu().numpy())


if __name__ == "__main__":
    # test()
    # test_humman()
    generate_smplx_point()
