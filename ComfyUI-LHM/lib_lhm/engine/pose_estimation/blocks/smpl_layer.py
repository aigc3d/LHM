# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license
# import sys
# sys.path.append("./pose_utils")
import os
import sys
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir+"/../")
import pose_utils
import roma
import smplx
import torch
from pose_utils import inverse_perspective_projection, perspective_projection
from pose_utils.rot6d import rotation_6d_to_matrix
from smplx.lbs import vertices2joints
from torch import nn


class SMPL_Layer(nn.Module):
    """
    Extension of the SMPL Layer with information about the camera for (inverse) projection the camera plane.
    """

    def __init__(
        self,
        smpl_dir,
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center=None,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Args
        assert type == "smplx"
        self.type = type
        self.kid = kid
        self.num_betas = num_betas
        self.bm_x = smplx.create(
            smpl_dir,
            "smplx",
            gender=gender,
            use_pca=False,
            flat_hand_mean=True,
            num_betas=num_betas,
        )

        # Primary keypoint - root
        self.joint_names = eval(f"pose_utils.get_{self.type}_joint_names")()
        self.person_center = person_center
        self.person_center_idx = None
        if self.person_center is not None:
            self.person_center_idx = self.joint_names.index(self.person_center)

    def forward(
        self,
        pose,
        shape,
        loc,
        dist,
        transl,
        K,
        expression=None,  # facial expression
        rot6d=False,
        j_regressor=None,
    ):
        """
        Args:
            - pose: pose of the person in axis-angle - torch.Tensor [bs,24,3]
            - shape: torch.Tensor [bs,10]
            - loc: 2D location of the pelvis in pixel space - torch.Tensor [bs,2]
            - dist: distance of the pelvis from the camera in m - torch.Tensor [bs,1]
        Return:
            - dict containing a bunch of useful information about each person
        """

        if loc is not None and dist is not None:
            assert pose.shape[0] == shape.shape[0] == loc.shape[0] == dist.shape[0]
        POSE_TYPE_LENGTH = 6 if rot6d else 3
        if self.type == "smpl":
            assert len(pose.shape) == 3 and list(pose.shape[1:]) == [
                24,
                POSE_TYPE_LENGTH,
            ]
        elif self.type == "smplx":
            assert len(pose.shape) == 3 and list(pose.shape[1:]) == [
                53,
                POSE_TYPE_LENGTH,
            ]  # taking root_orient, body_pose, lhand, rhan and jaw for the moment
        else:
            raise NameError
        assert len(shape.shape) == 2 and (
            list(shape.shape[1:]) == [self.num_betas]
            or list(shape.shape[1:]) == [self.num_betas + 1]
        )
        if loc is not None and dist is not None:
            assert len(loc.shape) == 2 and list(loc.shape[1:]) == [2]
            assert len(dist.shape) == 2 and list(dist.shape[1:]) == [1]

        bs = pose.shape[0]

        out = {}

        # No humans
        if bs == 0:
            return {}

        # Low dimensional parameters
        kwargs_pose = {
            "betas": shape,
        }
        kwargs_pose["global_orient"] = self.bm_x.global_orient.repeat(bs, 1)
        kwargs_pose["body_pose"] = pose[:, 1:22].flatten(1)
        kwargs_pose["left_hand_pose"] = pose[:, 22:37].flatten(1)
        kwargs_pose["right_hand_pose"] = pose[:, 37:52].flatten(1)
        kwargs_pose["jaw_pose"] = pose[:, 52:53].flatten(1)

        if expression is not None:
            kwargs_pose["expression"] = expression.flatten(1)  # [bs,10]
        else:
            kwargs_pose["expression"] = self.bm_x.expression.repeat(bs, 1)

        # default - to be generalized
        kwargs_pose["leye_pose"] = self.bm_x.leye_pose.repeat(bs, 1)
        kwargs_pose["reye_pose"] = self.bm_x.reye_pose.repeat(bs, 1)
        # kwargs_pose['pose2rot'] = not rot6d
        # Forward using the parametric 3d model SMPL-X layer
        output = self.bm_x(pose2rot=not rot6d, **kwargs_pose)
        verts = output.vertices
        j3d = output.joints  # 45 joints

        if rot6d:
            R = rotation_6d_to_matrix(pose[:, 0])
        else:
            R = roma.rotvec_to_rotmat(pose[:, 0])

        # Apply global orientation on 3D points
        pelvis = j3d[:, [0]]
        j3d = (R.unsqueeze(1) @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)

        # Apply global orientation on 3D points - bis
        verts = (R.unsqueeze(1) @ (verts - pelvis).unsqueeze(-1)).squeeze(-1)

        # Location of the person in 3D
        if transl is None:
            if K.dtype == torch.float16:
                # because of torch.inverse - not working with float16 at the moment
                transl = inverse_perspective_projection(
                    loc.unsqueeze(1).float(), K.float(), dist.unsqueeze(1).float()
                )[:, 0]
                transl = transl.half()
            else:
                transl = inverse_perspective_projection(
                    loc.unsqueeze(1), K, dist.unsqueeze(1)
                )[:, 0]

        # Updating transl if we choose a certain person center
        transl_up = transl.clone()

        # Definition of the translation depend on the args: 1) vanilla SMPL - 2) computed from a given joint
        if self.person_center_idx is None:
            # Add pelvis to transl - standard way for SMPLX layer
            transl_up = transl_up + pelvis[:, 0]
        else:
            # Center around the joint because teh translation is computed from this joint
            person_center = j3d[:, [self.person_center_idx]]
            verts = verts - person_center
            j3d = j3d - person_center

        # Moving into the camera coordinate system
        j3d_cam = j3d + transl_up.unsqueeze(1)
        verts_cam = verts + transl_up.unsqueeze(1)

        # Projection in camera plane
        if j_regressor is not None:
            # for smplify
            j3d_cam = vertices2joints(j_regressor, verts_cam)
        j2d = perspective_projection(j3d_cam, K)
        v2d = perspective_projection(verts_cam, K)

        out.update(
            {
                "v3d": verts_cam,  # in 3d camera space
                "j3d": j3d_cam,  # in 3d camera space
                "j2d": j2d,
                "v2d": v2d,
                "transl": transl,  # translation of the primary keypoint
                "transl_pelvis": transl.unsqueeze(1)
                - person_center
                - pelvis,  # root=pelvis
                "j3d_world": output.joints,
            }
        )

        return out

    def forward_local(self, pose, shape):
        N, J, L = pose.shape
        if N < 1:
            return None
        kwargs_pose = {
            "betas": shape,
        }
        if J == 53:
            kwargs_pose["global_orient"] = self.bm_x.global_orient.repeat(N, 1)
            kwargs_pose["body_pose"] = pose[:, 1:22].flatten(1)
            kwargs_pose["left_hand_pose"] = pose[:, 22:37].flatten(1)
            kwargs_pose["right_hand_pose"] = pose[:, 37:52].flatten(1)
            kwargs_pose["jaw_pose"] = pose[:, 52:53].flatten(1)
        elif J == 55:
            kwargs_pose["global_orient"] = self.bm_x.global_orient.repeat(N, 1)
            kwargs_pose["body_pose"] = pose[:, 1:22].flatten(1)
            kwargs_pose["left_hand_pose"] = pose[:, 25:40].flatten(1)
            kwargs_pose["right_hand_pose"] = pose[:, 40:55].flatten(1)
            kwargs_pose["jaw_pose"] = pose[:, 22:23].flatten(1)
        else:
            raise ValueError(f"pose dim error, should be 53 or 55, but got {J}")
        kwargs_pose["expression"] = self.bm_x.expression.repeat(N, 1)

        # default - to be generalized
        kwargs_pose["leye_pose"] = self.bm_x.leye_pose.repeat(N, 1)
        kwargs_pose["reye_pose"] = self.bm_x.reye_pose.repeat(N, 1)

        output = self.bm_x(**kwargs_pose)
        return output

    def convert_standard_pose(self, poses):
        # pose: N, J, 3
        n = poses.shape[0]
        poses = torch.cat(
            [
                poses[:, :22],
                poses[:, 52:53],
                self.bm_x.leye_pose.repeat(n, 1, 1),
                self.bm_x.reye_pose.repeat(n, 1, 1),
                poses[:, 22:52],
            ],
            dim=1,
        )
        return poses
