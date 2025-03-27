# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Peihao Li
# @Email         : liphao99@gmail.com
# @Time          : 2025-03-19 12:47:58
# @Function      : smplify-x
import torch
from pose_utils import (
    get_mapping,
    inverse_perspective_projection,
    perspective_projection,
)
from pose_utils.rot6d import (
    axis_angle_to_rotation_6d,
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
)
from tqdm import tqdm

KEYPOINT_THRESH = 0.5
ROOT_ORIENT_JITTER_THRESH = 1.0


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def compute_jitter(x):
    """
    Compute jitter for the input tensor
    """
    jitter = torch.linalg.norm(x[2:].detach() + x[:-2].detach() - 2 * x[1:-1], dim=-1)
    return jitter


class FastFirstFittingLoss(torch.nn.Module):
    def __init__(self, cam_intrinsics, j3d_idx, device):
        super().__init__()
        self.cam_intrinsics = cam_intrinsics
        self.j3d_idx = j3d_idx
        self.person_center_idx = 15  # head idx

    @torch.no_grad()
    def find_orient_jitter(self, root_orient, transl, j3d, input_keypoints, bbox):
        R = rotation_6d_to_matrix(root_orient)
        pelvis = j3d[:, [0]]
        j3d = (R @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)
        j3d = j3d - j3d[:, [self.person_center_idx]]
        j3d = j3d + transl.unsqueeze(1)
        j2d = perspective_projection(j3d, self.cam_intrinsics)

        scale = bbox[..., -1:].unsqueeze(-1)
        pred_keypoints = j2d[..., self.j3d_idx, :]
        mask = input_keypoints[..., -1:] > KEYPOINT_THRESH
        valid_mask = torch.sum(mask, dim=1) > 3
        valid_mask = valid_mask[:, 0]

        mask[~valid_mask] = False
        joints_conf = input_keypoints[..., -1:]
        joints_conf[~mask] = 0.0

        reprojection_error = (
            ((pred_keypoints - input_keypoints[..., :-1]) ** 2) * joints_conf
        ) / scale
        reprojection_error = torch.sum(reprojection_error, dim=(-2, -1))

        pose_jitter = compute_jitter(root_orient)

        mask1 = pose_jitter > 1
        mask2 = reprojection_error > 8

        mask2[2:] = mask2[2:] | mask1[:, 0]
        index = torch.where(mask2)[0]
        if len(index) < 1:
            return -1, -1
        return max(0, index.min() - 10), min(index.max() + 10, len(root_orient) - 1)

    def forward(
        self,
        root_orient,
        transl,
        j3d,
        input_keypoints,
        bbox,
        orient_smooth_weight=1,
        reprojection_weight=100.0,
        smooth_weight=30,
        sigma=10000,
    ):
        R = rotation_6d_to_matrix(root_orient)
        pelvis = j3d[:, [0]]
        j3d = (R @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)
        j3d = j3d - j3d[:, [self.person_center_idx]]
        j3d = j3d + transl.unsqueeze(1)
        j2d = perspective_projection(j3d, self.cam_intrinsics)

        scale = bbox[..., -1:].unsqueeze(-1)
        pred_keypoints = j2d[..., self.j3d_idx, :]
        mask = input_keypoints[..., -1:] > KEYPOINT_THRESH
        valid_mask = torch.sum(mask, dim=1) > 3
        valid_mask = valid_mask[:, 0]

        mask[~valid_mask] = False
        joints_conf = input_keypoints[..., -1:]
        joints_conf[~mask] = 0.0

        reprojection_error = (
            (pred_keypoints - input_keypoints[..., :-1]) ** 2 * joints_conf
        ) / scale

        reprojection_error = reprojection_error.sum() / mask.sum()

        dist_diff = compute_jitter(transl).mean()
        pose_diff = compute_jitter(root_orient).mean()
        smooth_error = dist_diff + orient_smooth_weight * pose_diff
        loss_dict = {
            "reprojection": reprojection_weight * reprojection_error,
            "smooth": smooth_weight * smooth_error,
        }

        loss = sum(loss_dict.values())

        return loss


class SMPLifyLoss(torch.nn.Module):
    def __init__(
        self,
        cam_intrinsics,
        init_pose,
        j3d_idx,
        device,
    ):

        super().__init__()

        self.cam_intrinsics = cam_intrinsics
        self.init_pose = init_pose.detach().clone()
        self.j3d_idx = j3d_idx

    def forward(
        self,
        output,
        params,
        input_keypoints,
        bbox,
        reprojection_weight=100.0,
        regularize_weight=100.0,
        consistency_weight=20.0,
        sprior_weight=0.04,
        smooth_weight=30,
        sigma=100,
    ):

        pose, shape, transl = params
        scale = bbox[..., -1:].unsqueeze(-1)

        # Loss 1. Data term
        pred_keypoints = output["j2d"][..., self.j3d_idx, :]
        joints_conf = input_keypoints[..., -1:]
        mask = input_keypoints[..., -1:] > KEYPOINT_THRESH
        joints_conf[~mask] = 0.0

        reprojection_error = gmof(pred_keypoints - input_keypoints[..., :-1], sigma)

        reprojection_error = ((reprojection_error * joints_conf) / scale).mean()

        # Loss 2. Regularization term
        regularize_error = torch.linalg.norm(pose - self.init_pose, dim=-1).mean()
        head_regularize_weight = 40
        head_regularize_error = (
            torch.linalg.norm(pose[:, 12:13] - self.init_pose[:, 12:13], dim=-1)
            + torch.linalg.norm(pose[:, 15:16] - self.init_pose[:, 15:16], dim=-1)
        ).mean()

        # Loss 3. Shape prior and consistency error
        consistency_error = shape.std(dim=0).mean()

        sprior_error = torch.linalg.norm(shape, dim=-1).mean()
        shape_error = (
            sprior_weight * sprior_error + consistency_weight * consistency_error
        )

        # Loss 4. Smooth loss
        pose_diff = compute_jitter(pose).mean()
        dist_diff = compute_jitter(transl).mean()
        smooth_error = pose_diff + dist_diff
        # Sum up losses
        loss = {
            "reprojection": reprojection_weight * reprojection_error,
            "regularize": regularize_weight * regularize_error
            + head_regularize_error * head_regularize_weight,
            "shape": shape_error,
            "smooth": smooth_weight * smooth_error,
        }

        return loss

    def create_closure(self, optimizer, smpl, params, bbox, input_keypoints):

        def closure():
            optimizer.zero_grad()
            poses = torch.cat([params[0], params[1]], dim=1)
            out = smpl(
                rotation_6d_to_axis_angle(poses),
                params[2],
                None,
                None,
                transl=params[3],
                K=self.cam_intrinsics,
            )
            loss_dict = self.forward(
                out, [poses, params[2], params[3]], input_keypoints, bbox
            )
            loss = sum(loss_dict.values())
            loss.backward()

            return loss

        return closure


class TemporalSMPLify:

    def __init__(self, smpl=None, lr=1e-2, num_iters=5, num_steps=100, device=None):

        self.smpl = smpl
        self.lr = lr
        self.num_iters = num_iters
        self.num_steps = num_steps
        self.device = device

        resutls = get_mapping("smplx", "coco_wholebody")
        full_mapping_list = resutls[-1]

        dst_idx = list(range(0, 23)) + list(range(91, 133))
        self.src_idx = []
        self.dst_idx = []
        for _dst_idx in dst_idx:
            _src_idx = full_mapping_list[_dst_idx]
            if _src_idx >= 0:
                self.src_idx.append(_src_idx)
                self.dst_idx.append(_dst_idx)

        # first fitting: optimize global_orient and translation with only 4 joints, left_shoulder ,right_shoulder, left_hip, right_hip
        first_fitting_dst_idx = [5, 6, 11, 12]
        self.first_fitting_dst_idx = []
        self.first_fitting_src_idx = []
        for _dst_idx in first_fitting_dst_idx:
            _src_idx = full_mapping_list[_dst_idx]
            if _src_idx >= 0:
                self.first_fitting_src_idx.append(_src_idx)
                self.first_fitting_dst_idx.append(_dst_idx)

    def fit(
        self,
        init_poses,
        init_betas,
        init_dist,
        init_loc,
        cam_intrinsic,
        keypoints_2d,
        bbox,
    ):

        def to_params(param):
            return param.detach().clone().requires_grad_(True)

        if not isinstance(init_poses, torch.Tensor):
            init_poses = torch.tensor(init_poses, device=self.device)
            init_betas = torch.tensor(init_betas, device=self.device)
            init_dist = torch.tensor(init_dist, device=self.device)
            init_loc = torch.tensor(init_loc, device=self.device)

        init_poses = axis_angle_to_rotation_6d(init_poses)

        init_global_orient = init_poses[..., 0:1, :]
        init_body_poses = init_poses[..., 1:, :]

        init_betas = torch.mean(init_betas, dim=0, keepdim=True).repeat(
            init_poses.shape[0], 1
        )

        if cam_intrinsic.dtype == torch.float16:
            init_transl = inverse_perspective_projection(
                init_loc.unsqueeze(1).float(),
                cam_intrinsic.float(),
                init_dist.unsqueeze(1).float(),
            )[:, 0].half()
        else:
            init_transl = inverse_perspective_projection(
                init_loc.unsqueeze(1), cam_intrinsic, init_dist.unsqueeze(1)
            )[:, 0]

        # confidence of toe is related to the ankle
        # left ankle: 15, left_bigtoe: 17, left_smalltoe: 18, left_heel: 19
        # right ankle: 16, right_bigtoe: 20, right_smalltoe: 21, right_heel: 22
        keypoints_2d[:, [17, 18, 19], 2] = (
            keypoints_2d[:, [17, 18, 19], 2] * keypoints_2d[:, 15:16, 2]
        )
        keypoints_2d[:, [20, 21, 22], 2] = (
            keypoints_2d[:, [20, 21, 22], 2] * keypoints_2d[:, 16:17, 2]
        )

        k2d_orient_fitting = keypoints_2d[:, self.first_fitting_dst_idx]
        keypoints_2d = keypoints_2d[:, self.dst_idx]

        lr = self.lr

        # init_poses = axis_angle_to_rotation_6d(init_poses)
        # Stage 1. Optimize global_orient and translation
        params = [
            to_params(init_global_orient),
            to_params(init_body_poses),
            to_params(init_betas),
            to_params(init_transl),
        ]

        optim_params = [params[0], params[3]]  # loc seems unuseful

        optimizer = torch.optim.Adam(optim_params, lr=lr)

        with torch.no_grad():
            poses = torch.cat([params[0], params[1]], dim=1)
            out = self.smpl(
                rotation_6d_to_axis_angle(poses),
                params[2],
                None,
                None,
                transl=params[3],
                K=cam_intrinsic,
            )

            j3d = out["j3d_world"].detach().clone()
            del out

        first_step_loss = FastFirstFittingLoss(
            cam_intrinsics=cam_intrinsic,
            device=self.device,
            j3d_idx=self.first_fitting_src_idx,
        )

        for j in (j_bar := tqdm(range(30))):
            loss = first_step_loss(params[0], params[3], j3d, k2d_orient_fitting, bbox)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            msg = f"Loss: {loss.item():.1f}"
            j_bar.set_postfix_str(msg)

        del first_step_loss

        # Stage 2. Optimize all params

        init_poses_ = torch.cat(
            [params[0].detach().clone(), params[1].detach().clone()], dim=1
        )
        loss_fn = SMPLifyLoss(
            cam_intrinsics=cam_intrinsic,
            init_pose=init_poses_,
            device=self.device,
            j3d_idx=self.src_idx,
        )

        optimizer = torch.optim.Adam(params, lr=lr)
        closure = loss_fn.create_closure(
            optimizer, self.smpl, params, bbox, keypoints_2d
        )

        for j in (j_bar := tqdm(range(self.num_steps))):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            msg = f"Loss: {loss.item():.1f}"
            j_bar.set_postfix_str(msg)

        poses = torch.cat([params[0].detach(), params[1].detach()], dim=1)
        betas = params[2].detach()
        transl = params[3].detach()

        return rotation_6d_to_axis_angle(poses), betas, transl
