# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Peihao Li
# @Email         : liphao99@gmail.com
# @Time          : 2025-03-11 12:47:58
# @Function      : inference code for pose estimation

import os
import sys
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir+"/../../")

import pdb
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from engine.ouputs import BaseOutput
from engine.pose_estimation.model import load_model
from PIL import Image

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


@dataclass
class SMPLXOutput(BaseOutput):
    beta: np.ndarray
    is_full_body: bool
    msg: str


def normalize_rgb_tensor(img, imgenet_normalization=True):
    img = img / 255.0
    if imgenet_normalization:
        img = (
            img - torch.tensor(IMG_NORM_MEAN, device=img.device).view(1, 3, 1, 1)
        ) / torch.tensor(IMG_NORM_STD, device=img.device).view(1, 3, 1, 1)
    return img


class PoseEstimator:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.mhmr_model = load_model(
            os.path.join(model_path, "pose_estimate", "multiHMR_896_L.pt"),
            model_path=model_path,
            device=self.device,
        )
        self.pad_ratio = 0.2
        self.img_size = 896
        self.fov = 60
    
    def to(self, device):
        self.device = device
        self.mhmr_model.to(device)
        return self

    def get_camera_parameters(self):
        K = torch.eye(3)
        # Get focal length.
        focal = self.img_size / (2 * np.tan(np.radians(self.fov) / 2))
        K[0, 0], K[1, 1] = focal, focal

        K[0, -1], K[1, -1] = self.img_size // 2, self.img_size // 2

        # Add batch dimension
        K = K.unsqueeze(0).to(self.device)
        return K

    def img_center_padding(self, img_np):

        ori_h, ori_w = img_np.shape[:2]

        w = round((1 + self.pad_ratio) * ori_w)
        h = round((1 + self.pad_ratio) * ori_h)

        img_pad_np = np.zeros((h, w, 3), dtype=np.uint8)
        offset_h, offset_w = (h - img_np.shape[0]) // 2, (w - img_np.shape[1]) // 2
        img_pad_np[
            offset_h : offset_h + img_np.shape[0] :,
            offset_w : offset_w + img_np.shape[1],
        ] = img_np

        return img_pad_np, offset_w, offset_h

    def _preprocess(self, img_np):

        raw_img_size = max(img_np.shape[:2])

        img_tensor = (
            torch.Tensor(img_np).to(self.device).unsqueeze(0).permute(0, 3, 1, 2)
        )

        _, _, h, w = img_tensor.shape
        scale_factor = min(self.img_size / w, self.img_size / h)
        img_tensor = F.interpolate(
            img_tensor, scale_factor=scale_factor, mode="bilinear"
        )

        _, _, h, w = img_tensor.shape
        pad_left = (self.img_size - w) // 2
        pad_top = (self.img_size - h) // 2
        pad_right = self.img_size - w - pad_left
        pad_bottom = self.img_size - h - pad_top
        img_tensor = F.pad(
            img_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

        resize_img = normalize_rgb_tensor(img_tensor)

        annotation = (
            pad_left,
            pad_top,
            scale_factor,
            self.img_size / scale_factor,
            raw_img_size,
        )

        return resize_img, annotation

    @torch.no_grad()
    def __call__(self, img_path):
        # image_tensor H W C

        img_np = np.asarray(Image.open(img_path).convert("RGB"))

        raw_h, raw_w, _ = img_np.shape

        # pad image for more accurate pose estimation
        img_np, offset_w, offset_h = self.img_center_padding(img_np)
        img_tensor, annotation = self._preprocess(img_np)
        K = self.get_camera_parameters()

        with torch.cuda.amp.autocast(enabled=True):
            target_human = self.mhmr_model(
                img_tensor,
                is_training=False,
                nms_kernel_size=int(3),
                det_thresh=0.3,
                K=K,
                idx=None,
                max_dist=None,
            )
        if not len(target_human) == 1:
            return SMPLXOutput(
                beta=None,
                is_full_body=False,
                msg=(
                    "more than one human detected"
                    if len(target_human) > 1
                    else "no human detected"
                ),
            )

        # check is full body
        pad_left, pad_top, scale_factor, _, _ = annotation
        j2d = target_human[0]["j2d"]
        # tranform to raw image space
        j2d = (
            j2d - torch.tensor([pad_left, pad_top], device=self.device).unsqueeze(0)
        ) / scale_factor
        j2d = j2d - torch.tensor([offset_w, offset_h], device=self.device).unsqueeze(0)

        # enable the full body contains 95% of the image
        scale_ratio = 0.025

        is_full_body = (
            (
                (j2d[..., 0] >= 0 - raw_w * scale_ratio)
                & (j2d[..., 0] < raw_w * (1 + scale_ratio))
                & (j2d[..., 1] >= 0 - raw_h * scale_ratio)
                & (j2d[..., 1] < raw_h * (1 + scale_ratio))
            )
            .all(dim=-1)
            .item()
        )

        return SMPLXOutput(
            beta=target_human[0]["shape"].cpu().numpy(),
            is_full_body=is_full_body,
            msg="success" if is_full_body else "no full-body human detected",
        )
