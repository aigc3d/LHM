import numpy as np
import torch
import torch.nn.functional as F
from pose_utils.rot6d import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle


def get_gaussian_kernel_1d(kernel_size, sigma, device):
    x = torch.arange(kernel_size).float() - (kernel_size // 2)
    g = torch.exp(-((x**2) / (2 * sigma**2)))
    g /= g.sum()

    kernel_weight = g.view(1, 1, -1).to(device)

    return kernel_weight


def gaussian_filter_1d(data, kernel_size=3, sigma=1.0, weight=None):
    kernel_weight = (
        get_gaussian_kernel_1d(kernel_size, sigma, data.device)
        if weight is None
        else weight
    )
    data = F.pad(data, (kernel_size // 2, kernel_size // 2), mode="replicate")
    return F.conv1d(data, kernel_weight)


def exponential_smoothing(x, d_x, alpha=0.5):
    return d_x + alpha * (x - d_x)


@torch.no_grad()
def smplx_gs_smooth(poses, betas, transl, fps=30):
    poses = axis_angle_to_rotation_6d(poses)
    N, J, _ = poses.shape
    poses = (
        gaussian_filter_1d(
            poses.view(N, 1, -1).permute(2, 1, 0),
            kernel_size=9,
            sigma=1 * fps / 30,
        )
        .permute(2, 1, 0)
        .view(N, J, -1)
    )
    betas = (
        gaussian_filter_1d(
            betas.view(-1, 1, 10).permute(2, 1, 0),
            kernel_size=11,
            sigma=5.0 * fps / 30,
        )
        .permute(2, 1, 0)
        .view(-1, 10)
    )
    transl[1:-1] = (
        gaussian_filter_1d(
            transl.view(N, 1, -1).permute(2, 1, 0),
            kernel_size=9,
            sigma=1.0 * fps / 30,
        )
        .permute(2, 1, 0)
        .view(N, -1)[1:-1]
    )

    poses = rotation_6d_to_axis_angle(poses)
    return poses, betas, transl


class OneEuroFilter:
    # param setting:
    #   realtime v2m: min_cutoff=1.0, beta=1.5
    #   motionshop 2d keypoint: min_cutoff=1.7, beta=0.3
    def __init__(
        self, min_cutoff=1.0, beta=0.0, sampling_rate=30, d_cutoff=1.0, device="cuda"
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.sampling_rate = sampling_rate
        self.x_prev = None
        self.dx_prev = None
        self.d_cutoff = d_cutoff
        self.pi = torch.tensor(torch.pi, device=device)

    def smoothing_factor(self, cutoff):

        r = 2 * self.pi * cutoff / self.sampling_rate
        return r / (1 + r)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = torch.zeros_like(x)
            return x

        a_d = self.smoothing_factor(self.d_cutoff)
        # 计算当前的速度
        dx = (x - self.x_prev) * self.sampling_rate

        dx_hat = exponential_smoothing(dx, self.dx_prev, a_d)

        cutoff = self.min_cutoff + self.beta * torch.abs(dx_hat)
        a = self.smoothing_factor(cutoff)

        x_hat = exponential_smoothing(x, self.x_prev, a)

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat
