# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn

__all__ = ['TVLoss']


class TVLoss(nn.Module):
    """
    Total variance loss.
    """

    def __init__(self):
        super().__init__()

    def numel_excluding_first_dim(self, x):
        return x.numel() // x.shape[0]

    @torch.compile
    def forward(self, x):
        """
        Assume batched and channel first with inner sizes.

        Args:
            x: [N, M, C, H, W]

        Returns:
            Mean-reduced TV loss with element-level scaling.
        """
        N, M, C, H, W = x.shape
        x = x.reshape(N*M, C, H, W)
        diff_i = x[..., 1:, :] - x[..., :-1, :]
        diff_j = x[..., :, 1:] - x[..., :, :-1]
        div_i = self.numel_excluding_first_dim(diff_i)
        div_j = self.numel_excluding_first_dim(diff_j)
        tv_i = diff_i.pow(2).sum(dim=[1,2,3]) / div_i
        tv_j = diff_j.pow(2).sum(dim=[1,2,3]) / div_j
        tv = tv_i + tv_j
        batch_tv = tv.reshape(N, M).mean(dim=1)
        all_tv = batch_tv.mean()
        return all_tv
