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

__all__ = ['PixelLoss']


class PixelLoss(nn.Module):
    """
    Pixel-wise loss between two images.
    """

    def __init__(self, option: str = 'mse'):
        super().__init__()
        self.loss_fn = self._build_from_option(option)

    @staticmethod
    def _build_from_option(option: str, reduction: str = 'none'):
        if option == 'mse':
            return nn.MSELoss(reduction=reduction)
        elif option == 'l1':
            return nn.L1Loss(reduction=reduction)
        else:
            raise NotImplementedError(f'Unknown pixel loss option: {option}')

    @torch.compile
    def forward(self, x, y):
        """
        Assume images are channel first.
        
        Args:
            x: [N, M, C, H, W]
            y: [N, M, C, H, W]
        
        Returns:
            Mean-reduced pixel loss across batch.
        """
        N, M, C, H, W = x.shape
        x = x.reshape(N*M, C, H, W)
        y = y.reshape(N*M, C, H, W)
        image_loss = self.loss_fn(x, y).mean(dim=[1, 2, 3])
        batch_loss = image_loss.reshape(N, M).mean(dim=1)
        all_loss = batch_loss.mean()
        return all_loss
