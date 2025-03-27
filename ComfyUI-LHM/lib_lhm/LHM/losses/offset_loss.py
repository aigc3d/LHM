# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-10 19:08:56
# @Function      : ACAP Loss
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ACAP_Loss", "Heuristic_ACAP_Loss"]


class ACAP_Loss(nn.Module):
    """As close as possibel loss"""

    def forward(self, offset, d=0.05625, **params):
        """Empirically, where d is the thresold of distance points leave from 1.8/32 = 0.0562."""

        offset_loss = torch.clamp(offset.norm(p=2, dim=-1), min=d) - d

        return offset_loss.mean()


class Heuristic_ACAP_Loss(nn.Module):
    """As close as possibel loss"""

    def __init__(self, group_dict, group_body_mapping):
        super(Heuristic_ACAP_Loss, self).__init__()

        self.group_dict = group_dict  # register weights fro different body parts
        self.group_body_mapping = group_body_mapping  # mapping of body parts to group

    def _heurisitic_loss(self, _offset_loss):

        _loss = 0.0
        for key in self.group_dict.keys():
            key_weights = self.group_dict[key]
            group_mapping_idx = self.group_body_mapping[key]
            _loss += key_weights * _offset_loss[:, group_mapping_idx].mean()

        return _loss

    def forward(self, offset, d=0.05625, **params):
        """Empirically, where d is the thresold of distance points leave from human prior model, 1.8/32 = 0.0562."""
        "human motion or rotation is very different in each body parts, for example, the head is more stable than the leg and hand, so we use heuristic_ball_loss"

        _offset_loss = torch.clamp(offset.norm(p=2, dim=-1), min=d) - d

        return self._heurisitic_loss(_offset_loss)
