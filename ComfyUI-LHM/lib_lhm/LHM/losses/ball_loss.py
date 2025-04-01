# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-10 19:08:35
# @Function      : ASAP loss
import pdb

import torch
import torch.nn as nn

__all__ = ["ASAP_Loss", "Heuristic_ASAP_Loss"]


class ASAP_Loss(nn.Module):

    def forward(self, scaling, r=1, **params):
        """where r is the radius of the ball between max-axis and min-axis."""
        raise NotImplementedError(
            "ASAP_Loss is not implemented yet in Inference version"
        )


class Heuristic_ASAP_Loss(nn.Module):
    def __init__(self, group_dict, group_body_mapping):
        super(Heuristic_ASAP_Loss, self).__init__()

        self.group_dict = group_dict  # register weights fro different body parts
        self.group_body_mapping = group_body_mapping  # mapping of body parts to group

    def _heurisitic_loss(self, _ball_loss):

        _loss = 0.0
        for key in self.group_dict.keys():
            key_weights = self.group_dict[key]
            group_mapping_idx = self.group_body_mapping[key]
            _loss += key_weights * _ball_loss[:, group_mapping_idx].mean()

        return _loss

    def forward(self, scaling, r=5, **params):
        """where r is the radius of the ball between max-axis and min-axis."""
        "human motion or rotation is very different in each body parts, for example, the head is more stable than the leg and hand, so we use heuristic_ball_loss"

        _scale = scaling

        _scale_min = torch.min(_scale, dim=-1)[0]
        _scale_max = torch.max(_scale, dim=-1)[0]

        scale_ratio = _scale_max / (_scale_min + 1e-6)

        _ball_loss = torch.clamp(scale_ratio, min=r) - r

        return self._heurisitic_loss(_ball_loss)
