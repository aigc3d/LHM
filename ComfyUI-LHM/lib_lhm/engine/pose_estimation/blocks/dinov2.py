# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license
import os
current_dir_path = os.path.dirname(__file__)
import torch
from torch import nn

class Dinov2Backbone(nn.Module):
    def __init__(self, name='dinov2_vitb14', pretrained=False, *args, **kwargs):
        super().__init__()
        self.name = name
        self.encoder = torch.hub.load(current_dir_path+'/../dinov2', self.name, pretrained=pretrained, source='local')
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim

    def forward(self, x):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert len(x.shape) == 4
        y = self.encoder.get_intermediate_layers(x)[0] # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        return y

