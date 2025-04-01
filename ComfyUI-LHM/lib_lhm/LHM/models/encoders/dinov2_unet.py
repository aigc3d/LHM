#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
import torchvision
import torch.nn as nn
import timm
from accelerate.logging import get_logger

logger = get_logger(__name__)


    
class DINOBase(nn.Module):
    def __init__(self, output_dim=128, only_global=False):
        super().__init__()
        self.only_global = only_global
        assert self.only_global == False
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        
        # self.encoder = timm.create_model("resnet18", pretrained=True)
        # del self.encoder.global_pool
        # del self.encoder.fc
        
        # model_name = "dinov2_vits14_reg"
        # modulation_dim = None
        # self.dino_model = self._build_dinov2(model_name, modulation_dim=modulation_dim)
        
        self.dino_normlize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        
        in_dim = self.dino_model.blocks[0].attn.qkv.in_features
        hidden_dims=256
        out_dims=[256, 512, 1024, 1024]
        # modules
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_dim, out_dim, kernel_size=1, stride=1, padding=0,
            ) for out_dim in out_dims
        ])

        self.resize_layers = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(
                    out_dims[0], out_dims[0], kernel_size=3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(
                    out_dims[0], out_dims[0], kernel_size=3, stride=1, padding=1)
                ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(
                    out_dims[1], out_dims[1], kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(
                    out_dims[2], out_dims[2], kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(
                    out_dims[3], out_dims[3], kernel_size=3, stride=2, padding=1)
            )
        ])
        # self.layer_rn = nn.ModuleList([
        #     nn.Conv2d(out_dims[0]+64, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.Conv2d(out_dims[1]+128, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.Conv2d(out_dims[2]+256, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.Conv2d(out_dims[3]+512, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
        # ])
        self.layer_rn = nn.ModuleList([
            nn.Conv2d(out_dims[0]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(out_dims[1]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(out_dims[2]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(out_dims[3]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
        ])
        # self.layer_rn = nn.ModuleList([
        #     nn.Conv2d(out_dims[0], hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.Conv2d(out_dims[1], hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.Conv2d(out_dims[2], hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.Conv2d(out_dims[3], hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
        # ])

        self.refinenet = nn.ModuleList([
            FeatureFusionBlock(hidden_dims, nn.ReLU(False), use_conv1=False),
            FeatureFusionBlock(hidden_dims, nn.ReLU(False)),
            FeatureFusionBlock(hidden_dims, nn.ReLU(False)),
            FeatureFusionBlock(hidden_dims, nn.ReLU(False)),
        ])
        self.output_conv = nn.Conv2d(hidden_dims, output_dim, kernel_size=3, stride=1, padding=1)
        # self.output_gloabl_proj = nn.Linear(384, output_dim)

    @staticmethod
    def _build_dinov2(model_name: str, modulation_dim: int = None, pretrained: bool = True):
        from importlib import import_module
        dinov2_hub = import_module(".dinov2.hub.backbones", package=__package__)
        model_fn = getattr(dinov2_hub, model_name)
        logger.debug(f"Modulation dim for Dinov2 is {modulation_dim}.")
        model = model_fn(modulation_dim=modulation_dim, pretrained=pretrained)
        return model
    
    def forward(self, images, output_size=None, is_training=True):
        # enc_output = self.encoder.forward_intermediates(images, stop_early=True, intermediates_only=True)
        # enc_out4 = enc_output[4]  # 32
        # enc_out3 = enc_output[3]  # 16
        # enc_out2 = enc_output[2]  # 8
        # enc_out1 = enc_output[1]  # 4

        images = self.dino_normlize(images)
        patch_h, patch_w = images.shape[-2]//14, images.shape[-1]//14
        
        image_features = self.dino_model.get_intermediate_layers(images, 4)
        
        out_features = []
        for i, feature in enumerate(image_features):
            feature = feature.permute(0, 2, 1).reshape(
                (feature.shape[0], feature.shape[-1], patch_h, patch_w)
            )
            feature = self.projects[i](feature)
            feature = self.resize_layers[i](feature)
            # print(enc_output[i+1].shape, feature.shape)
            feature = torch.cat([
                    nn.functional.interpolate(images, (feature.shape[-2], feature.shape[-1]), mode="bilinear", align_corners=True),
                    feature
                ], dim=1
            )
            out_features.append(feature)
        layer_rns = []
        for i, feature in enumerate(out_features):
            layer_rns.append(self.layer_rn[i](feature))

        path_4 = self.refinenet[0](layer_rns[3], size=layer_rns[2].shape[2:])
        path_3 = self.refinenet[1](path_4, layer_rns[2], size=layer_rns[1].shape[2:])
        path_2 = self.refinenet[2](path_3, layer_rns[1], size=layer_rns[0].shape[2:])
        path_1 = self.refinenet[3](path_2, layer_rns[0])
        out = self.output_conv(path_1)

        if output_size is not None:
            out = nn.functional.interpolate(out, output_size, mode="bilinear", align_corners=True)
        # out_global = image_features[-1][:, 0]
        # out_global = self.output_gloabl_proj(out_global)
        out_global = None
        return out, out_global


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)
        # return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, 
                 use_conv1=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        if use_conv1:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
            self.skip_add = nn.quantized.FloatFunctional()

        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output = output + res

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)
        return output
