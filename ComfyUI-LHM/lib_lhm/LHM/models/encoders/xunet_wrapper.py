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
import timm
from accelerate.logging import get_logger

logger = get_logger(__name__)

class XUNet(nn.Module):
    def __init__(self, model_name="swin_base_patch4_window12_384_in22k", encoder_feat_dim=384):
        super(XUNet, self).__init__()
        # Swin Transformer Encoder
        self.encoder = timm.create_model(model_name, pretrained=True)
        # swin
        # del self.encoder.head
        # del self.encoder.norm
        # resnet
        del self.encoder.global_pool
        del self.encoder.fc

        # Decoder layers
        # self.upconv4 = self.upconv_block(2048, 1024)  # Upsample
        # self.upconv3 = self.upconv_block(1024, 512)
        # self.upconv2 = self.upconv_block(512, 256)
        # self.upconv1 = self.upconv_block(256, 64)
        
        self.upconv4 = self.upconv_block(512, 256)  # Upsample
        self.upconv3 = self.upconv_block(256, 128)
        self.upconv2 = self.upconv_block(128, 64)
        # self.upconv1 = self.upconv_block(64, 64)
        
        self.out_conv = nn.Conv2d(64, encoder_feat_dim, kernel_size=1)
        

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder part using Swin Transformer
        enc_output = self.encoder.forward_intermediates(x, stop_early=True, intermediates_only=True)
        
        # for e in enc_output:
        #     print(e.shape, x.shape)
            
        # Assuming output of the encoder is a list of feature maps
        # Resize them according to UNet architecture
        enc_out4 = enc_output[4]  # Adjust according to the feature layers of Swin
        enc_out3 = enc_output[3]
        enc_out2 = enc_output[2]
        enc_out1 = enc_output[1]
        # enc_out0 = enc_output[0]

        # Decoder part
        x = self.upconv4(enc_out4) 
        x = x + enc_out3  # s16, Skip connection
        x = self.upconv3(x)
        x = x + enc_out2  # s8
        x = self.upconv2(x)
        x = x + enc_out1 # s4
        # x = self.upconv1(x)
        # x = x + enc_out0  # s2

        x = self.out_conv(x)
        return x


class XnetWrapper(nn.Module):
    """
    XnetWrapper using original implementation, hacked with modulation.
    """
    def __init__(self, model_name: str, modulation_dim: int = None, freeze: bool = True, encoder_feat_dim: int = 384):
        super().__init__()
        self.modulation_dim = modulation_dim
        self.model = XUNet(model_name=model_name, encoder_feat_dim=encoder_feat_dim)

        if freeze:
            if modulation_dim is not None:
                raise ValueError("Modulated SwinUnetWrapper requires training, freezing is not allowed.")
            self._freeze()

    def _freeze(self):
        logger.warning(f"======== Freezing SwinUnetWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @torch.compile
    def forward(self, image: torch.Tensor, mod: torch.Tensor = None):
        # image: [N, C, H, W]
        # mod: [N, D] or None
        # RGB image with [0,1] scale and properly sized
        outs = self.model(image)
        ret = outs.permute(0, 2, 3, 1).flatten(1, 2)
        return ret
