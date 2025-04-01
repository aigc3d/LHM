"""
Ported from Paella
"""

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

import functools
# import torch.nn as nn
from taming.modules.util import ActNorm


# Discriminator model ported from Paella https://github.com/dome272/Paella/blob/main/src_distributed/vqgan.py
class Discriminator(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels=3, cond_channels=0, hidden_channels=512, depth=6):
        super().__init__()
        d = max(depth - 3, 3)
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, hidden_channels // (2**d), kernel_size=3, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2),
        ]
        for i in range(depth - 1):
            c_in = hidden_channels // (2 ** max((d - i), 0))
            c_out = hidden_channels // (2 ** max((d - 1 - i), 0))
            layers.append(nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2))
        self.encoder = nn.Sequential(*layers)
        self.shuffle = nn.Conv2d(
            (hidden_channels + cond_channels) if cond_channels > 0 else hidden_channels, 1, kernel_size=1
        )
        # self.logits = nn.Sigmoid()
        
        
    def forward(self, x, cond=None):
        x = self.encoder(x)
        if cond is not None:
            cond = cond.view(
                cond.size(0),
                cond.size(1),
                1,
                1,
            ).expand(-1, -1, x.size(-2), x.size(-1))
            x = torch.cat([x, cond], dim=1)
        x = self.shuffle(x)
        # x = self.logits(x)
        return x




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            # use_bias = norm_layer.func != nn.BatchNorm2d
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            # use_bias = norm_layer != nn.BatchNorm2d
            use_bias = norm_layer != nn.InstanceNorm2d
        
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, False)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)
        
    def forward(self, input):
        """Standard forward."""
        return self.main(input)
