# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-10 17:38:29
# @Function      : Arc-Similarity Loss
import sys

sys.path.append(".")

import pdb
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel


def conv3x3(inplanes, outplanes, stride=1):
    """A simple wrapper for 3x3 convolution with padding.

    Args:
        inplanes (int): Channel number of inputs.
        outplanes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
    """
    return nn.Conv2d(
        inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """Basic residual block used in the ResNetArcFace architecture.

    Args:
        inplanes (int): Channel number of inputs.
        planes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
        downsample (nn.Module): The downsample module. Default: None.
    """

    expansion = 1  # output channel expansion ratio

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IRBlock(nn.Module):
    """Improved residual block (IR Block) used in the ResNetArcFace architecture.

    Args:
        inplanes (int): Channel number of inputs.
        planes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
        downsample (nn.Module): The downsample module. Default: None.
        use_se (bool): Whether use the SEBlock (squeeze and excitation block). Default: True.
    """

    expansion = 1  # output channel expansion ratio

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block used in the ResNetArcFace architecture.

    Args:
        inplanes (int): Channel number of inputs.
        planes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
        downsample (nn.Module): The downsample module. Default: None.
    """

    expansion = 4  # output channel expansion ratio

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    """The squeeze-and-excitation block (SEBlock) used in the IRBlock.

    Args:
        channel (int): Channel number of inputs.
        reduction (int): Channel reduction ration. Default: 16.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(
            1
        )  # pool to 1x1 without spatial information
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetArcFace(nn.Module):
    """ArcFace with ResNet architectures.

    Ref: ArcFace: Additive Angular Margin Loss for Deep Face Recognition.

    Args:
        block (str): Block used in the ArcFace architecture.
        layers (tuple(int)): Block numbers in each layer.
        use_se (bool): Whether use the SEBlock (squeeze and excitation block). Default: True.
    """

    def __init__(
        self,
        block="IRBlock",
        layers=[2, 2, 2, 2],
        use_se=False,
        pretrain_model="./pretrained_models/arcface_resnet18.pth",
    ):
        if block == "IRBlock":
            block = IRBlock
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetArcFace, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        if pretrain_model is not None:
            self.load_network(self, pretrain_model, strict=True, param_key=None)
        else:
            raise ValueError("Please specify the pretrain model path.")

        self.freeze()

    @staticmethod
    def load_network(net, load_path, strict=True, param_key=None):

        def get_bare_model(net):
            if isinstance(net, (DataParallel, DistributedDataParallel)):
                net = net.module
            return net

        net = get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and "params" in load_net:
                param_key = "params"
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith("module."):
                load_net[k[7:]] = v
                load_net.pop(k)
        ret = net.load_state_dict(load_net, strict=strict)
        print(ret)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, use_se=self.use_se)
        )
        self.inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)

        return x

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    model = ResNetArcFace()
    model.cuda()
    model.eval()
    # model.eval()

    set1 = [
        "./debug/face_debug/gt/head_gt_0.png",
        "./debug/face_debug/gt/head_gt_1.png",
        "./debug/face_debug/gt/head_gt_2.png",
        "./debug/face_debug/gt/head_gt_3.png",
        "./debug/face_debug/gt/head_gt_4.png",
        "./debug/face_debug/gt/head_gt_5.png",
        "./debug/face_debug/gt/head_gt_6.png",
    ]
    import cv2

    img_set1 = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in set1]

    F1_list = []

    f1_scores = []
    for img in img_set1:
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) / 255.0
        img = img.cuda()
        F1 = model(img)
        F1_list.append(F1)
    for i in range(len(F1_list)):
        for j in range(len(F1_list)):
            f1_scores.append(F.l1_loss(F1_list[i], F1_list[j]))

    print(len(f1_scores))

    f1_scores = torch.tensor(f1_scores)
    print(f1_scores)
    f1_scores = f1_scores.view(len(F1_list), len(F1_list))
    print(f1_scores)
