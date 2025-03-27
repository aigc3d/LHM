import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose

# from openlrm.models.encoders.dpt_util.dinov2 import DINOv2
from LHM.models.encoders.dpt_util.blocks import FeatureFusionBlock, _make_scratch
from LHM.models.encoders.dpt_util.transform import NormalizeImage, PrepareForNet, Resize


def _make_fusion_block(features, use_bn, size=None, use_conv1=True):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        use_conv1=use_conv1,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        out_channel=384,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        # self.resize_layers = nn.ModuleList([
        #     nn.ConvTranspose2d(
        #         in_channels=out_channels[0],
        #         out_channels=out_channels[0],
        #         kernel_size=4,
        #         stride=4,
        #         padding=0),
        #     nn.ConvTranspose2d(
        #         in_channels=out_channels[1],
        #         out_channels=out_channels[1],
        #         kernel_size=2,
        #         stride=2,
        #         padding=0),
        #     nn.Identity(),
        #     nn.Conv2d(
        #         in_channels=out_channels[3],
        #         out_channels=out_channels[3],
        #         kernel_size=3,
        #         stride=2,
        #         padding=1)
        # ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn, use_conv1=False)

        head_features_1 = features
        head_features_2 = 32

        # self.scratch.output_conv1 = nn.Conv2d(head_features_1, out_channnels, kernel_size=3, stride=1, padding=1)

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, out_channel, kernel_size=1, stride=1, padding=0
        )

        # self.scratch.output_conv2 = nn.Sequential(
        #     nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True),
        #     nn.Identity(),
        # )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            # x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(
            layer_4_rn, size=layer_3_rn.shape[2:], scale_factor=1
        )
        path_3 = self.scratch.refinenet3(
            path_4, layer_3_rn, size=layer_2_rn.shape[2:], scale_factor=1
        )
        path_2 = self.scratch.refinenet2(
            path_3, layer_2_rn, size=layer_1_rn.shape[2:], scale_factor=1
        )
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, scale_factor=1)

        # path_4 = self.scratch.refinenet4(layer_1_rn, size=layer_2_rn.shape[2:], scale_factor=1)
        # path_3 = self.scratch.refinenet3(path_4, layer_2_rn, size=layer_3_rn.shape[2:], scale_factor=1)
        # path_2 = self.scratch.refinenet2(path_3, layer_3_rn, size=layer_4_rn.shape[2:], scale_factor=1)
        # path_1 = self.scratch.refinenet1(path_2, layer_4_rn, scale_factor=1)

        out = self.scratch.output_conv1(path_1)
        # out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        # out = self.scratch.output_conv2(out)

        return out


class DINODPT(nn.Module):
    def __init__(
        self, model_name="vitb", out_dim=384, use_bn=False, use_clstoken=False
    ):
        super(DINODPT, self).__init__()

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }

        encoder = model_configs[model_name]["encoder"]
        features = model_configs[model_name]["features"]
        out_channels = model_configs[model_name]["out_channels"]

        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }

        self.encoder = encoder

        # self.dino_model = DINOv2(model_name=encoder)
        self.dino_model = torch.hub.load(
            "facebookresearch/dinov2", f"dinov2_{encoder}14", pretrained=True
        )
        self.dense_head = DPTHead(
            self.dino_model.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            out_channel=out_dim,
        )

        self.dino_normlize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, x, is_training=True):
        x = self.dino_normlize(x)

        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.dino_model.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder], return_class_token=True
        )

        feat = self.dense_head(features, patch_h, patch_w)
        # print(x.shape, feat.shape)
        # depth = F.relu(depth)
        # return depth.squeeze(1)
        out_global = None
        return feat, out_global

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)

        depth = self.forward(image)

        depth = F.interpolate(
            depth[:, None], (h, w), mode="bilinear", align_corners=True
        )[0, 0]

        return depth.cpu().numpy()

    def image2tensor(self, raw_image, input_size=518):
        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0)

        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        image = image.to(DEVICE)

        return image, (h, w)
