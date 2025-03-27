# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2024-08-30 16:26:10
# @Function      : SAM2 Segment class

import sys

sys.path.append("./")
import copy
import os
import pdb
import tempfile
import time
from bisect import bisect_left
from dataclasses import dataclass

import cv2
import numpy as np
import PIL
import torch
from pytorch3d.ops import sample_farthest_points
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision import transforms

from engine.BiRefNet.models.birefnet import BiRefNet
from engine.ouputs import BaseOutput
from engine.SegmentAPI.base import BaseSeg, Bbox
from engine.SegmentAPI.img_utils import load_image_file

SAM2_WEIGHT = "pretrained_models/sam2/sam2.1_hiera_large.pt"
BIREFNET_WEIGHT = "pretrained_models/BiRefNet-general-epoch_244.pth"


def avaliable_device():
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device


@dataclass
class SegmentOut(BaseOutput):
    masks: np.ndarray
    processed_img: np.ndarray
    alpha_img: np.ndarray


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def FPS(sample, num):
    n = sample.shape[0]
    center = np.mean(sample, axis=0)
    select_p = []
    L = []
    for i in range(n):
        L.append(distance(sample[i], center))
    p0 = np.argmax(L)
    select_p.append(p0)
    L = []
    for i in range(n):
        L.append(distance(p0, sample[i]))
    select_p.append(np.argmax(L))
    for i in range(num - 2):
        for p in range(n):
            d = distance(sample[select_p[-1]], sample[p])
            if d <= L[p]:
                L[p] = d
        select_p.append(np.argmax(L))
    return select_p, sample[select_p]


def fill_mask(alpha):
    # alpha = np.pad(alpha, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    h, w = alpha.shape[:2]

    mask = np.zeros((h + 2, w + 2), np.uint8)
    alpha = (alpha * 255).astype(np.uint8)
    im_floodfill = alpha.copy()
    retval, image, mask, rect = cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    alpha = alpha | im_floodfill_inv
    alpha = alpha.astype(np.float32) / 255.0

    # return alpha[1 : h - 1, 1 : w - 1, ...]
    return alpha


def erode_and_dialted(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)

    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=iterations)

    return dilated_mask


def eroded(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)

    return eroded_mask


def model_type(model):
    print(next(model.parameters()).device)


class SAM2Seg(BaseSeg):
    RATIO_MAP = [[512, 1], [1280, 0.6], [1920, 0.4], [3840, 0.2]]

    def tocpu(self):
        self.box_prior.cpu()
        self.image_predictor.model.cpu()
        torch.cuda.empty_cache()

    def tocuda(self):
        self.box_prior.cuda()
        self.image_predictor.model.cuda()

    def __init__(
        self,
        config="sam2.1_hiera_l.yaml",
        matting_config="resnet50",
        background=(1.0, 1.0, 1.0),
        wo_supres=False,
    ):
        super().__init__()

        self.device = avaliable_device()

        try:
            sam2_image_model = build_sam2(config, SAM2_WEIGHT)
        except:
            config = os.path.join("./configs/sam2.1/", config)  # sam2.1 case
            sam2_image_model = build_sam2(config, SAM2_WEIGHT)

        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        self.box_prior = None

        # Robust-Human-Matting

        # self.matting_predictor = MattingNetwork(matting_config).eval().cuda()
        # self.matting_predictor.load_state_dict(torch.load(MATTING_WEIGHT))

        self.background = background
        self.wo_supers = wo_supres

    def clean_up(self):
        self.tmp.cleanup()

    def collect_inputs(self, inputs):
        return dict(
            img_path=inputs["img_path"],
            bbox=inputs["bbox"],
        )

    def _super_resolution(self, input_path):

        low = os.path.abspath(input_path)
        high = self.tmp.name

        super_weights = os.path.abspath("./pretrained_models/RealESRGAN_x4plus.pth")
        hander = os.path.join(SUPRES_PATH, "inference_realesrgan.py")

        cmd = f"python {hander} -n RealESRGAN_x4plus -i {low} -o {high} --model_path {super_weights} -s 2"

        os.system(cmd)

        return os.path.join(high, os.path.basename(input_path))

    def predict_bbox(self, img, scale=1.0):

        ratio = self.ratio_mapping(img)

        # uint8
        # [0 1]
        img = np.asarray(img).astype(np.float32) / 255.0
        height, width, _ = img.shape

        # [C H W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        bgr = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1).cuda()  # Green background.
        rec = [None] * 4  # Initial recurrent states.

        # predict matting
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            fgr, pha, *rec = self.matting_predictor(
                img_tensor.to(self.device),
                *rec,
                downsample_ratio=ratio,
            )  # Cycle the recurrent states.

        pha[pha < 0.5] = 0.0
        pha[pha >= 0.5] = 1.0
        pha = pha[0].permute(1, 2, 0).detach().cpu().numpy()

        # obtain bbox
        _h, _w, _ = np.where(pha == 1)

        whwh = [
            _w.min().item(),
            _h.min().item(),
            _w.max().item(),
            _h.max().item(),
        ]

        box = Bbox(whwh)

        # scale box to 1.05
        scale_box = box.scale(1.00, width=width, height=height)

        return scale_box, pha[..., 0]

    def birefnet_predict_bbox(self, img, scale=1.0):

        # img: RGB-order

        if self.box_prior == None:
            from engine.BiRefNet.utils import check_state_dict

            birefnet = BiRefNet(bb_pretrained=False)
            state_dict = torch.load(BIREFNET_WEIGHT, map_location="cpu")
            state_dict = check_state_dict(state_dict)
            birefnet.load_state_dict(state_dict)
            device = avaliable_device()
            torch.set_float32_matmul_precision(["high", "highest"][0])

            birefnet.to(device)
            self.box_prior = birefnet
            self.box_prior.eval()
            self.box_transform = transforms.Compose(
                [
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            print("BiRefNet is ready to use.")
        else:
            device = avaliable_device()
            self.box_prior.to(device)

        height, width, _ = img.shape

        image = PIL.Image.fromarray(img)

        input_images = self.box_transform(image).unsqueeze(0).to("cuda")
        with torch.no_grad():
            preds = self.box_prior(input_images)[-1].sigmoid().cpu()
        pha = (preds[0]).squeeze(0).detach().numpy()

        pha = cv2.resize(pha, (width, height))

        masks = copy.deepcopy(pha[..., None])

        masks[masks < 0.3] = 0.0
        masks[masks >= 0.3] = 1.0

        # obtain bbox
        _h, _w, _ = np.where(masks == 1)

        whwh = [
            _w.min().item(),
            _h.min().item(),
            _w.max().item(),
            _h.max().item(),
        ]

        box = Bbox(whwh)

        # scale box to 1.05
        scale_box = box.scale(scale=scale, width=width, height=height)

        return scale_box, pha

    def rembg_predict_bbox(self, img, scale=1.0):

        height, width, _ = img.shape

        with torch.no_grad():
            img_rmbg = img[..., ::-1]  # rgb2bgr
            img_rmbg = remove(img_rmbg)
            img_rmbg = img_rmbg[..., :3]
            pha = copy.deepcopy(img_rmbg[..., -1:])

        masks = copy.deepcopy(pha)

        masks[masks < 1.0] = 0.0
        masks[masks >= 1.0] = 1.0

        # obtain bbox
        _h, _w, _ = np.where(masks == 1)

        whwh = [
            _w.min().item(),
            _h.min().item(),
            _w.max().item(),
            _h.max().item(),
        ]

        box = Bbox(whwh)

        # scale box to 1.05
        scale_box = box.scale(scale=scale, width=width, height=height)

        return scale_box, pha[..., 0].astype(np.float32) / 255.0

    def yolo_predict_bbox(self, img, scale=1.0, threshold=0.2):
        if self.prior == None:
            from ultralytics import YOLO

            pdb.set_trace()

        height, width, _ = img.shape

        with torch.no_grad():
            results = yolo_seg(img[..., ::-1])
            for result in results:
                masks = result.masks.data[result.boxes.cls == 0]
                if masks.shape[0] >= 1:
                    masks[masks >= threshold] = 1
                    masks[masks < threshold] = 0
                    masks = masks.sum(dim=0)

        pha = masks.detach().cpu().numpy()
        pha = cv2.resize(pha, (width, height), interpolation=cv2.INTER_AREA)[..., None]

        pha[pha >= 0.5] = 1
        pha[pha < 0.5] = 0

        masks = copy.deepcopy(pha)

        pha = pha * 255.0
        # obtain bbox
        _h, _w, _ = np.where(masks == 1)

        whwh = [
            _w.min().item(),
            _h.min().item(),
            _w.max().item(),
            _h.max().item(),
        ]

        box = Bbox(whwh)

        # scale box to 1.05
        scale_box = box.scale(scale=scale, width=width, height=height)

        return scale_box, pha[..., 0].astype(np.float32) / 255.0

    def ratio_mapping(self, img):

        my_ratio_map = self.RATIO_MAP

        ratio_landmarks = [v[0] for v in my_ratio_map]

        ratio_v = [v[1] for v in my_ratio_map]
        h, w, _ = img.shape

        max_length = min(h, w)

        low_bound = bisect_left(
            ratio_landmarks, max_length, lo=0, hi=len(ratio_landmarks)
        )

        if 0 == low_bound:
            return 1.0
        elif low_bound == len(ratio_landmarks):
            return ratio_v[-1]
        else:
            _l = ratio_v[low_bound - 1]
            _r = ratio_v[low_bound]

            _l_land = ratio_landmarks[low_bound - 1]
            _r_land = ratio_landmarks[low_bound]
            cur_ratio = _l + (_r - _l) * (max_length - _l_land) / (_r_land - _l_land)

            return cur_ratio

    def get_img(self, img_path, sup_res=True):

        img = cv2.imread(img_path)
        img = img[..., ::-1].copy()  # bgr2rgb

        if self.wo_supers:
            return img

        return img

    def compute_coords(self, pha, bbox):

        node_prompts = []

        H, W = pha.shape
        y_indices, x_indices = np.indices((H, W))
        coors = np.stack((x_indices, y_indices), axis=-1)

        # reduce the effect from pha
        # pha = eroded((pha * 255).astype(np.uint8), 3, 3) / 255.0

        pha_coors = np.repeat(pha[..., None], 2, axis=2)
        coors_points = (coors * pha_coors).sum(axis=0).sum(axis=0) / (pha.sum() + 1e-6)
        node_prompts.append(coors_points.tolist())

        _h, _w = np.where(pha > 0.5)

        sample_ps = torch.from_numpy(np.stack((_w, _h), axis=-1).astype(np.float32)).to(
            avaliable_device()
        )

        # positive prompts
        node_prompts_fps, _ = sample_farthest_points(sample_ps[None], K=5)
        node_prompts_fps = (
            node_prompts_fps[0].detach().cpu().numpy().astype(np.int32).tolist()
        )

        node_prompts.extend(node_prompts_fps)
        node_prompts_label = [1 for _ in range(len(node_prompts))]

        return node_prompts, node_prompts_label

    def _forward(self, img_path, bbox, sup_res=True):

        img = self.get_img(img_path, sup_res)

        if bbox is None:
            # bbox, pha = self.predict_bbox(img)
            # bbox, pha = self.rembg_predict_bbox(img, 1.01)
            # bbox, pha = self.yolo_predict_bbox(img)
            bbox, pha = self.birefnet_predict_bbox(img, 1.01)

        box = bbox.to_whwh()
        bbox = box.get_box()

        point_coords, point_coords_label = self.compute_coords(pha, bbox)

        self.image_predictor.set_image(img)

        masks, scores, logits = self.image_predictor.predict(
            point_coords=point_coords,
            point_labels=point_coords_label,
            box=bbox,
            multimask_output=False,
        )

        alpha = masks[0]

        # fill-mask NO USE
        # alpha = fill_mask(alpha)
        # alpha = erode_and_dialted(
        #     (alpha * 255).astype(np.uint8), kernel_size=3, iterations=3
        # )
        # alpha = alpha.astype(np.float32) / 255.0

        img_float = img.astype(np.float32) / 255.0
        process_img = (
            img_float * alpha[..., None] + (1 - alpha[..., None]) * self.background
        )
        process_img = (process_img * 255).astype(np.uint8)

        # using for draw box
        # process_img = cv2.rectangle(process_img, bbox[:2], bbox[2:], (0, 0, 255), 2)
        process_img = process_img.astype(np.float) / 255.0

        process_pha_img = (
            img_float * pha[..., None] + (1 - pha[..., None]) * self.background
        )

        return SegmentOut(
            masks=alpha, processed_img=process_img, alpha_img=process_pha_img[...]
        )

    @torch.no_grad()
    def __call__(self, **inputs):

        self.tmp = tempfile.TemporaryDirectory()

        self.collect_inputs(inputs)

        out = self._forward(**inputs)

        self.clean_up()
        return out


def get_parse():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", required=True, help="input path")
    parser.add_argument("-o", "--output", required=True, help="output path")
    parser.add_argument("--mask", action="store_true", help="mask bool")
    parser.add_argument(
        "--wo_super_reso", action="store_true", help="whether using super_resolution"
    )
    args = parser.parse_args()
    return args


def main():

    opt = get_parse()
    img_list = os.listdir(opt.input)
    img_names = [os.path.join(opt.input, img_name) for img_name in img_list]

    os.makedirs(opt.output, exist_ok=True)

    model = SAM2Seg(wo_supres=opt.wo_super_reso)

    for img in img_names:

        print(f"processing {img}")
        out = model(img_path=img, bbox=None)

        save_path = os.path.join(opt.output, os.path.basename(img))

        alpha = fill_mask(out.masks)
        alpha = erode_and_dialted(
            (alpha * 255).astype(np.uint8), kernel_size=3, iterations=3
        )
        save_img = alpha
        cv2.imwrite(save_path, save_img)


if __name__ == "__main__":

    main()
