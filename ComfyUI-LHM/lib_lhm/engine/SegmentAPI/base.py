# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2024-08-30 20:50:27
# @Function      : The class defines bbox, base-seg module

import copy

import cv2
import numpy as np
import torch


class BaseModel(object):
    """
    Simple BaseModel
    """

    def cuda(self):
        self.model.cuda()
        return self

    def cpu(self):
        self.model.cpu()
        return self

    def float(self):
        self.model.float()
        return self

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()

        return self

    def train(self):
        self.model.train()
        return self

    def __call__(self, x):
        raise NotImplementedError

    def __repr__(self):

        return f"model: \n{self.model}"


def get_dtype_string(arr):
    if arr.dtype == np.uint8:
        return "uint8"
    elif arr.dtype == np.float32:
        return "float32"
    elif arr.dtype == np.float64:
        return "float"
    else:
        return "unknow"


class BaseSeg(BaseModel):
    def __init__(self):
        pass


class Bbox:
    def __init__(self, box, mode="whwh"):

        assert len(box) == 4
        assert mode in ["whwh", "xywh"]
        self.box = box
        self.mode = mode

    def to_xywh(self):

        if self.mode == "whwh":

            l, t, r, b = self.box

            center_x = (l + r) / 2
            center_y = (t + b) / 2
            width = r - l
            height = b - t
            return Bbox([center_x, center_y, width, height], mode="xywh")
        else:
            return self

    def to_whwh(self):

        if self.mode == "whwh":
            return self
        else:

            cx, cy, w, h = self.box
            l = cx - w // 2
            t = cy - h // 2
            r = cx + w - (w // 2)
            b = cy + h - (h // 2)

            return Bbox([l, t, r, b], mode="whwh")

    def area(self):

        box = self.to_xywh()
        _, __, w, h = box.box

        return w * h

    def get_box(self):
        return list(map(int, self.box))

    def scale(self, scale, width, height):
        new_box = self.to_xywh()
        cx, cy, w, h = new_box.get_box()
        w = w * scale
        h = h * scale

        l = cx - w // 2
        t = cy - h // 2
        r = cx + w - (w // 2)
        b = cy + h - (h // 2)

        l = int(max(l, 0))
        t = int(max(t, 0))
        r = int(min(r, width))
        b = int(min(b, height))

        return Bbox([l, t, r, b], mode="whwh")

    def __repr__(self):
        box = self.to_whwh()
        l, t, r, b = box.box

        return f"BBox(left={l}, top={t}, right={r}, bottom={b})"


class Image:
    """TODO need to debug"""

    TYPE_ORDER = ["uint8", "float32", "float"]
    ORDER = ["RGB", "BGR"]
    MODE = ["numpy"]

    def __init__(self, input, order="RGB", type_mode="uint8"):
        """Only support 3 Channel Image"""
        if isinstance(input, str):
            self.data = self.read_image(input, type_mode, order)
        else:
            self.data = self.get_image(input, type_mode, order)

        self.order = order
        self.type_mode = type_mode

    def get_image(self, input, type_mode, order):
        if isinstance(input, Image):
            return input.to_numpy(type_mode, order)
        elif isinstance(input, np.ndarray):
            self.data = input
            self.order = "RGB"  # default
            self.type_mode = get_dtype_string(input)

            return self.to_numpy(type_mode, order)
        else:
            raise NotImplementedError

    def to_numpy(self, type_mode="uint8", order="RGB"):

        data = copy.deepcopy(self.data)

        if not order == self.order:
            return data[..., ::-1]  # only support RGB -> BGR or BGR -> RGB

        if self.type_mode == type_mode:
            return data
        else:
            if self.type_mode == "float32":
                return (self.data / 255.0).astype(np.float32)
            elif self.type_mode == "float":
                return (self.data / 255.0).astype(np.float64)

    def to_tensor(self, order):
        data = self.to_numpy(type_mode="float32", order=order)
        return torch.from_numpy(data)

    def read_image(
        self,
        path,
        mode,
        order,
    ):
        """read an image file into various formats and color mode.

        Args:
            path (str): path to the image file.
            mode (Literal["float", "uint8", "pil", "torch", "tensor"], optional): returned image format. Defaults to "float".
                float: float32 numpy array, range [0, 1];
                uint8: uint8 numpy array, range [0, 255];
                pil: PIL image;
                torch/tensor: float32 torch tensor, range [0, 1];
            order (Literal["RGB", "RGBA", "BGR", "BGRA"], optional): channel order. Defaults to "RGB".

        Note:
            By default this function will convert RGBA image to white-background RGB image. Use ``order="RGBA"`` to keep the alpha channel.

        Returns:
            Union[np.ndarray, PIL.Image, torch.Tensor]: the image array.
        """

        if mode == "pil":
            return Image.open(path).convert(order)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # cvtColor
        if len(img.shape) == 3:  # ignore if gray scale
            if order in ["RGB", "RGBA"]:
                if img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                elif img.shape[-1] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # mix background
            if img.shape[-1] == 4 and "A" not in order:
                img = img.astype(np.float32) / 255
                img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:])

        # mode
        if mode == "uint8":
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
        elif mode == "float":
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255
        else:
            raise ValueError(f"Unknown read_image mode {mode}")

        return img
