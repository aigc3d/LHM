# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2024-08-30 20:50:27
# @Function      : The class defines Bbox

import copy

import cv2
import numpy as np
import torch


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

    def offset(self, offset_w, offset_h):

        assert self.mode == "whwh"

        self.box[0] += offset_w
        self.box[1] += offset_h
        self.box[2] += offset_w
        self.box[3] += offset_h

    def get_box(self):

        return list(map(int, self.box))

    def to_xywh_ratio(self, ori_w, ori_h):

        cx, cy, w, h = self.to_xywh().get_box()
        cx = cx / ori_w
        cy = cy / ori_h
        w = w / ori_w
        h = h / ori_h

        return cx, cy, w, h

    def scale_bbox(self, ori_w, ori_h, new_w, new_h):
        """scale bbox as image scale"""

        assert self.mode == "whwh"

        cx, cy, w, h = self.to_xywh_ratio(ori_w, ori_h)

        cx = cx * new_w
        cy = cy * new_h
        w = w * new_w
        h = h * new_h

        l = cx - w // 2
        t = cy - h // 2
        r = cx + w - (w // 2)
        b = cy + h - (h // 2)

        return Bbox([l, t, r, b], mode="whwh")

    def scale(self, scale, width, height):
        """scale bbox with scale factor"""
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
