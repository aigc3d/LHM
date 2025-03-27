# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Peihao Li & Lingteng Qiu & Xiaodong Gu & Qi Zuo
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-10 18:47:56
# @Function      : dataset base

import json
import pdb
import traceback
from abc import ABC, abstractmethod

import numpy as np
import torch
from megfile import smart_exists, smart_open, smart_path_join
from PIL import Image


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root_dirs: str, meta_path: str):
        super().__init__()
        self.root_dirs = root_dirs
        self.uids = self._load_uids(meta_path)

    def __len__(self):
        return len(self.uids)

    @abstractmethod
    def inner_get_item(self, idx):
        pass

    def __getitem__(self, idx):
        try:
            return self.inner_get_item(idx)
        except Exception as e:
            traceback.print_exc()
            print(f"[DEBUG-DATASET] Error when loading {self.uids[idx]}")
            # raise e
            return self.__getitem__((idx + 1) % self.__len__())

    @staticmethod
    def _load_uids(meta_path: str):
        # meta_path is a json file
        if meta_path == None:
            uids = []
        else:
            with open(meta_path, "r") as f:
                uids = json.load(f)

        return uids

    @staticmethod
    def _load_rgba_image(file_path, bg_color: float = 1.0):
        """Load and blend RGBA image to RGB with certain background, 0-1 scaled"""
        rgba = np.array(Image.open(smart_open(file_path, "rb")))
        rgba = torch.from_numpy(rgba).float() / 255.0
        rgba = rgba.permute(2, 0, 1).unsqueeze(0)
        rgb = rgba[:, :3, :, :] * rgba[:, 3:4, :, :] + bg_color * (
            1 - rgba[:, 3:, :, :]
        )
        # rgba[:, :3, ...] * rgba[:, 3:, ...] + (1 - rgba[:, 3:, ...])
        return rgb

    @staticmethod
    def _locate_datadir(root_dirs, uid, locator: str):
        for root_dir in root_dirs:
            datadir = smart_path_join(root_dir, uid, locator)
            if smart_exists(datadir):
                return root_dir
        raise FileNotFoundError(f"Cannot find valid data directory for uid {uid}")
