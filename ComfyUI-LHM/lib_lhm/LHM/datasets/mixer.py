# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");:
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


import math
import pdb
from functools import partial

import torch

__all__ = ["MixerDataset"]


class MixerDataset(torch.utils.data.Dataset):
    """Reference"""

    def __init__(
        self,
        split: str,
        subsets: dict,
        **dataset_kwargs,
    ):

        self.subsets = [
            self._dataset_fn(subset, split)(
                use_flame=subset["use_flame"],
                src_head_size=subset.get("src_head_size", 448),
                **dataset_kwargs,
            )
            for subset in subsets
        ]
        self.virtual_lens = [
            math.ceil(subset_config["sample_rate"] * len(subset_obj))
            for subset_config, subset_obj in zip(subsets, self.subsets)
        ]

    @staticmethod
    def _dataset_fn(subset_config: dict, split: str):
        name = subset_config["name"]

        dataset_cls = None
        if name == "video_human":
            from .video_human import VideoHumanDataset

        else:
            raise NotImplementedError(f"Dataset {name} not implemented")

        return partial(
            dataset_cls,
            root_dirs=subset_config["root_dirs"],
            meta_path=subset_config["meta_path"][split],
        )

    def __len__(self):
        return sum(self.virtual_lens)

    def __getitem__(self, idx):
        subset_idx = 0
        virtual_idx = idx
        while virtual_idx >= self.virtual_lens[subset_idx]:
            virtual_idx -= self.virtual_lens[subset_idx]
            subset_idx += 1
        real_idx = virtual_idx % len(self.subsets[subset_idx])
        return self.subsets[subset_idx][real_idx]
