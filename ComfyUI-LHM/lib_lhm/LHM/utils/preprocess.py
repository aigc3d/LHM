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


import numpy as np
import rembg
import cv2


class Preprocessor:

    """
    Preprocessing under cv2 conventions.
    """

    def __init__(self):
        self.rembg_session = rembg.new_session(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def preprocess(self, image_path: str, save_path: str, rmbg: bool = True, recenter: bool = True, size: int = 512, border_ratio: float = 0.2):
        image = self.step_load_to_size(image_path=image_path, size=size*2)
        if rmbg:
            image = self.step_rembg(image_in=image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        if recenter:
            image = self.step_recenter(image_in=image, border_ratio=border_ratio, square_size=size)
        else:
            image = cv2.resize(
                src=image,
                dsize=(size, size),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.imwrite(save_path, image)

    def step_rembg(self, image_in: np.ndarray) -> np.ndarray:
        image_out = rembg.remove(
            data=image_in,
            session=self.rembg_session,
        )
        return image_out

    def step_recenter(self, image_in: np.ndarray, border_ratio: float, square_size: int) -> np.ndarray:
        assert image_in.shape[-1] == 4, "Image to recenter must be RGBA"
        mask = image_in[..., -1] > 0
        ijs = np.nonzero(mask)
        # find bbox
        i_min, i_max = ijs[0].min(), ijs[0].max()
        j_min, j_max = ijs[1].min(), ijs[1].max()
        bbox_height, bbox_width = i_max - i_min, j_max - j_min
        # recenter and resize
        desired_size = int(square_size * (1 - border_ratio))
        scale = desired_size / max(bbox_height, bbox_width)
        desired_height, desired_width = int(bbox_height * scale), int(bbox_width * scale)
        desired_i_min, desired_j_min = (square_size - desired_height) // 2, (square_size - desired_width) // 2
        desired_i_max, desired_j_max = desired_i_min + desired_height, desired_j_min + desired_width
        # create new image
        image_out = np.zeros((square_size, square_size, 4), dtype=np.uint8)
        image_out[desired_i_min:desired_i_max, desired_j_min:desired_j_max] = cv2.resize(
            src=image_in[i_min:i_max, j_min:j_max],
            dsize=(desired_width, desired_height),
            interpolation=cv2.INTER_AREA,
        )
        return image_out

    def step_load_to_size(self, image_path: str, size: int) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        height, width = image.shape[:2]
        scale = size / max(height, width)
        height, width = int(height * scale), int(width * scale)
        image_out = cv2.resize(
            src=image,
            dsize=(width, height),
            interpolation=cv2.INTER_AREA,
        )
        return image_out
