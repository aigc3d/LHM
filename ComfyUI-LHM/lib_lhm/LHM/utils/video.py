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


import os
import numpy as np
import imageio
import torch

def images_to_video(images, output_path, fps, gradio_codec: bool, verbose=False):
    # images: torch.tensor (T, C, H, W), 0-1  or numpy: (T, H, W, 3) 0-255
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        if isinstance(images, torch.Tensor):
            frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
                f"Frame shape mismatch: {frame.shape} vs {images.shape}"
            assert frame.min() >= 0 and frame.max() <= 255, \
                f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        else:
            frame = images[i]
        frames.append(frame)
    frames = np.stack(frames)
    if gradio_codec:
        imageio.mimwrite(output_path, frames, fps=fps, quality=10)
    else:
        # imageio.mimwrite(output_path, frames, fps=fps, codec='mpeg4', quality=10)
        imageio.mimwrite(output_path, frames, fps=fps, quality=10)

    if verbose:
        print(f"Using gradio codec option {gradio_codec}")
        print(f"Saved video to {output_path}")
