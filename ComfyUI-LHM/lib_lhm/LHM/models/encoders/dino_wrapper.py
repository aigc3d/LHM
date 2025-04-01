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


import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel
from accelerate.logging import get_logger


logger = get_logger(__name__)


class DinoWrapper(nn.Module):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """
    def __init__(self, model_name: str, freeze: bool = True, encoder_feat_dim: int = 384):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        if freeze:
            self._freeze()

    @torch.compile
    def forward_model(self, inputs):
        return self.model(**inputs, interpolate_pos_encoding=True)

    def forward(self, image):
        # image: [N, C, H, W], on cpu
        # RGB image with [0,1] scale and properly sized
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False, do_resize=False).to(self.model.device)
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.forward_model(inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    def _freeze(self):
        logger.warning(f"======== Freezing DinoWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            model = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
            processor = ViTImageProcessor.from_pretrained(model_name)
            return model, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying ({proxy_error_retries}) in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err
