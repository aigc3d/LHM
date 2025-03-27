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


from accelerate.logging import get_logger


logger = get_logger(__name__)


def configure_dynamo(config: dict):
    try:
        import torch._dynamo
        logger.debug(f'Configuring torch._dynamo.config with {config}')
        for k, v in config.items():
            if v is None:
                logger.debug(f'Skipping torch._dynamo.config.{k} with None')
                continue
            if hasattr(torch._dynamo.config, k):
                logger.warning(f'Overriding torch._dynamo.config.{k} from {getattr(torch._dynamo.config, k)} to {v}')
                setattr(torch._dynamo.config, k, v)
    except ImportError:
        logger.debug('torch._dynamo not found, skipping')
        pass
