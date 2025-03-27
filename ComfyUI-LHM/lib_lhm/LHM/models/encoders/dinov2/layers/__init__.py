# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# ******************************************************************************
#   Code modified by Zexin He in 2023-2024.
#   Modifications are marked with clearly visible comments
#   licensed under the Apache License, Version 2.0.
# ******************************************************************************

from .dino_head import DINOHead
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
# ********** Modified by Zexin He in 2023-2024 **********
# Avoid using nested tensor for now, deprecating usage of NestedTensorBlock
from .block import Block, BlockWithModulation
# ********************************************************
from .attention import MemEffAttention
