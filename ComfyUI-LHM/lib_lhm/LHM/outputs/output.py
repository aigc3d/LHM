"""A class to define GSNet Output
"""

from dataclasses import dataclass

import numpy as np
from torch import Tensor

from .base import BaseOutput


@dataclass
class GaussianAppOutput(BaseOutput):
    """
    Output of the Gaussian Appearance output.

    Attributes:

    """

    offset_xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor
    use_rgb: bool
