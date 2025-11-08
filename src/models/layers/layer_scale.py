# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://arxiv.org/abs/2103.17239

import torch
from torch import nn


class LayerScale(nn.Module):
    """Layer scale module.

    References:
        - CaiT: Going deeper with Image Transformers (https://arxiv.org/abs/2103.17239)
        - DINOv2 implementation

    Args:
        dim (int): Input dimension.
        init_values (float): Initial value for layer scale. Default: 1e-5.
        inplace (bool): Whether to perform inplace operation. Default: False.
    """

    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
