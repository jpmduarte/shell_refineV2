"""The refiner's network: common/model.py's U-Net with two inputs and a probability out.

The inputs are the image and the first pass's prior (its SDF), so the network sees where the first
pass put the surface and only has to correct it; the output is the sigmoid probability of "inside"
(trained with Dice + BCE). The default size (depth 3, kernel 3, 16 channels) is the measured one: 1.40M parameters.
"""
import torch

from common.model import UNet3D

INPUT_CHANNELS = 2   # image, prior


class RefinerUNet(UNet3D):

    def __init__(self, in_channels: int = INPUT_CHANNELS, base_channels: int = 16, depth: int = 3,
                 kernel: int = 3):
        super().__init__(in_channels=in_channels, base_channels=base_channels, depth=depth, kernel=kernel)

    def forward(self, x):
        return torch.sigmoid(self.head(self.features(x)))
