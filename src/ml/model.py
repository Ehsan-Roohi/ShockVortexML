"""A compact multi-label U-Net for CPU-friendly pilot experiments."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(input_channels + skip_channels, output_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = functional.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.block(torch.cat([inputs, skip], dim=1))


class MultiHeadUNet(nn.Module):
    """Predict independent shock, vortex-core, and background/other logits.

    The independent sigmoid heads intentionally permit shock/vortex overlap.
    """

    output_names = ("shock", "vortex_core", "background_other")

    def __init__(self, input_channels: int = 5, base_channels: int = 8) -> None:
        super().__init__()
        width = base_channels
        self.encoder_1 = ConvBlock(input_channels, width)
        self.encoder_2 = ConvBlock(width, width * 2)
        self.encoder_3 = ConvBlock(width * 2, width * 4)
        self.bottleneck = ConvBlock(width * 4, width * 8)
        self.pool = nn.MaxPool2d(2)
        self.decoder_3 = UpBlock(width * 8, width * 4, width * 4)
        self.decoder_2 = UpBlock(width * 4, width * 2, width * 2)
        self.decoder_1 = UpBlock(width * 2, width, width)
        self.head = nn.Conv2d(width, len(self.output_names), 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        level_1 = self.encoder_1(inputs)
        level_2 = self.encoder_2(self.pool(level_1))
        level_3 = self.encoder_3(self.pool(level_2))
        latent = self.bottleneck(self.pool(level_3))
        decoded = self.decoder_3(latent, level_3)
        decoded = self.decoder_2(decoded, level_2)
        decoded = self.decoder_1(decoded, level_1)
        return self.head(decoded)

