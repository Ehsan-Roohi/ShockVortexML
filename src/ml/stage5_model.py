"""Dense multi-task model for joint shock/vortex segmentation.

The primary heads are independent logits so a shock and a vortex core can
occupy the same pixel.  A shared encoder is followed by task-specific decoders;
the training script optionally applies PCGrad only to the shared encoder.
"""

from __future__ import annotations

import torch
from torch import nn

from ml.model import ConvBlock, UpBlock


class TaskDecoder(nn.Module):
    """One full-resolution task decoder with no cross-task feature mixing."""

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        width = base_channels
        self.decoder_3 = UpBlock(width * 8, width * 4, width * 4)
        self.decoder_2 = UpBlock(width * 4, width * 2, width * 2)
        self.decoder_1 = UpBlock(width * 2, width, width)

    def forward(
        self,
        latent: torch.Tensor,
        level_3: torch.Tensor,
        level_2: torch.Tensor,
        level_1: torch.Tensor,
    ) -> torch.Tensor:
        decoded = self.decoder_3(latent, level_3)
        decoded = self.decoder_2(decoded, level_2)
        return self.decoder_1(decoded, level_1)


class Stage5JointNet(nn.Module):
    """Predict six independent dense maps from primitive CFD fields only.

    ``shock`` and ``vortex_core`` are deliberately not a softmax pair.  The two
    uncertainty heads mean weak-label review/ignore propensity; they are not
    claimed to be calibrated epistemic uncertainty.
    """

    output_names = (
        "shock",
        "vortex_core",
        "background_other",
        "geometry",
        "shock_uncertainty",
        "vortex_uncertainty",
    )
    physical_output_names = ("shock", "vortex_core", "background_other")

    def __init__(self, input_channels: int = 4, base_channels: int = 8) -> None:
        super().__init__()
        width = base_channels
        self.encoder_1 = ConvBlock(input_channels, width)
        self.encoder_2 = ConvBlock(width, width * 2)
        self.encoder_3 = ConvBlock(width * 2, width * 4)
        self.bottleneck = ConvBlock(width * 4, width * 8)
        self.pool = nn.MaxPool2d(2)

        self.shock_decoder = TaskDecoder(width)
        self.vortex_decoder = TaskDecoder(width)
        self.shock_head = nn.Conv2d(width, 1, 1)
        self.vortex_head = nn.Conv2d(width, 1, 1)
        self.shock_uncertainty_head = nn.Conv2d(width, 1, 1)
        self.vortex_uncertainty_head = nn.Conv2d(width, 1, 1)
        self.background_fusion = nn.Sequential(
            nn.Conv2d(width * 2, width, 1, bias=False),
            nn.GroupNorm(1, width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 1, 1),
        )
        # Geometry is an auxiliary learned output from primitive-field features;
        # the exact STL mask is intentionally not a model input.
        self.geometry_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(1, width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 1, 1),
        )

    def shared_encoder_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.encoder_1,
            self.encoder_2,
            self.encoder_3,
            self.bottleneck,
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def forward(
        self, inputs: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        level_1 = self.encoder_1(inputs)
        level_2 = self.encoder_2(self.pool(level_1))
        level_3 = self.encoder_3(self.pool(level_2))
        latent = self.bottleneck(self.pool(level_3))
        shock_features = self.shock_decoder(latent, level_3, level_2, level_1)
        vortex_features = self.vortex_decoder(latent, level_3, level_2, level_1)

        logits = torch.cat(
            (
                self.shock_head(shock_features),
                self.vortex_head(vortex_features),
                self.background_fusion(
                    torch.cat((shock_features, vortex_features), dim=1)
                ),
                self.geometry_head(level_1),
                self.shock_uncertainty_head(shock_features),
                self.vortex_uncertainty_head(vortex_features),
            ),
            dim=1,
        )
        if return_features:
            return logits, {
                "shared_latent": latent,
                "shock": shock_features,
                "vortex_core": vortex_features,
            }
        return logits
