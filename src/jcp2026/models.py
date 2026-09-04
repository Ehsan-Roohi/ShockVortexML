from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from ml.stage5_model import Stage5JointNet


class SegformerBaseline(nn.Module):
    """Official HF SegFormer-B0; only input/output channels and loss adapted."""
    output_names = Stage5JointNet.output_names

    def __init__(self, input_channels: int):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        config = SegformerConfig(num_channels=input_channels, num_labels=6,
                                 depths=[2,2,2,2], hidden_sizes=[32,64,160,256],
                                 decoder_hidden_size=256)
        self.network = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def shared_encoder_parameters(self):
        return list(self.network.segformer.parameters())


def make_model(spec: dict, width: int = 10) -> nn.Module:
    n = 7 if spec["diagnostics"] else 4
    if spec["architecture"] == "joint_unet":
        return Stage5JointNet(input_channels=n, base_channels=width)
    if spec["architecture"] == "segformer_b0":
        return SegformerBaseline(n)
    raise ValueError("Unknown architecture")
