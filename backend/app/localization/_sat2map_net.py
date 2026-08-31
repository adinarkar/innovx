"""
U-Net used for aerial/satellite -> map-style translation (spec Phase 2).

Kept deliberately small and deterministic: this is a localisation aid, not a
photorealistic GAN. A supervised L1/SSIM/edge objective on this architecture
preserves road geometry and block boundaries, which is what the downstream
feature matcher needs.

This is the single source of truth for the architecture. Training code in
``backend/training/sat2map`` imports :class:`UNetTranslator` from here so the
checkpoint an operator drops into ``backend/weights`` always matches what the
inference engine expects.

Importing this module requires PyTorch; callers import it lazily.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetTranslator(nn.Module):
    """Classic 4-level U-Net. Input and output are 3-channel images in [0, 1]."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base: int = 48):
        super().__init__()
        self.enc1 = _block(in_channels, base)
        self.enc2 = _block(base, base * 2)
        self.enc3 = _block(base * 2, base * 4)
        self.enc4 = _block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = _block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _block(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))
