"""
Model + losses for aerial -> map translation.

The architecture lives in ``app.localization._sat2map_net`` so that the
checkpoint produced here always matches what :class:`DomainTranslationEngine`
loads at inference time. This module re-exports it and adds the training
objective.

Default objective (deterministic, supervised): L1 + SSIM + edge-consistency.
Geometric consistency matters more than photorealism, so there is no GAN loss
in the default path (an optional adversarial experiment can be added separately).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from app.localization._sat2map_net import UNetTranslator

__all__ = ["UNetTranslator", "TranslationLoss"]

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = _SOBEL_X.t()


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def _edges(x: torch.Tensor) -> torch.Tensor:
    g = _to_gray(x)
    kx = _SOBEL_X.to(x).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(x).view(1, 1, 3, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _ssim(a: torch.Tensor, b: torch.Tensor, window: int = 7) -> torch.Tensor:
    mu_a = F.avg_pool2d(a, window, 1, window // 2)
    mu_b = F.avg_pool2d(b, window, 1, window // 2)
    sa = F.avg_pool2d(a * a, window, 1, window // 2) - mu_a ** 2
    sb = F.avg_pool2d(b * b, window, 1, window // 2) - mu_b ** 2
    sab = F.avg_pool2d(a * b, window, 1, window // 2) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / \
           ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return ssim.clamp(0, 1).mean()


class TranslationLoss(torch.nn.Module):
    def __init__(self, l1: float = 1.0, ssim: float = 0.5, edge: float = 0.25):
        super().__init__()
        self.w_l1, self.w_ssim, self.w_edge = l1, ssim, edge

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l1 = F.l1_loss(pred, target)
        ssim_loss = 1.0 - _ssim(pred, target)
        edge_loss = F.l1_loss(_edges(pred), _edges(target))
        total = self.w_l1 * l1 + self.w_ssim * ssim_loss + self.w_edge * edge_loss
        return total, {"l1": float(l1), "ssim_loss": float(ssim_loss),
                       "edge_loss": float(edge_loss), "total": float(total)}
