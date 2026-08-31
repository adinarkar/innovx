"""
Phase 10 - evaluation.

Reports mapping-relevant metrics, not just visual quality:

    - L1 / MAE
    - SSIM
    - edge overlap (IoU of thresholded Sobel edges)
    - structural IoU (IoU of "dark structure" masks: roads/blocks)

Also writes qualitative triptychs: input satellite | ground-truth map |
predicted map, side by side.

    python -m training.sat2map.evaluate \
        --dataset ./datasets/sat2maps --checkpoint ./weights/sat2map/sat2map_best.pt \
        --out ./weights/sat2map/eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from training.sat2map.dataset import PairedTranslationDataset
from training.sat2map.model import UNetTranslator


def _edge_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (cv2.Canny(gray, 60, 160) > 0)


def _structure_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (gray < 110)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the Sat2Map translator.")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--aerial-dir", default="satellite")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, default=Path("./weights/sat2map/eval"))
    ap.add_argument("--max-triptychs", type=int, default=24)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(args.checkpoint), map_location=device)
    model = UNetTranslator(**ckpt.get("model_kwargs", {})).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = PairedTranslationDataset(args.dataset, args.split, args.aerial_dir, augment=False)
    dl = DataLoader(ds, batch_size=8)
    args.out.mkdir(parents=True, exist_ok=True)

    agg = {"mae": 0.0, "ssim": 0.0, "edge_iou": 0.0, "structure_iou": 0.0}
    n = saved = 0
    with torch.inference_mode():
        for aerial, target in dl:
            pred = model(aerial.to(device)).clamp(0, 1).cpu()
            for i in range(aerial.size(0)):
                a = (aerial[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                t = (target[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                p = (pred[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                agg["mae"] += float(np.abs(p.astype(int) - t.astype(int)).mean())
                agg["ssim"] += _ssim(p, t)
                agg["edge_iou"] += _iou(_edge_mask(p), _edge_mask(t))
                agg["structure_iou"] += _iou(_structure_mask(p), _structure_mask(t))
                n += 1
                if saved < args.max_triptychs:
                    strip = np.concatenate([a, t, p], axis=1)
                    cv2.imencode(".png", cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))[1] \
                        .tofile(str(args.out / f"triptych_{saved:03d}.png"))
                    saved += 1

    metrics = {k: round(v / max(n, 1), 4) for k, v in agg.items()}
    metrics["samples"] = n
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()
