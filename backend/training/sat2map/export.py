"""
Phase 10 - checkpoint export.

Strips optimizer state from a training checkpoint and (optionally) traces the
model to TorchScript, producing a small artefact for
``app.localization.domain_translation.DomainTranslationEngine`` or an
edge-device runtime.

    python -m training.sat2map.export \
        --checkpoint ./weights/sat2map/sat2map_best.pt \
        --out ./weights/sat2map_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training.sat2map.model import UNetTranslator


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a Sat2Map checkpoint for inference.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("./weights/sat2map_best.pt"))
    ap.add_argument("--torchscript", type=Path, default=None,
                    help="Optional path to also write a traced .ts module")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    model_kwargs = ckpt.get("model_kwargs", {})
    model = UNetTranslator(**model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_kwargs": model_kwargs,
                "size": ckpt.get("size", args.size)}, args.out)
    print(f"Inference checkpoint written to {args.out}")

    if args.torchscript:
        example = torch.zeros(1, 3, args.size, args.size)
        traced = torch.jit.trace(model, example)
        traced.save(str(args.torchscript))
        print(f"TorchScript module written to {args.torchscript}")


if __name__ == "__main__":
    main()
