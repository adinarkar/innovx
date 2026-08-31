"""
Phase 10 - training entry point, independent of the FastAPI server.

    python -m training.sat2map.train \
        --dataset ./datasets/sat2maps \
        --epochs 50 --batch-size 8 --output ./weights/sat2map

Provides: deterministic seed, CPU/GPU detection, checkpoint saving, resume,
best-checkpoint tracking, train/val loss logging and a config dump.

The checkpoint written to ``<output>/sat2map_best.pt`` is directly loadable by
``app.localization.domain_translation.DomainTranslationEngine`` - copy or point
``SAT2MAP_MODEL_PATH`` at it and set ``SAT2MAP_ENABLED=true``.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.sat2map.dataset import PairedTranslationDataset
from training.sat2map.model import TranslationLoss, UNetTranslator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> dict:
    train = optimizer is not None
    model.train(train)
    totals: dict = {}
    n = 0
    for aerial, target in loader:
        aerial, target = aerial.to(device), target.to(device)
        with torch.set_grad_enabled(train):
            pred = model(aerial)
            loss, parts = loss_fn(pred, target)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        bs = aerial.size(0)
        n += bs
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v * bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Sat2Map translator.")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--aerial-dir", default="satellite",
                    help="'satellite' for Sat2Maps, 'aerial' for a drone fine-tune")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base-channels", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--output", type=Path, default=Path("./weights/sat2map"))
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = detect_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "device": str(device)}, indent=2))
    print(f"Training on {device}. Config dumped to {args.output/'config.json'}.")

    model_kwargs = {"base": args.base_channels}
    train_ds = PairedTranslationDataset(args.dataset, "train", args.aerial_dir,
                                        args.size, augment=True, seed=args.seed)
    val_ds = PairedTranslationDataset(args.dataset, "val", args.aerial_dir,
                                      args.size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    model = UNetTranslator(base=args.base_channels).to(device)
    loss_fn = TranslationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch, best_val = 0, float("inf")
    if args.resume and args.resume.exists():
        ckpt = torch.load(str(args.resume), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")

    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_dl, loss_fn, device, optimizer)
        va = run_epoch(model, val_dl, loss_fn, device)
        dt = time.time() - t0
        row = {"epoch": epoch, "train": tr, "val": va, "seconds": round(dt, 1)}
        history.append(row)
        print(f"epoch {epoch:03d}  train {tr['total']:.4f}  val {va['total']:.4f}  {dt:.1f}s")

        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "epoch": epoch, "best_val": best_val, "model_kwargs": model_kwargs,
                   "size": args.size}
        torch.save(payload, args.output / "sat2map_last.pt")
        if va["total"] < best_val:
            best_val = va["total"]
            payload["best_val"] = best_val
            torch.save(payload, args.output / "sat2map_best.pt")
            print(f"  new best ({best_val:.4f}) -> sat2map_best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2))

    print(f"Done. Best val loss {best_val:.4f}. Weights in {args.output}.")


if __name__ == "__main__":
    main()
