"""
Fetch a Sat2Maps-style dataset from the Hugging Face Hub.

The canonical source in the spec -
``http://efrosgans.eecs.berkeley.edu/datasets/larger_sat2maps_cleaned.tar``
(taesungp/larger-google-sat2maps-dataset, ~92k pairs) - is a UC Berkeley host
that is frequently unreachable. ``huggan/maps`` is the same
satellite<->Google-roadmap pairing (the original pix2pix "maps" set, 1096 train
/ 1098 val, 600x600) and is a drop-in substitute for prototyping.

It writes the ``--already-split`` layout that ``prepare_dataset.py`` consumes:

    <out>/satellite/000001.png
    <out>/map/000001.png

Then run:

    python -m training.sat2map.prepare_dataset \
        --src <out> --out ./datasets/sat2maps --already-split --size 256

Usage:
    python -m training.sat2map.fetch_hf_maps --repo huggan/maps --out ./raw/sat2maps
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np


def _to_bgr(value) -> np.ndarray:
    """Accept a PIL image, or the ``{'bytes': ..., 'path': ...}`` dict some
    Hugging Face parquet datasets store instead of a decoded Image feature."""
    import cv2
    from PIL import Image

    if isinstance(value, dict):
        if value.get("bytes"):
            img = Image.open(io.BytesIO(value["bytes"]))
        elif value.get("path"):
            img = Image.open(value["path"])
        else:
            raise ValueError(f"cannot decode image dict: keys={list(value)}")
    else:
        img = value
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a Sat2Maps-style set from HF.")
    ap.add_argument("--repo", default="huggan/maps")
    ap.add_argument("--out", type=Path, default=Path("./raw/sat2maps"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--aerial-col", default="imageA",
                    help="column holding the satellite/aerial image")
    ap.add_argument("--map-col", default="imageB",
                    help="column holding the Google-roadmap image")
    args = ap.parse_args()

    import cv2
    from datasets import load_dataset

    (args.out / "satellite").mkdir(parents=True, exist_ok=True)
    (args.out / "map").mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.repo)
    idx = 0
    for split in ds:
        rows = ds[split]
        n = len(rows) if not args.limit else min(len(rows), args.limit)
        for i in range(n):
            row = rows[i]
            sat = _to_bgr(row[args.aerial_col])
            mp = _to_bgr(row[args.map_col])
            name = f"{idx:06d}.png"
            cv2.imencode(".png", sat)[1].tofile(str(args.out / "satellite" / name))
            cv2.imencode(".png", mp)[1].tofile(str(args.out / "map" / name))
            idx += 1
        if args.limit and idx >= args.limit:
            break
    print(f"Wrote {idx} pairs to {args.out}/satellite and {args.out}/map")


if __name__ == "__main__":
    main()
