"""
Phase 1 - dataset preparation.

Converts raw Sat2Maps-style samples (satellite | roadmap concatenated
side-by-side, as in taesungp/larger-google-sat2maps-dataset) into the layout
the training code expects:

    dataset/
        train/satellite/000001.png
        train/map/000001.png
        val/satellite/000001.png
        val/map/000001.png

Exact pair correspondence is preserved: ``satellite/NNNNNN.png`` and
``map/NNNNNN.png`` describe the same geographic area.

This utility is run by hand, never at application startup. It does not download
anything - point ``--src`` at an already-downloaded directory of images.

Usage:
    python -m training.sat2map.prepare_dataset \
        --src ./raw/sat2maps --out ./datasets/sat2maps \
        --val-split 0.05 --size 256
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _split_pair(img: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    A Sat2Maps tile is ``[satellite | roadmap]`` side by side (width == 2 * height).
    Returns ``(satellite, roadmap)`` or ``None`` if the aspect ratio is wrong.
    """
    h, w = img.shape[:2]
    if w >= 2 * h - 4 and w <= 2 * h + 4:
        half = w // 2
        return img[:, :half], img[:, half:half * 2]
    return None


def _is_corrupt(img: np.ndarray | None) -> bool:
    if img is None or img.size == 0:
        return True
    if img.std() < 1.0:  # flat / blank tile
        return True
    return False


def prepare(src: Path, out: Path, val_split: float, size: int, seed: int,
            already_split: bool) -> dict:
    rng = random.Random(seed)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"No images found under {src}")

    for part in ("train", "val"):
        for domain in ("satellite", "map"):
            (out / part / domain).mkdir(parents=True, exist_ok=True)

    kept = skipped = 0
    index = 0
    for path in files:
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if _is_corrupt(img):
            skipped += 1
            continue

        if already_split:
            # src/satellite/x.png + src/map/x.png already paired
            if path.parent.name != "satellite":
                continue
            sat = img
            map_path = path.parent.parent / "map" / path.name
            mp = cv2.imdecode(np.fromfile(str(map_path), np.uint8), cv2.IMREAD_COLOR) \
                if map_path.exists() else None
            if _is_corrupt(mp):
                skipped += 1
                continue
        else:
            pair = _split_pair(img)
            if pair is None:
                skipped += 1
                continue
            sat, mp = pair

        sat = cv2.resize(sat, (size, size), interpolation=cv2.INTER_AREA)
        mp = cv2.resize(mp, (size, size), interpolation=cv2.INTER_AREA)

        part = "val" if rng.random() < val_split else "train"
        name = f"{index:06d}.png"
        cv2.imencode(".png", sat)[1].tofile(str(out / part / "satellite" / name))
        cv2.imencode(".png", mp)[1].tofile(str(out / part / "map" / name))
        kept += 1
        index += 1

    summary = {"source_files": len(files), "pairs_written": kept, "skipped": skipped,
               "val_split": val_split, "size": size, "output": str(out)}
    print(summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a Sat2Maps-style dataset.")
    ap.add_argument("--src", type=Path, required=True,
                    help="Directory of raw side-by-side tiles (or paired subdirs)")
    ap.add_argument("--out", type=Path, default=Path("./datasets/sat2maps"))
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--already-split", action="store_true",
                    help="src already has satellite/ and map/ subdirectories")
    args = ap.parse_args()
    prepare(args.src, args.out, args.val_split, args.size, args.seed, args.already_split)


if __name__ == "__main__":
    main()
