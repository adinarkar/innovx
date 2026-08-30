"""
Offline evaluation harness - runs the localisation pipeline over a generated
test set and reports Top-1 accuracy, no-match detection and timings.

    python -m scripts.evaluate --data ../test_data/generated

Run from the ``backend`` directory.  Useful for checking a threshold change
without going through the UI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                       # noqa: E402
from app.localization.imaging import imread           # noqa: E402
from app.localization.pipeline import build_map_index, localize  # noqa: E402
from app.store import MapRecord, new_id               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../test_data/generated")
    ap.add_argument("--tolerance", type=int, default=150,
                    help="Position error in map pixels still counted as correct")
    ap.add_argument("--only", default=None, help="Substring filter on case filenames")
    args = ap.parse_args()

    data = Path(args.data).resolve()
    truth = json.loads((data / "ground_truth.json").read_text(encoding="utf-8"))
    map_path = data / truth["reference_map"]

    img = imread(map_path)
    h, w = img.shape[:2]
    rec = MapRecord(map_id=new_id("map"), path=map_path, width=w, height=h,
                    filename=map_path.name, file_size=map_path.stat().st_size)

    t0 = time.time()
    build_map_index(rec)
    print(f"Indexed {len(rec.tiles)} tiles in {time.time() - t0:.1f}s "
          f"({rec.embedding_backend})\n")

    rows, correct, evaluated, no_match_ok, no_match_total, false_loc = [], 0, 0, 0, 0, 0
    for case in truth["cases"]:
        if args.only and args.only not in case["file"]:
            continue
        started = time.time()
        res = localize(rec, data / case["file"], settings.processed_dir / new_id("eval"))
        elapsed = time.time() - started

        status = res["status"]
        px = res.get("map_pixel")
        err = None
        ok = None
        if case["expect_no_match"]:
            no_match_total += 1
            ok = status == "NO_MATCH"
            no_match_ok += int(ok)
            false_loc += int(status == "MATCH_FOUND")
        else:
            evaluated += 1
            if px:
                err = ((px["x"] - case["expected_x"]) ** 2 +
                       (px["y"] - case["expected_y"]) ** 2) ** 0.5
                ok = err <= args.tolerance
                correct += int(ok)
                false_loc += int(not ok and status == "MATCH_FOUND")
            else:
                ok = False

        rows.append((case["file"], status, res["confidence"],
                     res["feature_metrics"]["ransac_inliers"],
                     "-" if err is None else f"{err:6.1f}",
                     "PASS" if ok else "FAIL", f"{elapsed:5.1f}s"))

    width = max(len(r[0]) for r in rows) + 2
    print(f"{'case':<{width}}{'status':<16}{'conf':>6}{'inl':>6}{'err_px':>9}"
          f"{'result':>8}{'time':>8}")
    print("-" * (width + 53))
    for name, status, conf, inl, err, ok, t in rows:
        print(f"{name:<{width}}{status:<16}{conf:>6.2f}{inl:>6}{err:>9}{ok:>8}{t:>8}")

    print(f"\nTop-1 accuracy      : {correct}/{evaluated}")
    print(f"No-match detection  : {no_match_ok}/{no_match_total}")
    print(f"False localizations : {false_loc}")
    return 0 if (correct == evaluated and no_match_ok == no_match_total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
