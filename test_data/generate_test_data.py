"""
Synthetic test-data generator for innovX VisualNav.

Builds a procedural "satellite" reference map (road grid, buildings, fields,
water) plus the eight drone-capture variants from the testing strategy in the
build spec, and writes ``ground_truth.json`` with the true centre of each crop
so the Developer Mode page can score them.

    python test_data/generate_test_data.py --out test_data/generated
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

RNG = np.random.default_rng(20240817)


# --------------------------------------------------------------------------
def build_reference_map(width: int = 3000, height: int = 3000) -> np.ndarray:
    """Procedural aerial scene: terrain, fields, a road grid and buildings."""
    # Base terrain from smoothed noise so the map has broad tonal structure.
    coarse = RNG.random((height // 40, width // 40)).astype(np.float32)
    terrain = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    terrain = cv2.GaussianBlur(terrain, (0, 0), 12)
    img = np.dstack([
        (70 + terrain * 55), (95 + terrain * 60), (85 + terrain * 50)
    ]).astype(np.uint8)

    # Agricultural blocks.
    for _ in range(38):
        w = int(RNG.integers(140, 460)); h = int(RNG.integers(140, 460))
        x = int(RNG.integers(0, width - w)); y = int(RNG.integers(0, height - h))
        tone = RNG.integers(-28, 34, size=3)
        patch = img[y:y + h, x:x + w].astype(np.int16) + tone
        img[y:y + h, x:x + w] = np.clip(patch, 0, 255).astype(np.uint8)
        cv2.rectangle(img, (x, y), (x + w, y + h), (60, 78, 66), 2, cv2.LINE_AA)

    # Water body - a large low-texture region that should NOT attract matches.
    cv2.ellipse(img, (int(width * 0.78), int(height * 0.2)),
                (int(width * 0.11), int(height * 0.07)), 25, 0, 360,
                (128, 96, 62), cv2.FILLED, cv2.LINE_AA)

    # Road grid with slight irregularity.
    road = (58, 58, 58)
    for i in range(1, 9):
        y = int(height * i / 9 + RNG.integers(-45, 45))
        cv2.line(img, (0, y), (width, y + int(RNG.integers(-30, 30))), road,
                 int(RNG.integers(9, 20)), cv2.LINE_AA)
    for i in range(1, 9):
        x = int(width * i / 9 + RNG.integers(-45, 45))
        cv2.line(img, (x, 0), (x + int(RNG.integers(-30, 30)), height), road,
                 int(RNG.integers(9, 20)), cv2.LINE_AA)
    # A diagonal arterial gives the scene a unique large-scale cue.
    cv2.line(img, (0, int(height * 0.86)), (width, int(height * 0.12)),
             (48, 48, 48), 22, cv2.LINE_AA)

    # Buildings: rotated rectangles with roof tone variation and shadows.
    for _ in range(900):
        w = int(RNG.integers(18, 90)); h = int(RNG.integers(18, 90))
        cx = int(RNG.integers(w, width - w)); cy = int(RNG.integers(h, height - h))
        angle = float(RNG.integers(0, 90))
        box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.int32)
        roof = tuple(int(v) for v in RNG.integers(120, 225, size=3))
        shadow = box + np.array([6, 8])
        cv2.fillPoly(img, [shadow], (45, 45, 45), cv2.LINE_AA)
        cv2.fillPoly(img, [box], roof, cv2.LINE_AA)
        cv2.polylines(img, [box], True, (70, 70, 70), 1, cv2.LINE_AA)

    # Fine sensor-like grain keeps local descriptors realistic.
    noise = RNG.normal(0, 4.0, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
def crop_fraction(img: np.ndarray, fraction: float, cx: float, cy: float) -> tuple:
    """Crop a square covering ``fraction`` of the map area, centred at cx, cy."""
    h, w = img.shape[:2]
    side = int(round(math.sqrt(fraction * w * h)))
    side = min(side, h, w)
    x = int(np.clip(cx - side / 2, 0, w - side))
    y = int(np.clip(cy - side / 2, 0, h - side))
    return img[y:y + side, x:x + side].copy(), (x + side // 2, y + side // 2)


def rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the centre, cropping to the largest inscribed square."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    side = int(min(w, h) / math.sqrt(2))
    x = (w - side) // 2
    y = (h - side) // 2
    return rotated[y:y + side, x:x + side]


def perspective(img: np.ndarray, strength: float = 0.10) -> np.ndarray:
    h, w = img.shape[:2]
    d = strength * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[d, d * 0.5], [w - d * 0.6, 0], [w, h - d], [d * 0.4, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderMode=cv2.BORDER_REFLECT_101)


def unrelated_image(size: int = 700) -> np.ndarray:
    """A scene with plenty of structure that is absent from the map."""
    img = np.full((size, size, 3), 200, np.uint8)
    for _ in range(60):
        c = tuple(int(v) for v in RNG.integers(0, 255, size=3))
        p1 = tuple(int(v) for v in RNG.integers(0, size, size=2))
        p2 = tuple(int(v) for v in RNG.integers(0, size, size=2))
        cv2.rectangle(img, p1, p2, c, -1, cv2.LINE_AA)
    for _ in range(40):
        cv2.circle(img, tuple(int(v) for v in RNG.integers(0, size, size=2)),
                   int(RNG.integers(10, 70)),
                   tuple(int(v) for v in RNG.integers(0, 255, size=3)), 3, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------
def build_plan(center_lat: float, center_lon: float, out: Path) -> None:
    """A minimal but valid QGroundControl .plan for the mission-metadata demo."""
    def item(seq, lat, lon, alt):
        return {"AMSLAltAboveTerrain": None, "Altitude": alt, "AltitudeMode": 1,
                "autoContinue": True, "command": 16, "doJumpId": seq, "frame": 3,
                "params": [0, 0, 0, None, lat, lon, alt], "type": "SimpleItem"}

    d = 0.0035
    plan = {
        "fileType": "Plan", "version": 1, "groundStation": "QGroundControl",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "rallyPoints": {"points": [], "version": 2},
        "mission": {
            "cruiseSpeed": 15, "hoverSpeed": 5, "firmwareType": 12, "vehicleType": 2,
            "globalPlanAltitudeMode": 1, "version": 2,
            "plannedHomePosition": [center_lat - d, center_lon - d, 512.0],
            "items": [
                item(1, center_lat - d, center_lon - d, 120),
                item(2, center_lat - d, center_lon + d, 120),
                item(3, center_lat + d, center_lon + d, 120),
                item(4, center_lat + d, center_lon - d, 120),
                item(5, center_lat, center_lon, 100),
            ],
        },
    }
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate innovX VisualNav test data")
    ap.add_argument("--out", default=str(Path(__file__).parent / "generated"))
    ap.add_argument("--size", type=int, default=3000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Building reference map ...")
    ref = build_reference_map(args.size, args.size)
    cv2.imwrite(str(out / "reference_map.jpg"), ref, [cv2.IMWRITE_JPEG_QUALITY, 92])

    h, w = ref.shape[:2]
    cx, cy = int(w * 0.62), int(h * 0.41)
    base, truth = crop_fraction(ref, 0.15, cx, cy)

    cases = []

    def emit(name: str, img: np.ndarray, center, expect_no_match=False, note=""):
        cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cases.append({"file": f"{name}.jpg", "note": note,
                      "expected_x": None if expect_no_match else int(center[0]),
                      "expected_y": None if expect_no_match else int(center[1]),
                      "expect_no_match": expect_no_match})
        print(f"  {name}.jpg  {img.shape[1]}x{img.shape[0]}  {note}")

    print("Building drone captures ...")
    emit("test1_direct_crop", base, truth, note="Direct 15% crop")
    emit("test2_rotated_30", rotate(base, 30), truth, note="Rotated 30 degrees")
    emit("test3_rotated_90", rotate(base, 90), truth, note="Rotated 90 degrees")
    emit("test4_brightness",
         cv2.convertScaleAbs(base, alpha=1.28, beta=34), truth,
         note="Brightness and contrast shift")
    emit("test5_blur", cv2.GaussianBlur(base, (0, 0), 2.2), truth,
         note="Mild Gaussian blur")
    emit("test6_resized",
         cv2.resize(base, (int(base.shape[1] * 0.45), int(base.shape[0] * 0.45)),
                    interpolation=cv2.INTER_AREA), truth,
         note="Significant downscale")
    emit("test7_perspective", perspective(base), truth, note="Perspective distortion")
    emit("test8_unrelated", unrelated_image(), (0, 0), expect_no_match=True,
         note="Unrelated scene - expect NO_MATCH")

    # A second, different location to exercise repeat runs.
    other, other_truth = crop_fraction(ref, 0.12, int(w * 0.25), int(h * 0.72))
    emit("test9_other_area", other, other_truth, note="Different 12% region")

    build_plan(12.9670, 77.5980, out / "mission.plan")

    (out / "ground_truth.json").write_text(json.dumps({
        "reference_map": "reference_map.jpg",
        "map_width": w, "map_height": h,
        "georeference_hint": {"north": 12.9760, "south": 12.9580,
                              "west": 77.5880, "east": 77.6080},
        "cases": cases,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {len(cases) + 2} files to {out.resolve()}")
    print("Suggested georeference for the demo: "
          "N 12.9760  S 12.9580  W 77.5880  E 77.6080")


if __name__ == "__main__":
    main()
