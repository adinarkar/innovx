"""
Optional georeferencing of map pixels (spec section 32).

GPS is *never* invented.  A geographic coordinate is only produced when the
operator has supplied a georeference for the reference map, either as a
north/south/west/east bounding box or as four explicit corner coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from app.logging_config import get_logger

log = get_logger(__name__)


class GeoreferenceError(ValueError):
    """Raised for georeference input that cannot describe a real map extent."""


@dataclass
class Georeference:
    """
    Either an axis-aligned lat/lon box or a full 4-corner (homographic) fit.

    ``corners`` are ordered TL, TR, BR, BL in *map pixel* order and each entry
    is ``[latitude, longitude]``.
    """
    kind: str                                    # "bbox" | "corners"
    north: Optional[float] = None
    south: Optional[float] = None
    west: Optional[float] = None
    east: Optional[float] = None
    corners: Optional[List[List[float]]] = None
    width: int = 0
    height: int = 0

    # ----- construction --------------------------------------------------
    @staticmethod
    def from_bbox(north: float, south: float, west: float, east: float,
                  width: int, height: int) -> "Georeference":
        for name, value, lo, hi in (("north", north, -90, 90), ("south", south, -90, 90),
                                    ("west", west, -180, 180), ("east", east, -180, 180)):
            if value is None or not np.isfinite(value):
                raise GeoreferenceError(f"{name} latitude/longitude is missing or not a number.")
            if not (lo <= float(value) <= hi):
                raise GeoreferenceError(f"{name} must be between {lo} and {hi}, got {value}.")
        if float(north) <= float(south):
            raise GeoreferenceError("North latitude must be greater than south latitude.")
        if float(east) == float(west):
            raise GeoreferenceError("East and west longitude must differ.")
        if width <= 0 or height <= 0:
            raise GeoreferenceError("Map dimensions are unknown - upload the map first.")
        return Georeference("bbox", float(north), float(south), float(west), float(east),
                            None, int(width), int(height))

    @staticmethod
    def from_corners(corners: Sequence[Sequence[float]], width: int, height: int) -> "Georeference":
        pts = [[float(a), float(b)] for a, b in corners]
        if len(pts) != 4:
            raise GeoreferenceError("Exactly four corner coordinates are required (TL, TR, BR, BL).")
        for lat, lon in pts:
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                raise GeoreferenceError(f"Corner ({lat}, {lon}) is outside valid lat/lon range.")
        if width <= 0 or height <= 0:
            raise GeoreferenceError("Map dimensions are unknown - upload the map first.")
        return Georeference("corners", corners=pts, width=int(width), height=int(height))

    # ----- projection ----------------------------------------------------
    def _pixel_corners(self) -> np.ndarray:
        w, h = float(self.width), float(self.height)
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)

    def _homography(self) -> np.ndarray:
        """Pixel -> (lon, lat) transform fitted through the four corners."""
        dst = np.array([[lon, lat] for lat, lon in self.corners], dtype=np.float64)
        H, _ = cv2.findHomography(self._pixel_corners().reshape(-1, 1, 2),
                                  dst.reshape(-1, 1, 2), 0)
        if H is None:
            raise GeoreferenceError("The four corner coordinates are degenerate.")
        return H

    def pixel_to_latlon(self, x: float, y: float) -> dict:
        """Convert one map pixel to latitude/longitude."""
        if self.kind == "bbox":
            u = float(x) / max(self.width, 1)
            v = float(y) / max(self.height, 1)
            lat = self.north + (self.south - self.north) * v
            lon = self.west + (self.east - self.west) * u
        else:
            pt = cv2.perspectiveTransform(
                np.array([[[float(x), float(y)]]], dtype=np.float64), self._homography())
            lon, lat = float(pt[0, 0, 0]), float(pt[0, 0, 1])
        return {"latitude": round(lat, 7), "longitude": round(lon, 7)}

    def polygon_to_latlon(self, poly: Sequence[Sequence[float]]) -> List[dict]:
        return [self.pixel_to_latlon(x, y) for x, y in poly]

    def ground_sample_distance(self) -> Optional[float]:
        """
        Approximate metres per pixel, useful context for the operator.

        Uses a local equirectangular approximation, which is accurate enough
        over the few kilometres a prototype reference map covers.
        """
        if self.kind != "bbox":
            return None
        mid_lat = np.radians((self.north + self.south) / 2.0)
        lat_span_m = abs(self.north - self.south) * 111_320.0
        lon_span_m = abs(self.east - self.west) * 111_320.0 * float(np.cos(mid_lat))
        if self.width <= 0 or self.height <= 0:
            return None
        return round(float(np.mean([lat_span_m / self.height, lon_span_m / self.width])), 4)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "north": self.north, "south": self.south,
            "west": self.west, "east": self.east,
            "corners": self.corners,
            "width": self.width, "height": self.height,
            "ground_sample_distance_m": self.ground_sample_distance(),
        }

    @staticmethod
    def from_dict(data: Optional[dict]) -> Optional["Georeference"]:
        if not data:
            return None
        try:
            if data.get("kind") == "corners":
                return Georeference.from_corners(data["corners"], data["width"], data["height"])
            return Georeference.from_bbox(data["north"], data["south"], data["west"],
                                          data["east"], data["width"], data["height"])
        except Exception as exc:
            log.warning("Discarding stored georeference: %s", exc)
            return None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres - used to relate results to .plan waypoints."""
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))
