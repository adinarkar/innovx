"""
Multi-scale overlapping tiling of the reference map (spec section 7).

The drone frame typically covers 10-20% of the map area, so instead of
squashing the whole map down to the drone resolution we cut candidate windows
whose *area* is a fixed fraction of the map area, at several scales, with 25%
overlap between neighbours.  Every tile keeps its map-space geometry so a
homography solved in tile space can be lifted back to map pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

from app.config import settings
from app.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class Tile:
    tile_id: int
    x: int
    y: int
    width: int
    height: int
    scale: float          # fraction of the map AREA this tile covers

    def to_dict(self) -> dict:
        d = asdict(self)
        d["center"] = [self.x + self.width // 2, self.y + self.height // 2]
        return d

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y:self.y + self.height, self.x:self.x + self.width]

    def polygon(self) -> List[List[int]]:
        return [
            [self.x, self.y],
            [self.x + self.width, self.y],
            [self.x + self.width, self.y + self.height],
            [self.x, self.y + self.height],
        ]


def _positions(total: int, window: int, overlap: float) -> List[int]:
    """Start offsets covering ``total`` with ``overlap`` fraction shared."""
    if window >= total:
        return [0]
    step = max(1, int(round(window * (1.0 - overlap))))
    pos = list(range(0, total - window + 1, step))
    if pos[-1] != total - window:
        pos.append(total - window)
    return pos


def plan_tiles(map_w: int, map_h: int,
               scales: List[float] | None = None,
               overlap: float | None = None,
               max_tiles: int | None = None) -> List[Tile]:
    """
    Build the tile grid.  Tiles are square (drone frames have no guaranteed
    orientation), sized so that ``w*h ~= scale * map_area``.
    """
    scales = scales or settings.tile_scales
    overlap = settings.tile_overlap if overlap is None else overlap
    max_tiles = max_tiles or settings.max_tiles

    map_area = float(map_w * map_h)
    shortest = min(map_w, map_h)

    tiles: List[Tile] = []
    tid = 0
    for scale in sorted(scales):
        side = int(round(math.sqrt(scale * map_area)))
        side = max(64, min(side, shortest))          # never exceed the map
        for y in _positions(map_h, side, overlap):
            for x in _positions(map_w, side, overlap):
                tiles.append(Tile(tid, int(x), int(y), side, side, float(scale)))
                tid += 1

    if len(tiles) > max_tiles:
        # Keep the grid statistically uniform by evenly subsampling rather than
        # truncating (truncation would drop entire scales).
        idx = np.linspace(0, len(tiles) - 1, max_tiles).round().astype(int)
        tiles = [tiles[i] for i in sorted(set(idx.tolist()))]
        for new_id, t in enumerate(tiles):
            t.tile_id = new_id
        log.info("Tile budget hit - subsampled to %d tiles.", len(tiles))

    log.info("Planned %d tiles for a %dx%d map across %d scales.",
             len(tiles), map_w, map_h, len(scales))
    return tiles


def tile_to_map(tile: Tile, pts: np.ndarray, tile_render_scale: float) -> np.ndarray:
    """
    Lift points from *rendered tile* pixel space into map pixel space.

    ``tile_render_scale`` is the factor the native-resolution crop was resized
    by before feature extraction.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    out = pts / max(tile_render_scale, 1e-9)
    out[:, 0] += tile.x
    out[:, 1] += tile.y
    return out


def map_bounds_of(tiles: List[Tile]) -> Tuple[int, int, int, int]:
    xs = [t.x for t in tiles] + [t.x + t.width for t in tiles]
    ys = [t.y for t in tiles] + [t.y + t.height for t in tiles]
    return min(xs), min(ys), max(xs), max(ys)
