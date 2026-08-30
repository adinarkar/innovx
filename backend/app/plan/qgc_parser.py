"""
QGroundControl ``.plan`` parsing (spec sections 5 and 33).

A .plan file is JSON describing a mission: a planned home position and a list
of mission items with lat/lon/altitude.  It carries **no imagery** - the
reference map always comes from a separate upload - so this module only
extracts mission metadata and, optionally, a suggested map extent the operator
can accept as a georeference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.logging_config import get_logger

log = get_logger(__name__)

DISCLAIMER = ("The .plan file contains mission coordinates. "
              "It does not contain satellite imagery.")

# MAV_CMD values whose params 5/6/7 are lat/lon/alt.
_NAV_COMMANDS = {16: "WAYPOINT", 17: "LOITER_UNLIM", 18: "LOITER_TURNS",
                 19: "LOITER_TIME", 21: "LAND", 22: "TAKEOFF", 82: "SPLINE_WAYPOINT",
                 189: "DO_LAND_START", 195: "ROI_LOCATION"}


class PlanError(ValueError):
    """Raised when a file is not a usable QGroundControl plan."""


@dataclass
class Waypoint:
    seq: int
    command: int
    command_name: str
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    frame: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "seq": self.seq, "command": self.command, "command_name": self.command_name,
            "latitude": self.latitude, "longitude": self.longitude,
            "altitude": self.altitude, "frame": self.frame,
        }


@dataclass
class MissionPlan:
    filename: str
    file_type: Optional[str] = None
    version: Optional[int] = None
    ground_station: Optional[str] = None
    vehicle_type: Optional[int] = None
    cruise_speed: Optional[float] = None
    hover_speed: Optional[float] = None
    planned_home_position: Optional[Dict[str, float]] = None
    waypoints: List[Waypoint] = field(default_factory=list)
    geofence_polygons: int = 0
    rally_points: int = 0
    warnings: List[str] = field(default_factory=list)

    # ----- derived -------------------------------------------------------
    @property
    def coordinates(self) -> List[List[float]]:
        return [[w.latitude, w.longitude] for w in self.waypoints
                if w.latitude is not None and w.longitude is not None]

    def bounds(self) -> Optional[Dict[str, float]]:
        """Lat/lon extent of the mission, if any coordinates were found."""
        coords = list(self.coordinates)
        if self.planned_home_position:
            coords.append([self.planned_home_position["latitude"],
                           self.planned_home_position["longitude"]])
        coords = [c for c in coords if c[0] is not None and c[1] is not None]
        if not coords:
            return None
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        return {"north": max(lats), "south": min(lats),
                "west": min(lons), "east": max(lons)}

    def suggested_georeference(self, padding: float = 0.25) -> Optional[Dict[str, float]]:
        """
        A padded mission bounding box the operator may adopt as the map extent.

        Offered as a *suggestion only*: the actual map image may cover a
        different area, so it never becomes a georeference automatically.
        """
        b = self.bounds()
        if not b:
            return None
        dlat = max((b["north"] - b["south"]) * padding, 1e-4)
        dlon = max((b["east"] - b["west"]) * padding, 1e-4)
        return {"north": round(b["north"] + dlat, 7), "south": round(b["south"] - dlat, 7),
                "west": round(b["west"] - dlon, 7), "east": round(b["east"] + dlon, 7)}

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "version": self.version,
            "ground_station": self.ground_station,
            "vehicle_type": self.vehicle_type,
            "cruise_speed": self.cruise_speed,
            "hover_speed": self.hover_speed,
            "planned_home_position": self.planned_home_position,
            "waypoints": [w.to_dict() for w in self.waypoints],
            "waypoint_count": len(self.waypoints),
            "coordinates": self.coordinates,
            "bounds": self.bounds(),
            "suggested_georeference": self.suggested_georeference(),
            "geofence_polygons": self.geofence_polygons,
            "rally_points": self.rally_points,
            "warnings": self.warnings,
            "disclaimer": DISCLAIMER,
        }


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f          # drop NaN


def _parse_items(items: List[dict], plan: MissionPlan, seq_offset: int = 0) -> None:
    """Walk mission items, recursing into ComplexItems (surveys, corridors)."""
    for idx, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "ComplexItem":
            nested = item.get("TransectStyleComplexItem", {}).get("Items") or item.get("Items")
            if nested:
                _parse_items(nested, plan, seq_offset + idx)
            else:
                plan.warnings.append(
                    f"Complex mission item '{item.get('complexItemType', 'unknown')}' "
                    "was not expanded.")
            continue

        params = item.get("params") or []
        lat = _num(params[4]) if len(params) > 4 else None
        lon = _num(params[5]) if len(params) > 5 else None
        alt = _num(params[6]) if len(params) > 6 else _num(item.get("Altitude"))
        cmd = int(item.get("command", 0) or 0)
        # QGC writes 0/0 for commands that carry no position.
        if lat == 0 and lon == 0:
            lat = lon = None
        plan.waypoints.append(Waypoint(
            seq=int(item.get("doJumpId", seq_offset + idx + 1)),
            command=cmd, command_name=_NAV_COMMANDS.get(cmd, f"CMD_{cmd}"),
            latitude=lat, longitude=lon, altitude=alt,
            frame=item.get("frame"),
        ))


def parse_plan(path: Path) -> MissionPlan:
    """Parse a .plan file, raising :class:`PlanError` with a useful message."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"Not valid JSON ({exc.msg} at line {exc.lineno}). "
                        "A QGroundControl .plan file is a JSON document.") from exc
    except OSError as exc:
        raise PlanError(f"Could not read the file: {exc}") from exc

    if not isinstance(raw, dict):
        raise PlanError("The .plan file must contain a JSON object at the top level.")

    plan = MissionPlan(filename=path.name,
                       file_type=raw.get("fileType"),
                       version=raw.get("version"),
                       ground_station=raw.get("groundStation"))

    mission = raw.get("mission")
    if not isinstance(mission, dict):
        raise PlanError("No 'mission' section found - this does not look like a "
                        "QGroundControl plan.")

    plan.vehicle_type = mission.get("vehicleType")
    plan.cruise_speed = _num(mission.get("cruiseSpeed"))
    plan.hover_speed = _num(mission.get("hoverSpeed"))

    home = mission.get("plannedHomePosition")
    if isinstance(home, list) and len(home) >= 3:
        plan.planned_home_position = {
            "latitude": _num(home[0]), "longitude": _num(home[1]),
            "altitude": _num(home[2]),
        }
    else:
        plan.warnings.append("No plannedHomePosition in this plan.")

    _parse_items(mission.get("items") or [], plan)
    if not plan.waypoints:
        plan.warnings.append("The mission contains no waypoint items.")

    fence = raw.get("geoFence") or {}
    plan.geofence_polygons = len(fence.get("polygons") or [])
    plan.rally_points = len((raw.get("rallyPoints") or {}).get("points") or [])

    log.info("Parsed %s: %d waypoints, home=%s",
             path.name, len(plan.waypoints), bool(plan.planned_home_position))
    return plan
