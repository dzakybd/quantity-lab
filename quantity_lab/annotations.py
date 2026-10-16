"""Read a CubiCasa5K `model.svg`: room polygons (`Space <type>`), wall polygons, door and window counts.

Group `transform="matrix(...)"` attributes are applied, so multi-floor plans land in image pixel coordinates.
"""

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from shapely import affinity
from shapely.geometry import Polygon

from quantity_lab.geometry import Area, repair

NS = "{http://www.w3.org/2000/svg}"
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


@dataclass
class PlanRoom:
    id: str
    label: str
    polygon: Area
    repaired: bool
    outdoor: bool


@dataclass
class Plan:
    size: tuple[int, int]  # (width, height) in px, from the SVG
    rooms: list[PlanRoom] = field(default_factory=list)
    walls: list[Area] = field(default_factory=list)
    door_count: int = 0
    window_count: int = 0
    excluded: dict[str, int] = field(default_factory=dict)  # other classes seen, not used

    def indoor(self) -> list[PlanRoom]:
        return [r for r in self.rooms if not r.outdoor]


def _matrix(attr: str | None) -> tuple[float, ...]:
    if not attr:
        return IDENTITY
    m = re.match(r"\s*matrix\(([^)]*)\)", attr)
    if m:
        return tuple(float(v) for v in re.split(r"[ ,]+", m.group(1).strip()))
    t = re.match(r"\s*translate\(([^)]*)\)", attr)
    if t:
        v = [float(x) for x in re.split(r"[ ,]+", t.group(1).strip())]
        return (1.0, 0.0, 0.0, 1.0, v[0], v[1] if len(v) > 1 else 0.0)
    raise ValueError(f"unsupported transform: {attr}")


def _compose(outer, inner):
    a, b, c, d, e, f = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a * a2 + c * b2,
        b * a2 + d * b2,
        a * c2 + c * d2,
        b * c2 + d * d2,
        a * e2 + c * f2 + e,
        b * e2 + d * f2 + f,
    )


def _polygon(el: ET.Element, m) -> Polygon | None:
    poly = el.find(f"{NS}polygon")
    if poly is None:
        return None
    nums = [float(v) for v in re.split(r"[ ,]+", poly.get("points", "").strip()) if v]
    pts = list(zip(nums[::2], nums[1::2]))
    if len(pts) < 3:
        return None
    a, b, c, d, e, f = _compose(m, _matrix(poly.get("transform")))
    return affinity.affine_transform(Polygon(pts), [a, c, b, d, e, f])


def load_plan(svg: Path) -> Plan:
    if not Path(svg).exists():
        raise FileNotFoundError(svg)
    root = ET.parse(svg).getroot()
    plan = Plan(size=(math.ceil(float(root.get("width", 0))), math.ceil(float(root.get("height", 0)))))

    def walk(el: ET.Element, m):
        m = _compose(m, _matrix(el.get("transform")))
        cls = el.get("class", "")
        if cls.startswith("Space "):
            poly = _polygon(el, m)
            if poly is not None:
                fixed, repaired = repair(poly)
                label = cls.split()[1]
                plan.rooms.append(PlanRoom(el.get("id", ""), label, fixed, repaired, label == "Outdoor"))
        elif cls == "Wall" or cls.startswith("Wall "):
            poly = _polygon(el, m)
            if poly is not None:
                plan.walls.append(repair(poly)[0])
        elif cls.startswith("Door"):
            plan.door_count += 1
        elif cls.startswith("Window"):
            plan.window_count += 1
        elif cls and el.tag == f"{NS}g" and cls.split()[0] not in ("Model", "Floor", "Floorplan"):
            plan.excluded[cls.split()[0]] = plan.excluded.get(cls.split()[0], 0) + 1
        for child in el:
            walk(child, m)

    walk(root, IDENTITY)
    return plan
