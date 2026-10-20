"""Draw each held-out CubiCasa5K plan as a vector PDF from its annotation, with the rooms as an answer key.

For floorplan-parser's lines -> walls -> rooms step, which reads vector PDFs: CubiCasa5K's drawings are raster,
but its annotations hold real wall, door and room polygons. Walls are drawn the CAD way, as their outlines, with
the door openings cut out and a quarter-circle swing at each door; nothing else is drawn. One PDF point is one image
pixel. This tests the step from walls to rooms on real layouts, not the reading of messy CAD linework.

Writes runs/inputs/vector[-val]/<plan>/plan.pdf and rooms.json (gitignored: derived from CubiCasa5K).

Usage: uv run python -m eval.export_vector [--val]
"""

import itertools
import json
import math

import pymupdf
from shapely import union_all

from eval.common import INPUTS, TEST, plan_dir, val_plans
from quantity_lab.annotations import load_plan


def export(plan: str, folder: str = "vector") -> dict:
    d = plan_dir(plan)
    p = load_plan(d / "model.svg")
    walls = union_all(p.walls)
    if p.doors:
        walls = walls.difference(union_all([g.buffer(0.5, join_style="mitre") for g in p.doors]))
    out = INPUTS / folder / plan.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    w, h = p.size
    page = doc.new_page(width=w, height=h)
    for poly in getattr(walls, "geoms", [walls]):
        for ring in [poly.exterior, *poly.interiors]:
            c = list(ring.coords)
            for a, b in itertools.pairwise(c):
                page.draw_line(a, b, width=0.5)
    for door in p.doors:  # a swing arc on the long side of each opening, as CAD draws doors
        rect = door.minimum_rotated_rectangle
        pts = list(rect.exterior.coords)[:4]
        sides = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        (a, b) = max(sides, key=lambda s: math.dist(*s))
        r = math.dist(a, b)
        ux, uy = (b[0] - a[0]) / r, (b[1] - a[1]) / r
        arc = [(a[0] + r * (ux * math.cos(t) - uy * math.sin(t)), a[1] + r * (uy * math.cos(t) + ux * math.sin(t)))
               for t in [k * math.pi / 2 / 24 for k in range(25)]]  # fmt: skip
        page.draw_polyline(arc, width=0.3)
        page.draw_line(a, arc[-1], width=0.3)
    doc.save(out / "plan.pdf")
    rooms = [r.polygon.wkt for r in p.indoor()]  # WKT, so multi-part rooms survive
    info = {"plan": plan, "size_pt": [w, h], "rooms": rooms, "walls": len(p.walls), "doors_cut": len(p.doors)}
    (out / "rooms.json").write_text(json.dumps(info))
    return info


def main() -> None:
    import sys

    val = "--val" in sys.argv  # the validation plans, for choosing floorplan-parser's room rules
    for plan in val_plans() if val else TEST:
        info = export(plan, "vector-val" if val else "vector")
        print(plan, "rooms", len(info["rooms"]), "doors cut", info["doors_cut"])


if __name__ == "__main__":
    main()
