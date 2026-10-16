"""Synthetic floor plans in CubiCasa5K's format: a PNG drawing plus a model.svg with Space and Wall polygons.

Used for tests and for every public picture (CubiCasa5K images are never published). Each plan is a set of
rectangular rooms on a grid, with thick walls, doorways with swing arcs, thin furniture lines and room labels.
`open_pairs` lists rooms that share an opening with no wall, like an open kitchen next to a living room.

Usage: uv run python -m eval.synth runs/inputs/synth
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WALL = 10  # px
FONT = "/System/Library/Fonts/Helvetica.ttc"


@dataclass
class SynthPlan:
    name: str
    size: tuple[int, int]
    rooms: list[tuple[str, tuple[int, int, int, int]]]  # label, (x0, y0, x1, y1) of the room's inner face
    doors: list[tuple[int, int, str]] = field(default_factory=list)  # (x, y, "h"|"v") centre of a doorway
    open_pairs: list[tuple[int, int]] = field(default_factory=list)  # room indices with no wall between them


def _plans() -> list[SynthPlan]:
    w = WALL
    simple = SynthPlan(
        "synth-simple",
        (900, 620),
        [
            ("Bedroom", (60, 60, 420, 560)),
            ("LivingRoom", (420 + w, 60, 840, 330)),
            ("Bath", (420 + w, 330 + w, 840, 560)),
        ],
        doors=[(425, 200, "v"), (630, 335, "h")],
    )
    medium = SynthPlan(
        "synth-medium",
        (1100, 760),
        [
            ("Bedroom", (60, 60, 380, 360)),
            ("Bedroom", (380 + w, 60, 700, 360)),
            ("Bath", (700 + w, 60, 1040, 260)),
            ("Kitchen", (60, 360 + w, 420, 700)),
            ("LivingRoom", (420, 360 + w, 1040, 700)),  # open to the kitchen: no wall at x = 420
            ("Entry", (700 + w, 260 + w, 1040, 360)),
        ],
        doors=[(220, 365, "h"), (540, 365, "h"), (870, 265, "h"), (705, 310, "v"), (870, 365, "h")],
        open_pairs=[(3, 4)],
    )
    rooms, doors = [], []
    xs, ys = [60, 330, 600, 870, 1140], [60, 330, 600, 870]
    for j in range(3):
        for i in range(4):
            x0, y0 = xs[i] + (w if i else 0), ys[j] + (w if j else 0)
            label = ["Bedroom", "Bath", "Office", "Storage"][(i + j) % 4]
            rooms.append((label, (x0, y0, xs[i + 1], ys[j + 1])))
            if i < 3:
                doors.append((xs[i + 1] + w // 2, ys[j] + 135, "v"))
            if j < 2 and i % 2 == 0:
                doors.append((xs[i] + 135, ys[j + 1] + w // 2, "h"))
    dense = SynthPlan("synth-dense", (1200, 930), rooms, doors=doors)
    return [simple, medium, dense]


def _walls(plan: SynthPlan) -> list[tuple[int, int, int, int]]:
    """Wall rectangles: every room edge thickened outwards by WALL, except shared open edges."""
    out = []
    for i, (_, (x0, y0, x1, y1)) in enumerate(plan.rooms):
        out += [(x0 - WALL, y0 - WALL, x1 + WALL, y0), (x0 - WALL, y1, x1 + WALL, y1 + WALL)]
        left_open = any(i == b and plan.rooms[a][1][2] == x0 for a, b in plan.open_pairs)
        right_open = any(i == a and plan.rooms[b][1][0] == x1 for a, b in plan.open_pairs)
        if not left_open:
            out.append((x0 - WALL, y0, x0, y1))
        if not right_open:
            out.append((x1, y0, x1 + WALL, y1))
    return out


def render(plan: SynthPlan, out_dir: Path) -> Path:
    d = out_dir / plan.name
    d.mkdir(parents=True, exist_ok=True)
    im = Image.new("L", plan.size, 255)
    dr = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in _walls(plan):
        dr.rectangle((x0, y0, x1 - 1, y1 - 1), fill=30)
    for x, y, o in plan.doors:  # doorway: clear 70 px of wall, draw a thin leaf and swing
        if o == "h":
            dr.rectangle((x - 35, y - WALL, x + 35, y + WALL), fill=255)
            dr.line((x - 35, y, x - 35, y - 70), fill=60, width=2)
            dr.arc((x - 105, y - 70, x + 35, y + 70), 270, 360, fill=60, width=2)
        else:
            dr.rectangle((x - WALL, y - 35, x + WALL, y + 35), fill=255)
            dr.line((x, y - 35, x + 70, y - 35), fill=60, width=2)
            dr.arc((x - 70, y - 35, x + 70, y + 105), 0, 90, fill=60, width=2)
    font = ImageFont.truetype(FONT, 22)
    for k, (label, (x0, y0, x1, y1)) in enumerate(plan.rooms):
        dr.text(((x0 + x1) / 2 - 40, (y0 + y1) / 2 - 10), label.upper(), fill=40, font=font)
        if label in ("Bedroom", "Office"):  # a bed or desk outline: thin lines, not walls
            dr.rectangle((x0 + 30, y0 + 30, x0 + 130, y0 + 110), outline=70, width=2)
        if label == "Kitchen":  # a counter, thicker than furniture lines
            dr.rectangle((x0 + 10, y0 + 10, x1 - 10, y0 + 50), outline=50, width=3)
    for a, b in plan.open_pairs:  # the open boundary is often drawn as a thin dashed line
        x = plan.rooms[a][1][2]
        y0, y1 = plan.rooms[b][1][1], plan.rooms[b][1][3]
        for y in range(y0, y1, 24):
            dr.line((x, y, x, min(y + 12, y1)), fill=120, width=1)
    im.save(d / "F1_scaled.png")
    (d / "model.svg").write_text(_svg(plan))
    return d


def _svg(plan: SynthPlan) -> str:
    w, h = plan.size
    parts = []
    for k, (label, (x0, y0, x1, y1)) in enumerate(plan.rooms):
        pts = f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"
        parts.append(f'<g id="s{k}" class="Space {label}" fill="#ffffff"><polygon points="{pts}"/></g>')
    for k, (x0, y0, x1, y1) in enumerate(_walls(plan)):
        parts.append(f'<g id="w{k}" class="Wall"><polygon points="{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"/></g>')
    for k, _ in enumerate(plan.doors):
        parts.append(f'<g id="d{k}" class="Door Swing Beside"></g>')
    body = "\n".join(parts)
    return (
        f'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" height="{h}" width="{w}" version="1.1">'
        f'<g id="Model" class="Model v1-1"><g class="Floor"><g id="Floor-1" class="Floorplan Floor-1">\n{body}\n'
        "</g></g></g></svg>\n"
    )


def make_synthetic(out_dir: Path) -> list[Path]:
    return [render(p, out_dir) for p in _plans()]


if __name__ == "__main__":
    print("\n".join(str(p) for p in make_synthetic(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/inputs/synth"))))
