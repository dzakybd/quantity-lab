"""Room geometry: measure, repair, and the raster round trip (polygons -> label mask -> polygons).

Coordinates are image pixels: pixel (row r, col c) covers the square [c, c+1] x [r, r+1].
"""

import cv2
import numpy as np
from shapely import contains_xy, make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

Area = Polygon | MultiPolygon


def measure(geom: Area) -> tuple[float, float]:
    """Area in px^2 and perimeter in px (exterior plus hole boundaries)."""
    return float(geom.area), float(geom.length)


def repair(geom: BaseGeometry) -> tuple[Area, bool]:
    """Make a geometry valid, keeping only its polygonal parts. Returns (geometry, was_repaired)."""
    if geom.is_valid and isinstance(geom, (Polygon, MultiPolygon)):
        return geom, False
    fixed = make_valid(geom)
    if isinstance(fixed, GeometryCollection):
        parts = [g for g in fixed.geoms if isinstance(g, (Polygon, MultiPolygon))]
        polys = [p for g in parts for p in (g.geoms if isinstance(g, MultiPolygon) else [g])]
        fixed = MultiPolygon(polys) if len(polys) > 1 else (polys[0] if polys else Polygon())
    return fixed, True


def rasterize(rooms: list[Area], size: tuple[int, int]) -> np.ndarray:
    """Label mask (H, W) int32: 0 = no room, i + 1 = rooms[i]. A pixel belongs to a room when its centre is inside.
    `size` is (width, height). Later rooms win overlaps."""
    w, h = size
    mask = np.zeros((h, w), np.int32)
    for i, room in enumerate(rooms, 1):
        if room.is_empty:
            continue
        x0, y0, x1, y1 = room.bounds
        c0, r0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        c1, r1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        if c1 <= c0 or r1 <= r0:
            continue
        xs, ys = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5)
        inside = contains_xy(room, xs, ys)
        mask[r0:r1, c0:c1][inside] = i
    return mask


def polygonize(mask: np.ndarray, min_area: float = 1.0) -> dict[int, Area]:
    """Polygon per label (holes kept), traced on pixel boundaries so a filled w x h block has area w*h."""
    out = {}
    for label in np.unique(mask):
        if label == 0:
            continue
        binary = (mask == label).astype(np.uint8)
        contours, hier = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        polys = []
        for i, c in enumerate(contours):
            if hier[0][i][3] != -1:  # a hole; added with its parent
                continue
            shell = c[:, 0, :].astype(float) + 0.5
            holes = [contours[j][:, 0, :].astype(float) + 0.5 for j in range(len(contours)) if hier[0][j][3] == i]
            if len(shell) < 3:
                shell = np.array(
                    [
                        [shell[0][0] - 0.5, shell[0][1] - 0.5],
                        [shell[0][0] + 0.5, shell[0][1] - 0.5],
                        [shell[0][0] + 0.5, shell[0][1] + 0.5],
                        [shell[0][0] - 0.5, shell[0][1] + 0.5],
                    ]
                )
                polys.append(Polygon(shell))
                continue
            # contour runs through pixel centres; grow by half a pixel to reach pixel boundaries
            p = Polygon(shell, [h for h in holes if len(h) >= 3]).buffer(0.5, join_style="mitre")
            polys.extend(p.geoms if isinstance(p, MultiPolygon) else [p])
        polys = [p for p in polys if p.area >= min_area]
        if polys:
            out[int(label)] = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    return out
