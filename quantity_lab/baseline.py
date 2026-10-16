"""Predicted rooms without a model: find thick dark strokes (walls), close door openings, label the free space.

Classical computer vision only (OpenCV). Rooms with no wall between them, such as an open kitchen next to a
living room, cannot be told apart this way.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BaselineParams:
    threshold: int = 140  # gray level below which a pixel is ink
    wall_px: int = 5  # strokes thinner than this are not walls (opening with a wall_px square)
    door_px: int = 60  # wall gaps up to this long are closed (doorways), along rows and columns
    min_room_px2: int = 2000  # free-space pieces smaller than this are not rooms
    work_long_side: int = 1500  # segment at this long side (0 = native); plans come at very different resolutions

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineParams":
        return cls(**d)


def wall_mask(gray: np.ndarray, p: BaselineParams) -> np.ndarray:
    """Thick strokes, with doorways closed. Gaps are closed only between collinear wall pieces (horizontal strokes
    along rows, vertical strokes along columns), so a narrow corridor between two walls stays open."""
    ink = (gray < p.threshold).astype(np.uint8)
    walls = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((p.wall_px, p.wall_px), np.uint8))
    run = 4 * p.wall_px
    for along, across in (
        (np.ones((1, run), np.uint8), np.ones((1, p.door_px), np.uint8)),
        (np.ones((run, 1), np.uint8), np.ones((p.door_px, 1), np.uint8)),
    ):
        strokes = cv2.morphologyEx(walls, cv2.MORPH_OPEN, along)
        walls = walls | cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, across)
    return walls


def segment_rooms(gray: np.ndarray, p: BaselineParams) -> np.ndarray:
    """Room label mask (0 = not a room), same size as `gray`. Free-space pieces touching the border are outside."""
    h, w = gray.shape
    f = p.work_long_side / max(h, w) if p.work_long_side else 1.0
    work = cv2.resize(gray, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f != 1.0 else gray
    free = (wall_mask(work, p) == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    wh, ww = work.shape
    out = np.zeros((wh, ww), np.int32)
    k = 1
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < p.min_room_px2 or x == 0 or y == 0 or x + bw == ww or y + bh == wh:
            continue
        out[lab == i] = k
        k += 1
    if f != 1.0:
        out = cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return out
