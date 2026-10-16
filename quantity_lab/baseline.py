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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineParams":
        return cls(**d)


def wall_mask(gray: np.ndarray, p: BaselineParams) -> np.ndarray:
    ink = (gray < p.threshold).astype(np.uint8)
    walls = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((p.wall_px, p.wall_px), np.uint8))
    for k in (np.ones((1, p.door_px), np.uint8), np.ones((p.door_px, 1), np.uint8)):
        walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, k)
    return walls


def segment_rooms(gray: np.ndarray, p: BaselineParams) -> np.ndarray:
    """Room label mask (0 = not a room). Free-space pieces touching the image border are outside."""
    free = (wall_mask(gray, p) == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    h, w = gray.shape
    out = np.zeros((h, w), np.int32)
    k = 1
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < p.min_room_px2 or x == 0 or y == 0 or x + bw == w or y + bh == h:
            continue
        out[lab == i] = k
        k += 1
    return out
