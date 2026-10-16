"""Controlled errors on a room label mask (0 = not a room, k = room k). Each is a pure function of its inputs."""

import cv2
import numpy as np
from scipy import ndimage


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    """Pull every room boundary in by `px` pixels (rooms come out too small)."""
    out = np.zeros_like(mask)
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    for label in np.unique(mask[mask > 0]):
        out[cv2.erode((mask == label).astype(np.uint8), k) > 0] = label
    return out


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Push room boundaries out by `px` pixels into non-room pixels (walls, outside); rooms never overwrite rooms."""
    if not mask.any():
        return mask.copy()
    dist, (ri, ci) = ndimage.distance_transform_edt(mask == 0, return_indices=True)
    out = mask.copy()
    grow = (mask == 0) & (dist <= px)
    out[grow] = mask[ri[grow], ci[grow]]
    return out


def shift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate the whole mask by (dx, dy) pixels; pixels moved in from the edge are empty."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    out[yd, xd] = mask[ys, xs]
    return out


def scale(mask: np.ndarray, s: float) -> np.ndarray:
    """Scale about the image origin (a wrong drawing scale), nearest-neighbour, cropped or padded to the same size."""
    h, w = mask.shape
    big = cv2.resize(mask.astype(np.float32), (round(w * s), round(h * s)), interpolation=cv2.INTER_NEAREST)
    out = np.zeros_like(mask)
    hh, ww = min(h, big.shape[0]), min(w, big.shape[1])
    out[:hh, :ww] = big[:hh, :ww].astype(mask.dtype)
    return out


def _neighbours(mask: np.ndarray, gap: int) -> list[tuple[int, int, int]]:
    """Room pairs within `gap` px of each other, with the length of their shared front, longest first."""
    k = np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)
    pairs = []
    labels = [int(v) for v in np.unique(mask[mask > 0])]
    for a in labels:
        grown = cv2.dilate((mask == a).astype(np.uint8), k) > 0
        for b in np.unique(mask[grown & (mask > 0) & (mask != a)]):
            if a < b:
                pairs.append((int((grown & (mask == b)).sum()), a, int(b)))
    return sorted(pairs, reverse=True)


def merge_adjacent(mask: np.ndarray, gap: int = 15) -> np.ndarray:
    """A missing divider or a gap in a wall: the two rooms with the longest shared wall become one room, and the
    wall between them becomes floor."""
    pairs = _neighbours(mask, gap)
    if not pairs:
        return mask.copy()
    _, a, b = pairs[0]
    out = mask.copy()
    both = ((mask == a) | (mask == b)).astype(np.uint8)
    closed = cv2.morphologyEx(both, cv2.MORPH_CLOSE, np.ones((gap + 1, gap + 1), np.uint8)) > 0
    out[closed & ((mask == 0) | (mask == a) | (mask == b))] = a
    return out


def split_largest(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """A spurious line (a dimension line or furniture edge read as a wall) cuts the largest room in two across its
    longer side."""
    labels, counts = np.unique(mask[mask > 0], return_counts=True)
    if len(labels) == 0:
        return mask.copy()
    big = int(labels[np.argmax(counts)])
    rows, cols = np.nonzero(mask == big)
    out = mask.copy()
    cut = np.zeros_like(mask, bool)
    if np.ptp(cols) >= np.ptp(rows):
        x = int(np.median(cols))
        cut[:, x - width // 2 : x - width // 2 + width] = True
    else:
        y = int(np.median(rows))
        cut[y - width // 2 : y - width // 2 + width, :] = True
    out[cut & (mask == big)] = 0
    pieces, n = ndimage.label(out == big)
    new = int(mask.max()) + 1
    for i in range(2, n + 1):
        out[pieces == i] = new
        new += 1
    return out
