"""Predicted rooms from a trained model: the CubiCasa5K multi-task network (Kalervo et al., 2019).

The network labels every pixel with a room class (wall, kitchen, bedroom, ...) and an icon class (door, window,
...). Rooms are then the connected floor regions between predicted walls, optionally closed at predicted doors
and split where the room class changes. `predict` needs PyTorch and the CubiCasa5K code and weights (CC BY-NC 4.0,
not part of this repo; see README); `rooms_from_maps` is plain NumPy/OpenCV.
"""

import sys
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np

# room channels of the CubiCasa5K model, in output order
ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath", "Hallway", "Railing",
                "Storage", "Garage", "Other rooms"]  # fmt: skip
OUTDOOR, WALL, KITCHEN, LIVING, RAILING, OTHER = 1, 2, 3, 4, 8, 11
FLOOR = [3, 4, 5, 6, 7, 9, 10, 11]  # indoor room classes
DOOR = 2  # icon channel: Empty, Window, Door, ...
SPLIT = (21, 12, 11)  # output channels: heatmaps, rooms, icons

THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"


@dataclass(frozen=True)
class SegParams:
    close_doors: bool = False  # treat predicted door icons as walls, so rooms do not leak through doorways
    split_by_class: bool = False  # a change of predicted room class also separates rooms (open kitchens)
    min_room_share: float = 0.01  # floor pieces below this share of all predicted floor are not rooms (scale-free)
    neck_share: float = 0.06  # > 0: split floor at openings narrower than 2 x this x sqrt(floor area) (0 = off)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SegParams":
        return cls(**d)


@cache
def _model(third_party: Path):
    import torch

    sys.path.insert(0, str(third_party / "CubiCasa5k"))
    from floortrans.models.hg_furukawa_original import hg_furukawa_original

    model = hg_furukawa_original(n_classes=51)  # built as in CubiCasa5K eval.py, then the 44-channel head
    model.conv4_ = torch.nn.Conv2d(256, sum(SPLIT), bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(sum(SPLIT), sum(SPLIT), kernel_size=4, stride=4)
    ckpt = torch.load(third_party / "model_best_val_loss_var.pkl", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return model.eval().to(device), device


def predict(rgb: np.ndarray, third_party: Path = THIRD_PARTY) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel room class and icon class (argmax), same size as `rgb` (H x W x 3, uint8)."""
    import torch

    model, device = _model(third_party)
    h, w = rgb.shape[:2]
    x = np.pad(rgb, ((0, -h % 32), (0, -w % 32), (0, 0)), constant_values=255).astype(np.float32)
    x = torch.from_numpy(2 * (x / 255.0) - 1).permute(2, 0, 1)[None].to(device)  # the model's [-1, 1] input
    with torch.no_grad():
        y = model(x)[0, :, :h, :w]
    rooms = y[SPLIT[0] : SPLIT[0] + SPLIT[1]].argmax(0)
    icons = y[SPLIT[0] + SPLIT[1] :].argmax(0)
    return rooms.cpu().numpy(), icons.cpu().numpy()


def split_necks(floor: np.ndarray, neck_share: float) -> np.ndarray:
    """Label mask that splits `floor` at narrow openings such as doorways. Cores farther than
    r = neck_share * sqrt(floor area) from any non-floor pixel seed one room each; the cores then grow together, one
    pixel per step and through floor only, so every floor pixel joins the core it is nearest to along the floor.
    A floor region with no core stays one room."""
    dist = cv2.distanceTransform(floor.astype(np.uint8), cv2.DIST_L2, 5)
    _, seeds = cv2.connectedComponents((dist > neck_share * np.sqrt(floor.sum())).astype(np.uint8), connectivity=4)
    nc, comp = cv2.connectedComponents(floor.astype(np.uint8), connectivity=4)
    seeded = np.zeros(nc, bool)
    seeded[np.unique(comp[seeds > 0])] = True
    unseeded = np.flatnonzero(~seeded[1:]) + 1
    extra = np.zeros(nc, np.int32)
    extra[unseeded] = seeds.max() + 1 + np.arange(len(unseeded))
    lab = np.where(seeds > 0, seeds, extra[comp]).astype(np.int32)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    todo = floor & (lab == 0)
    while todo.any():
        grown = cv2.dilate(lab.astype(np.uint16), cross).astype(np.int32)  # a free pixel takes a labelled neighbour
        take = todo & (grown > 0)
        if not take.any():
            break
        lab[take] = grown[take]
        todo &= ~take
    lab[~floor] = 0
    return lab


def rooms_from_maps(rooms: np.ndarray, icons: np.ndarray, p: SegParams) -> np.ndarray:
    """Room label mask (0 = not a room): 4-connected floor regions between walls, railings and (optionally) doors."""
    floor = np.isin(rooms, FLOOR)
    if p.close_doors:
        floor &= icons != DOOR
    out = np.zeros(rooms.shape, np.int32)
    min_px2 = p.min_room_share * floor.sum()
    k = 1
    groups = [rooms == c for c in FLOOR] if p.split_by_class else [np.ones_like(floor)]
    for g in groups:
        part = floor & g
        if p.neck_share > 0:
            lab = split_necks(part, p.neck_share)
        else:
            _, lab = cv2.connectedComponents(part.astype(np.uint8), connectivity=4)
        sizes = np.bincount(lab.ravel())
        for i in np.flatnonzero(sizes >= min_px2):
            if i == 0:
                continue
            out[lab == i] = k
            k += 1
    return out
