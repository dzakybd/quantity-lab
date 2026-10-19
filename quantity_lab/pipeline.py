"""One plan, one mode: oracle (annotated rooms), perturbed (a controlled error on the oracle mask), predicted
(the OpenCV baseline) or segmented (the trained CubiCasa5K model). Each returns the measured rooms plus, for all
but oracle, a comparison with the oracle."""

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from quantity_lab import perturb
from quantity_lab.annotations import load_plan
from quantity_lab.baseline import BaselineParams, segment_rooms
from quantity_lab.geometry import Area, measure, polygonize, rasterize, repair
from quantity_lab.metrics import compare
from quantity_lab.pdfio import render_gray, render_rgb
from quantity_lab.segmentation import SegParams, rooms_from_maps

PERTURBATIONS = {
    "erode": lambda m, px: perturb.erode(m, px),
    "dilate": lambda m, px: perturb.dilate(m, px),
    "shift": lambda m, px: perturb.shift(m, px, 0),
    "scale": lambda m, px: perturb.scale(m, 1 + px / 100),  # --pixels 2 means x1.02
    "merge": lambda m, px: perturb.merge_adjacent(m, gap=px),  # --pixels is the widest wall bridged
    "split": lambda m, px: perturb.split_largest(m, width=px),
}


def _room(i: int, geom: Area, label: str | None, repaired: bool) -> dict:
    fixed, again = repair(geom)
    area, perim = measure(fixed)
    return {
        "id": f"r{i}",
        "label": label,
        "room_area_px2": round(area, 2),
        "room_perimeter_px": round(perim, 2),
        "valid": not (repaired or again),
        "repaired": repaired or again,
        "geometry": fixed,
    }


def run(
    pdf: Path,
    svg: Path,
    mode: str,
    perturbation: str | None = None,
    pixels: int = 2,
    params: BaselineParams | None = None,
    seg_params: SegParams | None = None,
) -> dict:
    t = {}
    t0 = time.perf_counter()
    gray = render_gray(pdf)
    t["render_s"] = time.perf_counter() - t0
    size = (gray.shape[1], gray.shape[0])
    t0 = time.perf_counter()
    plan = load_plan(svg)
    t["annotations_s"] = time.perf_counter() - t0
    gt_rooms = plan.indoor()
    gt = [r.polygon for r in gt_rooms]
    comparison = None
    t0 = time.perf_counter()
    audit = {}
    if mode == "oracle":
        rooms = [_room(i, r.polygon, r.label, r.repaired) for i, r in enumerate(gt_rooms, 1)]
        t["measure_s"] = time.perf_counter() - t0
    else:
        if mode == "perturbed":
            mask = PERTURBATIONS[perturbation](rasterize(gt, size), pixels)
        elif mode == "predicted":
            mask = segment_rooms(gray, params or BaselineParams())
        elif mode == "segmented":
            from quantity_lab.segmentation import predict  # needs PyTorch: uv sync --group seg

            mask = rooms_from_maps(*predict(render_rgb(pdf)), seg_params or SegParams())
        else:
            raise ValueError(f"unknown mode {mode}")
        t["segment_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        polys = polygonize(mask, audit=audit)
        t["polygonize_s"] = time.perf_counter() - t0
        rooms = [_room(i, p, None, label in audit["repaired_labels"]) for i, (label, p) in enumerate(polys.items(), 1)]
        t0 = time.perf_counter()
        comparison = compare([r["geometry"] for r in rooms], gt, size)
        t["score_s"] = time.perf_counter() - t0
    result = {
        "drawing_id": Path(pdf).stem,
        "unit": "px2",
        "mode": mode,
        "perturbation": {"name": perturbation, "pixels": pixels} if mode == "perturbed" else None,
        "image_size_px": list(size),
        "rooms": rooms,
        "summary": {
            "room_count": len(rooms),
            "total_area_px2": round(sum(r["room_area_px2"] for r in rooms), 2),
            "total_perimeter_px": round(sum(r["room_perimeter_px"] for r in rooms), 2),
            "repaired": sum(r["repaired"] for r in rooms),
            "door_count": plan.door_count if mode == "oracle" else None,
            "window_count": plan.window_count if mode == "oracle" else None,
        },
        "timing_s": {k: round(v, 6) for k, v in t.items()},
        "validity": {
            "rooms": len(rooms),
            "valid_before_repair_rate": sum(r["valid"] for r in rooms) / len(rooms) if rooms else None,
            "repaired_fraction": sum(r["repaired"] for r in rooms) / len(rooms) if rooms else None,
            "valid_after_repair_rate": sum(r["geometry"].is_valid for r in rooms) / len(rooms) if rooms else None,
            "contour_audit": audit,
        },
    }
    if comparison is not None:
        result["comparison"] = comparison
    result["_gray"], result["_gt"] = gray, gt
    return result


def write(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rooms = [{k: v for k, v in r.items() if k != "geometry"} for r in result["rooms"]]
    public = {k: v for k, v in result.items() if not k.startswith("_")} | {"rooms": rooms}
    (out / "rooms.json").write_text(json.dumps(public, indent=1))
    with (out / "quantities.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "drawing_id",
                "unit",
                "mode",
                "id",
                "label",
                "room_area_px2",
                "room_perimeter_px",
                "valid",
                "repaired",
            ],
        )
        w.writeheader()
        w.writerows(
            {"drawing_id": result["drawing_id"], "unit": result["unit"], "mode": result["mode"]} | r for r in rooms
        )
    overlay(result["_gray"], result["_gt"], [r["geometry"] for r in result["rooms"]], out / "overlay.png")


PALETTE = [(214, 120, 42), (58, 175, 27), (234, 104, 52), (164, 123, 232), (0, 161, 237), (129, 66, 74)]  # BGR


def overlay(gray: np.ndarray, gt: list[Area], rooms: list[Area], path: Path) -> Path:
    """Plan faded, measured rooms filled in colour, reference rooms outlined in dark grey."""
    faded = (255 - (255 - gray.astype(np.float32)) * 0.35).astype(np.uint8)
    img = cv2.cvtColor(faded, cv2.COLOR_GRAY2BGR)
    fill = img.copy()

    def rings(g):
        for p in getattr(g, "geoms", [g]):
            yield np.round(np.asarray(p.exterior.coords)).astype(np.int32)

    for i, g in enumerate(rooms):
        for ring in rings(g):
            cv2.fillPoly(fill, [ring], PALETTE[i % len(PALETTE)])
    img = cv2.addWeighted(fill, 0.45, img, 0.55, 0)
    for g in gt:
        for ring in rings(g):
            cv2.polylines(img, [ring], True, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return path
