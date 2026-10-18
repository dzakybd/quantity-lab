"""Run the three modes on the held-out CubiCasa5K test plans and on the synthetic plans.

A. oracle: annotated rooms measured directly, plus the raster round trip (rasterize -> polygonize) against them.
B. controlled errors on the oracle mask, each compared with the oracle.
C. predicted: the OpenCV baseline with the parameters frozen by eval.tune.

Writes runs/metrics.json, per_room.csv, perturbation_results.csv, category_map.json, experiment_config.json and
failure_cases/*.md. Overlays and PDFs go to runs/ subfolders that are gitignored.

Usage: uv run python -m eval.benchmark
"""

import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from collections import Counter
from importlib.metadata import version

import numpy as np
from shapely import union_all

from eval.common import INPUTS, RUNS, SYNTH_SRC, TEST, plan_dir, train_plans
from eval.synth import make_synthetic
from quantity_lab.annotations import load_plan
from quantity_lab.baseline import BaselineParams
from quantity_lab.geometry import measure, polygonize, rasterize
from quantity_lab.metrics import MATCH_IOU, MIN_REL_AREA, PART_SHARE, compare
from quantity_lab.pdfio import image_to_pdf
from quantity_lab.pipeline import overlay, run, write

PERTURBATIONS = [
    ("erode", 2), ("erode", 5), ("dilate", 2), ("dilate", 5), ("shift", 2), ("shift", 5),
    ("scale", -2), ("scale", 2), ("merge", 20), ("split", 3),
]  # fmt: skip
KEYS = ["gt_rooms", "pred_rooms", "count_precision", "count_recall", "room_count_err", "mask_iou",
        "background_mask_iou", "area_abs_err_mean_px2", "area_abs_err_max_px2", "perimeter_abs_err_mean_px", "count_f1", "matched", "merged", "split", "matched_iou_mean", "area_signed_err_pct_mean",
        "area_abs_err_pct_median", "perimeter_abs_err_pct_median", "total_area_err_pct"]  # fmt: skip


def hardware() -> dict:
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    mem = sysctl("hw.memsize")
    return {
        "platform": platform.platform(),
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(int(mem) / 2**30) if mem else None,
    }


def plans():
    for p in TEST:
        yield p.replace("/", "_"), "held-out", plan_dir(p)
    for d in make_synthetic(SYNTH_SRC):
        yield d.name, "synthetic", d


def main() -> None:
    params = BaselineParams.from_dict(json.loads((RUNS / "params.json").read_text())["chosen"])
    metrics, per_room, perturb_rows, categories = {}, [], [], {}
    for pid, dataset, d in plans():
        out = INPUTS / pid
        out.mkdir(parents=True, exist_ok=True)
        pdf = image_to_pdf(d / "F1_scaled.png", out / "plan.pdf")
        svg = d / "model.svg"
        plan = load_plan(svg)
        categories[pid] = {
            "rooms_by_type": dict(Counter(r.label for r in plan.rooms)),
            "outdoor_excluded": sum(r.outdoor for r in plan.rooms),
            "repaired_polygons": sum(r.repaired for r in plan.rooms),
            "walls": len(plan.walls), "doors": plan.door_count, "windows": plan.window_count,
            "other_classes_ignored": plan.excluded,
        }  # fmt: skip
        ov = RUNS / "overlays" / pid
        ov.mkdir(parents=True, exist_ok=True)

        oracle = run(pdf, svg, "oracle")
        gt = oracle["_gt"]
        size = tuple(oracle["image_size_px"])
        mask = rasterize(gt, size)
        rt = compare(list(polygonize(mask).values()), gt, size)
        visible = set(np.unique(mask))
        categories[pid]["fully_occluded_indoor_rooms"] = [
            r.id for i, r in enumerate(plan.indoor(), 1) if i not in visible
        ]
        categories[pid]["indoor_overlap_area_px2"] = round(sum(g.area for g in gt) - union_all(gt).area, 4)
        write(oracle, RUNS / "outputs" / pid / "oracle")
        entry = {"dataset": dataset, "image_size_px": list(size), "oracle": oracle["summary"] | oracle["timing_s"] | {"validity": oracle["validity"]},
                 "raster_roundtrip": {k: rt[k] for k in KEYS}}  # fmt: skip
        overlay(oracle["_gray"], gt, gt, ov / "oracle.png")

        entry["perturbed"] = {}
        for name, px in PERTURBATIONS:
            r = run(pdf, svg, "perturbed", name, px)
            c = r["comparison"]
            entry["perturbed"][f"{name}_{px}"] = {k: c[k] for k in KEYS} | {
                "validity": r["validity"],
                "timing_s": r["timing_s"],
            }
            write(r, RUNS / "outputs" / pid / f"{name}_{px}")
            perturb_rows.append(
                {"plan": pid, "dataset": dataset, "perturbation": name, "pixels": px} | {k: c[k] for k in KEYS}
            )
            overlay(r["_gray"], gt, [x["geometry"] for x in r["rooms"]], ov / f"perturbed_{name}_{px}.png")

        times, stages = [], []
        for _ in range(3):
            t0 = time.perf_counter()
            pred = run(pdf, svg, "predicted", params=params)
            times.append(time.perf_counter() - t0)
            stages.append(pred["timing_s"])
        c = pred["comparison"]
        entry["predicted"] = {k: c[k] for k in KEYS} | {"pred_rooms": c["pred_rooms"], "gt_rooms": c["gt_rooms"],
                                                         "mask_iou": c["mask_iou"], "room_count_err": c["room_count_err"],
                                                         "seconds_median": round(statistics.median(times), 4),
                                                         "timing_s": {k: round(statistics.median(t[k] for t in stages), 6) for k in stages[0]},
                                                         "validity": pred["validity"]}  # fmt: skip
        write(pred, RUNS / "outputs" / pid / "predicted")
        overlay(pred["_gray"], gt, [x["geometry"] for x in pred["rooms"]], ov / "predicted.png")
        labels = [r.label for r in plan.indoor()]
        matched = {g: (p, iou) for p, g, iou in c["pairs"]}
        for gi, g in enumerate(gt):
            a_g, p_g = measure(g)
            row = {"plan": pid, "dataset": dataset, "room": gi, "label": labels[gi], "gt_area_px2": round(a_g, 1),
                   "gt_perimeter_px": round(p_g, 1), "matched": gi in matched}  # fmt: skip
            if gi in matched:
                pi, iou = matched[gi]
                a_p, p_p = measure(pred["rooms"][pi]["geometry"])
                row |= {"iou": iou, "pred_area_px2": round(a_p, 1), "area_err_pct": round((a_p - a_g) / a_g * 100, 2),
                        "pred_perimeter_px": round(p_p, 1), "perimeter_err_pct": round((p_p - p_g) / p_g * 100, 2)}  # fmt: skip
            per_room.append(row)
        metrics[pid] = entry
        print(
            pid,
            "predicted F1",
            c["count_f1"],
            "| erode_2 area",
            entry["perturbed"]["erode_2"]["area_signed_err_pct_mean"],
        )

    _csv(RUNS / "per_room.csv", per_room)
    _csv(RUNS / "perturbation_results.csv", perturb_rows)
    (RUNS / "metrics.json").write_text(json.dumps(metrics, indent=1, allow_nan=False))
    (RUNS / "category_map.json").write_text(json.dumps(
        {"room": "every <g class='Space <Type>'> polygon; Type = second word of the class",
         "excluded_from_quantities": "Space Outdoor (balconies, terraces)",
         "walls": "class 'Wall' and 'Wall External' polygons (drawn, not measured)",
         "counted": "Door*, Window* classes", "per_plan": categories}, indent=1))  # fmt: skip
    (RUNS / "experiment_config.json").write_text(json.dumps({
        "params_sha256": hashlib.sha256((RUNS / "params.json").read_bytes()).hexdigest(),
        "sample_manifest": json.loads((RUNS / "sample_manifest.json").read_text()),
        "source_sha256": {p: {f: hashlib.sha256((plan_dir(p) / f).read_bytes()).hexdigest()
                              for f in ("model.svg", "F1_scaled.png")} for p in TEST},
        "held_out": TEST, "tuned_on": train_plans(), "units": "image pixels (px, px^2); no physical scale",
        "pdf_path": "F1_scaled.png wrapped as an image-only PDF, 1 pt = 1 px, rendered back at 72 DPI",
        "matching": {"iou": MATCH_IOU, "part_share_for_split_merge": PART_SHARE, "min_area_for_relative_px2": MIN_REL_AREA},
        "perturbations": PERTURBATIONS, "baseline_params": params.to_dict(),
        "timing": "predicted mode: median of 3 full runs; separate median render, annotation, segment, polygonize and score stages; PDF wrapping excluded",
        "packages": {p: version(p) for p in ("pymupdf", "opencv-python-headless", "shapely", "scipy", "numpy")},
        "python": platform.python_version(), "hardware": hardware()}, indent=1))  # fmt: skip
    failures(metrics, per_room)


def failures(metrics: dict, per_room: list[dict]) -> None:
    out = RUNS / "failure_cases"
    out.mkdir(exist_ok=True)
    held = {k: v for k, v in metrics.items() if v["dataset"] == "held-out"}
    worst = min(held, key=lambda k: held[k]["predicted"]["count_f1"])
    best = max(held, key=lambda k: held[k]["predicted"]["count_f1"])
    for case, pid in (("baseline-weakest", worst), ("baseline-best", best)):
        p = held[pid]["predicted"]
        rooms = [r for r in per_room if r["plan"] == pid]
        missed = Counter(r["label"] for r in rooms if not r["matched"])
        lines = [f"# {case}: {pid}", "", (f"Reference rooms {p['gt_rooms']}, predicted {p['pred_rooms']}, matched "
                 f"{p['matched']} (IoU >= {MATCH_IOU}), merged {p['merged']}, split {p['split']}."),
                 f"Count F1 {p['count_f1']}, total area error {p['total_area_err_pct']}%, mask IoU {p['mask_iou']}.", "",
                 "Missed rooms by type: " + ", ".join(f"{k} {v}" for k, v in missed.most_common()), "",
                 f"Overlay (local only): runs/overlays/{pid}/predicted.png"]  # fmt: skip
        (out / f"{case}.md").write_text("\n".join(lines) + "\n")


def _csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
