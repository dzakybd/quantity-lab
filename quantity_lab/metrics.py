"""Compare predicted rooms with reference rooms: matching, count F1, splits and merges, area and perimeter errors.

Everything is in image pixels (px, px^2); there is no physical scale.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from quantity_lab.geometry import Area, measure, rasterize

MATCH_IOU = 0.5  # a predicted room matches a reference room at IoU >= 0.5 (one-to-one, Hungarian)
PART_SHARE = 0.3  # for split/merge: a piece counts when it covers >= 30% of the room it overlaps
MIN_REL_AREA = 1000.0  # px^2; relative errors only for reference rooms at least this large


def _overlaps(pred: list[Area], gt: list[Area], size: tuple[int, int]):
    pm, gm = rasterize(pred, size), rasterize(gt, size)
    n_p, n_g = len(pred), len(gt)
    inter = np.bincount(pm.ravel() * (n_g + 1) + gm.ravel(), minlength=(n_p + 1) * (n_g + 1))
    inter = inter.reshape(n_p + 1, n_g + 1)[1:, 1:].astype(float)
    pa = np.bincount(pm.ravel(), minlength=n_p + 1)[1:].astype(float)
    ga = np.bincount(gm.ravel(), minlength=n_g + 1)[1:].astype(float)
    union = pa[:, None] + ga[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    fg_iou = float(((pm > 0) & (gm > 0)).sum() / max(1, ((pm > 0) | (gm > 0)).sum()))
    bg_iou = float(((pm == 0) & (gm == 0)).sum() / max(1, ((pm == 0) | (gm == 0)).sum()))
    return inter, pa, ga, iou, fg_iou, bg_iou


def compare(pred: list[Area], gt: list[Area], size: tuple[int, int]) -> dict:
    """`size` is the image (width, height). Returns a flat dict of metrics; see the README for definitions."""
    inter, pa, ga, iou, fg_iou, bg_iou = _overlaps(pred, gt, size)
    pairs = []
    if len(pred) and len(gt):
        # Maximize eligible matches first, then IoU; sub-threshold pairs must not displace valid matches.
        rows, cols = linear_sum_assignment((iou >= MATCH_IOU) * -(min(iou.shape) + 1) - iou)
        pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= MATCH_IOU]
    n = len(pairs)
    precision = n / len(pred) if pred else 0.0
    recall = n / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    gt_share = np.divide(inter, ga[None, :], out=np.zeros_like(inter), where=ga[None, :] > 0) if len(pred) else inter
    pred_share = np.divide(inter, pa[:, None], out=np.zeros_like(inter), where=pa[:, None] > 0) if len(gt) else inter
    merged = int(((gt_share >= PART_SHARE).sum(axis=1) >= 2).sum()) if len(pred) and len(gt) else 0
    split = int(((pred_share >= PART_SHARE).sum(axis=0) >= 2).sum()) if len(pred) and len(gt) else 0

    abs_a, rel_a, abs_p, rel_p, ious = [], [], [], [], []
    for r, c in pairs:
        a_p, p_p = measure(pred[r])
        a_g, p_g = measure(gt[c])
        abs_a.append(abs(a_p - a_g))
        abs_p.append(abs(p_p - p_g))
        ious.append(iou[r, c])
        if a_g >= MIN_REL_AREA:
            rel_a.append((a_p - a_g) / a_g * 100)
            rel_p.append((p_p - p_g) / p_g * 100)
    gt_total = sum(g.area for g in gt)
    pred_total = sum(p.area for p in pred)

    def stat(xs, fn):
        return round(float(fn(xs)), 4) if xs else None

    return {
        "gt_rooms": len(gt),
        "pred_rooms": len(pred),
        "matched": n,
        "count_precision": round(precision, 4),
        "count_recall": round(recall, 4),
        "count_f1": round(f1, 4),
        "room_count_err": len(pred) - len(gt),
        "merged": merged,
        "split": split,
        "mask_iou": round(fg_iou, 4),
        "background_mask_iou": round(bg_iou, 4),
        "matched_iou_mean": stat(ious, np.mean),
        "area_abs_err_mean_px2": stat(abs_a, np.mean),
        "area_abs_err_max_px2": stat(abs_a, np.max),
        "area_signed_err_pct_mean": stat(rel_a, np.mean),
        "area_abs_err_pct_median": stat([abs(x) for x in rel_a], np.median),
        "perimeter_abs_err_mean_px": stat(abs_p, np.mean),
        "perimeter_abs_err_pct_median": stat([abs(x) for x in rel_p], np.median),
        "total_area_err_pct": round((pred_total - gt_total) / gt_total * 100, 4) if gt_total else None,
        "pairs": [[int(r), int(c), round(float(iou[r, c]), 4)] for r, c in pairs],  # [pred index, gt index, IoU]
    }
