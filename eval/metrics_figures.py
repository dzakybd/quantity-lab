"""Plot held-out numeric results only; never publish dataset images."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.common import RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    held = [v for v in json.loads((RUNS / "metrics.json").read_text()).values() if v["dataset"] == "held-out"]
    for lang in ("en", "id"):
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        for perturbation, label, color in [
            ("erode_2", "−2 px", "#225ba8"),
            ("erode_5", "−5 px", "#5a8bd0"),
            ("dilate_2", "+2 px", "#248866"),
            ("dilate_5", "+5 px", "#6eb095"),
            ("shift_5", "shift 5 px" if lang == "en" else "geser 5 px", "#8c689c"),
            ("merge_20", "merge" if lang == "en" else "gabung", "#bd943e"),
            ("split_3", "split" if lang == "en" else "pecah", "#d97c43"),
        ]:
            rows = [v["perturbed"][perturbation] for v in held]
            ax.scatter(
                [r["mask_iou"] for r in rows],
                [r["area_abs_err_pct_median"] for r in rows],
                s=36,
                alpha=0.8,
                color=color,
                label=label,
            )
        ax.set_xlabel("Indoor mask IoU" if lang == "en" else "IoU mask ruangan dalam")
        ax.set_ylabel("Median matched-room area error (%)" if lang == "en" else "Median galat luas ruangan cocok (%)")
        ax.set_title(
            "Overlap and quantity error tell different stories · 10 test plans"
            if lang == "en"
            else "Overlap dan galat kuantitas berbeda · 10 denah uji",
            loc="left",
        )
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(alpha=0.15)
        fig.text(
            0.12,
            0.01,
            "One point per plan and perturbation. Matched rooms ≥ 1,000 px²; merges can hide in the median."
            if lang == "en"
            else "Satu titik per denah dan gangguan. Ruangan cocok ≥ 1.000 px²; median dapat menyamarkan penggabungan.",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(args.out / f"quantity-lab-iou-area{'-id' if lang == 'id' else ''}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
