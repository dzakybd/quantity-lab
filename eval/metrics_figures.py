"""Plot held-out numeric results only; never publish dataset images."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from eval.common import RUNS


def decimal_comma(ax) -> None:
    """Indonesian axes: decimal comma on both axes."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


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
        if lang == "id":
            decimal_comma(ax)
        fig.savefig(args.out / f"quantity-lab-iou-area{'-id' if lang == 'id' else ''}.png")
        plt.close(fig)
        iou_f1(held, lang, args.out)


def iou_f1(held: list[dict], lang: str, out: Path) -> None:
    """Room F1 against indoor-mask IoU for both predictors: high overlap does not mean the rooms are right."""
    en = lang == "en"
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    for mode, label, color, marker in [
        ("predicted", "OpenCV baseline" if en else "Baseline OpenCV", "#eb6834", "s"),
        ("segmented", "Trained model (CubiCasa5K)" if en else "Model terlatih (CubiCasa5K)", "#2a78d6", "o"),
    ]:
        ax.scatter([v[mode]["mask_iou"] for v in held], [v[mode]["count_f1"] for v in held], s=70, color=color,
                   marker=marker, edgecolor="#fcfcfb", linewidth=2, label=label, zorder=3)  # fmt: skip
    worst = min(held, key=lambda v: v["segmented"]["count_f1"])["segmented"]
    ax.annotate(
        f"IoU {worst['mask_iou']:.2f}, total area {worst['total_area_err_pct']:+.1f}%,\n"
        f"but {worst['gt_rooms']} rooms came out as {worst['pred_rooms']}"
        if en
        else f"IoU {worst['mask_iou']:.2f}, luas total {worst['total_area_err_pct']:+.1f}%,\n"
        f"tapi {worst['gt_rooms']} ruangan terbaca jadi {worst['pred_rooms']}".replace(".", ","),
        (worst["mask_iou"], worst["count_f1"]),
        xytext=(0.55, 0.3),
        arrowprops={"arrowstyle": "-", "color": "#6b6b66"},
        fontsize=11,
        color="#1d1d1b",
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.03, 1.02)
    ax.set_xlabel("Indoor mask IoU (pixel overlap)" if en else "IoU mask ruangan dalam (overlap piksel)")
    ax.set_ylabel("Room F1 (rooms matched one-to-one)" if en else "F1 ruangan (pencocokan satu-satu)")
    ax.set_title(
        "Pixel overlap is not room accuracy · 10 test plans"
        if en
        else "Overlap piksel bukan akurasi ruangan · 10 denah uji",
        loc="left",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.15)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    if not en:
        decimal_comma(ax)
    fig.savefig(out / f"quantity-lab-iou-f1{'-id' if lang == 'id' else ''}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
