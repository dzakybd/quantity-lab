"""Figures for the write-up, from synthetic plans and committed numbers only (no CubiCasa5K image is drawn).

Usage: uv run python -m eval.figures --out ../dzakyfawwaz/content/quantity-lab/media [--lang en|id]
(macOS fonts; run after python -m eval.benchmark)
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from eval.common import RUNS

FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
INK, MUTED = (29, 29, 27), (107, 107, 102)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
PLANS = ["high_quality_488", "high_quality_architectural_9453", "high_quality_architectural_6276"]

TEXT = {
    "en": {
        "oracle": "Reference rooms",
        "erode5": "Boundary 5 px too small",
        "pred": "OpenCV baseline",
        "pred_two": ["Baseline, simple plan: rooms merge", "Baseline, dense plan: everything merges"],
        "model_two": ["Trained model: doorways split, three rooms", "Open kitchen: still one room"],
        "merge": "Missing divider",
        "split": "Spurious line",
        "legend": "fill: measured rooms, one colour each · dark outline: reference rooms",
        "chart_title": "Room area error from a boundary error (three held-out CubiCasa5K plans)",
        "chart_x": "Boundary error in pixels (negative: rooms too small)",
        "chart_y": "Mean room area error (%)",
        "plans": ["Simple (5 rooms)", "Medium (9 rooms)", "Dense (28 rooms)"],
    },
    "id": {
        "oracle": "Ruangan referensi",
        "erode5": "Batas 5 px terlalu kecil",
        "pred": "Baseline OpenCV",
        "pred_two": ["Baseline, denah sederhana: ruangan menyatu", "Baseline, denah padat: semua menyatu"],
        "model_two": ["Model terlatih: dipisah di pintu, tiga ruangan", "Dapur terbuka: tetap satu ruangan"],
        "merge": "Sekat hilang",
        "split": "Garis palsu",
        "legend": "isian: ruangan terukur, satu warna per ruangan · garis gelap: ruangan referensi",
        "chart_title": "Galat luas ruangan akibat galat batas (tiga denah CubiCasa5K uji)",
        "chart_x": "Galat batas dalam piksel (negatif: ruangan terlalu kecil)",
        "chart_y": "Rata-rata galat luas ruangan (%)",
        "plans": ["Sederhana (5 ruangan)", "Sedang (9 ruangan)", "Padat (28 ruangan)"],
    },
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def panel(paths: list[Path], labels: list[str], width: int, footer: str | None) -> Image.Image:
    pad, head = 24, 52
    ims = [Image.open(p).convert("RGB") for p in paths]
    for im in ims:
        im.thumbnail((width, width), Image.LANCZOS)
    h = max(im.height for im in ims)
    out = Image.new("RGB", (pad + len(ims) * (width + pad), head + h + pad + (44 if footer else 0)), "white")
    d = ImageDraw.Draw(out)
    for i, (im, lb) in enumerate(zip(ims, labels)):
        x = pad + i * (width + pad)
        d.text((x, 16), lb, fill=INK, font=font(24, bold=True))
        out.paste(im, (x, head))
    if footer:
        d.text((pad, head + h + pad - 6), footer, fill=MUTED, font=font(20))
    return out


def _val(rows: list[dict], pid: str, name: str, px: int) -> float:
    return next(
        float(r["area_signed_err_pct_mean"])
        for r in rows
        if r["plan"] == pid and r["perturbation"] == name and r["pixels"] == str(px)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lang", choices=["en", "id"], default="en")
    args = ap.parse_args()
    t, sfx = TEXT[args.lang], "" if args.lang == "en" else "-id"
    args.out.mkdir(parents=True, exist_ok=True)
    ov = RUNS / "overlays"

    m = ov / "synth-medium"
    panel([m / "oracle.png", m / "perturbed_erode_5.png", m / "predicted.png"], [t["oracle"], t["erode5"], t["pred"]],
          520, t["legend"]).save(args.out / f"quantity-lab-three-modes{sfx}.png")  # fmt: skip
    panel([m / "oracle.png", m / "perturbed_merge_20.png", m / "perturbed_split_3.png"],
          [t["oracle"], t["merge"], t["split"]], 520, t["legend"]).save(args.out / f"quantity-lab-merge-split{sfx}.png")  # fmt: skip
    panel([ov / "synth-simple" / "predicted.png", ov / "synth-dense" / "predicted.png"], t["pred_two"], 700,
          t["legend"]).save(args.out / f"quantity-lab-baseline{sfx}.png")  # fmt: skip
    panel([ov / "synth-simple" / "segmented.png", ov / "synth-medium" / "segmented.png"], t["model_two"], 700,
          t["legend"]).save(args.out / f"quantity-lab-model{sfx}.png")  # fmt: skip

    rows = list(csv.DictReader((RUNS / "perturbation_results.csv").open()))
    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    xs = [-5, -2, 0, 2, 5]
    for k, (pid, color, label) in enumerate(zip(PLANS, SERIES, t["plans"])):
        ys = [
            _val(rows, pid, "erode", 5),
            _val(rows, pid, "erode", 2),
            0.0,
            _val(rows, pid, "dilate", 2),
            _val(rows, pid, "dilate", 5),
        ]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8, markeredgecolor="#fcfcfb", label=label)
    ax.axhline(0, color="#c9c8c2", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xlim(-5.6, 5.6)
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlabel(t["chart_x"], color="#52514e")
    ax.set_ylabel(t["chart_y"], color="#52514e")
    ax.set_title(t["chart_title"], loc="left", fontsize=14, fontweight="bold", color="#0b0b0b", pad=14)
    ax.grid(axis="y", color="#e6e5e0", linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c9c8c2")
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors="#52514e", length=0)
    fig.tight_layout()
    fig.savefig(args.out / f"quantity-lab-sensitivity{sfx}.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"figures ({args.lang}) -> {args.out}")


if __name__ == "__main__":
    main()
