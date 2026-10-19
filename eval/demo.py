"""15-second demo video (1080x1350) from real runs on a synthetic plan, ending on the held-out CubiCasa5K numbers.

Usage: uv run python -m eval.demo --out ../dzakyfawwaz/content/quantity-lab/media [--lang en|id]
(macOS fonts, needs ffmpeg; run after python -m eval.benchmark)
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from eval.common import RUNS

W, H, FPS = 1080, 1350, 20
PX, PY, PW = 60, 250, 960
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
INK, MUTED, PRIMARY, SOFT = (9, 9, 11), (82, 82, 91), (67, 56, 202), (238, 240, 255)
PLAN = "synth-medium"

TEXT = {
    "en": {
        "kicker": "FLOOR-PLAN TAKEOFF  ·  HOW ERRORS SPREAD",
        "scenes": [
            ("plan", "A floor plan", "Six rooms, one open kitchen"),
            ("oracle", "Measure the rooms", "Area and perimeter of each room, in pixels"),
            ("perturbed_erode_2", "Boundary 2 px too small", "Room area {area}%"),
            ("perturbed_erode_5", "Boundary 5 px too small", "Room area {area}%"),
            ("perturbed_merge_20", "One missing divider", "Total area {total}%, room count wrong"),
            ("predicted", "A classical OpenCV baseline", "Found {found} rooms for {ref}: room F1 {f1}"),
        ],
        "end_title": "Ten real floor plans (CubiCasa5K)",
        "end_rows": [
            ("2 px boundary error", "{e2} room area"),
            ("One missing divider", "total area within {mt}%, room count wrong"),
            ("OpenCV baseline", "mean room F1 {bf1}"),
            ("Right total, wrong rooms", "{rt} total area, 0 rooms matched"),
        ],
        "end_foot": "Pixels, not metres. Pictures: synthetic plan.",
        "url": "dzaky.ai.id/en/blog/quantity-lab-takeoff-errors",
    },
    "id": {
        "kicker": "TAKEOFF DENAH  ·  BAGAIMANA GALAT MENYEBAR",
        "scenes": [
            ("plan", "Sebuah denah", "Enam ruangan, satu dapur terbuka"),
            ("oracle", "Ukur ruangannya", "Luas dan keliling tiap ruangan, dalam piksel"),
            ("perturbed_erode_2", "Batas 2 px terlalu kecil", "Luas ruangan {area}%"),
            ("perturbed_erode_5", "Batas 5 px terlalu kecil", "Luas ruangan {area}%"),
            ("perturbed_merge_20", "Satu sekat hilang", "Luas total {total}%, jumlah ruangan salah"),
            ("predicted", "Baseline OpenCV klasik", "Menemukan {found} ruangan dari {ref}: F1 ruangan {f1}"),
        ],
        "end_title": "Sepuluh denah nyata (CubiCasa5K)",
        "end_rows": [
            ("Galat batas 2 px", "luas ruangan {e2}"),
            ("Satu sekat hilang", "luas total maks. {mt}%, jumlah ruangan salah"),
            ("Baseline OpenCV", "rata-rata F1 ruangan {bf1}"),
            ("Total benar, ruangan salah", "luas total {rt}, 0 ruangan cocok"),
        ],
        "end_foot": "Piksel, bukan meter. Gambar: denah sintetis.",
        "url": "dzaky.ai.id/id/blog/quantity-lab-takeoff-errors",
    },
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def base(t: dict, title: str, sub: str) -> Image.Image:
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((PX, 56), t["kicker"], fill=PRIMARY, font=font(26, bold=True))
    d.text((PX, 100), title, fill=INK, font=font(56, bold=True))
    d.text((PX, 172), sub, fill=MUTED, font=font(32))
    d.text((PX, H - 70), t["url"], fill=MUTED, font=font(26))
    return im


def with_panel(im: Image.Image, img: Image.Image) -> Image.Image:
    im = im.copy()
    p = img.convert("RGB")
    p.thumbnail((PW, PW), Image.LANCZOS)
    im.paste(p, (PX, PY))
    return im


def pct(v: float) -> str:
    return f"{v:+.1f}".replace("-", "−")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lang", choices=["en", "id"], default="en")
    args = ap.parse_args()
    t, sfx = TEXT[args.lang], "" if args.lang == "en" else "-id"
    m = json.loads((RUNS / "metrics.json").read_text())
    s = m[PLAN]
    ov = RUNS / "overlays" / PLAN
    plan_img = Image.open(RUNS / "inputs" / "synth" / PLAN / "F1_scaled.png")
    frames: list[Image.Image] = []

    def hold(img, sec):
        frames.extend([img] * round(sec * FPS))

    for key, title, sub in t["scenes"]:
        if key == "plan":
            hold(with_panel(base(t, title, sub), plan_img), 2.0)
            continue
        if key == "oracle":
            img = Image.open(ov / "oracle.png")
            for k in range(1, 21):  # fade the room fill in
                frames.append(
                    with_panel(
                        base(t, title, sub),
                        Image.blend(plan_img.convert("RGB").resize(img.size), img.convert("RGB"), k / 20),
                    )
                )
            hold(frames[-1], 1.2)
            continue
        if key == "predicted":
            p = s["predicted"]
            sub = sub.format(found=p["pred_rooms"], ref=p["gt_rooms"], f1=f"{p['count_f1']:.2f}")
        else:
            e = s["perturbed"][key.removeprefix("perturbed_")]
            sub = sub.format(area=pct(e["area_signed_err_pct_mean"]), total=pct(e["total_area_err_pct"]))
        hold(with_panel(base(t, title, sub), Image.open(ov / f"{key}.png")), 1.8)

    held = [v for v in m.values() if v["dataset"] == "held-out"]
    e2 = [v["perturbed"]["erode_2"]["area_signed_err_pct_mean"] for v in held]
    mt = max(abs(v["perturbed"]["merge_20"]["total_area_err_pct"]) for v in held)
    bf1 = [v["predicted"]["count_f1"] for v in held]
    zero = next(v for v in held if v["predicted"]["matched"] == 0)
    vals = {"e2": f"{pct(max(e2))} to {pct(min(e2))}%", "mt": f"{mt:.2f}",
            "bf1": f"{sum(bf1) / len(bf1):.3f}", "rt": pct(zero["predicted"]["total_area_err_pct"]) + "%"}  # fmt: skip
    if args.lang == "id":
        vals = {k: v.replace(" to ", " sampai ").replace(".", ",") for k, v in vals.items()}
    end = base(t, t["end_title"], t["end_foot"])
    d = ImageDraw.Draw(end)
    for i, (label, value) in enumerate(t["end_rows"]):
        y = 300 + i * 190
        d.rounded_rectangle((PX - 10, y - 20, W - PX + 10, y + 130), radius=18, fill=SOFT if i % 2 == 0 else None)
        d.text((PX + 10, y), label, fill=INK, font=font(38, bold=True))
        d.text((PX + 10, y + 56), value.format(**vals), fill=PRIMARY, font=font(42, bold=True))
    hold(end, 3.0)

    tmp = Path(tempfile.mkdtemp())
    for i, f in enumerate(frames):
        f.save(tmp / f"f{i:04d}.png")
    mp4, gif = args.out / f"quantity-lab-demo{sfx}.mp4", args.out / f"quantity-lab-demo{sfx}.gif"
    ff = ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "f%04d.png")]
    subprocess.run([*ff, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)], check=True)
    vf = "fps=12,scale=540:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none"
    subprocess.run([*ff, "-vf", vf, str(gif)], check=True)
    shutil.rmtree(tmp)
    print(f"{len(frames)} frames, {len(frames) / FPS:.1f} s -> {mp4.name}, {gif.name}")


if __name__ == "__main__":
    main()
