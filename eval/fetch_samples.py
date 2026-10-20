"""Download the held-out test plans and a few train plans from the Kaggle mirror of CubiCasa5K, file by file.

Needs a Kaggle API token (KAGGLE_API_TOKEN). Files land in samples/, which is gitignored: CubiCasa5K
(CC BY-NC-SA 4.0, https://zenodo.org/records/2613548) is never committed.

Train plans: the first `high_quality` entry and the first two `high_quality_architectural` entries of train.txt,
skipping plans whose annotation covers less than 60% of the drawing's ink (partly annotated sheets).

Usage: uv run python -m eval.fetch_samples [--val]
"""

import argparse
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from eval.common import SAMPLES, TEST, TRAIN_LIST, VAL_LIST, plan_dir
from quantity_lab.annotations import load_plan
from quantity_lab.geometry import rasterize

DATASET = "qmarva/cubicasa5k"
FILES = ("model.svg", "F1_scaled.png")


def kaggle_get(path: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    name = path.rsplit("/", 1)[-1]
    if not (dest / name).exists():
        subprocess.run(
            [
                "uvx",
                "kaggle",
                "datasets",
                "download",
                DATASET,
                "-f",
                f"cubicasa5k/cubicasa5k/{path}",
                "-p",
                str(dest),
                "-q",
            ],
            check=True,
            capture_output=True,
        )
    return dest / name


def fetch(plan: str) -> Path:
    d = plan_dir(plan)
    for f in FILES:
        kaggle_get(f"{plan}/{f}", d)
    return d


def coverage(d: Path) -> float:
    """Share of dark pixels that fall inside annotated rooms or walls (dilated a little)."""
    plan = load_plan(d / "model.svg")
    gray = np.asarray(Image.open(d / "F1_scaled.png").convert("L"))
    h, w = gray.shape
    covered = rasterize([r.polygon for r in plan.rooms] + plan.walls, (w, h)) > 0
    import cv2

    covered = cv2.dilate(covered.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    ink = gray < 128
    return float((ink & covered).sum() / max(1, ink.sum()))


def fetch_val(n: int = 10, seed: int = 20261021) -> list[str]:
    """Plans from the validation split for tuning the trained model's post-processing. The model was trained on the
    train split, so its output there is better than on unseen plans; val was only used to pick its checkpoint.
    The val list is shuffled with `seed`; the first `n` plans with >= 60% coverage are kept."""
    with tempfile.TemporaryDirectory() as tmp:
        lst = [e.strip("/") for e in kaggle_get("val.txt", Path(tmp)).read_text().split()]
    random.Random(seed).shuffle(lst)
    chosen = []
    for plan in lst:
        cov = coverage(fetch(plan))
        print("val candidate", plan, f"coverage {cov:.2f}")
        if cov >= 0.6:
            chosen.append(plan)
        if len(chosen) == n:
            break
    VAL_LIST.write_text("\n".join(chosen) + "\n")
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", action="store_true", help="only fetch the validation plans used by eval.tune_seg")
    if ap.parse_args().val:
        print("val:", fetch_val())
        return
    kaggle_get("test.txt", SAMPLES)
    for plan in TEST:
        print("test", plan, fetch(plan))
    with tempfile.TemporaryDirectory() as tmp:
        lst = kaggle_get("train.txt", Path(tmp)).read_text().split()
    want = {"high_quality": 1, "high_quality_architectural": 2}
    chosen = []
    for entry in lst:
        plan = entry.strip("/")
        kind = plan.split("/")[0]
        if want.get(kind, 0) == 0:
            continue
        cov = coverage(fetch(plan))
        print("train candidate", plan, f"coverage {cov:.2f}")
        if cov >= 0.6:
            chosen.append(plan)
            want[kind] -= 1
        if not any(want.values()):
            break
    TRAIN_LIST.write_text("\n".join(chosen) + "\n")
    print("train:", chosen)


if __name__ == "__main__":
    main()
