"""Pack synthetic overlays and the benchmark numbers into space/data for the Hugging Face Space.

Only synthetic plans are pictured. CubiCasa5K results appear as numbers only.

Usage: uv run python space/build.py   (after python -m eval.benchmark)
"""

import csv
import json
import shutil
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
VIEWS = ["oracle", "perturbed_erode_2", "perturbed_erode_5", "perturbed_dilate_5", "perturbed_merge_20",
         "perturbed_split_3", "predicted"]  # fmt: skip
TITLES = {"synth-simple": "Simple: 3 rooms", "synth-medium": "Medium: open kitchen", "synth-dense": "Dense: 12 rooms"}
WEB = 1100


def main() -> None:
    data = REPO / "space" / "data"
    shutil.rmtree(data, ignore_errors=True)
    metrics = json.loads((RUNS / "metrics.json").read_text())
    plans = []
    for pid, title in TITLES.items():
        assert pid.startswith("synth-"), pid
        (data / pid).mkdir(parents=True)
        for v in VIEWS:
            im = Image.open(RUNS / "overlays" / pid / f"{v}.png").convert("RGB")
            im.thumbnail((WEB, WEB), Image.LANCZOS)
            im.save(data / pid / f"{v}.png", optimize=True)
        m = metrics[pid]
        scores = {"oracle": {"count_f1": 1.0, "total_area_err_pct": 0.0, "area_signed_err_pct_mean": 0.0}}
        scores |= {v: m["perturbed"][v.removeprefix("perturbed_")] for v in VIEWS if v.startswith("perturbed_")}
        scores["predicted"] = m["predicted"]
        plans.append({"id": pid, "title": title, "rooms": m["oracle"]["room_count"], "scores": scores})
    held = {k: v for k, v in metrics.items() if v["dataset"] == "held-out"}
    out = {"plans": plans, "held_out": held,
           "perturbations": [r for r in csv.DictReader((RUNS / "perturbation_results.csv").open()) if r["dataset"] == "held-out"],
           "params": json.loads((RUNS / "params.json").read_text())["chosen"]}  # fmt: skip
    (data / "results.json").write_text(json.dumps(out, indent=1))
    print(f"{len(plans)} synthetic plans x {len(VIEWS)} views -> {data}")


if __name__ == "__main__":
    main()
