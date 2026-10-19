"""Pick the trained model's post-processing on CubiCasa5K train plans only, then freeze it in runs/seg_params.json.

The model itself is not retrained; only how its pixel classes become rooms (close at doors? split by class?
smallest room?) is chosen here, with the same objective as eval.tune. The test plans are never read.

Usage: uv run --group seg python -m eval.tune_seg
"""

import itertools
import json
import statistics

import numpy as np
from PIL import Image

from eval.common import RUNS, plan_dir, train_plans
from quantity_lab.annotations import load_plan
from quantity_lab.geometry import polygonize
from quantity_lab.metrics import compare
from quantity_lab.segmentation import SegParams, predict, rooms_from_maps

GRID = {
    "close_doors": [False, True],
    "split_by_class": [False, True],
    "min_room_share": [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03],
}


def main() -> None:
    plans = train_plans()
    data = []
    for plan in plans:
        d = plan_dir(plan)
        rgb = np.asarray(Image.open(d / "F1_scaled.png").convert("RGB"))
        data.append((predict(rgb), [r.polygon for r in load_plan(d / "model.svg").indoor()], rgb.shape))
    trials = []
    for values in itertools.product(*GRID.values()):
        p = SegParams(**dict(zip(GRID, values)))
        ms = [compare(list(polygonize(rooms_from_maps(*maps, p)).values()), gt, (s[1], s[0])) for maps, gt, s in data]
        trials.append(
            {
                "params": p.to_dict(),
                "count_f1": round(statistics.mean(m["count_f1"] for m in ms), 4),
                "abs_total_area_err_pct": round(statistics.mean(abs(m["total_area_err_pct"]) for m in ms), 2),
            }
        )
    best = max(trials, key=lambda t: (t["count_f1"], -t["abs_total_area_err_pct"]))
    out = {
        "tuned_on": plans,
        "objective": "mean room count F1, ties broken by mean |total area error|",
        "model": "CubiCasa5K hg_furukawa_original, model_best_val_loss_var.pkl (trained on the train split, not retrained)",
        "grid": GRID,
        "chosen": best["params"],
        "chosen_score": best,
        "all_trials": trials,
    }
    (RUNS / "seg_params.json").write_text(json.dumps(out, indent=1))
    for t in sorted(trials, key=lambda t: -t["count_f1"]):
        print(t["params"], t["count_f1"], t["abs_total_area_err_pct"])


if __name__ == "__main__":
    main()
