"""Pick the baseline's parameters on CubiCasa5K train plans only, then freeze them in runs/params.json.

The three test plans are held out: this script reads only the plans listed in samples/TRAIN.txt.

Usage: uv run python -m eval.tune
"""

import itertools
import json
import statistics

import numpy as np
from PIL import Image

from eval.common import RUNS, plan_dir, train_plans
from quantity_lab.annotations import load_plan
from quantity_lab.baseline import BaselineParams, segment_rooms
from quantity_lab.geometry import polygonize
from quantity_lab.metrics import compare

GRID = {
    "threshold": [100, 140, 180, 220],
    "wall_px": [3, 5, 8],
    "door_px": [40, 60, 90, 130],
    "min_room_px2": [1000, 2000, 3000],
    "work_long_side": [1500, 2500],
}


def main() -> None:
    plans = train_plans()
    data = []
    for plan in plans:
        d = plan_dir(plan)
        gray = np.asarray(Image.open(d / "F1_scaled.png").convert("L"))
        data.append((gray, [r.polygon for r in load_plan(d / "model.svg").indoor()]))
    trials = []
    for values in itertools.product(*GRID.values()):
        p = BaselineParams(**dict(zip(GRID, values)))
        ms = [compare(list(polygonize(segment_rooms(g, p)).values()), gt, (g.shape[1], g.shape[0])) for g, gt in data]
        trials.append(
            {
                "params": p.to_dict(),
                "count_f1": round(statistics.mean(m["count_f1"] for m in ms), 4),
                "abs_total_area_err_pct": round(statistics.mean(abs(m["total_area_err_pct"]) for m in ms), 2),
            }
        )
    best = max(trials, key=lambda t: (t["count_f1"], -t["abs_total_area_err_pct"]))
    default = next(t for t in trials if t["params"] == BaselineParams().to_dict())
    out = {
        "tuned_on": plans,
        "objective": "mean room count F1, ties broken by mean |total area error|",
        "grid": GRID,
        "chosen": best["params"],
        "chosen_score": best,
        "default_score": default,
        "trials": len(trials),
    }
    RUNS.mkdir(exist_ok=True)
    (RUNS / "params.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("chosen", "chosen_score", "default_score")}, indent=1))


if __name__ == "__main__":
    main()
