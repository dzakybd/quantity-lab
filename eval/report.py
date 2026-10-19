"""Summarize held-out distributions without pooling plans or silently treating missing matches as zero."""

import json
import statistics

from eval.common import RUNS


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    metrics = json.loads((RUNS / "metrics.json").read_text())
    held = [v for v in metrics.values() if v["dataset"] == "held-out"]
    summary = {
        "n_plans": len(held),
        "aggregation": "Each plan has equal weight; relative quantity errors use matched rooms >= 1000 px2 only. Missing matches are null, never zero.",
        "modes": {},
    }
    for mode in ["raster_roundtrip", "predicted", "segmented", *held[0]["perturbed"]]:
        rows = [v[mode] if mode in v else v["perturbed"][mode] for v in held]
        fields = [k for k, v in rows[0].items() if isinstance(v, (int, float)) or v is None]
        entry = {k: distribution([r[k] for r in rows]) for k in fields}
        entry["room_count_mae"] = statistics.mean(abs(r["room_count_err"]) for r in rows)
        if "validity" in rows[0]:
            entry["validity"] = {
                k: distribution([r["validity"][k] for r in rows])
                for k in ("valid_before_repair_rate", "repaired_fraction", "valid_after_repair_rate")
            }
        if "timing_s" in rows[0]:
            entry["timing_s"] = {k: distribution([r["timing_s"][k] for r in rows]) for k in rows[0]["timing_s"]}
        summary["modes"][mode] = entry
    summary["oracle_validity"] = {
        k: distribution([v["oracle"]["validity"][k] for v in held])
        for k in ("valid_before_repair_rate", "repaired_fraction", "valid_after_repair_rate")
    }
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print("Summarized", len(held), "held-out plans")


if __name__ == "__main__":
    main()
