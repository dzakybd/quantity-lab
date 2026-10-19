"""Verify all exports and rerun the frozen baseline and trained-model settings on the ten held-out plans."""

import csv
import hashlib
import json
import subprocess
import sys

from PIL import Image

from eval.common import INPUTS, RUNS, SAMPLES, TEST, plan_dir
from quantity_lab.baseline import BaselineParams
from quantity_lab.pipeline import run, write
from quantity_lab.segmentation import SegParams


def main():
    config = json.loads((RUNS / "experiment_config.json").read_text())
    m = json.loads((RUNS / "metrics.json").read_text())
    digest = hashlib.sha256((RUNS / "params.json").read_bytes()).hexdigest()
    assert digest == config["params_sha256"]
    seg_digest = hashlib.sha256((RUNS / "seg_params.json").read_bytes()).hexdigest()
    assert seg_digest == config["seg_params_sha256"]
    seg = json.loads((RUNS / "seg_params.json").read_text())
    assert not set(TEST) & set(seg["tuned_on"])
    seg_params = SegParams.from_dict(seg["chosen"])
    assert len(TEST) == len(set(TEST)) == 10 and not set(TEST) & set(config["tuned_on"])
    official_test = {line.strip("/") for line in (SAMPLES / "test.txt").read_text().split()}
    assert set(TEST) <= official_test
    params = BaselineParams.from_dict(json.loads((RUNS / "params.json").read_text())["chosen"])
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False, capture_output=True, text=True)
    (RUNS / "test_results.txt").write_text(tests.stdout + tests.stderr)
    assert tests.returncode == 0, tests.stdout + tests.stderr
    checked = 0
    for pid, entry in m.items():
        modes = ["oracle", "predicted", "segmented", *entry["perturbed"]]
        for mode in modes:
            out = RUNS / "outputs" / pid / mode
            data = json.loads((out / "rooms.json").read_text())
            assert data["unit"] == "px2"
            assert data["summary"]["room_count"] == len(data["rooms"])
            with (out / "quantities.csv").open() as f:
                assert len(list(csv.DictReader(f))) == len(data["rooms"])
            with Image.open(out / "overlay.png") as im:
                assert list(im.size) == data["image_size_px"]
            assert all(r["room_area_px2"] > 0 and r["room_perimeter_px"] > 0 for r in data["rooms"])
            checked += 1
    repeated = []
    for p in TEST:
        pid = p.replace("/", "_")
        for f, digest_source in config["source_sha256"][p].items():
            assert hashlib.sha256((plan_dir(p) / f).read_bytes()).hexdigest() == digest_source
        for mode, kw in (("predicted", {"params": params}), ("segmented", {"seg_params": seg_params})):
            result = run(INPUTS / pid / "plan.pdf", plan_dir(p) / "model.svg", mode, **kw)
            for key, expected in m[pid][mode].items():
                if key in result["comparison"]:
                    assert result["comparison"][key] == expected, (pid, mode, key)
            write(result, RUNS / "outputs" / pid / mode)
        repeated.append(pid)
    assert digest == hashlib.sha256((RUNS / "params.json").read_bytes()).hexdigest()
    assert seg_digest == hashlib.sha256((RUNS / "seg_params.json").read_bytes()).hexdigest()
    (RUNS / "verification.json").write_text(
        json.dumps(
            {
                "pytest": {
                    "command": "python -m pytest -q",
                    "exit_code": tests.returncode,
                    "output": tests.stdout.strip(),
                },
                "exports_checked": checked,
                "repeated_predicted_plans": repeated,
                "repeated_modes": ["predicted", "segmented"],
                "params_unchanged": True,
                "source_hashes_verified": True,
                "checks": "JSON/CSV room counts and units; nonzero quantities; overlay image size; no train/test overlap for either tuning; repeated baseline and trained-model metrics identical",
            },
            indent=2,
        )
    )
    print(f"Verified {checked} exports and {len(repeated)} deterministic held-out predictions")


if __name__ == "__main__":
    main()
