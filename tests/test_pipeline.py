import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from quantity_lab.pdfio import image_to_pdf
from quantity_lab.segmentation import THIRD_PARTY

PLAN = Path(__file__).parent / "fixtures" / "plan.svg"


def _pdf(tmp_path):
    g = np.full((200, 300), 255, np.uint8)
    g[10:20, 10:290] = 0
    g[180:190, 10:290] = 0
    g[10:190, 10:20] = 0
    g[10:190, 280:290] = 0
    g[10:190, 140:150] = 0
    p = tmp_path / "plan.png"
    Image.fromarray(g).save(p)
    return image_to_pdf(p, tmp_path / "plan.pdf")


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "quantity_lab", *map(str, args)], check=False, capture_output=True, text=True
    )


def _check(out, mode):
    data = json.loads((out / "rooms.json").read_text())
    assert data["unit"] == "px2" and data["mode"] == mode
    assert data["summary"]["room_count"] == len(data["rooms"]) > 0
    for r in data["rooms"]:
        assert r["room_area_px2"] > 0 and r["room_perimeter_px"] > 0
    assert (out / "quantities.csv").exists() and (out / "overlay.png").exists()
    return data


def test_oracle_mode(tmp_path):
    r = _run("--input", _pdf(tmp_path), "--annotations", PLAN, "--mode", "oracle", "--output", tmp_path / "o")
    assert r.returncode == 0, r.stderr
    data = _check(tmp_path / "o", "oracle")
    assert data["summary"]["room_count"] == 3  # outdoor room excluded
    assert data["rooms"][0]["room_area_px2"] == 120 * 160
    assert data["summary"]["repaired"] == 1


def test_perturbed_mode(tmp_path):
    r = _run(
        "--input",
        _pdf(tmp_path),
        "--annotations",
        PLAN,
        "--mode",
        "perturbed",
        "--perturbation",
        "erode",
        "--pixels",
        "3",
        "--output",
        tmp_path / "p",
    )
    assert r.returncode == 0, r.stderr
    data = _check(tmp_path / "p", "perturbed")
    assert data["rooms"][0]["room_area_px2"] < 120 * 160
    assert "comparison" in data


def test_predicted_mode(tmp_path):
    r = _run("--input", _pdf(tmp_path), "--annotations", PLAN, "--mode", "predicted", "--output", tmp_path / "c")
    assert r.returncode == 0, r.stderr
    data = _check(tmp_path / "c", "predicted")
    assert data["comparison"]["matched"] >= 1


def test_missing_annotation_exit_1(tmp_path):
    r = _run(
        "--input",
        _pdf(tmp_path),
        "--annotations",
        tmp_path / "none.svg",
        "--mode",
        "oracle",
        "--output",
        tmp_path / "x",
    )
    assert r.returncode == 1 and "not found" in r.stderr


def test_repair_flags_and_validity_survive_export(tmp_path):
    from quantity_lab.pipeline import run, write

    result = run(_pdf(tmp_path), PLAN, "oracle")
    write(result, tmp_path / "audit")
    data = json.loads((tmp_path / "audit" / "rooms.json").read_text())
    assert data["validity"]["repaired_fraction"] == 1 / 3
    assert data["validity"]["valid_before_repair_rate"] == 2 / 3
    assert data["validity"]["valid_after_repair_rate"] == 1.0
    import csv

    row = next(csv.DictReader((tmp_path / "audit" / "quantities.csv").open()))
    assert row["mode"] == "oracle" and row["unit"] == "px2"


@pytest.mark.skipif(not (THIRD_PARTY / "model_best_val_loss_var.pkl").exists(), reason="CubiCasa5K weights not fetched")
def test_cli_segmented(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "seg"
    r = _run("--input", _pdf(tmp_path), "--annotations", PLAN, "--mode", "segmented", "--output", out)
    assert r.returncode == 0, r.stderr
    data = json.loads((out / "rooms.json").read_text())
    assert data["mode"] == "segmented" and "comparison" in data


def test_cli_reads_a_tuning_output_as_params(tmp_path):
    tuned = tmp_path / "params.json"
    tuned.write_text(json.dumps({"tuned_on": ["x"], "chosen": {"threshold": 140, "min_room_px2": 500}}))
    out = tmp_path / "pred"
    r = _run(
        "--input", _pdf(tmp_path), "--annotations", PLAN, "--mode", "predicted", "--params", tuned, "--output", out
    )
    assert r.returncode == 0, r.stderr
