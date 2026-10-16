"""Paths and plan lists shared by the eval scripts."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "samples"
RUNS = REPO / "runs"
INPUTS = RUNS / "inputs"
SYNTH_SRC = INPUTS / "synth"

# CubiCasa5K test split, held out: never used for tuning. Picked simple / medium / dense, fully annotated.
TEST = ["high_quality/488", "high_quality_architectural/9453", "high_quality_architectural/6276"]
# CubiCasa5K train split: the only plans the baseline is tuned on (chosen by eval/fetch_samples.py, see TRAIN.txt)
TRAIN_LIST = SAMPLES / "TRAIN.txt"


def plan_dir(plan: str) -> Path:
    return SAMPLES / plan.replace("/", "_")


def train_plans() -> list[str]:
    return [line.strip() for line in TRAIN_LIST.read_text().splitlines() if line.strip()]
