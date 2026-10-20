"""Paths and plan lists shared by the eval scripts."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "samples"
RUNS = REPO / "runs"
INPUTS = RUNS / "inputs"
SYNTH_SRC = INPUTS / "synth"

# CubiCasa5K test split, held out: never used for tuning. Picked simple / medium / dense, fully annotated.
TEST = [
    "high_quality/488",
    "high_quality_architectural/9453",
    "high_quality_architectural/6276",
    "colorful/13502",
    "high_quality_architectural/2035",
    "high_quality_architectural/875",
    "high_quality/9334",
    "high_quality_architectural/9433",
    "high_quality/5632",
    "high_quality/1140",
]
# CubiCasa5K train split: the only plans the baseline is tuned on (chosen by eval/fetch_samples.py, see TRAIN.txt)
TRAIN_LIST = SAMPLES / "TRAIN.txt"
# CubiCasa5K validation split: where the trained model's post-processing is tuned (eval/fetch_samples.py --val)
VAL_LIST = SAMPLES / "VAL.txt"


def plan_dir(plan: str) -> Path:
    return SAMPLES / plan.replace("/", "_")


def train_plans() -> list[str]:
    return [line.strip() for line in TRAIN_LIST.read_text().splitlines() if line.strip()]


def val_plans() -> list[str]:
    return [line.strip() for line in VAL_LIST.read_text().splitlines() if line.strip()]
