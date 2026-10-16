# quantity-lab

Floor-plan room quantities (area, perimeter, count) in three separate modes: oracle (annotated polygons),
controlled errors (perturbed masks) and predicted (an OpenCV free-space baseline). Units are image pixels.

## Stack
- Python >= 3.11 (dev on 3.12), uv. PyMuPDF, opencv-python-headless, shapely 2, scipy. Tests: pytest + hypothesis.

## Commands
- Install: `uv sync`
- Tests: `uv run pytest -q`
- Format / lint: `uvx ruff format . && uvx ruff check .`
- CLI: `uv run python -m quantity_lab --input plan.pdf --annotations model.svg --mode oracle --output out/plan`
- Samples (needs KAGGLE_API_TOKEN): `uv run python -m eval.fetch_samples`

## Layout
- `quantity_lab/`: annotations, geometry, perturb, baseline, metrics, pdfio, pipeline, CLI.
- `eval/`: fetch, synthetic plans, tune (train split only), benchmark, figures, demo. `runs/`: committed JSON/CSV.
- `space/`: static Hugging Face Space (synthetic plans only).

## Forbidden
- Never commit CubiCasa5K files or anything rendered from them (`samples/`, `runs/inputs/`, images under `runs/`).
  CubiCasa5K is CC BY-NC-SA 4.0. Public images use synthetic plans only.
- Never tune on the test plans (`eval/tune.py` reads `samples/TRAIN.txt` plans only).
- No physical units (m, m^2): there is no validated scale.
