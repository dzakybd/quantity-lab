# quantity-lab

How do small geometry errors turn into wrong quantities? quantity-lab measures room area, perimeter and count on floor plans in three separate modes, so the error sources stay apart:

- **A. Oracle:** the annotated room polygons, measured directly. It also runs the raster round trip (polygons → mask → polygons) to show the error that rasterizing alone adds.
- **B. Controlled errors:** known damage applied to the oracle mask, then re-measured. The damage types are boundary erosion and dilation (2 and 5 px), a 2 and 5 px shift, a ±2% scale, a missing divider (two rooms merge) and a spurious line (one room splits).
- **C. Predicted:** a classical OpenCV baseline with no model. It keeps thick dark strokes as walls, closes doorways between collinear wall pieces, and labels the free space between walls as rooms.

Units are **image pixels** (px, px²). No physical scale is claimed.

Demo (synthetic plans, precomputed): [huggingface.co/spaces/dzakybd/quantity-lab](https://huggingface.co/spaces/dzakybd/quantity-lab) · Write-up: [dzaky.ai.id/en/blog/quantity-lab-takeoff-errors](https://dzaky.ai.id/en/blog/quantity-lab-takeoff-errors)

## Install and run

```bash
uv sync
uv run python -m quantity_lab --input plan.pdf --annotations model.svg --mode oracle --output out/plan
uv run python -m quantity_lab --input plan.pdf --annotations model.svg --mode perturbed --perturbation erode --pixels 2 --output out/plan_erode2
uv run python -m quantity_lab --input plan.pdf --annotations model.svg --mode predicted --params runs/params.json --output out/plan_pred
```

- **Inputs:**
  - `--input` is a plan PDF; an image-only PDF is fine. To make one from an image, use `quantity_lab.pdfio.image_to_pdf`, which maps one PDF point to one image pixel.
  - `--annotations` is a CubiCasa5K-style `model.svg`. Its `Space <Type>` polygons are the reference rooms; `Space Outdoor` is excluded from the quantities.
- **`--pixels`:** the size of the error. For `scale` it is a percentage (2 means ×1.02); for `merge` it is the widest wall that gets bridged.
- **Outputs:** `rooms.json`, `quantities.csv` and `overlay.png`.
  - Metric names carry their units: `room_area_px2`, `room_perimeter_px`, `room_count`.
  - Polygons that needed `make_valid` are flagged `repaired`.
  - The perturbed and predicted modes add a `comparison` with the oracle.

## How it is measured

- **Matching:** each predicted room is matched one-to-one to a reference room at IoU ≥ 0.5 (Hungarian assignment). Count precision, recall and F1 follow from the matches.
- **Merged and split rooms:**
  - *merged:* one predicted room covers at least 30% of two or more reference rooms;
  - *split:* one reference room has two or more predicted pieces each covering at least 30% of it.
- **Errors:** area and perimeter errors are computed on matched rooms. Relative errors are reported only for reference rooms of at least 1,000 px². The total-area error compares the sums.
- **Data:** ten held-out CubiCasa5K **test** plans (`runs/sample_manifest.json`): the first three examples (`high_quality/488`, `high_quality_architectural/9453`, `high_quality_architectural/6276`) plus seven taken in order from the test list shuffled with seed 20261019, keeping plans whose annotations cover at least 60% of the dark pixels. One candidate (`2204`) failed that check and is logged as excluded. No predicted score was used to pick plans.
- **Tuning:** the baseline was tuned on three **train** plans (`samples/TRAIN.txt`) and frozen in `runs/params.json`.

## Results (held-out test plans)

**Mode A: the measuring code.**
- Nine of ten plans keep every room through the raster round trip. `high_quality/5632` loses one: a later annotation fully covers an Undefined room in the label mask (`runs/failure_cases/oracle-overlap.md`).
- Median room-area error is 0.07–0.36%, and total area is within 0.30%.

**Mode B: controlled errors.** Range across the ten plans of each plan's mean signed room-area error (`runs/perturbation_results.csv`):

| Error | Room area | Perimeter (median abs.) | Count F1 |
|---|---|---|---|
| boundary pulled in 2 px | −4.1 to −3.2% | 1.3–2.0% | 0.94–1.00 |
| boundary pulled in 5 px | −10.1 to −7.6% | 3.5–4.9% | 0.94–1.00 |
| boundary pushed out 2 px | +2.6 to +3.9% | 0.6–1.3% | 0.94–1.00 |
| boundary pushed out 5 px | +6.6 to +9.7% | 2.1–3.9% | 0.94–1.00 |
| whole plan shifted 5 px | −0.2 to +0.2% | 0.0–0.1% | 0.94–1.00 |
| scale −2% | −4.2 to −3.6% | 1.8–2.1% | 0.82–1.00 |
| scale +2% | +3.8 to +4.2% | 1.9–2.2% | 0.82–1.00 |
| one missing divider (merge) | total area within 0.83% | — | 0.80–0.98 |
| one spurious line (split) | total area within 0.45% | — | 0.73–0.95 |

What this shows:
- **Boundary errors scale with perimeter.** A 2 px error costs smaller rooms more.
- **A shifted plan keeps its areas.** Misregistration hurts overlap, not area.
- **Count F1 below 1.0 on boundary errors comes from one plan** (`5632`, the overlap case above), not from the perturbation.
- **Merges and splits leave the total area almost unchanged but get the room count wrong.**

**Mode C: OpenCV baseline (count F1 / total-area error).**

| Plan | Reference rooms | Predicted | Matched | Count F1 | Total area error |
|---|---|---|---|---|---|
| HQ/488 | 5 | 31 | 2 | 0.11 | +33.8% |
| HQA/9453 | 9 | 23 | 0 | 0.00 | −1.3% |
| HQA/6276 | 28 | 71 | 5 | 0.10 | −4.5% |
| colorful/13502 | 6 | 29 | 3 | 0.17 | +5.6% |
| HQA/2035 | 6 | 39 | 4 | 0.18 | +11.5% |
| HQA/875 | 8 | 103 | 4 | 0.07 | +2.8% |
| HQ/9334 | 11 | 2 | 0 | 0.00 | −99.3% |
| HQA/9433 | 17 | 35 | 3 | 0.12 | −74.1% |
| HQ/5632 | 9 | 42 | 3 | 0.12 | +7.1% |
| HQ/1140 | 8 | 47 | 4 | 0.15 | +3.8% |

The baseline fails as a room counter. It splits rooms at furniture and fixture lines, merges rooms through open doorways and open plans, and its settings did not carry over between drawing styles: tuned on three train plans, it scored 0.33 count F1 even there. Across the ten plans, mean count F1 is 0.101 and the room-count MAE is 33.3. Note `9453`: **total area within 1.3% while matching zero rooms**. A right total can hide wrong rooms. HQ = `high_quality`, HQA = `high_quality_architectural`.

Mask IoU, area/perimeter errors, polygon validity before and after repair, and stage timings (mean, median, min, max) are in `runs/summary.json`. Rooms are matched one-to-one at IoU ≥ 0.5. `runs/verification.json` records the export checks and the repeat run of every prediction.

Synthetic plans (clean, CubiCasa-style) are in `runs/metrics.json`. They use the same frozen parameters and get count F1 0.00–0.44, because their doorways, scaled to the working resolution, are wider than the tuned gap closing (130 px).

## Reproduce

```bash
uv run python -m eval.fetch_samples   # needs KAGGLE_API_TOKEN; Kaggle mirror qmarva/cubicasa5k
uv run python -m eval.tune            # train plans only
uv run python -m eval.benchmark
uv run python -m eval.report
uv run python -m eval.verify
uv run python space/build.py
uv run pytest -q
```

## Limitations

- **Ten test plans,** three of which were inspected before the set was expanded. These show how the errors behave, not a general accuracy figure.
- **No physical units.** A real takeoff needs a validated scale (a scale bar or a known dimension).
- **The baseline is deliberately classical.** A trained segmentation model (such as the one in the CubiCasa5K paper) is the natural next step for mode C.
- **Wall length is not measured.** The area of a wall polygon is not wall length, and a centreline method was out of scope.

## Data and licence

Code: MIT.

CubiCasa5K: A. Kalervo, J. Ylioinas, M. Häikiö, A. Karhu, J. Kannala, "CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis", SCIA 2019. Data: https://zenodo.org/records/2613548 (CC BY-NC-SA 4.0). Mirror used for per-file download: https://www.kaggle.com/datasets/qmarva/cubicasa5k.

**No CubiCasa5K data is included in this repository.** `samples/` and every image rendered from it stay local; only numeric results are committed. Public pictures are synthetic plans made by `eval/synth.py`.
