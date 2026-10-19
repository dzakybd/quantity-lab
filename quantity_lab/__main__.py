"""CLI: python -m quantity_lab --input plan.pdf --annotations model.svg --mode oracle|perturbed|predicted|segmented --output dir"""

import argparse
import json
import sys
from pathlib import Path

from quantity_lab.baseline import BaselineParams
from quantity_lab.pipeline import PERTURBATIONS, run, write
from quantity_lab.segmentation import SegParams


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="quantity_lab", description="Measure floor-plan rooms (px, px^2).")
    ap.add_argument("--input", type=Path, required=True, help="plan PDF (an image-only PDF is fine)")
    ap.add_argument("--annotations", type=Path, required=True, help="CubiCasa5K model.svg with the reference rooms")
    ap.add_argument("--mode", choices=["oracle", "perturbed", "predicted", "segmented"], required=True)
    ap.add_argument("--perturbation", choices=sorted(PERTURBATIONS), help="perturbed mode: which controlled error")
    ap.add_argument("--pixels", type=int, default=2, help="perturbed mode: size of the error (see README)")
    ap.add_argument("--params", type=Path, help="predicted / segmented mode: JSON of BaselineParams / SegParams fields")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.mode == "perturbed" and not args.perturbation:
        ap.error("--perturbation is required in perturbed mode")
    for path in (args.input, args.annotations):
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1
    try:
        raw = json.loads(args.params.read_text()) if args.params else None
        params = BaselineParams.from_dict(raw) if raw and args.mode == "predicted" else None
        seg_params = SegParams.from_dict(raw) if raw and args.mode == "segmented" else None
        result = run(args.input, args.annotations, args.mode, args.perturbation, args.pixels, params, seg_params)
    except (OSError, ValueError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    write(result, args.output)
    print(f"{result['summary']['room_count']} rooms ({args.mode}) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
