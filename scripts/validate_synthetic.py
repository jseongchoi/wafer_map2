"""Validate generated synthetic wafer samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.data import load_sample
from wafermap.evaluation import validate_synthetic_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/synthetic/debug",
        help="Directory containing synth_* sample folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data)
    failures = 0
    sample_dirs = sorted(path for path in root.glob("synth_*") if path.is_dir())
    if not sample_dirs:
        raise SystemExit(f"No samples found under {root}")

    for sample_dir in sample_dirs:
        sample = load_sample(sample_dir)
        errors = validate_synthetic_sample(sample)
        if errors:
            failures += 1
            print(f"{sample.sample_id}: FAIL")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"{sample.sample_id}: OK")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
