"""Extract wafer-level feature vectors from synthetic samples."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.data import load_sample
from wafermap.features import extract_feature_vector, extract_validation_feature_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/synthetic/debug",
        help="Directory containing synth_* sample folders.",
    )
    parser.add_argument(
        "--out",
        default="outputs/reports/synthetic_features.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--include-validation-fields",
        action="store_true",
        help="Include synthetic-only pattern mask ratios. Do not use for real inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data)
    rows = []
    for sample_dir in sorted(root.glob("synth_*")):
        sample = load_sample(sample_dir)
        row = {
            "sample_id": sample.sample_id,
            "actual_net_die": sample.metadata["actual_net_die"],
            **extract_feature_vector(sample),
        }
        if args.include_validation_fields:
            row.update(extract_validation_feature_vector(sample))
        rows.append(row)

    if not rows:
        raise SystemExit(f"No samples found under {root}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} feature rows to {out_path}")


if __name__ == "__main__":
    main()
