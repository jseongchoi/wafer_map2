"""Generate synthetic wafer map samples from a JSON config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.synth.generator import SyntheticConfig, generate_sample, save_sample
from wafermap.viz import save_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/synth/debug.json",
        help="Path to a synthetic generation JSON config.",
    )
    parser.add_argument(
        "--out",
        default="data/synthetic/debug",
        help="Output directory for generated samples.",
    )
    parser.add_argument("--count", type=int, default=None, help="Override sample count.")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip preview PNG rendering for large CPU-only validation batches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples that already have arrays.npz and metadata.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    out_dir = Path(args.out)
    with config_path.open("r", encoding="utf-8") as f:
        config_data = json.load(f)

    config = SyntheticConfig.from_mapping(config_data)
    if args.count is not None:
        config = SyntheticConfig(
            seed=config.seed,
            count=args.count,
            target_net_die=config.target_net_die,
            chip_width=config.chip_width,
            chip_height=config.chip_height,
            grade_thresholds=config.grade_thresholds,
            pattern_probabilities=config.pattern_probabilities,
            stby_min_chips=config.stby_min_chips,
            stby_max_chips=config.stby_max_chips,
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    for idx in range(config.count):
        sample_id = f"synth_{idx:06d}"
        sample_dir = out_dir / sample_id
        if args.resume and (sample_dir / "arrays.npz").exists() and (sample_dir / "metadata.json").exists():
            skipped += 1
            continue
        sample = generate_sample(config, idx)
        save_sample(sample, sample_dir)
        if not args.no_preview:
            save_preview(
                sample_dir / "preview.png",
                sample.severity,
                sample.wafer_mask,
                sample.stby_mask,
                title=sample.sample_id,
            )
        generated += 1

    manifest = {
        "config": str(config_path),
        "count": config.count,
        "sample_ids": [f"synth_{idx:06d}" for idx in range(config.count)],
        "preview_rendered": not args.no_preview,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Generated {generated} synthetic samples in {out_dir}; skipped {skipped}")


if __name__ == "__main__":
    np.set_printoptions(edgeitems=3, threshold=16)
    main()
