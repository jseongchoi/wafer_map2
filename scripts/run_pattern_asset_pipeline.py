"""Run the Pattern Asset learning-readiness pipeline end to end."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.assets import (
    DEFAULT_PROCEDURAL_FAMILIES,
    scan_pattern_assets,
)
from wafermap.data import load_sample
from wafermap.reporting.pattern_asset_project_report import project_report_html as clean_project_report_html


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sample-dir", default="data/synthetic/fbm_grouping_scale_pilot/synth_000000")
    parser.add_argument("--assets-root", default="data/pattern_assets")
    parser.add_argument("--composed-dir", default="data/synthetic/asset_composed")
    parser.add_argument("--work-dir", default="outputs/pattern_asset_pipeline")
    parser.add_argument("--report-out", default="outputs/reports/pattern_asset_project_report.html")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--assets-per-wafer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--placement-mode", choices=("source_jitter", "random_valid"), default="source_jitter")
    parser.add_argument("--jitter-pixels", type=int, default=48)
    parser.add_argument(
        "--procedural-families",
        default=",".join(DEFAULT_PROCEDURAL_FAMILIES),
        help="Comma-separated code-generated families. Use none to disable.",
    )
    parser.add_argument("--output-size", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load script: {path}")
    spec.loader.exec_module(module)
    return module


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    composer = load_script("compose_synthetic_from_assets.py")
    readiness = load_script("build_segmentation_readiness.py")
    segmentation_smoke = load_script("train_segmentation_smoke.py")
    embedding_smoke = load_script("train_embedding_smoke.py")
    unet_train = load_script("train_unet_segmentation.py")

    assets_root = resolve_repo_path(args.assets_root)
    base_sample_dir = resolve_repo_path(args.base_sample_dir)
    composed_dir = resolve_repo_path(args.composed_dir)
    work_dir = resolve_repo_path(args.work_dir)
    report_out = resolve_repo_path(args.report_out)
    work_dir.mkdir(parents=True, exist_ok=True)

    procedural_families = composer.parse_family_list(args.procedural_families)
    assets = composer.load_assets(assets_root, require_assets=not procedural_families)
    base = load_sample(base_sample_dir)
    rng = random.Random(args.seed)
    composed_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(args.count):
        sample_id = f"asset_composed_{idx:06d}"
        sample = composer.compose_sample(
            base,
            assets,
            rng,
            args.assets_per_wafer,
            sample_id,
            placement_mode=args.placement_mode,
            jitter_pixels=args.jitter_pixels,
            procedural_families=procedural_families,
        )
        composer.write_sample(sample, composed_dir / sample_id)

    manifest = work_dir / "asset_segmentation_manifest.csv"
    readiness_metrics = work_dir / "asset_segmentation_readiness_metrics.json"
    readiness_report = work_dir / "asset_segmentation_readiness.html"
    gallery = work_dir / "asset_segmentation_gallery.png"
    readiness_outputs = readiness.build_outputs(
        SimpleNamespace(
            data=str(composed_dir),
            out=str(readiness_report),
            metrics=str(readiness_metrics),
            manifest=str(manifest),
            gallery=str(gallery),
            val_fraction=0.2,
            split_seed=args.seed,
            max_gallery_rows=6,
            overlap_stride=4,
        )
    )

    segmentation_report = work_dir / "asset_segmentation_smoke.html"
    segmentation_metrics = work_dir / "asset_segmentation_smoke_metrics.json"
    segmentation_figure = work_dir / "asset_segmentation_smoke_loss.png"
    segmentation_smoke.main(
        [
            "--manifest",
            str(manifest),
            "--out",
            str(segmentation_report),
            "--metrics",
            str(segmentation_metrics),
            "--figure",
            str(segmentation_figure),
            "--output-size",
            str(args.output_size),
            "--max-train-samples",
            "8",
            "--max-val-samples",
            "4",
            "--steps",
            "6",
        ]
    )

    embedding_report = work_dir / "asset_embedding_smoke.html"
    embedding_metrics = work_dir / "asset_embedding_smoke_metrics.json"
    embeddings_csv = work_dir / "asset_embedding_vectors.csv"
    embedding_smoke.main(
        [
            "--manifest",
            str(manifest),
            "--out",
            str(embedding_report),
            "--metrics",
            str(embedding_metrics),
            "--embeddings-out",
            str(embeddings_csv),
            "--output-size",
            str(args.output_size),
            "--embedding-dim",
            str(args.embedding_dim),
            "--top-k",
            str(args.top_k),
            "--max-train-samples",
            "16",
            "--max-val-samples",
            "8",
        ]
    )

    unet_report = work_dir / "asset_unet_segmentation.html"
    unet_metrics = work_dir / "asset_unet_segmentation_metrics.json"
    unet_model = work_dir / "asset_unet_segmentation.pt"
    unet_train.main(
        [
            "--manifest",
            str(manifest),
            "--out",
            str(unet_report),
            "--metrics",
            str(unet_metrics),
            "--model-out",
            str(unet_model),
            "--output-size",
            str(args.output_size),
            "--check-deps",
        ]
    )

    scanned_assets = scan_pattern_assets(assets_root)
    payload = {
        "assets": scanned_assets,
        "asset_count": len(scanned_assets),
        "composed_count": int(args.count),
        "assets_root": assets_root,
        "base_sample_dir": base_sample_dir,
        "composed_dir": composed_dir,
        "work_dir": work_dir,
        "report_out": report_out,
        "readiness_outputs": readiness_outputs,
        "readiness_metrics": json.loads(readiness_metrics.read_text(encoding="utf-8")),
        "segmentation_metrics": json.loads(segmentation_metrics.read_text(encoding="utf-8")),
        "embedding_metrics": json.loads(embedding_metrics.read_text(encoding="utf-8")),
        "unet_metrics": json.loads(unet_metrics.read_text(encoding="utf-8")),
        "outputs": {
            "manifest": manifest,
            "readiness_report": readiness_report,
            "segmentation_report": segmentation_report,
            "embedding_report": embedding_report,
            "embeddings_csv": embeddings_csv,
            "unet_report": unet_report,
            "unet_metrics": unet_metrics,
            "project_report": report_out,
        },
        "placement_mode": args.placement_mode,
        "jitter_pixels": int(args.jitter_pixels),
        "procedural_families": list(procedural_families),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(clean_project_report_html(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_pipeline(args)
    print(f"Wrote project report: {payload['outputs']['project_report']}")
    print(f"Wrote manifest: {payload['outputs']['manifest']}")
    print(f"Wrote embedding vectors: {payload['outputs']['embeddings_csv']}")


if __name__ == "__main__":
    main()
