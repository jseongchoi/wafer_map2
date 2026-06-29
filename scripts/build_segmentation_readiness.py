"""Build a CPU-only segmentation readiness report from synthetic FBM samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.data import PATTERN_CLASSES, SyntheticSample, load_sample
from wafermap.reporting.segmentation_readiness_report import html_report
from wafermap.training.segmentation import INPUT_CHANNELS, TARGET_CHANNELS

PATTERN_TO_INDEX = {name: idx for idx, name in enumerate(PATTERN_CLASSES)}
FOCUS_CLASSES = ("scratch", "ring", "local", "stby_pattern")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic/fbm_grouping_scale_pilot")
    parser.add_argument("--out", default="outputs/reports/fbm_segmentation_readiness_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_segmentation_readiness_metrics.json")
    parser.add_argument("--manifest", default="outputs/reports/fbm_segmentation_manifest.csv")
    parser.add_argument("--gallery", default="outputs/figures/fbm_segmentation_readiness_gallery.png")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260617)
    parser.add_argument("--max-gallery-rows", type=int, default=6)
    parser.add_argument("--overlap-stride", type=int, default=4)
    return parser.parse_args(argv)


def sample_dirs(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if (path / "arrays.npz").exists())


def stable_split(sample_id: str, val_fraction: float, seed: int) -> str:
    score = zlib.crc32(f"{seed}:{sample_id}".encode("utf-8")) / float(2**32)
    return "val" if score < val_fraction else "train"


def mask_ratio(mask: np.ndarray, denominator: int) -> float:
    return float(mask.sum() / max(denominator, 1))


def sample_manifest_row(sample_dir: Path, sample: SyntheticSample, split: str, root: Path) -> dict[str, Any]:
    wafer = sample.wafer_mask > 0
    valid = wafer & (sample.valid_test_mask > 0)
    denominator = int(valid.sum())
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "split": split,
        "arrays_path": relpath(sample_dir / "arrays.npz", root),
        "metadata_path": relpath(sample_dir / "metadata.json", root),
        "actual_net_die": sample.metadata.get("actual_net_die", ""),
        "wafer_pixel_count": int(wafer.sum()),
        "valid_test_pixel_count": denominator,
        "input_channels": "|".join(INPUT_CHANNELS),
        "target_channels": "|".join(TARGET_CHANNELS),
    }
    active_count = 0
    for class_name in TARGET_CHANNELS:
        class_mask = sample.pattern_masks[PATTERN_TO_INDEX[class_name]] > 0
        ratio = mask_ratio(class_mask & valid, denominator)
        row[f"has_{class_name}"] = int(ratio > 0.0)
        row[f"{class_name}_mask_ratio"] = ratio
        active_count += int(ratio > 0.0)
    row["active_target_count"] = active_count
    return row


def save_gallery(samples: list[SyntheticSample], out: Path, max_rows: int) -> None:
    selected = samples[:max_rows]
    if not selected:
        return
    columns = ("input", *FOCUS_CLASSES)
    fig, axes = plt.subplots(
        len(selected),
        len(columns),
        figsize=(3.1 * len(columns), 3.0 * len(selected)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    colors = {
        "scratch": "autumn",
        "ring": "winter",
        "local": "spring",
        "stby_pattern": "gray",
    }
    for row_idx, sample in enumerate(selected):
        stride = max(1, int(np.ceil(max(sample.shape) / 650)))
        base = sample.severity.astype(np.float32) / 7.0
        wafer = sample.wafer_mask > 0
        base = base.copy()
        base[~wafer] = np.nan
        for col_idx, column in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ax.imshow(base[::stride, ::stride], cmap="turbo", vmin=0.0, vmax=1.0, interpolation="nearest")
            if column != "input":
                mask = sample.stby_mask > 0 if column == "stby_pattern" else sample.pattern_masks[PATTERN_TO_INDEX[column]] > 0
                overlay = np.ma.masked_where(~mask[::stride, ::stride], mask[::stride, ::stride])
                ax.imshow(overlay, cmap=colors[column], alpha=0.58, interpolation="nearest")
            title = f"{sample.sample_id}\n{column}" if col_idx == 0 else column
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def update_gallery_candidates(
    rows: list[dict[str, Any]],
    sample_dir: Path,
    sample: SyntheticSample,
) -> None:
    scores: dict[str, float] = {"sample_id": sample.sample_id, "path": str(sample_dir)}
    for class_name in FOCUS_CLASSES:
        scores[class_name] = float((sample.stby_mask > 0).sum()) if class_name == "stby_pattern" else float((sample.pattern_masks[PATTERN_TO_INDEX[class_name]] > 0).sum())
    scratch = sample.pattern_masks[PATTERN_TO_INDEX["scratch"]] > 0
    stby = sample.stby_mask > 0
    scores["scratch_stby_overlap"] = float((scratch & stby).sum())
    scores["active_target_count"] = float(sum((sample.pattern_masks[PATTERN_TO_INDEX[name]] > 0).any() for name in TARGET_CHANNELS))
    rows.append(scores)


def selected_gallery_paths(candidates: list[dict[str, Any]], max_rows: int) -> list[Path]:
    selected: list[Path] = []
    seen: set[str] = set()
    selectors = [*FOCUS_CLASSES, "scratch_stby_overlap", "active_target_count"]
    for selector in selectors:
        if len(selected) >= max_rows:
            break
        pool = [row for row in candidates if row["sample_id"] not in seen]
        if not pool:
            break
        best = max(pool, key=lambda row: float(row[selector]))
        if float(best[selector]) <= 0.0 and selected:
            continue
        selected.append(Path(str(best["path"])))
        seen.add(str(best["sample_id"]))
    return selected


def summarize_class_accumulators(all_ratios: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows = []
    for class_name in TARGET_CHANNELS:
        ratios = all_ratios[class_name]
        positives = [value for value in ratios if value > 0.0]
        positive_array = np.array(positives, dtype=np.float64)
        pixel_prevalence = float(np.mean(ratios)) if ratios else 0.0
        pos_weight = float((1.0 - pixel_prevalence) / max(pixel_prevalence, 1e-9))
        rows.append(
            {
                "class": class_name,
                "positive_samples": int(len(positives)),
                "sample_presence_rate": float(len(positives) / max(len(ratios), 1)),
                "mean_pixel_ratio": pixel_prevalence,
                "median_positive_ratio": float(np.median(positive_array)) if len(positive_array) else 0.0,
                "p95_positive_ratio": float(np.percentile(positive_array, 95)) if len(positive_array) else 0.0,
                "suggested_pos_weight_capped": float(min(pos_weight, 50.0)),
            }
        )
    return rows


def summarize_overlap_accumulators(
    pair_sums: dict[tuple[str, str], float],
    pair_cooccurrences: dict[tuple[str, str], int],
    sample_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for pair, overlap_sum in pair_sums.items():
        rows.append(
            {
                "pair": f"{pair[0]}+{pair[1]}",
                "cooccurrence_rate": float(pair_cooccurrences[pair] / max(sample_count, 1)),
                "mean_overlap_pixel_ratio": float(overlap_sum / max(sample_count, 1)),
            }
        )
    return sorted(rows, key=lambda row: row["mean_overlap_pixel_ratio"], reverse=True)


def dataset_quality_checks(
    class_summary: list[dict[str, Any]],
    split_counts: dict[str, int],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[str] = []
    if split_counts.get("train", 0) <= 0:
        issues.append("missing train split")
    if split_counts.get("val", 0) <= 0:
        issues.append("missing validation split")
    empty_targets = sum(1 for row in manifest_rows if int(row.get("active_target_count", 0)) == 0)
    if empty_targets:
        issues.append(f"{empty_targets} samples have no valid target pixels")
    for row in class_summary:
        if int(row["positive_samples"]) == 0:
            issues.append(f"{row['class']} has no positive target samples")
    return {"status": "PASS" if not issues else "CHECK", "issues": issues}


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def relpath(target: Path, base: Path) -> str:
    return os.path.relpath(Path(target).resolve(), Path(base).resolve()).replace("\\", "/")


def repo_path(target: Path) -> str:
    path = Path(target)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(target)


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data)
    dirs = sample_dirs(data_root)
    if not dirs:
        raise SystemExit(f"No synthetic samples found under {data_root}")
    root = ROOT
    manifest_rows = []
    class_ratios: dict[str, list[float]] = {name: [] for name in TARGET_CHANNELS}
    pair_sums: dict[tuple[str, str], float] = {}
    pair_cooccurrences: dict[tuple[str, str], int] = {}
    for left_idx, left_name in enumerate(TARGET_CHANNELS):
        for right_name in TARGET_CHANNELS[left_idx + 1 :]:
            pair = (left_name, right_name)
            pair_sums[pair] = 0.0
            pair_cooccurrences[pair] = 0
    scratch_count = 0
    ring_cooccur = 0
    local_cooccur = 0
    stby_cooccur = 0
    stby_overlap_ratios = []
    gallery_candidates: list[dict[str, Any]] = []

    for path in dirs:
        sample = load_sample(path)
        wafer = sample.wafer_mask > 0
        valid = wafer & (sample.valid_test_mask > 0)
        denominator = int(valid.sum())
        split = stable_split(sample.sample_id, args.val_fraction, args.split_seed)
        manifest_rows.append(sample_manifest_row(path, sample, split, root))
        update_gallery_candidates(gallery_candidates, path, sample)

        class_masks: dict[str, np.ndarray] = {}
        raw_class_masks: dict[str, np.ndarray] = {}
        for class_name in TARGET_CHANNELS:
            raw_class_mask = (sample.pattern_masks[PATTERN_TO_INDEX[class_name]] > 0) & wafer
            class_mask = raw_class_mask & valid
            raw_class_masks[class_name] = raw_class_mask
            class_masks[class_name] = class_mask
            class_ratios[class_name].append(mask_ratio(class_mask, denominator))
        stby_mask = (sample.stby_mask > 0) & wafer

        overlap_valid = valid[:: args.overlap_stride, :: args.overlap_stride]
        overlap_denominator = int(overlap_valid.sum())
        for left_idx, left_name in enumerate(TARGET_CHANNELS):
            left = class_masks[left_name]
            for right_name in TARGET_CHANNELS[left_idx + 1 :]:
                right = class_masks[right_name]
                pair = (left_name, right_name)
                pair_cooccurrences[pair] += int(left.any() and right.any())
                pair_sums[pair] += mask_ratio(
                    left[:: args.overlap_stride, :: args.overlap_stride]
                    & right[:: args.overlap_stride, :: args.overlap_stride]
                    & overlap_valid,
                    overlap_denominator,
                )

        scratch = raw_class_masks["scratch"]
        if scratch.any():
            scratch_count += 1
            ring_cooccur += int(raw_class_masks["ring"].any())
            local_cooccur += int(raw_class_masks["local"].any())
            stby_cooccur += int(stby_mask.any())
            stby_overlap_ratios.append(float((scratch & stby_mask).sum() / max(int(scratch.sum()), 1)))

    split_counts = {
        "train": sum(1 for row in manifest_rows if row["split"] == "train"),
        "val": sum(1 for row in manifest_rows if row["split"] == "val"),
    }
    class_summary = summarize_class_accumulators(class_ratios)
    metrics = {
        "sample_count": len(dirs),
        "data_root": repo_path(data_root),
        "input_channels": list(INPUT_CHANNELS),
        "target_channels": list(TARGET_CHANNELS),
        "split_counts": split_counts,
        "overlap_stride": int(args.overlap_stride),
        "class_summary": class_summary,
        "overlap_summary": summarize_overlap_accumulators(pair_sums, pair_cooccurrences, len(dirs)),
        "quality_checks": dataset_quality_checks(class_summary, split_counts, manifest_rows),
        "scratch_risk": {
            "scratch_positive_samples": scratch_count,
            "scratch_ring_cooccurrence_rate": float(ring_cooccur / max(scratch_count, 1)),
            "scratch_local_cooccurrence_rate": float(local_cooccur / max(scratch_count, 1)),
            "scratch_stby_cooccurrence_rate": float(stby_cooccur / max(scratch_count, 1)),
            "scratch_pixels_hidden_by_stby_mean": float(np.mean(stby_overlap_ratios)) if stby_overlap_ratios else 0.0,
            "scratch_pixels_hidden_by_stby_p95": float(np.percentile(stby_overlap_ratios, 95)) if stby_overlap_ratios else 0.0,
        },
    }
    manifest_path = Path(args.manifest)
    metrics_path = Path(args.metrics)
    gallery_path = Path(args.gallery)
    out_path = Path(args.out)
    write_manifest(manifest_path, manifest_rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    gallery_samples = [load_sample(path) for path in selected_gallery_paths(gallery_candidates, args.max_gallery_rows)]
    save_gallery(gallery_samples, gallery_path, args.max_gallery_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_report(metrics, gallery_path, manifest_path, metrics_path, out_path), encoding="utf-8")
    return {
        "manifest": manifest_path,
        "metrics": metrics_path,
        "gallery": gallery_path,
        "report": out_path,
        "sample_count": len(dirs),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = build_outputs(args)
    print(f"Wrote segmentation readiness report: {outputs['report']}")
    print(f"Wrote metrics: {outputs['metrics']}")
    print(f"Wrote manifest: {outputs['manifest']}")
    print(f"Wrote gallery: {outputs['gallery']}")


if __name__ == "__main__":
    main()
