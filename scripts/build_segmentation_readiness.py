"""Build a CPU-only segmentation readiness report from synthetic FBM samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.data import PATTERN_CLASSES, SyntheticSample, load_sample

PATTERN_TO_INDEX = {name: idx for idx, name in enumerate(PATTERN_CLASSES)}
FOCUS_CLASSES = ("scratch", "ring", "local", "stby_pattern")
INPUT_CHANNELS = ("severity", "wafer_mask", "valid_test_mask", "stby_mask")


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
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    score = int(digest[:12], 16) / float(16**12)
    return "val" if score < val_fraction else "train"


def mask_ratio(mask: np.ndarray, denominator: int) -> float:
    return float(mask.sum() / max(denominator, 1))


def sample_manifest_row(sample_dir: Path, sample: SyntheticSample, split: str, root: Path) -> dict[str, Any]:
    wafer = sample.wafer_mask > 0
    denominator = int(wafer.sum())
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "split": split,
        "arrays_path": relpath(sample_dir / "arrays.npz", root),
        "metadata_path": relpath(sample_dir / "metadata.json", root),
        "actual_net_die": sample.metadata.get("actual_net_die", ""),
        "input_channels": "|".join(INPUT_CHANNELS),
        "target_channels": "|".join(PATTERN_CLASSES),
    }
    active_count = 0
    for class_name in PATTERN_CLASSES:
        class_mask = sample.pattern_masks[PATTERN_TO_INDEX[class_name]] > 0
        ratio = mask_ratio(class_mask & wafer, denominator)
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
                mask = sample.pattern_masks[PATTERN_TO_INDEX[column]] > 0
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
        scores[class_name] = float((sample.pattern_masks[PATTERN_TO_INDEX[class_name]] > 0).sum())
    scratch = sample.pattern_masks[PATTERN_TO_INDEX["scratch"]] > 0
    stby = sample.pattern_masks[PATTERN_TO_INDEX["stby_pattern"]] > 0
    scores["scratch_stby_overlap"] = float((scratch & stby).sum())
    scores["active_target_count"] = float(sum((sample.pattern_masks[idx] > 0).any() for idx in range(len(PATTERN_CLASSES))))
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
    for class_name in PATTERN_CLASSES:
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


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def relpath(target: Path, base: Path) -> str:
    return os.path.relpath(Path(target).resolve(), Path(base).resolve()).replace("\\", "/")


def repo_path(target: Path) -> str:
    path = Path(target)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(target)


def summary_rows(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{html.escape(str(row['class']))}</td>"
            f"<td>{row['positive_samples']}</td>"
            f"<td>{fmt(row['sample_presence_rate'])}</td>"
            f"<td>{fmt(row['mean_pixel_ratio'])}</td>"
            f"<td>{fmt(row['median_positive_ratio'])}</td>"
            f"<td>{fmt(row['p95_positive_ratio'])}</td>"
            f"<td>{fmt(row['suggested_pos_weight_capped'])}</td>"
            "</tr>"
        )
    return "\n".join(out)


def overlap_rows(rows: list[dict[str, Any]], limit: int = 10) -> str:
    out = []
    for row in rows[:limit]:
        out.append(
            "<tr>"
            f"<td>{html.escape(str(row['pair']))}</td>"
            f"<td>{fmt(row['cooccurrence_rate'])}</td>"
            f"<td>{fmt(row['mean_overlap_pixel_ratio'])}</td>"
            "</tr>"
        )
    return "\n".join(out)


def html_report(metrics: dict[str, Any], gallery: Path, manifest: Path, metrics_path: Path, out: Path) -> str:
    scratch = metrics["scratch_risk"]
    conclusion = (
        "scratch는 ring/local/stby와 자주 중첩되므로 wafer-level retrieval feature만으로 분리하기 어렵다. "
        "다음 단계는 synthetic mask 기반 multi-label segmentation 또는 scratch-specific representation으로 가는 것이 맞다."
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Segmentation Readiness 중간점검</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #eef6ff; border-left: 4px solid #2563eb; padding: 12px 14px; margin: 14px 0; }}
    .warn {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    img {{ width: 100%; max-width: 1400px; border: 1px solid #d8dee9; border-radius: 8px; background: white; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM Segmentation Readiness 중간점검</h1>
  <p>목적은 GPU 학습 전에 synthetic target mask가 어떤 난이도와 불균형을 갖는지 확인하고, scratch/local을 다음 모델 단계로 넘길 근거를 정리하는 것이다.</p>
  <div class="note">Synthetic mask는 학습/검증 target이다. 실제 inference feature에는 <code>pattern_masks</code>, <code>label_*</code>, <code>*_mask_ratio</code>를 넣지 않는다.</div>

  <h2>Executive Summary</h2>
  <ul>
    <li>데이터셋: {metrics['sample_count']} samples, train {metrics['split_counts']['train']} / val {metrics['split_counts']['val']}</li>
    <li>입력 semantic channels: {html.escape(', '.join(INPUT_CHANNELS))}</li>
    <li>target channels: {html.escape(', '.join(PATTERN_CLASSES))}</li>
    <li>{html.escape(conclusion)}</li>
  </ul>

  <h2>Scratch Risk Check</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>해석</th></tr>
    <tr><td>scratch positive samples</td><td>{scratch['scratch_positive_samples']}</td><td>scratch target 학습 후보 수</td></tr>
    <tr><td>scratch + ring co-occurrence</td><td>{fmt(scratch['scratch_ring_cooccurrence_rate'])}</td><td>ring과 섞이면 wafer-level feature가 scratch를 놓치기 쉽다.</td></tr>
    <tr><td>scratch + local co-occurrence</td><td>{fmt(scratch['scratch_local_cooccurrence_rate'])}</td><td>작은 blob과 같이 보이면 관심 기준별 검색이 흔들릴 수 있다.</td></tr>
    <tr><td>scratch + stby co-occurrence</td><td>{fmt(scratch['scratch_stby_cooccurrence_rate'])}</td><td>scratch 시작점/충돌점이 stby로 가려질 가능성.</td></tr>
    <tr><td>scratch pixels hidden by stby mean/p95</td><td>{fmt(scratch['scratch_pixels_hidden_by_stby_mean'])} / {fmt(scratch['scratch_pixels_hidden_by_stby_p95'])}</td><td>stby가 scratch 관측을 얼마나 가리는지의 synthetic proxy.</td></tr>
  </table>

  <h2>Class Balance</h2>
  <table>
    <tr><th>Class</th><th>Positive samples</th><th>Presence</th><th>Mean pixel ratio</th><th>Median positive</th><th>P95 positive</th><th>Suggested pos weight</th></tr>
    {summary_rows(metrics['class_summary'])}
  </table>

  <h2>Overlap Top Pairs</h2>
  <table>
    <tr><th>Pair</th><th>Sample co-occurrence</th><th>Mean overlap pixel ratio</th></tr>
    {overlap_rows(metrics['overlap_summary'])}
  </table>

  <h2>Mask Gallery</h2>
  <p>각 행은 대표 sample이다. 왼쪽은 input severity이고, 오른쪽은 scratch/ring/local/stby target mask overlay다.</p>
  <img src="{html.escape(relpath(gallery, out.parent))}" alt="segmentation readiness gallery">

  <h2>다음 확인 단계</h2>
  <ol>
    <li>local은 현재 morphology baseline 결과를 expert review form에 연결한다.</li>
    <li>scratch는 이 manifest를 이용해 작은 U-Net/SegFormer 계열 multi-label segmentation으로 넘긴다.</li>
    <li>학습 target은 class별 sigmoid mask이며 overlap을 허용한다.</li>
    <li>성공 기준은 전체 mIoU가 아니라 scratch recall, stby-hidden scratch recall, local small-blob recall이다.</li>
  </ol>

  <h2>Outputs</h2>
  <ul>
    <li>Manifest CSV: <code>{html.escape(relpath(manifest, out.parent))}</code></li>
    <li>Metrics JSON: <code>{html.escape(relpath(metrics_path, out.parent))}</code></li>
    <li>Gallery: <code>{html.escape(relpath(gallery, out.parent))}</code></li>
  </ul>
</body>
</html>
"""


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data)
    dirs = sample_dirs(data_root)
    if not dirs:
        raise SystemExit(f"No synthetic samples found under {data_root}")
    root = ROOT
    manifest_rows = []
    class_ratios: dict[str, list[float]] = {name: [] for name in PATTERN_CLASSES}
    pair_sums: dict[tuple[str, str], float] = {}
    pair_cooccurrences: dict[tuple[str, str], int] = {}
    for left_idx, left_name in enumerate(PATTERN_CLASSES):
        for right_name in PATTERN_CLASSES[left_idx + 1 :]:
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
        denominator = int(wafer.sum())
        split = stable_split(sample.sample_id, args.val_fraction, args.split_seed)
        manifest_rows.append(sample_manifest_row(path, sample, split, root))
        update_gallery_candidates(gallery_candidates, path, sample)

        class_masks: dict[str, np.ndarray] = {}
        for class_name in PATTERN_CLASSES:
            class_mask = (sample.pattern_masks[PATTERN_TO_INDEX[class_name]] > 0) & wafer
            class_masks[class_name] = class_mask
            class_ratios[class_name].append(mask_ratio(class_mask, denominator))

        overlap_wafer = wafer[:: args.overlap_stride, :: args.overlap_stride]
        overlap_denominator = int(overlap_wafer.sum())
        for left_idx, left_name in enumerate(PATTERN_CLASSES):
            left = class_masks[left_name]
            for right_name in PATTERN_CLASSES[left_idx + 1 :]:
                right = class_masks[right_name]
                pair = (left_name, right_name)
                pair_cooccurrences[pair] += int(left.any() and right.any())
                pair_sums[pair] += mask_ratio(
                    left[:: args.overlap_stride, :: args.overlap_stride]
                    & right[:: args.overlap_stride, :: args.overlap_stride]
                    & overlap_wafer,
                    overlap_denominator,
                )

        scratch = class_masks["scratch"]
        if scratch.any():
            scratch_count += 1
            ring_cooccur += int(class_masks["ring"].any())
            local_cooccur += int(class_masks["local"].any())
            stby_cooccur += int(class_masks["stby_pattern"].any())
            stby_overlap_ratios.append(float((scratch & class_masks["stby_pattern"]).sum() / max(int(scratch.sum()), 1)))

    split_counts = {
        "train": sum(1 for row in manifest_rows if row["split"] == "train"),
        "val": sum(1 for row in manifest_rows if row["split"] == "val"),
    }
    metrics = {
        "sample_count": len(dirs),
        "data_root": repo_path(data_root),
        "input_channels": list(INPUT_CHANNELS),
        "target_channels": list(PATTERN_CLASSES),
        "split_counts": split_counts,
        "overlap_stride": int(args.overlap_stride),
        "class_summary": summarize_class_accumulators(class_ratios),
        "overlap_summary": summarize_overlap_accumulators(pair_sums, pair_cooccurrences, len(dirs)),
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
