"""Summarize segmentation masks into structured defect feature tables."""

from __future__ import annotations

import argparse
import csv
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

from wafermap.data import PATTERN_CLASSES, load_sample
from wafermap.reporting import summarize_sample_defects


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic/fbm_grouping_scale_pilot")
    parser.add_argument("--out", default="outputs/reports/fbm_defect_location_summary_report.html")
    parser.add_argument("--csv-out", default="outputs/reports/fbm_defect_location_summary.csv")
    parser.add_argument("--metrics", default="outputs/reports/fbm_defect_location_summary_metrics.json")
    parser.add_argument("--gallery", default="outputs/figures/fbm_defect_location_summary_gallery.png")
    parser.add_argument("--max-gallery-rows", type=int, default=6)
    return parser.parse_args(argv)


def sample_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if (path / "arrays.npz").exists())


def summary_to_row(summary: Any) -> dict[str, Any]:
    return {
        "sample_id": summary.sample_id,
        "class_name": summary.class_name,
        "feature_key": summary.feature_key,
        "pixel_ratio": summary.pixel_ratio,
        "centroid_clock": summary.centroid_clock,
        "location_label": summary.location_label,
        "radial_zone": summary.radial_zone,
        "top_clock_positions": "|".join(summary.top_clock_positions),
        "top_sector_share": summary.top_sector_share,
        "stby_overlap_ratio": summary.stby_overlap_ratio,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_gallery_samples(sample_summaries: dict[str, list[Any]], max_rows: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    selectors = [
        lambda rows: max((row.stby_overlap_ratio for row in rows if row.class_name == "scratch"), default=-1.0),
        lambda rows: max((row.pixel_ratio for row in rows if row.class_name == "scratch"), default=-1.0),
        lambda rows: max((row.pixel_ratio for row in rows if row.class_name == "local"), default=-1.0),
        lambda rows: max((row.pixel_ratio for row in rows if row.class_name == "ring"), default=-1.0),
        lambda rows: max((row.pixel_ratio for row in rows if row.class_name == "edge"), default=-1.0),
        lambda rows: len(rows),
    ]
    for selector in selectors:
        if len(selected) >= max_rows:
            break
        candidates = [(sample_id, rows) for sample_id, rows in sample_summaries.items() if sample_id not in seen]
        if not candidates:
            break
        sample_id, rows = max(candidates, key=lambda item: selector(item[1]))
        if selector(rows) <= 0 and selected:
            continue
        selected.append(sample_id)
        seen.add(sample_id)
    return selected


def save_gallery(data_root: Path, selected_ids: list[str], out: Path) -> None:
    if not selected_ids:
        return
    fig, axes = plt.subplots(len(selected_ids), 2, figsize=(9, 3.6 * len(selected_ids)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    colors = {
        "scratch": "autumn",
        "ring": "winter",
        "edge": "summer",
        "local": "spring",
        "shot_grid": "cool",
        "stby_pattern": "gray",
    }
    for row_idx, sample_id in enumerate(selected_ids):
        sample = load_sample(data_root / sample_id)
        stride = max(1, int(np.ceil(max(sample.shape) / 700)))
        base = sample.severity.astype(np.float32) / 7.0
        base[sample.wafer_mask == 0] = np.nan
        ax0, ax1 = axes[row_idx]
        ax0.imshow(base[::stride, ::stride], cmap="turbo", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax0.set_title(f"{sample_id} input", fontsize=9)
        ax0.axis("off")
        ax1.imshow(base[::stride, ::stride], cmap="turbo", vmin=0.0, vmax=1.0, interpolation="nearest")
        for class_name, cmap in colors.items():
            idx = PATTERN_CLASSES.index(class_name)
            mask = sample.pattern_masks[idx] > 0
            overlay = np.ma.masked_where(~mask[::stride, ::stride], mask[::stride, ::stride])
            ax1.imshow(overlay, cmap=cmap, alpha=0.42, interpolation="nearest")
        ax1.set_title(f"{sample_id} oracle masks", fontsize=9)
        ax1.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def metrics_from_rows(rows: list[dict[str, Any]], sample_count: int) -> dict[str, Any]:
    by_class = {}
    for class_name in ("scratch", "ring", "edge", "local", "shot_grid", "stby_pattern"):
        class_rows = [row for row in rows if row["class_name"] == class_name]
        hidden = [float(row["stby_overlap_ratio"]) for row in class_rows]
        by_class[class_name] = {
            "sample_count": len({row["sample_id"] for row in class_rows}),
            "presence_rate": float(len({row["sample_id"] for row in class_rows}) / max(sample_count, 1)),
            "mean_pixel_ratio": float(np.mean([float(row["pixel_ratio"]) for row in class_rows])) if class_rows else 0.0,
            "mean_stby_overlap": float(np.mean(hidden)) if hidden else 0.0,
            "p95_stby_overlap": float(np.percentile(hidden, 95)) if hidden else 0.0,
        }
    return {
        "sample_count": sample_count,
        "summary_row_count": len(rows),
        "by_class": by_class,
    }


def relpath(target: Path, base_file: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.resolve().parent).replace("\\", "/")


def class_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for class_name, item in metrics["by_class"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(class_name)}</td>"
            f"<td>{item['sample_count']}</td>"
            f"<td>{item['presence_rate']:.3f}</td>"
            f"<td>{item['mean_pixel_ratio']:.5f}</td>"
            f"<td>{item['mean_stby_overlap']:.3f}</td>"
            f"<td>{item['p95_stby_overlap']:.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def selected_feature_rows(sample_summaries: dict[str, list[Any]], selected_ids: list[str]) -> str:
    rows = []
    for sample_id in selected_ids:
        for summary in sample_summaries[sample_id][:4]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(sample_id)}</td>"
                f"<td>{html.escape(summary.feature_key)}</td>"
                f"<td>{html.escape(summary.class_name)}</td>"
                f"<td>{html.escape(summary.location_label)}</td>"
                f"<td>{html.escape(summary.radial_zone)}</td>"
                f"<td>{summary.pixel_ratio:.5f}</td>"
                f"<td>{summary.stby_overlap_ratio:.3f}</td>"
                "</tr>"
            )
    return "\n".join(rows)


def html_report(
    metrics: dict[str, Any],
    sample_summaries: dict[str, list[Any]],
    selected_ids: list[str],
    gallery: Path,
    csv_path: Path,
    metrics_path: Path,
    out: Path,
) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Defect Feature Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    img {{ width: 100%; max-width: 1400px; border: 1px solid #d8dee9; border-radius: 8px; background: white; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM Defect Feature Summary</h1>
  <div class="note">이 리포트는 synthetic oracle mask를 사용한 upper-bound feature extraction 검증이다. 실제 운영에서는 model-predicted mask가 같은 후처리 입력이 된다.</div>
  <h2>Downstream Feature Row 예시</h2>
  <table>
    <tr><th>Sample</th><th>Feature key</th><th>Class</th><th>Location</th><th>Radial zone</th><th>Area ratio</th><th>Stby overlap</th></tr>
    {selected_feature_rows(sample_summaries, selected_ids)}
  </table>
  <h2>Class별 Defect Feature 통계</h2>
  <table>
    <tr><th>Class</th><th>Samples</th><th>Presence</th><th>Mean pixel ratio</th><th>Mean stby overlap</th><th>P95 stby overlap</th></tr>
    {class_rows(metrics)}
  </table>
  <h2>대표 Sample Gallery</h2>
  <img src="{html.escape(relpath(gallery, out))}" alt="defect feature summary gallery">
  <h2>Outputs</h2>
  <ul>
    <li>Summary CSV: <code>{html.escape(relpath(csv_path, out))}</code></li>
    <li>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>Gallery: <code>{html.escape(relpath(gallery, out))}</code></li>
  </ul>
</body>
</html>
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data)
    dirs = sample_dirs(data_root)
    rows: list[dict[str, Any]] = []
    sample_summaries: dict[str, list[Any]] = {}
    for sample_dir in dirs:
        sample = load_sample(sample_dir)
        summaries = summarize_sample_defects(sample)
        sample_summaries[sample.sample_id] = summaries
        rows.extend(summary_to_row(summary) for summary in summaries)
    if not rows:
        raise SystemExit(f"No defect summaries generated under {data_root}")
    metrics = metrics_from_rows(rows, len(dirs))
    selected_ids = select_gallery_samples(sample_summaries, args.max_gallery_rows)
    csv_path = Path(args.csv_out)
    metrics_path = Path(args.metrics)
    gallery_path = Path(args.gallery)
    out_path = Path(args.out)
    write_csv(csv_path, rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_gallery(data_root, selected_ids, gallery_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        html_report(metrics, sample_summaries, selected_ids, gallery_path, csv_path, metrics_path, out_path),
        encoding="utf-8",
    )
    return {
        "report": out_path,
        "csv": csv_path,
        "metrics": metrics_path,
        "gallery": gallery_path,
        "rows": len(rows),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run(args)
    print(f"Wrote defect feature summary report: {outputs['report']}")
    print(f"Wrote summary rows: {outputs['csv']}")
    print(f"Wrote metrics: {outputs['metrics']}")
    print(f"Wrote gallery: {outputs['gallery']}")


if __name__ == "__main__":
    main()
