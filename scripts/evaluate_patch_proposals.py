"""Evaluate low-resolution proposal maps for high-resolution FBM defect review."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.data import PATTERN_CLASSES
from wafermap.features.spatial_pool import grid_edges, pooled_mean

PROPOSAL_CLASSES = ("scratch", "ring", "edge", "local", "shot_grid", "stby_pattern")


@dataclass
class ProposalSample:
    sample_id: str
    sample_dir: Path
    severity: NDArray[np.uint8]
    wafer_mask: NDArray[np.uint8]
    valid_test_mask: NDArray[np.uint8]
    stby_mask: NDArray[np.uint8]
    pattern_masks: NDArray[np.uint8]


@dataclass(frozen=True)
class Window:
    y0: int
    y1: int
    x0: int
    x1: int
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic/fbm_grouping_scale_pilot")
    parser.add_argument("--out", default="outputs/reports/fbm_patch_proposal_scale_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_patch_proposal_scale_metrics.json")
    parser.add_argument("--details", default="outputs/reports/fbm_patch_proposal_scale_details.csv")
    parser.add_argument("--gallery", default="outputs/figures/fbm_patch_proposal_scale_gallery.png")
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--window-cells", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def sample_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("synth_*") if (path / "arrays.npz").exists())


def load_proposal_sample(sample_dir: Path) -> ProposalSample:
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    arrays = np.load(sample_dir / "arrays.npz")
    return ProposalSample(
        sample_id=str(metadata["sample_id"]),
        sample_dir=sample_dir,
        severity=arrays["severity"],
        wafer_mask=arrays["wafer_mask"],
        valid_test_mask=arrays["valid_test_mask"],
        stby_mask=arrays["stby_mask"],
        pattern_masks=arrays["pattern_masks"],
    )


def grid_window_mean(score: NDArray[np.float32], window_cells: int) -> NDArray[np.float32]:
    if window_cells < 1:
        raise ValueError("window_cells must be >= 1")
    height, width = score.shape
    if window_cells > height or window_cells > width:
        raise ValueError("window_cells must fit inside the score grid")
    integral = np.pad(score.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")
    y0 = np.arange(0, height - window_cells + 1, dtype=np.int32)
    y1 = y0 + window_cells
    x0 = np.arange(0, width - window_cells + 1, dtype=np.int32)
    x1 = x0 + window_cells
    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    return sums / float(window_cells * window_cells)


def window_iou(left: Window, right: Window) -> float:
    y0 = max(left.y0, right.y0)
    y1 = min(left.y1, right.y1)
    x0 = max(left.x0, right.x0)
    x1 = min(left.x1, right.x1)
    inter = max(0, y1 - y0) * max(0, x1 - x0)
    if inter == 0:
        return 0.0
    left_area = (left.y1 - left.y0) * (left.x1 - left.x0)
    right_area = (right.y1 - right.y0) * (right.x1 - right.x0)
    return float(inter / max(left_area + right_area - inter, 1))


def select_top_windows(
    score: NDArray[np.float32],
    window_cells: int,
    top_k: int,
    nms_iou: float = 0.2,
) -> list[Window]:
    pooled = grid_window_mean(score, window_cells)
    candidates = [
        Window(int(y), int(y + window_cells), int(x), int(x + window_cells), float(pooled[y, x]))
        for y, x in np.ndindex(pooled.shape)
    ]
    candidates.sort(key=lambda item: item.score, reverse=True)
    selected: list[Window] = []
    for candidate in candidates:
        if all(window_iou(candidate, existing) <= nms_iou for existing in selected):
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def select_random_windows(
    valid_score: NDArray[np.float32],
    window_cells: int,
    top_k: int,
    rng: np.random.Generator,
) -> list[Window]:
    pooled = grid_window_mean(valid_score, window_cells)
    valid_positions = np.argwhere(pooled > 0.15)
    if len(valid_positions) == 0:
        valid_positions = np.argwhere(np.ones_like(pooled, dtype=bool))
    take = min(top_k, len(valid_positions))
    choices = rng.choice(len(valid_positions), size=take, replace=False)
    return [
        Window(int(valid_positions[idx, 0]), int(valid_positions[idx, 0] + window_cells), int(valid_positions[idx, 1]), int(valid_positions[idx, 1] + window_cells), 0.0)
        for idx in choices
    ]


def semantic_score_maps(sample: ProposalSample, grid_size: int) -> dict[str, NDArray[np.float32]]:
    y_edges, x_edges = grid_edges(sample.severity.shape, grid_size)
    severity = np.clip(sample.severity.astype(np.float32), 0, 7) / 7.0
    wafer = sample.wafer_mask > 0
    valid = sample.valid_test_mask > 0
    stby = sample.stby_mask > 0
    valid_w = valid.astype(np.float32)
    wafer_w = wafer.astype(np.float32)
    severity_mean = pooled_mean(severity, valid_w, y_edges, x_edges)
    fail_density = pooled_mean((sample.severity > 0).astype(np.float32), valid_w, y_edges, x_edges)
    high_grade = pooled_mean((sample.severity >= 6).astype(np.float32), valid_w, y_edges, x_edges)
    stby_ratio = pooled_mean(stby.astype(np.float32), wafer_w, y_edges, x_edges)
    wafer_ratio = pooled_mean(wafer.astype(np.float32), np.ones_like(wafer_w), y_edges, x_edges)
    yy, xx = np.mgrid[0:grid_size, 0:grid_size].astype(np.float32)
    cy = (grid_size - 1) / 2.0
    cx = (grid_size - 1) / 2.0
    radius = np.sqrt(((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2)
    radius = np.clip(radius, 0.0, 1.0).astype(np.float32)
    return {
        "severity": severity_mean,
        "fail": fail_density,
        "high": high_grade,
        "stby": stby_ratio,
        "wafer": wafer_ratio,
        "outer": radius,
    }


def class_score_map(class_name: str, maps: dict[str, NDArray[np.float32]]) -> NDArray[np.float32]:
    base = 0.45 * maps["severity"] + 0.35 * maps["fail"] + 0.35 * maps["high"] + 0.35 * maps["stby"]
    if class_name == "edge":
        score = base * (0.35 + maps["outer"])
    elif class_name == "local":
        score = 0.45 * maps["high"] + 0.35 * maps["severity"] + 0.25 * maps["fail"] + 0.25 * maps["stby"]
    elif class_name == "stby_pattern":
        score = maps["stby"] + 0.20 * maps["high"] + 0.10 * maps["severity"]
    elif class_name == "shot_grid":
        score = 0.55 * maps["fail"] + 0.35 * maps["severity"] + 0.20 * maps["high"]
    else:
        score = base
    return (score * maps["wafer"]).astype(np.float32)


def pixel_boxes(windows: list[Window], y_edges: NDArray[np.int32], x_edges: NDArray[np.int32]) -> list[tuple[int, int, int, int, float]]:
    return [
        (int(y_edges[w.y0]), int(y_edges[w.y1]), int(x_edges[w.x0]), int(x_edges[w.x1]), w.score)
        for w in windows
    ]


def proposal_recall(target: NDArray[np.bool_], boxes: list[tuple[int, int, int, int, float]]) -> float:
    target_pixels = int(target.sum())
    if target_pixels == 0:
        return 0.0
    covered = np.zeros(target.shape, dtype=bool)
    for y0, y1, x0, x1, _score in boxes:
        covered[y0:y1, x0:x1] = True
    return float(np.logical_and(target, covered).sum() / target_pixels)


def serialize_boxes(boxes: list[tuple[int, int, int, int, float]]) -> str:
    return ";".join(f"{y0}:{y1}:{x0}:{x1}:{score:.4f}" for y0, y1, x0, x1, score in boxes)


def evaluate_sample(
    sample: ProposalSample,
    grid_size: int,
    window_cells: int,
    top_k: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    y_edges, x_edges = grid_edges(sample.severity.shape, grid_size)
    maps = semantic_score_maps(sample, grid_size)
    rows: list[dict[str, Any]] = []
    for class_name in PROPOSAL_CLASSES:
        class_idx = PATTERN_CLASSES.index(class_name)
        target = sample.pattern_masks[class_idx] > 0
        target_pixels = int(target.sum())
        if target_pixels == 0:
            continue
        score = class_score_map(class_name, maps)
        proposals = pixel_boxes(select_top_windows(score, window_cells, top_k), y_edges, x_edges)
        random_windows = pixel_boxes(select_random_windows(maps["wafer"], window_cells, top_k, rng), y_edges, x_edges)
        proposal_cov = proposal_recall(target, proposals)
        random_cov = proposal_recall(target, random_windows)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "class_name": class_name,
                "target_pixels": target_pixels,
                "target_area_ratio": float(target_pixels / max(int(sample.wafer_mask.sum()), 1)),
                "proposal_recall": proposal_cov,
                "random_recall": random_cov,
                "hit_at_10": int(proposal_cov >= 0.10),
                "hit_at_30": int(proposal_cov >= 0.30),
                "random_hit_at_10": int(random_cov >= 0.10),
                "random_hit_at_30": int(random_cov >= 0.30),
                "proposal_boxes": serialize_boxes(proposals),
                "random_boxes": serialize_boxes(random_windows),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for class_name in PROPOSAL_CLASSES:
        items = [row for row in rows if row["class_name"] == class_name]
        if not items:
            summary[class_name] = {
                "positive_count": 0,
                "mean_proposal_recall": 0.0,
                "mean_random_recall": 0.0,
                "recall_lift": 0.0,
                "hit_at_10": 0.0,
                "hit_at_30": 0.0,
                "random_hit_at_10": 0.0,
                "random_hit_at_30": 0.0,
            }
            continue
        proposal = np.array([float(row["proposal_recall"]) for row in items], dtype=np.float32)
        random = np.array([float(row["random_recall"]) for row in items], dtype=np.float32)
        summary[class_name] = {
            "positive_count": len(items),
            "mean_target_area_ratio": float(np.mean([float(row["target_area_ratio"]) for row in items])),
            "mean_proposal_recall": float(proposal.mean()),
            "mean_random_recall": float(random.mean()),
            "recall_lift": float(proposal.mean() / max(float(random.mean()), 1e-9)),
            "hit_at_10": float(np.mean([int(row["hit_at_10"]) for row in items])),
            "hit_at_30": float(np.mean([int(row["hit_at_30"]) for row in items])),
            "random_hit_at_10": float(np.mean([int(row["random_hit_at_10"]) for row in items])),
            "random_hit_at_30": float(np.mean([int(row["random_hit_at_30"]) for row in items])),
        }
    return summary


def render_sample(sample: ProposalSample) -> NDArray[np.float32]:
    values = np.clip(sample.severity.astype(np.float32), 0, 7) / 7.0
    image = plt.get_cmap("turbo")(values)
    image[(sample.wafer_mask == 0) | ((sample.severity == 0) & (sample.stby_mask == 0))] = (0.0, 0.0, 0.0, 1.0)
    image[sample.stby_mask > 0] = (1.0, 1.0, 1.0, 1.0)
    return image


def parse_boxes(value: str) -> list[tuple[int, int, int, int, float]]:
    boxes = []
    if not value:
        return boxes
    for item in value.split(";"):
        y0, y1, x0, x1, score = item.split(":")
        boxes.append((int(y0), int(y1), int(x0), int(x1), float(score)))
    return boxes


def save_gallery(rows: list[dict[str, Any]], sample_by_id: dict[str, Path], out: Path) -> None:
    examples = []
    for class_name in ("scratch", "local", "stby_pattern", "edge"):
        candidates = [row for row in rows if row["class_name"] == class_name]
        if candidates:
            examples.append(max(candidates, key=lambda row: float(row["target_area_ratio"])))
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 2, figsize=(8.0, 4.0 * len(examples)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_idx, row in enumerate(examples):
        sample = load_proposal_sample(sample_by_id[row["sample_id"]])
        class_idx = PATTERN_CLASSES.index(row["class_name"])
        target = sample.pattern_masks[class_idx] > 0
        image = render_sample(sample)
        overlay = np.zeros((*target.shape, 4), dtype=np.float32)
        overlay[target] = (1.0, 0.0, 0.0, 0.38)
        boxes = parse_boxes(row["proposal_boxes"])
        for col, show_overlay in enumerate((False, True)):
            ax = axes[row_idx, col]
            ax.imshow(image, interpolation="nearest")
            if show_overlay:
                ax.imshow(overlay, interpolation="nearest")
            for y0, y1, x0, x1, score in boxes:
                rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#00ffff", linewidth=1.5)
                ax.add_patch(rect)
                ax.text(x0, y0, f"{score:.2f}", color="#00ffff", fontsize=7, va="top")
            title = f"{row['class_name']} {row['sample_id']} recall={float(row['proposal_recall']):.2f}"
            if show_overlay:
                title += " / oracle overlay"
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_details(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def relpath(target: Path, base_file: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.resolve().parent).replace("\\", "/")


def summary_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for class_name, item in metrics["summary_by_class"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(class_name)}</td>"
            f"<td>{item['positive_count']}</td>"
            f"<td>{item.get('mean_target_area_ratio', 0.0):.4f}</td>"
            f"<td>{item['mean_proposal_recall']:.3f}</td>"
            f"<td>{item['mean_random_recall']:.3f}</td>"
            f"<td>{item['recall_lift']:.2f}x</td>"
            f"<td>{item['hit_at_10']:.3f}</td>"
            f"<td>{item['hit_at_30']:.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_report(metrics: dict[str, Any], gallery: Path, details: Path, metrics_path: Path, out: Path) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>FBM Patch Proposal Readiness</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #eef6ff; border-left: 4px solid #2563eb; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    img {{ width: 100%; max-width: 1400px; border: 1px solid #d8dee9; border-radius: 8px; background: white; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM Patch Proposal Readiness</h1>
  <p>리사이즈 표현을 유사맵 검색 대체재로 쓰는 대신, 저해상도 semantic score map으로 원본 FBM에서 검토할 후보 window를 제안하는 실험이다.</p>
  <div class="note">Synthetic oracle mask는 proposal recall 채점과 gallery overlay에만 사용한다. Proposal score는 severity, fail density, high-grade density, stby ratio, wafer occupancy에서만 계산한다.</div>

  <h2>Class별 Proposal Recall</h2>
  <table>
    <tr><th>Class</th><th>Positive</th><th>Target Area</th><th>Proposal Recall</th><th>Random Recall</th><th>Lift</th><th>Hit@10%</th><th>Hit@30%</th></tr>
    {summary_rows(metrics)}
  </table>

  <h2>Proposal Gallery</h2>
  <img src="{html.escape(relpath(gallery, out))}" alt="patch proposal gallery">

  <h2>설정</h2>
  <table>
    <tr><td>Samples</td><td>{metrics['sample_count']}</td></tr>
    <tr><td>Grid size</td><td>{metrics['grid_size']}</td></tr>
    <tr><td>Window cells</td><td>{metrics['window_cells']}</td></tr>
    <tr><td>Top-K proposals</td><td>{metrics['top_k']}</td></tr>
  </table>

  <h2>Outputs</h2>
  <ul>
    <li>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>Details CSV: <code>{html.escape(relpath(details, out))}</code></li>
    <li>Gallery: <code>{html.escape(relpath(gallery, out))}</code></li>
  </ul>
</body>
</html>
"""


def evaluate(
    dirs: list[Path],
    grid_size: int,
    window_cells: int,
    top_k: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Path]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    sample_by_id: dict[str, Path] = {}
    for sample_dir in dirs:
        sample = load_proposal_sample(sample_dir)
        sample_by_id[sample.sample_id] = sample_dir
        rows.extend(evaluate_sample(sample, grid_size, window_cells, top_k, rng))
    metrics = {
        "sample_count": len(dirs),
        "grid_size": grid_size,
        "window_cells": window_cells,
        "top_k": top_k,
        "proposal_classes": list(PROPOSAL_CLASSES),
        "summary_by_class": summarize(rows),
    }
    return metrics, rows, sample_by_id


def main() -> None:
    args = parse_args()
    dirs = sample_dirs(Path(args.data))
    if not dirs:
        raise SystemExit(f"No samples found under {args.data}")
    metrics, rows, sample_by_id = evaluate(dirs, args.grid_size, args.window_cells, args.top_k, args.seed)
    metrics_path = Path(args.metrics)
    details_path = Path(args.details)
    gallery_path = Path(args.gallery)
    out_path = Path(args.out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_details(details_path, rows)
    save_gallery(rows, sample_by_id, gallery_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_report(metrics, gallery_path, details_path, metrics_path, out_path), encoding="utf-8")
    print(f"Wrote patch proposal report: {out_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote details: {details_path}")
    print(f"Wrote gallery: {gallery_path}")


if __name__ == "__main__":
    main()
