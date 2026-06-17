"""Create an HTML report for a generated synthetic wafer review batch."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.data import PATTERN_CLASSES, SyntheticSample, load_sample
from wafermap.evaluation import validate_synthetic_sample
from wafermap.features import extract_feature_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic/final_review")
    parser.add_argument("--out", default="outputs/reports/final_review_report.html")
    parser.add_argument("--gallery", default="outputs/figures/final_review_gallery.png")
    parser.add_argument("--metrics", default="outputs/reports/final_review_metrics.json")
    parser.add_argument("--features", default="outputs/reports/final_review_features.csv")
    return parser.parse_args()


def sample_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("synth_*") if path.is_dir())


def radial_masks(sample: SyntheticSample) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(sample.shape)
    cx = (sample.shape[1] - 1) / 2.0
    cy = (sample.shape[0] - 1) / 2.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_distance = float(distance[sample.wafer_mask > 0].max())
    radius = distance / max(max_distance, 1.0)
    valid = sample.valid_test_mask > 0
    center = valid & (radius < 0.55)
    edge = valid & (radius > 0.78)
    return center, edge


def edge_chip_face_masks(sample: SyntheticSample) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(sample.shape)
    cx = (sample.shape[1] - 1) / 2.0
    cy = (sample.shape[0] - 1) / 2.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_distance = float(distance[sample.wafer_mask > 0].max())
    radius = distance / max(max_distance, 1.0)
    valid = sample.valid_test_mask > 0
    inner_face = np.zeros(sample.shape, dtype=bool)
    outer_face = np.zeros(sample.shape, dtype=bool)

    for chip_id in np.unique(sample.chip_index[sample.chip_index >= 0]):
        chip = sample.chip_index == chip_id
        chip_radius = radius[chip]
        if chip_radius.max() < 0.78:
            continue
        radius_min = chip_radius.min()
        radius_max = chip_radius.max()
        if radius_max <= radius_min:
            continue
        local_rank = (radius - radius_min) / (radius_max - radius_min)
        inner_face |= chip & valid & (local_rank < 0.34)
        outer_face |= chip & valid & (local_rank > 0.66)
    return inner_face, outer_face


def summarize_sample(sample: SyntheticSample) -> dict[str, Any]:
    valid = sample.valid_test_mask > 0
    center, edge = radial_masks(sample)
    edge_chip_inner, edge_chip_outer = edge_chip_face_masks(sample)
    grade_hist = {str(g): int(((sample.severity == g) & valid).sum()) for g in range(8)}
    valid_count = max(int(valid.sum()), 1)
    patterns = sample.metadata.get("patterns", [])
    stby_area = int(sample.stby_mask.sum())
    chip_area = int(sample.metadata["chip_blocks"]["width"] * sample.metadata["chip_blocks"]["height"])
    edge_chip_inner_density = float(
        ((sample.severity > 0) & edge_chip_inner).sum() / max(int(edge_chip_inner.sum()), 1)
    )
    edge_chip_outer_density = float(
        ((sample.severity > 0) & edge_chip_outer).sum() / max(int(edge_chip_outer.sum()), 1)
    )
    return {
        "sample_id": sample.sample_id,
        "shape": list(sample.shape),
        "actual_net_die": int(sample.metadata["actual_net_die"]),
        "grade_hist": grade_hist,
        "grade0_ratio": grade_hist["0"] / valid_count,
        "center_mean_grade": float(sample.severity[center].mean()) if center.any() else 0.0,
        "edge_mean_grade": float(sample.severity[edge].mean()) if edge.any() else 0.0,
        "center_fail_density": float(((sample.severity > 0) & center).sum() / max(int(center.sum()), 1)),
        "edge_fail_density": float(((sample.severity > 0) & edge).sum() / max(int(edge.sum()), 1)),
        "edge_chip_inner_face_fail_density": edge_chip_inner_density,
        "edge_chip_outer_face_fail_density": edge_chip_outer_density,
        "edge_chip_outer_minus_inner_fail_density": edge_chip_outer_density - edge_chip_inner_density,
        "stby_pixels": stby_area,
        "stby_chips_est": stby_area / max(chip_area, 1),
        "patterns": patterns,
        "has_flow": any(p.get("type") == "flow" for p in patterns),
        "validation_errors": validate_synthetic_sample(sample),
    }


def write_feature_csv(samples: list[SyntheticSample], path: Path) -> None:
    rows = []
    for sample in samples:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "actual_net_die": sample.metadata["actual_net_die"],
                **extract_feature_vector(sample),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compact_pattern_label(pattern: dict[str, Any]) -> str:
    ptype = str(pattern.get("type", "?"))
    params = pattern.get("parameters", {})
    mode = str(params.get("mode", ""))
    if ptype == "stby_pattern":
        seeded = int(params.get("seeded_stby_chip_count", 0))
        latent = int(params.get("latent_weighted_stby_chip_count_est", 0))
        if "seeded_stby_chip_count" not in params and mode.startswith("origin_coupled"):
            return "stby:origin-unverified"
        if seeded > 0 and latent > 0:
            return "stby:origin+latent"
        if seeded > 0:
            return "stby:origin"
        if latent > 0 or mode.startswith("latent_weighted"):
            return "stby:latent"
        return "stby"
    if ptype == "local":
        return {
            "single_blob": "local:single",
            "double_blob": "local:double",
            "triple_triangle": "local:tri",
        }.get(mode, "local")
    if ptype == "shot_grid":
        anchor = str(pattern.get("parameters", {}).get("anchor_region", ""))
        return f"shot:{anchor.replace('lower_left', 'LL')}" if anchor else "shot"
    if ptype in {"scratch", "ring"} and mode:
        return f"{ptype}:{mode}"
    return ptype


def create_gallery(samples: list[SyntheticSample], path: Path) -> None:
    cols = 3
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), constrained_layout=True)
    axes = np.array(axes).ravel()
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#050505")

    for ax, sample in zip(axes, samples, strict=False):
        image = sample.severity.astype(float)
        image[sample.wafer_mask == 0] = np.nan
        image[sample.stby_mask > 0] = 7.0
        im = ax.imshow(image, vmin=0, vmax=7, cmap=cmap, interpolation="nearest")
        labels = [compact_pattern_label(p) for p in sample.metadata["patterns"]]
        label = ", ".join(labels) or "background"
        ax.set_title(f"{sample.sample_id}\n{label}", fontsize=7)
        ax.axis("off")

    for ax in axes[len(samples) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.72, label="Grade")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def batch_metrics(samples: list[SyntheticSample]) -> dict[str, Any]:
    summaries = [summarize_sample(sample) for sample in samples]
    pattern_counter: Counter[str] = Counter()
    mode_counter: Counter[str] = Counter()
    local_mode_counter: Counter[str] = Counter()
    scratch_mode_counter: Counter[str] = Counter()
    ring_mode_counter: Counter[str] = Counter()
    shot_mode_counter: Counter[str] = Counter()
    seeded_stby_chip_count = 0
    latent_stby_chip_count = 0
    legacy_unverified_stby_origin_count = 0
    for summary in summaries:
        for pattern in summary["patterns"]:
            ptype = pattern.get("type", "?")
            params = pattern.get("parameters", {})
            mode = params.get("mode")
            pattern_counter[str(ptype)] += 1
            if ptype == "stby_pattern":
                if "seeded_stby_chip_count" not in params and str(mode).startswith("origin_coupled"):
                    legacy_unverified_stby_origin_count += 1
                seeded_stby_chip_count += int(params.get("seeded_stby_chip_count", 0))
                latent_stby_chip_count += int(params.get("latent_weighted_stby_chip_count_est", 0))
            if mode:
                mode_counter[str(mode)] += 1
                if ptype == "local":
                    local_mode_counter[str(mode)] += 1
                if ptype == "scratch":
                    scratch_mode_counter[str(mode)] += 1
                if ptype == "ring":
                    ring_mode_counter[str(mode)] += 1
                if ptype == "shot_grid":
                    shot_mode_counter[str(mode)] += 1

    grade0 = [s["grade0_ratio"] for s in summaries]
    edge_minus_center = [s["edge_fail_density"] - s["center_fail_density"] for s in summaries]
    edge_chip_outer_minus_inner = [
        s["edge_chip_outer_minus_inner_fail_density"] for s in summaries
    ]
    all_errors = {s["sample_id"]: s["validation_errors"] for s in summaries if s["validation_errors"]}
    acceptance = {
        "all_samples_internal_valid": not all_errors,
        "no_flow_patterns": not any(s["has_flow"] for s in summaries),
        "pattern_classes_no_flow": "flow" not in PATTERN_CLASSES,
        "pattern_classes_include_ring": "ring" in PATTERN_CLASSES,
        "pattern_classes_include_shot_grid": "shot_grid" in PATTERN_CLASSES,
        "grade0_present_all_samples": all(s["grade0_ratio"] > 0 for s in summaries),
        "edge_fail_density_higher_majority": sum(v > 0 for v in edge_minus_center) >= math.ceil(len(samples) * 0.6),
        "edge_chip_outer_face_higher_majority": sum(v > 0 for v in edge_chip_outer_minus_inner)
        >= math.ceil(len(samples) * 0.6),
        "local_modes_cover_single_double_triple": {"single_blob", "double_blob", "triple_triangle"}.issubset(
            set(local_mode_counter)
        ),
        "stby_present_all_samples": all(s["stby_pixels"] > 0 for s in summaries),
        "origin_coupled_stby_present": mode_counter["origin_coupled_or_random_chip_missing"] > 0,
        "shot_grid_present": pattern_counter["shot_grid"] > 0,
    }
    return {
        "sample_count": len(samples),
        "pattern_classes": list(PATTERN_CLASSES),
        "pattern_counts": dict(pattern_counter),
        "mode_counts": dict(mode_counter),
        "local_mode_counts": dict(local_mode_counter),
        "scratch_mode_counts": dict(scratch_mode_counter),
        "ring_mode_counts": dict(ring_mode_counter),
        "shot_mode_counts": dict(shot_mode_counter),
        "seeded_stby_chip_count": seeded_stby_chip_count,
        "latent_weighted_stby_chip_count": latent_stby_chip_count,
        "legacy_unverified_stby_origin_count": legacy_unverified_stby_origin_count,
        "grade0_ratio_mean": float(np.mean(grade0)),
        "grade0_ratio_min": float(np.min(grade0)),
        "grade0_ratio_max": float(np.max(grade0)),
        "edge_minus_center_fail_density_mean": float(np.mean(edge_minus_center)),
        "edge_fail_density_higher_count": int(sum(v > 0 for v in edge_minus_center)),
        "edge_chip_outer_minus_inner_fail_density_mean": float(np.mean(edge_chip_outer_minus_inner)),
        "edge_chip_outer_face_higher_count": int(sum(v > 0 for v in edge_chip_outer_minus_inner)),
        "acceptance": acceptance,
        "validation_errors": all_errors,
        "samples": summaries,
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def relpath(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), base_file.resolve().parent)).as_posix()


def html_report(metrics: dict[str, Any], gallery: Path, features: Path, metrics_path: Path, out: Path) -> str:
    acceptance_rows = "\n".join(
        f"<tr><td>{html.escape(key)}</td><td class=\"{'pass' if value else 'fail'}\">{'PASS' if value else 'FAIL'}</td></tr>"
        for key, value in metrics["acceptance"].items()
    )
    sample_rows = []
    for sample in metrics["samples"]:
        patterns = []
        for pattern in sample["patterns"]:
            mode = pattern.get("parameters", {}).get("mode", "")
            label = f"{pattern.get('type')}" + (f":{mode}" if mode else "")
            patterns.append(label)
        sample_rows.append(
            "<tr>"
            f"<td>{html.escape(sample['sample_id'])}</td>"
            f"<td>{pct(sample['grade0_ratio'])}</td>"
            f"<td>{sample['center_fail_density']:.4f}</td>"
            f"<td>{sample['edge_fail_density']:.4f}</td>"
            f"<td>{sample['edge_chip_outer_minus_inner_fail_density']:.4f}</td>"
            f"<td>{sample['stby_chips_est']:.1f}</td>"
            f"<td>{html.escape(', '.join(patterns))}</td>"
            "</tr>"
        )
    sample_table = "\n".join(sample_rows)
    local_modes = ", ".join(f"{k}: {v}" for k, v in metrics["local_mode_counts"].items())
    scratch_modes = ", ".join(f"{k}: {v}" for k, v in metrics["scratch_mode_counts"].items())
    ring_modes = ", ".join(f"{k}: {v}" for k, v in metrics["ring_mode_counts"].items())
    shot_modes = ", ".join(f"{k}: {v}" for k, v in metrics["shot_mode_counts"].items())
    pattern_counts = ", ".join(f"{k}: {v}" for k, v in metrics["pattern_counts"].items())
    stby_origin_summary = (
        f"seeded origin chips: {metrics['seeded_stby_chip_count']}, "
        f"latent-weighted chips: {metrics['latent_weighted_stby_chip_count']}, "
        f"legacy unverified origin instances: {metrics['legacy_unverified_stby_origin_count']}"
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>WaferMap Synthetic FBM Review Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; background: #f8fafc; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    .pass {{ color: #096b3b; font-weight: 700; }}
    .fail {{ color: #b42318; font-weight: 700; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee9; border-radius: 8px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>WaferMap Synthetic Fail Bit Map Review</h1>
  <p>This report was generated from the final review dataset <code>data/synthetic/final_review</code>. The goal is to build a plausible synthetic Fail Bit Map dataset without using confidential real wafer data, then verify whether it is suitable as a foundation for FBM information extraction, similarity search, coarse grouping, defect scoring, and later segmentation modeling. ANOVA is treated as a downstream analysis after process metadata is joined.</p>

  <div class="note">
    <strong>Key revision:</strong> flow and water-streak-like defects are excluded. The active synthetic classes are <code>scratch</code>, <code>ring</code>, <code>edge</code>, <code>local</code>, <code>random</code>, <code>shot_grid</code>, and <code>stby_pattern</code>. The local class includes single, double, and triple-triangle blob variants. <code>shot_grid</code> represents repeated photo-shot behavior where the same relative region inside each shot field, such as lower-left or shot-edge area, has subtly higher fail bits. Stby can now be origin-coupled, meaning a scratch contact point or local impact origin can become a chip-level missing-test region that hides the fail-bit pattern underneath.
  </div>

  <div class="summary">
    <div class="card"><div>Samples</div><div class="metric">{metrics['sample_count']}</div></div>
    <div class="card"><div>Mean Grade 0 Ratio</div><div class="metric">{pct(metrics['grade0_ratio_mean'])}</div></div>
    <div class="card"><div>Edge &gt; Center Count</div><div class="metric">{metrics['edge_fail_density_higher_count']}/{metrics['sample_count']}</div></div>
    <div class="card"><div>Edge-Chip Outer &gt; Inner</div><div class="metric">{metrics['edge_chip_outer_face_higher_count']}/{metrics['sample_count']}</div></div>
  </div>

  <h2>Gallery</h2>
  <p>This gallery uses a Grade colormap. None-wafer is black, and in-wafer Grade 0 is also visually dark, so <code>wafer_mask</code> remains mandatory. Stby chips are separate missing-test masks and are rendered using the Grade 7 color for visual review only. In origin-coupled cases, the Stby chip is intentionally allowed to cover the defect source. Edge statistics use center-distance polar radius, normalized as <code>r=0</code> at wafer center and <code>r=1</code> at the farthest in-wafer cell. Edge-chip face checks compare the inner third of each edge chip against the outer third closer to wafer edge.</p>
  <img src="{html.escape(relpath(gallery, out))}" alt="final review gallery">

  <h2>Acceptance Checks</h2>
  <table>
    <tr><th>Check</th><th>Status</th></tr>
    {acceptance_rows}
  </table>

  <h2>Pattern And Mode Coverage</h2>
  <table>
    <tr><th>Item</th><th>Observed</th></tr>
    <tr><td>Pattern classes</td><td>{html.escape(', '.join(metrics['pattern_classes']))}</td></tr>
    <tr><td>Pattern counts</td><td>{html.escape(pattern_counts)}</td></tr>
    <tr><td>Local blob modes</td><td>{html.escape(local_modes)}</td></tr>
    <tr><td>Scratch modes</td><td>{html.escape(scratch_modes)}</td></tr>
    <tr><td>Ring modes</td><td>{html.escape(ring_modes)}</td></tr>
    <tr><td>Shot-grid modes</td><td>{html.escape(shot_modes)}</td></tr>
    <tr><td>Stby origin split</td><td>{html.escape(stby_origin_summary)}</td></tr>
  </table>

  <h2>Per-Sample Metrics</h2>
  <table>
    <tr>
      <th>Sample</th><th>Grade 0 Ratio</th><th>Center Fail Density</th><th>Edge Fail Density</th><th>Edge-Chip Delta</th><th>Stby Chips Est.</th><th>Patterns</th>
    </tr>
    {sample_table}
  </table>

  <h2>Artifacts</h2>
  <ul>
    <li>Feature CSV: <code>{html.escape(relpath(features, out))}</code></li>
    <li>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>Gallery PNG: <code>{html.escape(relpath(gallery, out))}</code></li>
  </ul>

  <h2>Residual Risks</h2>
  <ul>
    <li>No confidential real wafer data was used, so final realism must still be judged by expert review.</li>
    <li>Stby is represented as a chip-level missing-test signal and can be coupled to defect origins, but real coupling frequency must be calibrated later with expert feedback.</li>
    <li>The current debug/review set cycles local blob modes to guarantee visual coverage of single, double, and triple-triangle cases. Pilot/training configs can switch this back to probabilistic sampling.</li>
  </ul>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data_root = Path(args.data)
    out = Path(args.out)
    gallery = Path(args.gallery)
    metrics_path = Path(args.metrics)
    features = Path(args.features)

    samples = [load_sample(path) for path in sample_dirs(data_root)]
    if not samples:
        raise SystemExit(f"No samples found under {data_root}")

    write_feature_csv(samples, features)
    create_gallery(samples, gallery)
    metrics = batch_metrics(samples)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_report(metrics, gallery, features, metrics_path, out), encoding="utf-8")
    print(f"Wrote report: {out}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote features: {features}")
    print(f"Wrote gallery: {gallery}")


if __name__ == "__main__":
    main()
