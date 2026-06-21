"""Score real-unlabeled wafer manifests with the CPU shared encoder model."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

import extract_real_unlabeled_features as real_loader  # noqa: E402
from wafermap.training.cpu_encoder import load_cpu_encoder_model, predict_cpu_encoder  # noqa: E402
from wafermap.training.segmentation import sample_to_input_tensor  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/models/fbm_cpu_encoder_model.npz")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions-out", default="outputs/reports/real_cpu_encoder_predictions.csv")
    parser.add_argument("--neighbors-out", default="outputs/reports/real_cpu_encoder_neighbors.csv")
    parser.add_argument("--sanity-out", default="outputs/reports/real_cpu_encoder_sanity.json")
    parser.add_argument("--report-out", default="outputs/reports/real_cpu_encoder_report.html")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--allow-output-outside-root",
        action="store_true",
        help="Allow derived outputs outside outputs/. Use only for isolated synthetic smoke tests.",
    )
    return parser.parse_args(argv)


def output_path(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def report_output_path(path: str | Path, *, allow_outside_root: bool = False) -> Path:
    resolved = output_path(path)
    if not allow_outside_root and not real_loader._is_inside(resolved, real_loader.OUTPUT_ROOT):
        raise ValueError(f"Output path must be under {real_loader.OUTPUT_ROOT}: {path}")
    return resolved


def relpath(target: Path, base_file: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.resolve().parent).replace("\\", "/")


def load_samples(manifest_path: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    manifest = real_loader.read_json(manifest_path)
    real_loader.validate_manifest(manifest)
    samples = []
    sanity = []
    for entry in manifest["samples"]:
        sample = real_loader.load_real_like_sample(entry, manifest_path)
        errors, warnings = real_loader.validate_real_like_sample(sample)
        samples.append(sample)
        sanity.append(
            {
                "sample_id": sample.sample_id,
                "errors": errors,
                "warnings": warnings,
                "shape": list(sample.shape),
                "valid_pixel_count": int((sample.valid_test_mask > 0).sum()),
                "stby_pixel_count": int((sample.stby_mask > 0).sum()),
            }
        )
    return samples, sanity


def write_predictions(
    path: Path,
    samples: list[Any],
    sanity: list[dict[str, Any]],
    label_names: tuple[str, ...],
    probs: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "status",
        "scoring_status",
        "errors",
        "warnings",
        *[f"prob_{label}" for label in label_names],
        "top_synthetic_label_hint",
        "top_synthetic_probability",
        "confidence_margin",
        "mean_binary_entropy",
        "review_priority",
        *[f"embedding_{idx:02d}" for idx in range(embeddings.shape[1])],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sample in enumerate(samples):
            scored = not sanity[idx]["errors"] and np.isfinite(probs[idx]).all()
            row: dict[str, object] = {
                "sample_id": sample.sample_id,
                "status": "PASS" if not sanity[idx]["errors"] else "FAIL",
                "scoring_status": "SCORED" if scored else "SKIPPED",
                "errors": "; ".join(sanity[idx]["errors"]),
                "warnings": "; ".join(sanity[idx]["warnings"]),
            }
            if scored:
                top_idx = int(np.argmax(probs[idx]))
                sorted_probs = np.sort(probs[idx])
                margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else float(sorted_probs[-1])
                entropy = mean_binary_entropy(probs[idx])
                row.update(
                    {
                        "top_synthetic_label_hint": label_names[top_idx],
                        "top_synthetic_probability": f"{float(probs[idx, top_idx]):.6f}",
                        "confidence_margin": f"{margin:.6f}",
                        "mean_binary_entropy": f"{entropy:.6f}",
                        "review_priority": review_priority(float(probs[idx, top_idx]), margin, entropy),
                    }
                )
                for label_idx, label in enumerate(label_names):
                    row[f"prob_{label}"] = f"{float(probs[idx, label_idx]):.6f}"
                for emb_idx in range(embeddings.shape[1]):
                    row[f"embedding_{emb_idx:02d}"] = f"{float(embeddings[idx, emb_idx]):.8f}"
            else:
                row.update(
                    {
                        "top_synthetic_label_hint": "",
                        "top_synthetic_probability": "",
                        "confidence_margin": "",
                        "mean_binary_entropy": "",
                        "review_priority": "",
                    }
                )
                for label in label_names:
                    row[f"prob_{label}"] = ""
                for emb_idx in range(embeddings.shape[1]):
                    row[f"embedding_{emb_idx:02d}"] = ""
            writer.writerow(row)


def mean_binary_entropy(probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities.astype(np.float64), 1e-6, 1.0 - 1e-6)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    return float(entropy.mean())


def review_priority(top_probability: float, margin: float, entropy: float) -> str:
    if entropy >= 0.60 or margin < 0.10:
        return "review_high_uncertain"
    if top_probability >= 0.75:
        return "review_high_strong_synthetic_match"
    return "review_medium"


def write_neighbors(
    path: Path,
    samples: list[Any],
    embeddings: np.ndarray,
    reference: dict[str, Any],
    label_names: tuple[str, ...],
    top_k: int,
) -> bool:
    if not reference or not samples:
        return False
    ref_embeddings = reference["embeddings"]
    similarities = embeddings @ ref_embeddings.T
    k = min(top_k, ref_embeddings.shape[0])
    fieldnames = [
        "query_sample_id",
        "rank",
        "neighbor_sample_id",
        "similarity",
        *[f"neighbor_has_{label}" for label in label_names],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for query_idx, sample in enumerate(samples):
            order = np.argsort(-similarities[query_idx])[:k]
            for rank, ref_idx in enumerate(order, start=1):
                row: dict[str, object] = {
                    "query_sample_id": sample.sample_id,
                    "rank": rank,
                    "neighbor_sample_id": reference["sample_ids"][int(ref_idx)],
                    "similarity": f"{float(similarities[query_idx, ref_idx]):.6f}",
                }
                for label_idx, label in enumerate(label_names):
                    row[f"neighbor_has_{label}"] = int(reference["labels"][int(ref_idx), label_idx])
                writer.writerow(row)
    return True


def valid_sample_indices(sanity: list[dict[str, Any]]) -> list[int]:
    return [idx for idx, row in enumerate(sanity) if not row["errors"]]


def html_report(
    sanity: list[dict[str, Any]],
    predictions_path: Path,
    neighbors_path: Path | None,
    sanity_path: Path,
    report_path: Path,
) -> str:
    passed = sum(1 for row in sanity if not row["errors"])
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['sample_id'])}</td>"
        f"<td>{'PASS' if not row['errors'] else 'FAIL'}</td>"
        f"<td>{html.escape('; '.join(row['errors']) or '-')}</td>"
        f"<td>{html.escape('; '.join(row['warnings']) or '-')}</td>"
        "</tr>"
        for row in sanity
    )
    neighbor_link = (
        f'<li>Nearest synthetic neighbors: <code>{html.escape(relpath(neighbors_path, report_path))}</code></li>'
        if neighbors_path is not None
        else "<li>Nearest synthetic neighbors: not available in the model file</li>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Real-Unlabeled CPU Encoder Scores</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Real-Unlabeled CPU Encoder Scores</h1>
  <div class="note">Predictions are synthetic-trained pre-real-data scores. Use them for triage and review prioritization, not as calibrated production defect calls.</div>
  <ul>
    <li>Samples: {len(sanity)}</li>
    <li>Sanity pass: {passed}</li>
    <li>Sanity fail: {len(sanity) - passed}</li>
  </ul>
  <h2>Sanity Snapshot</h2>
  <table>
    <tr><th>Sample</th><th>Status</th><th>Errors</th><th>Warnings</th></tr>
    {rows}
  </table>
  <h2>Outputs</h2>
  <ul>
    <li>Predictions: <code>{html.escape(relpath(predictions_path, report_path))}</code></li>
    {neighbor_link}
    <li>Sanity JSON: <code>{html.escape(relpath(sanity_path, report_path))}</code></li>
  </ul>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model, reference = load_cpu_encoder_model(output_path(args.model))
    manifest_path = Path(args.manifest).resolve()
    samples, sanity = load_samples(manifest_path)
    valid_indices = valid_sample_indices(sanity)
    probs = np.full((len(samples), len(model.label_names)), np.nan, dtype=np.float32)
    embeddings = np.full((len(samples), model.embedding_dim), np.nan, dtype=np.float32)
    valid_samples = [samples[idx] for idx in valid_indices]
    valid_embeddings = np.zeros((0, model.embedding_dim), dtype=np.float32)
    if valid_samples:
        vectors = np.stack(
            [sample_to_input_tensor(sample, model.output_size).reshape(-1) for sample in valid_samples],
            axis=0,
        ).astype(np.float32)
        valid_embeddings, _, valid_probs = predict_cpu_encoder(model, vectors)
        for out_idx, sample_idx in enumerate(valid_indices):
            probs[sample_idx] = valid_probs[out_idx]
            embeddings[sample_idx] = valid_embeddings[out_idx]

    predictions_path = report_output_path(args.predictions_out, allow_outside_root=args.allow_output_outside_root)
    neighbors_path = report_output_path(args.neighbors_out, allow_outside_root=args.allow_output_outside_root)
    sanity_path = report_output_path(args.sanity_out, allow_outside_root=args.allow_output_outside_root)
    report_path = report_output_path(args.report_out, allow_outside_root=args.allow_output_outside_root)
    write_predictions(predictions_path, samples, sanity, model.label_names, probs, embeddings)
    wrote_neighbors = write_neighbors(neighbors_path, valid_samples, valid_embeddings, reference, model.label_names, args.top_k)
    sanity_path.parent.mkdir(parents=True, exist_ok=True)
    sanity_path.write_text(json.dumps({"samples": sanity}, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        html_report(sanity, predictions_path, neighbors_path if wrote_neighbors else None, sanity_path, report_path),
        encoding="utf-8",
    )
    print(f"Wrote CPU encoder predictions: {predictions_path}")
    if wrote_neighbors:
        print(f"Wrote nearest synthetic neighbors: {neighbors_path}")
    print(f"Wrote sanity JSON: {sanity_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
