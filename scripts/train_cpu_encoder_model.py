"""Train a CPU-only shared encoder model on synthetic wafer tensors."""

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
sys.path.insert(0, str(ROOT / "src"))

from wafermap.training.cpu_encoder import (  # noqa: E402
    initialize_cpu_encoder,
    predict_cpu_encoder,
    save_cpu_encoder_model,
    train_cpu_encoder,
)
from wafermap.training.embedding import load_embedding_dataset, select_label_covered_rows  # noqa: E402
from wafermap.training.segmentation import TARGET_CHANNELS, load_manifest_rows  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/reports/fbm_segmentation_manifest.csv")
    parser.add_argument("--model-out", default="outputs/models/fbm_cpu_encoder_model.npz")
    parser.add_argument("--out", default="outputs/reports/fbm_cpu_encoder_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_cpu_encoder_metrics.json")
    parser.add_argument("--predictions-out", default="outputs/reports/fbm_cpu_encoder_val_predictions.csv")
    parser.add_argument("--output-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--pairwise-weight", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260618)
    return parser.parse_args(argv)


def relpath(target: Path, base_file: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.resolve().parent).replace("\\", "/")


def write_prediction_csv(
    path: Path,
    sample_ids: list[str],
    labels: np.ndarray,
    probs: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        *[f"has_{label}" for label in TARGET_CHANNELS],
        *[f"prob_{label}" for label in TARGET_CHANNELS],
        "top_predicted_label",
        "top_predicted_probability",
        *[f"embedding_{idx:02d}" for idx in range(embeddings.shape[1])],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, sample_id in enumerate(sample_ids):
            top_idx = int(np.argmax(probs[row_idx]))
            row: dict[str, object] = {
                "sample_id": sample_id,
                "top_predicted_label": TARGET_CHANNELS[top_idx],
                "top_predicted_probability": f"{float(probs[row_idx, top_idx]):.6f}",
            }
            for label_idx, label in enumerate(TARGET_CHANNELS):
                row[f"has_{label}"] = int(labels[row_idx, label_idx])
                row[f"prob_{label}"] = f"{float(probs[row_idx, label_idx]):.6f}"
            for emb_idx in range(embeddings.shape[1]):
                row[f"embedding_{emb_idx:02d}"] = f"{float(embeddings[row_idx, emb_idx]):.8f}"
            writer.writerow(row)


def class_metric_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for row in metrics["validation"]["per_class"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['class']))}</td>"
            f"<td>{row['positive_samples']}</td>"
            f"<td>{row['predicted_positive_samples']}</td>"
            f"<td>{float(row['precision']):.3f}</td>"
            f"<td>{float(row['recall']):.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def history_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for row in metrics["history"]:
        rows.append(
            "<tr>"
            f"<td>{row['epoch']}</td>"
            f"<td>{float(row['loss']['total']):.4f}</td>"
            f"<td>{float(row['loss']['bce']):.4f}</td>"
            f"<td>{float(row['loss']['pairwise']):.4f}</td>"
            f"<td>{float(row['val_bce']):.4f}</td>"
            f"<td>{float(row['val_top1_mean_jaccard']):.4f}</td>"
            f"<td>{float(row['val_lift_vs_baseline']):.4f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_report(metrics: dict[str, Any], model_path: Path, metrics_path: Path, predictions_path: Path, out: Path) -> str:
    gate = metrics["readiness_gate"]
    verdict = gate["status"]
    gate_issues = "".join(f"<li>{html.escape(issue)}</li>" for issue in gate["issues"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FBM CPU Shared Encoder</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #eef6ff; border-left: 4px solid #2563eb; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FBM CPU Shared Encoder</h1>
  <div class="note">This is the CPU baseline model. It uses a shared MLP encoder, multi-label BCE, and a pairwise label-similarity loss to keep the embedding useful for retrieval. It is intended as a complete pre-real-data pipeline, not a final production accuracy claim.</div>
  <h2>판정 결과: {verdict}</h2>
  <ul>
    {gate_issues or '<li>No gate issues detected.</li>'}
  </ul>
  <ul>
    <li>Model: hidden {metrics['hidden_dim']}, embedding {metrics['embedding_dim']}, tensor {metrics['output_size']} x {metrics['output_size']}</li>
    <li>Validation BCE: {metrics['validation']['bce']:.4f}</li>
    <li>Validation top-1 label Jaccard: {metrics['validation']['retrieval']['top1_mean_jaccard']:.4f}</li>
    <li>Validation top-k best label Jaccard: {metrics['validation']['retrieval']['topk_best_mean_jaccard']:.4f}</li>
    <li>Lift vs baseline: {metrics['validation']['retrieval']['lift_vs_baseline']:.4f}x</li>
  </ul>

  <h2>Training History</h2>
  <table>
    <tr><th>Epoch</th><th>Total Loss</th><th>BCE</th><th>Pairwise</th><th>Val BCE</th><th>Val Top-1 Jaccard</th><th>Val Lift</th></tr>
    {history_rows(metrics)}
  </table>

  <h2>Validation Class Snapshot</h2>
  <table>
    <tr><th>Class</th><th>Positive</th><th>Predicted Positive</th><th>Precision</th><th>Recall</th></tr>
    {class_metric_rows(metrics)}
  </table>

  <h2>Outputs</h2>
  <ul>
    <li>Model NPZ: <code>{html.escape(relpath(model_path, out))}</code></li>
    <li>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>Validation predictions CSV: <code>{html.escape(relpath(predictions_path, out))}</code></li>
  </ul>
</body>
</html>
"""


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Any, Any, Any]:
    train_rows = load_manifest_rows(args.manifest, split="train")
    val_rows = load_manifest_rows(args.manifest, split="val")
    if not train_rows:
        raise SystemExit(f"No train rows found in {args.manifest}")
    selected_train_rows = select_label_covered_rows(train_rows, args.max_train_samples)
    selected_val_rows = select_label_covered_rows(val_rows or train_rows, args.max_val_samples)
    train = load_embedding_dataset(selected_train_rows, ROOT, args.output_size, args.max_train_samples)
    val = load_embedding_dataset(selected_val_rows, ROOT, args.output_size, args.max_val_samples)
    model = initialize_cpu_encoder(
        train.vectors,
        output_size=args.output_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )
    model, metrics = train_cpu_encoder(
        model,
        train,
        val,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        pairwise_weight=args.pairwise_weight,
        l2=args.l2,
        top_k=args.top_k,
    )
    metrics["train_samples"] = len(train.sample_ids)
    metrics["val_samples"] = len(val.sample_ids)
    metrics["readiness_gate"] = readiness_gate(metrics)
    return metrics, model, train, val


def readiness_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    issues = []
    if metrics["val_samples"] < 10:
        issues.append(f"validation sample count is small: {metrics['val_samples']} < 10")
    for row in metrics["validation"]["per_class"]:
        if int(row["positive_samples"]) == 0:
            issues.append(f"class {row['class']} has no positive validation samples; metric is not evaluable")
    if not np.isfinite(metrics["validation"]["bce"]):
        issues.append("validation BCE is not finite")
    return {
        "status": "PASS" if not issues else "CHECK",
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metrics, model, train, val = run(args)
    train_embeddings, _, _ = predict_cpu_encoder(model, train.vectors)
    val_embeddings, _, val_probs = predict_cpu_encoder(model, val.vectors)

    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics)
    predictions_path = Path(args.predictions_out)
    out_path = Path(args.out)
    save_cpu_encoder_model(
        model_path,
        model,
        reference_sample_ids=train.sample_ids,
        reference_embeddings=train_embeddings,
        reference_labels=train.labels,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_prediction_csv(predictions_path, val.sample_ids, val.labels, val_probs, val_embeddings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_report(metrics, model_path, metrics_path, predictions_path, out_path), encoding="utf-8")
    print(f"Wrote CPU encoder model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote validation predictions: {predictions_path}")
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
