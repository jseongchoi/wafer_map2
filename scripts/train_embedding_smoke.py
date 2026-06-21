"""Run a synthetic wafer embedding smoke check."""

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

from wafermap.training.embedding import (
    EmbeddingDataset,
    fit_pca_model,
    load_embedding_dataset,
    retrieval_metrics,
    select_label_covered_rows,
    transform_embeddings,
)
from wafermap.training.segmentation import TARGET_CHANNELS, load_manifest_rows


class EmbeddingSmokeResult:
    def __init__(
        self,
        metrics: dict[str, Any],
        train: EmbeddingDataset,
        val: EmbeddingDataset,
        train_embeddings: np.ndarray,
        val_embeddings: np.ndarray,
    ) -> None:
        self.metrics = metrics
        self.train = train
        self.val = val
        self.train_embeddings = train_embeddings
        self.val_embeddings = val_embeddings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/reports/fbm_segmentation_manifest.csv")
    parser.add_argument("--out", default="outputs/reports/fbm_embedding_smoke_report.html")
    parser.add_argument("--metrics", default="outputs/reports/fbm_embedding_smoke_metrics.json")
    parser.add_argument("--embeddings-out", default="outputs/reports/fbm_embedding_smoke_embeddings.csv")
    parser.add_argument("--output-size", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-val-samples", type=int, default=32)
    return parser.parse_args(argv)


def relpath(target: Path, base_file: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.resolve().parent).replace("\\", "/")


def repo_path(target: Path) -> str:
    try:
        return target.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(target)


def write_embeddings_csv(
    path: Path,
    train_sample_ids: list[str],
    train_labels: np.ndarray,
    train_embeddings: np.ndarray,
    val_sample_ids: list[str],
    val_labels: np.ndarray,
    val_embeddings: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    embedding_dim = int(train_embeddings.shape[1])
    fieldnames = [
        "split",
        "sample_id",
        *[f"has_{label_name}" for label_name in TARGET_CHANNELS],
        *[f"embedding_{idx:02d}" for idx in range(embedding_dim)],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split, sample_ids, labels, embeddings in (
            ("train", train_sample_ids, train_labels, train_embeddings),
            ("val", val_sample_ids, val_labels, val_embeddings),
        ):
            for row_idx, sample_id in enumerate(sample_ids):
                row: dict[str, object] = {"split": split, "sample_id": sample_id}
                for label_idx, label_name in enumerate(TARGET_CHANNELS):
                    row[f"has_{label_name}"] = int(labels[row_idx, label_idx])
                for embedding_idx in range(embedding_dim):
                    row[f"embedding_{embedding_idx:02d}"] = f"{float(embeddings[row_idx, embedding_idx]):.8f}"
                writer.writerow(row)


def neighbor_table_rows(metrics: dict[str, Any], limit: int = 12) -> str:
    rows = []
    for row in metrics["retrieval"]["neighbors"][:limit]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['query_sample_id']))}</td>"
            f"<td>{html.escape(str(row['top1_sample_id']))}</td>"
            f"<td>{float(row['top1_similarity']):.4f}</td>"
            f"<td>{float(row['top1_jaccard']):.3f}</td>"
            f"<td>{float(row['topk_best_jaccard']):.3f}</td>"
            f"<td>{html.escape(', '.join(str(value) for value in row['topk_sample_ids']))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_report(metrics: dict[str, Any], metrics_path: Path, embeddings_path: Path, out: Path) -> str:
    verdict = "PASS" if metrics["train_samples"] > 0 and np.isfinite(metrics["retrieval"]["top1_mean_jaccard"]) else "CHECK"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FBM Embedding Smoke Check</title>
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
  <h1>FBM Embedding Smoke Check</h1>
  <div class="note">This is a representation-pipeline check, not a final deep-learning model. It verifies that synthetic wafer tensors can be projected into an embedding space and evaluated with label-aware retrieval.</div>
  <h2>판정 결과: {verdict}</h2>
  <ul>
    <li>Train samples: {metrics['train_samples']}, validation samples: {metrics['val_samples']}</li>
    <li>Tensor size: {metrics['output_size']} x {metrics['output_size']}</li>
    <li>Requested/effective embedding dim: {metrics['requested_embedding_dim']} / {metrics['effective_embedding_dim']}</li>
    <li>Top-1 mean label Jaccard: {metrics['retrieval']['top1_mean_jaccard']:.4f}</li>
    <li>Top-{metrics['retrieval']['top_k']} best mean label Jaccard: {metrics['retrieval']['topk_best_mean_jaccard']:.4f}</li>
    <li>Baseline mean label Jaccard: {metrics['retrieval']['baseline_mean_jaccard']:.4f}</li>
    <li>Top-1 lift vs baseline: {metrics['retrieval']['lift_vs_baseline']:.4f}x</li>
  </ul>

  <h2>Nearest Neighbor Snapshot</h2>
  <table>
    <tr><th>Query</th><th>Top-1 neighbor</th><th>Similarity</th><th>Top-1 Jaccard</th><th>Top-k Best Jaccard</th><th>Top-k IDs</th></tr>
    {neighbor_table_rows(metrics)}
  </table>

  <h2>Outputs</h2>
  <ul>
    <li>Metrics JSON: <code>{html.escape(relpath(metrics_path, out))}</code></li>
    <li>Embeddings CSV: <code>{html.escape(relpath(embeddings_path, out))}</code></li>
  </ul>
</body>
</html>
"""


def run(args: argparse.Namespace) -> EmbeddingSmokeResult:
    manifest = Path(args.manifest)
    train_rows = load_manifest_rows(manifest, split="train")
    val_rows = load_manifest_rows(manifest, split="val")
    if not train_rows:
        raise SystemExit(f"No train rows found in {manifest}")
    selected_train_rows = select_label_covered_rows(train_rows, args.max_train_samples)
    selected_val_rows = select_label_covered_rows(val_rows or train_rows, args.max_val_samples)
    train = load_embedding_dataset(selected_train_rows, ROOT, args.output_size, args.max_train_samples)
    val = load_embedding_dataset(selected_val_rows, ROOT, args.output_size, args.max_val_samples)
    model = fit_pca_model(train.vectors, args.embedding_dim)
    train_embeddings = transform_embeddings(model, train.vectors)
    val_embeddings = transform_embeddings(model, val.vectors)
    metrics = {
        "manifest": repo_path(manifest),
        "output_size": int(args.output_size),
        "requested_embedding_dim": int(args.embedding_dim),
        "effective_embedding_dim": int(train_embeddings.shape[1]),
        "train_samples": len(train.sample_ids),
        "val_samples": len(val.sample_ids),
        "target_channels": list(TARGET_CHANNELS),
        "projection": "standardized_pca",
        "retrieval": retrieval_metrics(
            train.sample_ids,
            train_embeddings,
            train.labels,
            val.sample_ids,
            val_embeddings,
            val.labels,
            args.top_k,
        ),
    }
    return EmbeddingSmokeResult(metrics, train, val, train_embeddings, val_embeddings)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)

    metrics_path = Path(args.metrics)
    embeddings_path = Path(args.embeddings_out)
    out_path = Path(args.out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_embeddings_csv(
        embeddings_path,
        result.train.sample_ids,
        result.train.labels,
        result.train_embeddings,
        result.val.sample_ids,
        result.val.labels,
        result.val_embeddings,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_report(result.metrics, metrics_path, embeddings_path, out_path), encoding="utf-8")
    print(f"Wrote embedding smoke report: {out_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote embeddings: {embeddings_path}")


if __name__ == "__main__":
    main()
