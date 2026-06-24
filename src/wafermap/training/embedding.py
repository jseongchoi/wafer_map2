"""Synthetic wafer embedding helpers for representation smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wafermap.training.segmentation import TARGET_CHANNELS
from wafermap.training.segmentation import load_segmentation_tensor


@dataclass(frozen=True)
class EmbeddingDataset:
    """Flat wafer tensors and multi-label targets used for embedding checks."""

    sample_ids: list[str]
    vectors: NDArray[np.float32]
    labels: NDArray[np.bool_]
    label_names: tuple[str, ...]


@dataclass(frozen=True)
class PCAModel:
    """Small deterministic projection model used before a neural encoder exists."""

    mean: NDArray[np.float32]
    scale: NDArray[np.float32]
    components: NDArray[np.float32]


def select_label_covered_rows(
    rows: list[dict[str, str]],
    max_samples: int,
    label_names: tuple[str, ...] = TARGET_CHANNELS,
) -> list[dict[str, str]]:
    """Pick a small row set while trying to cover every positive class once."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for label_name in label_names:
        if len(selected) >= max_samples:
            break
        key = f"has_{label_name}"
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in seen:
                continue
            if row.get(key) == "1":
                selected.append(row)
                seen.add(sample_id)
                break
    for row in rows:
        if len(selected) >= max_samples:
            break
        sample_id = row["sample_id"]
        if sample_id in seen:
            continue
        selected.append(row)
        seen.add(sample_id)
    return selected


def load_embedding_dataset(
    rows: list[dict[str, str]],
    repo_root: str | Path,
    output_size: int,
    max_samples: int,
    label_names: tuple[str, ...] = TARGET_CHANNELS,
) -> EmbeddingDataset:
    """Load synthetic manifest rows as flattened segmentation input tensors."""

    selected = rows[:max_samples]
    sample_ids: list[str] = []
    vectors = []
    labels = []
    for row in selected:
        sample_dir = _sample_dir_from_manifest_row(row, repo_root)
        inputs, _ = load_segmentation_tensor(sample_dir, output_size=output_size)
        sample_ids.append(row["sample_id"])
        vectors.append(inputs.reshape(-1))
        labels.append(_label_vector(row, label_names))
    if not vectors:
        raise ValueError("No samples available for embedding dataset")
    return EmbeddingDataset(
        sample_ids=sample_ids,
        vectors=np.stack(vectors, axis=0).astype(np.float32),
        labels=np.stack(labels, axis=0).astype(np.bool_),
        label_names=label_names,
    )


def fit_pca_model(vectors: NDArray[np.float32], embedding_dim: int) -> PCAModel:
    """Fit a deterministic PCA projection as a lightweight encoder baseline."""

    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("vectors must be a non-empty 2D array")
    mean = vectors.mean(axis=0).astype(np.float32)
    scale = vectors.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    standardized = ((vectors - mean) / scale).astype(np.float32)
    _, _, vt = np.linalg.svd(standardized, full_matrices=False)
    rank = min(int(embedding_dim), int(vt.shape[0]))
    components = vt[:rank].astype(np.float32)
    return PCAModel(mean=mean, scale=scale, components=components)


def transform_embeddings(model: PCAModel, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Project vectors and L2-normalize them for cosine-style retrieval."""

    standardized = ((vectors - model.mean) / model.scale).astype(np.float32)
    embeddings = standardized @ model.components.T
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / np.maximum(norms, 1e-12)).astype(np.float32)


def retrieval_metrics(
    train_sample_ids: list[str],
    train_embeddings: NDArray[np.float32],
    train_labels: NDArray[np.bool_],
    val_sample_ids: list[str],
    val_embeddings: NDArray[np.float32],
    val_labels: NDArray[np.bool_],
    top_k: int,
) -> dict[str, object]:
    """Evaluate whether nearby embeddings share synthetic defect labels."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if train_embeddings.shape[0] != train_labels.shape[0]:
        raise ValueError("train embeddings and labels have different row counts")
    if val_embeddings.shape[0] != val_labels.shape[0]:
        raise ValueError("validation embeddings and labels have different row counts")

    train_count = int(train_embeddings.shape[0])
    capped_top_k = min(int(top_k), train_count)
    similarities = val_embeddings @ train_embeddings.T
    neighbor_rows: list[dict[str, object]] = []
    top1_scores: list[float] = []
    topk_best_scores: list[float] = []
    topk_mean_scores: list[float] = []
    baseline_scores: list[float] = []

    for val_idx, sample_id in enumerate(val_sample_ids):
        scores = similarities[val_idx].copy()
        same_id_indices = [idx for idx, train_id in enumerate(train_sample_ids) if train_id == sample_id]
        if len(same_id_indices) < train_count:
            scores[same_id_indices] = -np.inf
        order = np.argsort(-scores)[:capped_top_k]
        jaccards = [_label_jaccard(val_labels[val_idx], train_labels[train_idx]) for train_idx in order]
        top1_scores.append(jaccards[0])
        topk_best_scores.append(float(max(jaccards)))
        topk_mean_scores.append(float(np.mean(jaccards)))

        baseline_candidates = [
            idx for idx, train_id in enumerate(train_sample_ids) if train_id != sample_id
        ] or list(range(train_count))
        baseline_idx = baseline_candidates[(val_idx * 997 + 17) % len(baseline_candidates)]
        baseline_scores.append(_label_jaccard(val_labels[val_idx], train_labels[baseline_idx]))

        neighbor_rows.append(
            {
                "query_sample_id": sample_id,
                "top1_sample_id": train_sample_ids[int(order[0])],
                "top1_similarity": float(scores[int(order[0])]),
                "top1_jaccard": float(jaccards[0]),
                "topk_best_jaccard": float(max(jaccards)),
                "topk_mean_jaccard": float(np.mean(jaccards)),
                "topk_sample_ids": [train_sample_ids[int(idx)] for idx in order],
            }
        )

    return {
        "top_k": capped_top_k,
        "top1_mean_jaccard": float(np.mean(top1_scores)) if top1_scores else 0.0,
        "topk_best_mean_jaccard": float(np.mean(topk_best_scores)) if topk_best_scores else 0.0,
        "topk_mean_jaccard": float(np.mean(topk_mean_scores)) if topk_mean_scores else 0.0,
        "baseline_mean_jaccard": float(np.mean(baseline_scores)) if baseline_scores else 0.0,
        "lift_vs_baseline": float(
            (np.mean(top1_scores) + 1e-9) / (np.mean(baseline_scores) + 1e-9)
        )
        if top1_scores
        else 0.0,
        "neighbors": neighbor_rows,
    }


def _sample_dir_from_manifest_row(row: dict[str, str], repo_root: str | Path) -> Path:
    arrays_path = Path(row["arrays_path"])
    if not arrays_path.is_absolute():
        arrays_path = Path(repo_root) / arrays_path
    return arrays_path.parent


def _label_vector(row: dict[str, str], label_names: tuple[str, ...]) -> NDArray[np.bool_]:
    return np.array([row.get(f"has_{label_name}") == "1" for label_name in label_names], dtype=np.bool_)


def _label_jaccard(left: NDArray[np.bool_], right: NDArray[np.bool_]) -> float:
    union = np.logical_or(left, right).sum()
    if int(union) == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)
