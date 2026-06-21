"""CPU-only shared encoder baseline for synthetic wafer maps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wafermap.data import PATTERN_CLASSES
from wafermap.training.embedding import EmbeddingDataset, retrieval_metrics


MODEL_VERSION = "cpu_shared_encoder/v1"


@dataclass(frozen=True)
class CPUEncoderModel:
    """Small MLP encoder with a multi-label classification head."""

    output_size: int
    input_dim: int
    hidden_dim: int
    embedding_dim: int
    label_names: tuple[str, ...]
    mean: NDArray[np.float32]
    scale: NDArray[np.float32]
    w1: NDArray[np.float32]
    b1: NDArray[np.float32]
    w2: NDArray[np.float32]
    b2: NDArray[np.float32]
    wc: NDArray[np.float32]
    bc: NDArray[np.float32]


def initialize_cpu_encoder(
    vectors: NDArray[np.float32],
    output_size: int,
    hidden_dim: int,
    embedding_dim: int,
    label_names: tuple[str, ...] = PATTERN_CLASSES,
    seed: int = 20260618,
) -> CPUEncoderModel:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("vectors must be a non-empty 2D array")
    if hidden_dim <= 0 or embedding_dim <= 0:
        raise ValueError("hidden_dim and embedding_dim must be positive")
    rng = np.random.default_rng(seed)
    mean = vectors.mean(axis=0).astype(np.float32)
    scale = vectors.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    input_dim = int(vectors.shape[1])
    output_dim = len(label_names)
    w1 = rng.normal(0.0, np.sqrt(2.0 / max(input_dim, 1)), size=(input_dim, hidden_dim)).astype(np.float32)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    w2 = rng.normal(0.0, np.sqrt(2.0 / max(hidden_dim, 1)), size=(hidden_dim, embedding_dim)).astype(np.float32)
    b2 = np.zeros(embedding_dim, dtype=np.float32)
    wc = rng.normal(0.0, np.sqrt(1.0 / max(embedding_dim, 1)), size=(embedding_dim, output_dim)).astype(np.float32)
    bc = np.zeros(output_dim, dtype=np.float32)
    return CPUEncoderModel(output_size, input_dim, hidden_dim, embedding_dim, label_names, mean, scale, w1, b1, w2, b2, wc, bc)


def predict_cpu_encoder(
    model: CPUEncoderModel,
    vectors: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    cache = _forward(model, vectors)
    embeddings = _normalize_rows(cache["z"])
    return embeddings, cache["logits"], _sigmoid(cache["logits"])


def train_cpu_encoder(
    model: CPUEncoderModel,
    train: EmbeddingDataset,
    val: EmbeddingDataset,
    *,
    epochs: int,
    learning_rate: float,
    pairwise_weight: float,
    l2: float,
    top_k: int,
) -> tuple[CPUEncoderModel, dict[str, Any]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if train.vectors.shape[1] != model.input_dim:
        raise ValueError("train vectors do not match model input_dim")

    current = model
    history = []
    train_targets = train.labels.astype(np.float32)
    pair_targets = _pairwise_label_similarity(train.labels)
    for epoch in range(epochs):
        current, losses = _train_step(
            current,
            train.vectors,
            train_targets,
            pair_targets,
            learning_rate=learning_rate,
            pairwise_weight=pairwise_weight,
            l2=l2,
        )
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 5) == 0:
            train_metrics = evaluate_cpu_encoder(current, train, train, top_k=top_k)
            val_metrics = evaluate_cpu_encoder(current, train, val, top_k=top_k)
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": losses,
                    "train_bce": train_metrics["bce"],
                    "val_bce": val_metrics["bce"],
                    "val_top1_mean_jaccard": val_metrics["retrieval"]["top1_mean_jaccard"],
                    "val_lift_vs_baseline": val_metrics["retrieval"]["lift_vs_baseline"],
                }
            )

    final_train = evaluate_cpu_encoder(current, train, train, top_k=top_k)
    final_val = evaluate_cpu_encoder(current, train, val, top_k=top_k)
    metrics = {
        "model_version": MODEL_VERSION,
        "output_size": current.output_size,
        "input_dim": current.input_dim,
        "hidden_dim": current.hidden_dim,
        "embedding_dim": current.embedding_dim,
        "target_channels": list(current.label_names),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "pairwise_weight": float(pairwise_weight),
        "l2": float(l2),
        "history": history,
        "train": final_train,
        "validation": final_val,
    }
    return current, metrics


def evaluate_cpu_encoder(
    model: CPUEncoderModel,
    reference: EmbeddingDataset,
    query: EmbeddingDataset,
    *,
    top_k: int,
) -> dict[str, Any]:
    ref_embeddings, _, _ = predict_cpu_encoder(model, reference.vectors)
    query_embeddings, query_logits, query_probs = predict_cpu_encoder(model, query.vectors)
    bce = _binary_cross_entropy(query_logits, query.labels.astype(np.float32))
    retrieval = retrieval_metrics(
        reference.sample_ids,
        ref_embeddings,
        reference.labels,
        query.sample_ids,
        query_embeddings,
        query.labels,
        top_k,
    )
    return {
        "bce": float(bce),
        "retrieval": retrieval,
        "per_class": _per_class_metrics(query.labels, query_probs, model.label_names),
    }


def save_cpu_encoder_model(
    path: str | Path,
    model: CPUEncoderModel,
    *,
    reference_sample_ids: list[str] | None = None,
    reference_embeddings: NDArray[np.float32] | None = None,
    reference_labels: NDArray[np.bool_] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_version": MODEL_VERSION,
        "output_size": model.output_size,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "embedding_dim": model.embedding_dim,
        "label_names": list(model.label_names),
    }
    payload: dict[str, Any] = {
        "metadata_json": np.array(json.dumps(metadata), dtype=np.str_),
        "mean": model.mean,
        "scale": model.scale,
        "w1": model.w1,
        "b1": model.b1,
        "w2": model.w2,
        "b2": model.b2,
        "wc": model.wc,
        "bc": model.bc,
    }
    if reference_sample_ids is not None and reference_embeddings is not None and reference_labels is not None:
        payload["reference_sample_ids"] = np.array(reference_sample_ids, dtype=np.str_)
        payload["reference_embeddings"] = reference_embeddings.astype(np.float32)
        payload["reference_labels"] = reference_labels.astype(np.uint8)
    np.savez_compressed(path, **payload)


def load_cpu_encoder_model(path: str | Path) -> tuple[CPUEncoderModel, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata_json"]))
        if metadata.get("model_version") != MODEL_VERSION:
            raise ValueError(f"Unsupported CPU encoder model_version: {metadata.get('model_version')}")
        model = CPUEncoderModel(
            output_size=int(metadata["output_size"]),
            input_dim=int(metadata["input_dim"]),
            hidden_dim=int(metadata["hidden_dim"]),
            embedding_dim=int(metadata["embedding_dim"]),
            label_names=tuple(metadata["label_names"]),
            mean=arrays["mean"].astype(np.float32),
            scale=arrays["scale"].astype(np.float32),
            w1=arrays["w1"].astype(np.float32),
            b1=arrays["b1"].astype(np.float32),
            w2=arrays["w2"].astype(np.float32),
            b2=arrays["b2"].astype(np.float32),
            wc=arrays["wc"].astype(np.float32),
            bc=arrays["bc"].astype(np.float32),
        )
        _validate_model_arrays(model)
        reference: dict[str, Any] = {}
        if "reference_sample_ids" in arrays.files:
            reference = {
                "sample_ids": [str(value) for value in arrays["reference_sample_ids"]],
                "embeddings": arrays["reference_embeddings"].astype(np.float32),
                "labels": arrays["reference_labels"].astype(bool),
            }
            _validate_reference_arrays(reference, model)
    return model, reference


def _validate_model_arrays(model: CPUEncoderModel) -> None:
    label_count = len(model.label_names)
    expected_shapes = {
        "mean": (model.input_dim,),
        "scale": (model.input_dim,),
        "w1": (model.input_dim, model.hidden_dim),
        "b1": (model.hidden_dim,),
        "w2": (model.hidden_dim, model.embedding_dim),
        "b2": (model.embedding_dim,),
        "wc": (model.embedding_dim, label_count),
        "bc": (label_count,),
    }
    actual = {
        "mean": model.mean.shape,
        "scale": model.scale.shape,
        "w1": model.w1.shape,
        "b1": model.b1.shape,
        "w2": model.w2.shape,
        "b2": model.b2.shape,
        "wc": model.wc.shape,
        "bc": model.bc.shape,
    }
    mismatches = [f"{name}: expected {expected_shapes[name]}, got {actual[name]}" for name in expected_shapes if actual[name] != expected_shapes[name]]
    if mismatches:
        raise ValueError(f"Invalid CPU encoder model array shapes: {'; '.join(mismatches)}")


def _validate_reference_arrays(reference: dict[str, Any], model: CPUEncoderModel) -> None:
    sample_ids = reference["sample_ids"]
    embeddings = reference["embeddings"]
    labels = reference["labels"]
    if embeddings.ndim != 2 or embeddings.shape[1] != model.embedding_dim:
        raise ValueError(
            f"Invalid reference embedding shape: expected Nx{model.embedding_dim}, got {embeddings.shape}"
        )
    if labels.ndim != 2 or labels.shape[1] != len(model.label_names):
        raise ValueError(
            f"Invalid reference label shape: expected Nx{len(model.label_names)}, got {labels.shape}"
        )
    if len(sample_ids) != embeddings.shape[0] or len(sample_ids) != labels.shape[0]:
        raise ValueError("Reference sample_ids, embeddings, and labels must have matching row counts")


def _train_step(
    model: CPUEncoderModel,
    vectors: NDArray[np.float32],
    targets: NDArray[np.float32],
    pair_targets: NDArray[np.float32],
    *,
    learning_rate: float,
    pairwise_weight: float,
    l2: float,
) -> tuple[CPUEncoderModel, dict[str, float]]:
    cache = _forward(model, vectors)
    probs = _sigmoid(cache["logits"])
    sample_count, label_count = targets.shape
    bce = _binary_cross_entropy(cache["logits"], targets)
    dlogits = (probs - targets) / float(max(sample_count * label_count, 1))

    z = cache["z"]
    pair_scores = (z @ z.T) / float(max(model.embedding_dim, 1))
    pair_diff = pair_scores - pair_targets
    pair_loss = float(np.mean(pair_diff * pair_diff))
    dz_pair = (4.0 / float(max(sample_count * sample_count * model.embedding_dim, 1))) * (pair_diff @ z)

    dwc = cache["z"].T @ dlogits + l2 * model.wc
    dbc = dlogits.sum(axis=0)
    dz = dlogits @ model.wc.T + pairwise_weight * dz_pair
    dz_pre = dz * (1.0 - cache["z"] * cache["z"])
    dw2 = cache["h"].T @ dz_pre + l2 * model.w2
    db2 = dz_pre.sum(axis=0)
    dh = dz_pre @ model.w2.T
    dh_pre = dh * (1.0 - cache["h"] * cache["h"])
    dw1 = cache["x"].T @ dh_pre + l2 * model.w1
    db1 = dh_pre.sum(axis=0)

    updated = CPUEncoderModel(
        model.output_size,
        model.input_dim,
        model.hidden_dim,
        model.embedding_dim,
        model.label_names,
        model.mean,
        model.scale,
        (model.w1 - learning_rate * dw1).astype(np.float32),
        (model.b1 - learning_rate * db1).astype(np.float32),
        (model.w2 - learning_rate * dw2).astype(np.float32),
        (model.b2 - learning_rate * db2).astype(np.float32),
        (model.wc - learning_rate * dwc).astype(np.float32),
        (model.bc - learning_rate * dbc).astype(np.float32),
    )
    l2_loss = 0.5 * l2 * float(
        np.sum(model.w1 * model.w1) + np.sum(model.w2 * model.w2) + np.sum(model.wc * model.wc)
    )
    return updated, {
        "total": float(bce + pairwise_weight * pair_loss + l2_loss),
        "bce": float(bce),
        "pairwise": pair_loss,
        "l2": l2_loss,
    }


def _forward(model: CPUEncoderModel, vectors: NDArray[np.float32]) -> dict[str, NDArray[np.float32]]:
    if vectors.ndim != 2 or vectors.shape[1] != model.input_dim:
        raise ValueError("vectors do not match model input_dim")
    x = ((vectors - model.mean) / model.scale).astype(np.float32)
    h = np.tanh(x @ model.w1 + model.b1).astype(np.float32)
    z = np.tanh(h @ model.w2 + model.b2).astype(np.float32)
    logits = (z @ model.wc + model.bc).astype(np.float32)
    return {"x": x, "h": h, "z": z, "logits": logits}


def _sigmoid(values: NDArray[np.float32]) -> NDArray[np.float32]:
    return (1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))).astype(np.float32)


def _binary_cross_entropy(logits: NDArray[np.float32], targets: NDArray[np.float32]) -> float:
    probs = _sigmoid(logits)
    eps = 1e-6
    loss = -(targets * np.log(probs + eps) + (1.0 - targets) * np.log(1.0 - probs + eps))
    return float(loss.mean())


def _normalize_rows(values: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, 1e-12)).astype(np.float32)


def _pairwise_label_similarity(labels: NDArray[np.bool_]) -> NDArray[np.float32]:
    n = labels.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    for row in range(n):
        left = labels[row]
        for col in range(n):
            right = labels[col]
            union = np.logical_or(left, right).sum()
            out[row, col] = 1.0 if int(union) == 0 else float(np.logical_and(left, right).sum() / union)
    return out


def _per_class_metrics(
    labels: NDArray[np.bool_],
    probs: NDArray[np.float32],
    label_names: tuple[str, ...],
) -> list[dict[str, float | int | str]]:
    predictions = probs >= 0.5
    rows: list[dict[str, float | int | str]] = []
    for idx, label_name in enumerate(label_names):
        target = labels[:, idx]
        pred = predictions[:, idx]
        tp = int(np.logical_and(target, pred).sum())
        fp = int(np.logical_and(~target, pred).sum())
        fn = int(np.logical_and(target, ~pred).sum())
        rows.append(
            {
                "class": label_name,
                "positive_samples": int(target.sum()),
                "predicted_positive_samples": int(pred.sum()),
                "precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)),
            }
        )
    return rows
