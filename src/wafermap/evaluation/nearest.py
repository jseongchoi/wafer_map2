"""Small nearest-neighbor utilities shared by validation scripts."""

from __future__ import annotations

import numpy as np

EPSILON = 1e-6


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float32)
    mu = values.mean(axis=0, keepdims=True)
    sigma = np.maximum(values.std(axis=0, keepdims=True), EPSILON)
    return mu.astype(np.float32), sigma.astype(np.float32)


def apply_standardizer(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - mu) / sigma).astype(np.float32)


def standardize(x: np.ndarray) -> np.ndarray:
    mu, sigma = fit_standardizer(x)
    return apply_standardizer(x, mu, sigma)


def euclidean_distance_matrix(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float32)
    right_values = left_values if right is None else np.asarray(right, dtype=np.float32)
    left_norms = np.sum(left_values * left_values, axis=1, keepdims=True)
    right_norms = np.sum(right_values * right_values, axis=1, keepdims=True).T
    distances_sq = np.maximum(left_norms + right_norms - 2.0 * (left_values @ right_values.T), 0.0)
    return np.sqrt(distances_sq).astype(np.float32)


def nearest_neighbor_indices(
    x: np.ndarray,
    top_k: int,
    *,
    standardize_input: bool = True,
    exclude_self: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    z = standardize(x) if standardize_input else np.asarray(x, dtype=np.float32)
    distances = euclidean_distance_matrix(z)
    if exclude_self:
        np.fill_diagonal(distances, np.inf)
    k = min(top_k, max(len(z) - 1 if exclude_self else len(z), 1))
    return np.argsort(distances, axis=1)[:, :k].astype(np.int32), distances


def cross_nearest_neighbor_indices(
    query_x: np.ndarray,
    reference_x: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    ref_mu, ref_sigma = fit_standardizer(reference_x)
    ref_z = apply_standardizer(reference_x, ref_mu, ref_sigma)
    query_z = apply_standardizer(query_x, ref_mu, ref_sigma)
    distances = euclidean_distance_matrix(query_z, ref_z)
    k = min(top_k, len(reference_x))
    return np.argsort(distances, axis=1)[:, :k].astype(np.int32), distances
