"""Feature-column selection helpers for observable FBM tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_EXCLUDED_COLUMNS = frozenset({"sample_id", "actual_net_die", "cluster_id", "pca_0", "pca_1"})


def _first_row(rows_or_row: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if isinstance(rows_or_row, Mapping):
        return rows_or_row
    if not rows_or_row:
        raise ValueError("Feature rows are empty")
    return rows_or_row[0]


def is_observable_feature_name(
    name: str,
    *,
    extra_excluded: Iterable[str] = (),
    include_location_aware: bool = False,
) -> bool:
    excluded = set(DEFAULT_EXCLUDED_COLUMNS)
    excluded.update(extra_excluded)
    if name in excluded:
        return False
    if name.startswith("label_") or name.endswith("_mask_ratio"):
        return False
    if not include_location_aware and (name.startswith("polar_") or name.startswith("stby_polar_")):
        return False
    return True


def observable_feature_names(
    rows_or_row: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    extra_excluded: Iterable[str] = (),
    include_location_aware: bool = False,
) -> list[str]:
    row = _first_row(rows_or_row)
    return [
        name
        for name in row
        if is_observable_feature_name(
            str(name),
            extra_excluded=extra_excluded,
            include_location_aware=include_location_aware,
        )
    ]


def compact_observable_feature_names(
    rows_or_row: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    extra_excluded: Iterable[str] = (),
) -> list[str]:
    return observable_feature_names(
        rows_or_row,
        extra_excluded=extra_excluded,
        include_location_aware=False,
    )


def shared_observable_feature_names(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    extra_excluded: Iterable[str] = (),
    include_location_aware: bool = False,
) -> list[str]:
    right_names = observable_feature_names(
        right_rows,
        extra_excluded=extra_excluded,
        include_location_aware=include_location_aware,
    )
    left_keys = set(left_rows[0])
    return [name for name in right_names if name in left_keys]


def feature_matrix(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    return np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
