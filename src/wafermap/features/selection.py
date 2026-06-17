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
    values: list[list[float]] = []
    for row_idx, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", f"row_{row_idx}"))
        row_values: list[float] = []
        for name in feature_names:
            if name not in row:
                raise ValueError(f"Missing feature column '{name}' at row {row_idx} sample_id={sample_id}")
            raw_value = row[name]
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric feature value for '{name}' at row {row_idx} sample_id={sample_id}: {raw_value!r}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite feature value for '{name}' at row {row_idx} sample_id={sample_id}: {raw_value!r}"
                )
            row_values.append(value)
        values.append(row_values)
    return np.array(values, dtype=np.float32)
