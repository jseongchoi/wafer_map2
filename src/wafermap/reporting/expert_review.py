"""Expert-review template helpers for nearest-neighbor outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

REQUIRED_NEIGHBOR_COLUMNS = ("query_sample_id", "rank", "neighbor_sample_id", "distance")
REVIEW_DECISIONS = ("same_family", "partial_match", "mismatch", "not_sure")
DOMINANT_DEFECTS = (
    "edge",
    "shot_grid",
    "stby_pattern",
    "stby_hidden_origin",
    "ring",
    "scratch",
    "local",
    "random",
    "mixed",
    "none",
    "unknown",
)
CLOCK_POSITION_MATCHES = ("yes", "partial", "no", "not_applicable")
MISSED_MAJOR_DEFECT_VALUES = ("yes", "no", "not_sure")
REVIEW_DEFECT_FAMILIES = (
    "edge",
    "shot_grid",
    "stby_pattern",
    "stby_hidden_origin",
    "ring",
    "scratch",
    "local",
    "random",
    "mixed",
    "none",
    "unknown",
)
RETRIEVAL_FAILURE_MODES = (
    "none",
    "wrong_family",
    "right_family_wrong_location",
    "missed_query_defect",
    "missed_scratch",
    "missed_shot_grid",
    "missed_ring",
    "missed_local",
    "missed_stby_origin",
    "scratch_orientation_span_mismatch",
    "shot_phase_layout_mismatch",
    "ring_radius_width_mismatch",
    "local_blob_topology_mismatch",
    "stby_hidden_origin_mismatch",
    "severity_scale_mismatch",
    "parser_or_mask_issue",
    "wrong_clock_position",
    "overweighted_stby",
    "overweighted_edge",
    "mixed_confounder",
    "insufficient_evidence",
    "other",
    "not_sure",
)
NEXT_ACTIONS = (
    "keep_baseline",
    "feature_weight_tuning",
    "tune_observable_feature",
    "add_location_aware_feature",
    "add_scratch_component_features",
    "add_shot_phase_features",
    "add_ring_radius_width_features",
    "add_local_topology_features",
    "add_stby_origin_coupling_features",
    "segmentation_candidate",
    "scratch_specific_track",
    "parser_validation",
    "review_more_samples",
    "deprioritize_pair",
    "not_sure",
)
TEMPLATE_COLUMNS = (
    "review_case_id",
    "query_sample_id",
    "rank",
    "neighbor_sample_id",
    "distance",
    "reviewer_decision",
    "query_defect_family",
    "neighbor_defect_family",
    "dominant_defect",
    "clock_position_match",
    "missed_major_defect",
    "retrieval_failure_mode",
    "next_action",
    "review_comment",
)


def validate_neighbor_rows(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        raise ValueError("Neighbor CSV has no rows")
    missing = [name for name in REQUIRED_NEIGHBOR_COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(f"Neighbor CSV missing required columns: {missing}")
    warnings: list[str] = []
    label_columns = [name for name in rows[0] if name.startswith("label_")]
    if label_columns:
        warnings.append(f"Ignored reference label columns to avoid reviewer bias: {', '.join(label_columns)}")
    return warnings


def format_distance(value: str) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def build_template_rows(neighbor_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings = validate_neighbor_rows(neighbor_rows)
    rows: list[dict[str, Any]] = []
    review_case_ids: set[str] = set()
    for item in neighbor_rows:
        query_id = str(item["query_sample_id"]).strip()
        rank = str(item["rank"]).strip()
        neighbor_id = str(item["neighbor_sample_id"]).strip()
        review_case_id = f"{query_id}__rank{rank}__{neighbor_id}"
        if review_case_id in review_case_ids:
            raise ValueError(f"Duplicate review_case_id generated from neighbor rows: {review_case_id}")
        review_case_ids.add(review_case_id)
        rows.append(
            {
                "review_case_id": review_case_id,
                "query_sample_id": query_id,
                "rank": rank,
                "neighbor_sample_id": neighbor_id,
                "distance": format_distance(str(item.get("distance", ""))),
                "reviewer_decision": "",
                "query_defect_family": "",
                "neighbor_defect_family": "",
                "dominant_defect": "",
                "clock_position_match": "",
                "missed_major_defect": "",
                "retrieval_failure_mode": "",
                "next_action": "",
                "review_comment": "",
            }
        )
    return rows, warnings


def write_template_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TEMPLATE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
