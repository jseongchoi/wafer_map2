"""Defect-family scoring from observable wafer feature rows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

DEFECT_FAMILIES: tuple[str, ...] = (
    "edge",
    "ring",
    "local",
    "scratch",
    "shot_grid",
    "stby_pattern",
    "random",
)

_POLAR_RE = re.compile(r"^(stby_)?polar_r(?P<radial>\d+)_a(?P<angular>\d+)_(?P<kind>severity|fail_density|ratio)$")


@dataclass(frozen=True)
class DefectScore:
    sample_id: str
    defect_family: str
    score: float
    confidence: str
    location: str
    evidence: str
    primary_feature: str
    interpretation: str

    def as_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "defect_family": self.defect_family,
            "score": round(self.score, 2),
            "confidence": self.confidence,
            "location": self.location,
            "evidence": self.evidence,
            "primary_feature": self.primary_feature,
            "interpretation": self.interpretation,
        }


def score_feature_row(row: dict[str, Any]) -> list[DefectScore]:
    """Return one score per defect family for a feature row.

    The first version is deliberately rule-based. It makes the current feature
    layer usable as an interpreter, while leaving room for later calibration
    from reviewed real-wafer labels or learned embeddings.
    """

    sample_id = str(row.get("sample_id", "unknown"))
    edge = _edge_score(row, sample_id)
    ring = _ring_score(row, sample_id)
    local = _local_score(row, sample_id)
    scratch = _scratch_score(row, sample_id)
    shot = _shot_score(row, sample_id)
    stby = _stby_score(row, sample_id)
    random = _random_score(row, sample_id, (edge, ring, local, scratch, shot))
    scores = [edge, ring, local, scratch, shot, stby, random]
    return sorted(scores, key=lambda item: item.score, reverse=True)


def score_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.extend(score.as_row() for score in score_feature_row(row))
    return out


def top_defects(
    rows: list[dict[str, Any]],
    *,
    min_score: float = 15.0,
    max_per_sample: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        grouped.setdefault(sample_id, [])
        if float(row["score"]) >= min_score and len(grouped[sample_id]) < max_per_sample:
            grouped[sample_id].append(row)
    return grouped


def _edge_score(row: dict[str, Any], sample_id: str) -> DefectScore:
    parts = (
        ("edge_density", 0.22, 0.28),
        ("edge_minus_center_density", 0.12, 0.22),
        ("edge_chip_outer_minus_inner_density", 0.12, 0.20),
        ("edge_chip_peak_contrast", 0.20, 0.15),
        ("edge_sector_peak_contrast", 0.16, 0.15),
    )
    score = _weighted_score(row, parts)
    location = _polar_location(row, radial_preference=(2,), prefix="", label="edge")
    evidence = _evidence(row, [name for name, _, _ in parts])
    primary = _primary_feature(row, [name for name, _, _ in parts])
    return _make_score(
        sample_id,
        "edge",
        score,
        location,
        evidence,
        primary,
        "wafer edge 쪽 fail density나 edge-facing chip contrast가 높은지 보는 점수입니다.",
    )


def _ring_score(row: dict[str, Any], sample_id: str) -> DefectScore:
    contrast = _value(row, "ring_radial_peak_contrast")
    width = _value(row, "ring_radial_peak_width_ratio")
    width_fit = _clamp01(1.0 - abs(width - 0.16) / 0.22)
    score = 100.0 * _clamp01(0.72 * _sat(contrast, 0.16) + 0.28 * width_fit * _sat(contrast, 0.07))
    radial_zone = _strongest_radial_zone(row)
    evidence = _evidence(row, ["ring_radial_peak_contrast", "ring_radial_peak_width_ratio"])
    primary = _primary_feature(row, ["ring_radial_peak_contrast", "ring_radial_peak_width_ratio"])
    return _make_score(
        sample_id,
        "ring",
        score,
        radial_zone,
        evidence,
        primary,
        "radial profile에서 특정 radius에 peak가 생기는지 보는 점수입니다.",
    )


def _local_score(row: dict[str, Any], sample_id: str) -> DefectScore:
    parts = (
        ("local_hotspot_peak_contrast", 0.24, 0.26),
        ("local_hotspot_top3_mean_contrast", 0.18, 0.22),
        ("local_hotspot_count_ratio", 0.06, 0.14),
        ("local_component_largest_ratio", 0.05, 0.17),
        ("local_component_compactness", 0.55, 0.13),
        ("morph_hot_chip_ratio", 0.08, 0.08),
    )
    score = _weighted_score(row, parts)
    location = _polar_location(row, radial_preference=(0, 1, 2), prefix="", label="hotspot")
    evidence = _evidence(row, [name for name, _, _ in parts])
    primary = _primary_feature(row, [name for name, _, _ in parts])
    return _make_score(
        sample_id,
        "local",
        score,
        location,
        evidence,
        primary,
        "작은 영역에 fail이 몰려 있는 hotspot 또는 connected component를 보는 점수입니다.",
    )


def _scratch_score(row: dict[str, Any], sample_id: str) -> DefectScore:
    parts = (
        ("scratch_angular_peak_contrast", 0.16, 0.22),
        ("scratch_component_linear_score", 0.30, 0.28),
        ("scratch_component_elongation", 0.55, 0.20),
        ("scratch_component_radial_span", 0.38, 0.16),
        ("scratch_component_angular_span", 0.28, 0.14),
    )
    score = _weighted_score(row, parts)
    location = _angular_location(row, label="line direction")
    evidence = _evidence(row, [name for name, _, _ in parts])
    primary = _primary_feature(row, [name for name, _, _ in parts])
    return _make_score(
        sample_id,
        "scratch",
        score,
        location,
        evidence,
        primary,
        "가늘고 길게 이어진 line 성분과 angular concentration을 보는 점수입니다.",
    )


def _shot_score(row: dict[str, Any], sample_id: str) -> DefectScore:
    parts = (
        ("shot_best_contrast", 0.13, 0.42),
        ("shot_lower_left_contrast", 0.12, 0.20),
        ("shot_bottom_edge_contrast", 0.12, 0.19),
        ("shot_left_edge_contrast", 0.12, 0.19),
    )
    score = _weighted_score(row, parts)
    primary = _primary_feature(row, [name for name, _, _ in parts])
    location = {
        "shot_lower_left_contrast": "repeated lower-left shot area",
        "shot_bottom_edge_contrast": "repeated bottom-edge shot area",
        "shot_left_edge_contrast": "repeated left-edge shot area",
    }.get(primary, "repeated shot-relative layout")
    evidence = _evidence(row, [name for name, _, _ in parts])
    return _make_score(
        sample_id,
        "shot_grid",
        score,
        location,
        evidence,
        primary,
        "shot layout 안의 반복 위치에서 fail contrast가 생기는지 보는 점수입니다.",
    )


def _stby_score(row: dict[str, Any], sample_id: str) -> DefectScore:
    stby_ratio = _value(row, "stby_ratio")
    polar_peak = _max_feature(row, prefix="stby_polar_", suffix="_ratio")[1]
    score = 100.0 * _clamp01(0.78 * _sat(stby_ratio, 0.08) + 0.22 * _sat(polar_peak, 0.20))
    location = _polar_location(row, radial_preference=(0, 1, 2), prefix="stby_", label="stby")
    evidence = _evidence(row, ["stby_ratio"]) + (f"; stby polar peak={polar_peak:.3f}" if polar_peak > 0 else "")
    primary = "stby_ratio"
    return _make_score(
        sample_id,
        "stby_pattern",
        score,
        location,
        evidence,
        primary,
        "stby 또는 미측정 chip이 wafer 안에서 얼마나 크고 집중되어 있는지 보는 점수입니다.",
    )


def _random_score(
    row: dict[str, Any],
    sample_id: str,
    structured_scores: tuple[DefectScore, ...],
) -> DefectScore:
    base = 100.0 * _clamp01(
        0.58 * _sat(_value(row, "total_fail_density"), 0.16)
        + 0.42 * _sat(_value(row, "grade_weighted_severity"), 0.10)
    )
    structured = max((item.score for item in structured_scores), default=0.0)
    penalty = 1.0 - 0.72 * _clamp01(structured / 100.0)
    score = base * penalty
    evidence = _evidence(row, ["total_fail_density", "grade_weighted_severity"])
    primary = _primary_feature(row, ["total_fail_density", "grade_weighted_severity"])
    return _make_score(
        sample_id,
        "random",
        score,
        "wafer-wide background",
        evidence,
        primary,
        "뚜렷한 구조 없이 전체 fail density가 높은지 보는 보조 점수입니다.",
    )


def _make_score(
    sample_id: str,
    family: str,
    score: float,
    location: str,
    evidence: str,
    primary_feature: str,
    interpretation: str,
) -> DefectScore:
    score = round(max(0.0, min(100.0, score)), 2)
    return DefectScore(
        sample_id=sample_id,
        defect_family=family,
        score=score,
        confidence=_confidence(score),
        location=location,
        evidence=evidence or "-",
        primary_feature=primary_feature or "-",
        interpretation=interpretation,
    )


def _weighted_score(row: dict[str, Any], parts: tuple[tuple[str, float, float], ...]) -> float:
    total_weight = sum(weight for _, _, weight in parts)
    if total_weight <= 0:
        return 0.0
    value = sum(_sat(_value(row, name), target) * weight for name, target, weight in parts) / total_weight
    return 100.0 * _clamp01(value)


def _evidence(row: dict[str, Any], names: list[str], limit: int = 4) -> str:
    ranked = sorted(((name, _value(row, name)) for name in names), key=lambda item: abs(item[1]), reverse=True)
    return "; ".join(f"{name}={value:.3f}" for name, value in ranked[:limit] if abs(value) > 1e-9)


def _primary_feature(row: dict[str, Any], names: list[str]) -> str:
    if not names:
        return "-"
    return max(names, key=lambda name: abs(_value(row, name)))


def _value(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    raw = row.get(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value != value or value in (float("inf"), float("-inf")):
        return default
    return value


def _sat(value: float, target: float) -> float:
    return _clamp01(max(value, 0.0) / max(target, 1e-9))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _confidence(score: float) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    if score >= 15:
        return "low"
    return "none"


def _max_feature(row: dict[str, Any], *, prefix: str, suffix: str) -> tuple[str, float]:
    best_name = ""
    best_value = 0.0
    for name in row:
        if str(name).startswith(prefix) and str(name).endswith(suffix):
            value = _value(row, str(name))
            if value > best_value:
                best_name = str(name)
                best_value = value
    return best_name, best_value


def _polar_location(
    row: dict[str, Any],
    *,
    radial_preference: tuple[int, ...],
    prefix: str,
    label: str,
) -> str:
    best_name = ""
    best_value = -1.0
    for name in row:
        text = str(name)
        match = _POLAR_RE.match(text)
        if not match:
            continue
        has_stby = bool(match.group(1))
        if prefix == "stby_" and not has_stby:
            continue
        if prefix != "stby_" and has_stby:
            continue
        radial = int(match.group("radial"))
        if radial_preference and radial not in radial_preference:
            continue
        value = _value(row, text)
        if value > best_value:
            best_name = text
            best_value = value
    if not best_name or best_value <= 0:
        return "not localized"
    match = _POLAR_RE.match(best_name)
    if not match:
        return "not localized"
    radial = int(match.group("radial"))
    if radial == 0:
        return f"{label} near center / radial zone 0"
    return f"{label} near {_sector_to_clock(int(match.group('angular')))} / radial zone {radial}"


def _angular_location(row: dict[str, Any], *, label: str) -> str:
    best_name, value = _max_feature(row, prefix="angular_sector_", suffix="_severity")
    if not best_name or value <= 0:
        return "not localized"
    match = re.search(r"angular_sector_(\d+)_severity", best_name)
    if not match:
        return "not localized"
    return f"{label} near {_sector_to_clock(int(match.group(1)))}"


def _strongest_radial_zone(row: dict[str, Any]) -> str:
    best_name, value = _max_feature(row, prefix="radial_zone_", suffix="_severity")
    if not best_name or value <= 0:
        return "no clear radial zone"
    match = re.search(r"radial_zone_(\d+)_severity", best_name)
    if not match:
        return "no clear radial zone"
    idx = int(match.group(1))
    label = "center" if idx <= 1 else "middle" if idx <= 3 else "edge-side"
    return f"{label} radial zone {idx}"


def _sector_to_clock(angular_idx: int, bins: int = 12) -> str:
    hour = int(round(((angular_idx + 0.5) / bins) * 12.0)) % 12
    return "12:00" if hour == 0 else f"{hour:02d}:00"
