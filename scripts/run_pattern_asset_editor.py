"""Run the legacy local web editor for FBM pattern asset fallback work."""

from __future__ import annotations

import argparse
import base64
import html
import importlib.util
import io
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.assets import (
    FAMILY_COLORS,
    FAMILY_LABELS,
    TARGET_FAMILIES,
    connected_components,
    load_prediction_masks,
    mask_to_rle,
    preview_rgb,
    rle_to_mask,
    save_pattern_assets,
    scan_pattern_assets,
)

NEAREST_RESAMPLE = getattr(Image, "Resampling", Image).NEAREST


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="real_unlabeled_manifest/v1 JSON path.")
    parser.add_argument("--sample-id", help="Sample ID to edit. Defaults to the first manifest sample.")
    parser.add_argument("--assets-root", default="data/pattern_assets", help="Output root for family asset folders.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--margin-ratio", type=float, default=0.20)
    parser.add_argument("--prediction-json", help="Optional fbm_prediction_masks/v1 JSON to prefill editable masks.")
    parser.add_argument("--proposal-json", help="Optional fbm_model_proposals/v1 JSON to preview/apply model proposals.")
    parser.add_argument(
        "--editor-max-size",
        type=int,
        default=768,
        help="Maximum editor canvas width/height. Use 0 to edit at source resolution.",
    )
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def load_manifest_sample(manifest_path: Path, sample_id: str | None) -> Any:
    module = _load_real_feature_module()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    module.validate_manifest(manifest)
    entries = list(manifest.get("samples", []))
    if not entries:
        raise ValueError(f"manifest has no samples: {manifest_path}")
    if sample_id is None:
        selected = entries[0]
    else:
        matches = [entry for entry in entries if str(entry.get("sample_id", "")) == sample_id]
        if not matches:
            raise ValueError(f"--sample-id not found in manifest: {sample_id}")
        selected = matches[0]
    return module.load_real_like_sample(selected, manifest_path)


def _load_real_feature_module() -> Any:
    path = ROOT / "scripts" / "extract_real_unlabeled_features.py"
    spec = importlib.util.spec_from_file_location("extract_real_unlabeled_features_for_pattern_editor", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load real feature extraction module: {path}")
    spec.loader.exec_module(module)
    return module


def png_bytes(array: np.ndarray, mode: str | None = None) -> bytes:
    with io.BytesIO() as buffer:
        image = Image.fromarray(array, mode=mode) if mode else Image.fromarray(array)
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def bytes_b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array.astype(np.uint8)).tobytes()).decode("ascii")


def editor_shape(source_shape: tuple[int, int], max_size: int) -> tuple[int, int]:
    if max_size <= 0:
        return source_shape
    height, width = source_shape
    longest = max(height, width)
    if longest <= max_size:
        return source_shape
    scale = max_size / longest
    return max(1, round(height * scale)), max(1, round(width * scale))


def resize_array_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(array.shape[:2]) == shape:
        return array
    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    resized = image.resize((shape[1], shape[0]), resample=NEAREST_RESAMPLE)
    return np.asarray(resized, dtype=np.uint8)


def resize_rgb_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(array.shape[:2]) == shape:
        return array
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    resized = image.resize((shape[1], shape[0]), resample=NEAREST_RESAMPLE)
    return np.asarray(resized, dtype=np.uint8)


def resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = resize_array_nearest(np.asarray(mask, dtype=np.uint8), shape)
    return resized > 0


def prediction_masks_for_editor(
    masks: dict[str, list[list[int]]],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> dict[str, list[list[int]]]:
    if source_shape == target_shape:
        return masks
    out: dict[str, list[list[int]]] = {}
    for family in TARGET_FAMILIES:
        source_mask = rle_to_mask(masks.get(family, []), source_shape)
        out[family] = mask_to_rle(resize_mask_nearest(source_mask, target_shape))
    return out


def load_model_proposals(
    path: Path | None,
    sample_id: str,
    source_shape: tuple[int, int],
    editor_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    entry = _proposal_entry_for_sample(payload, sample_id)
    if entry is None:
        return []

    proposals: list[dict[str, Any]] = []
    source = str(entry.get("source") or payload.get("source") or payload.get("model_name") or "model")
    base_parameters = {"source": source, "schema_version": str(payload.get("schema_version", ""))}
    for item in entry.get("proposals", []):
        proposal = _coerce_model_proposal(item, payload, entry, base_parameters, source_shape, editor_shape)
        if proposal is not None:
            proposals.append(proposal)
    for family, runs in dict(entry.get("masks") or {}).items():
        item = {
            "family": family,
            "rle": runs,
            "confidence": entry.get("confidence", payload.get("confidence", 0.5)),
            "description": "model mask proposal",
            "parameters": base_parameters,
        }
        proposal = _coerce_model_proposal(item, payload, entry, base_parameters, source_shape, editor_shape)
        if proposal is not None:
            proposals.append(proposal)
    return proposals


def _proposal_entry_for_sample(payload: dict[str, Any], sample_id: str) -> dict[str, Any] | None:
    entries = payload.get("samples")
    if entries is None:
        return payload if str(payload.get("sample_id", sample_id)) == sample_id else None
    for entry in entries:
        if str(entry.get("sample_id", "")) == sample_id:
            return entry
    return None


def _coerce_model_proposal(
    item: dict[str, Any],
    payload: dict[str, Any],
    entry: dict[str, Any],
    base_parameters: dict[str, Any],
    source_shape: tuple[int, int],
    editor_shape: tuple[int, int],
) -> dict[str, Any] | None:
    family = str(item.get("family", ""))
    if family not in TARGET_FAMILIES:
        return None
    runs = item.get("rle", item.get("mask_rle", []))
    shape = _proposal_shape(item, entry, payload, default=source_shape)
    mask = resize_mask_nearest(rle_to_mask(runs, shape), editor_shape)
    pixel_count = int(mask.sum())
    if pixel_count == 0:
        return None
    parameters = dict(base_parameters)
    parameters.update(dict(item.get("parameters") or {}))
    return _proposal_payload(
        family=family,
        mask=mask,
        confidence=_safe_float(item.get("confidence"), default=0.5),
        description=str(item.get("description") or "model proposal"),
        parameters=parameters,
    )


def _proposal_shape(
    item: dict[str, Any],
    entry: dict[str, Any],
    payload: dict[str, Any],
    *,
    default: tuple[int, int],
) -> tuple[int, int]:
    for value in (item.get("shape"), entry.get("shape"), payload.get("shape")):
        parsed = _parse_shape(value)
        if parsed is not None:
            return parsed
    for mapping in (item, entry, payload):
        if "height" in mapping and "width" in mapping:
            return int(mapping["height"]), int(mapping["width"])
    return default


def _parse_shape(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, dict) and "height" in value and "width" in value:
        return int(value["height"]), int(value["width"])
    return None


def polar_geometry(wafer_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wafer = np.asarray(wafer_mask, dtype=bool)
    ys, xs = np.indices(wafer.shape)
    if not wafer.any():
        return np.zeros(wafer.shape, dtype=np.float32), np.zeros(wafer.shape, dtype=np.float32)
    wy, wx = np.nonzero(wafer)
    cx = (float(wx.min()) + float(wx.max())) / 2.0
    cy = (float(wy.min()) + float(wy.max())) / 2.0
    distance = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_distance = float(distance[wafer].max()) if wafer.any() else 1.0
    radius = np.clip(distance / max(max_distance, 1.0), 0.0, 1.0)
    theta = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360.0) % 360.0
    return radius.astype(np.float32), theta.astype(np.float32)


def analyze_pattern_proposals(sample: Any, shape: tuple[int, int], min_grade: int) -> list[dict[str, Any]]:
    severity = resize_array_nearest(sample.severity, shape)
    wafer = resize_array_nearest(sample.wafer_mask, shape) > 0
    valid = resize_array_nearest(sample.valid_test_mask, shape) > 0
    high = wafer & valid & (severity >= min_grade)
    high_count = int(high.sum())
    if high_count == 0:
        return []

    radius, theta = polar_geometry(wafer)
    proposals: list[dict[str, Any]] = []
    ring_mask = _ring_proposal_mask(high, radius)
    edge_mask = high & (radius >= 0.88)
    if int(edge_mask.sum()) >= max(8, int(high_count * 0.05)):
        proposals.append(
            _proposal_payload(
                family="edge",
                mask=edge_mask,
                confidence=min(0.99, float(edge_mask.sum() / max(high_count, 1)) * 1.5),
                description="high-grade pixels near wafer boundary",
                parameters={
                    "min_grade": int(min_grade),
                    "radial_min": 0.88,
                    **_angle_parameters(theta[edge_mask]),
                },
            )
        )

    if ring_mask is not None:
        ring_pixels = int(ring_mask.sum())
        interior_pixels = int((high & (radius < 0.88)).sum())
        proposals.append(
            _proposal_payload(
                family="ring",
                mask=ring_mask,
                confidence=min(0.99, float(ring_pixels / max(interior_pixels, 1)) * 2.0),
                description="high-grade radial band away from edge",
                parameters={
                    "min_grade": int(min_grade),
                    "radial_mean": round(float(radius[ring_mask].mean()), 4),
                    "radial_width": round(float(radius[ring_mask].max() - radius[ring_mask].min()), 4),
                    **_angle_parameters(theta[ring_mask]),
                },
            )
        )

    local_source = high & (radius < 0.88)
    if ring_mask is not None:
        local_source &= ~ring_mask
    local_mask = _largest_local_component(local_source)
    if local_mask is not None:
        proposals.append(
            _proposal_payload(
                family="local",
                mask=local_mask,
                confidence=min(0.95, float(local_mask.sum() / max(high_count, 1)) * 2.5),
                description="largest compact high-grade component",
                parameters={
                    "min_grade": int(min_grade),
                    "radial_mean": round(float(radius[local_mask].mean()), 4),
                    **_bbox_parameters(local_mask),
                },
            )
        )

    return sorted(proposals, key=lambda item: (-float(item["confidence"]), str(item["family"])))


def _ring_proposal_mask(high: np.ndarray, radius: np.ndarray) -> np.ndarray | None:
    interior = high & (radius < 0.88)
    values = radius[interior]
    if values.size < 8:
        return None
    bins = 40
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 0.9))
    peak_idx = int(np.argmax(hist))
    peak_count = int(hist[peak_idx])
    if peak_count < max(8, int(values.size * 0.08)):
        return None
    center = float((edges[peak_idx] + edges[peak_idx + 1]) / 2.0)
    half_width = max(0.025, float(edges[1] - edges[0]) * 1.5)
    mask = interior & (np.abs(radius - center) <= half_width)
    return mask if int(mask.sum()) >= max(8, int(values.size * 0.08)) else None


def _largest_local_component(mask: np.ndarray) -> np.ndarray | None:
    components = connected_components(mask)
    if not components:
        return None
    largest = max(components, key=lambda component: int(component.sum()))
    return largest if int(largest.sum()) >= 8 else None


def _proposal_payload(
    *,
    family: str,
    mask: np.ndarray,
    confidence: float,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "family_label": FAMILY_LABELS[family],
        "pixel_count": int(mask.sum()),
        "confidence": round(float(confidence), 3),
        "description": description,
        "parameters": parameters,
        "rle": mask_to_rle(mask),
    }


def _angle_parameters(theta_values: np.ndarray) -> dict[str, Any]:
    if theta_values.size == 0:
        return {}
    sorted_theta = np.sort(np.asarray(theta_values, dtype=np.float32) % 360.0)
    if sorted_theta.size == 1:
        angle = round(float(sorted_theta[0]), 1)
        return {"theta_start_deg": angle, "theta_end_deg": angle, "theta_span_deg": 0.0}
    wrapped = np.concatenate([sorted_theta, sorted_theta[:1] + 360.0])
    gaps = np.diff(wrapped)
    gap_idx = int(np.argmax(gaps))
    span = 360.0 - float(gaps[gap_idx])
    start = float(sorted_theta[(gap_idx + 1) % sorted_theta.size] % 360.0)
    end = float(sorted_theta[gap_idx] % 360.0)
    return {
        "theta_start_deg": round(start, 1),
        "theta_end_deg": round(end, 1),
        "theta_span_deg": round(span, 1),
    }


def _bbox_parameters(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1]}


class PatternAssetEditorHandler(BaseHTTPRequestHandler):
    sample: Any
    manifest_path: Path
    assets_root: Path
    margin_ratio: float
    prediction_masks: dict[str, list[list[int]]]
    model_proposals: list[dict[str, Any]]
    editor_shape: tuple[int, int]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.send_bytes(EDITOR_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
        elif path == "/library":
            self.send_bytes(
                asset_library_html(scan_pattern_assets(self.assets_root), Path("/library")).encode("utf-8"),
                content_type="text/html; charset=utf-8",
            )
        elif path == "/api/sample":
            edit_height, edit_width = self.editor_shape
            payload = {
                "sample_id": self.sample.sample_id,
                "width": int(edit_width),
                "height": int(edit_height),
                "source_width": int(self.sample.shape[1]),
                "source_height": int(self.sample.shape[0]),
                "editor_downsampled": self.editor_shape != self.sample.shape,
                "families": list(TARGET_FAMILIES),
                "family_labels": FAMILY_LABELS,
                "family_colors": FAMILY_COLORS,
                "assets_root": str(self.assets_root),
                "margin_ratio": float(self.margin_ratio),
                "stby_target_excluded": True,
                "composition_rule": "max",
                "severity_b64": bytes_b64(resize_array_nearest(self.sample.severity, self.editor_shape)),
                "wafer_mask_b64": bytes_b64(resize_array_nearest(self.sample.wafer_mask, self.editor_shape)),
                "valid_mask_b64": bytes_b64(resize_array_nearest(self.sample.valid_test_mask, self.editor_shape)),
                "stby_mask_b64": bytes_b64(resize_array_nearest(self.sample.stby_mask, self.editor_shape)),
            }
            self.send_json(payload)
        elif path == "/api/assets":
            self.send_json({"assets": scan_pattern_assets(self.assets_root)})
        elif path == "/api/predictions":
            self.send_json(
                {
                    "sample_id": self.sample.sample_id,
                    "masks": prediction_masks_for_editor(self.prediction_masks, self.sample.shape, self.editor_shape),
                }
            )
        elif path == "/api/model-proposals":
            self.send_json(
                {
                    "schema_version": "fbm_editor_proposals/v1",
                    "sample_id": self.sample.sample_id,
                    "proposals": self.model_proposals,
                }
            )
        elif path == "/api/auto-proposals":
            params = parse_qs(parsed.query)
            min_grade = _safe_int(params.get("min_grade", ["3"])[0], default=3)
            min_grade = min(7, max(1, min_grade))
            self.send_json(
                {
                    "sample_id": self.sample.sample_id,
                    "min_grade": min_grade,
                    "proposals": analyze_pattern_proposals(self.sample, self.editor_shape, min_grade),
                }
            )
        elif path == "/base.png":
            self.send_bytes(
                png_bytes(resize_rgb_nearest(preview_rgb(self.sample), self.editor_shape), mode="RGB"),
                content_type="image/png",
            )
        elif path.startswith("/assets/"):
            self.send_asset_file(path.removeprefix("/assets/"))
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/save-assets":
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        masks_payload = payload.get("masks", {})
        save_mode = str(payload.get("save_mode", "family"))
        if save_mode not in {"family", "components"}:
            save_mode = "family"
        masks_by_family = {
            family: resize_mask_nearest(rle_to_mask(masks_payload.get(family, []), self.editor_shape), self.sample.shape)
            for family in TARGET_FAMILIES
        }
        saved = save_pattern_assets(
            sample=self.sample,
            masks_by_family=masks_by_family,
            assets_root=self.assets_root,
            margin_ratio=self.margin_ratio,
            source_manifest=self.manifest_path,
            split_components=save_mode == "components",
        )
        self.send_json({"saved_count": len(saved), "save_mode": save_mode, "saved": saved})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, payload: dict[str, Any]) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), content_type="application/json")

    def send_asset_file(self, relative_url_path: str) -> None:
        relative_path = Path(unquote(relative_url_path))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            self.send_error(404)
            return
        candidate = (self.assets_root / relative_path).resolve()
        if not _is_inside(candidate, self.assets_root.resolve()) or not candidate.is_file():
            self.send_error(404)
            return
        content_type = "image/png" if candidate.suffix.lower() == ".png" else "application/json"
        self.send_bytes(candidate.read_bytes(), content_type=content_type)

    def send_bytes(self, payload: bytes, *, content_type: str) -> None:
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def asset_library_html(assets: list[dict[str, Any]], _base_file: Path) -> str:
    family_counts = {family: 0 for family in TARGET_FAMILIES}
    for asset in assets:
        family_counts[str(asset["family"])] = family_counts.get(str(asset["family"]), 0) + 1
    count_cards = "\n".join(
        f"""<div class="metric"><strong>{family_counts.get(family, 0)}</strong><span>{html.escape(FAMILY_LABELS[family])}</span></div>"""
        for family in TARGET_FAMILIES
    )
    cards = "\n".join(_asset_card(asset, f"/assets/{asset['relative_path']}") for asset in assets)
    if not cards:
        cards = '<p class="muted">No saved pattern assets yet.</p>'
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FBM Pattern Asset Library</title>
  <style>
    body {{ margin: 0; background: #eef2f1; color: #17211f; font-family: "Segoe UI", "Noto Sans KR", Arial, sans-serif; line-height: 1.55; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px 16px 52px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 24px 0 12px; font-size: 20px; }}
    .muted {{ color: #66736f; }}
    .metrics {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin: 16px 0; }}
    .metric, .asset {{ border: 1px solid #d4ddda; border-radius: 8px; background: #fff; padding: 12px; }}
    .metric strong {{ display: block; font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .asset img {{ width: 100%; border: 1px solid #d4ddda; border-radius: 6px; background: #17211f; image-rendering: pixelated; }}
    .asset h3 {{ display: flex; justify-content: space-between; gap: 8px; margin: 0 0 8px; font-size: 14px; }}
    .asset dl {{ display: grid; grid-template-columns: 94px 1fr; gap: 4px 8px; margin: 10px 0 0; font-size: 13px; }}
    .asset dt {{ color: #66736f; }}
    .asset dd {{ margin: 0; overflow-wrap: anywhere; }}
    .swatch {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
    @media (max-width: 900px) {{ .metrics, .grid {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 620px) {{ .metrics, .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>FBM Pattern Asset Library</h1>
  <p class="muted">저장된 defect 누끼 asset을 검수하는 리포트입니다. 각 asset은 <code>grade.png</code>, <code>mask.png</code>, <code>preview.png</code>, <code>metadata.json</code>으로 구성됩니다.</p>
  <section class="metrics">{count_cards}</section>
  <h2>Saved Assets</h2>
  <section class="grid">{cards}</section>
</main>
</body>
</html>
"""


def _asset_card(asset: dict[str, Any], url_prefix: str) -> str:
    family = str(asset["family"])
    color = FAMILY_COLORS.get(family, "#8e8e93")
    return f"""<article class="asset">
  <h3><span><span class="swatch" style="background:{html.escape(color)}"></span>{html.escape(str(asset["family_label"]))}</span><span>{html.escape(str(asset["asset_id"]))}</span></h3>
  <img src="{html.escape(url_prefix)}/preview.png" alt="{html.escape(str(asset["asset_id"]))} preview">
  <dl>
    <dt>valid</dt><dd>{html.escape(str(asset["valid"]))}</dd>
    <dt>pixels</dt><dd>{html.escape(str(asset["mask_pixel_count"]))}</dd>
    <dt>grade</dt><dd>{html.escape(str(asset["grade_min"]))} - {html.escape(str(asset["grade_max"]))}</dd>
    <dt>bbox</dt><dd>{html.escape(json.dumps(asset["bbox_xywh"], ensure_ascii=False))}</dd>
    <dt>source</dt><dd>{html.escape(str(asset["source_sample_id"]))}</dd>
  </dl>
</article>"""


EDITOR_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FBM Pattern Asset Builder</title>
  <style>
    body { margin: 0; overflow: hidden; background: #eef2f1; color: #17211f; font-family: "Segoe UI", "Noto Sans KR", Arial, sans-serif; }
    .app { height: 100vh; min-height: 0; display: grid; grid-template-columns: minmax(300px, 1fr) minmax(280px, 320px); grid-template-rows: 52px minmax(0, 1fr); }
    header { grid-column: 1 / -1; min-width: 0; display: flex; align-items: center; gap: 14px; padding: 0 14px; border-bottom: 1px solid #d4ddda; background: #fff; }
    h1 { margin: 0; font-size: 18px; }
    .meta, .status { min-width: 0; overflow: hidden; color: #66736f; font-size: 13px; text-overflow: ellipsis; }
    main { min-width: 0; min-height: 0; display: grid; place-items: start center; padding: 12px; overflow: hidden; }
    .canvas-wrap { position: relative; width: min(100%, calc(100vh - 84px)); aspect-ratio: 1 / 1; background: #1c2421; border: 1px solid #d4ddda; border-radius: 8px; overflow: hidden; box-shadow: 0 16px 34px rgba(23, 33, 31, 0.12); touch-action: none; }
    .canvas-stage { position: absolute; inset: 0; transform-origin: 0 0; will-change: transform; }
    canvas { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; }
    .proposal-canvas { pointer-events: none; }
    .lasso-canvas { pointer-events: none; }
    aside { min-height: 0; padding: 12px; border-left: 1px solid #d4ddda; background: #fff; overflow: auto; }
    .panel { padding: 12px 0; border-bottom: 1px solid #d4ddda; }
    .panel:first-child { padding-top: 0; }
    .panel:last-child { border-bottom: 0; }
    details.panel > summary { display: flex; align-items: center; justify-content: space-between; gap: 10px; min-height: 34px; list-style: none; cursor: pointer; font-size: 14px; font-weight: 600; }
    details.panel > summary::-webkit-details-marker { display: none; }
    details.panel > summary::after { content: "+"; color: #66736f; font-weight: 700; }
    details.panel[open] > summary::after { content: "-"; }
    details.panel .panel-body { padding: 8px 0 2px; }
    h2 { margin: 0 0 10px; font-size: 14px; }
    button, select { border: 1px solid #d4ddda; border-radius: 8px; background: #f9fbfa; color: #17211f; min-height: 34px; padding: 7px 9px; font: inherit; }
    button { cursor: pointer; }
    button.active { border-color: #25745d; box-shadow: 0 0 0 2px rgba(37, 116, 93, 0.18); }
    button.primary { background: #25745d; color: white; border-color: #25745d; }
    button.danger { color: #9a3d38; }
    .families, .toolbar { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .families { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .families button { min-height: 32px; padding: 6px; font-size: 12px; }
    .row { display: grid; grid-template-columns: 112px 1fr; gap: 8px; align-items: center; margin: 10px 0; }
    .row span { color: #66736f; font-size: 13px; }
    .radio-grid { display: grid; gap: 8px; }
    .option { display: flex; align-items: flex-start; gap: 8px; border: 1px solid #d4ddda; border-radius: 8px; background: #fbfcfc; padding: 9px; font-size: 13px; cursor: pointer; }
    .option input { margin-top: 2px; }
    .option strong { display: block; font-size: 13px; }
    .option span { display: block; color: #66736f; }
    input[type="range"] { width: 100%; }
    select { width: 100%; }
    .swatch { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
    .asset-list { display: grid; gap: 10px; max-height: 360px; overflow: auto; padding-right: 2px; }
    .proposal-list { display: grid; gap: 8px; margin-top: 10px; }
    .proposal-card { border: 1px solid #d4ddda; border-radius: 8px; background: #fbfcfc; padding: 9px; font-size: 12px; }
    .proposal-card strong { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 5px; font-size: 13px; }
    .proposal-card p { margin: 4px 0 8px; color: #66736f; }
    .proposal-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    .proposal-card button { width: 100%; min-height: 32px; padding: 6px 8px; font-size: 12px; }
    .asset-card { border: 1px solid #d4ddda; border-radius: 8px; background: #fbfcfc; padding: 8px; }
    .asset-card img { width: 100%; border: 1px solid #d4ddda; border-radius: 6px; background: #17211f; image-rendering: pixelated; }
    .asset-title { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; margin-bottom: 6px; }
    .asset-meta { color: #66736f; font-size: 12px; margin-top: 6px; }
    pre { margin: 8px 0 0; max-height: 210px; overflow: auto; background: #f5f7f8; border: 1px solid #d4ddda; border-radius: 8px; padding: 10px; font-size: 12px; white-space: pre-wrap; }
    @media (max-width: 560px) { body { overflow: auto; } .app { height: auto; min-height: 100vh; grid-template-columns: 1fr; grid-template-rows: 52px auto auto; } main { overflow: visible; } aside { overflow: visible; border-left: 0; border-top: 1px solid #d4ddda; } .canvas-wrap { width: min(100%, 760px); } }
  </style>
</head>
<body>
<div class="app">
  <header><h1>FBM Pattern Asset Builder</h1><div class="meta" id="meta"></div></header>
  <main><div class="canvas-wrap" id="canvasWrap"><div class="canvas-stage" id="canvasStage"><canvas id="base"></canvas><canvas id="mask"></canvas><canvas id="proposalPreview" class="proposal-canvas"></canvas><canvas id="lasso" class="lasso-canvas"></canvas></div></div></main>
  <aside>
    <section class="panel"><h2>Family</h2><div class="families" id="families"></div></section>
    <section class="panel">
      <h2>Edit</h2>
      <div class="toolbar"><button id="paint" class="active" title="Paint into the selected defect family">Paint</button><button id="erase" title="Erase from the selected defect family">Erase</button></div>
      <div class="toolbar" style="margin-top:8px"><button id="pan" title="Move around the wafer map without editing">Pan</button><button id="fitView" title="Reset pan and zoom">Fit View</button></div>
      <div class="toolbar" style="margin-top:8px"><button id="zoomOut" title="Zoom out">Zoom -</button><button id="zoomIn" title="Zoom in">Zoom +</button></div>
      <div class="row"><span>Map Colors</span><select id="colorScheme"><option value="process">Process</option><option value="contrast">High Contrast</option><option value="thermal">Thermal</option><option value="mono">Mono Review</option></select></div>
      <div class="row"><span>Size</span><input id="brush" type="range" min="1" max="80" value="14"></div>
      <div class="row"><span>Opacity</span><input id="opacity" type="range" min="15" max="90" value="55"></div>
    </section>
    <section class="panel">
      <h2>Save</h2>
      <div class="toolbar"><button id="undo" title="Undo the last mask edit">Undo</button><button id="save" class="primary" title="Write selected masks as reusable pattern assets">Save Assets</button></div>
      <p class="status" id="status">Loading</p>
    </section>
    <section class="panel">
      <h2>Fit</h2>
      <div class="row"><span>Min Grade</span><input id="minGrade" type="range" min="1" max="7" value="3"></div>
      <div class="toolbar"><button id="growPaint" title="Expand from painted seed pixels at or above Min Grade">Grow Seed</button><button id="addGrade" title="Add all wafer pixels at or above Min Grade">Grade Area</button></div>
      <div class="toolbar" style="margin-top:8px"><button id="lassoFit" title="Draw a rough loop; matching high-grade pixels are added">Lasso Fit</button><button id="traceLine" title="Extend a scratch-like line from painted seed pixels">Trace Line</button></div>
      <div class="toolbar" style="margin-top:8px"><button id="clearAssist" class="danger" title="Clear the active family mask">Clear Active</button></div>
    </section>
    <details class="panel">
      <summary>Proposals</summary>
      <div class="panel-body">
      <div class="toolbar"><button id="analyzeAuto" title="Find ring, edge, and compact high-grade candidates">Analyze</button><button id="loadModelProposals" title="Load external model proposals from --proposal-json">Load Model</button></div>
      <div class="toolbar" style="margin-top:8px; grid-template-columns:1fr"><button id="applyAllProposals" title="Apply every loaded proposal">Apply All</button></div>
      <p class="status" id="proposalStatus">No proposals yet</p>
      <div class="proposal-list" id="proposalList"></div>
      </div>
    </details>
    <details class="panel">
      <summary>Geometry Fit</summary>
      <div class="panel-body">
      <div class="toolbar"><button id="edgeFit">Edge Fit</button><button id="ringFit">Ring Fit</button></div>
      <div class="row"><span>Radius</span><input id="fitRadius" type="range" min="5" max="95" value="55"></div>
      <div class="row"><span>Width</span><input id="fitWidth" type="range" min="1" max="20" value="5"></div>
      <div class="row"><span>Angle Start</span><input id="fitThetaStart" type="range" min="0" max="359" value="0"></div>
      <div class="row"><span>Angle End</span><input id="fitThetaEnd" type="range" min="0" max="359" value="359"></div>
      <div class="toolbar"><button id="previewGeometry">Preview Fit</button><button id="applyGeometry">Apply Fit</button></div>
      <p class="status" id="geometryStatus">Edge Fit uses width from wafer boundary. Ring Fit uses radius and width.</p>
      </div>
    </details>
    <details class="panel">
      <summary>Save Mode</summary>
      <div class="panel-body">
      <div class="radio-grid">
        <label class="option"><input type="radio" name="saveMode" value="family" checked><span><strong>One Family Asset</strong><span>ring처럼 끊긴 패턴도 하나로 저장</span></span></label>
        <label class="option"><input type="radio" name="saveMode" value="components"><span><strong>Split Components</strong><span>독립 blob 여러 개를 따로 저장</span></span></label>
      </div>
      </div>
    </details>
    <details class="panel">
      <summary>Utilities</summary>
      <div class="panel-body">
      <div class="toolbar"><button id="clearFamily" class="danger" title="Clear the active family mask">Clear Active</button><button id="loadPrediction" title="Load prediction masks into the editor">Load Prediction</button></div>
      <div class="toolbar" style="margin-top:8px; grid-template-columns:1fr"><button id="clearAll" class="danger" title="Clear every family mask">Clear All</button></div>
      </div>
    </details>
    <details class="panel">
      <summary>Output</summary>
      <div class="panel-body">
      <pre id="preview"></pre>
      </div>
    </details>
    <details class="panel">
      <summary>Saved Assets</summary>
      <div class="panel-body">
      <div class="toolbar"><button id="refreshAssets">Refresh</button><button id="openReport">Open Report</button></div>
      <p class="status" id="assetStatus">No assets loaded</p>
      <div class="asset-list" id="assetList"></div>
      </div>
    </details>
  </aside>
</div>
<script>
let sample = null;
let W = 0;
let H = 0;
let N = 0;
let activeFamily = "";
let mode = "paint";
let drawing = false;
let history = [];
let masks = {};
let severity = null;
let waferMask = null;
let validMask = null;
let stbyMask = null;
let radiusNorm = null;
let thetaDeg = null;
let lassoPoints = [];
let autoProposals = [];
let proposalPreviewIndex = -1;
let geometryFamily = "edge";
let view = { zoom: 1, panX: 0, panY: 0 };
let panStart = null;
const canvasWrap = document.getElementById("canvasWrap");
const canvasStage = document.getElementById("canvasStage");
const base = document.getElementById("base");
const mask = document.getElementById("mask");
const proposalPreview = document.getElementById("proposalPreview");
const lasso = document.getElementById("lasso");
const baseCtx = base.getContext("2d");
const maskCtx = mask.getContext("2d");
const proposalCtx = proposalPreview.getContext("2d");
const lassoCtx = lasso.getContext("2d");
const COLOR_SCHEMES = {
  process: {
    outside: [28, 36, 33],
    stby: [166, 216, 240],
    grades: [[247, 248, 248], [88, 166, 255], [35, 203, 167], [118, 219, 87], [234, 221, 72], [245, 151, 54], [220, 72, 66], [122, 24, 28]]
  },
  contrast: {
    outside: [16, 18, 20],
    stby: [122, 206, 244],
    grades: [[236, 238, 241], [75, 114, 255], [35, 210, 186], [109, 226, 93], [255, 230, 73], [255, 142, 47], [235, 52, 68], [95, 18, 32]]
  },
  thermal: {
    outside: [22, 24, 30],
    stby: [142, 199, 224],
    grades: [[245, 246, 242], [44, 123, 182], [23, 180, 177], [113, 206, 77], [253, 217, 65], [252, 141, 42], [215, 48, 39], [112, 0, 38]]
  },
  mono: {
    outside: [30, 34, 36],
    stby: [157, 206, 222],
    grades: [[247, 247, 247], [220, 230, 238], [188, 205, 218], [156, 178, 194], [124, 151, 169], [92, 124, 145], [61, 96, 121], [32, 67, 92]]
  }
};

init();

async function init() {
  sample = await (await fetch("/api/sample")).json();
  W = sample.width; H = sample.height; N = W * H;
  base.width = mask.width = proposalPreview.width = lasso.width = W;
  base.height = mask.height = proposalPreview.height = lasso.height = H;
  activeFamily = sample.families[0];
  masks = Object.fromEntries(sample.families.map(name => [name, new Uint8Array(N)]));
  severity = decodeBytes(sample.severity_b64, N);
  waferMask = decodeBytes(sample.wafer_mask_b64, N);
  validMask = decodeBytes(sample.valid_mask_b64, N);
  stbyMask = sample.stby_mask_b64 ? decodeBytes(sample.stby_mask_b64, N) : new Uint8Array(N);
  const polar = computePolarGeometry();
  radiusNorm = polar.radius;
  thetaDeg = polar.theta;
  const sourceSize = sample.editor_downsampled ? `source ${sample.source_width} x ${sample.source_height}, ` : "";
  document.getElementById("meta").textContent = `${sample.sample_id} - ${sourceSize}edit ${W} x ${H} - assets: ${sample.assets_root}`;
  buildFamilies();
  bindEvents();
  setGeometryFamily("edge");
  setMode("paint");
  fitView();
  renderBase();
  render();
  updatePreview();
  loadAssets();
}

function buildFamilies() {
  const root = document.getElementById("families");
  root.innerHTML = "";
  for (const name of sample.families) {
    const button = document.createElement("button");
    button.dataset.family = name;
    button.innerHTML = `<span class="swatch" style="background:${sample.family_colors[name]}"></span>${sample.family_labels[name]}`;
    button.onclick = () => {
      setActiveFamily(name);
    };
    root.appendChild(button);
  }
  root.querySelector("button").classList.add("active");
}

function setActiveFamily(name) {
  activeFamily = name;
  document.querySelectorAll("[data-family]").forEach(btn => btn.classList.toggle("active", btn.dataset.family === name));
  clearProposalPreview();
  clearLasso();
  updatePreview();
}

function computePolarGeometry() {
  const radius = new Float32Array(N);
  const theta = new Float32Array(N);
  let minX = W, maxX = -1, minY = H, maxY = -1;
  for (let y = 0; y < H; y += 1) {
    for (let x = 0; x < W; x += 1) {
      const idx = y * W + x;
      if (!waferMask[idx]) continue;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  if (maxX < minX || maxY < minY) return { radius, theta };
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  let maxDistance = 1;
  for (let i = 0; i < N; i += 1) {
    if (!waferMask[i]) continue;
    const x = i % W;
    const y = Math.floor(i / W);
    const distance = Math.hypot(x - cx, y - cy);
    if (distance > maxDistance) maxDistance = distance;
  }
  for (let i = 0; i < N; i += 1) {
    const x = i % W;
    const y = Math.floor(i / W);
    radius[i] = Math.min(1, Math.hypot(x - cx, y - cy) / maxDistance);
    theta[i] = (Math.atan2(y - cy, x - cx) * 180 / Math.PI + 360) % 360;
  }
  return { radius, theta };
}

function bindEvents() {
  document.getElementById("paint").onclick = () => setMode("paint");
  document.getElementById("erase").onclick = () => setMode("erase");
  document.getElementById("pan").onclick = () => setMode("pan");
  document.getElementById("fitView").onclick = fitView;
  document.getElementById("zoomOut").onclick = () => zoomAtCenter(1 / 1.25);
  document.getElementById("zoomIn").onclick = () => zoomAtCenter(1.25);
  document.getElementById("undo").onclick = undo;
  document.getElementById("clearFamily").onclick = clearFamily;
  document.getElementById("clearAll").onclick = clearAll;
  document.getElementById("save").onclick = saveAssets;
  document.getElementById("loadPrediction").onclick = loadPrediction;
  document.getElementById("growPaint").onclick = growFromPaint;
  document.getElementById("addGrade").onclick = addGradeArea;
  document.getElementById("lassoFit").onclick = () => setMode("lasso");
  document.getElementById("traceLine").onclick = traceScratchLine;
  document.getElementById("clearAssist").onclick = clearFamily;
  document.getElementById("analyzeAuto").onclick = analyzeAuto;
  document.getElementById("loadModelProposals").onclick = loadModelProposals;
  document.getElementById("applyAllProposals").onclick = applyAllProposals;
  document.getElementById("edgeFit").onclick = () => setGeometryFamily("edge");
  document.getElementById("ringFit").onclick = () => setGeometryFamily("ring");
  document.getElementById("previewGeometry").onclick = previewGeometryFit;
  document.getElementById("applyGeometry").onclick = applyGeometryFit;
  document.getElementById("refreshAssets").onclick = loadAssets;
  document.getElementById("openReport").onclick = () => window.open("/library", "_blank");
  document.getElementById("opacity").oninput = render;
  document.getElementById("colorScheme").onchange = renderBase;
  document.getElementById("minGrade").oninput = () => { clearProposalPreview(); updatePreview(); updateGeometryStatus(); };
  for (const id of ["fitRadius", "fitWidth", "fitThetaStart", "fitThetaEnd"]) {
    document.getElementById(id).oninput = () => { clearProposalPreview(); updateGeometryStatus(); };
  }
  document.querySelectorAll("input[name='saveMode']").forEach(input => { input.onchange = updatePreview; });
  mask.onpointerdown = startPointer;
  mask.onpointermove = movePointer;
  mask.onpointerup = endPointer;
  mask.onpointercancel = cancelPointer;
  canvasWrap.onwheel = zoomWheel;
  window.onresize = () => applyView();
}

function setMode(next) {
  mode = next;
  document.getElementById("paint").classList.toggle("active", mode === "paint");
  document.getElementById("erase").classList.toggle("active", mode === "erase");
  document.getElementById("lassoFit").classList.toggle("active", mode === "lasso");
  document.getElementById("pan").classList.toggle("active", mode === "pan");
  canvasWrap.style.cursor = mode === "pan" ? "grab" : "crosshair";
  clearLasso();
}

function fitView() {
  view = { zoom: 1, panX: 0, panY: 0 };
  applyView();
}

function zoomWheel(event) {
  event.preventDefault();
  const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
  zoomAt(event.clientX, event.clientY, factor);
}

function zoomAtCenter(factor) {
  const rect = canvasWrap.getBoundingClientRect();
  zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, factor);
}

function zoomAt(clientX, clientY, factor) {
  const rect = canvasWrap.getBoundingClientRect();
  const anchorX = clientX - rect.left;
  const anchorY = clientY - rect.top;
  const nextZoom = Math.max(1, Math.min(12, view.zoom * factor));
  const stageX = (anchorX - view.panX) / view.zoom;
  const stageY = (anchorY - view.panY) / view.zoom;
  view.zoom = nextZoom;
  view.panX = anchorX - stageX * nextZoom;
  view.panY = anchorY - stageY * nextZoom;
  clampView();
  applyView();
}

function clampView() {
  const rect = canvasWrap.getBoundingClientRect();
  const minX = rect.width * (1 - view.zoom);
  const minY = rect.height * (1 - view.zoom);
  view.panX = Math.min(0, Math.max(minX, view.panX));
  view.panY = Math.min(0, Math.max(minY, view.panY));
  if (view.zoom === 1) {
    view.panX = 0;
    view.panY = 0;
  }
}

function applyView() {
  clampView();
  canvasStage.style.transform = `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`;
}

function point(event) {
  const rect = mask.getBoundingClientRect();
  return { x: Math.floor((event.clientX - rect.left) * W / rect.width), y: Math.floor((event.clientY - rect.top) * H / rect.height) };
}

function startPointer(event) {
  drawing = true;
  mask.setPointerCapture(event.pointerId);
  if (mode === "pan") {
    panStart = { x: event.clientX, y: event.clientY, panX: view.panX, panY: view.panY };
    canvasWrap.style.cursor = "grabbing";
    return;
  }
  if (mode === "lasso") {
    startLasso(event);
    return;
  }
  saveHistory();
  paintAt(event);
}

function movePointer(event) {
  if (!drawing) return;
  if (mode === "pan") {
    if (!panStart) return;
    view.panX = panStart.panX + event.clientX - panStart.x;
    view.panY = panStart.panY + event.clientY - panStart.y;
    applyView();
    return;
  }
  if (mode === "lasso") {
    updateLasso(event);
    return;
  }
  paintAt(event);
}

function endPointer(event) {
  if (!drawing) return;
  drawing = false;
  if (mode === "pan") {
    panStart = null;
    canvasWrap.style.cursor = "grab";
    return;
  }
  if (mode === "lasso") {
    finishLasso(event);
    return;
  }
  updatePreview();
}

function cancelPointer() {
  drawing = false;
  panStart = null;
  canvasWrap.style.cursor = mode === "pan" ? "grab" : "crosshair";
  clearLasso();
}

function paintAt(event) {
  const p = point(event);
  const radius = Number(document.getElementById("brush").value);
  const target = masks[activeFamily];
  const value = mode === "paint" ? 1 : 0;
  const r2 = radius * radius;
  const minX = Math.max(0, p.x - radius);
  const maxX = Math.min(W - 1, p.x + radius);
  const minY = Math.max(0, p.y - radius);
  const maxY = Math.min(H - 1, p.y + radius);
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const dx = x - p.x, dy = y - p.y;
      if (dx * dx + dy * dy <= r2) target[y * W + x] = value;
    }
  }
  renderDirty(minX, minY, maxX, maxY);
}

function clampPoint(p) {
  return { x: Math.max(0, Math.min(W - 1, p.x)), y: Math.max(0, Math.min(H - 1, p.y)) };
}

function startLasso(event) {
  lassoPoints = [clampPoint(point(event))];
  drawLasso();
  document.getElementById("status").textContent = "Drawing lasso";
}

function updateLasso(event) {
  const p = clampPoint(point(event));
  const last = lassoPoints[lassoPoints.length - 1];
  if (last) {
    const dx = p.x - last.x;
    const dy = p.y - last.y;
    if (dx * dx + dy * dy < 4) return;
  }
  lassoPoints.push(p);
  drawLasso();
}

function finishLasso(event) {
  updateLasso(event);
  if (lassoPoints.length < 3) {
    clearLasso();
    document.getElementById("status").textContent = "Lasso needs at least three points";
    return;
  }
  const result = applyLassoFit();
  clearLasso();
  render();
  updatePreview();
  document.getElementById("preview").textContent = JSON.stringify({
    lassoFit: result,
    activeFamily,
    minGrade: Number(document.getElementById("minGrade").value),
    rule: "inside lasso, includes severity >= minGrade valid pixels and STBY fail chips"
  }, null, 2);
}

function drawLasso() {
  lassoCtx.clearRect(0, 0, W, H);
  if (lassoPoints.length < 2) return;
  const lineWidth = Math.max(1, Math.round(Math.max(W, H) / 384));
  lassoCtx.save();
  lassoCtx.lineWidth = lineWidth;
  lassoCtx.lineJoin = "round";
  lassoCtx.lineCap = "round";
  const rgb = hexToRgb(sample.family_colors[activeFamily] || "#ffffff");
  lassoCtx.strokeStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.96)`;
  lassoCtx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.22)`;
  lassoCtx.beginPath();
  lassoCtx.moveTo(lassoPoints[0].x, lassoPoints[0].y);
  for (const p of lassoPoints.slice(1)) lassoCtx.lineTo(p.x, p.y);
  if (lassoPoints.length > 2) {
    lassoCtx.closePath();
    lassoCtx.fill();
  }
  lassoCtx.stroke();
  lassoCtx.strokeStyle = "rgba(255, 255, 255, 0.82)";
  lassoCtx.lineWidth = Math.max(1, lineWidth - 1);
  lassoCtx.stroke();
  lassoCtx.restore();
}

function clearLasso() {
  lassoPoints = [];
  lassoCtx.clearRect(0, 0, W, H);
}

function clearProposalPreview() {
  proposalPreviewIndex = -1;
  proposalCtx.clearRect(0, 0, W, H);
}

function showProposalPreview(index) {
  const proposal = autoProposals[index];
  if (!proposal) return;
  showProposalObject(proposal, "previewProposal");
  proposalPreviewIndex = index;
}

function showProposalObject(proposal, label) {
  clearProposalPreview();
  const proposalMask = fromRle(proposal.rle);
  const imageData = proposalCtx.createImageData(W, H);
  const rgb = hexToRgb(sample.family_colors[proposal.family] || "#ffffff");
  for (let i = 0; i < N; i += 1) {
    if (!proposalMask[i]) continue;
    const p = i * 4;
    imageData.data[p] = rgb.r;
    imageData.data[p + 1] = rgb.g;
    imageData.data[p + 2] = rgb.b;
    imageData.data[p + 3] = 120;
  }
  proposalCtx.putImageData(imageData, 0, 0);
  document.getElementById("preview").textContent = JSON.stringify({
    [label]: proposal.family,
    pixel_count: proposal.pixel_count,
    confidence: proposal.confidence,
    parameters: proposal.parameters
  }, null, 2);
}

function setGeometryFamily(family) {
  geometryFamily = family;
  document.getElementById("edgeFit").classList.toggle("active", geometryFamily === "edge");
  document.getElementById("ringFit").classList.toggle("active", geometryFamily === "ring");
  document.getElementById("fitRadius").disabled = geometryFamily === "edge";
  setActiveFamily(family);
  updateGeometryStatus();
}

function geometryParams() {
  const minGrade = Number(document.getElementById("minGrade").value);
  const radius = Number(document.getElementById("fitRadius").value);
  const width = Number(document.getElementById("fitWidth").value);
  const thetaStart = Number(document.getElementById("fitThetaStart").value);
  const thetaEnd = Number(document.getElementById("fitThetaEnd").value);
  const params = { min_grade: minGrade, width_pct: width, theta_start_deg: thetaStart, theta_end_deg: thetaEnd };
  if (geometryFamily === "ring") params.radius_pct = radius;
  if (geometryFamily === "edge") params.radial_min_pct = 100 - width;
  return params;
}

function updateGeometryStatus() {
  if (!sample) return;
  const params = geometryParams();
  const detail = geometryFamily === "edge"
    ? `edge >= r${params.radial_min_pct}%, angle ${params.theta_start_deg}-${params.theta_end_deg}, grade >= ${params.min_grade}`
    : `ring r${params.radius_pct}% ± ${params.width_pct}%, angle ${params.theta_start_deg}-${params.theta_end_deg}, grade >= ${params.min_grade}`;
  document.getElementById("geometryStatus").textContent = detail;
}

function makeGeometryProposal() {
  const params = geometryParams();
  const out = new Uint8Array(N);
  let count = 0;
  for (let i = 0; i < N; i += 1) {
    if (!waferMask[i] || !validMask[i] || severity[i] < params.min_grade) continue;
    if (!angleInRange(thetaDeg[i], params.theta_start_deg, params.theta_end_deg)) continue;
    const radiusPct = radiusNorm[i] * 100;
    const inGeometry = geometryFamily === "edge"
      ? radiusPct >= params.radial_min_pct
      : Math.abs(radiusPct - params.radius_pct) <= params.width_pct;
    if (!inGeometry) continue;
    out[i] = 1;
    count += 1;
  }
  return {
    family: geometryFamily,
    family_label: sample.family_labels[geometryFamily],
    pixel_count: count,
    confidence: 1,
    description: `${geometryFamily} manual geometry fit`,
    parameters: params,
    rle: toRle(out)
  };
}

function angleInRange(value, start, end) {
  if (start === 0 && end === 359) return true;
  if (start <= end) return value >= start && value <= end;
  return value >= start || value <= end;
}

function previewGeometryFit() {
  const proposal = makeGeometryProposal();
  if (proposal.pixel_count === 0) {
    clearProposalPreview();
    document.getElementById("geometryStatus").textContent = "No pixels match this geometry fit";
    return;
  }
  showProposalObject(proposal, "geometryPreview");
  document.getElementById("geometryStatus").textContent = `${proposal.family} preview: ${proposal.pixel_count} px`;
}

function applyGeometryFit() {
  const proposal = makeGeometryProposal();
  if (proposal.pixel_count === 0) {
    clearProposalPreview();
    document.getElementById("geometryStatus").textContent = "No pixels to apply";
    return;
  }
  clearProposalPreview();
  saveHistory();
  const added = mergeProposalMask(proposal);
  setActiveFamily(proposal.family);
  render();
  updatePreview();
  document.getElementById("geometryStatus").textContent = `${proposal.family} applied: ${added} new px`;
  document.getElementById("preview").textContent = JSON.stringify({
    appliedGeometryFit: proposal.family,
    addedPixels: added,
    pixel_count: proposal.pixel_count,
    parameters: proposal.parameters
  }, null, 2);
}

function applyLassoFit() {
  const target = masks[activeFamily];
  const minGrade = Number(document.getElementById("minGrade").value);
  const minX = Math.max(0, Math.floor(Math.min(...lassoPoints.map(p => p.x))));
  const maxX = Math.min(W - 1, Math.ceil(Math.max(...lassoPoints.map(p => p.x))));
  const minY = Math.max(0, Math.floor(Math.min(...lassoPoints.map(p => p.y))));
  const maxY = Math.min(H - 1, Math.ceil(Math.max(...lassoPoints.map(p => p.y))));
  const candidates = [];
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      if (!pointInPolygon(x + 0.5, y + 0.5, lassoPoints)) continue;
      const idx = y * W + x;
      if (!lassoCandidateAt(idx, minGrade)) continue;
      candidates.push(idx);
    }
  }
  if (candidates.length === 0) return { addedPixels: 0, candidatePixels: 0 };
  saveHistory();
  let added = 0;
  let stbyPixels = 0;
  let gradePixels = 0;
  for (const idx of candidates) {
    if (!target[idx]) added += 1;
    if (stbyMask[idx]) stbyPixels += 1;
    else gradePixels += 1;
    target[idx] = 1;
  }
  return {
    addedPixels: added,
    candidatePixels: candidates.length,
    gradePixels,
    stbyPixels,
    bbox: [minX, minY, maxX - minX + 1, maxY - minY + 1]
  };
}

function lassoCandidateAt(idx, minGrade) {
  if (!waferMask[idx]) return false;
  if (stbyMask[idx]) return true;
  return validMask[idx] && severity[idx] >= minGrade;
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    const crosses = (yi > y) !== (yj > y);
    if (!crosses) continue;
    const xAtY = ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (x < xAtY) inside = !inside;
  }
  return inside;
}

function renderBase() {
  if (!sample) return;
  const selected = document.getElementById("colorScheme").value;
  const scheme = COLOR_SCHEMES[selected] || COLOR_SCHEMES.process;
  const imageData = baseCtx.createImageData(W, H);
  for (let i = 0; i < N; i += 1) {
    const p = i * 4;
    let rgb = scheme.outside;
    if (waferMask[i]) {
      rgb = stbyMask[i] ? scheme.stby : scheme.grades[Math.max(0, Math.min(7, severity[i] || 0))];
    }
    imageData.data[p] = rgb[0];
    imageData.data[p + 1] = rgb[1];
    imageData.data[p + 2] = rgb[2];
    imageData.data[p + 3] = 255;
  }
  baseCtx.putImageData(imageData, 0, 0);
}

function render() {
  if (!sample) return;
  renderDirty(0, 0, W - 1, H - 1);
}

function renderDirty(x0, y0, x1, y1) {
  if (!sample) return;
  x0 = Math.max(0, Math.floor(x0));
  y0 = Math.max(0, Math.floor(y0));
  x1 = Math.min(W - 1, Math.ceil(x1));
  y1 = Math.min(H - 1, Math.ceil(y1));
  if (x1 < x0 || y1 < y0) return;
  const width = x1 - x0 + 1;
  const height = y1 - y0 + 1;
  const imageData = maskCtx.createImageData(width, height);
  const alpha = Math.round(Number(document.getElementById("opacity").value) * 2.55);
  for (const name of sample.families) {
    const rgb = hexToRgb(sample.family_colors[name]);
    const target = masks[name];
    for (let y = 0; y < height; y += 1) {
      const sourceOffset = (y0 + y) * W + x0;
      const imageOffset = y * width;
      for (let x = 0; x < width; x += 1) {
        if (!target[sourceOffset + x]) continue;
        const p = (imageOffset + x) * 4;
        imageData.data[p] = rgb.r;
        imageData.data[p + 1] = rgb.g;
        imageData.data[p + 2] = rgb.b;
        imageData.data[p + 3] = alpha;
      }
    }
  }
  maskCtx.putImageData(imageData, x0, y0);
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return { r: parseInt(value.slice(0, 2), 16), g: parseInt(value.slice(2, 4), 16), b: parseInt(value.slice(4, 6), 16) };
}

function decodeBytes(encoded, expectedLength) {
  const raw = atob(encoded);
  const array = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) array[i] = raw.charCodeAt(i);
  if (array.length !== expectedLength) throw new Error(`decoded array length mismatch: ${array.length} !== ${expectedLength}`);
  return array;
}

function saveHistory() {
  history.push(Object.fromEntries(sample.families.map(name => [name, masks[name].slice()])));
  if (history.length > 20) history.shift();
}

function undo() {
  const previous = history.pop();
  if (!previous) return;
  for (const name of sample.families) masks[name].set(previous[name]);
  render();
  updatePreview();
}

function clearFamily() {
  saveHistory();
  clearProposalPreview();
  masks[activeFamily].fill(0);
  render();
  updatePreview();
}

function clearAll() {
  saveHistory();
  clearProposalPreview();
  for (const name of sample.families) masks[name].fill(0);
  render();
  updatePreview();
}

function growFromPaint() {
  const target = masks[activeFamily];
  const minGrade = Number(document.getElementById("minGrade").value);
  const queue = new Int32Array(N);
  const visited = new Uint8Array(N);
  let head = 0;
  let tail = 0;
  for (let i = 0; i < N; i += 1) {
    if (target[i] && waferMask[i] && validMask[i] && severity[i] >= minGrade) {
      visited[i] = 1;
      queue[tail] = i;
      tail += 1;
    }
  }
  if (tail === 0) {
    document.getElementById("status").textContent = `No seed pixels at grade >= ${minGrade}`;
    return;
  }
  saveHistory();
  while (head < tail) {
    const current = queue[head];
    head += 1;
    const x = current % W;
    const y = Math.floor(current / W);
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (dx === 0 && dy === 0) continue;
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const next = ny * W + nx;
        if (visited[next] || !waferMask[next] || !validMask[next] || severity[next] < minGrade) continue;
        visited[next] = 1;
        queue[tail] = next;
        tail += 1;
      }
    }
  }
  for (let i = 0; i < N; i += 1) if (visited[i]) target[i] = 1;
  render();
  updatePreview();
}

function addGradeArea() {
  const target = masks[activeFamily];
  const minGrade = Number(document.getElementById("minGrade").value);
  saveHistory();
  for (let i = 0; i < N; i += 1) {
    if (waferMask[i] && validMask[i] && severity[i] >= minGrade) target[i] = 1;
  }
  render();
  updatePreview();
}

function traceScratchLine() {
  const target = masks[activeFamily];
  const minGrade = Number(document.getElementById("minGrade").value);
  const seed = [];
  for (let i = 0; i < N; i += 1) {
    if (target[i] && waferMask[i] && validMask[i]) seed.push([i % W, Math.floor(i / W)]);
  }
  if (seed.length < 2) {
    document.getElementById("status").textContent = "Trace needs at least two painted seed pixels";
    return;
  }
  saveHistory();
  const meanX = seed.reduce((sum, p) => sum + p[0], 0) / seed.length;
  const meanY = seed.reduce((sum, p) => sum + p[1], 0) / seed.length;
  let sxx = 0, sxy = 0, syy = 0;
  for (const [x, y] of seed) {
    const dx = x - meanX;
    const dy = y - meanY;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const ux = Math.cos(angle);
  const uy = Math.sin(angle);
  let minT = Infinity, maxT = -Infinity;
  for (const [x, y] of seed) {
    const t = (x - meanX) * ux + (y - meanY) * uy;
    minT = Math.min(minT, t);
    maxT = Math.max(maxT, t);
  }
  const brush = Number(document.getElementById("brush").value);
  const halfWidth = Math.max(1, Math.round(brush / 3));
  const extension = Math.max(4, brush * 2);
  minT -= extension;
  maxT += extension;
  const minX = Math.max(0, Math.floor(meanX + Math.min(minT * ux, maxT * ux) - halfWidth - 2));
  const maxX = Math.min(W - 1, Math.ceil(meanX + Math.max(minT * ux, maxT * ux) + halfWidth + 2));
  const minY = Math.max(0, Math.floor(meanY + Math.min(minT * uy, maxT * uy) - halfWidth - 2));
  const maxY = Math.min(H - 1, Math.ceil(meanY + Math.max(minT * uy, maxT * uy) + halfWidth + 2));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const idx = y * W + x;
      if (!waferMask[idx] || !validMask[idx] || severity[idx] < minGrade) continue;
      const dx = x - meanX;
      const dy = y - meanY;
      const t = dx * ux + dy * uy;
      if (t < minT || t > maxT) continue;
      const distance = Math.abs(dx * -uy + dy * ux);
      if (distance <= halfWidth) target[idx] = 1;
    }
  }
  render();
  updatePreview();
}

function toRle(array) {
  const runs = [];
  let start = -1;
  for (let i = 0; i < array.length; i += 1) {
    if (array[i] && start < 0) start = i;
    if ((!array[i] || i === array.length - 1) && start >= 0) {
      const end = array[i] && i === array.length - 1 ? i + 1 : i;
      runs.push([start, end - start]);
      start = -1;
    }
  }
  return runs;
}

function fromRle(runs) {
  const array = new Uint8Array(N);
  for (const run of runs || []) {
    const start = Number(run[0]);
    const length = Number(run[1]);
    for (let i = start; i < start + length && i < N; i += 1) array[i] = 1;
  }
  return array;
}

function payload() {
  const saveMode = document.querySelector("input[name='saveMode']:checked").value;
  return { sample_id: sample.sample_id, save_mode: saveMode, masks: Object.fromEntries(sample.families.map(name => [name, toRle(masks[name])])) };
}

function pixelCount(array) {
  let count = 0;
  for (let i = 0; i < array.length; i += 1) count += array[i] ? 1 : 0;
  return count;
}

function updatePreview() {
  if (!sample) return;
  const counts = Object.fromEntries(sample.families.map(name => [name, pixelCount(masks[name])]));
  const minGrade = Number(document.getElementById("minGrade").value);
  const saveMode = document.querySelector("input[name='saveMode']:checked").value;
  const activeLabel = sample.family_labels[activeFamily] || activeFamily;
  const nonzero = Object.entries(counts).filter(([, value]) => value > 0).map(([name, value]) => `${name}:${value}`);
  document.getElementById("status").textContent = Object.entries(counts).map(([k, v]) => `${k}:${v}`).join(" · ");
  document.getElementById("status").textContent = `${activeLabel} active | ${nonzero.length ? nonzero.join(" | ") : "no mask pixels yet"}`;
  document.getElementById("preview").textContent = JSON.stringify({ activeFamily, counts, minGrade, saveMode, rule: "multi-label masks, crop bbox + 20% margin, max composition" }, null, 2);
}

async function saveAssets() {
  document.getElementById("status").textContent = "Saving assets...";
  const response = await fetch("/api/save-assets", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload()) });
  const result = await response.json();
  document.getElementById("status").textContent = `Saved ${result.saved_count} assets`;
  document.getElementById("preview").textContent = JSON.stringify(result, null, 2);
  loadAssets();
}

async function loadPrediction() {
  const response = await fetch("/api/predictions");
  const result = await response.json();
  if (!result.masks || Object.keys(result.masks).length === 0) {
    document.getElementById("status").textContent = "No prediction masks for this sample";
    return;
  }
  saveHistory();
  clearProposalPreview();
  for (const name of sample.families) {
    if (result.masks[name]) masks[name].set(fromRle(result.masks[name]));
  }
  render();
  updatePreview();
  document.getElementById("preview").textContent = JSON.stringify({ loadedPrediction: result.sample_id, counts: Object.fromEntries(sample.families.map(name => [name, pixelCount(masks[name])])) }, null, 2);
}

async function analyzeAuto() {
  const minGrade = Number(document.getElementById("minGrade").value);
  document.getElementById("proposalStatus").textContent = "Analyzing...";
  clearProposalPreview();
  const response = await fetch(`/api/auto-proposals?min_grade=${minGrade}`);
  const result = await response.json();
  autoProposals = result.proposals || [];
  renderProposalList();
  document.getElementById("proposalStatus").textContent = `${autoProposals.length} proposals at grade >= ${result.min_grade}`;
  document.getElementById("preview").textContent = JSON.stringify({
    autoProposals: autoProposals.map(proposal => ({
      family: proposal.family,
      pixel_count: proposal.pixel_count,
      confidence: proposal.confidence,
      parameters: proposal.parameters
    }))
  }, null, 2);
}

async function loadModelProposals() {
  document.getElementById("proposalStatus").textContent = "Loading model proposals...";
  clearProposalPreview();
  const response = await fetch("/api/model-proposals");
  const result = await response.json();
  autoProposals = result.proposals || [];
  renderProposalList();
  document.getElementById("proposalStatus").textContent = autoProposals.length
    ? `${autoProposals.length} model proposals loaded`
    : "No model proposals configured";
  document.getElementById("preview").textContent = JSON.stringify({
    modelProposals: autoProposals.map(proposal => ({
      family: proposal.family,
      pixel_count: proposal.pixel_count,
      confidence: proposal.confidence,
      parameters: proposal.parameters
    }))
  }, null, 2);
}

function renderProposalList() {
  const root = document.getElementById("proposalList");
  root.innerHTML = "";
  if (autoProposals.length === 0) {
    root.innerHTML = '<p class="status">No proposals loaded.</p>';
    return;
  }
  autoProposals.forEach((proposal, index) => {
    const card = document.createElement("article");
    card.className = "proposal-card";
    const color = sample.family_colors[proposal.family] || "#8e8e93";
    card.innerHTML = `
      <strong>
        <span><span class="swatch" style="background:${color}"></span>${escapeHtml(proposal.family_label)}</span>
        <span>${Math.round(Number(proposal.confidence) * 100)}%</span>
      </strong>
      <p>${escapeHtml(proposal.description)} · ${proposal.pixel_count} px</p>
      <div class="proposal-actions">
        <button data-preview-index="${index}">Preview</button>
        <button data-apply-index="${index}">Apply</button>
      </div>`;
    const description = card.querySelector("p");
    if (description) description.textContent = `${proposal.description} - ${proposal.pixel_count} px`;
    card.querySelector("[data-preview-index]").onclick = () => showProposalPreview(index);
    card.querySelector("[data-apply-index]").onclick = () => applyProposal(index);
    root.appendChild(card);
  });
}

function applyProposal(index) {
  const proposal = autoProposals[index];
  if (!proposal || !masks[proposal.family]) return;
  clearProposalPreview();
  saveHistory();
  const added = mergeProposalMask(proposal);
  setActiveFamily(proposal.family);
  render();
  updatePreview();
  document.getElementById("preview").textContent = JSON.stringify({
    appliedProposal: proposal.family,
    addedPixels: added,
    parameters: proposal.parameters
  }, null, 2);
}

function applyAllProposals() {
  if (autoProposals.length === 0) {
    document.getElementById("proposalStatus").textContent = "No proposals to apply";
    return;
  }
  clearProposalPreview();
  saveHistory();
  const added = {};
  for (const proposal of autoProposals) {
    if (!masks[proposal.family]) continue;
    added[proposal.family] = (added[proposal.family] || 0) + mergeProposalMask(proposal);
  }
  render();
  updatePreview();
  document.getElementById("preview").textContent = JSON.stringify({ appliedAllProposals: added }, null, 2);
}

function mergeProposalMask(proposal) {
  const proposalMask = fromRle(proposal.rle);
  const target = masks[proposal.family];
  let added = 0;
  for (let i = 0; i < N; i += 1) {
    if (!proposalMask[i]) continue;
    if (!target[i]) added += 1;
    target[i] = 1;
  }
  return added;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

async function loadAssets() {
  const response = await fetch("/api/assets");
  const result = await response.json();
  const assets = result.assets || [];
  const root = document.getElementById("assetList");
  document.getElementById("assetStatus").textContent = `${assets.length} saved assets`;
  root.innerHTML = "";
  for (const asset of assets.slice().reverse()) {
    const card = document.createElement("article");
    card.className = "asset-card";
    const color = sample.family_colors[asset.family] || "#8e8e93";
    card.innerHTML = `
      <div class="asset-title">
        <strong><span class="swatch" style="background:${color}"></span>${asset.family_label}</strong>
        <span>${asset.asset_id}</span>
      </div>
      <img src="/assets/${asset.relative_path}/preview.png?ts=${Date.now()}" alt="${asset.asset_id} preview">
      <div class="asset-meta">
        pixels ${asset.mask_pixel_count} · grade ${asset.grade_min}-${asset.grade_max}<br>
        bbox ${JSON.stringify(asset.bbox_xywh)} · valid ${asset.valid}
      </div>`;
    root.appendChild(card);
  }
}
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = Path(args.manifest).resolve()
    sample = load_manifest_sample(manifest, args.sample_id)
    assets_root = Path(args.assets_root)
    if not assets_root.is_absolute():
        assets_root = ROOT / assets_root
    handler = PatternAssetEditorHandler
    handler.sample = sample
    handler.manifest_path = manifest
    handler.assets_root = assets_root.resolve()
    handler.margin_ratio = float(args.margin_ratio)
    handler.editor_shape = editor_shape(sample.shape, int(args.editor_max_size))
    prediction_path = Path(args.prediction_json).resolve() if args.prediction_json else None
    proposal_path = Path(args.proposal_json).resolve() if args.proposal_json else None
    handler.prediction_masks = load_prediction_masks(prediction_path, sample.sample_id, sample.shape)
    handler.model_proposals = load_model_proposals(proposal_path, sample.sample_id, sample.shape, handler.editor_shape)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Pattern Asset Builder: {url}")
    print(f"Saving assets under: {handler.assets_root}")
    if handler.editor_shape != sample.shape:
        print(
            "Editing downsampled canvas: "
            f"{handler.editor_shape[1]}x{handler.editor_shape[0]} from {sample.shape[1]}x{sample.shape[0]}"
        )
    if handler.model_proposals:
        print(f"Loaded model proposals: {len(handler.model_proposals)}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Pattern Asset Builder")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
