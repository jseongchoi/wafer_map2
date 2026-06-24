"""Run a local web editor for extracting FBM defect pattern assets."""

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
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.assets import FAMILY_COLORS, FAMILY_LABELS, TARGET_FAMILIES, scan_pattern_assets
DISPLAY_COLORS: dict[int, tuple[int, int, int]] = {
    0: (247, 248, 248),
    1: (88, 166, 255),
    2: (35, 203, 167),
    3: (118, 219, 87),
    4: (234, 221, 72),
    5: (245, 151, 54),
    6: (220, 72, 66),
    7: (122, 24, 28),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="real_unlabeled_manifest/v1 JSON path.")
    parser.add_argument("--sample-id", help="Sample ID to edit. Defaults to the first manifest sample.")
    parser.add_argument("--assets-root", default="data/pattern_assets", help="Output root for family asset folders.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--margin-ratio", type=float, default=0.20)
    parser.add_argument("--prediction-json", help="Optional fbm_prediction_masks/v1 JSON to prefill editable masks.")
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


def preview_rgb(sample: Any) -> np.ndarray:
    image = np.zeros((*sample.shape, 3), dtype=np.uint8)
    image[:] = (28, 36, 33)
    wafer = sample.wafer_mask > 0
    for grade, color in DISPLAY_COLORS.items():
        image[wafer & (sample.severity == grade)] = color
    image[sample.stby_mask > 0] = (166, 216, 240)
    return image


def png_bytes(array: np.ndarray, mode: str | None = None) -> bytes:
    with io.BytesIO() as buffer:
        image = Image.fromarray(array, mode=mode) if mode else Image.fromarray(array)
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def bytes_b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array.astype(np.uint8)).tobytes()).decode("ascii")


def rle_to_mask(runs: list[list[int]], shape: tuple[int, int]) -> np.ndarray:
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for run in runs:
        if len(run) != 2:
            raise ValueError(f"invalid RLE run: {run}")
        start, length = int(run[0]), int(run[1])
        if start < 0 or length < 0 or start + length > flat.size:
            raise ValueError(f"RLE run outside mask bounds: {run}")
        flat[start : start + length] = 1
    return flat.reshape(shape).astype(bool)


def mask_to_rle(mask: np.ndarray) -> list[list[int]]:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    runs: list[list[int]] = []
    start = -1
    for idx, value in enumerate(flat):
        if value and start < 0:
            start = idx
        if (not value or idx == flat.size - 1) and start >= 0:
            end = idx + 1 if value and idx == flat.size - 1 else idx
            runs.append([int(start), int(end - start)])
            start = -1
    return runs


def load_prediction_masks(path: Path | None, sample_id: str, shape: tuple[int, int]) -> dict[str, list[list[int]]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != "fbm_prediction_masks/v1":
        raise ValueError(f"unsupported prediction schema: {payload.get('schema_version')}")
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("prediction JSON samples must be a list")
    for sample in samples:
        if str(sample.get("sample_id", "")) != sample_id:
            continue
        masks = sample.get("masks", {})
        if not isinstance(masks, dict):
            raise ValueError("prediction sample masks must be an object")
        out: dict[str, list[list[int]]] = {}
        for family in TARGET_FAMILIES:
            runs = masks.get(family, [])
            mask = rle_to_mask(runs, shape)
            out[family] = mask_to_rle(mask)
        return out
    return {}


def save_pattern_assets(
    *,
    sample: Any,
    masks_by_family: dict[str, np.ndarray],
    assets_root: Path,
    margin_ratio: float,
    source_manifest: Path | None = None,
    split_components: bool = False,
) -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    for family in TARGET_FAMILIES:
        mask = masks_by_family.get(family)
        if mask is None:
            continue
        mask = mask & (sample.wafer_mask > 0)
        components = connected_components(mask) if split_components else ([mask] if mask.any() else [])
        for component in components:
            bbox = bbox_with_margin(component, sample.shape, margin_ratio)
            asset_id = next_asset_id(assets_root / family, sample.sample_id, family)
            asset_dir = assets_root / family / asset_id
            write_asset(
                asset_dir=asset_dir,
                sample=sample,
                family=family,
                component=component,
                bbox=bbox,
                margin_ratio=margin_ratio,
                source_manifest=source_manifest,
            )
            saved.append({"family": family, "asset_id": asset_id, "path": str(asset_dir), "bbox": bbox})
    return saved


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            coords.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        component = np.zeros(mask.shape, dtype=bool)
        for cy, cx in coords:
            component[cy, cx] = True
        components.append(component)
    return components


def bbox_with_margin(component: np.ndarray, shape: tuple[int, int], margin_ratio: float) -> list[int]:
    ys, xs = np.nonzero(component)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    margin_y = max(1, int(round((y1 - y0) * margin_ratio)))
    margin_x = max(1, int(round((x1 - x0) * margin_ratio)))
    y0 = max(0, y0 - margin_y)
    x0 = max(0, x0 - margin_x)
    y1 = min(shape[0], y1 + margin_y)
    x1 = min(shape[1], x1 + margin_x)
    return [x0, y0, x1 - x0, y1 - y0]


def next_asset_id(family_root: Path, sample_id: str, family: str) -> str:
    family_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{safe_name(sample_id)}_{family}_"
    existing = [path.name for path in family_root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    numbers = []
    for name in existing:
        try:
            numbers.append(int(name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    return f"{prefix}{(max(numbers) + 1 if numbers else 1):04d}"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "sample"


def write_asset(
    *,
    asset_dir: Path,
    sample: Any,
    family: str,
    component: np.ndarray,
    bbox: list[int],
    margin_ratio: float,
    source_manifest: Path | None,
) -> None:
    x, y, width, height = bbox
    crop = np.s_[y : y + height, x : x + width]
    asset_dir.mkdir(parents=True, exist_ok=True)
    local_mask = component[crop]
    grade_patch = sample.severity[crop].astype(np.uint8)
    mask_png = (local_mask.astype(np.uint8) * 255)
    preview = preview_rgb(sample)[crop]
    Image.fromarray(grade_patch, mode="L").save(asset_dir / "grade.png")
    Image.fromarray(mask_png, mode="L").save(asset_dir / "mask.png")
    Image.fromarray(preview, mode="RGB").save(asset_dir / "preview.png")
    masked_values = grade_patch[local_mask]
    metadata = {
        "schema_version": "fbm_pattern_asset/v1",
        "family": family,
        "family_label": FAMILY_LABELS[family],
        "source_sample_id": sample.sample_id,
        "bbox_xywh": bbox,
        "source_image_shape": {"height": int(sample.shape[0]), "width": int(sample.shape[1])},
        "margin_ratio": float(margin_ratio),
        "composition_rule": "max",
        "mask_pixel_count": int(local_mask.sum()),
        "grade_min": int(masked_values.min()) if len(masked_values) else 0,
        "grade_max": int(masked_values.max()) if len(masked_values) else 0,
        "multi_label": True,
        "stby_target_excluded": True,
    }
    if source_manifest is not None:
        metadata["source_manifest_name"] = source_manifest.name
    (asset_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


class PatternAssetEditorHandler(BaseHTTPRequestHandler):
    sample: Any
    manifest_path: Path
    assets_root: Path
    margin_ratio: float
    prediction_masks: dict[str, list[list[int]]]

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_bytes(EDITOR_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
        elif path == "/library":
            self.send_bytes(
                asset_library_html(scan_pattern_assets(self.assets_root), Path("/library")).encode("utf-8"),
                content_type="text/html; charset=utf-8",
            )
        elif path == "/api/sample":
            payload = {
                "sample_id": self.sample.sample_id,
                "width": int(self.sample.shape[1]),
                "height": int(self.sample.shape[0]),
                "families": list(TARGET_FAMILIES),
                "family_labels": FAMILY_LABELS,
                "family_colors": FAMILY_COLORS,
                "assets_root": str(self.assets_root),
                "margin_ratio": float(self.margin_ratio),
                "stby_target_excluded": True,
                "composition_rule": "max",
                "severity_b64": bytes_b64(self.sample.severity),
                "wafer_mask_b64": bytes_b64(self.sample.wafer_mask),
                "valid_mask_b64": bytes_b64(self.sample.valid_test_mask),
            }
            self.send_json(payload)
        elif path == "/api/assets":
            self.send_json({"assets": scan_pattern_assets(self.assets_root)})
        elif path == "/api/predictions":
            self.send_json({"sample_id": self.sample.sample_id, "masks": self.prediction_masks})
        elif path == "/base.png":
            self.send_bytes(png_bytes(preview_rgb(self.sample)), content_type="image/png")
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
            family: rle_to_mask(masks_payload.get(family, []), self.sample.shape)
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
    body { margin: 0; background: #eef2f1; color: #17211f; font-family: "Segoe UI", "Noto Sans KR", Arial, sans-serif; }
    .app { min-height: 100vh; display: grid; grid-template-columns: 1fr 380px; grid-template-rows: 56px 1fr; }
    header { grid-column: 1 / -1; display: flex; align-items: center; gap: 14px; padding: 0 16px; border-bottom: 1px solid #d4ddda; background: #fff; }
    h1 { margin: 0; font-size: 18px; }
    .meta, .status { color: #66736f; font-size: 13px; }
    main { min-width: 0; display: grid; place-items: center; padding: 14px; overflow: auto; }
    .canvas-wrap { position: relative; width: min(100%, calc(100vh - 96px)); aspect-ratio: 1 / 1; background: #1c2421; border: 1px solid #d4ddda; border-radius: 8px; overflow: hidden; box-shadow: 0 16px 34px rgba(23, 33, 31, 0.12); }
    canvas { position: absolute; inset: 0; width: 100%; height: 100%; image-rendering: pixelated; }
    aside { padding: 14px; border-left: 1px solid #d4ddda; background: #fff; overflow: auto; }
    section { padding: 14px 0; border-bottom: 1px solid #d4ddda; }
    section:first-child { padding-top: 0; }
    section:last-child { border-bottom: 0; }
    h2 { margin: 0 0 10px; font-size: 14px; }
    button { border: 1px solid #d4ddda; border-radius: 8px; background: #f9fbfa; color: #17211f; min-height: 38px; padding: 8px 10px; font: inherit; cursor: pointer; }
    button.active { border-color: #25745d; box-shadow: 0 0 0 2px rgba(37, 116, 93, 0.18); }
    button.primary { background: #25745d; color: white; border-color: #25745d; }
    button.danger { color: #9a3d38; }
    .families, .toolbar { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .row { display: grid; grid-template-columns: 112px 1fr; gap: 8px; align-items: center; margin: 10px 0; }
    .row span { color: #66736f; font-size: 13px; }
    .radio-grid { display: grid; gap: 8px; }
    .option { display: flex; align-items: flex-start; gap: 8px; border: 1px solid #d4ddda; border-radius: 8px; background: #fbfcfc; padding: 9px; font-size: 13px; cursor: pointer; }
    .option input { margin-top: 2px; }
    .option strong { display: block; font-size: 13px; }
    .option span { display: block; color: #66736f; }
    input[type="range"] { width: 100%; }
    .swatch { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
    .asset-list { display: grid; gap: 10px; max-height: 360px; overflow: auto; padding-right: 2px; }
    .asset-card { border: 1px solid #d4ddda; border-radius: 8px; background: #fbfcfc; padding: 8px; }
    .asset-card img { width: 100%; border: 1px solid #d4ddda; border-radius: 6px; background: #17211f; image-rendering: pixelated; }
    .asset-title { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; margin-bottom: 6px; }
    .asset-meta { color: #66736f; font-size: 12px; margin-top: 6px; }
    pre { margin: 8px 0 0; max-height: 210px; overflow: auto; background: #f5f7f8; border: 1px solid #d4ddda; border-radius: 8px; padding: 10px; font-size: 12px; white-space: pre-wrap; }
    @media (max-width: 940px) { .app { grid-template-columns: 1fr; grid-template-rows: 56px minmax(360px, 1fr) auto; } aside { border-left: 0; border-top: 1px solid #d4ddda; } .canvas-wrap { width: min(100%, 760px); } }
  </style>
</head>
<body>
<div class="app">
  <header><h1>FBM Pattern Asset Builder</h1><div class="meta" id="meta"></div></header>
  <main><div class="canvas-wrap"><canvas id="base"></canvas><canvas id="mask"></canvas></div></main>
  <aside>
    <section><h2>Family</h2><div class="families" id="families"></div></section>
    <section>
      <h2>Brush</h2>
      <div class="toolbar"><button id="paint" class="active">Paint</button><button id="erase">Erase</button></div>
      <div class="row"><span>Size</span><input id="brush" type="range" min="1" max="80" value="14"></div>
      <div class="row"><span>Opacity</span><input id="opacity" type="range" min="15" max="90" value="55"></div>
    </section>
    <section>
      <h2>Smart Fit</h2>
      <div class="row"><span>Min Grade</span><input id="minGrade" type="range" min="1" max="7" value="3"></div>
      <div class="toolbar"><button id="growPaint">Grow From Paint</button><button id="addGrade">Add Grade Area</button></div>
      <div class="toolbar" style="margin-top:8px"><button id="traceLine">Trace Scratch Line</button><button id="clearAssist" class="danger">Clear Assist</button></div>
    </section>
    <section>
      <h2>Save Mode</h2>
      <div class="radio-grid">
        <label class="option"><input type="radio" name="saveMode" value="family" checked><span><strong>One Family Asset</strong><span>ring처럼 끊긴 패턴도 하나로 저장</span></span></label>
        <label class="option"><input type="radio" name="saveMode" value="components"><span><strong>Split Components</strong><span>독립 blob 여러 개를 따로 저장</span></span></label>
      </div>
    </section>
    <section>
      <h2>History</h2>
      <div class="toolbar"><button id="undo">Undo</button><button id="clearFamily" class="danger">Clear Family</button></div>
      <div class="toolbar" style="margin-top:8px"><button id="loadPrediction">Load Prediction</button><button id="clearAll" class="danger">Clear All</button></div>
      <div class="toolbar" style="margin-top:8px"><button id="save" class="primary">Save Assets</button></div>
    </section>
    <section>
      <h2>Output</h2>
      <p class="status" id="status">Loading</p>
      <pre id="preview"></pre>
    </section>
    <section>
      <h2>Saved Asset Library</h2>
      <div class="toolbar"><button id="refreshAssets">Refresh</button><button id="openReport">Open Report</button></div>
      <p class="status" id="assetStatus">No assets loaded</p>
      <div class="asset-list" id="assetList"></div>
    </section>
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
const base = document.getElementById("base");
const mask = document.getElementById("mask");
const baseCtx = base.getContext("2d");
const maskCtx = mask.getContext("2d");

init();

async function init() {
  sample = await (await fetch("/api/sample")).json();
  W = sample.width; H = sample.height; N = W * H;
  base.width = mask.width = W;
  base.height = mask.height = H;
  activeFamily = sample.families[0];
  masks = Object.fromEntries(sample.families.map(name => [name, new Uint8Array(N)]));
  severity = decodeBytes(sample.severity_b64, N);
  waferMask = decodeBytes(sample.wafer_mask_b64, N);
  validMask = decodeBytes(sample.valid_mask_b64, N);
  document.getElementById("meta").textContent = `${sample.sample_id} · ${W} x ${H} · assets: ${sample.assets_root}`;
  buildFamilies();
  bindEvents();
  const image = new Image();
  image.onload = () => baseCtx.drawImage(image, 0, 0, W, H);
  image.src = "/base.png";
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
      activeFamily = name;
      document.querySelectorAll("[data-family]").forEach(btn => btn.classList.toggle("active", btn.dataset.family === name));
      updatePreview();
    };
    root.appendChild(button);
  }
  root.querySelector("button").classList.add("active");
}

function bindEvents() {
  document.getElementById("paint").onclick = () => setMode("paint");
  document.getElementById("erase").onclick = () => setMode("erase");
  document.getElementById("undo").onclick = undo;
  document.getElementById("clearFamily").onclick = clearFamily;
  document.getElementById("clearAll").onclick = clearAll;
  document.getElementById("save").onclick = saveAssets;
  document.getElementById("loadPrediction").onclick = loadPrediction;
  document.getElementById("growPaint").onclick = growFromPaint;
  document.getElementById("addGrade").onclick = addGradeArea;
  document.getElementById("traceLine").onclick = traceScratchLine;
  document.getElementById("clearAssist").onclick = clearFamily;
  document.getElementById("refreshAssets").onclick = loadAssets;
  document.getElementById("openReport").onclick = () => window.open("/library", "_blank");
  document.getElementById("opacity").oninput = render;
  document.getElementById("minGrade").oninput = updatePreview;
  document.querySelectorAll("input[name='saveMode']").forEach(input => { input.onchange = updatePreview; });
  mask.onpointerdown = event => { saveHistory(); drawing = true; paintAt(event); mask.setPointerCapture(event.pointerId); };
  mask.onpointermove = event => { if (drawing) paintAt(event); };
  mask.onpointerup = () => { drawing = false; updatePreview(); };
  mask.onpointercancel = () => { drawing = false; };
}

function setMode(next) {
  mode = next;
  document.getElementById("paint").classList.toggle("active", mode === "paint");
  document.getElementById("erase").classList.toggle("active", mode === "erase");
}

function point(event) {
  const rect = mask.getBoundingClientRect();
  return { x: Math.floor((event.clientX - rect.left) * W / rect.width), y: Math.floor((event.clientY - rect.top) * H / rect.height) };
}

function paintAt(event) {
  const p = point(event);
  const radius = Number(document.getElementById("brush").value);
  const target = masks[activeFamily];
  const value = mode === "paint" ? 1 : 0;
  const r2 = radius * radius;
  for (let y = Math.max(0, p.y - radius); y <= Math.min(H - 1, p.y + radius); y += 1) {
    for (let x = Math.max(0, p.x - radius); x <= Math.min(W - 1, p.x + radius); x += 1) {
      const dx = x - p.x, dy = y - p.y;
      if (dx * dx + dy * dy <= r2) target[y * W + x] = value;
    }
  }
  render();
}

function render() {
  if (!sample) return;
  const imageData = maskCtx.createImageData(W, H);
  const alpha = Math.round(Number(document.getElementById("opacity").value) * 2.55);
  for (const name of sample.families) {
    const rgb = hexToRgb(sample.family_colors[name]);
    const target = masks[name];
    for (let i = 0; i < N; i += 1) {
      if (!target[i]) continue;
      const p = i * 4;
      imageData.data[p] = rgb.r;
      imageData.data[p + 1] = rgb.g;
      imageData.data[p + 2] = rgb.b;
      imageData.data[p + 3] = alpha;
    }
  }
  maskCtx.putImageData(imageData, 0, 0);
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
  masks[activeFamily].fill(0);
  render();
  updatePreview();
}

function clearAll() {
  saveHistory();
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
  document.getElementById("status").textContent = Object.entries(counts).map(([k, v]) => `${k}:${v}`).join(" · ");
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
  for (const name of sample.families) {
    if (result.masks[name]) masks[name].set(fromRle(result.masks[name]));
  }
  render();
  updatePreview();
  document.getElementById("preview").textContent = JSON.stringify({ loadedPrediction: result.sample_id, counts: Object.fromEntries(sample.families.map(name => [name, pixelCount(masks[name])])) }, null, 2);
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
    prediction_path = Path(args.prediction_json).resolve() if args.prediction_json else None
    handler.prediction_masks = load_prediction_masks(prediction_path, sample.sample_id, sample.shape)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Pattern Asset Builder: {url}")
    print(f"Saving assets under: {handler.assets_root}")
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
