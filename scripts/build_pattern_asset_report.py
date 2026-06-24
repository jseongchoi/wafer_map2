"""Build an HTML review report for saved FBM pattern assets."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.assets import FAMILY_COLORS, FAMILY_LABELS, TARGET_FAMILIES, scan_pattern_assets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", default="data/pattern_assets")
    parser.add_argument("--out", default="outputs/reports/pattern_asset_library_report.html")
    return parser.parse_args(argv)


def relpath(target: str | Path, base_file: Path) -> str:
    return os.path.relpath(Path(target).resolve(), base_file.resolve().parent).replace("\\", "/")


def html_report(assets: list[dict[str, Any]], out: Path) -> str:
    family_counts = {family: 0 for family in TARGET_FAMILIES}
    for asset in assets:
        family_counts[str(asset["family"])] = family_counts.get(str(asset["family"]), 0) + 1
    count_cards = "\n".join(
        f"""<div class="metric"><strong>{family_counts.get(family, 0)}</strong><span>{html.escape(FAMILY_LABELS[family])}</span></div>"""
        for family in TARGET_FAMILIES
    )
    cards = "\n".join(asset_card(asset, out) for asset in assets)
    if not cards:
        cards = '<p class="muted">No saved pattern assets yet.</p>'
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FBM Pattern Asset Library Report</title>
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
  <h1>FBM Pattern Asset Library Report</h1>
  <p class="muted">저장된 defect 누끼 asset 검수 리포트입니다. 이 리포트는 학습 전 데이터 품질을 확인하기 위한 산출물입니다.</p>
  <section class="metrics">{count_cards}</section>
  <h2>Saved Assets</h2>
  <section class="grid">{cards}</section>
</main>
</body>
</html>
"""


def asset_card(asset: dict[str, Any], out: Path) -> str:
    family = str(asset["family"])
    color = FAMILY_COLORS.get(family, "#8e8e93")
    return f"""<article class="asset">
  <h3><span><span class="swatch" style="background:{html.escape(color)}"></span>{html.escape(str(asset["family_label"]))}</span><span>{html.escape(str(asset["asset_id"]))}</span></h3>
  <img src="{html.escape(relpath(asset["preview_path"], out))}" alt="{html.escape(str(asset["asset_id"]))} preview">
  <dl>
    <dt>valid</dt><dd>{html.escape(str(asset["valid"]))}</dd>
    <dt>pixels</dt><dd>{html.escape(str(asset["mask_pixel_count"]))}</dd>
    <dt>grade</dt><dd>{html.escape(str(asset["grade_min"]))} - {html.escape(str(asset["grade_max"]))}</dd>
    <dt>bbox</dt><dd>{html.escape(json.dumps(asset["bbox_xywh"], ensure_ascii=False))}</dd>
    <dt>source</dt><dd>{html.escape(str(asset["source_sample_id"]))}</dd>
    <dt>files</dt><dd><a href="{html.escape(relpath(asset["metadata_path"], out))}">metadata</a> · <a href="{html.escape(relpath(asset["mask_path"], out))}">mask</a> · <a href="{html.escape(relpath(asset["grade_path"], out))}">grade</a></dd>
  </dl>
</article>"""


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    assets_root = Path(args.assets_root)
    if not assets_root.is_absolute():
        assets_root = ROOT / assets_root
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    assets = scan_pattern_assets(assets_root)
    out.write_text(html_report(assets, out), encoding="utf-8")
    print(f"Wrote pattern asset report: {out}")
    print(f"Assets: {len(assets)}")


if __name__ == "__main__":
    main()
