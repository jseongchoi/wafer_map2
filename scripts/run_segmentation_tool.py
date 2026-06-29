"""Run the in-repo FBM segmentation tool."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ENGINE_PATH = Path(__file__).resolve().with_name("run_pattern_asset_editor.py")

spec = importlib.util.spec_from_file_location("_wafermap_segmentation_tool_engine", ENGINE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load segmentation tool engine: {ENGINE_PATH}")

_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_engine)

for _name in dir(_engine):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_engine, _name)


if __name__ == "__main__":
    sys.exit(main())
