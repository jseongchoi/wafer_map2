# Architecture

이 문서는 WaferMap repository를 처음 보는 사람이 “무엇을 어디에서 고쳐야 하는지” 바로 판단할 수 있게 하는 구조 가이드입니다.

## Product Boundary

WaferMap은 CVAT를 대체하는 annotation UI가 아닙니다. 이 repository의 책임은 wafer-specific data pipeline입니다.

```text
CVAT:
  task management, annotator UI, polygon/brush labeling, review workflow

WaferMap:
  wafer manifest
  CVAT image package export
  CVAT annotation import
  reusable pattern asset library
  hybrid synthetic data generation
  segmentation readiness / smoke validation
  training and evaluation entrypoints
```

## Top-Level Folders

| Path | Responsibility |
|---|---|
| `configs/` | Reproducible configuration and schemas. Label definitions live in `configs/cvat/`. |
| `data/` | Local/private wafer data, pattern assets, synthetic samples. Ignored by Git except `.gitkeep`. |
| `docs/` | User guides, architecture notes, roadmap, operator manuals. |
| `outputs/` | Local generated reports, manifests, predictions, model artifacts. Ignored by Git except `.gitkeep`. |
| `scripts/` | Stable command-line entrypoints and orchestration scripts. See [scripts command map](../scripts/README.md). |
| `src/wafermap/` | Reusable library code. New shared logic should live here, not inside scripts. |
| `tests/` | Regression tests for data contracts, scripts, docs, and model/data wiring. |

## Package Boundaries

| Package | Owns | Should not own |
|---|---|---|
| `wafermap.assets` | Pattern asset format, CVAT label schema helpers, asset scan/save/load helpers | Full pipeline orchestration |
| `wafermap.real` | Real/raw wafer manifest and PNG ingestion contracts | Model training |
| `wafermap.synth` | Synthetic wafer generation and procedural defect geometry | CVAT XML parsing |
| `wafermap.data` | Core sample schema and disk IO | UI behavior |
| `wafermap.training` | Segmentation, embedding, CPU encoder training code | Raw wafer parsing |
| `wafermap.reporting` | HTML/PNG/CSV report generation | Data mutation |
| `wafermap.features` | Feature extraction and vector utilities | Annotation import/export |
| `wafermap.evaluation` | Synthetic checks and nearest-neighbor diagnostics | Production data ingestion |

Rule of thumb: scripts orchestrate, `src/wafermap` implements reusable behavior.

## Main Functional Lanes

### 1. Real Wafer Ingestion

Entry scripts:

- `scripts/analyze_png_raw_folders.py`
- `scripts/extract_real_unlabeled_features.py`

Library owners:

- `src/wafermap/real/`
- `src/wafermap/data/`

Output:

```text
outputs/manifests/<run>_manifest.json
outputs/reports/<run>/
```

### 2. CVAT Annotation Bridge

Entry scripts:

- `scripts/export_cvat_wafer_images.py`
- `scripts/import_cvat_annotations.py`

Library owners:

- `src/wafermap/assets/cvat.py`
- `src/wafermap/assets/library.py`

Config:

- `configs/cvat/wafer_defect_labels.json`

Output:

```text
data/cvat_exports/<task>/
data/pattern_assets/<family>/<asset_id>/
```

### 3. Pattern Asset Library

Entry scripts:

- `scripts/build_pattern_asset_report.py`
- `scripts/run_pattern_asset_editor.py` for legacy fallback only

Library owner:

- `src/wafermap/assets/`

Asset contract:

```text
grade.png
mask.png
preview.png
metadata.json
```

### 4. Hybrid Synthetic Dataset

Entry scripts:

- `scripts/compose_synthetic_from_assets.py`
- `scripts/run_pattern_asset_pipeline.py`

Library owners:

- `src/wafermap/synth/`
- `src/wafermap/data/`
- `src/wafermap/reporting/`

Output:

```text
data/synthetic/<run>/<sample_id>/
outputs/<run>/asset_segmentation_manifest.csv
```

### 5. Model Readiness And Training

Entry scripts:

- `scripts/build_segmentation_readiness.py`
- `scripts/train_segmentation_smoke.py`
- `scripts/train_embedding_smoke.py`
- `scripts/train_unet_segmentation.py`

Library owners:

- `src/wafermap/training/`
- `src/wafermap/reporting/`

### 6. Review, Retrieval, And Legacy Diagnostics

These are useful but secondary to the CVAT dataset pipeline.

Entry scripts:

- `scripts/interpret_fbm.py`
- `scripts/evaluate_*`
- `scripts/summarize_*`
- `scripts/make_*`

Library owners:

- `src/wafermap/features/`
- `src/wafermap/evaluation/`
- `src/wafermap/reporting/`

## Where To Add New Code

| New need | Preferred location |
|---|---|
| New CVAT label | `configs/cvat/wafer_defect_labels.json` |
| New label parsing rule | `src/wafermap/assets/cvat.py` |
| New pattern asset metadata field | `src/wafermap/assets/library.py` and related tests |
| New procedural defect | `src/wafermap/synth/procedural_patterns.py` |
| New training tensor behavior | `src/wafermap/training/segmentation.py` |
| New report | `src/wafermap/reporting/` plus a thin script if needed |
| New command | `scripts/` with an entry in [scripts command map](../scripts/README.md) |
| New operator-facing process | `docs/operator_manual.md` |

## Known Technical Debt

- `scripts/` is intentionally flat for now because existing tests and user commands reference those filenames directly.
- `scripts/run_pattern_asset_editor.py` is large and legacy. Do not expand it as the main annotation product.
- Several historical `evaluate_*` scripts are retained as research diagnostics. They are not the primary workflow.
- Some orchestration scripts still dynamically load other scripts. New shared logic should move to `src/wafermap/` instead of deepening that pattern.

## Refactoring Rule

Do not move public command filenames unless wrapper scripts are kept. A user should be able to follow README commands without knowing internal package layout.
