# Validation Protocol

This document defines how to validate the WaferMap pipeline and how to choose between multi-label segmentation, weak/unsupervised methods, and retrieval-style methods.

Canonical workflow: [End-To-End Workflow](end_to_end_workflow.md).

## Methodology Position

The primary modeling path is multi-label segmentation.

Why:

- The project needs pixel-level labels: where the defect is, not only whether a wafer is abnormal.
- One wafer can contain multiple defect families, so sigmoid multi-channel output is a better fit than single-class softmax.
- Synthetic composition already knows the oracle mask for each pasted defect. That label should be used directly.
- The local segmentation tool creates real-data pattern assets, so human correction naturally improves segmentation masks.

Unsupervised or self-supervised learning is useful, but not sufficient as the main method:

- it can rank unusual wafers for review;
- it can propose candidate regions for the segmentation tool;
- it can mine underrepresented real patterns for asset extraction;
- it can provide embeddings for similarity search;
- it usually cannot guarantee family-specific, pixel-accurate labels without human correction.

Therefore the recommended stack is:

```text
multi-label segmentation = primary training target
weak/unsupervised anomaly detection = candidate generation and triage
self-supervised/embedding retrieval = mining, grouping, and review prioritization
human correction = source of trusted real-data labels
```

## Input Validation

Supported inputs:

| Input | Use |
|---|---|
| `png_grayscale_raw` | Real grade 0-7 wafer PNG batch. |
| `npz_semantic_arrays` | Synthetic or processed arrays containing `severity`, masks, and metadata. |
| `real_unlabeled_manifest/v1` | Tool input manifest for real or real-like wafers. |

Checks:

- gray values map cleanly to grade 0-7 or known STBY values;
- wafer shape matches product geometry;
- `wafer_mask`, `valid_test_mask`, and `stby_mask` are mutually interpretable;
- sample ids are stable and unique;
- invalid or missing-test regions are not treated as positive defect targets.

## Pattern Asset Validation

Pattern assets must be family-specific and reusable.

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

Checks:

- `mask.png` is tight around the visible defect;
- family assignment matches `local`, `scratch`, `ring`, `edge`, `shot_grid`, or `random`;
- one physical ring is not accidentally split into unrelated assets;
- STBY-only regions are not labeled as physical defects;
- location summary is present for radial/angular/edge-distance-aware composition.

## Synthetic Dataset Validation

Synthetic samples must include image arrays, target masks, and metadata.

```text
data/synthetic/asset_composed/<sample_id>/
  arrays.npz
  metadata.json
```

Required arrays:

```text
severity
wafer_mask
valid_test_mask
stby_mask
pattern_masks
pattern_intensity
chip_index
```

Checks:

- `pattern_masks` has one channel per target family;
- target masks are clipped to `wafer_mask & valid_test_mask`;
- same-family overlap uses max composition;
- multi-family overlap is allowed and visible in readiness metrics;
- gallery samples look physically plausible on the wafer;
- procedural fallback does not dominate real-data asset families once enough assets exist.

## Segmentation Model Validation

Baseline target: coordinate-aware multi-label U-Net.

Output:

```text
family probability masks:
local, scratch, ring, edge, shot_grid, random
```

Primary metrics:

- per-family Dice/F1;
- per-family IoU;
- recall for small local defects;
- scratch continuity recall;
- ring/edge continuity and false-positive behavior;
- missed major defect rate after human review.

Training gates:

- train split has positive samples for every target family;
- validation split gaps are reported and not over-interpreted;
- prediction export uses `fbm_prediction_masks/v1`;
- exported predictions can be loaded by `run_segmentation_tool.py --prediction-json`.

## Unsupervised And Self-Supervised Validation

Use these methods as assistive models, not as the final label source.

Recommended uses:

| Method family | Use |
|---|---|
| Reconstruction/anomaly detection | Find unusual wafers or high-interest regions. |
| One-class anomaly detection | Flag out-of-distribution wafers when normal examples dominate. |
| Patch/feature anomaly localization | Propose candidate masks for tool review. |
| Self-supervised embeddings | Cluster real wafers, find near-duplicates, and mine rare patterns. |
| Retrieval | Prioritize samples similar to known high-quality assets. |

Validation metrics:

- review hit rate: fraction of proposed wafers/regions accepted by a human;
- missed major defect rate;
- false positive review burden;
- family discovery value: how often proposals create useful new assets;
- downstream improvement after adding corrected assets to segmentation training.

Failure modes:

- anomaly methods may highlight normal process variation;
- rare normal layouts can look anomalous;
- broad wafer-level anomaly scores do not provide family masks;
- reconstruction models can learn to reconstruct defects if trained on contaminated data;
- clustering groups similar wafers but does not prove defect family or mask boundaries.

## Human Review Loop

Review fields:

```text
query_sample_id
model_family
reviewer_family
position_correct
mask_quality
missed_major_defect
false_positive_family
comment
```

Decision loop:

1. Run model or anomaly proposals.
2. Load candidates in the segmentation tool.
3. Correct family and mask boundaries.
4. Save corrected pattern assets.
5. Recompose synthetic data.
6. Retrain or re-evaluate segmentation.
7. Track whether the correction loop improves downstream segmentation metrics.

## Test Commands

Fast default:

```powershell
python -m pytest -q --basetemp .pytest_tmp
```

Full slow validation:

```powershell
python -m pytest -q --run-slow --basetemp .pytest_tmp_full
```

Targeted validation:

```powershell
python -m pytest tests/test_pattern_asset_pipeline.py tests/test_segmentation_training.py tests/test_segmentation_readiness.py -q --basetemp .pytest_tmp_validation
```
