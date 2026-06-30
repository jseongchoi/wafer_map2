# Semiconductor AI Data Science Review

This review keeps WaferMap aligned to the primary workflow:

```text
FBM maps
-> defect generation
-> multi-defect synthetic maps
-> multi-defect segmentation training and validation
-> local segmentation tool for real-data pattern asset extraction
```

## Current Technical Decisions

### Methodology Choice

Multi-label segmentation is the primary method for this project because the target is a pixel-level family mask. Unsupervised and self-supervised methods should be used as assistive methods for candidate mining, anomaly triage, proposal generation, and embedding retrieval.

Do not replace the segmentation target with wafer-level anomaly scores. An anomaly score can say "review this wafer", but it does not reliably answer "which pixels are local, scratch, ring, edge, shot_grid, or random".

### Observed Targets Only

Primary segmentation targets must be limited to observed testable pixels:

```text
target_mask = pattern_mask & wafer_mask & valid_test_mask
```

STBY or missing-test regions can be tracked as context through `stby_mask`, but they should not become positive training pixels for defect families. A model cannot learn a reliable visual/severity cue for a hidden defect under an unobserved region.

### Input Severity Channels

The segmentation input keeps three severity-derived channels:

```text
severity_mean
severity_max
fail_density
```

`severity_mean` preserves average block behavior, `severity_max` preserves high-grade spikes, and `fail_density` preserves how many die/pixels are failing. The former binary high-grade channel was intentionally removed because it is derivable from `severity_max`.

### Label Resize Policy

Target masks use max pooling after valid-test clipping. This preserves small local and scratch defects when moving to lower resolution. Severity inputs use mean/max/density channels instead of a single max-pooled severity image, because max-only inputs can overstate a single high-grade pixel across the entire pooled cell.

### Asset Composition

Real-data pattern assets are composed with max severity and max pattern intensity rules. Metadata records both requested placement intent and actual placement behavior. If a requested location-aware placement cannot fit on valid pixels and falls back to random valid placement, that fallback must be visible in metadata.

### Location Prior

Asset metadata stores radial/angular summaries. `polar_jitter` should be preferred when the process location is part of the defect signature, while `random_valid` is reserved for stress-testing shape transfer.

## Checks Added By This Review

- Asset masks are clipped to `valid_test_mask` before saving.
- U-Net target tensors exclude invalid/STBY pixels before max pooling.
- Readiness class balance and overlap metrics are computed on valid-test target pixels.
- Readiness keeps hidden scratch/STBY overlap as a separate risk metric.
- Overlapping same-family asset intensity follows max composition.
- Placement metadata distinguishes requested mode from actual fallback mode.
- U-Net training checks train-split positive coverage for every target family before running, and reports validation coverage gaps so class metrics are not over-interpreted.

## Remaining Expert Review Questions

- Do large wafer-wide patterns need a separate global family or should they be decomposed into edge/ring/shot-grid/local masks?
- Should real operator masks be allowed to include grade-0 interior gaps for shape continuity, or should assets be strictly positive-grade pixels only?
- What minimum positive sample count per family should be required before production U-Net training starts?
- Should `polar_jitter` use tighter radial/angular tolerance gates once enough real assets exist?
