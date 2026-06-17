# Synthetic Review Presets

This project uses small review presets to keep the expert feedback loop fast.

## Presets

### `review_balanced`

Default expert-review setting.

- Keeps local blob, random background, stby, edge, scratch, ring, and shot-grid mixed.
- Use this when checking whether the full wafer map still looks plausible overall.

### `review_edge_heavy`

Edge-stress review setting.

- Raises the chance of localized edge defects.
- Slightly lowers grade thresholds to make edge baseline and edge-chip outer-face gradients easier to inspect.
- Use this when validating whether the wafer edge and edge chip interiors look realistic.

### `review_shot_relative`

Photo shot / reticle-coordinate review setting.

- Raises `shot_grid` frequency.
- Keeps shot-grid anchored to shot-relative regions such as lower-left or shot-edge bands.
- Use this when checking whether repeated shot-relative fail-bit increase looks subtle rather than like a hard grid overlay.

## Recommended Review Loop

1. Generate 9 to 20 samples from one preset.
2. Open the generated HTML report and gallery.
3. Decide which pattern is too strong, too weak, too geometric, or missing.
4. Adjust only one or two parameters.
5. Regenerate and compare against the prior report.

The current goal is not to create final training data yet. The current goal is to calibrate the synthetic generator until expert review says it is close enough to real Fail Bit Map behavior for EDA and baseline modeling.
