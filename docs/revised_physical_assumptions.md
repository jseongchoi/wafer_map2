# Revised Physical Assumptions

This document captures expert feedback that overrides earlier naive synthetic assumptions.

## Grade Semantics

- Grade 0 is valid inside the wafer.
- Grade 0 means fail bit count is exactly zero.
- Grade 1~7 are quantized fail bit count buckets.
- None-wafer and in-wafer Grade 0 can both look black in raw grayscale.
- `wafer_mask` is required to distinguish none-wafer from valid in-wafer Grade 0.

## Edge Behavior

- Wafer edge can naturally have higher fail bit density than wafer center.
- Synthetic background should not be spatially uniform.
- Edge regions should receive higher fail event probability and/or higher severity based on center-distance polar radius.
- The polar radius is normalized as `r=0` at wafer center and `r=1` at the farthest in-wafer cell.
- Within an edge chip, cells closer to the wafer edge should have a higher fail probability than cells on the center-facing side.
- This creates a cell-level edge-facing gradient inside the same chip, not only a chip-level edge effect.
- Localized edge defects are separate from this baseline edge lift.

## Scratch Behavior

Straight line scratches are not the main target pattern.

The generator should prioritize:

- `spin_arc`: circular or partial circular scratches caused by spin/rotation-based process steps.
- `radial`: scratches spreading from wafer center toward edge.

Straight chord-like scratches may exist in rare cases, but should not dominate debug or pilot synthetic sets.

## WM811K-Style Pattern Scope

The synthetic set should prioritize defect families that are plausible in wafer map pattern analysis:

- scratch
- ring/donut
- edge-local or edge-ring
- local/blob
- random
- shot/reticle grid
- near-full or center-like patterns in later phases

Flow-like water streak defects are excluded from the current generator because they are not representative for the target use case.

## Shot/Reticle Grid Behavior

Photo tools can expose multiple chips in one shot field.

The synthetic generator should support shot-level repeated structure:

- A shot field can be represented as a small chip grid such as `3x3`, `3x2`, or `2x3`.
- A weak shot-relative region can repeat at the same location across many shot fields.
- Common synthetic anchors include lower-left shot region, bottom/left shot edge, and subtle shot-edge bands.
- Repeated regions should not be perfectly clean; real-looking synthetic maps should include partial shot dropout and mild intensity jitter.
- This pattern is different from random fail, because cells share the same relative coordinate inside each shot grid.
- This pattern is also different from wafer-level edge/ring/scratch because its coordinate system is photo-shot grid position, not wafer polar position.

## Local Blob Variants

The `local` class should cover small droplet-like fail clusters:

- `single_blob`: one isolated small droplet.
- `double_blob`: two nearby droplets.
- `triple_triangle`: three small droplets whose centers form a triangular arrangement.

## Origin-Hidden Stby Coupling

Stby Fail Chip is not only random missing-test noise.

In some real-looking cases, the physical origin of a defect can become unmeasurable:

- A scratch may start at a chip that becomes Stby, hiding the first visible fail-bit evidence.
- A local impact or meteor-like drop point may become Stby, hiding the highest-severity core.
- The visible pattern can therefore look like a defect whose center or starting point is missing.

The synthetic generator should support this by first assigning Stby chips to defect-origin seed points, then filling the remaining Stby chips from the existing latent-severity-weighted random process.

## Validation Checks

For each synthetic batch:

- Grade 0 ratio inside wafer should be reported.
- Center and edge severity/fail density should be compared.
- Center and edge comparisons should use center-distance polar radius.
- Edge-chip validation should compare the outer third of edge chips against the inner third.
- Scratch metadata should report `mode=spin_arc` or `mode=radial`.
- Visual review should check that circular/radial scratch modes look plausible.
- Local metadata should report `mode=single_blob`, `mode=double_blob`, or `mode=triple_triangle`.
- Shot metadata should report shot layout, anchor region, affected slots, and touched shot/chip count.
- Stby metadata should report whether origin-coupled missing-test chips were generated.
