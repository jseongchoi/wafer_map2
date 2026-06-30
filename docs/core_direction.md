# Core Direction

This project stays centered on one workflow:

```text
FBM maps
-> defect generation
-> multi-defect synthetic maps
-> multi-defect segmentation training and validation
-> local segmentation tool for real-data pattern asset extraction
-> better synthetic data and correction loops
```

The executable version of this workflow is [End-To-End Workflow](end_to_end_workflow.md).

## Product Boundary

WaferMap is a wafer-specific segmentation dataset factory. The repository should make it faster to turn FBM maps into reusable defect masks, compose those masks into multi-defect synthetic wafers, and train or validate segmentation models against that data.

The local browser app is not a separate annotation product. It is the operator surface for extracting real-data pattern assets that feed the synthetic composition and segmentation pipeline.

## In Scope

- generating defect patterns from FBM maps;
- saving family-specific pattern assets from real or real-like wafers;
- composing multi-defect synthetic maps from pattern assets and procedural fallback;
- validating segmentation readiness, family coverage, and smoke training paths;
- loading model predictions or proposals back into the local segmentation tool for correction.

## Out Of Scope

- external annotation suites;
- generic image labeling workflows that do not preserve wafer-specific masks and metadata;
- non-segmentation platform features;
- standalone retrieval experiments unless they directly improve segmentation data quality.

## Decision Rule

When a change is ambiguous, keep it only if it makes this chain more reliable:

```text
FBM -> defects -> multi-defect synthetic data -> segmentation -> real-data asset feedback
```
