# Documentation Index

WaferMap documentation is organized around one product path:

```text
FBM maps
-> defect generation
-> multi-defect synthetic maps
-> multi-defect segmentation training and validation
-> real-data pattern asset extraction
```

## Start Here

1. [Core Direction](core_direction.md): the non-negotiable project scope and decision rule.

2. [End-To-End Workflow](end_to_end_workflow.md): the canonical command flow from wafer input to assets, synthetic maps, U-Net training, prediction export, and correction.

3. [Operator Manual](operator_manual.md): a shorter runbook for people operating the pipeline.

4. [Semiconductor AI Data Science Review](semiconductor_ai_review.md): the technical rationale for resize policy, target clipping, location-aware composition, and readiness checks.

## Reference Docs

- [Architecture](architecture.md): package boundaries, command ownership, and where new code belongs.
- [Project Overview](project_overview.md): current product shape, compatibility files, and next implementation order.
- [Segmentation Tool Workflow](segmentation_tool_workflow.md): tool controls, mask editing, asset saving, and prediction prefill.
- [FBM Data Flow Guide](fbm_data_flow_guide.md): file locations, manifest outputs, asset paths, and model artifacts.
- [Hybrid Synthetic Data Pipeline](fbm_pattern_asset_pipeline.md): pattern asset format, procedural fallback, readiness, and training.
- [Pattern Taxonomy](pattern_taxonomy.md): defect family meanings.
- [Data Schema](data_schema.md): arrays, masks, metadata, and manifest schemas.
- [Roadmap](roadmap.md): phased implementation status.
- [Glossary](glossary.md): shared terms such as `severity`, `manifest`, `pattern asset`, and `retrieval_failure_mode`.
- [scripts command map](../scripts/README.md): stable command-line entrypoints.

## Real Data And Review

- [Real PNG Operator Runbook](real_png_operator_runbook.md)
- [Real Unlabeled Workflow](real_unlabeled_workflow.md)
- [Expert Review Protocol](expert_review_protocol.md)
- [Real Wafer Review Checklist](real_wafer_review_checklist.md)

## Historical Or Secondary Docs

- [Experiment History](experiment_history.md)
- [Modeling Strategy](modeling_strategy.md)
- [Validation Protocol](validation_protocol.md)
- [Enterprise Readiness Assessment](enterprise_readiness_assessment.md)
- [Compatibility Pattern Asset Editor](legacy_pattern_asset_editor.md)

## Documentation Rules

- Keep [End-To-End Workflow](end_to_end_workflow.md) as the single source for the main command sequence.
- Keep tool-specific details in [Segmentation Tool Workflow](segmentation_tool_workflow.md).
- Keep artifact and folder contracts in [FBM Data Flow Guide](fbm_data_flow_guide.md).
- Keep synthesis and training details in [Hybrid Synthetic Data Pipeline](fbm_pattern_asset_pipeline.md).
- Register new scripts in [scripts command map](../scripts/README.md).
