# Documentation Index

이 문서 폴더의 현재 기준은 in-repo segmentation tool 기반 defect dataset workflow입니다. 오래된 feature/retrieval 실험 문서는 남아 있지만, 지금 의사결정의 중심은 다음 경로입니다.

```text
real/unlabeled wafer manifest
-> local segmentation tool
-> pattern asset library
-> hybrid synthetic data
-> segmentation readiness / smoke validation
```

## 먼저 읽을 문서

0. [Core Direction](core_direction.md)
   - FBM map, defect generation, multi-defect synthetic map, segmentation, real-data pattern asset extraction이라는 기준선을 고정합니다.

1. [Architecture](architecture.md)
   - repository folder, package boundary, command group, technical debt를 한 장으로 정리합니다.

2. [Operator Manual](operator_manual.md)
   - 실제 운영자가 manifest 생성, 직접 mask 작성, QC, synthetic dataset, training까지 따라 하는 절차서입니다.

3. [Segmentation Tool Workflow](segmentation_tool_workflow.md)
   - 직접 세그멘테이션 툴 실행, mask 저장, pattern asset 검증 방법을 설명합니다.

4. [Project Overview](project_overview.md)
   - 현재 프로젝트 목표, 남길 기능, 호환용 진입점, 다음 구현 순서를 정리합니다.

5. [FBM Data Flow Guide](fbm_data_flow_guide.md)
   - raw wafer, manifest, pattern assets, synthetic samples, segmentation manifest가 어디에 생기는지 정리합니다.

6. [Hybrid Synthetic Data Pipeline](fbm_pattern_asset_pipeline.md)
   - pattern asset과 procedural fallback을 합쳐 학습용 multi-label synthetic data를 만드는 방법을 설명합니다.

7. [Roadmap](roadmap.md)
   - segmentation tool, synthetic composition, model training, active learning을 단계별로 정리합니다.

8. [Semiconductor AI Data Science Review](semiconductor_ai_review.md)
   - Reviews target clipping, severity input channels, placement metadata, and dataset readiness from the wafer AI perspective.

9. [Pattern Taxonomy](pattern_taxonomy.md)
   - `local`, `scratch`, `ring`, `edge`, `shot_grid`, `random`, `stby_pattern`의 의미를 정리합니다.

10. [Glossary](glossary.md)
   - `severity`, `manifest`, `pattern asset`, `retrieval_failure_mode` 같은 용어를 정리합니다.

## 운영 문서

- [Real PNG Operator Runbook](real_png_operator_runbook.md)
- [Real Unlabeled Workflow](real_unlabeled_workflow.md)
- [Expert Review Protocol](expert_review_protocol.md)
- [Real Wafer Review Checklist](real_wafer_review_checklist.md)

## 보조/이력 문서

- [Experiment History](experiment_history.md)
- [Modeling Strategy](modeling_strategy.md)
- [Validation Protocol](validation_protocol.md)
- [Enterprise Readiness Assessment](enterprise_readiness_assessment.md)
- [Compatibility Pattern Asset Editor](legacy_pattern_asset_editor.md)

## 문서 사용 기준

- 새 annotation 기능은 [Segmentation Tool Workflow](segmentation_tool_workflow.md)에 먼저 연결합니다.
- `scripts/run_segmentation_tool.py`를 operator-facing 진입점으로 다룹니다.
- 새 command를 추가하면 [scripts command map](../scripts/README.md)에 등록합니다.
- 모델 학습 문서는 데이터셋 생성 경로가 안정화된 뒤 업데이트합니다.
- 생성된 HTML/JSON/PNG 결과는 기본적으로 `outputs/` 아래에 두고 Git에는 넣지 않습니다.
