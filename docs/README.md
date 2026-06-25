# Documentation Index

이 문서 폴더의 현재 기준은 CVAT-first defect dataset workflow입니다. 오래된 feature/retrieval 실험 문서는 남아 있지만, 지금 의사결정의 중심은 다음 경로입니다.

```text
real/unlabeled wafer manifest
-> CVAT image package
-> CVAT annotation export
-> pattern asset library
-> hybrid synthetic data
-> segmentation readiness / smoke validation
```

## 먼저 읽을 문서

1. [Architecture](architecture.md)
   - repository folder, package boundary, command group, technical debt를 한 장으로 정리합니다.

2. [Operator Manual](operator_manual.md)
   - 실제 운영자가 manifest 생성, CVAT annotation, import, QC, synthetic dataset, training까지 따라 하는 절차서입니다.

3. [CVAT Wafer Defect Annotation Workflow](cvat_wafer_annotation_workflow.md)
   - CVAT export/import 명령, label schema, STBY blob 처리, 검증 방법을 설명합니다.

4. [Project Overview](project_overview.md)
   - 현재 프로젝트 목표, 남길 기능, legacy로 분리할 기능, 다음 구현 순서를 정리합니다.

5. [FBM Data Flow Guide](fbm_data_flow_guide.md)
   - raw wafer, manifest, CVAT package, pattern assets, synthetic samples, segmentation manifest가 어디에 생기는지 정리합니다.

6. [Hybrid Synthetic Data Pipeline](fbm_pattern_asset_pipeline.md)
   - pattern asset과 procedural fallback을 합쳐 학습용 multi-label synthetic data를 만드는 방법을 설명합니다.

7. [Roadmap](roadmap.md)
   - CVAT annotation, synthetic composition, model training, active learning을 단계별로 정리합니다.

8. [Pattern Taxonomy](pattern_taxonomy.md)
   - `local`, `scratch`, `ring`, `edge`, `shot_grid`, `random`, `stby_blob`/`stby_pattern`의 의미를 정리합니다.

9. [Glossary](glossary.md)
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
- [Legacy Pattern Asset Editor](legacy_pattern_asset_editor.md)

## 문서 사용 기준

- 새 annotation 기능은 [CVAT workflow](cvat_wafer_annotation_workflow.md)에 먼저 연결합니다.
- `scripts/run_pattern_asset_editor.py`는 legacy fallback/reference로만 다룹니다.
- 새 command를 추가하면 [scripts command map](../scripts/README.md)에 등록합니다.
- 모델 학습 문서는 데이터셋 생성 경로가 안정화된 뒤 업데이트합니다.
- 생성된 HTML/JSON/PNG 결과는 기본적으로 `outputs/` 아래에 두고 Git에는 넣지 않습니다.
