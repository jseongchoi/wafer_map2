# WaferMap Defect Intelligence

보안상 실제 wafer Fail Bit Map을 repo에 둘 수 없는 조건에서, synthetic FBM을 만들고 검증하며 FBM 자체에서 정보를 추출하는 프로젝트입니다.

현재 1차 목표는 ANOVA가 아닙니다. ANOVA와 공정/설비/lot/recipe 기반 통계 검정은 나중에 process metadata와 FBM feature를 조인한 뒤 수행할 후속 분석입니다.

```text
현재 우선순위:
FBM feature 추출 -> 유사맵 검색 -> defect score -> scale 검증 -> real-unlabeled workflow
```

## 현재 목표

- 1채널 고해상도 Fail Bit Map에서 wafer-level observable feature를 추출한다.
- 비슷한 불량 패턴을 가진 wafer를 nearest-neighbor 방식으로 찾는다.
- hard label 하나가 아니라 edge, ring, scratch, local, shot-relative, stby 같은 여러 관점의 score를 만든다.
- coarse group, 유사맵 검색, feature family ablation으로 방법론이 유효한지 CPU 환경에서 먼저 검증한다.
- 이후 synthetic mask를 이용해 segmentation baseline을 만들고, 보안 환경 안에서 real unlabeled wafer sanity check로 확장한다.
- 실제 wafer inference에 쓰는 feature는 observable-only로 고정하고, `*_mask_ratio` 같은 synthetic oracle field는 검증 전용으로 분리한다.

## 현재 범위

- 600 net die급 synthetic wafer map 생성
- Grade 0~7, none wafer, valid test mask, stby fail chip 분리
- 계단식 wafer edge, center-to-edge fail gradient, edge-chip outer-face gradient
- spin arc/radial scratch, ring, edge, local blob, shot-relative, stby-origin-coupled defect 생성
- observable feature 기반 유사 wafer 검색, grouping, stability, parameter sweep, feature family ablation
- 위치-aware retrieval용 chip-level polar feature는 CSV에 보관하되 global grouping에는 조건부로 제외
- 전문가 육안 검증용 HTML report와 gallery 생성

## 주요 리포트

- [Project Progress Briefing](outputs/reports/project_progress_briefing.html)
- [Methodology Validation](outputs/reports/methodology_validation_report.html)
- [FBM Grouping](outputs/reports/fbm_grouping_report.html)
- [Grouping Stability](outputs/reports/fbm_grouping_stability_report.html)
- [Parameter Sweep](outputs/reports/fbm_grouping_parameter_sweep_report.html)
- [Feature Ablation](outputs/reports/fbm_feature_ablation_report.html)
- [Scale FBM Grouping](outputs/reports/fbm_grouping_scale_report.html)
- [Scale Grouping Stability](outputs/reports/fbm_grouping_scale_stability_report.html)
- [Scale Parameter Sweep](outputs/reports/fbm_grouping_scale_parameter_sweep_report.html)
- [Scale Feature Ablation](outputs/reports/fbm_feature_ablation_scale_report.html)
- [Scale Retrieval Confidence](outputs/reports/fbm_retrieval_confidence_scale_report.html)
- [Interest-Based Retrieval](outputs/reports/fbm_interest_retrieval_scale_report.html)
- [Scale Defect Feature Retrieval](outputs/reports/fbm_defect_feature_retrieval_scale_report.html)
- [Holdout Retrieval Confidence](outputs/reports/fbm_retrieval_confidence_holdout_report.html)
- [Holdout Interest Retrieval](outputs/reports/fbm_interest_retrieval_holdout_report.html)
- [Holdout Defect Feature Retrieval](outputs/reports/fbm_defect_feature_retrieval_holdout_report.html)
- [Segmentation Readiness](outputs/reports/fbm_segmentation_readiness_report.html)
- [Holdout Segmentation Readiness](outputs/reports/fbm_segmentation_readiness_holdout_report.html)
- [Segmentation Smoke Training](outputs/reports/fbm_segmentation_smoke_report.html)
- [Holdout Segmentation Smoke Training](outputs/reports/fbm_segmentation_smoke_holdout_report.html)
- [Defect Feature Summary](outputs/reports/fbm_defect_location_summary_report.html)
- [Holdout Defect Feature Summary](outputs/reports/fbm_defect_location_summary_holdout_report.html)
- [Scale Resize Benchmark](outputs/reports/fbm_resize_benchmark_scale_report.html)
- [Holdout Resize Benchmark](outputs/reports/fbm_resize_benchmark_holdout_report.html)
- [Scale Patch Proposal](outputs/reports/fbm_patch_proposal_scale_report.html)
- [Holdout Patch Proposal](outputs/reports/fbm_patch_proposal_holdout_report.html)
- [Scale Curve Proposal](outputs/reports/fbm_curve_proposal_scale_report.html)
- [Holdout Curve Proposal](outputs/reports/fbm_curve_proposal_holdout_report.html)
- [Expert Review Template](outputs/reports/expert_review_template.html)
- [Expert Review Summary](outputs/reports/expert_review_summary.html)

## 주요 문서

- [Problem Definition](docs/problem_definition.md)
- [Data Schema](docs/data_schema.md)
- [Pattern Taxonomy](docs/pattern_taxonomy.md)
- [Synthetic Data Plan](docs/synthetic_data_plan.md)
- [Validation Protocol](docs/validation_protocol.md)
- [Modeling Strategy](docs/modeling_strategy.md)
- [Methodology Validation](docs/methodology_validation.md)
- [Milestone Review](docs/milestone_review.md)
- [AI Data Science Validation Audit](docs/ai_ds_validation_audit.md)
- [Scale Pilot Review](docs/scale_pilot_review.md)
- [Next Tasks](docs/next_tasks.md)
- [Real-Unlabeled Workflow](docs/real_unlabeled_workflow.md)
- [Expert Review Protocol](docs/expert_review_protocol.md)
- [Interest-Based Retrieval](docs/interest_based_retrieval.md)
- [Segmentation Readiness](docs/segmentation_readiness.md)
- [Defect Feature Summary](docs/defect_location_summary.md)
- [Resize Strategy](docs/resize_strategy.md)
- [Current Milestone Checkpoint](docs/current_milestone_checkpoint.md)
- [Solution Roadmap Checkpoint](docs/solution_roadmap_checkpoint.md)
- [Curve Proposal Strategy](docs/curve_proposal_strategy.md)
- [Robustness Holdout](docs/robustness_holdout.md)
- [Revised Physical Assumptions](docs/revised_physical_assumptions.md)
- [Review Presets](docs/review_presets.md)
- [Roadmap](docs/roadmap.md)
- [Project Structure](docs/project_structure.md)

## 빠른 실행

```powershell
python -m pytest -q
python scripts/generate_synthetic.py --config configs/synth/debug.json --out data/synthetic/debug --count 3
python scripts/validate_synthetic.py --data data/synthetic/debug
python scripts/extract_features.py --data data/synthetic/debug --out outputs/reports/synthetic_features.csv
```

생성된 preview는 `data/synthetic/debug/synth_*/preview.png`에서 확인합니다. synthetic data와 report output은 git에 포함하지 않는 재생성 가능한 산출물입니다.
