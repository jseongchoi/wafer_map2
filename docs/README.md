# 문서 길잡이

이 문서 폴더의 현재 기준 문서는 **FBM Pattern Asset 기반 Hybrid Synthetic Data 파이프라인**입니다. 예전 feature/retrieval 문서는 보조 진단과 과거 실험 기록으로 남아 있지만, 지금 프로젝트의 주 경로는 pixel mask 기반 딥러닝 데이터 파이프라인입니다.

## 먼저 읽을 문서

1. [FBM 데이터 흐름 운영 가이드](fbm_data_flow_guide.md)
   - 실제 FBM을 어디에 두고, 누끼 asset과 합성 label 데이터가 어디에 생기는지 설명합니다.

2. [FBM Pattern Asset 기반 Hybrid Synthetic Data 파이프라인](fbm_pattern_asset_pipeline.md)
   - 현재 프로젝트의 본질, human asset과 procedural generator의 역할, U-Net 학습 경로를 설명합니다.

3. [프로젝트 개요](project_overview.md)
   - 지금 무엇이 구현됐고, 무엇이 아직 막혀 있는지 한 장으로 정리합니다.

4. [로드맵](roadmap.md)
   - 앞으로 어떤 순서로 완성도를 올릴지 단계별로 정리합니다.

5. [불량 패턴 정리](pattern_taxonomy.md)
   - `local`, `scratch`, `ring`, `edge`, `shot_grid`, `random`, `stby_pattern`의 의미를 정리합니다.

6. [용어와 변수 설명](glossary.md)
   - `severity`, `wafer_mask`, `manifest`, `top-k`, `retrieval_failure_mode`처럼 처음 보는 용어를 설명합니다.

7. [실제 raw PNG 운영 안내서](real_png_operator_runbook.md)
   - 실제 wafer raw PNG 폴더를 안전하게 읽고 결과를 만드는 운영 절차입니다.

8. [전문가 리뷰 절차](expert_review_protocol.md)
   - 모델/검색 결과를 사람이 어떻게 리뷰하고 다음 개선으로 연결할지 설명합니다.

## 현재 주 경로

```text
실제 wafer map
-> Pattern Asset Builder에서 local/scratch/ring 누끼 저장
-> scratch cold-start, edge, shot_grid, random은 procedural generator로 생성
-> Hybrid Synthetic Composer가 라벨 있는 synthetic wafer 생성
-> Segmentation readiness / smoke / embedding smoke 검증
-> PyTorch small U-Net 학습
-> 모델 prediction을 에디터로 되돌려 active learning loop 구성
-> encoder embedding으로 유사 wafer top-k 검색
```

## 문서 사용 기준

- 현재 의사결정은 `fbm_pattern_asset_pipeline.md`와 `project_overview.md`를 기준으로 합니다.
- `experiment_history.md`, `modeling_strategy.md`, `validation_protocol.md`는 과거 실험 판단과 보조 근거입니다.
- 실제 raw wafer 운영은 `real_png_operator_runbook.md`와 `real_unlabeled_workflow.md`를 따릅니다.
- 생성된 HTML/JSON 결과는 `outputs/` 아래에 있으며, 재생성 가능한 산출물입니다.
