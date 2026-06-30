# 문서 품질 검증표

이 문서는 WaferMap 문서들이 현재 프로젝트 목적에 맞는지 하나씩 검토한 결과입니다.
검토 기준은 아래 네 가지입니다.

1. 한국어로 읽히는가?
2. 작업자가 바로 따라 할 예시가 있는가?
3. “대표 불량 패턴 → 합성맵 → multi-label U-Net → correction loop” 목적과 맞는가?
4. 코드/파일 위치와 문서 설명이 어긋나지 않는가?

## 1. 전체 판정

현재 문서 세트는 아래 구조로 정리되어 있습니다.

```text
docs/index.html
-> 먼저 읽을 문서
-> 라벨 데이터 가이드
-> 학습 데이터 규격
-> 실행 흐름
-> 도구/데이터/검증 문서
-> 보조/이력 문서
```

사용자가 한 파일만 본다면 [문서 표지판](index.html)을 보면 됩니다.
브라우저에서 보기 좋은 HTML 문서는 `docs/pages/`에 생성됩니다.

## 2. 문서별 검증 결과

| 문서 | 현재 역할 | 검증 결과 | 보강된 예시 |
|---|---|---|---|
| `index.html` | 메인 표지판 | 한국어 목차와 구현 위치 지도 제공 | 불량별 라벨링 방식, 학습 데이터 구조 |
| `README.md` | repository 첫 설명 | 프로젝트 목적과 주요 command 안내 | segmentation family 표 |
| `docs/README.md` | 문서 목차 | 읽을 순서와 문서 관리 규칙 제공 | HTML 생성 명령 |
| `core_direction.md` | 프로젝트 경계 | classification이 아니라 segmentation dataset factory임을 명확화 | 좋은/나쁜 방향 예시 |
| `project_overview.md` | 제품 개요 | 전체 파이프라인과 완료 기준 정리 | 구성요소별 코드 위치 |
| `architecture.md` | 코드 구조 | 새 기능을 어디에 넣을지 설명 | 기능별 command와 패키지 지도 |
| `end_to_end_workflow.md` | 전체 실행 순서 | raw wafer부터 prediction correction까지 연결 | 단계별 PowerShell command |
| `operator_manual.md` | 작업자 실행 안내 | 실제 작업 순서와 troubleshooting 정리 | manifest, tool, readiness, U-Net 명령 |
| `segmentation_tool_workflow.md` | 도구 조작 기준 | bbox/mask 차이와 prediction prefill 설명 | shot_grid/ring/scratch parametric JSON |
| `label_data_guidelines.md` | 라벨 저장 방식 | wafer 1장 기준 labels/masks 구조 설명 | `labels.json`, `manual_mask`, `parametric_mask` |
| `training_data_contract.md` | 학습 데이터 규격 | `arrays.npz`, manifest, input/target channel 정의 | sample folder, metadata, CSV 예시 |
| `pattern_taxonomy.md` | defect family 정의 | family별 판단 기준과 애매한 경우 처리 | local/scratch/ring/edge/shot_grid 예시 |
| `fbm_data_flow_guide.md` | 데이터 경로 | raw, labels, assets, synthetic, model output 경로 연결 | folder tree, manifest JSON |
| `fbm_pattern_asset_pipeline.md` | 합성 파이프라인 | asset/rule을 학습 sample로 바꾸는 과정 설명 | asset metadata, 합성 명령 |
| `validation_protocol.md` | 검증 기준 | 데이터 계약, mask 품질, 모델 correction 검증 정리 | 테스트 명령과 통과 기준 |
| `semiconductor_ai_review.md` | AI 설계 판단 | 왜 segmentation/합성/좌표 channel을 쓰는지 설명 | input channel과 target clipping 예시 |
| `roadmap.md` | 단계 계획 | 현재/다음 phase와 완료 기준 정리 | phase별 명령과 산출물 |
| `data_schema.md` | 데이터 schema | 배열/manifest/metadata 필드 설명 | 좌표 기준과 sample 형식 |
| `glossary.md` | 용어 사전 | 변수와 용어 정의 제공 | feature, manifest, family 설명 |
| `real_png_operator_runbook.md` | 실제 PNG 입력 | raw PNG batch를 manifest/report로 만드는 절차 | geometry JSON, 실행 명령 |
| `real_unlabeled_workflow.md` | 라벨 없는 wafer 처리 | triage, asset 후보, prediction correction 연결 | manifest와 feature 추출 예시 |
| `real_wafer_review_checklist.md` | 실제 wafer 리뷰 | 리뷰어가 판단할 항목과 결과 형식 정리 | review CSV column |
| `expert_review_protocol.md` | 전문가 리뷰 | family 정의/asset 품질을 전문가가 검증하는 절차 | review CSV와 집계 명령 |
| `modeling_strategy.md` | 모델 이해 가이드 | U-Net 입력, target, sigmoid, loss, threshold, correction loop 설명 | input/target tensor, BCEWithLogitsLoss, prediction export |
| `experiment_history.md` | 실험 이력 | 왜 현재 방향으로 왔는지 기록 | resize-only, patch proposal, smoke test |
| `enterprise_readiness_assessment.md` | 운영 준비도 | 운영 투입 전 부족한 점 정리 | checklist |
| `legacy_pattern_asset_editor.md` | 호환성 안내 | 과거 command 이름을 왜 유지하는지 설명 | 제거 가능 조건 |
| `scripts/README.md` | 명령 지도 | script별 목적과 우선순위 정리 | 주력 pipeline command |

## 3. 목표 적합성 검증

문서 세트가 현재 목표와 맞는 이유:

- 모든 핵심 문서가 `pattern asset`, `synthetic dataset`, `multi-label segmentation`,
  `prediction correction` 흐름을 기준으로 설명합니다.
- `bbox_xywh`를 학습 정답으로 오해하지 않게 `mask`와 분리했습니다.
- `shot_grid`, `ring`, `edge`처럼 손마스킹이 비효율적인 유형은
  `parametric_mask`로 설명했습니다.
- 애매한 불량을 억지로 family에 넣지 않고 `mixed_unknown` 또는 review-only로
  분리하도록 문서화했습니다.
- 코드 위치와 command를 각 문서에 넣어 실제 구현과 연결했습니다.
- 모델 설명 문서에서 U-Net 구조, input/target tensor, sigmoid multi-label 예측,
  threshold 해석, correction loop까지 작업자 눈높이로 설명했습니다.

## 4. 문서 유지 규칙

문서를 수정한 뒤에는 아래 순서로 확인합니다.

```powershell
python scripts/build_static_docs.py
python -m pytest tests/test_documentation_quality.py -q --basetemp .pytest_tmp_docs
```

학습 데이터 계약까지 같이 확인하려면:

```powershell
python -m pytest tests/test_pattern_asset_pipeline.py tests/test_segmentation_training.py -q
```

## 5. 아직 남겨둘 영어

아래는 번역하지 않고 유지합니다.

- script 파일명: `train_unet_segmentation.py`
- schema key: `pattern_masks`, `bbox_xywh`, `metadata.json`
- family id: `local`, `scratch`, `ring`, `edge`, `shot_grid`, `random`
- model 이름: `U-Net`
- manifest/output path

이 값들은 코드와 데이터 계약의 일부라 한국어로 바꾸면 오히려 혼란이 커집니다.
