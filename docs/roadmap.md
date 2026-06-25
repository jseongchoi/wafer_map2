# Roadmap

현재 로드맵은 CVAT-first dataset pipeline을 기준으로 합니다. 모델은 중요하지만, 지금 병목은 학습 코드가 아니라 믿을 수 있는 defect mask 데이터셋입니다.

## Phase 0. Scope Lock

상태: 완료

- 입력은 wafer grade 0-7 map입니다.
- 1차 목표는 defect segmentation dataset 제작입니다.
- label은 CVAT schema에서 확장합니다.
- model training/evaluation은 dataset path 안정화 뒤 진행합니다.
- legacy local editor는 fallback/reference로만 유지합니다.

참조 문서:

- [Project Overview](project_overview.md)
- [CVAT Wafer Defect Annotation Workflow](cvat_wafer_annotation_workflow.md)

## Phase 1. CVAT Annotation Workflow

상태: 구현됨, 실제 wafer 운영 검증 필요

구현됨:

- `scripts/export_cvat_wafer_images.py`
- `configs/cvat/wafer_defect_labels.json`
- `scripts/import_cvat_annotations.py`
- CVAT polygon/box import
- `stby_blob` label과 grade override
- label alias mapping

다음 보강:

- 실제 wafer batch에서 annotator가 label을 헷갈리지 않는지 확인합니다.
- CVAT brush/mask가 주 workflow가 되면 native mask RLE import를 추가합니다.
- 제품/공정별 label set이 커질 때 schema 운영 규칙을 추가합니다.

## Phase 2. Pattern Asset Library

상태: 구현됨, 품질 축적 필요

구현됨:

- `grade.png`, `mask.png`, `preview.png`, `metadata.json` asset format
- CVAT annotation source metadata
- asset scan/report
- legacy Pattern Asset Builder fallback

다음 보강:

- `local`, `scratch`, `ring`, `edge` 실제 asset을 충분히 모읍니다.
- edge band와 ring처럼 전역적으로 큰 pattern의 라벨링 규칙을 문서화합니다.
- asset quality report에서 mask leakage, wrong family, split/merge issue를 바로 보이게 합니다.

## Phase 3. Hybrid Synthetic Data

상태: 구현됨

구현됨:

- `compose_synthetic_from_assets.py`
- `source_jitter` placement
- procedural fallback for `scratch`, `edge`, `shot_grid`, `random`
- multi-label `pattern_masks`

다음 보강:

- 실제 wafer feedback으로 edge sector, ring continuity, shot-grid realism을 조정합니다.
- product별 shot layout 정보가 있으면 `shot_grid` generator를 보정합니다.
- STBY/missing-test blob을 별도 model channel로 둘지 결정합니다.

## Phase 4. Readiness And Smoke Validation

상태: 구현됨

구현됨:

- `asset_segmentation_manifest.csv`
- segmentation readiness report
- Segmentation Smoke Test
- embedding smoke diagnostic
- full test suite baseline: `111 passed`

다음 보강:

- synthetic dataset versioning을 추가합니다.
- family별 최소 positive sample 기준을 운영 config로 뺍니다.
- CVAT import report와 readiness report를 더 직접 연결합니다.

## Phase 5. Small U-Net Training

상태: entrypoint 구현됨, 실제 학습은 PyTorch 환경 필요

구현 파일:

```text
scripts/train_unet_segmentation.py
```

학습 명령:

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/cvat_pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/cvat_pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/cvat_pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

완료 기준:

- family별 validation IoU/recall 계산
- scratch recall, local small-blob recall, ring continuity 확인
- edge/ring 같은 global pattern에서 failure case 수집
- model output을 `fbm_prediction_masks/v1`로 export

## Phase 6. Active Learning

상태: 설계 단계

목표:

이 단계는 model-assisted active learning loop를 만드는 단계입니다.

```text
model prediction
-> CVAT pre-annotation or review package
-> human correction
-> updated pattern assets / training labels
-> retraining
```

필요 작업:

- prediction mask export schema 정리
- CVAT pre-annotation import 방식 조사
- correction history를 asset metadata와 연결
- repeated wafer review loop 운영

## Phase 7. Retrieval And Similarity Search

상태: smoke diagnostic만 유지

embedding retrieval은 최종 제품에는 유용하지만 지금 1순위는 아닙니다. segmentation model이 안정화된 뒤 encoder embedding을 저장하고, cosine/FAISS 기반 similar wafer search로 확장합니다.

## Current Priority

1. 실제 wafer manifest를 CVAT package로 뽑아 annotation workflow를 검증합니다.
2. label schema를 운영 가능한 형태로 다듬습니다.
3. CVAT import 결과 asset 품질을 review합니다.
4. 합성 dataset report를 보고 edge/ring/global pattern realism을 보정합니다.
5. PyTorch 환경에서 small U-Net 학습을 시작합니다.
