# Legacy Pattern Asset Editor

`scripts/run_pattern_asset_editor.py`는 CVAT 도입 전에 만든 로컬 브라우저 기반 pattern asset editor입니다. 현재 메인 annotation surface는 CVAT이며, 이 editor는 다음 경우에만 사용합니다.

- CVAT import/export 경로를 검증하기 전 빠른 아이디어를 시험할 때
- CVAT가 특정 wafer UX를 처리하기 어려운지 비교할 때
- 모델 proposal, smart fit, lasso interaction 같은 custom interaction을 reference로 보존할 때
- emergency fallback으로 단일 wafer에서 pattern asset을 직접 저장해야 할 때

## 실행

```powershell
python scripts/run_pattern_asset_editor.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root data/pattern_assets
```

## 현재 역할

이 editor가 저장하는 asset 구조는 CVAT importer가 저장하는 구조와 같습니다.

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

따라서 legacy editor에서 만든 asset도 synthetic composer가 그대로 사용할 수 있습니다.

## 더 이상 확장하지 않는 범위

- CVAT와 중복되는 일반 polygon/brush annotation UI
- label 관리 UI
- 대량 wafer task 관리
- annotator assignment, review, audit trail

이 기능들은 CVAT가 더 잘 처리합니다. 이 repository에서는 CVAT package export/import와 pattern asset conversion 품질에 집중합니다.

## 보존할 reference 기능

- lasso selection preview
- smart fit / trace scratch line
- model proposal overlay
- low-resolution interaction path
- one-family asset and split-components save mode

이 기능들은 나중에 CVAT plugin, model-assisted annotation, 또는 별도 lightweight review tool을 만들 때 참고할 수 있습니다.
