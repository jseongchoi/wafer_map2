# 호환성 Pattern Asset Editor

이 문서는 `run_pattern_asset_editor.py` 파일명이 왜 남아 있는지 설명합니다.

## 1. 현재 방향

현재 제품 흐름의 중심 command는 아래입니다.

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --sample-id WAFER_0001 `
  --assets-root data/pattern_assets
```

`run_pattern_asset_editor.py`는 과거 이름을 쓰던 workflow와 문서를 깨지 않기 위해
남겨 둔 호환 entrypoint입니다.

## 2. 유지하는 이유

- 기존 테스트나 사용자가 과거 파일명을 호출할 수 있습니다.
- 급하게 삭제하면 문서 링크와 자동화가 깨질 수 있습니다.
- 실제 내부 구현을 공유하면 유지 비용이 크지 않습니다.

## 3. 새 문서에서 쓸 이름

새 문서와 작업자 안내에서는 `run_segmentation_tool.py`를 우선 사용합니다.

| 상황 | 권장 command |
|---|---|
| 새 작업자 안내 | `run_segmentation_tool.py` |
| 과거 script 호환 | `run_pattern_asset_editor.py` |
| 문서 예시 | `run_segmentation_tool.py` |

## 4. 제거 가능 조건

나중에 아래 조건이 만족되면 deprecated 처리할 수 있습니다.

- tests에서 과거 command 의존성이 사라짐
- docs에서 과거 이름이 호환 설명 외에는 사라짐
- 사용자 자동화에서 더 이상 호출하지 않음

그 전까지는 wrapper로 유지합니다.
