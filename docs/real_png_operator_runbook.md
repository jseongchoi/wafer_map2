# 실제 raw PNG 운영 안내서

이 문서는 실제 wafer PNG 폴더를 받았을 때 처음 실행하는 절차를 설명합니다.
목표는 raw PNG를 바로 모델에 넣는 것이 아니라, sample manifest와 검토 report를
만들어 segmentation asset 작업으로 연결하는 것입니다.

## 1. 입력 폴더

예시:

```text
data/raw/product_A/
  lot_001/
    WAFER_0001.png
    WAFER_0002.png
  lot_002/
    WAFER_0101.png
```

권장:

- 제품/공정/lot 단위가 경로에 드러나게 둡니다.
- 원본 파일명은 가능하면 바꾸지 않습니다.
- raw data는 git에 넣지 않습니다.

## 2. Geometry JSON

wafer center, radius, die size 같은 geometry가 있으면 JSON으로 둡니다.

```json
{
  "wafer_center_xy": [512, 512],
  "wafer_radius_px": 490,
  "die_pitch_xy": [12, 12],
  "notch_direction": "down"
}
```

geometry가 없으면 초기 분석은 가능하지만 `edge`, `ring`, `shot_grid` 같은
위치 기반 label 품질이 떨어질 수 있습니다.

## 3. manifest 생성

```powershell
python scripts/analyze_png_raw_folders.py `
  --input-root data/raw/product_A `
  --out-manifest outputs/manifests/real_png_batch_manifest.json `
  --out-report-dir outputs/reports/real_png_batch
```

geometry를 줄 수 있으면:

```powershell
python scripts/analyze_png_raw_folders.py `
  --input-root data/raw/product_A `
  --geometry-json configs/geometry/product_A.json `
  --out-manifest outputs/manifests/real_png_batch_manifest.json `
  --out-report-dir outputs/reports/real_png_batch
```

## 4. 생성 산출물

```text
outputs/manifests/real_png_batch_manifest.json
outputs/reports/real_png_batch/
  index.html
  sample_summary.csv
  thumbnails/
```

manifest는 이후 segmentation tool의 입력이 됩니다.

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id WAFER_0001 `
  --assets-root data/pattern_assets
```

## 5. 실행 후 확인 순서

1. report HTML을 열어 image가 정상적으로 보이는지 확인합니다.
2. sample 수가 예상과 맞는지 확인합니다.
3. wafer가 뒤집히거나 crop되지 않았는지 preview를 봅니다.
4. `actual_net_die=0` 같은 이상값이 있으면 geometry 또는 threshold를 의심합니다.
5. 대표 wafer 몇 장을 segmentation tool로 열어 asset 저장을 시작합니다.

## 6. 리뷰 작성

전문가가 wafer를 빠르게 triage할 때는 review CSV를 사용할 수 있습니다.

```powershell
python scripts/make_expert_review_template.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --out outputs/reviews/product_A_review_template.csv
```

리뷰어가 채우면:

```powershell
python scripts/summarize_expert_review.py `
  --review-csv outputs/reviews/product_A_review_template.csv `
  --out outputs/reviews/product_A_review_summary.json
```

## 7. 실패 시 판단

| 증상 | 원인 후보 | 조치 |
|---|---|---|
| PNG가 하나도 잡히지 않음 | 경로/확장자 문제 | `--input-root`와 파일 확장자 확인 |
| wafer가 검게만 보임 | gray scale 해석 문제 | raw image min/max 확인 |
| center/radius가 이상함 | geometry 누락/오류 | geometry JSON 추가 |
| sample id가 중복됨 | 파일명 충돌 | lot/path 정보를 sample id에 포함 |
| report는 되지만 tool이 못 엶 | manifest path 문제 | manifest의 `image_path` 존재 확인 |

## 8. 다음 단계

raw PNG 분석은 시작점일 뿐입니다.
이후 반드시 아래로 이어져야 합니다.

```text
manifest
-> segmentation tool
-> pattern asset 저장
-> asset report
-> synthetic dataset
-> readiness manifest
```
