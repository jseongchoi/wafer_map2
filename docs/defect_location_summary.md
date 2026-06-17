# Defect Feature Summary

## 목적

이 문서는 segmentation mask가 주어졌을 때 FBM downstream task에 사용할 수 있는 구조화 feature를 추출할 수 있는지 검증한 결과다.

원래 목표:

```text
입력 FBM 한 장에서
유사맵 검색,
관심 불량별 retrieval,
defect family별 score,
clock/radial 위치 feature,
stby-hidden 정도,
downstream 통계/검색용 feature table을 만든다.
```

이번 단계는 synthetic oracle mask를 사용한 upper-bound 후처리 검증이다. 실제 운영에서는 model-predicted mask가 같은 후처리 입력이 된다.

## 산출물

Scale pilot:

- `outputs/reports/fbm_defect_location_summary_report.html`
- `outputs/reports/fbm_defect_location_summary.csv`
- `outputs/reports/fbm_defect_location_summary_metrics.json`
- `outputs/figures/fbm_defect_location_summary_gallery.png`

Holdout stress:

- `outputs/reports/fbm_defect_location_summary_holdout_report.html`
- `outputs/reports/fbm_defect_location_summary_holdout.csv`
- `outputs/reports/fbm_defect_location_summary_holdout_metrics.json`
- `outputs/figures/fbm_defect_location_summary_holdout_gallery.png`

## 출력 Schema

후처리 결과는 class별 region feature row를 만든다.

```text
sample_id
class_name
feature_key
pixel_ratio
centroid_clock
location_label
radial_zone
top_clock_positions
top_sector_share
stby_overlap_ratio
```

예시:

```text
sample_id=synth_000000
class_name=scratch
feature_key=scratch__0300__center
pixel_ratio=0.0009
location_label=03:00
radial_zone=center
stby_overlap_ratio=0.064
```

## 현재 결론

Mask가 주어지면 원래 원했던 downstream feature extraction은 가능하다.

Scale 155장 기준:

- summary row: 519개
- scratch sample: 60장
- local sample: 113장
- local의 평균 stby overlap: 약 0.231
- scratch의 p95 stby overlap: 약 0.109

Holdout 120장 기준:

- summary row: 357개
- scratch sample: 78장
- local sample: 65장
- local의 평균 stby overlap: 약 0.225
- scratch의 p95 stby overlap: 약 0.123

## 해석

이 단계에서 확인한 것은 “mask에서 structured FBM feature table로 변환하는 후처리는 가능하다”는 점이다.

남은 어려운 부분은 mask prediction이다. 특히 scratch/local은 작은 면적과 stby overlap 때문에 단순 wafer-level feature나 1x1 baseline으로는 충분하지 않다. 따라서 다음 모델은 spatial context를 보는 U-Net/SegFormer 계열이어야 한다.
