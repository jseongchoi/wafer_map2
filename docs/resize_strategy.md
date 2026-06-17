# Resize Strategy

## 결론

리사이즈는 필요하지만, 현재 기준으로는 **단독 retrieval representation으로 쓰면 안 된다.**

이번 벤치마크는 다음 표현을 비교했다.

```text
compact_features
naive_gray_32 / naive_gray_64
semantic_pool_32 / semantic_pool_64
```

Scale 155장:

- compact feature: lift 약 1.36x
- naive gray 32: lift 약 0.51x
- semantic pool 32: lift 약 0.55x
- naive gray 64: lift 약 0.49x
- semantic pool 64: lift 약 0.49x

Holdout 120장:

- compact feature: lift 약 1.40x
- naive gray 32: lift 약 0.49x
- semantic pool 32: lift 약 0.41x
- naive gray 64: lift 약 0.41x
- semantic pool 64: lift 약 0.40x

## 해석

naive grayscale resize는 stby와 Grade7, none-wafer와 Grade0의 의미가 섞인다. 이 방식은 기대대로 가장 위험하다.

semantic pooling은 severity/fail/stby/valid/wafer를 분리했지만, flatten vector를 그대로 Euclidean nearest-neighbor에 쓰면 차원이 크고 위치 민감도가 과해져 global 유사맵 검색 품질이 떨어진다.

따라서 현재 정책은 다음과 같다.

```text
global grouping/retrieval:
  compact 50 observable features

location-aware feature_key retrieval:
  compact feature + chip-level polar feature

model input / segmentation:
  semantic resized tensor 사용 가능

small defect review:
  low-res candidate -> original-resolution patch 검토
```

## 다음 사용처

리사이즈는 버리는 것이 아니라, 다음 단계에서 모델 입력으로 쓴다.

- small U-Net / lightweight SegFormer 입력
- DINO/metric-learning embedding 후보
- low-res 후보 위치 탐색
- high-res patch crop의 proposal map

현재 CPU retrieval 기준으로는 `compact_features`가 제일 안정적이다.

## Patch Proposal 업데이트

리사이즈/semantic pooling을 retrieval vector로 쓰는 대신, 저해상도 score map에서 원본 해상도 후보 window를 뽑는 실험을 추가했다.

결론:

- edge/local/stby는 proposal map 방식이 유효하다.
- scratch는 hotspot window 방식으로는 여전히 약하다.
- shot은 반복 위치/레티클 상대좌표 feature로 보는 편이 낫고, 단일 hotspot proposal로 평가하면 약하게 나온다.
- ring은 넓고 얇은 구조라 window recall 지표만으로는 과소평가될 수 있다.

Scale 155장:

- edge recall lift 약 8.38x
- local recall lift 약 6.16x
- stby_pattern recall lift 약 6.30x
- scratch recall lift 약 0.91x

Holdout 120장:

- edge recall lift 약 9.26x
- local recall lift 약 6.18x
- stby_pattern recall lift 약 5.74x
- scratch recall lift 약 1.11x

이 결과는 resize가 “검색 feature 대체재”가 아니라 “원본 patch review를 위한 후보 생성기”로는 의미가 있음을 보여준다.

## 산출물

- `outputs/reports/fbm_resize_benchmark_scale_report.html`
- `outputs/reports/fbm_resize_benchmark_scale_metrics.json`
- `outputs/figures/fbm_resize_benchmark_scale_gallery.png`
- `outputs/reports/fbm_resize_benchmark_holdout_report.html`
- `outputs/reports/fbm_resize_benchmark_holdout_metrics.json`
- `outputs/figures/fbm_resize_benchmark_holdout_gallery.png`
- `outputs/reports/fbm_patch_proposal_scale_report.html`
- `outputs/reports/fbm_patch_proposal_scale_metrics.json`
- `outputs/figures/fbm_patch_proposal_scale_gallery.png`
- `outputs/reports/fbm_patch_proposal_holdout_report.html`
- `outputs/reports/fbm_patch_proposal_holdout_metrics.json`
- `outputs/figures/fbm_patch_proposal_holdout_gallery.png`
