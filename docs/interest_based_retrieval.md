# Interest-Based Retrieval

## 왜 필요한가

한 장의 wafer에도 edge, shot, ring, scratch, local, stby가 중첩될 수 있다. 따라서 “유사 wafer”라는 말은 항상 관심 기준을 함께 가져야 한다.

예를 들어 같은 query wafer라도 다음 질문은 서로 다른 검색 기준을 요구한다.

- edge 불량이 비슷한 wafer를 찾고 싶은가?
- shot/reticle 반복성이 비슷한 wafer를 찾고 싶은가?
- scratch나 ring처럼 선형/원형 구조가 비슷한 wafer를 찾고 싶은가?
- local blob 위치와 강도가 비슷한 wafer를 찾고 싶은가?
- stby chip이 많은 wafer를 찾고 싶은가?

## 구현

현재 스크립트:

```text
scripts/evaluate_interest_retrieval.py
scripts/evaluate_defect_feature_retrieval.py
```

입력:

- `outputs/reports/fbm_grouping_scale_features.csv`
- `data/synthetic/fbm_grouping_scale_pilot`

출력:

- `outputs/reports/fbm_interest_retrieval_scale_report.html`
- `outputs/reports/fbm_interest_retrieval_scale_metrics.json`
- `outputs/reports/fbm_interest_retrieval_scale_neighbors.csv`
- `outputs/figures/fbm_interest_neighbor_gallery_scale.png`
- `outputs/reports/fbm_defect_feature_retrieval_scale_report.html`
- `outputs/reports/fbm_defect_feature_retrieval_scale_metrics.json`
- `outputs/reports/fbm_defect_feature_retrieval_scale_neighbors.csv`
- `outputs/figures/fbm_defect_feature_retrieval_scale_gallery.png`

## 기준별 검색 방식

`overall`:

- 모든 observable feature 사용
- synthetic multi-label Jaccard로 검증

`edge_focus`:

- edge density
- center density
- edge minus center
- edge-chip outer-face gradient
- edge sector concentration
- outer radial zone

`shot_focus`:

- shot lower-left contrast
- shot bottom-edge contrast
- shot left-edge contrast
- shot best contrast

`stby_focus`:

- 현재는 stby ratio 중심
- stby의 공간 배열은 아직 약하므로 다음 feature 보강 후보

`ring_focus`:

- radial zone feature
- ring radial peak contrast
- ring width ratio

`scratch_focus`:

- angular sector feature
- scratch angular peak contrast
- scratch width ratio
- scratch component elongation
- scratch component linear/radial/angular span

`local_focus`:

- local hotspot peak
- top-3 hotspot mean
- top-3 spread
- hotspot count ratio
- connected-component largest ratio
- component compactness
- component triangle score

## 현재 결과

155장 scale pilot 기준:

- `overall`: lift 약 1.36x
- `edge_focus`: lift 약 1.61x
- `shot_focus`: lift 약 2.02x
- `stby_focus`: lift 약 1.24x
- `ring_focus`: lift 약 1.58x
- `scratch_focus`: lift 약 1.10x
- `local_focus`: lift 약 1.11x

120장 holdout stress 기준:

- `overall`: lift 약 1.40x
- `edge_focus`: lift 약 2.64x
- `shot_focus`: lift 약 1.23x
- `stby_focus`: lift 약 1.48x
- `ring_focus`: lift 약 1.52x
- `scratch_focus`: lift 약 0.92x
- `local_focus`: lift 약 1.33x

해석:

- shot과 edge는 관심 기준을 명시하면 검색 성능이 더 좋아진다.
- ring도 일정 수준 신호가 있다.
- local은 connected-component morphology를 넣으면 일부 개선된다.
- scratch는 morphology feature까지 넣어도 조건 변화에 취약하다.
- stby는 ratio만으로는 “많다/적다”는 잡지만, stby가 어디에 배열되는지는 아직 feature가 부족하다.

## Defect Feature Target 검색

class 단위 interest retrieval은 “edge가 있는 wafer끼리 가까운가?”를 본다. 하지만 실제 업무 질문은 더 세밀할 수 있다.

예:

```text
edge 전체
edge + edge-side radial
edge + 09:00 location
edge__0900__edge_side feature_key
```

이를 위해 `scripts/evaluate_defect_feature_retrieval.py`는 `outputs/reports/fbm_defect_location_summary.csv`에서 만든 structured target을 채점 기준으로 사용한다. 검색 feature는 기존과 동일하게 observable-only feature만 사용한다.

Scale 155장 기준:

- class target: mean P@K 약 0.824, mean lift 약 1.41x
- class+radial target: mean P@K 약 0.588, mean lift 약 1.49x
- class+location target: mean P@K 약 0.299, mean lift 약 2.50x
- feature_key target: mean P@K 약 0.340, mean lift 약 3.13x

Holdout 120장 기준:

- class target: mean P@K 약 0.743, mean lift 약 1.47x
- class+radial target: mean P@K 약 0.507, mean lift 약 1.59x
- class+location target: mean P@K 약 0.178, mean lift 약 1.86x
- feature_key target: mean P@K 약 0.191, mean lift 약 1.97x

해석:

- defect family 단위 검색은 이미 꽤 안정적이다.
- 위치와 radial zone까지 포함하면 target이 희소해져 P@K는 낮아진다.
- 그래도 random baseline 대비 lift가 1보다 높게 유지되어, 위치-aware 검색 신호는 존재한다.
- scratch/local/stby 위치 검색은 아직 불안정하므로 spatial feature 또는 segmentation/model representation으로 넘겨야 한다.
- polar spatial feature는 global similarity에 섞으면 잡음이 된다. 따라서 feature CSV에는 보관하되, global grouping/retrieval은 compact 50개 feature를 쓰고, `class_location`/`feature_key` target 검색에서만 polar feature를 사용한다.

## 중요한 원칙

Synthetic label은 검증용 정답지로만 사용한다.

실제 inference에서는 다음을 feature로 쓰지 않는다.

- `label_*`
- `*_mask_ratio`
- `pattern_masks`

실제 wafer에서는 이 리포트와 같은 관심 기준별 feature subset을 쓰고, 결과는 expert review protocol로 사람이 평가한다.
