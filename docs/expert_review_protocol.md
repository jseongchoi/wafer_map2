# 전문가 리뷰 절차

이 문서는 공정/품질 전문가가 wafer와 모델 결과를 검토할 때의 기준을 정리합니다.
전문가 리뷰의 목적은 모델 점수 하나를 만드는 것이 아니라, 라벨 정의와 asset 품질을
안정화하는 것입니다.

## 1. 목적

전문가 리뷰로 확인할 것:

- family 정의가 현업 관점에서 맞는가?
- `ring`과 `edge`처럼 헷갈리는 유형을 어떻게 나눌 것인가?
- `shot_grid`에 필요한 shot layout metadata를 얻을 수 있는가?
- 어떤 wafer를 학습 target으로 쓰면 안 되는가?
- 모델 prediction이 correction seed로 쓸 만한가?

## 2. 리뷰 단위

리뷰 단위는 세 가지입니다.

| 단위 | 설명 |
|---|---|
| wafer-level | wafer 전체에 어떤 defect family가 보이는지 |
| region-level | defect 후보 bbox/sector/polyline이 맞는지 |
| mask-level | 학습 target으로 쓸 pixel mask가 맞는지 |

처음부터 모든 wafer를 mask-level로 리뷰하지 않습니다.
먼저 wafer-level, region-level로 후보를 줄이고 clean한 예시만 mask-level로 갑니다.

## 3. 허용 라벨

추천 라벨:

```text
local
scratch
ring
edge
shot_grid
random
mixed_unknown
bad_input
no_defect
```

`mixed_unknown`은 실패가 아닙니다.
애매한 것을 억지로 학습 target에 넣지 않기 위한 안전장치입니다.

## 4. Review CSV 예시

```csv
sample_id,primary_family,secondary_family,review_decision,label_type,notes
WAFER_0001,local,scratch,usable_asset,manual_mask,"왼쪽 아래 local 명확, 중앙 scratch도 있음"
WAFER_0002,shot_grid,,needs_parametric_rule,parametric_mask,"shot마다 왼쪽 아래 반복"
WAFER_0003,mixed_unknown,,review_only,weak_region,"diffuse라 family 확정 어려움"
```

## 5. 실행

template 생성:

```powershell
python scripts/make_expert_review_template.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --out outputs/reviews/product_A_review_template.csv
```

리뷰 집계:

```powershell
python scripts/summarize_expert_review.py `
  --review-csv outputs/reviews/product_A_review_template.csv `
  --out outputs/reviews/product_A_review_summary.json
```

## 6. 집계 지표

| 지표 | 의미 |
|---|---|
| family별 sample 수 | 어떤 defect가 충분히 모였는지 |
| `usable_asset` 비율 | 학습에 바로 쓸 수 있는 예시 비율 |
| `needs_parametric_rule` 수 | rule UI/코드가 필요한 유형 |
| `mixed_unknown` 비율 | taxonomy가 부족하거나 불량 정의가 애매한 정도 |
| `bad_input` 비율 | 입력/geometry 품질 문제 |

## 7. 파일럿 성공 판정

성공:

- 전문가가 family 정의를 이해하고 일관되게 사용함
- family별 usable asset 후보가 생김
- 애매한 유형을 `mixed_unknown`으로 분리함
- `shot_grid`, `edge`, `ring`에 필요한 rule parameter가 정리됨

실패:

- 리뷰어마다 family 해석이 크게 다름
- 대부분이 `mixed_unknown`으로만 쌓임
- mask를 학습 target으로 쓰기 어려움
- shot/grid geometry를 얻을 방법이 없음

## 8. 다음 의사결정

| 리뷰 결과 | 다음 행동 |
|---|---|
| local/scratch가 많음 | manual mask asset 수집 확대 |
| ring/edge가 많음 | parametric mask generator 보강 |
| shot_grid가 많음 | shot layout metadata 확보 |
| mixed_unknown이 많음 | taxonomy 재정의 또는 별도 family 검토 |
| bad_input이 많음 | raw ingestion/geometry 보정부터 해결 |
