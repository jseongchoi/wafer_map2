# 실제 Wafer 리뷰 체크리스트

이 문서는 실제 wafer 결과를 사람이 검토할 때 체크할 항목을 정리합니다.
리뷰의 목적은 “예쁘게 점수 내기”가 아니라 다음 학습 cycle에 넣을 수 있는
좋은 asset과 correction 정보를 얻는 것입니다.

## 1. 리뷰 결론 형식

wafer 하나를 보고 아래 중 하나로 결론을 냅니다.

| 결론 | 의미 | 다음 행동 |
|---|---|---|
| `usable_asset` | 명확한 불량 mask로 저장 가능 | asset 저장 |
| `needs_parametric_rule` | 반복/규칙 패턴 | rule 작성 |
| `review_only` | 애매하지만 기록 필요 | 학습 제외 |
| `bad_input` | image/geometry 문제 | 입력 수정 |
| `no_defect` | 학습 target 없음 | negative/context로만 사용 |

## 2. 리뷰어가 보는 순서

1. wafer 전체 preview를 봅니다.
2. edge/ring/shot 반복처럼 큰 구조를 먼저 봅니다.
3. local blob과 scratch 후보를 봅니다.
4. family가 명확한 것만 mask로 저장합니다.
5. 애매한 것은 억지로 넣지 않고 review-only로 남깁니다.

## 3. 입력 준비

리뷰어에게 줄 파일:

```text
outputs/reports/real_png_batch/index.html
outputs/manifests/real_png_batch_manifest.json
outputs/reviews/product_A_review_template.csv
```

가능하면 함께 줄 정보:

- 제품/공정 설명
- wafer orientation
- shot layout
- die pitch
- known bad region rule

## 4. 실행 명령

review template 생성:

```powershell
python scripts/make_expert_review_template.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --out outputs/reviews/product_A_review_template.csv
```

segmentation tool 실행:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id WAFER_0001 `
  --assets-root data/pattern_assets
```

## 5. 리뷰어가 채울 것

권장 column:

| Column | 예시 | 설명 |
|---|---|---|
| `sample_id` | `WAFER_0001` | wafer id |
| `primary_family` | `scratch` | 가장 뚜렷한 family |
| `secondary_family` | `edge` | 함께 보이는 family |
| `review_decision` | `usable_asset` | 리뷰 결론 |
| `label_type` | `manual_mask` | mask/rule/review-only |
| `notes` | `중앙 scratch 명확함` | 사람이 보는 설명 |

## 6. Family별 리뷰 질문

| Family | 질문 |
|---|---|
| `local` | blob 경계가 충분히 명확한가? |
| `scratch` | 선 중심과 폭을 설명할 수 있는가? |
| `ring` | 중심/반지름/두께로 표현 가능한가? |
| `edge` | edge sector 범위를 말할 수 있는가? |
| `shot_grid` | shot layout과 affected slot을 알 수 있는가? |
| `random` | 구조 없는 sparse fail인지 확인했는가? |

## 7. 리뷰 후 요약

```powershell
python scripts/summarize_expert_review.py `
  --review-csv outputs/reviews/product_A_review_template.csv `
  --out outputs/reviews/product_A_review_summary.json
```

요약에서 볼 것:

- family별 usable asset 수
- review-only 비율
- parametric rule이 필요한 wafer 수
- bad input 비율

## 8. 나에게 주면 되는 결과

사용자가 리뷰를 마친 뒤 다음 세 가지를 주면 됩니다.

```text
1. 수정된 review CSV
2. 저장된 pattern assets 폴더
3. 리뷰 중 애매했던 예시 목록
```

이 정보로 다음 합성 데이터셋과 U-Net 재학습을 진행할 수 있습니다.

## 9. 파일럿 성공 기준

처음 파일럿은 작게 봅니다.

- 20~50장 wafer를 열어볼 수 있음
- family별로 최소 몇 개의 명확한 예시를 저장함
- `shot_grid` 또는 `edge`처럼 rule이 필요한 유형을 식별함
- asset report에서 품질을 확인함
- 합성 sample로 readiness manifest를 만들 수 있음

## 10. AI 모델 구현 상태

현재 AI 모델은 “완성된 자동 검사기”가 아닙니다.
현재 역할은 아래입니다.

```text
synthetic data로 small U-Net 학습
-> 실제 wafer에 prediction seed 제공
-> 사람이 수정
-> 수정 결과를 다시 asset으로 축적
```

따라서 리뷰어는 prediction을 정답으로 받아들이지 말고 수정 가능한 초안으로 봐야 합니다.
