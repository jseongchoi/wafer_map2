# 실험과 판단 기록

이 문서는 WaferMap 프로젝트가 왜 현재 구조로 정리됐는지 설명한다. 결론부터 말하면, 단순 rule 기반 feature만으로 끝내기보다 “실제 FBM에서 defect mask asset을 만들고, 그 asset으로 라벨이 있는 합성 데이터를 만들고, multi-label segmentation과 embedding retrieval로 확장하는 구조”가 현재 목표에 가장 맞다.

관련 문서:

- [프로젝트 개요](project_overview.md)
- [데이터 흐름 가이드](fbm_data_flow_guide.md)
- [패턴 asset 파이프라인](fbm_pattern_asset_pipeline.md)
- [모델링 전략](modeling_strategy.md)
- [검증 방법](validation_protocol.md)

## 1. 초기 접근: rule feature와 유사 wafer 검색

처음에는 wafer map에서 radial, angular, edge, hotspot, stby, density 같은 feature를 뽑고 nearest-neighbor 검색으로 비슷한 wafer를 찾는 방향을 검토했다.

좋았던 점:

- 라벨이 없어도 시작할 수 있다.
- 실제 wafer batch를 넣으면 sanity, drift, top-k neighbor를 바로 볼 수 있다.
- 전문가 리뷰용 후보를 빠르게 만들 수 있다.

한계:

- “어느 위치에 어떤 defect family가 있다”는 segmentation mask를 직접 주지는 않는다.
- blob, scratch, ring, random이 섞이면 수치 feature만으로 설명이 부족하다.
- 모델 학습용 정답 mask가 없으면 딥러닝 모델의 실패 원인을 해석하기 어렵다.

따라서 feature retrieval은 버리는 것이 아니라, segmentation 모델 이후의 embedding 검색과 리뷰 후보 생성에 붙이는 보조 축으로 남긴다.

## 2. resize-only representation

`resize-only representation`은 원본 FBM을 단순 resize해서 모델 입력으로 넣는 접근이다.

판단:

- 빠른 baseline으로는 유용하다.
- 하지만 wafer outside, stby, Grade 0 정상 영역, Grade 7 fail 영역이 섞이면 모델이 잘못된 shortcut을 배울 수 있다.
- 절대좌표가 중요한 FBM 특성상 x/y coordinate channel과 wafer mask를 함께 주는 편이 더 타당하다.

현재 방향:

```text
input channels =
  severity normalized
  wafer_mask
  valid_test_mask
  stby_mask
  x_coord
  y_coord
```

## 3. patch proposal

`patch proposal`은 wafer 전체를 바로 segmentation하기 전에, 의심 영역을 patch 단위로 잘라 후보를 만드는 접근이다.

판단:

- local blob이나 scratch 후보를 빠르게 줄이는 데는 좋다.
- 그러나 ring, edge, shot-relative pattern처럼 wafer 전체 맥락이 중요한 family는 patch만 보면 의미가 사라질 수 있다.

현재 방향:

- editor에서는 사람이 직접 defect mask를 만들 수 있게 한다.
- 모델은 wafer 전체 입력을 받는 U-Net 계열을 baseline으로 둔다.
- patch proposal은 추후 active learning에서 “리뷰할 후보 영역”을 줄이는 보조 기능으로 쓴다.

## 4. Segmentation Smoke Test

`Segmentation Smoke Test`는 synthetic mask를 정답으로 놓고 작은 모델이 최소한의 학습 신호를 받을 수 있는지 확인하는 단계다.

목표:

- image와 multi-channel mask가 서로 맞는지 확인한다.
- family별 label channel이 올바른지 확인한다.
- 학습 코드가 Dice/IoU를 계산하고 preview를 생성하는지 확인한다.

이 단계의 성공은 “실제 성능이 좋다”는 뜻이 아니다. 학습 파이프라인이 끊기지 않았고, 라벨 정의가 코드에서 일관되게 흐른다는 뜻이다.

## 5. 실제 FBM 누끼와 합성 데이터로 방향 전환

사용자 피드백에서 핵심이 명확해졌다.

```text
1. 실제 wafer map을 본다.
2. 불량 유형별로 사람이 누끼를 딴다.
3. family별 mask asset 폴더에 저장한다.
4. mask asset을 실제 또는 base wafer에 합성한다.
5. 어떤 위치에 어떤 defect를 넣었는지 metadata로 남긴다.
6. multi-label segmentation 모델을 학습한다.
7. 모델 출력 mask, 수치화, embedding retrieval을 실제 wafer 검증에 쓴다.
```

이 구조는 의료 AI의 병변 segmentation과 유사하다. 병변 mask를 만들고, augmentation과 합성 데이터를 통해 label을 늘리고, 모델이 위치와 형태를 함께 예측하게 만든다.

## 6. 현재 권장 모델

추천 baseline:

- Small U-Net
- multi-label sigmoid output
- family별 mask channel
- 좌표 channel 포함
- loss는 BCE + Dice 조합

출력:

- family probability mask
- family별 면적, 중심, bbox, radial/angle 위치
- wafer-level family score
- embedding vector
- top-k similar wafer

## 7. 계속 확인해야 할 질문

- 실제 raw PNG의 gray value contract가 제품별로 같은가?
- ring처럼 하나의 pattern이 여러 component로 쪼개지는 문제를 editor에서 어떻게 막을까?
- random defect는 누끼 asset보다 procedural generator가 나은가?
- edge defect는 rule generator로 충분한가, 아니면 실제 mask asset도 필요한가?
- synthetic label로 학습한 모델이 실제 FBM에서 어떤 family를 가장 자주 놓치는가?

이 질문에 대한 답은 실제 wafer 5~20장 batch와 전문가 리뷰에서 나온다.
