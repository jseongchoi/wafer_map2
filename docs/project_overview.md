# 프로젝트 개요

WaferMap은 FBM wafer map에서 불량 segmentation 학습 데이터를 만들기 위한
작업 도구와 파이프라인입니다. 최종 목표는 실제 wafer에서 여러 불량이 섞여
있을 때, 각 pixel이 어떤 family에 속하는지 예측하는 multi-label 모델을
만드는 것입니다.

## 1. 현재 제품 형태

현재 제품은 하나의 큰 흐름으로 보면 됩니다.

```text
실제 wafer 입력
-> segmentation tool에서 대표 불량 mask 저장
-> pattern asset library 축적
-> base wafer 위에 asset/rule 합성
-> 학습 가능한 arrays.npz + manifest 생성
-> small U-Net 학습
-> 실제 wafer prediction export
-> 사람이 prediction을 수정해 다시 asset으로 저장
```

이 흐름에서 중요한 것은 “합성 데이터”가 가짜 장난감 데이터가 아니라는 점입니다.
실제 wafer에서 추출한 대표 패턴을 재사용하고, 위치/강도/겹침을 통제해서
정답 mask를 아는 학습 sample을 만드는 장치입니다.

## 2. 주요 구성요소

| 구성요소 | 역할 | 대표 파일 |
|---|---|---|
| 실제 wafer ingestion | raw PNG와 metadata를 manifest로 정리 | `scripts/analyze_png_raw_folders.py` |
| segmentation tool | 사람이 불량 mask를 그리고 asset으로 저장 | `scripts/run_segmentation_tool.py` |
| pattern asset library | 재사용 가능한 불량 조각 관리 | `src/wafermap/assets/` |
| synthetic composer | base wafer에 asset/rule 합성 | `scripts/compose_synthetic_from_assets.py` |
| readiness builder | 학습 manifest와 coverage 점검 | `scripts/build_segmentation_readiness.py` |
| U-Net trainer | multi-label segmentation 학습 | `scripts/train_unet_segmentation.py` |
| prediction export | tool에서 다시 고칠 seed 생성 | `scripts/export_unet_predictions.py` |

## 3. Defect family

현재 primary target은 여섯 개입니다.

| Family | 뜻 | 예시 |
|---|---|---|
| `local` | 작은 국소 blob | 특정 die 주변만 밝게 튀는 패턴 |
| `scratch` | 선형/곡선형 긁힘 | wafer 중앙을 가로지르는 긴 선 |
| `ring` | 원형 또는 부분 원형 band | 특정 반지름 근처의 ring |
| `edge` | wafer edge 근처 band/sector | 오른쪽 edge만 두꺼운 불량 |
| `shot_grid` | shot 상대 위치 반복 불량 | shot마다 왼쪽 아래 die가 반복 fail |
| `random` | sparse baseline fail | 뚜렷한 구조 없는 산발 fail |

`stby_pattern`은 보조 context로 쓰며 현재 primary target에서는 제외합니다.

## 4. 왜 classification이 아니라 segmentation인가

wafer 하나에는 여러 불량이 동시에 있을 수 있습니다.

예시:

```text
왼쪽 아래: local blob
중앙: scratch
edge: 부분 edge band
```

이 wafer를 하나의 label로만 분류하면 중요한 정보가 사라집니다.
이 프로젝트가 필요한 정답은 아래처럼 family별 mask입니다.

```text
local_mask[y, x]
scratch_mask[y, x]
edge_mask[y, x]
```

## 5. 왜 합성맵을 쓰는가

실제 wafer만으로 학습하려면 모든 불량을 사람이 mask로 따야 합니다.
하지만 `ring`, `edge`, `shot_grid` 같은 패턴은 사람이 전부 칠하기 어렵고,
어떤 경우는 정의 자체가 애매합니다.

합성맵을 쓰면 아래 장점이 있습니다.

- 어떤 family를 어디에 넣었는지 알기 때문에 정답 mask가 자동으로 생깁니다.
- rare defect를 더 많이 만들 수 있습니다.
- `local + scratch`, `ring + edge`처럼 조합을 통제할 수 있습니다.
- 모델이 처음부터 완벽하지 않아도 실제 wafer correction loop를 시작할 수 있습니다.

## 6. 현재 검증 상태

현재 구현은 아래 흐름을 테스트로 확인합니다.

```text
pattern asset 저장
-> synthetic composition
-> segmentation readiness manifest
-> small U-Net input/target tensor contract
```

즉, 아직 “운영급 모델 성능”을 주장하는 단계가 아니라
학습 데이터 공장과 correction loop를 안정화하는 단계입니다.

## 7. 다음 구현 우선순위

1. 실제 wafer에서 family별 좋은 예시를 모읍니다.
2. 애매한 불량은 `mixed_unknown`으로 따로 보관합니다.
3. `shot_grid`, `ring`, `edge`는 parametric label UI/저장 구조를 강화합니다.
4. readiness report에서 family coverage와 mask ratio를 확인합니다.
5. U-Net 예측을 segmentation tool에서 수정하는 loop를 반복합니다.

## 8. 완료 기준

이 프로젝트의 “1차 성공”은 모델 점수 하나가 아닙니다.
아래가 가능해야 합니다.

- 작업자가 실제 wafer에서 대표 불량을 5분 안에 asset으로 저장할 수 있음
- 합성 데이터셋이 family별 mask를 포함함
- manifest가 학습 코드에서 바로 읽힘
- 모델 예측을 다시 tool에서 수정할 수 있음
- 수정된 mask가 다음 학습 asset으로 다시 들어감
