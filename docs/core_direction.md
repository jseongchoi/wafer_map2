# 핵심 방향

이 프로젝트의 목적은 FBM wafer map에서 “불량이 어디에 어떤 형태로 있는지”를
가르칠 수 있는 segmentation 학습 데이터를 만드는 것입니다.

중요한 결론은 하나입니다.

```text
모든 wafer를 사람이 전부 누끼 따는 프로젝트가 아니다.
대표 불량 패턴과 규칙을 모아,
라벨을 아는 합성 wafer map을 만들고,
그 데이터로 multi-label segmentation 모델을 시작한다.
```

## 1. 제품 경계

WaferMap은 현재 아래 흐름을 제품의 중심으로 둡니다.

```text
FBM maps
-> defect generation
-> multi-defect synthetic maps
-> multi-defect segmentation training and validation
-> real-data pattern asset extraction
```

한국어로 풀면 이렇습니다.

```text
실제 FBM wafer map 수집
-> 대표 불량 패턴 추출
-> 여러 불량이 섞인 합성 wafer map 생성
-> family별 mask를 가진 U-Net 학습/검증
-> 실제 wafer 예측 결과를 사람이 고쳐 다시 asset으로 축적
```

## 2. 지금 집중하는 것

현재 1순위는 “큰 모델을 바로 학습하는 것”이 아니라 학습 가능한 라벨 데이터를
만드는 것입니다.

구체적으로는 아래 네 가지입니다.

| 작업 | 의미 | 예시 |
|---|---|---|
| 실제 wafer에서 패턴 추출 | 좋은 대표 불량을 골라 asset으로 저장 | `local` blob 하나를 mask로 저장 |
| 규칙 기반 라벨 정의 | 손마스킹이 비효율적인 패턴을 파라미터로 표현 | `shot_grid`, `ring`, `edge` |
| 합성 wafer 생성 | base wafer 위에 asset/rule을 얹어 정답 mask를 자동 생성 | local + scratch + edge가 함께 있는 sample |
| 작은 U-Net 학습 | 완벽한 모델이 아니라 correction loop의 시작점 생성 | prediction을 tool에서 다시 수정 |

## 3. 범위 안

아래 작업은 이 프로젝트의 현재 범위 안입니다.

- 실제 raw PNG 또는 semantic `.npz`에서 wafer sample을 manifest로 정리합니다.
- segmentation tool에서 사람이 명확한 불량만 mask로 저장합니다.
- `local`, `scratch`, `ring`, `edge`, `shot_grid`, `random` family를 관리합니다.
- `ring`, `edge`, `shot_grid`, 긴 `scratch`는 parametric mask를 적극 사용합니다.
- pattern asset과 procedural pattern을 base wafer에 합성합니다.
- `arrays.npz`, `metadata.json`, manifest CSV를 만들어 U-Net 학습에 연결합니다.
- 모델 예측을 JSON/mask로 export해서 다시 사람이 고칠 수 있게 합니다.

## 4. 범위 밖

아래는 지금 우선순위가 아닙니다.

- 처음부터 완벽한 production AI 모델 만들기
- 모든 불량을 사람 손으로 pixel 단위 라벨링하기
- 애매한 불량을 억지로 known family에 끼워 넣기
- 좌표 정보 없이 이미지만 보고 일반 분류 모델을 만드는 것
- wafer 단위 단일 label classification으로 문제를 축소하는 것

예를 들어 wafer 한 장에 `ring`, `scratch`, `local`이 같이 있으면
“이 wafer는 scratch”라고 하나로 분류하면 안 됩니다.
우리에게 필요한 정답은 family별 mask입니다.

```text
target[local, y, x] = 1 또는 0
target[scratch, y, x] = 1 또는 0
target[ring, y, x] = 1 또는 0
```

## 5. 라벨링 판단 규칙

사람이 모든 pixel을 다 칠하려고 하면 프로젝트가 멈춥니다. 그래서 불량별로
라벨링 방식을 다르게 가져갑니다.

| 불량 유형 | 권장 라벨 방식 | 이유 |
|---|---|---|
| `local` blob | 수동 mask | 영역이 작고 경계가 비교적 명확함 |
| `scratch` | polyline + width 또는 수동 mask | 긴 선은 점 몇 개와 폭으로 표현 가능 |
| `ring` | center, radius, width | 원형/호형은 코드로 mask 생성이 쉬움 |
| `edge` | angle range, edge width | wafer edge sector는 규칙이 더 안정적 |
| `shot_grid` | shot layout, affected slot | 반복 구조라 손으로 칠하면 비효율적 |
| 애매한 diffuse | `mixed_unknown`, review-only | 억지 라벨은 모델을 망침 |

## 6. 의사결정 규칙

새 기능이나 문서를 추가할 때는 아래 질문으로 판단합니다.

1. 이 변경이 대표 불량 패턴을 더 잘 모으게 하는가?
2. 이 변경이 합성 wafer와 정답 mask를 더 안정적으로 만들게 하는가?
3. 이 변경이 U-Net 학습 데이터 규격과 직접 연결되는가?
4. 이 변경이 실제 wafer 예측을 사람이 고쳐 다시 asset으로 축적하는 loop를 돕는가?

위 질문에 “아니오”가 많으면 지금 할 일이 아닐 가능성이 큽니다.

## 7. 예시: 좋은 방향과 나쁜 방향

좋은 방향:

```text
실제 wafer에서 명확한 scratch 30개를 저장한다.
각 scratch의 mask와 metadata를 정리한다.
base wafer에 scratch를 여러 위치/강도로 합성한다.
scratch target mask가 들어간 manifest를 만든다.
U-Net을 학습하고 실제 wafer 예측을 다시 수정한다.
```

나쁜 방향:

```text
wafer 전체를 보고 "불량 A"라고만 적는다.
불량 위치 mask는 없다.
ring인지 scratch인지 애매한데 억지로 scratch라고 저장한다.
그 데이터를 바로 모델에 넣는다.
```

이 경우 모델은 “어디가 scratch인지” 배우지 못하고, 애매한 라벨 때문에
family 경계도 흐려집니다.
