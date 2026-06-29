# 운영 준비도 평가

이 문서는 WaferMap 프로젝트가 실제 FBM 업무에 투입되기 위해 무엇이 준비됐고 무엇이 남았는지 정리한다. 현재 목표는 “FBM wafer map에서 불량 패턴을 추출하고, 라벨이 있는 합성 데이터를 만들고, multi-label segmentation 모델로 위치와 패턴을 수치화하는 파이프라인”이다.

## 현재 판정

현재 평가는 문서, synthetic smoke test, raw PNG batch 준비 상태를 함께 본다. 실제 성능 판정은 real FBM batch와 전문가 리뷰 CSV가 들어온 뒤 가능하다.

## 단계별 기준

| 단계 | 현재 의미 | 통과 기준 |
| --- | --- | --- |
| 0. 문제 정의 | FBM grade, wafer mask, stby, defect family 정의 | 문서와 schema가 같은 용어를 사용 |
| 1. 원본 입력 | 실제 FBM raw PNG 또는 semantic npz 입력 | Grade 0~7, stby, wafer outside를 안정적으로 분리 |
| 2. 누끼/마스크 | 사람이 defect pattern mask를 만들 수 있음 | editor save 결과를 다시 열어 확인 가능 |
| 3. 마스크 라이브러리 | family별 defect mask asset을 축적 | `data/masks/<family>/` 단위로 재사용 가능 |
| 4. 합성 데이터 | 실제 wafer 배경에 defect mask를 합성 | image, multi-channel mask, metadata가 함께 생성 |
| 5. 기준 모델 | U-Net 계열 multi-label segmentation baseline | family별 Dice/IoU, 위치 오차, confidence 산출 |
| 6. 유사맵 검색 | encoder embedding 또는 segmentation feature 검색 | top-k neighbor가 전문가 기준으로 납득 가능 |
| 7. 실제 리뷰 | 실제 wafer 5~20장 이상으로 검증 | 틀린 family와 위치 오류가 기록됨 |
| 8. 반복 개선 | 실패 family 중심으로 generator/model 보정 | 같은 오류가 줄어드는지 재평가 |

## 현재 강점

- raw PNG 폴더를 읽어 manifest, feature, sanity report, neighbor 후보를 만들 수 있다.
- editor 기반으로 defect mask asset을 저장하고 확인하는 흐름이 있다.
- synthetic dataset generator, feature retrieval, CPU encoder baseline이 연결되어 있다.
- 문서상 입력/출력 위치가 정리되어 작업자가 어디에 파일을 둬야 하는지 알 수 있다.

## 아직 중요한 공백

- 실제 FBM batch 기준으로 defect family 정의가 충분히 축적되지 않았다.
- random처럼 사람이 누끼따기 어려운 family는 코드형 generator 또는 weak-label 전략이 필요하다.
- segmentation 모델 학습/평가 루프는 baseline을 만들고 real review feedback으로 조정해야 한다.
- 유사맵 검색은 feature baseline에서 시작하되, 최종적으로 encoder embedding과 segmentation output을 함께 쓰는 방향이 적합하다.

## 운영 투입 전 체크리스트

| 항목 | 기준 |
| --- | --- |
| 실제 입력 | `data/raw/<product>/*.png` 또는 사내 경로에 제품별 PNG 배치 |
| 누끼 asset | `data/masks/<family>/` 아래 family별 PNG/JSON 저장 |
| 합성 산출물 | `outputs/synthetic_dataset/<run>/` 아래 image, mask, metadata 생성 |
| 리포트 | `outputs/reports/<run>/report.html` 생성 |
| manifest | `outputs/manifests/<run>_manifest.json` 생성 |
| 리뷰 CSV | query-neighbor 판단, dominant defect, missed major defect 기록 |
| 모델 산출물 | `models/<run>/` 또는 `outputs/models/<run>/`에 checkpoint와 metrics 저장 |

## 결론

프로젝트는 “딥러닝으로 갈 수 있는 구조”를 만들고 있다. 단, 처음부터 모델만 던지는 방식은 라벨 정의와 합성 데이터 품질이 흔들리면 성능을 해석할 수 없다. 그래서 현재의 핵심은 defect mask asset을 잘 만들고, 그 asset으로 라벨이 명확한 synthetic dataset을 만든 뒤, U-Net 계열 모델과 embedding 검색을 붙이는 것이다.
