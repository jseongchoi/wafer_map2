# 용어와 변수 설명

이 문서는 프로젝트 문서와 HTML에서 반복해서 나오는 변수, 지표, 리뷰 용어를 설명한다.

## 1. 입력 배열 변수

| 이름 | 뜻 | 왜 필요한가 | 예시/주의 |
| --- | --- | --- | --- |
| `severity` | wafer map의 불량 grade 값. 현재 기준은 0~7이다. | fail 강도와 위치를 feature로 바꾸기 위한 핵심 입력이다. | wafer 밖 영역과 stby 영역은 0이어야 한다. |
| `wafer_mask` | wafer 내부인지 아닌지를 표시하는 mask. wafer 내부는 1, wafer 밖은 0이다. | wafer 밖 0과 wafer 안의 정상 Grade 0을 구분한다. | 이 값이 없으면 none-wafer와 정상 영역이 섞인다. |
| `valid_test_mask` | 실제 test가 수행된 pixel인지 표시한다. test된 곳은 1, test되지 않은 곳은 0이다. | stby나 미측정 영역을 실제 정상 영역과 구분한다. | stby 영역은 보통 0이다. |
| `stby_mask` | stby fail chip 영역을 표시한다. stby 영역은 1이다. | stby를 Grade 7 같은 불량으로 오해하지 않게 한다. | stby는 `severity=0`, `valid_test_mask=0`, `stby_mask=1`이어야 한다. |
| `chip_index` | die/chip id를 나타내는 배열이다. | chip 단위 geometry와 shot-relative feature를 안정적으로 계산하는 데 도움된다. | 권장 입력이다. wafer 밖은 -1로 둔다. |

## 2. Manifest 변수

| 이름 | 뜻 | 왜 필요한가 |
| --- | --- | --- |
| `manifest` | 여러 wafer sample의 raw PNG 또는 `.npz` 경로와 metadata를 적은 JSON 파일이다. | batch 실행, 에디터, feature 추출이 같은 입력 목록을 보게 한다. |
| `sample_id` | wafer sample을 가리키는 id다. | 리뷰와 검색 결과에서 wafer를 구분한다. 사람이 읽기 쉬운 이름을 써도 된다. |
| `png_grayscale_raw` | 실제 8-bit grayscale PNG를 직접 읽는 source type이다. | gray value를 Grade 0~7과 stby 후보로 변환한다. |
| `png_path` | raw PNG 파일 경로다. | 제품별 PNG batch나 manifest에서 입력 파일을 참조한다. |
| `arrays_npz` | 해당 sample의 `.npz` 파일 경로다. | `severity`, `wafer_mask`, `valid_test_mask`, `stby_mask`를 읽는 위치다. |
| `array_keys` | 원본 `.npz` 안의 key 이름이 표준 이름과 다를 때 쓰는 mapping이다. | 예를 들어 원본 key가 `grade`이면 이를 `severity`로 연결한다. |
| `parser_name` / `parser_version` | 실제 FBM을 `.npz`로 바꾼 parser의 이름과 버전이다. | 추후 parser 변경으로 결과가 달라졌는지 추적한다. |
| `orientation` | wafer map 방향 정보다. | 회전/flip이 있으면 clock 위치와 edge 위치 해석이 달라진다. |
| `chip_blocks` | 한 chip이 array에서 차지하는 block 크기다. | chip 단위 feature와 geometry 검증에 필요하다. |
| `grid` | wafer map의 chip row/column 개수다. | die layout과 예상 net die 수를 확인한다. |
| `actual_net_die` | 실제 wafer 안에서 유효한 die 수다. | generator 또는 parser 결과가 말이 되는지 sanity check에 쓴다. |
| `wafer_mask_strategy` | PNG에서 wafer 밖 0과 good 0을 나눌 때 쓰는 mask 추정 방식이다. | 기본값은 `centered_ellipse_from_png`이다. 제품 layout이 다르면 별도 geometry나 mask가 필요하다. |

## 3. 실제 raw PNG gray 기준

| Gray value | 의미 | 주의 |
| --- | --- | --- |
| `0` | Grade 0, good | wafer 밖도 0일 수 있어 mask 추정이 필요하다. |
| `31` | Grade 1 |  |
| `151` | Grade 2 |  |
| `175` | Grade 3 |  |
| `191` | Grade 4 |  |
| `207` | Grade 5 |  |
| `223` | Grade 6 |  |
| `255` | Grade 7 또는 stby 후보 | chip 전체가 255일 때만 stby로 분리한다. |

## 4. Feature와 검색 용어

| 이름 | 뜻 | 해석 방법 |
| --- | --- | --- |
| `feature` | wafer map에서 계산한 숫자 표현이다. | 예: fail density, edge density, ring contrast, stby ratio. |
| `compact feature 50개` | 전체 유사 wafer 검색에 쓰는 핵심 feature 묶음이다. | 실제 wafer에서도 계산 가능한 값만 사용한다. |
| `observable feature` | label이나 oracle mask 없이 FBM 자체에서 계산 가능한 feature다. | 실제 inference에 쓸 수 있는 feature라는 뜻이다. |
| `nearest-neighbor` | feature가 가장 가까운 wafer를 찾는 검색 방식이다. | 가까울수록 비슷한 wafer 후보로 본다. |
| `top-k` | query wafer 하나에 대해 가까운 neighbor k개를 보는 방식이다. | top-5면 가장 가까운 5개를 본다. |
| `retrieval lift` | random 검색보다 nearest-neighbor 검색이 얼마나 나은지 보는 비율이다. | 1보다 크면 random보다 낫다는 뜻이다. |
| `reference features` | 비교 기준이 되는 기존 feature table이다. | real wafer query를 synthetic/reference wafer와 비교할 때 쓴다. |
| `feature drift` | 새 wafer feature 분포가 reference와 얼마나 다른지 보는 값이다. | 성능 점수가 아니라 입력 분포 sanity signal이다. |
| `sanity check` | 입력 배열과 metadata가 정해진 형식을 지키는지 확인하는 검사다. | FAIL이면 검색보다 parser/export 문제를 먼저 봐야 한다. |

## 5. 검색에서 제외하는 값

전체 유사 wafer 검색에서는 아래 값을 쓰지 않는다.

| 이름 | 제외 이유 |
| --- | --- |
| `label_*` | 합성 데이터 검증용 label이라 실제 wafer inference에 없다. |
| `*_mask_ratio` | 합성 oracle mask에서 온 값일 수 있어 실제 feature에 섞으면 안 된다. |
| `pattern_masks` | 합성 데이터에서만 아는 정답 mask다. |
| `pattern_intensity` | 합성 데이터에서만 아는 pattern 강도다. |
| `polar_*` | 전체 유사 검색에 위치 편향을 강하게 줄 수 있다. |
| `stby_polar_*` | 전체 유사 검색에는 제외하고, 위치가 중요한 검색에서만 조건부로 쓴다. |

## 6. 전문가 리뷰 컬럼

| 이름 | 뜻 | 채우는 방법 |
| --- | --- | --- |
| `review_case_id` | query-neighbor pair를 구분하는 id다. | 자동 생성된다. |
| `query_sample_id` | 검색 기준 wafer의 id다. | 자동 생성되거나 manifest에서 온다. |
| `neighbor_sample_id` | 검색된 neighbor wafer의 id다. | 자동 생성되거나 manifest에서 온다. |
| `rank` | neighbor 순위다. | 1이면 가장 가까운 wafer다. |
| `distance` | feature 공간에서 query와 neighbor 사이의 거리다. | 낮을수록 feature 기준으로 가깝다. |
| `reviewer_decision` | 사람이 봤을 때 비슷한지 판단한 값이다. | `same_family`, `partial_match`, `mismatch`, `not_sure` 중 선택한다. |
| `query_defect_family` | query wafer에서 주로 보이는 defect family다. | 예: `edge`, `ring`, `scratch`, `local`, `stby_pattern`. |
| `neighbor_defect_family` | neighbor wafer에서 주로 보이는 defect family다. | query와 같은 계열인지 비교한다. |
| `dominant_defect` | pair 판단에 가장 큰 영향을 준 defect다. | 여러 defect가 섞였으면 가장 중요한 것을 적는다. |
| `clock_position_match` | 위치/방향이 비슷한지 보는 값이다. | `yes`, `partial`, `no`, `not_applicable`. |
| `missed_major_defect` | neighbor가 query의 중요한 defect를 놓쳤는지 표시한다. | 놓쳤으면 `yes`. |
| `retrieval_failure_mode` | 검색이 틀렸다면 왜 틀렸는지 적는 값이다. | 예: `wrong_family`, `right_family_wrong_location`, `parser_or_mask_issue`. |
| `next_action` | 다음에 무엇을 보강할지 적는 값이다. | 예: `feature_weight_tuning`, `segmentation_candidate`, `parser_validation`. |
| `review_comment` | 리뷰어 메모다. | 판단 근거를 자유롭게 적는다. |

## 7. Defect Family

| 이름 | 뜻 |
| --- | --- |
| `edge` | wafer edge 근처 fail density가 올라가는 패턴이다. |
| `shot_grid` | reticle/shot 위치와 맞물려 반복되는 패턴이다. |
| `stby_pattern` | stby fail chip 자체의 missing-test 패턴이다. |
| `stby_hidden_origin` | stby가 실제 defect origin을 가리는 경우다. |
| `ring` | wafer 중심 기준 ring 또는 partial arc 패턴이다. |
| `scratch` | 길고 좁은 선형/곡선형 defect다. |
| `local` | 국소 hotspot 또는 compact blob cluster다. |
| `random` | 구조가 약한 산발성 fail이다. |
| `mixed` | 여러 defect family가 동시에 보이는 경우다. |

## 8. 이 문서를 읽는 방법

1. 실제 PNG 폴더나 `.npz`를 준비할 때는 "입력 배열 변수", "Manifest 변수", "실제 raw PNG gray 기준"을 본다.
2. 검색 결과를 볼 때는 "Feature와 검색 용어"를 본다.
3. 리뷰 양식을 채울 때는 "전문가 리뷰 컬럼"을 본다.
4. 특정 불량 이름이 헷갈리면 "Defect Family"를 본다.
