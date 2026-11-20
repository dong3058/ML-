# ashrae-energy-prediction

Kaggle 대회 **[ASHRAE – Great Energy Predictor III](https://www.kaggle.com/competitions/ashrae-energy-prediction)** 를 기준으로 한 건물 에너지 사용량 예측 실험.
LightGBM 베이스라인 + Optuna 하이퍼파라미터 튜닝, MLflow로 실험 기록, ONNX로 모델 내보내기.

| 항목 | 내용 |
|---|---|
| 문제 | 지도학습 **회귀** — 건물 1,449개 × 미터 4종(전기·냉수·증기·온수)의 시간별 사용량(`meter_reading`) 예측 |
| 학습 / 예측 | 2016년 1년치 학습 → 2017~2018년 2년치 예측 (정답 비공개, Kaggle 제출로 채점) |
| 평가 지표 | **RMSLE** (모델은 `log1p(meter_reading)`에 L2 손실로 학습 = RMSLE 최소화) |
| 최고 점수 | **Kaggle private 1.458 / public 1.178** (제출 58600018) |

---

## 1. 폴더 구조

| 경로 | 내용 |
|---|---|
| `app/ashrae.py` | 데이터 추출·전처리(날씨 시간대 보정 포함)·튜닝·학습·등록·검증·제출 파일 생성 |
| `app/ml.py` | 범용 LightGBM 학습/튜닝, ONNX 변환(`_export_onnx`, float32 안전 기준값 처리) |
| `app/main.py`, `app/schemas.py` | FastAPI (포트 8081): `/train`, `/tune`, `/predict`, `/models/{id}/onnx` |
| `scripts/load_ashrae.py` | Kaggle 다운로드 → PostgreSQL 적재 |
| `scripts/train_ashrae.py` | 학습 CLI |
| `scripts/start_mlflow.ps1` | MLflow 서버 (PostgreSQL + S3 호환 저장소) |
| `.env.example` | 접속 정보 형식. 실제 값은 `.env`(git 제외)에 |

데이터(`data/`), 모델 파일(`models/`), 가상환경(`.venv/`), `.env`, Kaggle 토큰은 저장소에 올리지 않는다.

## 2. 실행 방법

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
cp .env.example .env            # 실제 값 채우기 (PostgreSQL, MLflow, S3, Kaggle)

.venv/Scripts/python scripts/load_ashrae.py --include-train          # Kaggle → PostgreSQL
powershell -ExecutionPolicy Bypass -File scripts/start_mlflow.ps1    # MLflow UI: http://localhost:5000

# 미터별 모델: 튜닝 → 학습 (검증 1~8월 / 9~12월)
.venv/Scripts/python scripts/train_ashrae.py --per-meter --trials 20 --timeout 1800 --max-trees 5000 [--no-save]
# 튜닝만 (정답값 정리 실험: --train-clean)
.venv/Scripts/python scripts/train_ashrae.py --per-meter --tune-only --trials 20
# 2016년 전체 학습 → test 예측 → Kaggle 제출 파일 (현재 최고 설정)
.venv/Scripts/python scripts/train_ashrae.py --full-submit [--with-ids | --no-floor]
```

## 3. 데이터 가공 흐름

```
Kaggle CSV ──(load_ashrae.py)──▶ PostgreSQL ashrae 스키마
   · timestamp → year / month / day / hour / day_of_week 분리 저장
   · PK·FK·인덱스 (site, meter_type, building_metadata, weather_*, train, test)
        │
        ▼ (load_dataset) train ⨝ building_metadata ⨝ weather_train  → parquet 캐시
   · site 0 전기 2016-05-20 이전 제외 (98.9%가 0인 알려진 결측)
   · primary_use → 정수 코드, is_weekend 생성
        │
        ▼ (load_weather_hourly) 날씨 정리 — 원본은 그대로, 학습 시 메모리에서만
   · 사이트별 1시간 격자 (빠진 시간 = 결측 행)
   · 강수 -1(미량) → 0
   · 결측: 정확히 1시간 전·후 값 평균, 한쪽만 있으면 그 값, 둘 다 없으면 NaN
   · 풍향 → sin/cos (무풍 → (0,0))
   · UTC → 사이트별 현지 시각 (표준 시차 + 서머타임, site 14는 서머타임 없음)
        │
        ▼ (preprocess)
   · 건물 1099 제외 (증기 최대 2,190만, 전체 사용량 합의 78%)
   · 24h 이상 연속 0 구간: 버전에 따라 제거 / 유지 (아래 표)
        │
        ▼ 피처 → 미터별 LightGBM (정답 log1p(meter_reading))
        │
        ├─▶ MLflow 기록 (파라미터·지표, 모델 레지스트리, ONNX)
        └─▶ test 예측 → expm1 → Kaggle 제출 CSV (41,697,600행)
```

### 원본 데이터 특징 (EDA)
- 결측: `floor_count` 82.7%, `year_built` 60.0%, `cloud_coverage` 43.7%, `precip_depth_1_hr` 18.5%
- 날씨는 **UTC**, 사용량은 **현지 시각** — 미국 13개 사이트에서 5~8시간 어긋남 (하루 최고 기온 시각이 19~23시)
- 0값의 대부분은 **계절 운휴** (냉수는 겨울, 증기·온수는 여름) — 이상값이 아님
- 패턴: 전기는 계절 변화가 작고(월별 패턴 지수 0.90~1.02) 하루 변화가 큼. 증기·온수는 함께 움직이고(상관 +0.99), 냉수는 그 반대(−0.85)

## 4. 시도한 버전

### 4-1. 내부 검증 단계 (1~8월 학습 / 9~12월 검증)

| 버전 | 모델 | 주요 변경 | 검증 RMSLE |
|---|---|---|---|
| v1 | 단일 | 베이스라인 (시간별 원본 날씨, `square_feet` log1p) | 1.1450 ※ |
| v2 | 단일 | `year_built` 결측→0, `floor_count` 제외 | 1.1410 ※ |
| v3 | 단일 | 건물 1099 제외, 24h 이상 0 구간 제거, `square_feet` 원본 | 0.7923 |
| v4 | 단일 | 날씨를 하루 평균으로 (튜닝 후 중단) | 튜닝 0.8105 |
| v5 | 단일 | 시간별 날씨 복귀, ±1h 결측 채움, 풍향 sin/cos (튜닝 후 중단) | 튜닝 0.8019 |
| M1 | 미터별 4개 | 미터당 trial 10 | 0.7945 |
| M2 | 미터별 4개 | trial 20, 트리 상한 5,000, `square_feet` log1p | 0.7921 |
| **M3** | 미터별 4개 | **날씨 UTC → 현지 시각 보정** | **0.7882** |

※ v1·v2는 검증셋에 0 구간·건물 1099가 포함돼 v3 이후와 직접 비교 불가.

| 미터 | v3 단일 | M2 | **M3** |
|---|---|---|---|
| 전기 | 0.4638 | 0.4629 | **0.4624** |
| 냉수 | 1.0244 | 1.0234 | **1.0205** |
| 증기 | 1.1194 | 1.1176 | **1.1075** |
| 온수 | 1.5296 | 1.5359 | **1.5255** |

**정답값 정리 실험 (튜닝만, 미터당 20회)** — 검증셋을 원본(0 구간 포함)으로도 채점

| 미터 | 원본 검증: 기준 → 정리 후 | 판정 |
|---|---|---|
| 전기 | 0.6651 → 0.6695 | 나빠짐 (값 고착 제거가 실제 일정 부하까지 지움) |
| 냉수 | 1.5162 → **1.3921** | 크게 개선 (겨울 운휴 0을 학습) |
| 증기 | 1.4713 → 1.4930 | 나빠짐 (운휴 계절이 검증 기간과 안 겹침) |
| 온수 | 1.8458 → **1.7147** | 크게 개선 (여름 운휴 0을 학습) |

→ 0 구간을 지운 검증셋 점수는 실제 성능을 과대평가함을 확인.

### 4-2. Kaggle 제출 단계 (2016년 전체 학습 → 2017~2018 예측)

공통: 미터별 LightGBM, M3의 미터별 최적 파라미터·트리 수, 0 구간 **유지**, `square_feet` 원본, 현지 시각 날씨.

| 제출 | 피처 변경 | public | **private** |
|---|---|---|---|
| **58600018 (기준)** | `building_id`·`site_id` 제외 → **`building_level`**(건물·미터별 2016년 log 평균), **`month`** 추가 (18개) | **1.178** | **1.458** |
| 58601688 (A) | + `building_id`, `site_id` | 1.185 | 1.462 |
| 58604228 | − `floor_count` | 1.179 | 1.459 |

- ID를 다시 넣으면 학습 데이터에는 더 맞지만(in-sample RMSLE 전 미터 하락) test에서는 나빠짐 → 2016년 건물별 세부를 외우는 과적합.
- `floor_count`는 효과가 거의 없음 (±0.001).

## 5. 점수 지표 정리

| 기준 | 값 | 의미 |
|---|---|---|
| 내부 검증 (0 구간 제거된 9~12월) | 0.788 (M3) | 낙관적. 같은 해 몇 달 뒤 + 이상 구간 제외 |
| 내부 검증 (원본 9~12월, 튜닝 샘플) | 전기 0.67 · 냉수 1.39~1.52 · 증기 1.47 · 온수 1.71~1.85 | 실제 조건에 더 가까움 |
| **Kaggle private** | **1.458** | 1~2년 뒤 예측, 공식 기준 |
| 참고: 1등 (Isamu & Matt) | 1.231 | 유출 데이터로 일부 예측을 실제 값으로 덮어씀 → 직접 비교 불가 |

RMSLE 1.458 ≈ 예측이 실제값에서 전형적으로 e^1.458 ≈ 4.3배 벗어나는 수준 (큰 오차 일부가 지배).

## 6. 주요 발견과 해결

1. **날씨 시간대** — 사이트별 표준 시차(겨울 기온 곡선 위상)와 서머타임 적용 여부(평일 전기 아침 상승 시각의 계절 차이)를 데이터로 추정해 보정. 단일 변경 중 가장 큰 개선(−0.0039, 증기 −0.0101).
2. **ONNX 불일치** — LightGBM의 double 분할 기준값이 ONNX에서 float32로 반올림되며 경계 값이 반대 가지로 감. 기준값을 "원래 값 이하인 가장 큰 float32"로 내려 변환해 불일치 행 0으로 해결.
3. **MLflow 응답 멈춤** — Windows에서 uvicorn 다중 워커 사용 시 발생 → `--workers 1`.
4. **패턴 지수 요약** — 0값이 많은 냉수·증기·온수는 중앙값이 0에 붙거나 하루 패턴을 묻으므로 평균, 전기는 중앙값이 적합.

## 7. 알려진 이슈 / 다음 단계

- `building_level`은 학습 데이터로 만든 피처라 학습 시 과신 가능 (교차검증에서는 학습 구간만으로 계산해야 함).
- 1등 솔루션 참고 후보: 건물·미터·시간대별 / 요일별 평균 사용량 피처, 기온 7h·73h 이동 평균·표준편차, site 0 전기 단위 보정(×0.2931), 공휴일, 1등의 고정 시차와 비교, `month` 제외 실험, 카테고리 피처, 앙상블.
- 냉수·온수: 켜짐/꺼짐 분류 + 사용량 회귀의 2단계 모델.

## 회고

- **검증 설계가 결과 해석을 좌우했다.** 1~8월/9~12월 분할에서 0 구간까지 지운 검증셋은 0.79라는 좋은 숫자를 줬지만 Kaggle에서는 1.46이었다. 계절 운휴 0을 "이상값"으로 보고 지운 것도 같은 착시에서 나왔다. 처음부터 원본 검증셋과 월 단위 교차검증(1등 방식)을 함께 봤어야 했다.
- **데이터를 먼저 본 것이 가장 효과적이었다.** 하이퍼파라미터 튜닝을 늘리는 것보다, 기온 최고 시각으로 날씨 시간대 어긋남을 찾아낸 것과 0값의 계절성을 확인한 것이 훨씬 큰 개선을 줬다.
- **한 번에 하나씩 바꿔야 효과를 말할 수 있다.** 초반에는 여러 변경을 묶어 적용해 무엇이 효과였는지 분리하지 못했다. Kaggle 제출 단계에서 한 가지씩 바꿔 비교한 뒤에야 ID 피처와 `floor_count`의 효과를 판단할 수 있었다.
- **도구 문제에 시간이 많이 들었다.** MLflow 다중 워커 멈춤, cp949 인코딩, ONNX float32 반올림, 폴더를 잡고 있는 프로세스 등은 모델과 무관하지만 매번 실험을 끊었다. 실행 환경을 처음에 단단히 해 두는 것이 실험 속도를 결정한다.
- **상위권과의 차이는 피처와 앙상블이다.** 1등은 건물·시간대별 평소 패턴, 기온 이동 통계, 이상 검침 제거, 여러 분할 기준의 모델 앙상블을 썼다. 유출 데이터를 빼더라도 이 부분에서 아직 개선 여지가 크다.
