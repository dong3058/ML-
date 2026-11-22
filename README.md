# ML

Kaggle 대회를 기준으로 한 머신러닝 실험 저장소. 프로젝트 폴더 이름은 참고한 Kaggle 대회 이름을 따른다.

| 프로젝트 | 대회 | 내용 | 최고 점수 |
|---|---|---|---|
| [`ashrae-energy-prediction`](ashrae-energy-prediction/README.md) | [ASHRAE – Great Energy Predictor III](https://www.kaggle.com/competitions/ashrae-energy-prediction) | 건물 미터별 시간당 에너지 사용량 예측. LightGBM + Optuna, MLflow, ONNX | Kaggle private 1.458 |
| [`house-prices-advanced-regression-techniques`](house-prices-advanced-regression-techniques/README.md) | [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | 주택 가격 예측. CatBoost + LightGBM + Lasso 블렌딩, 오차 분석, Optuna | Kaggle public 0.11870 |

## 공통 규칙

- 접속 정보·토큰 등 환경변수 값은 저장소에 올리지 않는다. 각 프로젝트의 `.env.example`을 `.env`로 복사해 채운다 (`.env`, `kaggleToken`은 `.gitignore`).
- 프로젝트마다 `README.md`에 시도한 버전, 점수 지표, 데이터 가공 흐름을 기록하고 맨 끝에 회고를 남긴다.
- 데이터, 학습된 모델 파일, 가상환경은 저장소에 올리지 않는다.

## 회고

- 첫 프로젝트(ASHRAE)에서 가장 큰 교훈은 **검증 방식이 실제 평가 조건과 맞아야 한다**는 점이다. 내부 점수가 좋아 보여도 Kaggle 점수와 크게 달랐다.
- 실험 기록(MLflow)과 버전별 정리를 처음부터 남긴 덕분에, 중간에 방향을 바꾸거나 되돌릴 때 근거를 바로 찾을 수 있었다.
- 다음 프로젝트에서는 처음부터 환경 설정(비밀값 분리, 인코딩, 서버 설정)과 대회 방식에 맞는 교차검증을 먼저 갖추고 시작한다.
