"""ASHRAE Great Energy Predictor III — LightGBM + Optuna 튜닝 + MLflow 기록/검증.

흐름
  1. load_dataset   : PostgreSQL(train ⨝ building_metadata ⨝ weather_train) → parquet 캐시
  2. split          : 시간 기준 분할 (1~8월 학습 / 9~12월 검증), target = log1p(meter_reading)
  3. tune           : 샘플 데이터로 Optuna(TPE + MedianPruner), trial 별 MLflow nested run
  4. fit_final      : 최적 파라미터로 전체 데이터 재학습 → RMSLE (전체/미터별)
  5. register       : MLflow Model Registry 등록 + ONNX/메타 artifact 업로드
  6. validate       : 레지스트리에서 모델을 다시 로드해 재채점 + ONNX 추론 결과 일치 검증

평가 지표: RMSLE = RMSE(log1p(y), log1p(ŷ))  (대회 공식 지표)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import onnxruntime as ort
import optuna
import pandas as pd
import psycopg
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from app.ml import MODEL_DIR, ONNX_OPSET, _export_onnx

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "ashrae" / "train_features.parquet"
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_ASHRAE", "ashrae-lgbm-optuna")
REGISTERED_MODEL = os.getenv("MLFLOW_REGISTERED_MODEL", "ashrae-lgbm")

TARGET = "meter_reading"
WEATHER_COLS = [
    "air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr",
    "sea_level_pressure", "wind_direction", "wind_speed",
]
LINEAR_WEATHER = [c for c in WEATHER_COLS if c != "wind_direction"]
# 시간별 날씨 피처. 풍향은 원형 값이라 각도 대신 단위벡터 성분(sin, cos)으로 넣는다
WEATHER_FEATURES = LINEAR_WEATHER + ["wind_dir_sin", "wind_dir_cos"]
FEATURES = [
    "building_id", "meter", "site_id", "primary_use", "square_feet", "year_built", "floor_count",
    "hour", "day_of_week", "is_weekend",
    *WEATHER_FEATURES,
]
VALID_FROM_MONTH = 9  # 9~12월 = 검증
TRAIN_YEAR = 2016

# site 0 의 전기(meter 0) 값은 2016-05-20 까지 0 으로 기록된 알려진 결측 → 제외
DATASET_SQL = """
SELECT t.building_id, t.meter, b.site_id, b.primary_use, b.square_feet, b.year_built, b.floor_count,
       t.month, t.day, t.hour, t.day_of_week,
       w.air_temperature, w.cloud_coverage, w.dew_temperature, w.precip_depth_1_hr,
       w.sea_level_pressure, w.wind_direction, w.wind_speed,
       t.meter_reading
FROM ashrae.train t
JOIN ashrae.building_metadata b USING (building_id)
LEFT JOIN ashrae.weather_train w
       ON w.site_id = b.site_id AND w.year = t.year AND w.month = t.month
      AND w.day = t.day AND w.hour = t.hour
WHERE NOT (b.site_id = 0 AND t.meter = 0 AND make_date(t.year, t.month, t.day) < DATE '2016-05-21')
"""

BASE_PARAMS = {
    "objective": "regression",
    "n_estimators": 3000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ---------------------------------------------------------------- 1. 데이터
def primary_use_codes(conn) -> dict[str, int]:
    rows = conn.execute("SELECT DISTINCT primary_use FROM ashrae.building_metadata ORDER BY 1").fetchall()
    return {r[0]: i for i, r in enumerate(rows)}


def load_dataset(refresh: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    with psycopg.connect(dbname=os.getenv("PGDATABASE", "ashrae")) as conn:
        codes = primary_use_codes(conn)
        if CACHE.exists() and not refresh:
            log.info("load cache %s", CACHE)
            return pd.read_parquet(CACHE), codes

        t0 = time.time()
        csv_path = CACHE.with_suffix(".csv")
        with conn.cursor() as cur, open(csv_path, "wb") as f:
            with cur.copy(f"COPY ({DATASET_SQL}) TO STDOUT WITH (FORMAT csv, HEADER true)") as copy:
                for chunk in copy:
                    f.write(chunk)
    log.info("export %.0fs", time.time() - t0)

    df = pd.read_csv(csv_path, dtype={c: "float32" for c in [
        "square_feet", "year_built", "floor_count", "air_temperature", "cloud_coverage",
        "dew_temperature", "precip_depth_1_hr", "sea_level_pressure", "wind_direction",
        "wind_speed", "meter_reading"]})
    df["primary_use"] = df["primary_use"].map(codes).astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 6).astype("int8")
    df["square_feet"] = np.log1p(df["square_feet"]).astype("float32")  # 스케일 완화
    for c in ["building_id", "meter", "site_id", "month", "day", "hour", "day_of_week"]:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    df.to_parquet(CACHE, index=False)
    csv_path.unlink()
    log.info("dataset %s rows, cached → %s", f"{len(df):,}", CACHE)
    return df, codes


def _fill_from_adjacent_hours(w: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    """결측을 같은 사이트의 '정확히 1시간 전'과 '1시간 후' 값으로 채운다 (in-place).
    둘 다 있으면 평균, 한쪽만 있으면 그 값, 둘 다 없으면 NaN 유지.
    w 는 사이트별로 빈 시간이 없는 1시간 간격 격자여야 한다."""
    filled = {}
    g = w.groupby("site_id")[cols]
    prev, nxt = g.shift(1), g.shift(-1)          # 격자이므로 shift 1 = 정확히 1시간
    for c in cols:
        before = int(w[c].isna().sum())
        w[c] = w[c].fillna(pd.concat([prev[c], nxt[c]], axis=1).mean(axis=1))   # NaN 은 평균에서 제외
        filled[c] = before - int(w[c].isna().sum())
    return filled


# 날씨는 UTC, 사용량은 현지 시각으로 기록돼 있어 사이트별로 날씨 시각을 현지 시각으로 바꿔야 한다.
# 표준 시차: 겨울(12~2월, 서머타임 없음) 하루 기온 곡선의 위상을 UTC+0 사이트(1·5·12)와 비교해 추정.
# 서머타임: 사용량 시각이 서머타임을 따르는지 학교·사무실 평일 전기의 아침 상승 시각(겨울 vs 여름)으로 확인.
#   대부분 사이트는 서머타임을 따름(차이 ≈ 0) → 해당 지역 규칙 적용.
#   site 14 는 여름이 1시간 이름(-0.96, 건물 76개) → 사용량이 표준시 → 서머타임 없이 고정 시차.
# 지역명은 같은 시차·서머타임 규칙을 갖는 대표 시간대일 뿐, 실제 위치를 뜻하지 않는다.
SITE_TIMEZONE = {
    0: "America/New_York",     # UTC-5
    1: "Europe/London",        # UTC+0
    2: "America/Los_Angeles",  # UTC-8
    3: "America/Chicago",      # UTC-6
    4: "America/Los_Angeles",  # UTC-8
    5: "Europe/London",        # UTC+0
    6: "America/Chicago",      # UTC-6
    7: "America/Chicago",      # UTC-6
    8: "America/New_York",     # UTC-5
    9: "America/Denver",       # UTC-7
    10: "America/Los_Angeles", # UTC-8
    11: "America/Chicago",     # UTC-6
    12: "Europe/London",       # UTC+0
    13: "America/Los_Angeles", # UTC-8
    14: "Etc/GMT+6",           # UTC-6 고정 (서머타임 없음)
    15: "America/New_York",    # UTC-5
}


def _to_local_time(w: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """UTC 격자(w.ts)를 사이트별 현지 시각으로 바꾼다.
    - 서머타임 종료일: 같은 현지 시각이 2번 생김 → 두 값 평균
    - 서머타임 시작일: 현지 시각 1시간이 비어 있음 → 1시간 전·후 값으로 채움"""
    parts = []
    for s, g in w.groupby("site_id", sort=True):
        local = g["ts"].dt.tz_localize("UTC").dt.tz_convert(SITE_TIMEZONE[int(s)]).dt.tz_localize(None)
        parts.append(g.assign(ts=local))
    w = pd.concat(parts, ignore_index=True)
    stats["tz_duplicate_hours_merged"] = int(w.duplicated(["site_id", "ts"]).sum())
    w = w.groupby(["site_id", "ts"], as_index=False)[WEATHER_FEATURES].mean()

    hours = pd.date_range(w.ts.min(), w.ts.max(), freq="h")
    grid = pd.MultiIndex.from_product([sorted(w.site_id.unique()), hours], names=["site_id", "ts"])
    w = w.set_index(["site_id", "ts"])
    new_row = ~grid.isin(w.index)          # 변환으로 새로 생긴 빈 시간 (서머타임 시작일 등)만 채운다
    w = w.reindex(grid).reset_index()
    g = w.groupby("site_id")[WEATHER_FEATURES]
    neighbor = (g.shift(1) + g.shift(-1)) / 2
    neighbor = neighbor.fillna(g.shift(1)).fillna(g.shift(-1))   # 한쪽만 있으면 그 값
    before = int(w.loc[new_row, WEATHER_FEATURES].isna().sum().sum())
    w.loc[new_row, WEATHER_FEATURES] = w.loc[new_row, WEATHER_FEATURES].fillna(neighbor[new_row])
    stats["tz_new_hours"] = int(new_row.sum())
    stats["tz_gap_cells_filled"] = before - int(w.loc[new_row, WEATHER_FEATURES].isna().sum().sum())
    return w


def load_weather_hourly(tables: tuple[str, ...] = ("ashrae.weather_train", "ashrae.weather_test"),
                        align_tz: bool = True) -> tuple[pd.DataFrame, dict]:
    """시간별 날씨 정리 (학습/예측 공통). 원본 테이블은 바꾸지 않고 메모리에서만 처리한다.
    1. 사이트별로 1시간 간격 격자를 만들어 기록 자체가 없는 시간도 결측 행으로 만든다
    2. 강수: -1(미량) → 0
    3. 결측: 1시간 전·후 값 평균 (한쪽만 있으면 그 값, 둘 다 없으면 NaN)
    4. 그래도 남은 결측은 강수 포함 모두 NaN 그대로
    5. 풍향: sin/cos 로 변환 (무풍 = 풍속 0 → (0, 0)), sin/cos 결측도 3번과 같은 방식
    6. (align_tz) UTC → 사이트별 현지 시각 (표준 시차 + 서머타임)
    train/test 날씨를 이어 붙여 처리하므로 연말·연초 경계에서도 현지 시각 변환으로 빈 시간이 생기지 않는다."""
    with psycopg.connect(dbname=os.getenv("PGDATABASE", "ashrae")) as conn:
        sql = " UNION ALL ".join(
            f"SELECT site_id, year, month, day, hour, {', '.join(WEATHER_COLS)} FROM {t}" for t in tables)
        cur = conn.execute(sql)
        w = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    w[WEATHER_COLS] = w[WEATHER_COLS].astype("float64")
    w["ts"] = pd.to_datetime(dict(year=w.year, month=w.month, day=w.day, hour=w.hour))

    # 1) 1시간 격자 (테이블 전체 기간 x 전체 사이트)
    hours = pd.date_range(w.ts.min().normalize(), w.ts.max().normalize() + pd.Timedelta(hours=23), freq="h")
    grid = pd.MultiIndex.from_product([sorted(w.site_id.unique()), hours], names=["site_id", "ts"])
    w = w.drop(columns=["year", "month", "day", "hour"]).set_index(["site_id", "ts"]).reindex(grid).reset_index()
    stats = {"weather_grid_rows": len(w),
             "weather_missing_hours_added": int(w[WEATHER_COLS].isna().all(axis=1).sum())}

    # 2) 강수 -1 → 0
    stats["precip_minus1_to_0"] = int((w["precip_depth_1_hr"] == -1).sum())
    w.loc[w["precip_depth_1_hr"] == -1, "precip_depth_1_hr"] = 0

    # 5) 풍향 → sin/cos (원본 기준 무풍이면 방향 없음 = (0, 0))
    calm = w["wind_speed"].eq(0)
    rad = np.deg2rad(w["wind_direction"])
    w["wind_dir_sin"], w["wind_dir_cos"] = np.sin(rad), np.cos(rad)
    w.loc[calm, ["wind_dir_sin", "wind_dir_cos"]] = 0.0

    # 3) 1시간 전·후로 채우기
    stats["missing_before"] = {c: int(w[c].isna().sum()) for c in WEATHER_FEATURES}
    stats["filled_adjacent"] = _fill_from_adjacent_hours(w, WEATHER_FEATURES)

    # 4) 1시간 전·후로 못 채운 결측은 강수 포함 모두 NaN 그대로 (LightGBM 결측 처리)
    stats["missing_after"] = {c: int(w[c].isna().sum()) for c in WEATHER_FEATURES}

    # 6) UTC → 현지 시각
    if align_tz:
        w = _to_local_time(w, stats)

    w["year"], w["month"], w["day"], w["hour"] = w.ts.dt.year, w.ts.dt.month, w.ts.dt.day, w.ts.dt.hour
    for c in ["site_id", "year", "month", "day", "hour"]:
        w[c] = pd.to_numeric(w[c], downcast="integer")
    out = w[["site_id", "year", "month", "day", "hour", *WEATHER_FEATURES]]
    return out.astype({c: "float32" for c in WEATHER_FEATURES}), stats


EXCLUDE_BUILDINGS = [1099]   # steam 최대 2,190만 — 전체 사용량 합의 78% 를 차지하는 이상 시계열
ZERO_RUN_HOURS = 24          # 이 시간 이상 연속으로 0 인 구간은 검침 누락으로 보고 제거

PREPROCESS = {
    "target": "log1p(meter_reading)",
    "square_feet": "log1p (v1~v2 방식)",
    "exclude_buildings": str(EXCLUDE_BUILDINGS),
    "zero_runs": f">= {ZERO_RUN_HOURS}h 연속 0 구간 제거 (building, meter 별)",
    "year_built": "결측 그대로",
    "floor_count": "피처 포함, 결측 그대로",
    "weather": "시간별 값 사용 (하루 평균 없음). 사이트별 1시간 격자로 빈 시간도 결측 처리",
    "weather_missing": "1시간 전·후 값 평균, 한쪽만 있으면 그 값, 둘 다 없으면 NaN",
    "precip": "-1(미량) → 0, 1시간 전·후로 못 채운 결측은 NaN 그대로",
    "wind_direction": "sin/cos 2개 피처, 무풍(풍속 0) → (0, 0), 결측은 1시간 전·후 평균",
    "timezone": "날씨 UTC → 사이트별 현지 시각 (표준 시차 + 서머타임, site 14 는 서머타임 없음). 학습 시에만 변환",
}


# ---- 학습 기간 전용 정답값 정리 (TRAIN_CLEAN=True 일 때). 검증 기간(9~12월)은 기존 규칙만 적용.
TRAIN_CLEAN = False
REMOVE_ZERO_RUNS = True  # False: 24h 이상 0 구간도 지우지 않음 (계절 운휴 0 을 그대로 학습)
SQFT_LOG = True          # False: square_feet 원본 값 사용 (캐시의 log1p 를 되돌림)
STUCK_HOURS = 48         # 0 이 아닌 같은 값이 이 시간 이상 이어지면 고착(미터 멈춤)으로 보고 제거
SPIKE_FACTOR = 10        # 앞뒤 24시간 중앙값보다 이 배수 이상 '위로' 튄 값 제거
# 24h 이상 0 구간이라도 계절 운휴로 보이면 학습에 남긴다: (구간 평균기온 조건, 기준 ℃)
SEASONAL_ZERO = {1: ("<=", 10.0),   # 냉수: 추운 날 꺼짐
                 2: (">=", 20.0),   # 증기: 더운 날 꺼짐
                 3: (">=", 20.0)}   # 온수: 더운 날 꺼짐


def _attach_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """캐시의 원본 날씨를 버리고 정리된 시간별 날씨(현지 시각)로 교체."""
    df = df.drop(columns=WEATHER_COLS)
    if "year" not in df.columns:
        df["year"] = TRAIN_YEAR                 # 캐시(train)에는 연도 컬럼이 없음 — train 은 2016년뿐
    return df.merge(weather, on=["site_id", "year", "month", "day", "hour"], how="left")


def raw_valid(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """정답값 정리를 전혀 하지 않은 검증 기간 데이터 (건물 1099·0 구간 포함). 실제 상황 기준 평가용."""
    return _attach_weather(df[df["month"] >= VALID_FROM_MONTH], weather).reset_index(drop=True)


def _run_lengths(flag: np.ndarray, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """flag 값이 같은 연속 구간의 (구간 id, 구간 길이) — series 경계에서 끊는다."""
    new = np.ones(len(flag), dtype=bool)
    new[1:] = (flag[1:] != flag[:-1]) | (series[1:] != series[:-1])
    rid = np.cumsum(new)
    return rid, np.bincount(rid)[rid]


def preprocess(df: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """캐시(parquet) 로드 후 적용하는 전처리.
    공통: 건물 1099 제외, 24h 이상 0 구간 제거.
    TRAIN_CLEAN 이면 학습 기간에만: 계절 운휴 0 구간은 남기고, 값 고착·위로 튄 값은 제거.
    weather: load_weather_hourly() 결과 (정리된 시간별 날씨)."""
    stats = {"rows_before": len(df)}

    # square_feet: 캐시에 log1p 로 저장된 값을 그대로 사용 (v1~v2 방식)
    # year_built / floor_count 결측은 NaN 그대로 (ONNX 입력에서도 NaN 으로 전달)
    df = _attach_weather(df, weather)
    stats["weather_missing_cells"] = int(df[WEATHER_FEATURES].isna().sum().sum())
    if not SQFT_LOG:   # 캐시에는 log1p 로 저장돼 있음 → 원본 정수로 복원
        df["square_feet"] = np.round(np.expm1(df["square_feet"].astype("float64"))).astype("float32")

    is_excl = df["building_id"].isin(EXCLUDE_BUILDINGS)
    stats["rows_removed_building"] = int(is_excl.sum())
    df = df[~is_excl]

    # (building, meter) 별 시간순 정렬 후, 연속된 0 구간의 길이를 계산
    df = df.sort_values(["building_id", "meter", "month", "day", "hour"], kind="stable").reset_index(drop=True)
    y = df["meter_reading"].to_numpy()
    zero = y == 0
    series = (df["building_id"].astype("int64") * 10 + df["meter"]).to_numpy()
    run_id, run_len = _run_lengths(zero, series)
    drop = zero & (run_len >= ZERO_RUN_HOURS) if REMOVE_ZERO_RUNS else np.zeros(len(df), dtype=bool)

    if TRAIN_CLEAN:
        is_tr = (df["month"] < VALID_FROM_MONTH).to_numpy()
        meter = df["meter"].to_numpy()
        # 1) 계절 운휴 0 구간은 학습에 남김
        run_temp = pd.Series(df["air_temperature"].to_numpy()).groupby(run_id).transform("mean").to_numpy()
        seasonal = np.zeros(len(df), dtype=bool)
        for m, (op, th) in SEASONAL_ZERO.items():
            cond = run_temp <= th if op == "<=" else run_temp >= th
            seasonal |= (meter == m) & cond
        keep = drop & is_tr & seasonal
        stats["train_rows_kept_seasonal_zero"] = int(keep.sum())
        drop &= ~keep
        # 2) 값 고착 (0 이 아닌 같은 값이 STUCK_HOURS 이상)
        _, same_len = _run_lengths(y, series)
        stuck = is_tr & (y > 0) & (same_len >= STUCK_HOURS)
        stats["train_rows_removed_stuck"] = int((stuck & ~drop).sum())
        # 3) 위로 튄 값 (앞뒤 24시간 중앙값의 SPIKE_FACTOR 배 이상)
        ly = pd.Series(np.log1p(y))
        med = ly.groupby(series).transform(lambda s: s.rolling(49, center=True, min_periods=12).median()).to_numpy()
        spike = is_tr & ((ly.to_numpy() - med) > np.log(SPIKE_FACTOR))
        stats["train_rows_removed_spike"] = int((spike & ~drop & ~stuck).sum())
        drop |= stuck | spike

    stats["rows_removed_zero_runs"] = int((drop & zero).sum())
    stats["rows_removed_total_target"] = int(drop.sum())
    df = df[~drop].reset_index(drop=True)

    stats["rows_after"] = len(df)
    log.info("preprocess %s", stats)
    return df, stats


def split(df: pd.DataFrame):
    is_valid = df["month"] >= VALID_FROM_MONTH
    tr, va = df[~is_valid], df[is_valid]
    to_xy = lambda d: (d[FEATURES].astype("float32"), np.log1p(d[TARGET]).astype("float32"))
    return (*to_xy(tr), *to_xy(va))


def rmsle_log(y_log, pred_log) -> float:
    """log1p 공간에서의 RMSE == RMSLE."""
    return float(np.sqrt(np.mean((np.asarray(y_log) - np.clip(pred_log, 0, None)) ** 2)))


# ---------------------------------------------------------------- 3. 튜닝
def _pruning_callback(trial: optuna.Trial, every: int = 25):
    def _cb(env: lgb.callback.CallbackEnv):
        if env.iteration % every:
            return
        score = env.evaluation_result_list[0][2]
        trial.report(score, env.iteration)
        if trial.should_prune():
            raise optuna.TrialPruned(f"pruned at iteration {env.iteration}")
    return _cb


def _suggest_params(trial: optuna.Trial) -> dict:
    """Optuna 탐색 공간 (trial 학습은 트리 상한 2,000 + early stopping)."""
    return {
        **BASE_PARAMS,
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 511, log=True),
        "max_depth": trial.suggest_int("max_depth", 6, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _new_study() -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100),
    )


def tune(X_tr, y_tr, X_va, y_va, n_trials: int, timeout: int | None, parent_run_id: str,
         extra_eval: tuple | None = None) -> optuna.Study:
    """extra_eval=(X, y): 끝까지 학습한 trial 을 추가 데이터(예: 원본 검증셋)로도 채점해 기록 (선택에는 안 씀)."""
    best: list[float] = []

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
            mlflow.log_params(trial.params)
            t0 = time.time()
            model = lgb.LGBMRegressor(**params)
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
                          callbacks=[lgb.early_stopping(50, verbose=False), _pruning_callback(trial)])
            except optuna.TrialPruned:
                mlflow.set_tag("state", "PRUNED")
                mlflow.log_metric("fit_seconds", time.time() - t0)
                raise
            score = rmsle_log(y_va, model.predict(X_va))
            trial.set_user_attr("best_iteration", int(model.best_iteration_))
            extra = {}
            if extra_eval is not None:
                raw_score = rmsle_log(extra_eval[1], model.predict(extra_eval[0]))
                trial.set_user_attr("raw_rmsle", raw_score)
                extra["val_rmsle_raw"] = raw_score
            mlflow.set_tag("state", "COMPLETE")
            mlflow.log_metrics({"val_rmsle": score, "best_iteration": model.best_iteration_,
                                "fit_seconds": time.time() - t0, **extra})

        best.append(min(best[-1], score) if best else score)
        mlflow.log_metrics({"trial_rmsle": score, "best_rmsle": best[-1]},
                           step=trial.number, run_id=parent_run_id)
        log.info("trial %d rmsle=%.5f best=%.5f", trial.number, score, best[-1])
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100),
    )
    # MLflow 통신 실패 등으로 trial 하나가 실패해도 study 는 계속 진행
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(MlflowException,))
    return study


# ---------------------------------------------------------------- 6. 검증
def validate(model_uri: str, onnx_file: Path, X_va, y_va, expected_rmsle: float, n: int = 200_000) -> dict:
    """레지스트리 모델 재로드 → 재채점, ONNX 추론과 LightGBM 결과 비교."""
    idx = np.random.default_rng(0).choice(len(X_va), size=min(n, len(X_va)), replace=False)
    Xs, ys = X_va.iloc[idx], y_va.iloc[idx]

    loaded = mlflow.lightgbm.load_model(model_uri)
    pred_reg = loaded.predict(Xs)
    sess = ort.InferenceSession(str(onnx_file))
    pred_onnx = sess.run(None, {"input": Xs.to_numpy(np.float32)})[0].ravel()

    full_rmsle = rmsle_log(y_va, loaded.predict(X_va))
    return {
        "check_registry_val_rmsle": full_rmsle,
        "check_registry_rmsle_diff": abs(full_rmsle - expected_rmsle),
        "check_onnx_max_abs_diff": float(np.max(np.abs(pred_onnx - pred_reg))),
        "check_onnx_sample_rmsle": rmsle_log(ys, pred_onnx),
    }


# ---------------------------------------------------------------- 전체 파이프라인
def run(n_trials: int = 30, timeout: int | None = None, tune_frac: float = 0.1,
        refresh_data: bool = False) -> dict:
    df, codes = load_dataset(refresh_data)
    weather, weather_stats = load_weather_hourly()
    log.info("weather %s", weather_stats)
    df, prep_stats = preprocess(df, weather)
    X_tr, y_tr, X_va, y_va = split(df)
    del df
    log.info("train %s / valid %s rows", f"{len(X_tr):,}", f"{len(X_va):,}")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="optuna-ashrae") as parent:
        pid = parent.info.run_id
        mlflow.log_params({
            "n_trials": n_trials, "timeout": timeout, "tune_frac": tune_frac,
            "valid_from_month": VALID_FROM_MONTH, "n_train": len(X_tr), "n_valid": len(X_va),
            "n_features": len(FEATURES), "target": "log1p(meter_reading)", "metric": "RMSLE",
        })
        mlflow.log_params({f"prep_{k}": v for k, v in PREPROCESS.items()})
        mlflow.log_metrics({f"prep_{k}": v for k, v in prep_stats.items()})
        mlflow.log_dict({"features": FEATURES, "primary_use_codes": codes, "preprocess": PREPROCESS},
                        "dataset/features.json")

        # 3. 샘플로 튜닝
        rng = np.random.default_rng(42)
        tr_i = rng.choice(len(X_tr), int(len(X_tr) * tune_frac), replace=False)
        va_i = rng.choice(len(X_va), int(len(X_va) * tune_frac), replace=False)
        t0 = time.time()
        study = tune(X_tr.iloc[tr_i], y_tr.iloc[tr_i], X_va.iloc[va_i], y_va.iloc[va_i],
                     n_trials, timeout, pid)
        states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({"tune_best_rmsle": study.best_value, "best_trial": study.best_trial.number,
                            "tune_seconds": time.time() - t0,
                            **{f"trials_{k.lower()}": v for k, v in states.items()}})
        log.info("tuning done: best %.5f (%s)", study.best_value, states)

        # 4. 전체 데이터 재학습
        params = {**BASE_PARAMS, **study.best_params, "subsample_freq": 1}
        t0 = time.time()
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
        pred = model.predict(X_va)
        final = rmsle_log(y_va, pred)
        per_meter = {f"val_rmsle_meter{m}": rmsle_log(y_va[X_va.meter == m], pred[X_va.meter == m])
                     for m in sorted(X_va.meter.unique().astype(int))}
        mlflow.log_metrics({"val_rmsle": final, "final_best_iteration": model.best_iteration_,
                            "final_fit_seconds": time.time() - t0, **per_meter})
        log.info("final rmsle %.5f  (iter %d)", final, model.best_iteration_)

        imp = pd.DataFrame({"feature": FEATURES, "gain": model.booster_.feature_importance("gain")}) \
            .sort_values("gain", ascending=False)
        mlflow.log_table(imp, "feature_importance.json")

        # 5. 레지스트리 등록 + ONNX
        sig = infer_signature(X_va.head(100), model.predict(X_va.head(100)))
        info = mlflow.lightgbm.log_model(model, name="model", signature=sig,
                                         input_example=X_va.head(5),
                                         registered_model_name=REGISTERED_MODEL)
        model_id = f"ashrae-{pid[:8]}"
        onnx_file = MODEL_DIR / f"{model_id}.onnx"
        onnx_ok = _export_onnx(model, len(FEATURES), onnx_file)
        meta = {"model_id": model_id, "task": "regression", "features": FEATURES,
                "primary_use_codes": codes, "target_transform": "log1p → 예측값에 expm1 적용",
                "square_feet_transform": "log1p", "year_built_missing": "NaN",
                "preprocess": PREPROCESS, "onnx": onnx_ok, "onnx_input": "input",
                "onnx_opset": ONNX_OPSET, "mlflow_run_id": pid,
                "registered_model": REGISTERED_MODEL, "registered_version": info.registered_model_version}
        meta_file = MODEL_DIR / f"{model_id}.json"
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(meta_file), "onnx")
        if onnx_ok:
            mlflow.log_artifact(str(onnx_file), "onnx")
        mlflow.set_tag("model_id", model_id)

        # 6. MLflow 에서 다시 받아서 검증 (레지스트리 모델 + 업로드된 ONNX)
        reg_uri = f"models:/{REGISTERED_MODEL}/{info.registered_model_version}"
        with tempfile.TemporaryDirectory() as tmp:
            dl = Path(mlflow.artifacts.download_artifacts(run_id=pid, artifact_path=f"onnx/{onnx_file.name}",
                                                          dst_path=tmp))
            checks = validate(reg_uri, dl, X_va, y_va, final)
        mlflow.log_metrics(checks)
        passed = checks["check_registry_rmsle_diff"] < 1e-6 and checks["check_onnx_max_abs_diff"] < 1e-3
        mlflow.set_tag("validation", "PASSED" if passed else "FAILED")
        log.info("validation %s %s", "PASSED" if passed else "FAILED", checks)

    return {"mlflow_run_id": pid, "model_id": model_id, "registered_model": reg_uri,
            "tune_best_rmsle": study.best_value, "val_rmsle": final, **per_meter,
            "best_params": study.best_params, "trials": states, "checks": checks,
            "validation": "PASSED" if passed else "FAILED"}


# ---------------------------------------------------------------- 미터별 모델
METER_NAMES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}
SAVE_ARTIFACTS = True   # False: 튜닝 → 학습만 (모델 등록·ONNX·파일 업로드·검증 생략, 지표/파라미터만 MLflow 기록)
TUNE_ONLY = False       # True: 튜닝까지만 (최종 학습·저장 생략)
METER_FEATURES = [f for f in FEATURES if f != "meter"]   # 미터별 모델에서 meter 는 상수라 제외


def _train_one_meter(m: int, X_tr, y_tr, X_va, y_va, n_trials: int, timeout: int | None,
                     tune_frac: float, codes: dict, raw_va: tuple | None = None) -> dict:
    """미터 하나: 튜닝 → 전체 재학습 → 레지스트리 등록 + ONNX → 검증. 현재 활성 run 아래 nested run 으로 기록."""
    name = METER_NAMES[m]
    reg_name = f"{REGISTERED_MODEL}-{name}"
    with mlflow.start_run(run_name=f"meter-{m}-{name}", nested=True) as mrun:
        rid = mrun.info.run_id
        mlflow.set_tag("meter", name)
        mlflow.log_params({"meter": m, "n_trials": n_trials, "timeout": timeout, "tune_frac": tune_frac,
                           "max_trees": BASE_PARAMS["n_estimators"],
                           "n_train": len(X_tr), "n_valid": len(X_va), "n_features": len(METER_FEATURES)})

        rng = np.random.default_rng(42)
        tr_i = rng.choice(len(X_tr), int(len(X_tr) * tune_frac), replace=False)
        va_i = rng.choice(len(X_va), int(len(X_va) * tune_frac), replace=False)
        extra = None
        if raw_va is not None:   # 원본(정리 안 한) 검증셋에서도 같은 비율로 샘플
            raw_i = rng.choice(len(raw_va[0]), int(len(raw_va[0]) * tune_frac), replace=False)
            extra = (raw_va[0].iloc[raw_i], raw_va[1][raw_i])
        t0 = time.time()
        study = tune(X_tr.iloc[tr_i], y_tr.iloc[tr_i], X_va.iloc[va_i], y_va.iloc[va_i], n_trials, timeout, rid,
                     extra_eval=extra)
        states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
        raw_best = study.best_trial.user_attrs.get("raw_rmsle")
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({"tune_best_rmsle": study.best_value, "best_trial": study.best_trial.number,
                            "tune_seconds": time.time() - t0,
                            **({"tune_best_rmsle_raw": raw_best} if raw_best is not None else {}),
                            **{f"trials_{k.lower()}": v for k, v in states.items()}})
        log.info("[%s] tuning done: best %.5f raw %s (%s)", name, study.best_value,
                 f"{raw_best:.5f}" if raw_best is not None else "-", states)
        if TUNE_ONLY:
            mlflow.set_tag("validation", "SKIPPED")
            return {"meter": name, "mlflow_run_id": rid, "tune_best_rmsle": study.best_value,
                    "tune_best_rmsle_raw": raw_best, "best_params": study.best_params, "trials": states,
                    "n_train": len(X_tr), "_pred": None}

        params = {**BASE_PARAMS, **study.best_params, "subsample_freq": 1}
        t0 = time.time()
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
        pred = model.predict(X_va)
        final = rmsle_log(y_va, pred)
        mlflow.log_metrics({"val_rmsle": final, "final_best_iteration": model.best_iteration_,
                            "final_fit_seconds": time.time() - t0})
        log.info("[%s] final rmsle %.5f (iter %d, %.0fs)", name, final, model.best_iteration_, time.time() - t0)
        imp = pd.DataFrame({"feature": METER_FEATURES, "gain": model.booster_.feature_importance("gain")}) \
            .sort_values("gain", ascending=False)
        if not SAVE_ARTIFACTS:
            # 파일 저장소(RustFS) 없이 실행: 지표·파라미터만 기록, 모델 등록/ONNX/검증은 건너뜀
            mlflow.log_metrics({f"gain_{r.feature}": float(r.gain) for r in imp.itertuples()})
            mlflow.set_tag("validation", "SKIPPED")
            return {"meter": name, "mlflow_run_id": rid, "tune_best_rmsle": study.best_value, "val_rmsle": final,
                    "best_iteration": int(model.best_iteration_), "best_params": study.best_params,
                    "trials": states, "validation": "SKIPPED", "_pred": pred}
        mlflow.log_table(imp, "feature_importance.json")

        sig = infer_signature(X_va.head(100), model.predict(X_va.head(100)))
        info = mlflow.lightgbm.log_model(model, name="model", signature=sig, input_example=X_va.head(5),
                                         registered_model_name=reg_name)
        model_id = f"ashrae-{name}-{rid[:8]}"
        onnx_file = MODEL_DIR / f"{model_id}.onnx"
        onnx_ok = _export_onnx(model, len(METER_FEATURES), onnx_file)
        meta = {"model_id": model_id, "task": "regression", "meter": m, "meter_name": name,
                "features": METER_FEATURES, "primary_use_codes": codes,
                "target_transform": "log1p → 예측값에 expm1 적용", "square_feet_transform": "log1p",
                "preprocess": PREPROCESS,
                "onnx": onnx_ok, "onnx_input": "input", "onnx_opset": ONNX_OPSET, "mlflow_run_id": rid,
                "registered_model": reg_name, "registered_version": info.registered_model_version}
        meta_file = MODEL_DIR / f"{model_id}.json"
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(meta_file), "onnx")
        if onnx_ok:
            mlflow.log_artifact(str(onnx_file), "onnx")
        mlflow.set_tag("model_id", model_id)

        reg_uri = f"models:/{reg_name}/{info.registered_model_version}"
        with tempfile.TemporaryDirectory() as tmp:
            dl = Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path=f"onnx/{onnx_file.name}",
                                                          dst_path=tmp))
            checks = validate(reg_uri, dl, X_va, y_va, final)
        mlflow.log_metrics(checks)
        passed = checks["check_registry_rmsle_diff"] < 1e-6 and checks["check_onnx_max_abs_diff"] < 1e-3
        mlflow.set_tag("validation", "PASSED" if passed else "FAILED")
        log.info("[%s] validation %s %s", name, "PASSED" if passed else "FAILED", checks)

    return {"meter": name, "mlflow_run_id": rid, "registered_model": reg_uri, "model_id": model_id,
            "tune_best_rmsle": study.best_value, "val_rmsle": final, "best_iteration": int(model.best_iteration_),
            "best_params": study.best_params, "trials": states, "checks": checks,
            "validation": "PASSED" if passed else "FAILED", "_pred": pred}


def run_per_meter(n_trials: int = 10, timeout: int | None = None, tune_frac: float = 0.1,
                  refresh_data: bool = False) -> dict:
    """미터(전기/냉수/증기/온수)별로 모델을 따로 튜닝·학습. 전처리는 run() 과 동일."""
    df, codes = load_dataset(refresh_data)
    weather, weather_stats = load_weather_hourly()
    rv = raw_valid(df, weather)                      # 정답값 정리 안 한 검증셋 (실제 상황 기준 평가용)
    rv_X, rv_y = rv[FEATURES].astype("float32"), np.log1p(rv[TARGET]).astype("float32").to_numpy()
    del rv
    df, prep_stats = preprocess(df, weather)
    X_tr, y_tr, X_va, y_va = split(df)
    del df
    log.info("train %s / valid %s rows (raw valid %s)", f"{len(X_tr):,}", f"{len(X_va):,}", f"{len(rv_X):,}")

    mlflow.set_experiment(EXPERIMENT)
    results, y_all, p_all = [], [], []
    with mlflow.start_run(run_name="per-meter-tune-ashrae" if TUNE_ONLY else "per-meter-ashrae") as parent:
        pid = parent.info.run_id
        mlflow.set_tags({"mode": "per-meter", "train_clean": str(TRAIN_CLEAN), "tune_only": str(TUNE_ONLY)})
        mlflow.log_params({"n_trials_per_meter": n_trials, "timeout_per_meter": timeout, "tune_frac": tune_frac,
                           "max_trees": BASE_PARAMS["n_estimators"],
                           "valid_from_month": VALID_FROM_MONTH, "n_train": len(X_tr), "n_valid": len(X_va),
                           "n_features": len(METER_FEATURES), "target": "log1p(meter_reading)", "metric": "RMSLE"})
        mlflow.log_params({f"prep_{k}": v for k, v in PREPROCESS.items()})
        mlflow.log_metrics({f"prep_{k}": v for k, v in prep_stats.items()})
        if SAVE_ARTIFACTS:
            mlflow.log_dict({"features": METER_FEATURES, "primary_use_codes": codes, "preprocess": PREPROCESS},
                            "dataset/features.json")

        for m in sorted(METER_NAMES):
            tr, va = (X_tr.meter == m).to_numpy(), (X_va.meter == m).to_numpy()
            log.info("[%s] train %s / valid %s rows", METER_NAMES[m], f"{tr.sum():,}", f"{va.sum():,}")
            rva = (rv_X.meter == m).to_numpy()
            r = _train_one_meter(m, X_tr.loc[tr, METER_FEATURES], y_tr[tr], X_va.loc[va, METER_FEATURES], y_va[va],
                                 n_trials, timeout, tune_frac, codes,
                                 raw_va=(rv_X.loc[rva, METER_FEATURES], rv_y[rva]))
            pred = r.pop("_pred")
            results.append(r)
            if TUNE_ONLY:
                continue
            y_all.append(y_va[va].to_numpy()); p_all.append(pred)
            mlflow.log_metric(f"val_rmsle_meter{m}", r["val_rmsle"])

        if TUNE_ONLY:
            mlflow.set_tag("validation", "SKIPPED")
            log.info("tune-only done: %s", {r["meter"]: (round(r["tune_best_rmsle"], 5),
                                                         round(r["tune_best_rmsle_raw"], 5)) for r in results})
            return {"mlflow_run_id": pid, "train_clean": TRAIN_CLEAN, "prep": prep_stats, "meters": results}

        overall = rmsle_log(np.concatenate(y_all), np.concatenate(p_all))
        status = ("SKIPPED" if not SAVE_ARTIFACTS
                  else "PASSED" if all(r["validation"] == "PASSED" for r in results) else "FAILED")
        mlflow.log_metric("val_rmsle", overall)
        mlflow.set_tag("validation", status)
        log.info("overall rmsle %.5f  validation %s", overall, status)

    return {"mlflow_run_id": pid, "val_rmsle": overall, "meters": results, "validation": status}


# ---------------------------------------------------------------- 1~12월 학습: 반기 교차검증 + test 예측
SUBMISSION_DIR = CACHE.parent

TEST_SQL = """
SELECT t.row_id, t.building_id, t.meter, b.site_id, b.primary_use, b.square_feet, b.year_built, b.floor_count,
       t.year, t.month, t.day, t.hour, t.day_of_week
FROM ashrae.test t JOIN ashrae.building_metadata b USING (building_id)
WHERE t.meter = {meter} AND t.year = {year}
"""


def _half_year_folds(month: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """반기 교차검증: (1~6월 학습 → 7~12월 평가), (7~12월 학습 → 1~6월 평가)."""
    h1 = month <= 6
    return [(h1, ~h1, "train H1 → valid H2"), (~h1, h1, "train H2 → valid H1")]


def tune_cv(folds: list[tuple], n_trials: int, timeout: int | None, parent_run_id: str) -> optuna.Study:
    """folds: [(X_tr, y_tr, X_va, y_va), ...] (샘플). 목적함수 = 두 fold RMSLE 평균.
    가지치기는 첫 번째 fold 학습 중간 점수로 판단한다."""
    best: list[float] = []

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
            mlflow.log_params(trial.params)
            t0, scores, iters = time.time(), [], []
            try:
                for i, (X_tr, y_tr, X_va, y_va) in enumerate(folds):
                    cbs = [lgb.early_stopping(50, verbose=False)] + ([_pruning_callback(trial)] if i == 0 else [])
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse", callbacks=cbs)
                    scores.append(rmsle_log(y_va, model.predict(X_va)))
                    iters.append(int(model.best_iteration_))
            except optuna.TrialPruned:
                mlflow.set_tag("state", "PRUNED")
                mlflow.log_metric("fit_seconds", time.time() - t0)
                raise
            score = float(np.mean(scores))
            trial.set_user_attr("best_iterations", iters)
            mlflow.set_tag("state", "COMPLETE")
            mlflow.log_metrics({"cv_rmsle": score, **{f"fold{i}_rmsle": s for i, s in enumerate(scores)},
                                "fit_seconds": time.time() - t0})
        best.append(min(best[-1], score) if best else score)
        mlflow.log_metrics({"trial_rmsle": score, "best_rmsle": best[-1]}, step=trial.number, run_id=parent_run_id)
        log.info("trial %d cv_rmsle=%.5f best=%.5f", trial.number, score, best[-1])
        return score

    study = _new_study()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(MlflowException,))
    return study


def _load_test_chunk(meter: int, year: int, codes: dict, weather: pd.DataFrame) -> pd.DataFrame:
    """test 한 덩어리(미터 × 연도)를 학습과 같은 방식으로 피처화."""
    import io
    buf = io.BytesIO()
    with psycopg.connect(dbname=os.getenv("PGDATABASE", "ashrae")) as conn, conn.cursor() as cur:
        with cur.copy(f"COPY ({TEST_SQL.format(meter=meter, year=year)}) TO STDOUT WITH (FORMAT csv, HEADER true)") as cp:
            for chunk in cp:
                buf.write(chunk)
    buf.seek(0)
    t = pd.read_csv(buf, dtype={"square_feet": "float32", "year_built": "float32", "floor_count": "float32"})
    t["primary_use"] = t["primary_use"].map(codes).astype("int16")
    t["is_weekend"] = (t["day_of_week"] >= 6).astype("int8")
    if SQFT_LOG:
        t["square_feet"] = np.log1p(t["square_feet"]).astype("float32")  # 학습 캐시와 동일 (log1p)
    return t.merge(weather, on=["site_id", "year", "month", "day", "hour"], how="left")


def predict_test(models: dict[int, lgb.LGBMRegressor], codes: dict, weather: pd.DataFrame, path: Path) -> dict:
    """test(2017~2018년) 전체 예측 → Kaggle 제출 형식 (row_id, meter_reading) gzip CSV."""
    ids, preds, stats = [], [], {}
    for m, model in models.items():
        for year in (2017, 2018):
            t = _load_test_chunk(m, year, codes, weather)
            p = np.expm1(np.clip(model.predict(t[METER_FEATURES].astype("float32")), 0, None))
            ids.append(t["row_id"].to_numpy(np.int32)); preds.append(p.astype(np.float32))
            stats[f"test_rows_{METER_NAMES[m]}_{year}"] = len(t)
            stats[f"test_weather_nan_pct_{METER_NAMES[m]}_{year}"] = float(t["air_temperature"].isna().mean() * 100)
            log.info("[test] %s %d: %s rows predicted", METER_NAMES[m], year, f"{len(t):,}")
    sub = pd.DataFrame({"row_id": np.concatenate(ids), "meter_reading": np.concatenate(preds)}).sort_values("row_id")
    sub.to_csv(path, index=False, float_format="%.4f", compression="gzip")
    stats["test_rows"] = len(sub)
    stats["test_pred_mean"] = float(sub.meter_reading.mean())
    log.info("submission saved → %s (%s rows)", path, f"{len(sub):,}")
    return stats


def run_cv_per_meter(n_trials: int = 30, timeout: int | None = None, tune_frac: float = 0.1,
                     refresh_data: bool = False, write_submission: bool = True) -> dict:
    """1~12월 전체 학습. 미터별로 반기 교차검증으로 튜닝·평가 → 1~12월 전체로 최종 학습 → test 예측."""
    df, codes = load_dataset(refresh_data)
    weather, weather_stats = load_weather_hourly()
    df, prep_stats = preprocess(df, weather)
    X_all = df[FEATURES].astype("float32")
    y_all = np.log1p(df[TARGET]).astype("float32").to_numpy()
    month = df["month"].to_numpy()
    del df
    log.info("full-year rows %s", f"{len(X_all):,}")

    mlflow.set_experiment(EXPERIMENT)
    results, models = [], {}
    oof = np.full(len(X_all), np.nan, dtype=np.float32)
    with mlflow.start_run(run_name="cv-per-meter-ashrae") as parent:
        pid = parent.info.run_id
        mlflow.set_tags({"mode": "per-meter half-year CV", "validation": "SKIPPED"})
        mlflow.log_params({"n_trials_per_meter": n_trials, "timeout_per_meter": timeout, "tune_frac": tune_frac,
                           "max_trees": BASE_PARAMS["n_estimators"], "cv": "H1(1~6)↔H2(7~12)",
                           "n_rows": len(X_all), "n_features": len(METER_FEATURES),
                           "target": "log1p(meter_reading)", "metric": "RMSLE"})
        mlflow.log_params({f"prep_{k}": v for k, v in PREPROCESS.items()})
        mlflow.log_metrics({f"prep_{k}": v for k, v in prep_stats.items()})

        for m in sorted(METER_NAMES):
            name = METER_NAMES[m]
            idx = np.where((X_all["meter"] == m).to_numpy())[0]
            Xm, ym, mo = X_all.iloc[idx][METER_FEATURES], y_all[idx], month[idx]
            folds = _half_year_folds(mo)
            log.info("[%s] rows %s (H1 %s / H2 %s)", name, f"{len(idx):,}", f"{(mo <= 6).sum():,}", f"{(mo > 6).sum():,}")
            with mlflow.start_run(run_name=f"meter-{m}-{name}", nested=True) as mrun:
                rid = mrun.info.run_id
                mlflow.set_tag("meter", name)
                mlflow.log_params({"meter": m, "n_rows": len(idx), "n_trials": n_trials, "timeout": timeout})

                # 1) 튜닝 (각 반기에서 10% 샘플)
                rng = np.random.default_rng(42)
                sampled = []
                for tr, va, _ in folds:
                    tr_i, va_i = np.where(tr)[0], np.where(va)[0]
                    tr_i = rng.choice(tr_i, int(len(tr_i) * tune_frac), replace=False)
                    va_i = rng.choice(va_i, int(len(va_i) * tune_frac), replace=False)
                    sampled.append((Xm.iloc[tr_i], ym[tr_i], Xm.iloc[va_i], ym[va_i]))
                t0 = time.time()
                study = tune_cv(sampled, n_trials, timeout, rid)
                del sampled
                states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
                mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
                mlflow.log_metrics({"tune_best_cv_rmsle": study.best_value, "best_trial": study.best_trial.number,
                                    "tune_seconds": time.time() - t0,
                                    **{f"trials_{k.lower()}": v for k, v in states.items()}})
                log.info("[%s] tuning done: best cv %.5f (%s)", name, study.best_value, states)

                # 2) 교차검증 점수 (전체 데이터, 반기씩 번갈아)
                params = {**BASE_PARAMS, **study.best_params, "subsample_freq": 1}
                fold_scores, iters = [], []
                for k, (tr, va, label) in enumerate(folds):
                    t0 = time.time()
                    model = lgb.LGBMRegressor(**params)
                    model.fit(Xm[tr], ym[tr], eval_set=[(Xm[va], ym[va])], eval_metric="rmse",
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                    p = model.predict(Xm[va])
                    oof[idx[va]] = p
                    fold_scores.append(rmsle_log(ym[va], p)); iters.append(int(model.best_iteration_))
                    log.info("[%s] fold %d (%s): rmsle %.5f iter %d (%.0fs)", name, k, label, fold_scores[-1],
                             iters[-1], time.time() - t0)
                cv = rmsle_log(ym, oof[idx])
                mlflow.log_metrics({"cv_rmsle": cv, "fold0_rmsle": fold_scores[0], "fold1_rmsle": fold_scores[1],
                                    "fold0_iter": iters[0], "fold1_iter": iters[1]})
                log.info("[%s] CV rmsle %.5f (folds %.5f / %.5f)", name, cv, *fold_scores)

                # 3) 최종 모델: 1~12월 전체, 트리 수 = 교차검증 best_iteration 평균
                n_est = max(int(np.mean(iters)), 50)
                t0 = time.time()
                final = lgb.LGBMRegressor(**{**params, "n_estimators": n_est})
                final.fit(Xm, ym)
                models[m] = final
                mlflow.log_metrics({"final_n_estimators": n_est, "final_fit_seconds": time.time() - t0})
                log.info("[%s] final model on full year: %d trees (%.0fs)", name, n_est, time.time() - t0)

            mlflow.log_metric(f"cv_rmsle_meter{m}", cv)
            results.append({"meter": name, "mlflow_run_id": rid, "tune_best_cv_rmsle": study.best_value,
                            "cv_rmsle": cv, "fold_rmsle": fold_scores, "fold_iters": iters,
                            "final_n_estimators": n_est, "best_params": study.best_params, "trials": states})

        overall = rmsle_log(y_all, oof)
        h1 = month <= 6
        by_half = {"cv_rmsle_valid_H1": rmsle_log(y_all[h1], oof[h1]), "cv_rmsle_valid_H2": rmsle_log(y_all[~h1], oof[~h1])}
        mlflow.log_metrics({"cv_rmsle": overall, **by_half})
        log.info("overall CV rmsle %.5f  %s", overall, by_half)

        sub_stats = {}
        if write_submission:
            path = SUBMISSION_DIR / f"submission_cv_{pid[:8]}.csv.gz"
            sub_stats = predict_test(models, codes, weather, path)
            mlflow.log_metrics(sub_stats)
            mlflow.set_tag("submission_file", str(path))

    return {"mlflow_run_id": pid, "cv_rmsle": overall, **by_half, "meters": results,
            "submission": sub_stats}


# ---------------------------------------------------------------- 2016년 전체 학습 → test 예측 → Kaggle 제출
# 피처: building_id·site_id 대신 건물·미터별 평소 사용 수준(log 평균), square_feet 원본, month 추가.
FULL_FEATURES = [
    "primary_use", "square_feet", "year_built", "floor_count", "building_level",
    "month", "hour", "day_of_week", "is_weekend",
    *WEATHER_FEATURES,
]
# 자체 검증이 없으므로 직전 최고(M3, run b5987d7a)의 미터별 최적 파라미터와 트리 수를 그대로 사용
FULL_PARAMS = {
    0: dict(learning_rate=0.024808977903657107, num_leaves=456, max_depth=14, min_child_samples=258,
            subsample=0.8231905839874377, colsample_bytree=0.5598138475638506,
            reg_alpha=7.313609163967875e-08, reg_lambda=0.0005703949522081839, n_estimators=781),
    1: dict(learning_rate=0.07191133758026799, num_leaves=441, max_depth=14, min_child_samples=54,
            subsample=0.7613531969953997, colsample_bytree=0.48065841407777754,
            reg_alpha=5.020005867243297e-08, reg_lambda=2.0088321117572198e-05, n_estimators=314),
    2: dict(learning_rate=0.0455875904753014, num_leaves=134, max_depth=10, min_child_samples=50,
            subsample=0.8059264473611898, colsample_bytree=0.4836963163912251,
            reg_alpha=4.258943089524393e-06, reg_lambda=1.9826980964985924e-05, n_estimators=771),
    3: dict(learning_rate=0.06020078164728183, num_leaves=52, max_depth=12, min_child_samples=43,
            subsample=0.8646527992085615, colsample_bytree=0.45216095171845483,
            reg_alpha=2.1233271403855056e-05, reg_lambda=0.0007057511496664353, n_estimators=397),
}


def building_levels(raw: pd.DataFrame) -> pd.DataFrame:
    """건물·미터별 평소 사용 수준 = 2016년 log1p(meter_reading) 평균. test 예측에도 같은 값을 쓴다."""
    return (raw.assign(_l=np.log1p(raw["meter_reading"]))
               .groupby(["building_id", "meter"], as_index=False)["_l"].mean()
               .rename(columns={"_l": "building_level"}))


def run_full_submit(path: Path | None = None, extra_features: list[str] | None = None, tag: str = "",
                    drop_features: list[str] | None = None) -> dict:
    """2016년 1~12월 전체로 미터별 최종 학습 → test(2017~2018) 예측 → 제출 파일.
    extra_features: 기본 FULL_FEATURES 에 더할 피처 (실험용, 예: ["building_id", "site_id"]).
    drop_features: 기본 FULL_FEATURES 에서 뺄 피처 (실험용, 예: ["floor_count"])."""
    global REMOVE_ZERO_RUNS, SQFT_LOG
    REMOVE_ZERO_RUNS, SQFT_LOG = False, False
    feats = [f for f in [*(extra_features or []), *FULL_FEATURES] if f not in (drop_features or [])]

    raw, codes = load_dataset()
    levels = building_levels(raw)                     # 1099 포함 원본 기준 (test 의 모든 건물에 값이 있도록)
    weather, _ = load_weather_hourly()
    df, prep_stats = preprocess(raw, weather)
    del raw
    df = df.merge(levels, on=["building_id", "meter"], how="left")
    log.info("full-year train rows %s", f"{len(df):,}")

    mlflow.set_experiment(EXPERIMENT)
    models = {}
    with mlflow.start_run(run_name=f"full-2016-submit{('-' + tag) if tag else ''}") as run:
        rid = run.info.run_id
        mlflow.set_tags({"mode": "full-year train + test submission", "validation": "kaggle"})
        mlflow.log_params({"features": ",".join(feats), "n_features": len(feats),
                           "params_from": "M3 b5987d7a", "zero_runs": "kept", "square_feet": "raw",
                           "building_id_site_id": ("kept + building_level" if extra_features and "building_id" in extra_features
                                                  else "dropped → building_level"), "month": "feature", "experiment": tag or "base",
                           "dropped_features": ",".join(drop_features or []) or "none"})
        mlflow.log_metrics({f"prep_{k}": v for k, v in prep_stats.items()})
        for m, name in METER_NAMES.items():
            d = df[df["meter"] == m]
            X, y = d[feats].astype("float32"), np.log1p(d[TARGET]).astype("float32")
            t0 = time.time()
            model = lgb.LGBMRegressor(**{**BASE_PARAMS, **FULL_PARAMS[m], "subsample_freq": 1})
            model.fit(X, y)
            fit_rmsle = rmsle_log(y, model.predict(X))
            mlflow.log_metrics({f"train_rows_{name}": len(d), f"train_fit_rmsle_{name}": fit_rmsle,
                                f"fit_seconds_{name}": time.time() - t0})
            imp = dict(zip(feats, model.booster_.feature_importance("gain")))
            top = sorted(imp.items(), key=lambda kv: -kv[1])[:5]
            log.info("[%s] trained %s rows, %d trees (%.0fs) in-sample rmsle %.4f | top gain %s", name, f"{len(d):,}",
                     FULL_PARAMS[m]["n_estimators"], time.time() - t0, fit_rmsle, [k for k, _ in top])
            models[m] = model
        del df

        path = path or SUBMISSION_DIR / f"submission_full{('_' + tag) if tag else ''}_{rid[:8]}.csv.gz"
        ids, preds = [], []
        for m, model in models.items():
            for year in (2017, 2018):
                t = _load_test_chunk(m, year, codes, weather).merge(levels, on=["building_id", "meter"], how="left")
                p = np.expm1(np.clip(model.predict(t[feats].astype("float32")), 0, None))
                ids.append(t["row_id"].to_numpy(np.int32)); preds.append(p.astype(np.float32))
                log.info("[test] %s %d: %s rows, building_level NaN %d", METER_NAMES[m], year, f"{len(t):,}",
                         int(t["building_level"].isna().sum()))
        sub = pd.DataFrame({"row_id": np.concatenate(ids), "meter_reading": np.concatenate(preds)}).sort_values("row_id")
        sub.to_csv(path, index=False, float_format="%.4f", compression="gzip")
        mlflow.log_metrics({"test_rows": len(sub), "test_pred_mean": float(sub.meter_reading.mean())})
        mlflow.set_tag("submission_file", str(path))
        log.info("submission saved → %s (%s rows)", path, f"{len(sub):,}")
    return {"mlflow_run_id": rid, "submission": str(path), "rows": len(sub)}
