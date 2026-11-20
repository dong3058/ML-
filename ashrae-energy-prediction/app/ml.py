"""LightGBM 베이스라인 학습 + Optuna 하이퍼파라미터 튜닝.

- 모든 학습/튜닝은 MLflow에 기록된다 (튜닝: 부모 run + trial별 nested run).
- 학습된 모델은 joblib(파이썬 예측용)과 ONNX(Spring 로컬 추론용)로 함께 저장된다.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# MLflow Tracking Server (scripts/start_mlflow.ps1: PostgreSQL + RustFS S3).
# 서버 없이 로컬 파일로만 쓰려면 MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
EXP_BASELINE = os.getenv("MLFLOW_EXPERIMENT_BASELINE", "lgbm-baseline")
EXP_TUNING = os.getenv("MLFLOW_EXPERIMENT_TUNING", "lgbm-optuna")

ONNX_OPSET = 15

BASE_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbose": -1,
}

EARLY_STOPPING_ROUNDS = 30


def load_data(records: list[dict] | None, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if not records:
        ds = load_breast_cancer(as_frame=True)
        return ds.data, ds.target
    df = pd.DataFrame(records)
    if target not in df.columns:
        raise ValueError(f"target 컬럼 '{target}'이(가) 데이터에 없습니다.")
    return df.drop(columns=[target]), df[target]


def _model(task: str, params: dict) -> lgb.LGBMModel:
    return lgb.LGBMRegressor(**params) if task == "regression" else lgb.LGBMClassifier(**params)


def _score(task: str, model: lgb.LGBMModel, X: pd.DataFrame, y: pd.Series) -> tuple[str, float]:
    if task == "binary":
        return "roc_auc", float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    if task == "multiclass":
        return "accuracy", float(accuracy_score(y, model.predict(X)))
    return "rmse", float(np.sqrt(mean_squared_error(y, model.predict(X))))


def _split(X, y, task: str, test_size: float):
    stratify = y if task != "regression" else None
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)


def _fit(task, params, X_tr, y_tr, X_va, y_va) -> lgb.LGBMModel:
    model = _model(task, params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


_F32_MAX = float(np.finfo(np.float32).max)


def _float32_safe_threshold(t: float) -> float:
    """double 분할 기준값 t 를 't 이하인 가장 큰 float32' 로 내린다.
    입력이 float32 이면 (x <= t) 와 (x <= 결과) 가 모든 x 에서 같아진다.
    (그냥 반올림하면 t 보다 커질 수 있어, 경계값의 행이 ONNX 에서 반대쪽 가지로 감)"""
    if t >= _F32_MAX:
        return _F32_MAX
    if t <= -_F32_MAX:
        return -_F32_MAX
    t32 = np.float32(t)
    if float(t32) > t:
        t32 = np.nextafter(t32, np.float32(-np.inf))
    return float(t32)


def _float32_safe_dump(booster: lgb.Booster) -> dict:
    dump = booster.dump_model()

    def fix(node):
        if "leaf_value" in node:
            return
        if node.get("decision_type") == "<=" and isinstance(node.get("threshold"), (int, float)):
            node["threshold"] = _float32_safe_threshold(float(node["threshold"]))
        fix(node["left_child"])
        fix(node["right_child"])

    for tree in dump["tree_info"]:
        fix(tree["tree_structure"])
    return dump


def _export_onnx(model: lgb.LGBMModel, n_features: int, path: Path) -> bool:
    """LightGBM → ONNX. 입력: float32 [batch, n_features] 'input'.
    출력: 분류 = label, probabilities / 회귀 = variable. 숫자형 피처만 지원.
    ONNX 트리는 기준값을 float32 로 저장하므로, 변환 전에 기준값을 float32 안전값으로 내려
    LightGBM(double) 과 분기 결과가 완전히 같도록 한다."""
    booster = model.booster_
    safe_dump = _float32_safe_dump(booster)
    # onnxmltools 는 booster.is_mock 이 True 면 booster.dump_model() 결과로 변환한다
    booster.is_mock = True
    booster.dump_model = lambda *args, **kwargs: safe_dump
    try:
        onx = convert_lightgbm(
            model,
            initial_types=[("input", FloatTensorType([None, n_features]))],
            target_opset=ONNX_OPSET,
            zipmap=False,  # 확률을 dict 대신 텐서로 → Java 에서 다루기 쉬움
        )
        path.write_bytes(onx.SerializeToString())
        return True
    except Exception:
        log.exception("ONNX export failed")
        return False
    finally:
        del booster.is_mock, booster.dump_model   # 원래 Booster 동작으로 복구


def _save(model: lgb.LGBMModel, features: list[str], task: str) -> tuple[str, Path | None]:
    model_id = uuid.uuid4().hex[:12]
    joblib.dump({"model": model, "features": features, "task": task}, MODEL_DIR / f"{model_id}.joblib")
    onnx_path = MODEL_DIR / f"{model_id}.onnx"
    ok = _export_onnx(model, len(features), onnx_path)
    meta = {"model_id": model_id, "task": task, "features": features,
            "onnx": ok, "onnx_input": "input", "onnx_opset": ONNX_OPSET}
    (MODEL_DIR / f"{model_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_id, onnx_path if ok else None


def _log_artifacts(model_id: str, onnx_path: Path | None) -> None:
    mlflow.set_tag("model_id", model_id)
    mlflow.log_artifact(str(MODEL_DIR / f"{model_id}.json"), artifact_path="model")
    if onnx_path:
        mlflow.log_artifact(str(onnx_path), artifact_path="model")


def train_baseline(X, y, task: str, test_size: float, params: dict | None) -> dict:
    p = {**BASE_PARAMS, **(params or {})}
    X_tr, X_va, y_tr, y_va = _split(X, y, task, test_size)

    mlflow.set_experiment(EXP_BASELINE)
    with mlflow.start_run(run_name=f"baseline-{task}") as run:
        mlflow.log_params({**p, "task": task, "test_size": test_size,
                           "n_rows": len(X), "n_features": X.shape[1]})
        model = _fit(task, p, X_tr, y_tr, X_va, y_va)
        metric, score = _score(task, model, X_va, y_va)
        mlflow.log_metrics({metric: score, "best_iteration": model.best_iteration_ or p["n_estimators"]})
        features = list(X.columns)
        model_id, onnx_path = _save(model, features, task)
        _log_artifacts(model_id, onnx_path)

    return {
        "model_id": model_id, "mlflow_run_id": run.info.run_id,
        "task": task, "metric": metric, "score": score,
        "params": p, "features": features,
    }


def tune(X, y, task: str, test_size: float, n_trials: int, timeout: int | None,
         params: dict | None) -> dict:
    X_tr, X_va, y_tr, y_va = _split(X, y, task, test_size)
    direction = "minimize" if task == "regression" else "maximize"
    better = min if direction == "minimize" else max

    mlflow.set_experiment(EXP_TUNING)
    with mlflow.start_run(run_name=f"optuna-{task}") as parent:
        mlflow.log_params({"task": task, "test_size": test_size, "n_trials": n_trials,
                           "timeout": timeout, "direction": direction,
                           "n_rows": len(X), "n_features": X.shape[1]})
        best_so_far: list[float] = []

        def objective(trial: optuna.Trial) -> float:
            p = {
                **BASE_PARAMS,
                **(params or {}),
                "n_estimators": 1000,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            # trial 하나 = nested run 하나
            with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
                mlflow.log_params(trial.params)
                model = _fit(task, p, X_tr, y_tr, X_va, y_va)
                score = _score(task, model, X_va, y_va)[1]
                trial.set_user_attr("best_iteration", model.best_iteration_)
                mlflow.log_metrics({"score": score, "best_iteration": model.best_iteration_ or 0})

            # 부모 run 에 진행 곡선 기록 → MLflow UI 에서 실시간 모니터링
            best_so_far.append(better(best_so_far[-1], score) if best_so_far else score)
            mlflow.log_metrics({"trial_score": score, "best_score": best_so_far[-1]},
                               step=trial.number, run_id=parent.info.run_id)
            return score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best = {
            **BASE_PARAMS, **(params or {}), **study.best_params,
            "subsample_freq": 1,
            "n_estimators": study.best_trial.user_attrs.get("best_iteration") or BASE_PARAMS["n_estimators"],
        }
        # 최적 파라미터로 재학습 후 저장
        model = _model(task, best).fit(X_tr, y_tr)
        metric, score = _score(task, model, X_va, y_va)
        features = list(X.columns)
        model_id, onnx_path = _save(model, features, task)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({f"final_{metric}": score, "best_trial": study.best_trial.number})
        _log_artifacts(model_id, onnx_path)

    return {
        "model_id": model_id, "mlflow_run_id": parent.info.run_id,
        "task": task, "metric": metric, "score": score,
        "params": best, "features": features,
        "n_trials": len(study.trials), "best_trial": study.best_trial.number,
    }


def predict(model_id: str, records: list[dict]) -> list:
    path = MODEL_DIR / f"{model_id}.joblib"
    if not path.exists():
        raise FileNotFoundError(model_id)
    bundle = joblib.load(path)
    X = pd.DataFrame(records)[bundle["features"]]
    return bundle["model"].predict(X).tolist()


def model_meta(model_id: str) -> dict:
    path = MODEL_DIR / f"{model_id}.json"
    if not path.exists():
        raise FileNotFoundError(model_id)
    return json.loads(path.read_text(encoding="utf-8"))


def onnx_path(model_id: str) -> Path:
    path = MODEL_DIR / f"{model_id}.onnx"
    if not path.exists():
        raise FileNotFoundError(model_id)
    return path
