"""House Prices – Advanced Regression Techniques: 파생 피처 + 스태킹.

구성
  - 카테고리형 44개      → CatBoost
  - 수치형 35개 + 파생 피처 → 선형(Ridge) / 비선형(LightGBM) 중 교차검증 점수가 좋은 쪽 하나
  - 메타 모델: 두 모델의 OOF(out-of-fold) 예측을 입력으로 받는 선형회귀(가중치 ≥ 0)

흐름
  1. 반복 K-fold(5-fold × 3회)로 기본 모델들의 OOF 예측과 점수 계산
  2. 메타 모델을 OOF 예측으로 학습, 스태킹 점수는 OOF 위에서 다시 5-fold 로 평가
  3. 기본 모델을 학습 데이터 전체로 재학습 → test 예측 → 메타 모델로 결합 → 제출 파일
  지표: log1p(SalePrice) 의 RMSE (대회 지표와 같음)

사용
  python house_prices_stacking.py            # 교차검증 + 전체 재학습 + 제출 파일
  python house_prices_stacking.py --submit   # 위 + Kaggle 제출 (스태킹 / CatBoost 단독 / 수치형 단독)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
COMPETITION = "house-prices-advanced-regression-techniques"
SEED = 42

# ---------------------------------------------------------------- 피처 구분
NUMERIC = [
    # 면적·크기
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF",
    "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
    # 개수
    "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "GarageCars",
    # 연도·월
    "YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold", "MoSold",
    # 점수형 등급
    "OverallQual", "OverallCond",
]
CATEGORICAL = [
    # 순서형 (품질 5단계)
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu",
    "GarageQual", "GarageCond", "PoolQC",
    # 순서형 (그 밖의 등급)
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageFinish", "Functional", "LotShape",
    "LandSlope", "PavedDrive", "Utilities", "Electrical",
    # 이진
    "Street", "CentralAir",
    # 명목형
    "MSSubClass", "MSZoning", "Alley", "LandContour", "LotConfig", "Neighborhood", "Condition1",
    "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
    "MasVnrType", "Foundation", "Heating", "GarageType", "Fence", "MiscFeature", "SaleType",
    "SaleCondition",
]
OUTLIER_IDS = [524, 1299]          # 거주 면적 4,000 sq ft 초과인데 가격 30만 미만
QUAL = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
QUAL_SUM_COLS = ["ExterQual", "KitchenQual", "BsmtQual", "GarageQual", "HeatingQC", "FireplaceQu"]
DERIVED = [
    "TotalSF", "TotalBath", "TotalPorchSF", "HouseAge", "RemodAge", "IsRemodeled", "IsNew",
    "HasPool", "Has2ndFlr", "HasGarage", "HasBsmt", "HasFireplace",
    "QualTotalSF", "QualGrLiv", "OverallGrade", "QualSum",
]
NUM_FEATURES = NUMERIC + DERIVED


# ---------------------------------------------------------------- 데이터
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """'NA'·'None' 을 결측으로 자동 변환하지 않고 읽는다 (대부분 '시설 없음' 이라는 값)."""
    tr = pd.read_csv(DATA / "train.csv", keep_default_na=False, na_values=[""])
    te = pd.read_csv(DATA / "test.csv", keep_default_na=False, na_values=[""])
    tr = tr[~tr["Id"].isin(OUTLIER_IDS)].reset_index(drop=True)
    return tr, te


def prepare(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """수치형 결측 처리 + 파생 피처. 채움 기준값은 학습 데이터에서만 계산한다."""
    out = []
    for df in (tr, te):
        df = df.copy()
        for c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")   # 'NA' 문자 → NaN
        for c in CATEGORICAL:
            df[c] = df[c].astype(str)
        out.append(df)
    tr, te = out

    lot_med = tr.groupby("Neighborhood")["LotFrontage"].median()
    lot_all = tr["LotFrontage"].median()
    for df in (tr, te):
        # 진짜 누락: 도로 접면 길이 → 동네별 중앙값
        df["LotFrontage"] = df["LotFrontage"].fillna(df["Neighborhood"].map(lot_med)).fillna(lot_all)
        # 차고 없음 → 차고 연도는 건축 연도로
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
        # 나머지 수치형 결측은 '시설 없음' (지하·차고·석조 외장 면적·개수) → 0
        for c in NUMERIC:
            df[c] = df[c].fillna(0)

        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
        df["TotalPorchSF"] = df[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]].sum(axis=1)
        df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
        df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
        df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
        df["IsNew"] = (df["HouseAge"] <= 1).astype(int)
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
        df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
        df["QualTotalSF"] = df["OverallQual"] * df["TotalSF"]
        df["QualGrLiv"] = df["OverallQual"] * df["GrLivArea"]
        df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
        df["QualSum"] = sum(df[c].map(QUAL).fillna(0) for c in QUAL_SUM_COLS)
    return tr, te


# ---------------------------------------------------------------- 모델
def make_catboost(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                             loss_function="RMSE", random_seed=seed, verbose=0, thread_count=-1,
                             cat_features=list(range(len(CATEGORICAL))), allow_writing_files=False)


class LinearNumeric:
    """선형: 치우친 피처 log1p → 표준화 → Ridge(교차검증으로 alpha 선택). 치우침 판정은 학습 데이터 기준."""

    def __init__(self, seed: int = 0):
        self.skewed: list[str] = []
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 30)))

    def _tf(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.skewed] = np.log1p(X[self.skewed].clip(lower=0))
        return X

    def fit(self, X: pd.DataFrame, y):
        sk = X.skew()
        self.skewed = [c for c in X.columns if abs(sk[c]) > 0.75 and X[c].min() >= 0]
        self.model.fit(self._tf(X), y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(self._tf(X))


class TreeNumeric:
    """비선형: LightGBM (작은 데이터용 보수적 설정)."""

    def __init__(self, seed: int = 0):
        self.model = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.01, num_leaves=8,
                                       min_child_samples=10, subsample=0.7, subsample_freq=1,
                                       colsample_bytree=0.5, reg_lambda=1.0, random_state=seed, verbose=-1)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class MLPNumeric:
    """비선형(신경망): 다층 퍼셉트론 (은닉층 128 → 64, ReLU).
    ReLU 신경망은 학습 범위 밖에서 직선으로 연장되므로, 입력을 학습 데이터의 최솟값~최댓값으로 잘라서 넣는다.
    입력: 치우친 피처 log1p → 학습 범위로 자르기 → 표준화. 데이터가 작아 seed 3개 모델의 평균을 쓴다."""

    def __init__(self, seed: int = 0, n_models: int = 3):
        from sklearn.neural_network import MLPRegressor
        self.skewed: list[str] = []
        self.lo = self.hi = None
        self.scaler = StandardScaler()
        self.y_mean = 0.0
        self.models = [MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-2,
                                    learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
                                    validation_fraction=0.1, n_iter_no_change=50, random_state=seed * 10 + k)
                       for k in range(n_models)]

    def _tf(self, X: pd.DataFrame):
        X = X.copy()
        X[self.skewed] = np.log1p(X[self.skewed].clip(lower=0))
        return self.scaler.transform(X.clip(self.lo, self.hi, axis=1))

    def fit(self, X: pd.DataFrame, y):
        sk = X.skew()
        self.skewed = [c for c in X.columns if abs(sk[c]) > 0.75 and X[c].min() >= 0]
        Xl = X.copy()
        Xl[self.skewed] = np.log1p(Xl[self.skewed].clip(lower=0))
        self.lo, self.hi = Xl.min(), Xl.max()
        self.scaler.fit(Xl)
        Z = self._tf(X)
        self.y_mean = float(np.mean(y))
        for m in self.models:
            m.fit(Z, np.asarray(y) - self.y_mean)
        return self

    def predict(self, X: pd.DataFrame):
        Z = self._tf(X)
        return np.mean([m.predict(Z) for m in self.models], axis=0) + self.y_mean


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


# ---------------------------------------------------------------- 교차검증
def oof_predictions(make_model, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame,
                    n_splits=5, n_repeats=3, label: str = "") -> tuple[np.ndarray, np.ndarray, list[float]]:
    """반복 K-fold. 반환: 반복 평균 OOF 예측, fold 모델들의 test 예측 평균, fold 별 점수."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    oof = np.zeros((n_repeats, len(X)))
    test_pred = np.zeros(len(X_test))
    scores = []
    for i, (tr_i, va_i) in enumerate(rkf.split(X)):
        rep = i // n_splits
        m = make_model(SEED + i).fit(X.iloc[tr_i], y[tr_i])
        oof[rep, va_i] = m.predict(X.iloc[va_i])
        test_pred += m.predict(X_test) / (n_splits * n_repeats)
        scores.append(rmse(y[va_i], oof[rep, va_i]))
        print(f"  [{label}] fold {i + 1}/{n_splits * n_repeats} RMSE {scores[-1]:.5f}", flush=True)
    return oof.mean(axis=0), test_pred, scores


def log_mlflow(params: dict, metrics: dict) -> None:
    """MLFLOW_TRACKING_URI 가 있고 서버가 응답하면 파라미터·지표만 기록 (파일 업로드 없음)."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        import mlflow
        import urllib.request
        urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=3)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("house-prices-stacking")
        with mlflow.start_run(run_name="stack-catboost-numeric"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
        print(f"MLflow 기록 완료 ({uri})")
    except Exception as e:  # noqa: BLE001
        print(f"MLflow 기록 생략: {type(e).__name__}")


def kaggle_submit(path: Path, message: str) -> None:
    env = dict(os.environ)
    tok = ROOT.parent / "kaggleToken"
    if "KAGGLE_API_TOKEN" not in env and tok.exists():
        env["KAGGLE_API_TOKEN"] = tok.read_text(encoding="utf-8").strip()
    subprocess.run([sys.executable, "-c", "from kaggle.cli import main; main()", "competitions", "submit",
                    "-c", COMPETITION, "-f", str(path), "-m", message], check=True, env=env)


# ---------------------------------------------------------------- 실행
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="Kaggle 에 제출 (스태킹 / CatBoost 단독 / 수치형 단독)")
    ap.add_argument("--numeric-model", choices=["auto", "linear", "tree", "mlp", "tree+mlp"], default="auto",
                    help="수치형 모델: auto = Ridge/LightGBM 중 교차검증으로 선택, linear = Ridge, tree = LightGBM, mlp = 신경망, tree+mlp = LightGBM·신경망 모두 스태킹")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    tr, te = load()
    tr, te = prepare(tr, te)
    y = np.log1p(tr["SalePrice"].to_numpy())
    Xc, Xc_te = tr[CATEGORICAL], te[CATEGORICAL]
    Xn, Xn_te = tr[NUM_FEATURES].astype(float), te[NUM_FEATURES].astype(float)
    print(f"train {len(tr)} (이상치 {len(OUTLIER_IDS)}개 제외) / test {len(te)} | "
          f"카테고리 {len(CATEGORICAL)} / 수치 {len(NUMERIC)} + 파생 {len(DERIVED)}")

    # 1) 수치형 모델: 하나를 고르거나(auto/linear/tree/mlp), 여러 개를 모두 스태킹에 넣는다(tree+mlp)
    all_models = {"linear_ridge": LinearNumeric, "tree_lightgbm": TreeNumeric, "nn_mlp": MLPNumeric}
    modes = {"auto": ["linear_ridge", "tree_lightgbm"], "linear": ["linear_ridge"], "tree": ["tree_lightgbm"],
             "mlp": ["nn_mlp"], "tree+mlp": ["tree_lightgbm", "nn_mlp"]}
    candidates = {k: all_models[k] for k in modes[args.numeric_model]}
    res = {}
    for name, cls in candidates.items():
        oof, tp, sc = oof_predictions(lambda s, c=cls: c(s), Xn, y, Xn_te, label=name)
        res[name] = dict(oof=oof, test=tp, cv=float(np.mean(sc)), cv_std=float(np.std(sc)))
        print(f"[수치형 {name}] CV RMSE {res[name]['cv']:.5f} ± {res[name]['cv_std']:.5f}")
    if args.numeric_model == "tree+mlp":
        num_names = list(res)
        print(f"→ 수치형 모델: {' + '.join(num_names)} (모두 스태킹)")
    else:
        num_names = [min(res, key=lambda k: res[k]["cv"])]
        print(f"→ 수치형 모델: {num_names[0]} ({'교차검증으로 선택' if args.numeric_model == 'auto' else '지정'})")
    tag = "+".join(num_names)

    # 2) CatBoost (카테고리형만)
    cat_oof, cat_tp, cat_sc = oof_predictions(make_catboost, Xc, y, Xc_te, label="catboost")
    print(f"[카테고리 catboost] CV RMSE {np.mean(cat_sc):.5f} ± {np.std(cat_sc):.5f}")

    # 3) 스태킹: OOF 예측 → 메타 모델 (가중치 ≥ 0 선형회귀)
    names = ["catboost", *num_names]
    Z = np.column_stack([cat_oof, *[res[n]["oof"] for n in num_names]])
    meta_oof = np.zeros(len(y))
    for tr_i, va_i in KFold(5, shuffle=True, random_state=SEED).split(Z):
        meta_oof[va_i] = LinearRegression(positive=True).fit(Z[tr_i], y[tr_i]).predict(Z[va_i])
    meta = LinearRegression(positive=True).fit(Z, y)
    stack_cv = rmse(y, meta_oof)
    avg_cv = rmse(y, Z.mean(axis=1))
    weights = {n: float(w) for n, w in zip(names, meta.coef_)}
    print(f"[스태킹] CV RMSE {stack_cv:.5f} | 메타 가중치 "
          + ", ".join(f"{n} {w:.3f}" for n, w in weights.items())
          + f", 절편 {meta.intercept_:.3f} | 참고: 단순 평균 {avg_cv:.5f}")
    corr = np.corrcoef(Z.T)
    print("[OOF 예측 상관] " + ", ".join(f"{names[i]}-{names[j]} {corr[i, j]:.3f}"
                                         for i in range(len(names)) for j in range(i + 1, len(names))))

    # 4) 기본 모델을 학습 데이터 전체로 재학습 → test 예측 → 메타 결합
    cat_full = make_catboost(SEED).fit(Xc, y)
    full = {"catboost": cat_full}
    te_pred = {"catboost": cat_full.predict(Xc_te)}
    for n in num_names:
        full[n] = candidates[n](SEED).fit(Xn, y)
        te_pred[n] = full[n].predict(Xn_te)
    stack_te = meta.predict(np.column_stack([te_pred[n] for n in names]))
    # 재학습 모델 점검: fold 평균 모델의 test 예측과 얼마나 같은지
    agree = {"catboost": float(np.corrcoef(te_pred["catboost"], cat_tp)[0, 1]),
             **{n: float(np.corrcoef(te_pred[n], res[n]["test"])[0, 1]) for n in num_names}}
    fit_rmse = {"catboost": rmse(y, cat_full.predict(Xc)),
                **{n: rmse(y, full[n].predict(Xn)) for n in num_names}}
    print(f"[전체 재학습] 학습 데이터 RMSE {fit_rmse} | fold 평균 모델과 test 예측 상관 {agree}")

    subs = {f"stack_{tag}": stack_te, "catboost_only": te_pred["catboost"],
            **{f"{n}_only": te_pred[n] for n in num_names}}

    # 5) MLP 범위 밖 예측 보정 (MLP 가 스태킹에 들어 있을 때)
    guard = {}
    if "nn_mlp" in num_names:
        mi = names.index("nn_mlp")
        # A. MLP 예측을 학습 데이터 정답 범위(log)로 자르기
        Zc = np.column_stack([te_pred[n] for n in names])
        Zc[:, mi] = np.clip(Zc[:, mi], y.min(), y.max())
        subs[f"stack_{tag}_clip"] = meta.predict(Zc)
        # B. 수치형 피처가 하나라도 학습 범위 밖인 test 집은 MLP 를 뺀 메타 모델로 예측
        lo, hi = Xn.min(), Xn.max()
        ood = ((Xn_te < lo) | (Xn_te > hi)).any(axis=1).to_numpy()
        keep = [i for i, n in enumerate(names) if n != "nn_mlp"]
        meta_wo = LinearRegression(positive=True).fit(Z[:, keep], y)
        gated = stack_te.copy()
        gated[ood] = meta_wo.predict(np.column_stack([te_pred[names[i]] for i in keep]))[ood]
        subs[f"stack_{tag}_gate"] = gated
        cols = ((Xn_te < lo) | (Xn_te > hi)).sum().sort_values(ascending=False)
        guard = {"ood_rows": int(ood.sum()), "ood_top_features": {k: int(v) for k, v in cols[cols > 0].head(8).items()},
                 "meta_without_mlp": {**{f"w_{names[i]}": float(w) for i, w in zip(keep, meta_wo.coef_)},
                                      "intercept": float(meta_wo.intercept_)},
                 "mlp_clipped_rows": int(((te_pred["nn_mlp"] < y.min()) | (te_pred["nn_mlp"] > y.max())).sum())}
        print(f"[MLP 보정] 학습 범위 밖 test 집 {guard['ood_rows']}채 (피처별 {guard['ood_top_features']}) | "
              f"MLP 예측이 정답 범위를 넘은 집 {guard['mlp_clipped_rows']}채 | MLP 뺀 메타 가중치 {guard['meta_without_mlp']}")
    paths = {}
    for k, p in subs.items():
        path = OUT / f"submission_{k}.csv"
        pd.DataFrame({"Id": te["Id"], "SalePrice": np.expm1(p)}).to_csv(path, index=False)
        paths[k] = path
    result = {
        "numeric_models": num_names,
        "cv": {**{k: v["cv"] for k, v in res.items()},
               "catboost": float(np.mean(cat_sc)), "stack": stack_cv, "simple_average": avg_cv},
        "meta": {**{f"w_{n}": w for n, w in weights.items()}, "intercept": float(meta.intercept_)},
        "full_refit_train_rmse": fit_rmse, "full_vs_fold_test_corr": agree, "mlp_guard": guard,
        "submissions": {k: str(v) for k, v in paths.items()}, "seconds": round(time.time() - t0, 1),
    }
    (OUT / f"result_{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log_mlflow({"numeric_models": tag, "n_categorical": len(CATEGORICAL), "n_numeric": len(NUM_FEATURES),
                "outliers_removed": len(OUTLIER_IDS), "cv": "RepeatedKFold 5x3"},
               {**{f"cv_{k}": v["cv"] for k, v in res.items()},
                "cv_catboost": float(np.mean(cat_sc)), "cv_stack": stack_cv, "cv_simple_avg": avg_cv,
                **{f"w_{n}": w for n, w in weights.items()}})
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.submit:
        for k, path in paths.items():
            kaggle_submit(path, f"{k}: catboost(categorical) + {tag}(numeric) stacking, CV {result['cv']}")


if __name__ == "__main__":
    main()
