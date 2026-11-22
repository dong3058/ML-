"""House Prices 피처 공유 실험: 오차 분석에서 나온 개선안 1·2 를 따로, 그리고 함께 적용해 비교한다.

  base    : CatBoost = 카테고리 44개, LightGBM = 수치형 51개 (기존 역할 분담)
  lgb_cat : LightGBM 에 품질 등급 10개(숫자로) + SaleCondition·MSZoning·Neighborhood·Functional(카테고리) 추가
  cat_num : CatBoost 에 수치형 51개 추가 (카테고리 + 수치형 전체)

조합 4가지(base / lgb_cat / cat_num / both)의 스태킹 교차검증과 test 예측을 만든다.
교차검증 예측(OOF)은 output/oof_feature_sharing.csv 에 저장하고, 기존 base 의 OOF 는
오차 분석(house_prices_error_analysis.py)이 저장한 output/oof_catboost_lightgbm.csv 를 재사용한다.
"""

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import house_prices_stacking as H

QUALITY_COLS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu",
                "GarageQual", "GarageCond", "PoolQC"]
LGB_CATS = ["SaleCondition", "MSZoning", "Neighborhood", "Functional"]
OOF_BASE = H.OUT / "oof_catboost_lightgbm.csv"
OOF_NEW = H.OUT / "oof_feature_sharing.csv"


def lgb_frame(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LightGBM 용: 수치형 + 품질 등급 숫자 + 핵심 카테고리(train·test 공통 범주로 고정)."""
    out = []
    for df in (tr, te):
        X = df[H.NUM_FEATURES].astype(float).copy()
        for c in QUALITY_COLS:
            X[f"{c}_ord"] = df[c].map(H.QUAL).fillna(0).astype(float)
        out.append(X)
    for c in LGB_CATS:
        cats = sorted(set(tr[c]) | set(te[c]))
        out[0][c] = pd.Categorical(tr[c], categories=cats)
        out[1][c] = pd.Categorical(te[c], categories=cats)
    return out[0], out[1]


class TreeWithCats(H.TreeNumeric):
    """기존 LightGBM 설정 + 작은 데이터용 카테고리 분기 설정 (기본값 min_data_per_group=100 은 1,458행에 너무 큼)."""

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.model.set_params(min_data_per_group=10, cat_smooth=10, max_cat_threshold=16)


def make_catboost_full(seed: int) -> CatBoostRegressor:
    m = H.make_catboost(seed)  # cat_features = 앞쪽 44개 위치 → 카테고리를 앞에, 수치형을 뒤에 붙인다
    return m


def stack(cols: list[np.ndarray], y: np.ndarray):
    Z = np.column_stack(cols)
    oof = np.zeros(len(y))
    for tr_i, va_i in KFold(5, shuffle=True, random_state=H.SEED).split(Z):
        oof[va_i] = LinearRegression(positive=True).fit(Z[tr_i], y[tr_i]).predict(Z[va_i])
    return H.rmse(y, oof), LinearRegression(positive=True).fit(Z, y), oof


def main() -> None:
    tr, te = H.load()
    tr, te = H.prepare(tr, te)
    y = np.log1p(tr["SalePrice"].to_numpy())

    X = {
        "cat_base": (tr[H.CATEGORICAL], te[H.CATEGORICAL], H.make_catboost),
        "cat_num": (tr[H.CATEGORICAL + H.NUM_FEATURES], te[H.CATEGORICAL + H.NUM_FEATURES], make_catboost_full),
        "lgb_base": (tr[H.NUM_FEATURES].astype(float), te[H.NUM_FEATURES].astype(float), lambda s: H.TreeNumeric(s)),
        "lgb_cat": (*lgb_frame(tr, te), lambda s: TreeWithCats(s)),
    }
    for k in ("cat_num",):
        X[k][0][H.NUM_FEATURES] = X[k][0][H.NUM_FEATURES].astype(float)
        X[k][1][H.NUM_FEATURES] = X[k][1][H.NUM_FEATURES].astype(float)

    # 1) OOF: base 는 저장된 것 재사용, 새 모델은 계산 후 저장
    base = pd.read_csv(OOF_BASE, usecols=["Id", "p_cat", "p_lgb"])
    assert (base.Id.to_numpy() == tr.Id.to_numpy()).all(), "저장된 OOF 의 집 순서가 다름"
    oof = {"cat_base": base.p_cat.to_numpy(), "lgb_base": base.p_lgb.to_numpy()}
    new = pd.read_csv(OOF_NEW) if OOF_NEW.exists() else pd.DataFrame({"Id": tr.Id})
    cv = {}
    for k in ("lgb_cat", "cat_num"):
        if k in new:
            oof[k] = new[k].to_numpy()
        else:
            Xtr, Xte, mk = X[k]
            oof[k], _, _ = H.oof_predictions(mk, Xtr, y, Xte, label=k)
            new[k] = oof[k]
            new.to_csv(OOF_NEW, index=False)
    for k, p in oof.items():
        cv[k] = H.rmse(y, p)
        print(f"[단독 {k}] OOF RMSE {cv[k]:.5f}", flush=True)

    # 2) 조합별 스태킹
    combos = {"base": ("cat_base", "lgb_base"), "lgb_cat": ("cat_base", "lgb_cat"),
              "cat_num": ("cat_num", "lgb_base"), "both": ("cat_num", "lgb_cat")}
    res = {}
    for name, (c, l) in combos.items():
        sc, meta, so = stack([oof[c], oof[l]], y)
        err = so - y
        cheap = tr.SalePrice <= tr.SalePrice.quantile(0.1)
        res[name] = {"cv": sc, "w_cat": float(meta.coef_[0]), "w_lgb": float(meta.coef_[1]),
                     "intercept": float(meta.intercept_), "corr": float(np.corrcoef(oof[c], oof[l])[0, 1]),
                     "rmse_C(all)": H.rmse(y[tr.MSZoning == "C (all)"], so[tr.MSZoning == "C (all)"]),
                     "rmse_Abnorml": H.rmse(y[tr.SaleCondition == "Abnorml"], so[tr.SaleCondition == "Abnorml"]),
                     "rmse_D1": H.rmse(y[cheap], so[cheap]), "bias_D1": float(err[cheap].mean()),
                     "meta": meta}
    print("\n=== 조합별 스태킹 (OOF) ===")
    show = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "meta"} for k, v in res.items()}).T
    print(show.round(4).to_string(), flush=True)

    # 3) 필요한 모델을 전체 데이터로 재학습 → test 예측 → 조합별 제출 파일
    te_pred = {}
    for k, (Xtr, Xte, mk) in X.items():
        te_pred[k] = mk(H.SEED).fit(Xtr, y).predict(Xte)
    i2550 = te.index[te.Id == 2550][0]
    for name, (c, l) in combos.items():
        p = res[name]["meta"].predict(np.column_stack([te_pred[c], te_pred[l]]))
        path = H.OUT / f"submission_share_{name}.csv"
        pd.DataFrame({"Id": te["Id"], "SalePrice": np.expm1(p)}).to_csv(path, index=False)
        res[name]["test_Id2550"] = float(np.expm1(p[i2550]))
        res[name]["test_max"] = float(np.expm1(p.max()))
        print(f"[{name}] {path.name} | Id 2550 {res[name]['test_Id2550']:,.0f} | 최댓값 {res[name]['test_max']:,.0f}")
    for v in res.values():
        v.pop("meta")
    (H.OUT / "result_feature_sharing.json").write_text(
        json.dumps({"single_cv": cv, "stack": res}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
    main()
