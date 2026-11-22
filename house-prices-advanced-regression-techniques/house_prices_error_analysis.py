"""House Prices 오차 분석: CatBoost(카테고리형) + LightGBM(수치형) 스태킹의 교차검증(OOF) 예측 기준.

OOF 예측은 output/oof_catboost_lightgbm.csv 에 저장하고, 있으면 다시 계산하지 않는다 (--refresh 로 재계산).
오차 = log1p(예측) − log1p(실제). 양수면 과대 예측, 음수면 과소 예측.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import house_prices_stacking as H

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
OOF_FILE = H.OUT / "oof_catboost_lightgbm.csv"


def build_oof() -> pd.DataFrame:
    tr, te = H.load()
    tr, te = H.prepare(tr, te)
    y = np.log1p(tr["SalePrice"].to_numpy())
    Xc, Xn = tr[H.CATEGORICAL], tr[H.NUM_FEATURES].astype(float)
    cat_oof, _, _ = H.oof_predictions(H.make_catboost, Xc, y, te[H.CATEGORICAL], label="catboost")
    lgb_oof, _, _ = H.oof_predictions(lambda s: H.TreeNumeric(s), Xn, y, te[H.NUM_FEATURES].astype(float),
                                      label="tree_lightgbm")
    Z = np.column_stack([cat_oof, lgb_oof])
    stack = np.zeros(len(y))
    for tr_i, va_i in KFold(5, shuffle=True, random_state=H.SEED).split(Z):
        stack[va_i] = LinearRegression(positive=True).fit(Z[tr_i], y[tr_i]).predict(Z[va_i])
    out = tr.copy()
    out["y"], out["p_cat"], out["p_lgb"], out["p_stack"] = y, cat_oof, lgb_oof, stack
    H.OUT.mkdir(exist_ok=True)
    out.to_csv(OOF_FILE, index=False)
    return out


def group_table(d: pd.DataFrame, col: str, min_n: int = 10) -> pd.DataFrame:
    g = d.groupby(col, observed=True)
    t = pd.DataFrame({
        "n": g.size(),
        "rmse": g["err"].apply(lambda e: np.sqrt(np.mean(e ** 2))),
        "bias": g["err"].mean(),                        # +: 과대 예측, −: 과소 예측
        "rmse_cat": g["err_cat"].apply(lambda e: np.sqrt(np.mean(e ** 2))),
        "rmse_lgb": g["err_lgb"].apply(lambda e: np.sqrt(np.mean(e ** 2))),
        "share_%": g["se"].sum() / d["se"].sum() * 100,
    })
    t = t[t.n >= min_n].sort_values("rmse", ascending=False)
    return t.round(4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    d = build_oof() if (args.refresh or not OOF_FILE.exists()) else pd.read_csv(
        OOF_FILE, keep_default_na=False, na_values=[""])
    for c in H.CATEGORICAL:
        d[c] = d[c].astype(str)
    d["err"], d["err_cat"], d["err_lgb"] = d.p_stack - d.y, d.p_cat - d.y, d.p_lgb - d.y
    d["se"] = d.err ** 2
    total = np.sqrt(d.se.mean())
    print(f"스태킹 OOF RMSE {total:.5f} (CatBoost {np.sqrt((d.err_cat ** 2).mean()):.5f}, "
          f"LightGBM {np.sqrt((d.err_lgb ** 2).mean()):.5f}) | 집 {len(d)}채")

    # 1) 오차 집중도
    s = d.se.sort_values(ascending=False).to_numpy()
    cum = np.cumsum(s) / s.sum() * 100
    for k in (10, 20, 50, int(len(d) * 0.05), int(len(d) * 0.10)):
        print(f"  상위 {k}채가 전체 제곱오차의 {cum[k - 1]:.1f}%")
    rest = np.sqrt(s[int(len(d) * 0.05):].mean())
    print(f"  상위 5% 를 뺀 나머지 RMSE {rest:.5f}")

    # 2) 가장 크게 틀린 집
    cols = ["Id", "Neighborhood", "MSZoning", "SaleType", "SaleCondition", "BldgType", "OverallQual", "OverallCond",
            "GrLivArea", "TotalSF", "YearBuilt", "YrSold", "SalePrice"]
    top = d.reindex(d.err.abs().sort_values(ascending=False).index).head(20).copy()
    top["pred"] = np.expm1(top.p_stack).round(0)
    top["err"], top["err_cat"], top["err_lgb"] = top.err.round(3), top.err_cat.round(3), top.err_lgb.round(3)
    print("\n=== 가장 크게 틀린 20채 (err: +과대 / −과소, log 기준) ===")
    print(top[cols + ["pred", "err", "err_cat", "err_lgb"]].to_string(index=False))

    # 3) 그룹별
    d["price_bin"] = pd.qcut(d.SalePrice, 10, labels=[f"D{i}" for i in range(1, 11)])
    d["age_bin"] = pd.cut(d.HouseAge, [-1, 5, 15, 30, 50, 80, 200], labels=["0~5", "6~15", "16~30", "31~50", "51~80", "80+"])
    d["sf_bin"] = pd.qcut(d.TotalSF, 5, labels=["Q1(작음)", "Q2", "Q3", "Q4", "Q5(큼)"])
    for col, title in [("SaleCondition", "판매 상황"), ("SaleType", "계약 방식"), ("MSZoning", "용도 지역"),
                       ("BldgType", "건물 유형"), ("OverallQual", "전체 품질"), ("OverallCond", "전체 상태"),
                       ("price_bin", "실제 가격 10분위 (D1 저가 ~ D10 고가)"), ("age_bin", "판매 시 집 나이"),
                       ("sf_bin", "총면적 5분위"), ("Functional", "기능 상태")]:
        print(f"\n=== {title} ===")
        print(group_table(d, col, min_n=5).to_string())
    print("\n=== 동네 (오차 큰 순, 10채 이상) ===")
    print(group_table(d, "Neighborhood").head(12).to_string())

    # 4) 오차와 피처의 관계 (부호 있는 오차와의 상관 = 편향 방향, 절대 오차와의 상관 = 불확실성)
    num = ["SalePrice", "OverallQual", "OverallCond", "GrLivArea", "TotalSF", "LotArea", "LotFrontage", "HouseAge",
           "RemodAge", "TotalBath", "GarageCars", "MasVnrArea", "TotalPorchSF", "QualSum"]
    rel = pd.DataFrame({"corr(err)": [d.err.corr(d[c]) for c in num],
                        "corr(|err|)": [d.err.abs().corr(d[c]) for c in num]}, index=num).round(3)
    print("\n=== 오차와 피처의 상관 ===")
    print(rel.sort_values("corr(err)").to_string())


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
    main()
