"""Playground S6E11 (Predicting Credit Card Fraud) LightGBM (GBDT) + 오차 분석.

v3: LightGBM, 범주형은 LightGBM 내장 categorical 처리, 수치형은 원본 그대로 (트리는 단조 변환에 무관).
선형 기준 모델(v2_log1p)의 오차 분석에서 곡선 관계로 어긋났던 4개 변수
(velocity_score, txn_count_last_24h, merchant_risk_score, customer_age) 와 Crypto Exchange 가
GBDT 에서 고르게 맞는지, 같은 오차 분석 함수로 비교한다.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import playground_s6e11_linear_baseline as L

PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.03, num_leaves=31,
    min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, cat_smooth=20, verbose=-1, n_jobs=-1, seed=L.SEED,
)
FOCUS = ["velocity_score", "txn_count_last_24h", "merchant_risk_score", "customer_age", "merchant_category"]


def prepare(df):
    X = df.drop(columns=["id", L.TARGET], errors="ignore").copy()
    for c in L.CATEGORICAL:
        X[c] = X[c].astype("category")
    X[L.BOOLEAN] = X[L.BOOLEAN].astype(int)
    return X


def oof_lgb(X, y, Xte):
    oof, pte, iters = np.zeros(len(y)), np.zeros(len(Xte)), []
    skf = StratifiedKFold(5, shuffle=True, random_state=L.SEED)
    for k, (tr_i, va_i) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[tr_i], y[tr_i])
        dva = lgb.Dataset(X.iloc[va_i], y[va_i], reference=dtr)
        m = lgb.train(PARAMS, dtr, 5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_i] = m.predict(X.iloc[va_i], num_iteration=m.best_iteration)
        pte += m.predict(Xte, num_iteration=m.best_iteration) / 5
        iters.append(m.best_iteration)
        print(f"  fold {k}: AUC {roc_auc_score(y[va_i], oof[va_i]):.5f}  iter {m.best_iteration}")
    imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    return oof, pte, iters, imp


def main():
    tr, te = L.load()
    y = tr[L.TARGET].astype(int).to_numpy()
    X, Xte = prepare(tr), prepare(te)
    Xte[L.CATEGORICAL] = Xte[L.CATEGORICAL].apply(
        lambda s: s.cat.set_categories(X[s.name].cat.categories))

    print("[v3_lightgbm]")
    oof, pte, iters, imp = oof_lgb(X, y, Xte)
    print(f"  OOF AUC {roc_auc_score(y, oof):.5f}  logloss {log_loss(y, oof):.5f}")
    print("\n변수 중요도 (gain, 마지막 fold, 비율 %):")
    print((imp / imp.sum() * 100).round(2).to_string())

    _, tables = L.error_analysis(tr, y, oof, "v3_lightgbm", tag="lightgbm")

    # 선형 모델과 같은 그룹에서 ratio 비교
    lin_path = L.OUT / "oof_linear.csv"
    if lin_path.exists():
        lin = pd.read_csv(lin_path)
        d = tr.copy()
        d["y"], d["p_lin"], d["p_lgb"] = y, lin["p"].to_numpy(), oof
        print("\n[선형(v2) vs LightGBM(v3) — 문제 구간 ratio 비교]")
        for c in FOCUS:
            key = pd.qcut(d[c], 10, duplicates="drop") if c in L.numeric_cols(tr) and tr[c].nunique() > 12 else d[c]
            g = d.groupby(key, observed=True)
            t = pd.DataFrame({
                "n": g.size(),
                "fraud%": g["y"].mean() * 100,
                "ratio_lin": g["p_lin"].mean() / g["y"].mean(),
                "ratio_lgb": g["p_lgb"].mean() / g["y"].mean(),
            })
            t = t[t.n >= 300].round(3)
            print(f"\n{c}  (|ratio-1| 평균: 선형 {(t.ratio_lin - 1).abs().mean():.3f}, "
                  f"LightGBM {(t.ratio_lgb - 1).abs().mean():.3f})")
            print(t.to_string())

    sub = pd.DataFrame({"id": te["id"], L.TARGET: pte})
    sub.to_csv(L.OUT / "submission_lightgbm.csv", index=False)
    print("\n제출 파일 저장(제출은 안 함):", L.OUT / "submission_lightgbm.csv", " 평균 iter", int(np.mean(iters)))


if __name__ == "__main__":
    main()
