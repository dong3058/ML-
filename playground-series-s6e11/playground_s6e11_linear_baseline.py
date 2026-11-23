"""Playground S6E11 (Predicting Credit Card Fraud) 선형 기준 모델 + 오차 분석.

v1: 로지스틱 회귀, 수치형 원본 그대로 (표준화만)
v2: v1 + 꼬리가 긴(왜도 > 1) 수치형을 log1p 변환
두 버전을 5-fold StratifiedKFold OOF ROC AUC 로 비교하고, 더 나은 버전의 OOF 예측으로 오차 분석을 한다.

오차 분석: 행별 오차 = log loss, 그룹별로
  - fraud%  : 실제 사기 비율
  - pred%   : 평균 예측 확률
  - ratio   : pred / fraud (1 이면 보정이 맞음, 1 보다 크면 과대, 작으면 과소 예측)
  - auc     : 그룹 안에서의 순위 성능
  - loss_share_% / n_share_% : 전체 log loss 중 그룹 몫 / 행 수 몫 (차이가 크면 오차가 몰림)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "output"
SEED = 42
TARGET = "is_fraud"
CATEGORICAL = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOLEAN = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
           "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def load():
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    return tr, te


def numeric_cols(df):
    return [c for c in df.columns if c not in CATEGORICAL + BOOLEAN + ["id", TARGET]]


def skewed_cols(df, threshold=1.0):
    sk = df[numeric_cols(df)].skew()
    print("수치형 왜도 (skew):")
    print(sk.sort_values(ascending=False).round(2).to_string(), "\n")
    return [c for c in sk.index if sk[c] > threshold and df[c].min() >= 0]


def make_matrix(df, log_cols):
    X = df[numeric_cols(df)].astype(float).copy()
    for c in log_cols:
        X[c] = np.log1p(X[c])
    X[BOOLEAN] = df[BOOLEAN].astype(int)
    dummies = pd.get_dummies(df[CATEGORICAL], drop_first=True, dtype=int)
    return pd.concat([X, dummies], axis=1)


def oof_logistic(X, y):
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for k, (tr_i, va_i) in enumerate(skf.split(X, y)):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X.iloc[tr_i], y[tr_i])
        oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
        print(f"  fold {k}: AUC {roc_auc_score(y[va_i], oof[va_i]):.5f}")
    return oof


def group_table(d, col):
    g = d.groupby(col, observed=True)
    t = pd.DataFrame({
        "n": g.size(),
        "fraud%": g["y"].mean() * 100,
        "pred%": g["p"].mean() * 100,
        "auc": g.apply(lambda s: roc_auc_score(s["y"], s["p"]) if s["y"].nunique() == 2 else np.nan,
                       include_groups=False),
        "logloss": g["loss"].mean(),
        "n_share_%": g.size() / len(d) * 100,
        "loss_share_%": g["loss"].sum() / d["loss"].sum() * 100,
    })
    t["ratio"] = t["pred%"] / t["fraud%"].replace(0, np.nan)
    return t.round(4)


def error_analysis(tr, y, p, label, tag="linear"):
    d = tr.copy()
    d["y"], d["p"] = y, p
    eps = 1e-15
    d["loss"] = -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))

    print(f"\n===== 오차 분석 ({label}) =====")
    print("\n[예측 확률 10분위별 보정] (ratio 1 이면 예측 확률이 실제 사기율과 일치)")
    d["p_decile"] = pd.qcut(d["p"], 10, labels=False, duplicates="drop")
    print(group_table(d, "p_decile")[["n", "fraud%", "pred%", "ratio", "loss_share_%"]].to_string())

    summary, tables = [], {}
    for c in CATEGORICAL + BOOLEAN + numeric_cols(tr):
        key = c
        if c in numeric_cols(tr) and tr[c].nunique() > 12:
            key = f"{c}_bin"
            d[key] = pd.qcut(d[c], 10, duplicates="drop")
        t = group_table(d, key)
        t = t[t["n"] >= 300]
        tables[c] = t
        summary.append({
            "feature": c,
            "groups": len(t),
            "ratio_min": t["ratio"].min(),
            "ratio_max": t["ratio"].max(),
            "auc_min": t["auc"].min(),
            "auc_max": t["auc"].max(),
            "loss_share_max_%": t["loss_share_%"].max(),
            "n_share_at_max_%": t.loc[t["loss_share_%"].idxmax(), "n_share_%"],
        })
    s = pd.DataFrame(summary)
    s["ratio_spread"] = s["ratio_max"] - s["ratio_min"]
    s = s.sort_values("ratio_spread", ascending=False).round(3)
    print("\n[변수별 요약] ratio_spread 가 클수록 그룹마다 보정이 어긋남 (오차가 고르지 않음)")
    print(s.to_string(index=False))

    for c in s["feature"].head(5):
        print(f"\n[상세] {c}")
        print(tables[c].to_string())

    OUT.mkdir(exist_ok=True)
    s.to_csv(OUT / f"error_summary_{tag}.csv", index=False)
    d[["id", "y", "p", "loss"]].to_csv(OUT / f"oof_{tag}.csv", index=False)
    return s, tables


def main():
    tr, _ = load()
    y = tr[TARGET].astype(int).to_numpy()
    log_cols = skewed_cols(tr)
    print("log1p 변환 대상:", log_cols, "\n")

    results = {}
    for name, cols in [("v1_raw", []), ("v2_log1p", log_cols)]:
        print(f"[{name}]")
        X = make_matrix(tr, cols)
        p = oof_logistic(X, y)
        results[name] = p
        print(f"  OOF AUC {roc_auc_score(y, p):.5f}  logloss {log_loss(y, p):.5f}\n")

    best = max(results, key=lambda k: roc_auc_score(y, results[k]))
    print("선택:", best)
    error_analysis(tr, y, results[best], best)


if __name__ == "__main__":
    main()
