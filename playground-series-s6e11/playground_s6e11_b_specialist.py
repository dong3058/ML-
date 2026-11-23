"""Playground S6E11 (Predicting Credit Card Fraud) B 피처 전문 모델 앙상블 (v8).

B 피처 = 사기율이 값에 따라 단조롭게 변하는 연속 점수형 (playground_s6e11_feature_group_profile.py 의 B 부류)
  merchant_risk_score, velocity_score, customer_age, txn_count_last_24h, hours_since_last_txn,
  prior_disputes, merchant_category, amount_usd

본 모델: v7 튜닝 LightGBM (전 피처, output/oof_lightgbm_tuned.csv 재사용, 같은 5-fold 분할)
전문 모델 (모두 같은 5-fold, OOF)
  S1: LightGBM, B 피처만, 전체 행 학습
  S2: LightGBM, B 피처만, 'A 신호 없음' 행만 학습 (A = CVV 재입력·인증 없음·위험 bool 6개)
      → 블렌드도 A 신호 없는 행에만 적용 (놓친 사기 유형 0 이 사는 곳)
  S3: 스플라인 로지스틱 회귀, B 피처만, 전체 행 (단조·매끈한 관계를 직접 모델링)
결합
  - 순위 평균: w·전문 + (1−w)·본 모델 (S2 는 해당 행 안에서 확률 평균)
  - 스태킹: 본 모델·전문 모델 logit 을 입력으로 하는 로지스틱 회귀 (바깥 fold 와 다른 5-fold 로 OOF)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
import json

import playground_s6e11_lightgbm as G
import playground_s6e11_linear_baseline as L

B_NUM = ["merchant_risk_score", "velocity_score", "customer_age", "txn_count_last_24h",
         "hours_since_last_txn", "prior_disputes", "amount_usd"]
B_CAT = ["merchant_category"]
B = B_NUM + B_CAT
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)


def lgb_params():
    p = {**G.PARAMS, **json.loads((L.OUT / "best_params_lightgbm.json").read_text(encoding="utf-8"))}
    p.update(learning_rate=0.05, feature_fraction=0.8)
    return p


def oof_lgb_b(X, y, splits, rows=None):
    params = lgb_params()
    oof = np.full(len(y), np.nan)
    for k, (tr_i, va_i) in enumerate(splits):
        if rows is not None:
            tr_i, va_i = tr_i[rows[tr_i]], va_i[rows[va_i]]
        dtr = lgb.Dataset(X.iloc[tr_i], y[tr_i])
        dva = lgb.Dataset(X.iloc[va_i], y[va_i], reference=dtr)
        m = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_i] = m.predict(X.iloc[va_i], num_iteration=m.best_iteration)
        print(f"    fold {k}: AUC {roc_auc_score(y[va_i], oof[va_i]):.5f}  iter {m.best_iteration}", flush=True)
    return oof


def oof_spline(df, y, splits):
    pre = ColumnTransformer([
        ("sp", make_pipeline(SplineTransformer(n_knots=6, degree=3, extrapolation="constant"), StandardScaler()), B_NUM),
        ("oh", OneHotEncoder(handle_unknown="ignore"), B_CAT)])
    oof = np.zeros(len(y))
    Xd = df[B].copy()
    Xd["amount_usd"] = np.log1p(Xd["amount_usd"])
    Xd["hours_since_last_txn"] = np.log1p(Xd["hours_since_last_txn"])
    for k, (tr_i, va_i) in enumerate(splits):
        m = make_pipeline(pre, LogisticRegression(max_iter=3000, C=1.0)).fit(Xd.iloc[tr_i], y[tr_i])
        oof[va_i] = m.predict_proba(Xd.iloc[va_i])[:, 1]
        print(f"    fold {k}: AUC {roc_auc_score(y[va_i], oof[va_i]):.5f}", flush=True)
    return oof


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fold_gains(y, p, base, fold_id):
    return [roc_auc_score(y[fold_id == f], p[fold_id == f]) - roc_auc_score(y[fold_id == f], base[fold_id == f])
            for f in range(5)]


def report(name, y, base, cands, fold_id):
    """cands: {w: 예측} → 최고 w, 전체 gain, 오른 fold 수"""
    b = roc_auc_score(y, base)
    res = [(w, roc_auc_score(y, p)) for w, p in cands.items()]
    w, a = max(res, key=lambda t: t[1])
    g = fold_gains(y, cands[w], base, fold_id)
    curve = "  ".join(f"{ww:.1f}:{aa - b:+.5f}" for ww, aa in res)
    print(f"  {name:28s} 최고 w={w:.1f}  AUC {a:.5f}  gain {a - b:+.5f}  fold 상승 {sum(x > 0 for x in g)}/5  "
          f"fold 최소 {min(g):+.5f}")
    print(f"    {curve}")
    return {"name": name, "w": w, "auc": a, "gain": a - b, "folds_up": sum(x > 0 for x in g), "fold_min": min(g)}


def main():
    tr, _ = L.load()
    y = tr[L.TARGET].astype(int).to_numpy()
    X = G.prepare(tr)[B]
    splits = list(StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(X, y))
    fold_id = np.zeros(len(y), dtype=int)
    for k, (_, va_i) in enumerate(splits):
        fold_id[va_i] = k

    main_oof = pd.read_csv(L.OUT / "oof_lightgbm_tuned.csv")
    assert (main_oof["id"].to_numpy() == tr["id"].to_numpy()).all()
    p_main = main_oof["p"].to_numpy()
    no_a = ((tr[L.BOOLEAN].sum(axis=1) == 0) & (tr.cvv_retry_count == 0)
            & (tr.auth_method != "No Authentication")).to_numpy()
    print(f"본 모델 v7 OOF AUC {roc_auc_score(y, p_main):.5f}")
    print(f"A 신호 없는 행: {no_a.sum()} (사기 {y[no_a].sum()}), 그 안에서 본 모델 AUC "
          f"{roc_auc_score(y[no_a], p_main[no_a]):.5f}\n")

    print("[S1] LightGBM, B 피처만, 전체 행", flush=True)
    s1 = oof_lgb_b(X, y, splits)
    print("[S2] LightGBM, B 피처만, A 신호 없는 행만", flush=True)
    s2 = oof_lgb_b(X, y, splits, rows=no_a)
    print("[S3] 스플라인 로지스틱, B 피처만, 전체 행", flush=True)
    s3 = oof_spline(tr, y, splits)

    print("\n[단독 성능]")
    print(f"  S1 전체 AUC {roc_auc_score(y, s1):.5f} | A 신호 없는 행 안 AUC {roc_auc_score(y[no_a], s1[no_a]):.5f}")
    print(f"  S2 A 신호 없는 행 안 AUC {roc_auc_score(y[no_a], s2[no_a]):.5f}")
    print(f"  S3 전체 AUC {roc_auc_score(y, s3):.5f} | A 신호 없는 행 안 AUC {roc_auc_score(y[no_a], s3[no_a]):.5f}")
    print(f"  본 모델 A 신호 없는 행 안 AUC {roc_auc_score(y[no_a], p_main[no_a]):.5f}")
    for nm, s in [("S1", s1), ("S3", s3)]:
        r_all = np.corrcoef(y - s, y - p_main)[0, 1]
        r_f = np.corrcoef((y - s)[y == 1], (y - p_main)[y == 1])[0, 1]
        print(f"  {nm} vs 본 모델 잔차 상관: 전체 {r_all:.3f}, 사기 행 {r_f:.3f}")

    print("\n[블렌드] gain = 본 모델 대비 전체 OOF AUC 변화")
    rm = rankdata(p_main) / len(y)
    rows = []
    for nm, s in [("S1 순위 평균", s1), ("S3 순위 평균", s3)]:
        rs = rankdata(s) / len(y)
        rows.append(report(nm, y, p_main, {w: w * rs + (1 - w) * rm for w in WEIGHTS}, fold_id))
    cands = {}
    for w in WEIGHTS:
        p = p_main.copy()
        p[no_a] = w * s2[no_a] + (1 - w) * p_main[no_a]
        cands[w] = p
    rows.append(report("S2 A신호없는 행만 확률 평균", y, p_main, cands, fold_id))

    # 스태킹
    Z = np.column_stack([logit(p_main), logit(s1), logit(s3), no_a.astype(float),
                         no_a * logit(np.nan_to_num(s2, nan=0.5))])
    stack = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=L.SEED + 7).split(Z, y):
        stack[b] = LogisticRegression(max_iter=2000).fit(Z[a], y[a]).predict_proba(Z[b])[:, 1]
    g = fold_gains(y, stack, p_main, fold_id)
    a = roc_auc_score(y, stack)
    print(f"  {'스태킹 (본+S1+S2+S3 logit)':28s} AUC {a:.5f}  gain {a - roc_auc_score(y, p_main):+.5f}  "
          f"fold 상승 {sum(x > 0 for x in g)}/5  fold 최소 {min(g):+.5f}")
    rows.append({"name": "stacking", "w": np.nan, "auc": a, "gain": a - roc_auc_score(y, p_main),
                 "folds_up": sum(x > 0 for x in g), "fold_min": min(g)})

    pd.DataFrame(rows).to_csv(L.OUT / "b_specialist_results.csv", index=False)
    pd.DataFrame({"id": tr["id"], "y": y, "p_main": p_main, "s1": s1, "s2": s2, "s3": s3, "stack": stack,
                  "fold": fold_id}).to_csv(L.OUT / "oof_b_specialist.csv", index=False)


if __name__ == "__main__":
    main()
