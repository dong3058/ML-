"""Playground S6E11 (Predicting Credit Card Fraud) CatBoost 단독 + LightGBM(v7) 비교·블렌드 (v15).

LightGBM(v3/v7) 과 같은 조건
  - 같은 5-fold StratifiedKFold (random_state 42), OOF ROC AUC
  - 원 피처 24개 그대로, 범주형 5개는 CatBoost 자체 범주 처리(ordered target statistics), bool 은 0/1
  - early stopping 200 (검증 fold 기준 — v3/v7 과 같은 방식)
CatBoost 설정: depth 6, learning_rate 0.05, l2_leaf_reg 3, eval_metric AUC, 최대 5000 반복.
비교: v7 OOF 와 AUC, 사기 행 잔차 상관, 순위 평균 블렌드 (가중치는 전체 OOF / 한 fold 빼고 고른 정직 추정).
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import playground_s6e11_linear_baseline as L

WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
PARAMS = dict(iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=3, eval_metric="AUC",
              od_type="Iter", od_wait=200, random_seed=L.SEED, thread_count=-1, verbose=0)


def prepare(df):
    X = df.drop(columns=["id", L.TARGET], errors="ignore").copy()
    X[L.BOOLEAN] = X[L.BOOLEAN].astype(int)
    return X


def rank(p):
    return pd.Series(p).rank().to_numpy() / len(p)


def main():
    tr, te = L.load()
    y = tr[L.TARGET].astype(int).to_numpy()
    X, Xte = prepare(tr), prepare(te)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(X, y))
    fold_id = np.zeros(len(y), dtype=int)
    oof, pte, imps = np.zeros(len(y)), np.zeros(len(te)), []
    for k, (tr_i, va_i) in enumerate(splits):
        fold_id[va_i] = k
        m = CatBoostClassifier(**PARAMS)
        m.fit(Pool(X.iloc[tr_i], y[tr_i], cat_features=L.CATEGORICAL),
              eval_set=Pool(X.iloc[va_i], y[va_i], cat_features=L.CATEGORICAL), use_best_model=True)
        oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
        pte += m.predict_proba(Xte)[:, 1] / 5
        imps.append(pd.Series(m.get_feature_importance(), index=X.columns))
        print(f"fold {k}: AUC {roc_auc_score(y[va_i], oof[va_i]):.5f}  iter {m.get_best_iteration()}", flush=True)

    v7 = pd.read_csv(L.OUT / "oof_lightgbm_tuned.csv")["p"].to_numpy()
    a_cb, a_v7 = roc_auc_score(y, oof), roc_auc_score(y, v7)
    g = [roc_auc_score(y[fold_id == f], oof[fold_id == f]) - roc_auc_score(y[fold_id == f], v7[fold_id == f])
         for f in range(5)]
    print(f"\nCatBoost OOF AUC {a_cb:.5f}  logloss {log_loss(y, oof):.5f}")
    print(f"LightGBM v7 OOF AUC {a_v7:.5f}  → 차이 {a_cb - a_v7:+.5f}, CatBoost 가 이긴 fold {sum(v > 0 for v in g)}/5 "
          f"{[round(v, 5) for v in g]}")
    print(f"잔차 상관 (CatBoost vs v7): 전체 {np.corrcoef(y - oof, y - v7)[0, 1]:.3f}, "
          f"사기 행 {np.corrcoef((y - oof)[y == 1], (y - v7)[y == 1])[0, 1]:.3f}")
    k = int(y.sum())
    top = lambda p: pd.Series(p).rank(ascending=False, method="first").to_numpy() <= k
    t_cb, t_v7 = top(oof) & (y == 1), top(v7) & (y == 1)
    print(f"상위 {k}건 기준 잡은 사기: CatBoost {t_cb.sum()}, v7 {t_v7.sum()}, 둘 다 {(t_cb & t_v7).sum()}, "
          f"CatBoost 만 {(t_cb & ~t_v7).sum()}, v7 만 {(~t_cb & t_v7).sum()}")

    r_cb, r_v7 = rank(oof), rank(v7)
    curve = {float(w): roc_auc_score(y, w * r_cb + (1 - w) * r_v7) - a_v7 for w in WEIGHTS}
    print("\n순위 평균 블렌드 (CatBoost 가중치: v7 대비 변화)")
    print("  " + "  ".join(f"{w:.1f}:{v:+.5f}" for w, v in curve.items()))
    w_all = max(curve, key=curve.get)
    p_hon, ws = np.zeros(len(y)), []
    for f in range(5):
        idx = fold_id != f
        w = max(WEIGHTS, key=lambda ww: roc_auc_score(y[idx], (ww * r_cb + (1 - ww) * r_v7)[idx]))
        ws.append(float(w))
        p_hon[fold_id == f] = (w * r_cb + (1 - w) * r_v7)[fold_id == f]
    gh = [roc_auc_score(y[fold_id == f], p_hon[fold_id == f]) - roc_auc_score(y[fold_id == f], v7[fold_id == f])
          for f in range(5)]
    pb = w_all * r_cb + (1 - w_all) * r_v7
    ga = [roc_auc_score(y[fold_id == f], pb[fold_id == f]) - roc_auc_score(y[fold_id == f], v7[fold_id == f])
          for f in range(5)]
    print(f"  (a) 전체 OOF 로 고른 w={w_all}: {curve[w_all]:+.5f}  fold 상승 {sum(v > 0 for v in ga)}/5 "
          f"{[round(v, 5) for v in ga]}")
    print(f"  (b) 정직 추정: fold 평균 {np.mean(gh):+.5f}  fold 상승 {sum(v > 0 for v in gh)}/5 "
          f"{[round(v, 5) for v in gh]}  fold별 w {ws}")

    imp = pd.concat(imps, axis=1).mean(axis=1)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    print("\nCatBoost 변수 중요도 (PredictionValuesChange %, 5 fold 평균)")
    print(imp.round(2).to_string())

    pd.DataFrame({"id": tr["id"], "y": y, "p": oof, "fold": fold_id}).to_csv(L.OUT / "oof_catboost.csv", index=False)
    pd.DataFrame({"id": te["id"], L.TARGET: pte}).to_csv(L.OUT / "submission_catboost.csv", index=False)
    print("\n제출 파일 생성(제출은 안 함): output/submission_catboost.csv")


if __name__ == "__main__":
    main()
