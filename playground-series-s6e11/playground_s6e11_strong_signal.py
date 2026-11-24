"""Playground S6E11 (Predicting Credit Card Fraud) 강한 신호 전문 모델 + 본 모델 앙상블 (v13).

강한 신호 피처 (7개): cvv_retry_count, auth_method, is_foreign_transaction, used_vpn, ip_country_mismatch,
                     billing_shipping_mismatch, is_ai_generated_scam_attempt
두 가지 '따로 모으기'
  S_col: 강한 신호 피처 7개만으로 전체 행 학습 (열을 모음)
  S_row: 강한 신호가 하나 이상 켜진 행만으로 전 피처 학습 (행을 모음). 그 밖의 행은 본 모델 그대로.
본 모델 = v7 (튜닝 LightGBM) OOF.
결합
  S_col: 순위 평균 w·rank(S_col) + (1−w)·rank(v7)
  S_row: 강한 신호 행 안에서 확률 가중 평균 w·S_row + (1−w)·v7, 밖은 v7
가중치는 (a) 전체 OOF 에서 고른 값, (b) 한 fold 빼고 고른 값을 빠진 fold 에 적용(정직한 추정).
검증: 기존과 같은 5-fold. early stopping 은 학습 부분 안 10% 홀드아웃으로 (검증 fold 미사용).
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import playground_s6e11_lightgbm as G
import playground_s6e11_linear_baseline as L

STRONG_BOOL = ["is_foreign_transaction", "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch",
               "is_ai_generated_scam_attempt"]
STRONG = ["cvv_retry_count", "auth_method"] + STRONG_BOOL
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)


def params(n_rows):
    p = {**G.PARAMS, **json.loads((L.OUT / "best_params_lightgbm.json").read_text(encoding="utf-8"))}
    p.update(learning_rate=0.05, min_child_samples=int(min(994, max(30, n_rows // 100))))
    return p


def strong_any(df):
    return ((df.cvv_retry_count > 0) | (df.auth_method == "No Authentication") |
            df[STRONG_BOOL].any(axis=1)).to_numpy()


def train_oof(X, y, splits, mask):
    oof = np.full(len(y), np.nan)
    for k, (tr_i, va_i) in enumerate(splits):
        rows = tr_i[mask[tr_i]]
        a, b = train_test_split(rows, test_size=0.1, stratify=y[rows], random_state=L.SEED)
        dtr = lgb.Dataset(X.iloc[a], y[a])
        dho = lgb.Dataset(X.iloc[b], y[b], reference=dtr)
        m = lgb.train(params(len(a)), dtr, 5000, valid_sets=[dho], callbacks=[lgb.early_stopping(200, verbose=False)])
        v = va_i[mask[va_i]]
        oof[v] = m.predict(X.iloc[v], num_iteration=m.best_iteration)
        print(f"    fold {k}: 학습 {len(a)}행, 검증 AUC {roc_auc_score(y[v], oof[v]):.4f}, iter {m.best_iteration}",
              flush=True)
    return oof


def rank(p):
    return pd.Series(p).rank().to_numpy() / len(p)


def honest(y, p_main, combine, fold_id):
    """combine(w) → 결합 예측. (a) 전체에서 고른 w, (b) 한 fold 빼고 고른 w"""
    base = roc_auc_score(y, p_main)
    curve = {float(w): roc_auc_score(y, combine(w)) - base for w in WEIGHTS}
    print("  전체 AUC 변화: " + "  ".join(f"{w:.1f}:{g:+.5f}" for w, g in curve.items()))
    w_all = max(curve, key=curve.get)
    pa = combine(w_all)
    g_all = [roc_auc_score(y[fold_id == f], pa[fold_id == f]) - roc_auc_score(y[fold_id == f], p_main[fold_id == f])
             for f in range(5)]
    p_hon, ws = p_main.copy(), []
    for f in range(5):
        idx = fold_id != f
        w = max(WEIGHTS, key=lambda ww: roc_auc_score(y[idx], combine(ww)[idx]))
        ws.append(float(w))
        p_hon[fold_id == f] = combine(w)[fold_id == f]
    # 순위 결합은 fold 마다 척도가 같아 그대로 이어 붙여도 됨
    g_hon = [roc_auc_score(y[fold_id == f], p_hon[fold_id == f]) - roc_auc_score(y[fold_id == f], p_main[fold_id == f])
             for f in range(5)]
    print(f"  (a) 전체 OOF 로 고른 w={w_all}: gain {curve[w_all]:+.5f}  fold 상승 {sum(g > 0 for g in g_all)}/5  "
          f"{[round(g, 5) for g in g_all]}")
    print(f"  (b) 정직 추정: gain {np.mean(g_hon):+.5f} (fold 평균)  fold 상승 {sum(g > 0 for g in g_hon)}/5  "
          f"{[round(g, 5) for g in g_hon]}  fold별 w {ws}")
    return {"w_all": w_all, "gain_all": curve[w_all], "gain_honest_fold_mean": float(np.mean(g_hon)),
            "folds_up_honest": sum(g > 0 for g in g_hon)}


def main():
    tr, _ = L.load()
    y = tr[L.TARGET].astype(int).to_numpy()
    X = G.prepare(tr)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(X, y))
    fold_id = np.zeros(len(y), dtype=int)
    for k, (_, va_i) in enumerate(splits):
        fold_id[va_i] = k
    p_main = pd.read_csv(L.OUT / "oof_lightgbm_tuned.csv")["p"].to_numpy()
    print(f"본 모델 v7 OOF AUC {roc_auc_score(y, p_main):.5f}")
    seg = strong_any(tr)
    print(f"강한 신호 1개 이상 행: {seg.sum()} ({seg.mean():.1%}), 그 안 사기 {y[seg].sum()} "
          f"({y[seg].mean():.2%}) = 전체 사기의 {y[seg].sum() / y.sum():.1%}")
    print(f"강한 신호 피처 7개의 서로 다른 값 조합 수: {tr[STRONG].drop_duplicates().shape[0]}")

    res = []
    print("\n[S_col] 강한 신호 피처 7개만, 전체 행")
    p_col = train_oof(X[STRONG], y, splits, np.ones(len(y), bool))
    print(f"  S_col 단독 AUC {roc_auc_score(y, p_col):.5f}, 사기 행 잔차 상관 "
          f"{np.corrcoef((1 - p_col)[y == 1], (1 - p_main)[y == 1])[0, 1]:.3f}")
    r_main, r_col = rank(p_main), rank(p_col)
    res.append({"model": "S_col", "alone_auc": roc_auc_score(y, p_col),
                **honest(y, p_main, lambda w: w * r_col + (1 - w) * r_main, fold_id)})

    print("\n[S_row] 강한 신호 행만, 전 피처")
    p_row = train_oof(X, y, splits, seg)
    print(f"  구간 안 AUC: 본 모델 {roc_auc_score(y[seg], p_main[seg]):.4f} / S_row {roc_auc_score(y[seg], p_row[seg]):.4f}")

    def comb_row(w):
        p = p_main.copy()
        p[seg] = w * p_row[seg] + (1 - w) * p_main[seg]
        return p
    res.append({"model": "S_row", "alone_auc": roc_auc_score(y[seg], p_row[seg]),
                **honest(y, p_main, comb_row, fold_id)})

    pd.DataFrame(res).to_csv(L.OUT / "strong_signal_results.csv", index=False)
    pd.DataFrame({"id": tr["id"], "y": y, "p_main": p_main, "p_col": p_col, "p_row": p_row, "strong": seg,
                  "fold": fold_id}).to_csv(L.OUT / "oof_strong_signal.csv", index=False)


if __name__ == "__main__":
    main()
