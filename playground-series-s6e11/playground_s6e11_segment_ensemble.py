"""Playground S6E11 (Predicting Credit Card Fraud) 구간 분리 앙상블 (v4).

아이디어: 두 모델이 모두 놓친 사기는 대부분 LightGBM 예측 확률이 애매한 구간(약 2~22%)에 있다.
이 구간만 따로 떼어 선형 모델(로지스틱 회귀)을 학습하고, 구간 안에서는 LightGBM 과 섞고,
구간 밖은 LightGBM 그대로 둔다.

누수 방지 (중첩 fold)
  바깥 5-fold (v1~v3 과 같은 분할). 각 바깥 fold 에서
   1) 학습 부분만으로 안쪽 3-fold LightGBM OOF 예측 → 학습 행의 '구간' 판정에 사용
   2) 학습 부분 전체로 LightGBM 학습 → 검증 행 예측 + 검증 행의 '구간' 판정
   3) 구간 학습 행만으로 선형 모델 학습 → 구간 검증 행 예측
  LightGBM 은 early stopping 없이 고정 반복수(v3 평균 669)로 학습해 검증 행 정보를 쓰지 않는다.

선형 모델 두 가지
  A: 원 피처만 (v2 와 같은 전처리, log1p 포함)
  B: 원 피처 + LightGBM 예측의 logit (구간 안에서 LightGBM 을 보정하는 형태)
결합: 구간 안 = w·선형 + (1−w)·LightGBM, 구간 밖 = LightGBM
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import playground_s6e11_lightgbm as G
import playground_s6e11_linear_baseline as L

N_ROUNDS = 669
LOG_COLS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
            "cvv_retry_count", "prior_disputes"]
BANDS = [(0.01, 0.22), (0.02, 0.22), (0.05, 0.22), (0.02, 0.40)]
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
PARAMS = {**G.PARAMS, "metric": "None"}


def fit_lgb(X, y):
    return lgb.train(PARAMS, lgb.Dataset(X, y), N_ROUNDS)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_linear(Xm, y):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    return m.fit(Xm, y)


def run():
    tr, _ = L.load()
    y = tr[L.TARGET].astype(int).to_numpy()
    X = G.prepare(tr)
    XL = L.make_matrix(tr, LOG_COLS)

    p_gbdt = np.zeros(len(y))
    lin = {(b, v): np.full(len(y), np.nan) for b in BANDS for v in "AB"}
    fold_id = np.zeros(len(y), dtype=int)

    outer = StratifiedKFold(5, shuffle=True, random_state=L.SEED)
    for k, (tr_i, va_i) in enumerate(outer.split(X, y)):
        fold_id[va_i] = k
        # 1) 안쪽 OOF (학습 행 구간 판정용)
        p_inner = np.zeros(len(tr_i))
        inner = StratifiedKFold(3, shuffle=True, random_state=L.SEED + 1)
        for a, b in inner.split(tr_i, y[tr_i]):
            m = fit_lgb(X.iloc[tr_i[a]], y[tr_i[a]])
            p_inner[b] = m.predict(X.iloc[tr_i[b]])
        # 2) 바깥 검증 예측
        m = fit_lgb(X.iloc[tr_i], y[tr_i])
        p_va = m.predict(X.iloc[va_i])
        p_gbdt[va_i] = p_va
        # 3) 구간별 선형 모델
        for band in BANDS:
            lo, hi = band
            s_tr = (p_inner >= lo) & (p_inner <= hi)
            s_va = (p_va >= lo) & (p_va <= hi)
            Xa_tr, Xa_va = XL.iloc[tr_i[s_tr]], XL.iloc[va_i[s_va]]
            lin[(band, "A")][va_i[s_va]] = fit_linear(Xa_tr, y[tr_i[s_tr]]).predict_proba(Xa_va)[:, 1]
            Xb_tr = Xa_tr.assign(gbdt_logit=logit(p_inner[s_tr]))
            Xb_va = Xa_va.assign(gbdt_logit=logit(p_va[s_va]))
            lin[(band, "B")][va_i[s_va]] = fit_linear(Xb_tr, y[tr_i[s_tr]]).predict_proba(Xb_va)[:, 1]
        print(f"fold {k}: LightGBM AUC {roc_auc_score(y[va_i], p_va):.5f}", flush=True)
    return tr, y, p_gbdt, lin, fold_id


def evaluate(y, p_gbdt, lin, fold_id, name="선형", tag="segment_ensemble"):
    base = roc_auc_score(y, p_gbdt)
    print(f"\n[기준] LightGBM 고정 {N_ROUNDS} 반복 OOF AUC {base:.5f}")
    rows, best = [], None
    for (band, v), p_lin in lin.items():
        seg = ~np.isnan(p_lin)
        auc_seg_g = roc_auc_score(y[seg], p_gbdt[seg])
        auc_seg_l = roc_auc_score(y[seg], p_lin[seg])
        for w in WEIGHTS:
            p = p_gbdt.copy()
            p[seg] = w * p_lin[seg] + (1 - w) * p_gbdt[seg]
            auc = roc_auc_score(y, p)
            folds = [roc_auc_score(y[fold_id == f], p[fold_id == f]) -
                     roc_auc_score(y[fold_id == f], p_gbdt[fold_id == f]) for f in range(5)]
            rows.append({"band": f"{band[0]:.2f}~{band[1]:.2f}", "linear": v, "w": w,
                         "seg_rows": int(seg.sum()), "seg_fraud": int(y[seg].sum()),
                         "seg_auc_gbdt": auc_seg_g, "seg_auc_linear": auc_seg_l,
                         "auc": auc, "gain": auc - base,
                         "folds_up": sum(f > 0 for f in folds), "fold_gain_min": min(folds)})
    t = pd.DataFrame(rows)
    print(f"\n[구간별 요약] 구간 안 AUC 비교 (LightGBM vs {name})")
    print(t[t.w == 0].drop(columns=["w", "auc", "gain", "folds_up", "fold_gain_min"])
          .round(5).to_string(index=False))
    print("\n[각 구간·선형 버전별 최고 가중치] gain = 전체 OOF AUC 변화, folds_up = 5 fold 중 오른 fold 수")
    bt = t.loc[t.groupby(["band", "linear"]).auc.idxmax()].sort_values("auc", ascending=False)
    print(bt[["band", "linear", "w", "auc", "gain", "folds_up", "fold_gain_min"]].round(5).to_string(index=False))
    best = bt.iloc[0]
    print(f"\n[가중치 곡선] {best.band} / {name} {best.linear}")
    cur = t[(t.band == best.band) & (t.linear == best.linear)]
    print("  " + "  ".join(f"{w:.1f}:{a:+.5f}" for w, a in zip(cur.w, cur.gain)))
    t.to_csv(L.OUT / f"{tag}_grid.csv", index=False)
    return best


def main():
    tr, y, p_gbdt, lin, fold_id = run()
    best = evaluate(y, p_gbdt, lin, fold_id)
    band = tuple(float(x) for x in best.band.split("~"))
    key = min(lin, key=lambda kv: abs(kv[0][0] - band[0]) + abs(kv[0][1] - band[1]) + (kv[1] != best.linear))
    pd.DataFrame({"id": tr["id"], "y": y, "p_gbdt": p_gbdt, "p_linear_seg": lin[key], "fold": fold_id}).to_csv(
        L.OUT / "oof_segment.csv", index=False)


if __name__ == "__main__":
    main()
