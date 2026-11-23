"""Playground S6E11 (Predicting Credit Card Fraud) 비선형 모델(MLP 신경망) 단독 + LightGBM 구간 앙상블.

v5: MLP 단독 (v2 와 같은 전처리: log1p + 원-핫 + 표준화), 5-fold OOF
v6: LightGBM + MLP 구간 앙상블 (v4 의 선형 자리에 MLP)
    - 구간 판정·중첩 fold 방식은 v4 와 동일 (playground_s6e11_segment_ensemble.py)
    - 구간 MLP 두 가지: A 원 피처만, B 원 피처 + LightGBM logit
참고로 MLP 단독과 LightGBM 의 전체 순위 평균 블렌드도 같이 본다.

LightGBM 중첩 예측(안쪽 3-fold OOF, 바깥 검증 예측)은 output/gbdt_nested.npz 에 저장해 재사용한다.
MLP 는 sklearn MLPClassifier. 불균형 데이터에서 내장 early stopping(정확도 기준)이 의미가 없어
고정 epoch + 학습 loss 정체 시 중단으로 학습한다.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import playground_s6e11_linear_baseline as L
import playground_s6e11_lightgbm as G
import playground_s6e11_segment_ensemble as S

NESTED = L.OUT / "gbdt_nested.npz"


def make_mlp(hidden, max_iter):
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=hidden, alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
        max_iter=max_iter, n_iter_no_change=3, tol=1e-5, random_state=L.SEED))


def gbdt_nested(X, y, splits):
    if NESTED.exists():
        z = np.load(NESTED, allow_pickle=True)
        return z["p_va"], list(z["p_inner"])
    p_va_all, p_inner_all = np.zeros(len(y)), []
    for k, (tr_i, va_i) in enumerate(splits):
        p_inner = np.zeros(len(tr_i))
        for a, b in StratifiedKFold(3, shuffle=True, random_state=L.SEED + 1).split(tr_i, y[tr_i]):
            p_inner[b] = S.fit_lgb(X.iloc[tr_i[a]], y[tr_i[a]]).predict(X.iloc[tr_i[b]])
        p_va_all[va_i] = S.fit_lgb(X.iloc[tr_i], y[tr_i]).predict(X.iloc[va_i])
        p_inner_all.append(p_inner)
        print(f"  LightGBM 중첩 fold {k} 완료", flush=True)
    np.savez(NESTED, p_va=p_va_all, p_inner=np.array(p_inner_all, dtype=object))
    return p_va_all, p_inner_all


def main():
    tr, _ = L.load()
    y = tr[L.TARGET].astype(int).to_numpy()
    X = G.prepare(tr)
    XL = L.make_matrix(tr, S.LOG_COLS).astype(float)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(X, y))
    fold_id = np.zeros(len(y), dtype=int)
    for k, (_, va_i) in enumerate(splits):
        fold_id[va_i] = k

    print("[LightGBM 중첩 예측]", flush=True)
    p_gbdt, p_inner_all = gbdt_nested(X, y, splits)

    # v5: MLP 단독
    print("\n[v5_mlp 단독]", flush=True)
    p_mlp = np.zeros(len(y))
    for k, (tr_i, va_i) in enumerate(splits):
        m = make_mlp((128, 64), 30).fit(XL.iloc[tr_i], y[tr_i])
        p_mlp[va_i] = m.predict_proba(XL.iloc[va_i])[:, 1]
        print(f"  fold {k}: AUC {roc_auc_score(y[va_i], p_mlp[va_i]):.5f}  "
              f"epoch {m[-1].n_iter_}", flush=True)
    print(f"  OOF AUC {roc_auc_score(y, p_mlp):.5f}  logloss {log_loss(y, p_mlp):.5f}")
    print(f"  LightGBM(고정 669) OOF AUC {roc_auc_score(y, p_gbdt):.5f}")
    r = np.corrcoef(y - p_mlp, y - p_gbdt)[0, 1]
    r1 = np.corrcoef((y - p_mlp)[y == 1], (y - p_gbdt)[y == 1])[0, 1]
    print(f"  잔차 상관 (MLP vs LightGBM): 전체 {r:.3f}, 사기 행 {r1:.3f}  (선형 vs LightGBM 은 0.979 / 0.956)")

    ra, rb = rankdata(p_mlp) / len(y), rankdata(p_gbdt) / len(y)
    print("  전체 순위 평균 블렌드 (MLP 가중치: AUC)")
    print("  " + "  ".join(f"{w:.1f}:{roc_auc_score(y, w * ra + (1 - w) * rb):.5f}" for w in S.WEIGHTS))

    # v6: 구간 MLP
    print("\n[v6 구간 MLP]", flush=True)
    seg = {(b, v): np.full(len(y), np.nan) for b in S.BANDS for v in "AB"}
    for k, (tr_i, va_i) in enumerate(splits):
        p_inner, p_va = p_inner_all[k].astype(float), p_gbdt[va_i]
        for band in S.BANDS:
            lo, hi = band
            s_tr = (p_inner >= lo) & (p_inner <= hi)
            s_va = (p_va >= lo) & (p_va <= hi)
            Xa_tr, Xa_va = XL.iloc[tr_i[s_tr]], XL.iloc[va_i[s_va]]
            seg[(band, "A")][va_i[s_va]] = make_mlp((64, 32), 50).fit(
                Xa_tr, y[tr_i[s_tr]]).predict_proba(Xa_va)[:, 1]
            Xb_tr = Xa_tr.assign(gbdt_logit=S.logit(p_inner[s_tr]))
            Xb_va = Xa_va.assign(gbdt_logit=S.logit(p_va[s_va]))
            seg[(band, "B")][va_i[s_va]] = make_mlp((64, 32), 50).fit(
                Xb_tr, y[tr_i[s_tr]]).predict_proba(Xb_va)[:, 1]
        print(f"  fold {k} 구간 MLP 완료", flush=True)
    S.evaluate(y, p_gbdt, seg, fold_id, name="MLP", tag="segment_mlp")

    pd.DataFrame({"id": tr["id"], "y": y, "p": p_mlp, "fold": fold_id}).to_csv(L.OUT / "oof_mlp.csv", index=False)


if __name__ == "__main__":
    main()
