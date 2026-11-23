"""Playground S6E11 (Predicting Credit Card Fraud) 놓친 사기 5유형 전문 모델 앙상블 (v9).

놓친 사기 군집(playground_s6e11_missed_fraud_clusters.py)의 5유형을 정답 없이 규칙으로 정의한 구간.
A 강신호 = CVV 재입력≥1, 인증 없음, 해외 거래, VPN, IP 국가 불일치, 청구·배송지 불일치, AI 생성 사기 시도
  T0 약한 신호형   : A 강신호 0개, 고위험 업종 아님
  T1 CVV형        : A 강신호가 CVV 재입력 하나뿐
  T2 무인증형      : A 강신호가 인증 없음 하나뿐
  T3 고위험 업종형  : A 강신호 0개, 가맹점 위험 ≥55 & 업종 ∈ {Gaming, Gift Cards, Electronics, Crypto Exchange}
  T4 청구·배송형   : A 강신호가 청구·배송지 불일치 하나뿐
  나머지(신호 2개 이상 등)는 본 모델 그대로.
구간마다 그 구간 행만으로 LightGBM 전문 모델(전 피처)을 학습하고, 구간 안에서 본 모델(v7)과 확률 가중 평균.
검증: 같은 5-fold OOF. 가중치는 (a) 전체 OOF 에서 고른 값과 (b) 한 fold 를 빼고 나머지 4 fold 에서 고른 값을
     빠진 fold 에 적용한 값(정직한 추정) 둘 다 보고.
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import playground_s6e11_lightgbm as G
import playground_s6e11_linear_baseline as L

STRONG = ["is_foreign_transaction", "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch",
          "is_ai_generated_scam_attempt"]
HIGH_CAT = ["Gaming", "Gift Cards", "Electronics", "Crypto Exchange"]
SEGS = ["T0_weak", "T1_cvv", "T2_noauth", "T3_highrisk", "T4_billing"]
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)


def segment(df):
    cvv = df.cvv_retry_count > 0
    na = df.auth_method == "No Authentication"
    n = cvv.astype(int) + na.astype(int) + df[STRONG].astype(int).sum(axis=1)
    hi = (df.merchant_risk_score >= 55) & df.merchant_category.isin(HIGH_CAT)
    s = pd.Series("other", index=df.index)
    s[(n == 0) & ~hi] = "T0_weak"
    s[(n == 0) & hi] = "T3_highrisk"
    s[(n == 1) & cvv] = "T1_cvv"
    s[(n == 1) & na] = "T2_noauth"
    s[(n == 1) & df.billing_shipping_mismatch] = "T4_billing"
    return s.to_numpy()


def params_for(n_rows):
    p = {**G.PARAMS, **json.loads((L.OUT / "best_params_lightgbm.json").read_text(encoding="utf-8"))}
    p.update(learning_rate=0.05, min_child_samples=int(min(994, max(30, n_rows // 100))))
    return p


def train_segment(X, y, splits, mask, Xte=None, mask_te=None):
    params = params_for(int(mask.sum()))
    oof = np.full(len(y), np.nan)
    pte = np.zeros(int(mask_te.sum())) if Xte is not None else None
    for k, (tr_i, va_i) in enumerate(splits):
        a, b = tr_i[mask[tr_i]], va_i[mask[va_i]]
        dtr = lgb.Dataset(X.iloc[a], y[a])
        dva = lgb.Dataset(X.iloc[b], y[b], reference=dtr)
        m = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
        if Xte is not None:
            pte += m.predict(Xte[mask_te], num_iteration=m.best_iteration) / len(splits)
        print(f"    fold {k}: 구간 AUC {roc_auc_score(y[b], oof[b]):.4f}  iter {m.best_iteration}", flush=True)
    return oof, pte


def blend(p_main, spec, seg, w):
    p = p_main.copy()
    for s, ww in w.items():
        k = seg == s
        p[k] = ww * spec[s][k] + (1 - ww) * p_main[k]
    return p


def choose(y, p_main, spec, seg, idx):
    """idx 행들에서 구간별 가중치를 차례로(좌표 탐색) 선택"""
    w = {s: 0.0 for s in SEGS}
    for _ in range(2):
        for s in SEGS:
            best = max(WEIGHTS, key=lambda ww: roc_auc_score(
                y[idx], blend(p_main, spec, seg, {**w, s: ww})[idx]))
            w[s] = float(best)
    return w


def main():
    tr, te = L.load()
    y = tr[L.TARGET].astype(int).to_numpy()
    X, Xte = G.prepare(tr), G.prepare(te)
    Xte[L.CATEGORICAL] = Xte[L.CATEGORICAL].apply(lambda s: s.cat.set_categories(X[s.name].cat.categories))
    splits = list(StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(X, y))
    fold_id = np.zeros(len(y), dtype=int)
    for k, (_, va_i) in enumerate(splits):
        fold_id[va_i] = k
    seg, seg_te = segment(tr), segment(te)

    main_oof = pd.read_csv(L.OUT / "oof_lightgbm_tuned.csv")
    p_main = main_oof["p"].to_numpy()
    p_main_te = pd.read_csv(L.OUT / "submission_lightgbm_tuned.csv")[L.TARGET].to_numpy()
    base = roc_auc_score(y, p_main)
    print(f"본 모델 v7 OOF AUC {base:.5f}\n")

    spec, spec_te = {}, {}
    summary = []
    for s in SEGS:
        k = seg == s
        print(f"[{s}] 행 {k.sum()}, 사기 {y[k].sum()} ({y[k].mean():.2%})", flush=True)
        oof, pte = train_segment(X, y, splits, k, Xte, seg_te == s)
        spec[s] = oof
        full = np.full(len(te), np.nan)
        full[seg_te == s] = pte
        spec_te[s] = full
        a_main, a_spec = roc_auc_score(y[k], p_main[k]), roc_auc_score(y[k], oof[k])
        r = np.corrcoef((y - oof)[k & (y == 1)], (y - p_main)[k & (y == 1)])[0, 1]
        curve = {float(w): roc_auc_score(y, blend(p_main, spec, seg, {s: w})) - base for w in WEIGHTS}
        print(f"  구간 AUC 본 {a_main:.4f} / 전문 {a_spec:.4f}, 사기 행 잔차 상관 {r:.3f}")
        print("  이 구간만 섞을 때 전체 AUC 변화: " + "  ".join(f"{w:.1f}:{g:+.5f}" for w, g in curve.items()))
        summary.append({"segment": s, "rows": int(k.sum()), "fraud": int(y[k].sum()), "auc_main": a_main,
                        "auc_spec": a_spec, "resid_corr_fraud": r,
                        "best_w": max(curve, key=curve.get), "best_gain": max(curve.values())})

    # (a) 전체 OOF 에서 고른 가중치
    w_all = choose(y, p_main, spec, seg, np.arange(len(y)))
    p_all = blend(p_main, spec, seg, w_all)
    g_all = [roc_auc_score(y[fold_id == f], p_all[fold_id == f]) - roc_auc_score(y[fold_id == f], p_main[fold_id == f])
             for f in range(5)]
    print(f"\n(a) 전체 OOF 로 고른 가중치 {w_all}")
    print(f"    AUC {roc_auc_score(y, p_all):.5f}  gain {roc_auc_score(y, p_all) - base:+.5f}  "
          f"fold 상승 {sum(g > 0 for g in g_all)}/5  fold별 {[round(g, 5) for g in g_all]}")

    # (b) 한 fold 빼고 고른 가중치 → 빠진 fold 에 적용
    p_hon = p_main.copy()
    ws = []
    for f in range(5):
        w = choose(y, p_main, spec, seg, np.where(fold_id != f)[0])
        ws.append(w)
        k = fold_id == f
        p_hon[k] = blend(p_main, spec, seg, w)[k]
    g_hon = [roc_auc_score(y[fold_id == f], p_hon[fold_id == f]) - roc_auc_score(y[fold_id == f], p_main[fold_id == f])
             for f in range(5)]
    print(f"(b) 빠진 fold 정직 추정: AUC {roc_auc_score(y, p_hon):.5f}  gain {roc_auc_score(y, p_hon) - base:+.5f}  "
          f"fold 상승 {sum(g > 0 for g in g_hon)}/5  fold별 {[round(g, 5) for g in g_hon]}")
    print("    fold별 선택 가중치:", ws)

    pd.DataFrame(summary).to_csv(L.OUT / "type_specialists_summary.csv", index=False)
    pd.DataFrame({"id": tr["id"], "y": y, "p_main": p_main, "segment": seg, **{f"spec_{s}": spec[s] for s in SEGS},
                  "p_blend": p_all, "fold": fold_id}).to_csv(L.OUT / "oof_type_specialists.csv", index=False)
    p_te = blend(p_main_te, spec_te, seg_te, w_all)
    pd.DataFrame({"id": te["id"], L.TARGET: p_te}).to_csv(L.OUT / "submission_type_specialists.csv", index=False)
    (L.OUT / "type_specialists_weights.json").write_text(json.dumps(w_all), encoding="utf-8")
    print("\n제출 파일 생성(제출은 안 함): output/submission_type_specialists.csv")


if __name__ == "__main__":
    main()
