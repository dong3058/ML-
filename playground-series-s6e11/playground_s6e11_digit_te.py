"""Playground S6E11 (Predicting Credit Card Fraud) 값·자릿수 target encoding (v10, 대회 전용 기법).

주의: 이 기법은 실제 사기 패턴이 아니라 합성 데이터 생성 과정의 흔적(특정 값·자릿수에 사기가 몰림)을 이용한다.
     실제 서비스 모델에는 쓰지 않는다. '정석' 기준 모델은 v7(playground_s6e11_lightgbm_tuning.py).

인코딩 키
  소수형 6개 (amount_usd, account_balance_usd, hours_since_last_txn, distance_from_home_km,
             velocity_score, merchant_risk_score) × 4종
     exact  : 값 그대로 (센트 단위 정수로 변환)
     int    : 정수 부분
     cents  : 소수 둘째 자리까지의 끝자리 (값×100 mod 100)
     lead2  : 앞 두 유효 숫자
  정수형 5개 (card_age_months, customer_age, txn_count_last_24h, time_of_day_hour, day_of_week) × exact
  → 29개 피처. 값 = (그 키의 사기 수 + m·전체 사기율) / (그 키의 행 수 + m), m = 20

누수 방지 (중첩)
  바깥 5-fold(v7 과 같은 분할) 마다
    - 학습 행의 인코딩: 학습 부분 안에서 다시 5-fold OOF 로 계산 (자기 정답이 섞이지 않게)
    - 검증·테스트 행의 인코딩: 학습 부분 전체로 계산
  모델: v7 튜닝 파라미터 LightGBM (learning_rate 0.03, early stopping 200) — v7 과 같은 조건 비교
결과: output/oof_digit_te.csv, output/submission_digit_te.csv (제출은 하지 않음), 변수 중요도
"""

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import playground_s6e11_lightgbm as G
import playground_s6e11_linear_baseline as L

FLOAT_COLS = ["amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km",
              "velocity_score", "merchant_risk_score"]
INT_COLS = ["card_age_months", "customer_age", "txn_count_last_24h", "time_of_day_hour", "day_of_week"]
M_SMOOTH = 20


def lead2(x):
    x = np.abs(x.to_numpy(dtype=float))
    out = np.zeros(len(x), dtype=np.int64)
    nz = x > 0
    e = np.floor(np.log10(x[nz]))
    out[nz] = np.floor(x[nz] / 10 ** (e - 1) + 1e-9).astype(np.int64)
    return out


def make_keys(df):
    k = {}
    for c in FLOAT_COLS:
        cents_total = np.round(df[c].to_numpy(dtype=float) * 100).astype(np.int64)
        k[f"{c}__exact"] = cents_total
        k[f"{c}__int"] = cents_total // 100
        k[f"{c}__cents"] = cents_total % 100
        k[f"{c}__lead2"] = lead2(df[c])
    for c in INT_COLS:
        k[f"{c}__exact"] = df[c].to_numpy(dtype=np.int64)
    return pd.DataFrame(k, index=df.index)


def te_map(key, y, prior):
    g = pd.DataFrame({"k": key, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + M_SMOOTH * prior) / (g["count"] + M_SMOOTH)


def te_apply(keys_fit, y_fit, keys_apply):
    prior = y_fit.mean()
    out = {}
    for c in keys_fit.columns:
        mp = te_map(keys_fit[c].to_numpy(), y_fit, prior)
        out["te_" + c] = pd.Series(keys_apply[c].to_numpy()).map(mp).fillna(prior).to_numpy()
    return pd.DataFrame(out, index=keys_apply.index)


def te_oof(keys, y, seed):
    out = pd.DataFrame(index=keys.index, columns=["te_" + c for c in keys.columns], dtype=float)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(keys, y):
        out.iloc[b] = te_apply(keys.iloc[a], y[a], keys.iloc[b]).to_numpy()
    return out


# 효과 분해용 키 묶음
B_FLOATS = ["merchant_risk_score", "velocity_score", "amount_usd", "hours_since_last_txn"]
GROUPS = {
    "all": [f"{c}__{k}" for c in FLOAT_COLS for k in ("exact", "int", "cents", "lead2")]
           + [f"{c}__exact" for c in INT_COLS],
    # (a) 의미 있는 B 피처를 거칠게 구간화한 것 (정수 부분·앞 두 자리, 정수형 B 는 값 그대로)
    "b_coarse": [f"{c}__{k}" for c in B_FLOATS for k in ("int", "lead2")]
                + ["customer_age__exact", "txn_count_last_24h__exact"],
    # (b) 합성 흔적 후보: 센트·정확값 전부 + 사기와 무관한 피처의 키 전부
    "artifact": [f"{c}__{k}" for c in FLOAT_COLS for k in ("cents", "exact")]
                + [f"{c}__{k}" for c in ("account_balance_usd", "distance_from_home_km") for k in ("int", "lead2")]
                + ["card_age_months__exact", "time_of_day_hour__exact", "day_of_week__exact"],
    # (c) 순수 합성 흔적: 의미 없는 키만 (센트 단위 전부 + 사기와 무관한 피처의 키). B 피처의 값·정수·앞자리는 제외
    "pure_artifact": [f"{c}__cents" for c in FLOAT_COLS]
                     + [f"{c}__{k}" for c in ("account_balance_usd", "distance_from_home_km")
                        for k in ("exact", "int", "lead2")]
                     + ["card_age_months__exact", "time_of_day_hour__exact", "day_of_week__exact"],
    # (a') 순수 정석: 업무적으로 설명되는 구간만. 앞 두 자리(크기 단위가 섞임)·나이 1세 단위 제외
    #      B 연속형 4개의 정수 부분 + 20분위 구간, 나이 5세 구간, 24시간 거래 수 값
    "pure_coarse": [f"{c}__int" for c in B_FLOATS] + [f"{c}__q20" for c in B_FLOATS]
                   + ["customer_age__age5", "txn_count_last_24h__exact"],
}


def add_business_keys(K, Kte, tr, te):
    """학습 데이터 분위수 경계로 20분위 구간, 나이 5세 구간 키 추가 (정답 미사용)"""
    for c in B_FLOATS:
        edges = np.unique(np.quantile(tr[c], np.linspace(0, 1, 21)))
        K[f"{c}__q20"] = np.searchsorted(edges[1:-1], tr[c].to_numpy(), side="right")
        Kte[f"{c}__q20"] = np.searchsorted(edges[1:-1], te[c].to_numpy(), side="right")
    K["customer_age__age5"] = tr["customer_age"].to_numpy() // 5
    Kte["customer_age__age5"] = te["customer_age"].to_numpy() // 5
    return K, Kte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=list(GROUPS))
    ap.add_argument("--ratio", action="store_true", help="가맹점 위험 점수 / (거래 속도 점수 + 1) 피처 추가 (v14)")
    ap.add_argument("--no-te", action="store_true", help="TE 없이 v7 + 추가 피처만")
    args = ap.parse_args()
    tag = ("" if args.group == "all" else "_" + args.group)
    if args.no_te:
        tag = "_v7"
    if args.ratio:
        tag += "_ratio"
    tr, te = L.load()
    y = tr[L.TARGET].astype(int).to_numpy()
    X, Xte = G.prepare(tr), G.prepare(te)
    if args.ratio:
        # 거래 속도 점수는 0 인 행이 있어 분모에 +1
        for d, src in [(X, tr), (Xte, te)]:
            d["risk_per_velocity"] = src.merchant_risk_score / (src.velocity_score + 1)
    Xte[L.CATEGORICAL] = Xte[L.CATEGORICAL].apply(lambda s: s.cat.set_categories(X[s.name].cat.categories))
    K, Kte = make_keys(tr), make_keys(te)
    K, Kte = add_business_keys(K, Kte, tr, te)
    if GROUPS[args.group] is not None:
        K, Kte = K[GROUPS[args.group]], Kte[GROUPS[args.group]]
    print(f"[{args.group}] 키 {K.shape[1]}개", flush=True)
    print("키별 고유값 수:", {c: int(K[c].nunique()) for c in K.columns}, "\n", flush=True)

    params = {**G.PARAMS, **json.loads((L.OUT / "best_params_lightgbm.json").read_text(encoding="utf-8")),
              "learning_rate": 0.03}
    splits = list(StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(X, y))
    oof, pte, imps = np.zeros(len(y)), np.zeros(len(te)), []
    for k, (tr_i, va_i) in enumerate(splits):
        if args.no_te:
            te_tr, te_va, te_te = (pd.DataFrame(index=range(n)) for n in (len(tr_i), len(va_i), len(te)))
        else:
            te_tr = te_oof(K.iloc[tr_i], y[tr_i], seed=L.SEED + 100 + k)
            te_va = te_apply(K.iloc[tr_i], y[tr_i], K.iloc[va_i])
            te_te = te_apply(K.iloc[tr_i], y[tr_i], Kte)
        Xa = pd.concat([X.iloc[tr_i].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
        Xb = pd.concat([X.iloc[va_i].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
        Xt = pd.concat([Xte.reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)
        dtr = lgb.Dataset(Xa, y[tr_i])
        dva = lgb.Dataset(Xb, y[va_i], reference=dtr)
        m = lgb.train(params, dtr, 6000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_i] = m.predict(Xb, num_iteration=m.best_iteration)
        pte += m.predict(Xt, num_iteration=m.best_iteration) / 5
        imps.append(pd.Series(m.feature_importance("gain"), index=Xa.columns))
        print(f"fold {k}: AUC {roc_auc_score(y[va_i], oof[va_i]):.5f}  iter {m.best_iteration}", flush=True)

    v7 = pd.read_csv(L.OUT / "oof_lightgbm_tuned.csv")["p"].to_numpy()
    fold_id = np.zeros(len(y), dtype=int)
    for k, (_, va_i) in enumerate(splits):
        fold_id[va_i] = k
    gains = [roc_auc_score(y[fold_id == f], oof[fold_id == f]) - roc_auc_score(y[fold_id == f], v7[fold_id == f])
             for f in range(5)]
    ref = L.OUT / ("oof_lightgbm_tuned.csv" if args.no_te else f"oof_digit_te{tag.replace('_ratio', '')}.csv")
    if args.ratio and ref.exists():
        r = pd.read_csv(ref)["p"].to_numpy()
        g2 = [roc_auc_score(y[fold_id == f], oof[fold_id == f]) - roc_auc_score(y[fold_id == f], r[fold_id == f])
              for f in range(5)]
        print(f"\n[비율 피처 효과] 같은 모델 비율 피처 없음({ref.name}) {roc_auc_score(y, r):.5f} → 있음 "
              f"{roc_auc_score(y, oof):.5f}  변화 {roc_auc_score(y, oof) - roc_auc_score(y, r):+.5f}  "
              f"fold 상승 {sum(g > 0 for g in g2)}/5  {[round(g, 5) for g in g2]}")
    print(f"\n이번 OOF AUC {roc_auc_score(y, oof):.5f}  logloss {log_loss(y, oof):.5f}")
    print(f"v7  OOF AUC {roc_auc_score(y, v7):.5f}  → 변화 {roc_auc_score(y, oof) - roc_auc_score(y, v7):+.5f}, "
          f"fold 상승 {sum(g > 0 for g in gains)}/5, fold별 {[round(g, 5) for g in gains]}")

    imp = pd.concat(imps, axis=1).mean(axis=1)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    print("\n변수 중요도 (gain %, 5 fold 평균) 상위 25")
    print(imp.head(25).round(2).to_string())
    te_share = imp[imp.index.str.startswith("te_")].sum()
    if args.ratio:
        print(f"risk_per_velocity 중요도 {imp.get('risk_per_velocity', 0):.2f}% "
              f"(순위 {list(imp.index).index('risk_per_velocity') + 1}/{len(imp)})")
    print(f"\nTE 피처 중요도 합 {te_share:.1f}%")
    by_kind = imp[imp.index.str.startswith("te_")].groupby(lambda x: x.split("__")[-1]).sum()
    print("TE 종류별 합:", by_kind.round(2).to_dict())

    imp.to_csv(L.OUT / f"digit_te_importance{tag}.csv", header=["gain_pct"])
    pd.DataFrame({"id": tr["id"], "y": y, "p": oof}).to_csv(L.OUT / f"oof_digit_te{tag}.csv", index=False)
    pd.DataFrame({"id": te["id"], L.TARGET: pte}).to_csv(L.OUT / f"submission_digit_te{tag}.csv", index=False)
    print(f"\n제출 파일 생성(제출은 안 함): output/submission_digit_te{tag}.csv")


if __name__ == "__main__":
    main()
