"""Playground S6E11 (Predicting Credit Card Fraud) 놓친 사기 부드러운 라벨 (v12).

답지: 잡힌 사기 1, 정상 0, 놓친 사기 0.5 로 바꿔 본 모델(v7 파라미터) 하나를 학습한다.
LightGBM 기본 binary 목적함수는 0/1 만 받으므로 0~1 값을 받는 cross_entropy 목적함수를 쓴다.
v12b: 답지는 0/1 그대로 두고 놓친 사기 행의 학습 가중치만 올리는 변형 (인자 w2, w3 …).
공정 비교를 위해 같은 목적함수·같은 절차에 0/1 답지를 쓴 기준선도 함께 학습한다.

누수 방지: 바깥 5-fold (기존과 같은 분할). 학습 행의 '놓침' 판정은 그 학습 행만으로 만든
안쪽 3-fold LightGBM OOF 예측(output/gbdt_nested.npz)으로 한다.
early stopping 은 학습 행 안 10% 홀드아웃의 원래 0/1 답지 AUC 로 한다 (검증 fold 미사용).
평가는 검증 fold 의 원래 답지(is_fraud) AUC.
"""

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import playground_s6e11_lightgbm as G
import playground_s6e11_linear_baseline as L
from playground_s6e11_missed_label_model import NESTED, missed_label

# 인자: 0~1 실수 = 놓친 사기 답지 값 x, 'w2' 처럼 w 로 시작 = 답지 1 유지 + 놓친 사기 학습 가중치 2배
ARGS = sys.argv[1:] or ["0.5"]


def params():
    p = {**G.PARAMS, **json.loads((L.OUT / "best_params_lightgbm.json").read_text(encoding="utf-8"))}
    p.update(objective="cross_entropy", metric="auc", learning_rate=0.05)
    return p


def main():
    tr, _ = L.load()
    y = tr[L.TARGET].astype(int).to_numpy()
    X = G.prepare(tr)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(X, y))
    fold_id = np.zeros(len(y), dtype=int)
    for k, (_, va_i) in enumerate(splits):
        fold_id[va_i] = k
    p_inner_all = list(np.load(NESTED, allow_pickle=True)["p_inner"])

    configs = {"base_01": None}
    for a in ARGS:
        configs[f"weight_{a[1:]}" if a.startswith("w") else f"soft_{a}"] = a
    oof = {c: np.zeros(len(y)) for c in configs}
    for k, (tr_i, va_i) in enumerate(splits):
        missed = missed_label(y[tr_i], p_inner_all[k].astype(float)) == 1
        a, b = train_test_split(np.arange(len(tr_i)), test_size=0.1, stratify=y[tr_i], random_state=L.SEED)
        dho = lgb.Dataset(X.iloc[tr_i[b]], y[tr_i[b]], free_raw_data=False)
        for c, s in configs.items():
            t, wt = y[tr_i].astype(float), np.ones(len(tr_i))
            if s is not None and s.startswith("w"):
                wt[missed] = float(s[1:])
            elif s is not None:
                t[missed] = float(s)
            dtr = lgb.Dataset(X.iloc[tr_i[a]], t[a], weight=wt[a])
            m = lgb.train(params(), dtr, 5000, valid_sets=[dho.create_valid(X.iloc[tr_i[b]], y[tr_i[b]])],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            oof[c][va_i] = m.predict(X.iloc[va_i], num_iteration=m.best_iteration)
            print(f"  fold {k} {c}: 놓친 사기 {missed.sum()}건, AUC {roc_auc_score(y[va_i], oof[c][va_i]):.5f}  "
                  f"iter {m.best_iteration}", flush=True)

    p7 = pd.read_csv(L.OUT / "oof_lightgbm_tuned.csv")["p"].to_numpy()
    missed_eval = missed_label(y, p7)
    sel = (y == 0) | (missed_eval == 1)
    base = oof["base_01"]
    print(f"\n참고: v7 OOF AUC {roc_auc_score(y, p7):.5f}")
    rows = []
    for c in configs:
        g = [roc_auc_score(y[fold_id == f], oof[c][fold_id == f]) - roc_auc_score(y[fold_id == f], base[fold_id == f])
             for f in range(5)]
        r = {"config": c, "auc": roc_auc_score(y, oof[c]), "gain_vs_base": roc_auc_score(y, oof[c]) - roc_auc_score(y, base),
             "folds_up": sum(v > 0 for v in g), "missed_vs_normal_auc": roc_auc_score(missed_eval[sel], oof[c][sel]),
             "mean_p_missed": oof[c][missed_eval == 1].mean(), "mean_p_normal": oof[c][y == 0].mean()}
        rows.append(r)
        print(f"[{c}] AUC {r['auc']:.5f}  기준선 대비 {r['gain_vs_base']:+.5f}  fold 상승 {r['folds_up']}/5 "
              f"{[round(v, 5) for v in g]}  놓친사기vs정상 {r['missed_vs_normal_auc']:.4f}  "
              f"평균 예측: 놓친 사기 {r['mean_p_missed']:.3f}, 정상 {r['mean_p_normal']:.4f}")
        if c != "base_01":
            for w in [0.3, 0.5, 0.7]:
                pb = w * pd.Series(oof[c]).rank().to_numpy() + (1 - w) * pd.Series(base).rank().to_numpy()
                print(f"    기준선과 순위 평균 w={w}: {roc_auc_score(y, pb) - roc_auc_score(y, base):+.5f}")
    pd.DataFrame(rows).to_csv(L.OUT / "soft_label_results.csv", index=False)
    pd.DataFrame({"id": tr["id"], "y": y, **oof, "fold": fold_id}).to_csv(L.OUT / "oof_soft_label.csv", index=False)


if __name__ == "__main__":
    main()
