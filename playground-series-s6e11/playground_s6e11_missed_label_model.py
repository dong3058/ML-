"""Playground S6E11 (Predicting Credit Card Fraud) '놓친 사기' 라벨 전문 모델 + 본 모델 앙상블 (v11).

아이디어: 원래 답지(is_fraud)에서 본 모델이 놓친 사기만 1, 나머지(정상 + 잡힌 사기)는 0 인 새 답지를 만들고,
이 답지로 LightGBM 하나를 학습한다. 이 모델 점수 = '본 모델이 놓칠 만한 사기일 가능성'.
본 모델(v7)과 섞는다.

누수 방지
  바깥 5-fold (v1~v10 과 같은 분할). 바깥 fold k 의 학습 행 라벨은 '그 학습 행만으로 만든'
  안쪽 3-fold LightGBM OOF 예측(output/gbdt_nested.npz, v4 에서 만든 것)으로 정한다.
  놓침 = 실제 사기이면서 안쪽 예측 순위가 상위 (학습 행 사기 수) 안에 못 든 행.
  early stopping 은 학습 행 안의 10% 홀드아웃으로 해서 검증 fold 정보를 쓰지 않는다.

라벨 두 가지
  A (요청 그대로): 놓친 사기 1 / 정상 + 잡힌 사기 0
  B (비교용)     : 놓친 사기 1 / 정상 0, 잡힌 사기는 학습에서 제외
결합 (본 모델 p_main, 전문 모델 p_spec)
  가산: p = p_main + w·p_spec   (P(사기) = P(잡힐 사기) + P(놓칠 사기) 라는 해석)
  가중치는 (a) 전체 OOF 에서 고른 값, (b) 한 fold 빼고 고른 값을 빠진 fold 에 적용(정직한 추정) 둘 다 보고.
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import playground_s6e11_lightgbm as G
import playground_s6e11_linear_baseline as L

NESTED = L.OUT / "gbdt_nested.npz"
WEIGHTS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5]


def params():
    p = {**G.PARAMS, **json.loads((L.OUT / "best_params_lightgbm.json").read_text(encoding="utf-8"))}
    p.update(learning_rate=0.05)
    return p


def missed_label(y, p):
    """실제 사기이면서 예측 순위가 상위 (사기 수) 안에 못 든 행"""
    k = int(y.sum())
    top = pd.Series(p).rank(ascending=False, method="first").to_numpy() <= k
    return ((y == 1) & ~top).astype(int)


def fit_spec(X, z, rows):
    a, b = train_test_split(rows, test_size=0.1, stratify=z[rows], random_state=L.SEED)
    dtr = lgb.Dataset(X.iloc[a], z[a])
    dva = lgb.Dataset(X.iloc[b], z[b], reference=dtr)
    m = lgb.train(params(), dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


def fold_gain(y, p, p_main, fold_id):
    return [roc_auc_score(y[fold_id == f], p[fold_id == f]) - roc_auc_score(y[fold_id == f], p_main[fold_id == f])
            for f in range(5)]


def choose_w(y, p_main, p_spec, idx):
    return max(WEIGHTS, key=lambda w: roc_auc_score(y[idx], (p_main + w * p_spec)[idx]))


def report(name, y, p_main, p_spec, fold_id, missed_eval):
    base = roc_auc_score(y, p_main)
    normal = y == 0
    sel = normal | (missed_eval == 1)
    print(f"\n[{name}]")
    print(f"  놓친 사기 vs 정상 구분 AUC: 본 모델 {roc_auc_score(missed_eval[sel], p_main[sel]):.4f}  "
          f"/ 전문 모델 {roc_auc_score(missed_eval[sel], p_spec[sel]):.4f}")
    print(f"  전문 모델 단독 is_fraud AUC {roc_auc_score(y, p_spec):.5f}")
    r = np.corrcoef(p_spec[y == 1], p_main[y == 1])[0, 1]
    print(f"  사기 행에서 두 점수 상관 {r:.3f}")
    curve = {w: roc_auc_score(y, p_main + w * p_spec) - base for w in WEIGHTS}
    print("  가산 결합 전체 AUC 변화: " + "  ".join(f"{w}:{g:+.5f}" for w, g in curve.items()))

    w_all = max(curve, key=curve.get)
    g_all = fold_gain(y, p_main + w_all * p_spec, p_main, fold_id)
    print(f"  (a) 전체 OOF 로 고른 w={w_all}: gain {curve[w_all]:+.5f}  fold 상승 {sum(g > 0 for g in g_all)}/5  "
          f"{[round(g, 5) for g in g_all]}")

    p_hon, ws = p_main.copy(), []
    for f in range(5):
        w = choose_w(y, p_main, p_spec, np.where(fold_id != f)[0])
        ws.append(w)
        k = fold_id == f
        p_hon[k] = (p_main + w * p_spec)[k]
    g_hon = fold_gain(y, p_hon, p_main, fold_id)
    print(f"  (b) 정직 추정: AUC {roc_auc_score(y, p_hon):.5f}  gain {roc_auc_score(y, p_hon) - base:+.5f}  "
          f"fold 상승 {sum(g > 0 for g in g_hon)}/5  {[round(g, 5) for g in g_hon]}  fold별 w {ws}")

    # 놓친 사기가 얼마나 구제됐나 (상위 k 기준)
    k = int(y.sum())
    p_best = p_main + w_all * p_spec
    top_main = pd.Series(p_main).rank(ascending=False, method="first").to_numpy() <= k
    top_new = pd.Series(p_best).rank(ascending=False, method="first").to_numpy() <= k
    print(f"  상위 {k}건 기준 잡은 사기: 본 {int((top_main & (y == 1)).sum())} → 결합 {int((top_new & (y == 1)).sum())}  "
          f"(놓친 사기 중 새로 잡음 {int((top_new & (missed_eval == 1)).sum())}, "
          f"원래 잡던 사기 중 빠짐 {int((top_main & ~top_new & (y == 1)).sum())})")
    return {"label": name, "w_all": w_all, "gain_all": curve[w_all], "gain_honest": roc_auc_score(y, p_hon) - base,
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
    missed_eval = missed_label(y, p_main)  # 평가용: v7 OOF 기준 놓친 사기
    print(f"평가용 놓친 사기 (v7 OOF 기준): {missed_eval.sum()}건 / 사기 {y.sum()}건")

    p_inner_all = list(np.load(NESTED, allow_pickle=True)["p_inner"])
    spec = {"A": np.zeros(len(y)), "B": np.zeros(len(y))}
    for k, (tr_i, va_i) in enumerate(splits):
        z = np.zeros(len(y), dtype=int)
        z[tr_i] = missed_label(y[tr_i], p_inner_all[k].astype(float))
        caught = (y == 1) & (z == 0)
        for lab, rows in [("A", tr_i), ("B", tr_i[~caught[tr_i]])]:
            m = fit_spec(X, z, rows)
            spec[lab][va_i] = m.predict(X.iloc[va_i], num_iteration=m.best_iteration)
            print(f"  fold {k} 라벨 {lab}: 학습 행 {len(rows)}, 놓친 사기 라벨 {z[rows].sum()}, "
                  f"iter {m.best_iteration}", flush=True)

    res = [report("A 놓친 사기 vs 나머지 전부", y, p_main, spec["A"], fold_id, missed_eval),
           report("B 놓친 사기 vs 정상 (잡힌 사기 제외)", y, p_main, spec["B"], fold_id, missed_eval)]
    pd.DataFrame(res).to_csv(L.OUT / "missed_label_results.csv", index=False)
    pd.DataFrame({"id": tr["id"], "y": y, "p_main": p_main, "spec_A": spec["A"], "spec_B": spec["B"],
                  "missed_eval": missed_eval, "fold": fold_id}).to_csv(L.OUT / "oof_missed_label.csv", index=False)


if __name__ == "__main__":
    main()
