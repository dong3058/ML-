"""Playground S6E11 (Predicting Credit Card Fraud) LightGBM Optuna 넓은 범위 재탐색 (v16).

v7 탐색(10개 파라미터, 60 trial)보다 범위·종류를 넓힌다.
  - 기존 10개 범위 확대 (예: num_leaves 4~1024, min_child_samples 5~10000, 규제 1e-8~100)
  - 새 파라미터: max_bin, bagging_freq, feature_fraction_bynode, min_sum_hessian_in_leaf, path_smooth,
    extra_trees, cat_l2, max_cat_threshold, scale_pos_weight, data_sample_strategy(bagging/goss),
    단조 제약 on/off (가맹점 위험 점수·거래 속도 점수·CVV 재입력 ↑ → 사기 확률 ↓ 금지)
  - TPE multivariate, 시작 무작위 trial 25개, v7 최적값을 첫 trial 로 넣어 기준점 확보
탐색 조건은 v7 과 같음: 3-fold, learning_rate 0.05, early stopping 100, 평균 AUC, MedianPruner.
study 는 output/optuna_lightgbm.db 의 'lightgbm_v16' (v7 study 와 별개).
확인: 최적 파라미터 + learning_rate 0.03 으로 기존과 같은 5-fold OOF → v7(0.94754) 과 fold 별 비교.
v7 결과 파일은 덮어쓰지 않는다 (출력: *_v16).

실행: python playground_s6e11_lightgbm_tuning_wide.py [--trials 200] [--timeout 10800] [--final-only]
"""

import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import playground_s6e11_lightgbm as G
import playground_s6e11_linear_baseline as L
from playground_s6e11_lightgbm_tuning import STORAGE

STUDY = "lightgbm_v16"
BEST_JSON = L.OUT / "best_params_lightgbm_v16.json"
MONO_UP = ["merchant_risk_score", "velocity_score", "cvv_retry_count"]
DATASET_KEYS = ["max_bin", "min_data_per_group", "max_cat_threshold", "cat_smooth", "cat_l2"]


def search_space(trial):
    p = {
        "num_leaves": trial.suggest_int("num_leaves", 4, 1024, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 10000, log=True),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-4, 50, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.3, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 100, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 100, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 5),
        "path_smooth": trial.suggest_float("path_smooth", 0, 100),
        "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255, 511, 1023]),
        "cat_smooth": trial.suggest_float("cat_smooth", 0.1, 200, log=True),
        "cat_l2": trial.suggest_float("cat_l2", 0.1, 100, log=True),
        "max_cat_threshold": trial.suggest_int("max_cat_threshold", 4, 64),
        "min_data_per_group": trial.suggest_int("min_data_per_group", 5, 2000, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 30, log=True),
        "monotone": trial.suggest_categorical("monotone", [False, True]),
        "data_sample_strategy": trial.suggest_categorical("data_sample_strategy", ["bagging", "goss"]),
    }
    if p["data_sample_strategy"] == "bagging":
        p["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.3, 1.0)
        p["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 10)
    return p


def to_lgb(p, columns):
    """trial 값 → LightGBM 파라미터 (단조 제약 표시를 실제 제약 벡터로)"""
    q = {**G.PARAMS, **p}
    mono = q.pop("monotone", False)
    if mono:
        q["monotone_constraints"] = [1 if c in MONO_UP else 0 for c in columns]
        q["monotone_constraints_method"] = "advanced"
    if q.get("data_sample_strategy") == "goss":
        q["bagging_fraction"], q["bagging_freq"] = 1.0, 0
    return q


def dataset(X, y, q, reference=None):
    return lgb.Dataset(X, y, reference=reference, params={k: q[k] for k in DATASET_KEYS if k in q})


def objective_factory(X, y):
    splits = list(StratifiedKFold(3, shuffle=True, random_state=L.SEED).split(X, y))

    def objective(trial):
        q = {**to_lgb(search_space(trial), X.columns), "learning_rate": 0.05}
        scores, iters = [], []
        for k, (tr_i, va_i) in enumerate(splits):
            dtr = dataset(X.iloc[tr_i], y[tr_i], q)
            dva = dataset(X.iloc[va_i], y[va_i], q, reference=dtr)
            m = lgb.train(q, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            scores.append(m.best_score["valid_0"]["auc"])
            iters.append(m.best_iteration)
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("iters", iters)
        return float(np.mean(scores))

    return objective


def tune(X, y, trials, timeout):
    study = optuna.create_study(
        study_name=STUDY, storage=STORAGE, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=L.SEED + 16, n_startup_trials=25, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=0))
    if len(study.trials) == 0:
        v7 = json.loads((L.OUT / "best_params_lightgbm.json").read_text(encoding="utf-8"))
        study.enqueue_trial({**v7, "min_sum_hessian_in_leaf": 1e-3, "feature_fraction_bynode": 1.0,
                             "path_smooth": 0.0, "extra_trees": False, "max_bin": 255, "cat_l2": 10.0,
                             "max_cat_threshold": 32, "scale_pos_weight": 1.0, "monotone": False,
                             "data_sample_strategy": "bagging", "bagging_freq": 1})

    def log(st, tr):
        if tr.state == optuna.trial.TrialState.COMPLETE:
            print(f"trial {tr.number:3d}: AUC {tr.value:.5f}  best {st.best_value:.5f} (#{st.best_trial.number})",
                  flush=True)
        else:
            print(f"trial {tr.number:3d}: {tr.state.name}", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective_factory(X, y), n_trials=trials, timeout=timeout, callbacks=[log])

    done = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    done.to_csv(L.OUT / "optuna_lightgbm_v16_trials.csv", index=False)
    print(f"\n완료 {sum(done.state == 'COMPLETE')}개, 중단(prune) {sum(done.state == 'PRUNED')}개")
    print(f"v7 파라미터 (trial 0) 3-fold AUC: {study.trials[0].value}")
    print(f"최고 trial #{study.best_trial.number}: {study.best_value:.5f}")
    print(json.dumps(study.best_params, indent=2))
    imp = optuna.importance.get_param_importances(study)
    print("\n파라미터 중요도:", {k: round(v, 3) for k, v in imp.items()})
    top = done[done.state == "COMPLETE"].sort_values("value", ascending=False).head(10)
    print("\n상위 10 trial")
    print(top.drop(columns=["state"]).round(4).to_string(index=False))
    BEST_JSON.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    return study.best_params


def final(X, y, Xte, te, best, train_ids):
    q = {**to_lgb(dict(best), X.columns), "learning_rate": 0.03}
    splits = list(StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(X, y))
    oof, pte, iters = np.zeros(len(y)), np.zeros(len(te)), []
    fold_id = np.zeros(len(y), dtype=int)
    print("\n[v16] 5-fold, learning_rate 0.03")
    for k, (tr_i, va_i) in enumerate(splits):
        fold_id[va_i] = k
        dtr = dataset(X.iloc[tr_i], y[tr_i], q)
        dva = dataset(X.iloc[va_i], y[va_i], q, reference=dtr)
        m = lgb.train(q, dtr, 8000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_i] = m.predict(X.iloc[va_i], num_iteration=m.best_iteration)
        pte += m.predict(Xte, num_iteration=m.best_iteration) / 5
        iters.append(m.best_iteration)
        print(f"  fold {k}: AUC {roc_auc_score(y[va_i], oof[va_i]):.5f}  iter {m.best_iteration}", flush=True)
    v7 = pd.read_csv(L.OUT / "oof_lightgbm_tuned.csv")["p"].to_numpy()
    g = [roc_auc_score(y[fold_id == f], oof[fold_id == f]) - roc_auc_score(y[fold_id == f], v7[fold_id == f])
         for f in range(5)]
    print(f"  OOF AUC {roc_auc_score(y, oof):.5f}  logloss {log_loss(y, oof):.5f}")
    print(f"  v7 {roc_auc_score(y, v7):.5f} 대비 {roc_auc_score(y, oof) - roc_auc_score(y, v7):+.5f}  "
          f"fold 상승 {sum(v > 0 for v in g)}/5  {[round(v, 5) for v in g]}")
    pd.DataFrame({"id": train_ids, "y": y, "p": oof}).to_csv(L.OUT / "oof_lightgbm_tuned_v16.csv", index=False)
    pd.DataFrame({"id": te["id"], L.TARGET: pte}).to_csv(L.OUT / "submission_lightgbm_tuned_v16.csv", index=False)
    print("  제출 파일 저장(제출은 안 함): output/submission_lightgbm_tuned_v16.csv, 평균 iter", int(np.mean(iters)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--final-only", action="store_true")
    args = ap.parse_args()

    tr, te = L.load()
    y = tr[L.TARGET].astype(int).to_numpy()
    X, Xte = G.prepare(tr), G.prepare(te)
    Xte[L.CATEGORICAL] = Xte[L.CATEGORICAL].apply(lambda s: s.cat.set_categories(X[s.name].cat.categories))
    best = json.loads(BEST_JSON.read_text(encoding="utf-8")) if args.final_only else tune(X, y, args.trials, args.timeout)
    final(X, y, Xte, te, best, tr["id"].to_numpy())


if __name__ == "__main__":
    main()
