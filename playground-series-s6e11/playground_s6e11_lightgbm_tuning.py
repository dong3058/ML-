"""Playground S6E11 (Predicting Credit Card Fraud) LightGBM Optuna 튜닝 (v7).

1단계 탐색: 3-fold StratifiedKFold, learning_rate 0.05 (빠르게), early stopping 100, 평균 AUC 최대화.
         fold 마다 중간 점수를 보고해 MedianPruner 로 가망 없는 trial 을 일찍 중단.
         study 는 output/optuna_lightgbm.db 에 저장 (중단 후 이어서 실행 가능).
2단계 확인: 최적 파라미터 + learning_rate 0.03 으로 v3 과 같은 5-fold OOF → v3(0.94667) 과 비교,
         제출 파일 output/submission_lightgbm_tuned.csv 생성 (제출은 하지 않음).

실행: python playground_s6e11_lightgbm_tuning.py [--trials 60] [--timeout 3600] [--final-only]
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

STORAGE = f"sqlite:///{(L.OUT / 'optuna_lightgbm.db').as_posix()}"
STUDY = "lightgbm_v7"
BEST_JSON = L.OUT / "best_params_lightgbm.json"


def search_space(trial):
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 30, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 1),
        "cat_smooth": trial.suggest_float("cat_smooth", 1, 100, log=True),
        "min_data_per_group": trial.suggest_int("min_data_per_group", 10, 1000, log=True),
    }


def objective_factory(X, y):
    splits = list(StratifiedKFold(3, shuffle=True, random_state=L.SEED).split(X, y))

    def objective(trial):
        params = {**G.PARAMS, **search_space(trial), "learning_rate": 0.05}
        scores, iters = [], []
        for k, (tr_i, va_i) in enumerate(splits):
            dtr = lgb.Dataset(X.iloc[tr_i], y[tr_i])
            dva = lgb.Dataset(X.iloc[va_i], y[va_i], reference=dtr)
            m = lgb.train(params, dtr, 3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
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
        sampler=optuna.samplers.TPESampler(seed=L.SEED, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    if len(study.trials) == 0:
        # v3 파라미터를 첫 trial 로 넣어 기준점 확보
        study.enqueue_trial({"num_leaves": 31, "max_depth": 12, "min_child_samples": 200,
                             "feature_fraction": 0.8, "bagging_fraction": 0.8, "lambda_l1": 1e-3,
                             "lambda_l2": 1.0, "min_gain_to_split": 0.0, "cat_smooth": 20,
                             "min_data_per_group": 100})

    def log(st, tr):
        if tr.state == optuna.trial.TrialState.COMPLETE:
            print(f"trial {tr.number:3d}: AUC {tr.value:.5f}  best {st.best_value:.5f} "
                  f"(#{st.best_trial.number})", flush=True)
        else:
            print(f"trial {tr.number:3d}: {tr.state.name}", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective_factory(X, y), n_trials=trials, timeout=timeout, callbacks=[log])

    done = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    done.to_csv(L.OUT / "optuna_lightgbm_trials.csv", index=False)
    print(f"\n완료 {sum(done.state == 'COMPLETE')}개, 중단(prune) {sum(done.state == 'PRUNED')}개")
    print(f"v3 파라미터 (trial 0) 3-fold AUC: {study.trials[0].value}")
    print(f"최고 trial #{study.best_trial.number}: {study.best_value:.5f}")
    print(json.dumps(study.best_params, indent=2))
    imp = optuna.importance.get_param_importances(study)
    print("\n파라미터 중요도:", {k: round(v, 3) for k, v in imp.items()})
    BEST_JSON.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    return study.best_params


def final(X, y, Xte, te, best, train_ids):
    G.PARAMS.update(best)
    G.PARAMS["learning_rate"] = 0.03
    print("\n[v7_lightgbm_tuned] 5-fold, learning_rate 0.03")
    oof, pte, iters, _ = G.oof_lgb(X, y, Xte)
    print(f"  OOF AUC {roc_auc_score(y, oof):.5f}  logloss {log_loss(y, oof):.5f}  (v3: 0.94667 / 0.04579)")
    pd.DataFrame({"id": train_ids, "y": y, "p": oof}).to_csv(L.OUT / "oof_lightgbm_tuned.csv", index=False)
    pd.DataFrame({"id": te["id"], L.TARGET: pte}).to_csv(L.OUT / "submission_lightgbm_tuned.csv", index=False)
    print("  제출 파일 저장(제출은 안 함): output/submission_lightgbm_tuned.csv, 평균 iter", int(np.mean(iters)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=3600)
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
