"""House Prices 하이퍼파라미터 튜닝(Optuna) + seed 평균. 대상은 피처 공유 실험의 최고 조합 'both':
  CatBoost  = 카테고리 44개 + 수치형 51개
  LightGBM  = 수치형 51개 + 품질 등급 10개 + SaleCondition·MSZoning·Neighborhood·Functional

단계
  tune  : 모델별 Optuna 탐색 (5-fold, 튜닝 전용 분할 random_state=7). 결과는 output/optuna_house_prices.db 에
          저장되어 중단해도 이어서 진행된다. 최적값은 output/best_params.json.
  final : 최적값으로 반복 5-fold×3 교차검증(분할 random_state=42, 튜닝과 다른 분할) → 스태킹.
          fold 마다 seed 3개를 학습해 '첫 seed 단독'과 'seed 3개 평균'을 함께 기록한다.
          전체 데이터 재학습(seed 평균) → output/submission_tuned_*.csv

실행: python house_prices_tuning.py tune [--lgb-minutes 15 --cat-minutes 40]
      python house_prices_tuning.py final
"""

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold

import house_prices_stacking as H
from house_prices_feature_sharing import lgb_frame

STORAGE = f"sqlite:///{(H.OUT / 'optuna_house_prices.db').as_posix()}"
BEST = H.OUT / "best_params.json"
SEEDS = [0, 1, 2]
TUNE_SPLIT = 7


def data():
    tr, te = H.load()
    tr, te = H.prepare(tr, te)
    y = np.log1p(tr["SalePrice"].to_numpy())
    cols = H.CATEGORICAL + H.NUM_FEATURES
    Xc, Xc_te = tr[cols].copy(), te[cols].copy()
    Xc[H.NUM_FEATURES] = Xc[H.NUM_FEATURES].astype(float)
    Xc_te[H.NUM_FEATURES] = Xc_te[H.NUM_FEATURES].astype(float)
    Xl, Xl_te = lgb_frame(tr, te)
    return tr, te, y, (Xc, Xc_te), (Xl, Xl_te)


def make_lgb(p: dict, seed: int):
    return lgb.LGBMRegressor(**p, subsample_freq=1, random_state=seed, verbose=-1, n_jobs=-1)


def make_cat(p: dict, seed: int):
    return CatBoostRegressor(**p, loss_function="RMSE", random_seed=seed, verbose=0, thread_count=-1,
                             cat_features=list(range(len(H.CATEGORICAL))), allow_writing_files=False)


BASELINE = {  # 피처 공유 실험(both)에서 쓴 기존 설정 = 튜닝 출발점
    "lgb": dict(n_estimators=3000, learning_rate=0.01, num_leaves=8, min_child_samples=10, subsample=0.7,
                colsample_bytree=0.5, reg_lambda=1.0, reg_alpha=1e-4, min_data_per_group=10, cat_smooth=10.0),
    "cat": dict(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, random_strength=1.0,
                bagging_temperature=1.0),
}


def space(trial: optuna.Trial, kind: str) -> dict:
    if kind == "lgb":
        return dict(n_estimators=trial.suggest_int("n_estimators", 800, 5000, step=200),
                    learning_rate=trial.suggest_float("learning_rate", 0.004, 0.05, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 4, 32),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.15, 0.8),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                    min_data_per_group=trial.suggest_int("min_data_per_group", 5, 50),
                    cat_smooth=trial.suggest_float("cat_smooth", 1, 30, log=True))
    return dict(iterations=trial.suggest_int("iterations", 600, 2000, step=200),
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                depth=trial.suggest_int("depth", 4, 8),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 15, log=True),
                random_strength=trial.suggest_float("random_strength", 0.1, 5, log=True),
                bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0))


def tune(kind: str, X: pd.DataFrame, y: np.ndarray, minutes: float) -> dict:
    make = make_lgb if kind == "lgb" else make_cat
    splits = list(KFold(5, shuffle=True, random_state=TUNE_SPLIT).split(X))

    def objective(trial: optuna.Trial) -> float:
        p = space(trial, kind)
        se = []
        for i, (a, b) in enumerate(splits):
            m = make(p, H.SEED).fit(X.iloc[a], y[a])
            se.append(np.mean((m.predict(X.iloc[b]) - y[b]) ** 2))
            trial.report(float(np.sqrt(np.mean(se))), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.sqrt(np.mean(se)))

    study = optuna.create_study(study_name=f"house_prices_both_{kind}", storage=STORAGE, direction="minimize",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=H.SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    if not study.trials:
        study.enqueue_trial(BASELINE[kind])  # 첫 trial = 기존 설정 → 튜닝 개선폭의 기준

    def log(st, t):
        if t.state == optuna.trial.TrialState.COMPLETE:
            print(f"  [{kind}] trial {t.number:3d} RMSE {t.value:.5f} | best {st.best_value:.5f} (#{st.best_trial.number})",
                  flush=True)
        else:
            print(f"  [{kind}] trial {t.number:3d} {t.state.name}", flush=True)

    study.optimize(objective, timeout=minutes * 60, callbacks=[log])
    base = study.trials[0].value
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"[튜닝 {kind}] 완료 {len(done)}개 / 전체 {len(study.trials)}개 | 기존 설정 {base:.5f} → 최적 "
          f"{study.best_value:.5f} (#{study.best_trial.number}) | {study.best_params}", flush=True)
    return {"params": study.best_params, "best": study.best_value, "baseline": base,
            "n_complete": len(done), "n_trials": len(study.trials)}


def oof_seeds(make, p: dict, X: pd.DataFrame, y: np.ndarray, X_te: pd.DataFrame, label: str):
    """반복 5-fold×3. fold 마다 seed 3개 학습 → (첫 seed OOF, seed 평균 OOF)."""
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=H.SEED)
    one, avg = np.zeros((3, len(X))), np.zeros((3, len(X)))
    for i, (a, b) in enumerate(rkf.split(X)):
        preds = [make(p, H.SEED + 100 * s + i).fit(X.iloc[a], y[a]).predict(X.iloc[b]) for s in SEEDS]
        one[i // 5, b], avg[i // 5, b] = preds[0], np.mean(preds, axis=0)
        print(f"  [{label}] fold {i + 1}/15 RMSE 첫 seed {H.rmse(y[b], preds[0]):.5f} | "
              f"seed 평균 {H.rmse(y[b], avg[i // 5, b]):.5f}", flush=True)
    return one.mean(axis=0), avg.mean(axis=0)


def stack_cv(a: np.ndarray, b: np.ndarray, y: np.ndarray):
    Z = np.column_stack([a, b])
    oof = np.zeros(len(y))
    for t, v in KFold(5, shuffle=True, random_state=H.SEED).split(Z):
        oof[v] = LinearRegression(positive=True).fit(Z[t], y[t]).predict(Z[v])
    return H.rmse(y, oof), LinearRegression(positive=True).fit(Z, y)


def final() -> None:
    tr, te, y, (Xc, Xc_te), (Xl, Xl_te) = data()
    best = json.loads(BEST.read_text(encoding="utf-8"))
    pc, pl = best["cat"]["params"], best["lgb"]["params"]
    cache = H.OUT / "oof_tuned.csv"
    if cache.exists():
        o = pd.read_csv(cache)
    else:
        o = pd.DataFrame({"Id": tr.Id})
        o["lgb_one"], o["lgb_avg"] = oof_seeds(make_lgb, pl, Xl, y, Xl_te, "lgb 튜닝")
        o["cat_one"], o["cat_avg"] = oof_seeds(make_cat, pc, Xc, y, Xc_te, "cat 튜닝")
        o.to_csv(cache, index=False)
    prev = pd.read_csv(H.OUT / "oof_feature_sharing.csv")  # 튜닝 전 both (기존 설정, 단일 seed)
    rows = {}
    for name, (c, l) in {"튜닝 전 (기존 설정)": (prev.cat_num, prev.lgb_cat),
                         "튜닝 후 단일 seed": (o.cat_one, o.lgb_one),
                         "튜닝 후 seed 3개 평균": (o.cat_avg, o.lgb_avg)}.items():
        sc, meta = stack_cv(c.to_numpy(), l.to_numpy(), y)
        rows[name] = {"cat": H.rmse(y, c), "lgb": H.rmse(y, l), "stack": sc,
                      "w_cat": meta.coef_[0], "w_lgb": meta.coef_[1], "_meta": meta}
    print("\n=== 교차검증 비교 (반복 5-fold×3) ===")
    print(pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "_meta"} for k, v in rows.items()}).T
          .round(5).to_string(), flush=True)

    # 전체 데이터 재학습 (seed 3개 평균) → 제출 파일
    cat_te = np.mean([make_cat(pc, H.SEED + 100 * s).fit(Xc, y).predict(Xc_te) for s in SEEDS], axis=0)
    lgb_te = np.mean([make_lgb(pl, H.SEED + 100 * s).fit(Xl, y).predict(Xl_te) for s in SEEDS], axis=0)
    meta = rows["튜닝 후 seed 3개 평균"]["_meta"]
    p = meta.predict(np.column_stack([cat_te, lgb_te]))
    outs = {"submission_tuned_seedavg_both.csv": p,
            "submission_tuned_seedavg_catboost_only.csv": cat_te}
    i2550 = te.index[te.Id == 2550][0]
    for f, v in outs.items():
        pd.DataFrame({"Id": te.Id, "SalePrice": np.expm1(v)}).to_csv(H.OUT / f, index=False)
        print(f"[제출 파일] {f} | Id 2550 {np.expm1(v[i2550]):,.0f} | 최댓값 {np.expm1(v.max()):,.0f}")
    res = {k: {kk: float(vv) for kk, vv in v.items() if kk != "_meta"} for k, v in rows.items()}
    (H.OUT / "result_tuned.json").write_text(json.dumps({"cv": res, "best_params": best}, ensure_ascii=False,
                                                        indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "final"])
    ap.add_argument("--lgb-minutes", type=float, default=15)
    ap.add_argument("--cat-minutes", type=float, default=40)
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if a.stage == "tune":
        _, _, y, (Xc, _), (Xl, _) = data()
        t0 = time.time()
        best = {"lgb": tune("lgb", Xl, y, a.lgb_minutes), "cat": tune("cat", Xc, y, a.cat_minutes)}
        BEST.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[튜닝 완료] {time.time() - t0:.0f}초 → {BEST}")
    else:
        final()


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
    main()
