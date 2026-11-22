"""House Prices v7 실험: Box-Cox + 선형 모델(Lasso / ElasticNet) 블렌딩 + 보호 장치. (Kaggle 제출 안 함)

  선형 모델 입력: 수치형 51개 (치우친 피처는 log1p 또는 Box-Cox) + 카테고리 44개 원-핫 → RobustScaler
    lasso_log : log1p 변환 + Lasso        ← Box-Cox 효과 비교용
    lasso_bc  : Box-Cox 변환 + Lasso
    enet_bc   : Box-Cox 변환 + ElasticNet
  변환 강도(λ)·치우침 판정·스케일 기준은 fold 의 학습 데이터에서만 계산한다.

  블렌딩: v6 의 CatBoost·LightGBM 교차검증 예측(output/oof_tuned.csv, seed 3개 평균) + 선형 모델 → 메타 선형회귀
  보호 장치: 수치형 피처가 하나라도 학습 범위 밖인 test 집은 선형 모델을 뺀 메타(CatBoost+LightGBM)로 예측
  결과: output/result_linear_blend.json, 제출용 파일 output/submission_v7_*.csv (제출은 하지 않음)
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

import house_prices_stacking as H
from house_prices_tuning import SEEDS, data, make_lgb

OOF_LIN = H.OUT / "oof_linear.csv"


class LinearBlendModel:
    def __init__(self, seed: int = 0, transform: str = "boxcox", kind: str = "lasso"):
        self.transform, self.kind = transform, kind
        self.lam: dict[str, float] = {}
        self.scaler = RobustScaler()
        alphas = np.logspace(-4.5, -2, 30)
        if kind == "lasso":
            self.model = LassoCV(alphas=alphas, cv=5, max_iter=50000, n_jobs=-1)
        else:
            self.model = ElasticNetCV(alphas=alphas, l1_ratio=[0.3, 0.6, 0.9], cv=5, max_iter=50000, n_jobs=-1)

    def _num(self, X: pd.DataFrame) -> pd.DataFrame:
        N = X[H.NUM_FEATURES].astype(float).copy()
        for c, lam in self.lam.items():
            v = N[c].clip(lower=0)
            N[c] = np.log1p(v) if self.transform == "log" else boxcox1p(v, lam)
        return N

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        D = pd.get_dummies(X[H.CATEGORICAL].astype(str), dtype=float)
        D = D.reindex(columns=self.dummy_cols, fill_value=0.0)
        return pd.concat([self._num(X).reset_index(drop=True), D.reset_index(drop=True)], axis=1)

    def fit(self, X: pd.DataFrame, y):
        N = X[H.NUM_FEATURES].astype(float)
        sk = N.skew()
        skewed = [c for c in H.NUM_FEATURES if abs(sk[c]) > 0.75 and N[c].min() >= 0]
        # Box-Cox λ: 피처마다 분포를 가장 정규에 가깝게 만드는 값 (boxcox1p 는 x+1 에 적용)
        self.lam = {c: (float(boxcox_normmax(N[c] + 1, method="mle")) if self.transform == "boxcox" else 0.0)
                    for c in skewed}
        self.lam = {c: float(np.clip(l, -2, 2)) for c, l in self.lam.items()}
        dummies = pd.get_dummies(X[H.CATEGORICAL].astype(str), dtype=float)
        self.dummy_cols = [c for c in dummies.columns if dummies[c].sum() >= 3]  # 3채 미만 범주는 버림
        Z = self.scaler.fit_transform(self._design(X))
        self.model.fit(Z, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(self.scaler.transform(self._design(X)))


def meta_cv(cols: list[np.ndarray], y: np.ndarray):
    Z = np.column_stack(cols)
    oof = np.zeros(len(y))
    for t, v in KFold(5, shuffle=True, random_state=H.SEED).split(Z):
        oof[v] = LinearRegression(positive=True).fit(Z[t], y[t]).predict(Z[v])
    return H.rmse(y, oof), LinearRegression(positive=True).fit(Z, y), oof


def main() -> None:
    tr, te, y, (Xc, Xc_te), (Xl, Xl_te) = data()
    Xall, Xall_te = tr, te
    variants = {"lasso_log": ("log", "lasso"), "lasso_bc": ("boxcox", "lasso"), "enet_bc": ("boxcox", "enet")}

    # 1) 선형 모델 OOF (v6 와 같은 반복 5-fold×3 분할)
    lin = pd.read_csv(OOF_LIN) if OOF_LIN.exists() else pd.DataFrame({"Id": tr.Id})
    for k, (t, kind) in variants.items():
        if k not in lin:
            lin[k], _, _ = H.oof_predictions(lambda s, t=t, kind=kind: LinearBlendModel(s, t, kind),
                                             Xall, y, Xall_te.iloc[:5], label=k)
            lin.to_csv(OOF_LIN, index=False)
    v6 = pd.read_csv(H.OUT / "oof_tuned.csv")
    assert (v6.Id.to_numpy() == tr.Id.to_numpy()).all() and (lin.Id.to_numpy() == tr.Id.to_numpy()).all()
    oof = {"catboost": v6.cat_avg.to_numpy(), "lightgbm": v6.lgb_avg.to_numpy(),
           **{k: lin[k].to_numpy() for k in variants}}
    single = {k: H.rmse(y, v) for k, v in oof.items()}
    print("\n=== 단독 교차검증 ===")
    for k, v in single.items():
        print(f"  {k:10s} {v:.5f}")
    print(f"  예측 상관: catboost-lasso_bc {np.corrcoef(oof['catboost'], oof['lasso_bc'])[0, 1]:.3f}, "
          f"lightgbm-lasso_bc {np.corrcoef(oof['lightgbm'], oof['lasso_bc'])[0, 1]:.3f}, "
          f"catboost-lightgbm {np.corrcoef(oof['catboost'], oof['lightgbm'])[0, 1]:.3f}")

    # 2) 블렌딩 조합
    combos = {"v6 (cat+lgb)": ["catboost", "lightgbm"],
              "+ lasso_log": ["catboost", "lightgbm", "lasso_log"],
              "+ lasso_bc": ["catboost", "lightgbm", "lasso_bc"],
              "+ enet_bc": ["catboost", "lightgbm", "enet_bc"],
              "+ lasso_bc + enet_bc": ["catboost", "lightgbm", "lasso_bc", "enet_bc"]}
    res, metas = {}, {}
    cheap = (tr.SalePrice <= tr.SalePrice.quantile(0.1)).to_numpy()
    for name, ks in combos.items():
        sc, meta, so = meta_cv([oof[k] for k in ks], y)
        metas[name] = (ks, meta)
        res[name] = {"cv": sc, **{f"w_{k}": float(w) for k, w in zip(ks, meta.coef_)},
                     "rmse_D1(싼 10%)": H.rmse(y[cheap], so[cheap]),
                     "rmse_Abnorml": H.rmse(y[(tr.SaleCondition == "Abnorml").to_numpy()],
                                            so[(tr.SaleCondition == "Abnorml").to_numpy()])}
    print("\n=== 블렌딩 교차검증 ===")
    print(pd.DataFrame(res).T.round(4).fillna("").to_string(), flush=True)

    # 3) test 예측: 전체 데이터 재학습 + 보호 장치
    cat_te = np.log1p(pd.read_csv(H.OUT / "submission_tuned_seedavg_catboost_only.csv").SalePrice.to_numpy())
    pl = json.loads((H.OUT / "best_params.json").read_text(encoding="utf-8"))["lgb"]["params"]
    lgb_te = np.mean([make_lgb(pl, H.SEED + 100 * s).fit(Xl, y).predict(Xl_te) for s in SEEDS], axis=0)
    te_pred = {"catboost": cat_te, "lightgbm": lgb_te}
    for k, (t, kind) in variants.items():
        te_pred[k] = LinearBlendModel(H.SEED, t, kind).fit(Xall, y).predict(Xall_te)
    lo, hi = tr[H.NUM_FEATURES].astype(float).min(), tr[H.NUM_FEATURES].astype(float).max()
    Nte = te[H.NUM_FEATURES].astype(float)
    ood = ((Nte < lo) | (Nte > hi)).any(axis=1).to_numpy()
    base_ks, base_meta = metas["v6 (cat+lgb)"]
    base_te = base_meta.predict(np.column_stack([te_pred[k] for k in base_ks]))
    i2550 = int(np.where(te.Id.to_numpy() == 2550)[0][0])
    print(f"\n학습 범위 밖 test 집 {ood.sum()}채")
    print(f"Id 2550 단독 예측: " + ", ".join(f"{k} {np.expm1(v[i2550]):,.0f}" for k, v in te_pred.items()))
    test_info = {}
    for name, (ks, meta) in metas.items():
        if name == "v6 (cat+lgb)":
            continue
        raw = meta.predict(np.column_stack([te_pred[k] for k in ks]))
        guarded = np.where(ood, base_te, raw)
        tag = name.replace("+ ", "").replace(" ", "_")
        for suffix, p in (("raw", raw), ("guard", guarded)):
            pd.DataFrame({"Id": te.Id, "SalePrice": np.expm1(p)}).to_csv(
                H.OUT / f"submission_v7_{tag}_{suffix}.csv", index=False)
        test_info[name] = {"Id2550_raw": float(np.expm1(raw[i2550])), "Id2550_guard": float(np.expm1(guarded[i2550])),
                           "max_raw": float(np.expm1(raw.max())), "max_guard": float(np.expm1(guarded.max())),
                           "corr_with_v6": float(np.corrcoef(guarded, base_te)[0, 1]),
                           "ood_mean_abs_diff_raw_vs_guard": float(np.abs(raw - guarded)[ood].mean())}
    print("\n=== test 예측 (보호 장치 전/후) ===")
    print(pd.DataFrame(test_info).T.round(4).to_string())
    (H.OUT / "result_linear_blend.json").write_text(json.dumps(
        {"single_cv": single, "blend_cv": res, "test": test_info, "ood_rows": int(ood.sum())},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
    main()
