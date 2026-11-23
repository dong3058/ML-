"""Playground S6E11 (Predicting Credit Card Fraud) v8 스태킹 제출 파일 생성.

v8 = 본 모델(v7 튜닝 LightGBM) + B 전문 모델 S1·S2·S3 의 logit 을 로지스틱 회귀로 스태킹
     (playground_s6e11_b_specialist.py 에서 OOF AUC 0.94779, v7 대비 +0.00025, 5/5 fold 상승)

테스트 예측
  - 본 모델: output/submission_lightgbm_tuned.csv (v7, 5-fold 모델 평균)
  - S1·S2·S3: 같은 5-fold 로 다시 학습해 fold 모델 평균 (S2 는 A 신호 없는 테스트 행에만)
  - 스태커: 학습 OOF 전체(output/oof_b_specialist.csv)로 학습 후 테스트에 적용
결과: output/submission_b_stacking.csv
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import playground_s6e11_b_specialist as B
import playground_s6e11_lightgbm as G
import playground_s6e11_linear_baseline as L


def no_a_mask(df):
    return ((df[L.BOOLEAN].sum(axis=1) == 0) & (df.cvv_retry_count == 0)
            & (df.auth_method != "No Authentication")).to_numpy()


def test_lgb(X, y, Xte, splits, rows=None, rows_te=None):
    params = B.lgb_params()
    pte = np.zeros(len(Xte))
    Xt = Xte if rows_te is None else Xte[rows_te]
    for k, (tr_i, va_i) in enumerate(splits):
        if rows is not None:
            tr_i, va_i = tr_i[rows[tr_i]], va_i[rows[va_i]]
        dtr = lgb.Dataset(X.iloc[tr_i], y[tr_i])
        dva = lgb.Dataset(X.iloc[va_i], y[va_i], reference=dtr)
        m = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        p = m.predict(Xt, num_iteration=m.best_iteration)
        if rows_te is None:
            pte += p / len(splits)
        else:
            pte[rows_te] += p / len(splits)
        print(f"    fold {k} 완료 (iter {m.best_iteration})", flush=True)
    return pte


def test_spline(tr, y, te, splits):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

    def prep(df):
        X = df[B.B].copy()
        X["amount_usd"] = np.log1p(X["amount_usd"])
        X["hours_since_last_txn"] = np.log1p(X["hours_since_last_txn"])
        return X

    Xtr, Xte = prep(tr), prep(te)
    pte = np.zeros(len(te))
    for k, (tr_i, _) in enumerate(splits):
        pre = ColumnTransformer([
            ("sp", make_pipeline(SplineTransformer(n_knots=6, degree=3, extrapolation="constant"), StandardScaler()), B.B_NUM),
            ("oh", OneHotEncoder(handle_unknown="ignore"), B.B_CAT)])
        m = make_pipeline(pre, LogisticRegression(max_iter=3000, C=1.0)).fit(Xtr.iloc[tr_i], y[tr_i])
        pte += m.predict_proba(Xte)[:, 1] / len(splits)
        print(f"    fold {k} 완료", flush=True)
    return pte


def features(p_main, s1, s2, s3, no_a):
    lg = B.logit
    return np.column_stack([lg(p_main), lg(s1), lg(s3), no_a.astype(float), no_a * lg(np.nan_to_num(s2, nan=0.5))])


def main():
    tr, te = L.load()
    y = tr[L.TARGET].astype(int).to_numpy()
    X, Xte = G.prepare(tr)[B.B], G.prepare(te)[B.B]
    Xte["merchant_category"] = Xte["merchant_category"].cat.set_categories(X["merchant_category"].cat.categories)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(X, y))
    na_tr, na_te = no_a_mask(tr), no_a_mask(te)

    sub_main = pd.read_csv(L.OUT / "submission_lightgbm_tuned.csv")
    assert (sub_main["id"].to_numpy() == te["id"].to_numpy()).all()
    p_main_te = sub_main[L.TARGET].to_numpy()

    print("[S1 테스트 예측]", flush=True)
    s1_te = test_lgb(X, y, Xte, splits)
    print("[S2 테스트 예측]", flush=True)
    s2_te = np.full(len(te), np.nan)
    s2_te[na_te] = test_lgb(X, y, Xte, splits, rows=na_tr, rows_te=na_te)[na_te]
    print("[S3 테스트 예측]", flush=True)
    s3_te = test_spline(tr, y, te, splits)

    oof = pd.read_csv(L.OUT / "oof_b_specialist.csv")
    Z = features(oof.p_main.to_numpy(), oof.s1.to_numpy(), oof.s2.to_numpy(), oof.s3.to_numpy(), na_tr)
    Zte = features(p_main_te, s1_te, s2_te, s3_te, na_te)
    stacker = LogisticRegression(max_iter=2000).fit(Z, y)
    print("\n스태커 계수 [본, S1, S3, A신호없음, S2]:", np.round(stacker.coef_[0], 3), "절편", round(stacker.intercept_[0], 3))
    print(f"(참고) 학습 OOF 에 같은 스태커 적용 AUC {roc_auc_score(y, stacker.predict_proba(Z)[:, 1]):.5f}  "
          f"/ 교차검증 스태킹 OOF {roc_auc_score(y, oof["stack"]):.5f}")
    p = stacker.predict_proba(Zte)[:, 1]
    r = np.corrcoef(pd.Series(p).rank(), pd.Series(p_main_te).rank())[0, 1]
    print(f"테스트: v7 과 순위 상관 {r:.5f}, 평균 확률 {p.mean():.4f} (v7 {p_main_te.mean():.4f})")
    pd.DataFrame({"id": te["id"], L.TARGET: p}).to_csv(L.OUT / "submission_b_stacking.csv", index=False)
    print("저장: output/submission_b_stacking.csv")


if __name__ == "__main__":
    main()
