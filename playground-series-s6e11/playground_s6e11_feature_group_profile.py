"""Playground S6E11 (Predicting Credit Card Fraud) 피처별 그룹 비교 (card_age_months 분석을 전 피처로 확장).

그룹: 정상 / 놓친 사기 / 잡힌 사기 (playground_s6e11_missed_fraud.py 정의)
피처마다
 1) 그룹 차이: 놓친 사기 vs 정상, 잡힌 사기 vs 정상
    - 숫자형: 표준화 평균차 d = (그룹 평균 − 정상 평균) / 정상 표준편차, KS 검정 p
    - 범주형·bool: 총변동거리 TVD = ½ Σ|비율 차이| (0 = 분포 같음, 1 = 완전히 다름), 카이제곱 p
    - 위치 = (놓친 − 정상) / (잡힌 − 정상) 방향의 비율 (숫자형 d 기준, 범주형 TVD 기준)
 2) 값에 따른 사기율: 구간(연속형 10분위) 또는 값별 사기율
    - chi2 p, 사기율 범위, 단조성(구간 순서와 사기율의 spearman, 연속형만), 이웃 구간 들쭉날쭉 정도
 3) 신호 조합: 신호별 하위집단(CVV≥1 / 인증 없음 / 위험점수≥50 / 신호 없음) 안에서
    피처 5분위(또는 값)별 사기율의 범위 — 하위집단 평균 대비 비율
결과: output/feature_group_profile.csv
"""

import contextlib
import io

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, spearmanr

import playground_s6e11_linear_baseline as L
import playground_s6e11_missed_fraud as M

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def bins_of(s):
    if s.dtype == object or str(s.dtype) in ("str", "string", "category", "bool") or s.nunique() <= 24:
        return s.astype(str), False
    return pd.qcut(s, 10, duplicates="drop"), True


def tvd(a, b):
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    return 0.5 * pa.subtract(pb, fill_value=0).abs().sum()


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        tr, _ = L.load()
        d = M.groups(tr)
    y = d[L.TARGET]
    feats = [c for c in tr.columns if c not in ("id", L.TARGET)]
    subs = {
        "CVV≥1": d.cvv_retry_count > 0,
        "인증없음": d.auth_method == "No Authentication",
        "위험≥50": d.merchant_risk_score >= 50,
        "신호없음": (d[L.BOOLEAN].sum(axis=1) == 0) & (d.cvv_retry_count == 0) & (d.auth_method != "No Authentication"),
    }
    gN, gM, gC = (d[d.group == g] for g in ("normal", "missed", "caught"))

    rows = []
    for c in feats:
        s = d[c]
        b, cont = bins_of(s)
        r = {"feature": c, "kind": "연속" if cont else ("범주" if s.dtype.kind not in "if" else "이산")}
        if s.dtype.kind in "if" and s.dtype != bool:
            sd = gN[c].std()
            r["d_missed"] = (gM[c].mean() - gN[c].mean()) / sd
            r["d_caught"] = (gC[c].mean() - gN[c].mean()) / sd
            r["ks_p_missed"] = ks_2samp(gM[c], gN[c]).pvalue
            r["ks_p_caught"] = ks_2samp(gC[c], gN[c]).pvalue
            r["position"] = r["d_missed"] / r["d_caught"] if abs(r["d_caught"]) > 0.05 else np.nan
        else:
            sm, sc, sn = gM[c].astype(str), gC[c].astype(str), gN[c].astype(str)
            r["tvd_missed"], r["tvd_caught"] = tvd(sm, sn), tvd(sc, sn)
            r["chi_p_missed"] = chi2_contingency(pd.crosstab(pd.concat([sm, sn]), [0] * len(sm) + [1] * len(sn)))[1]
            r["chi_p_caught"] = chi2_contingency(pd.crosstab(pd.concat([sc, sn]), [0] * len(sc) + [1] * len(sn)))[1]
            r["position"] = r["tvd_missed"] / r["tvd_caught"] if r["tvd_caught"] > 0.02 else np.nan

        # 값(구간)별 사기율
        t = y.groupby(b, observed=True).agg(["size", "mean"])
        t = t[t["size"] >= 500]
        r["rate_min_%"], r["rate_max_%"] = t["mean"].min() * 100, t["mean"].max() * 100
        r["rate_chi_p"] = chi2_contingency(pd.crosstab(b, y))[1]
        if cont:
            r["monotonic_rho"] = spearmanr(np.arange(len(t)), t["mean"])[0]
            # 이웃 구간 들쭉날쭉: 이웃 차이 절대값 합 / (최대−최소) (1 이면 단조, 클수록 들쭉날쭉)
            rng = t["mean"].max() - t["mean"].min()
            r["zigzag"] = np.abs(np.diff(t["mean"])).sum() / rng if rng > 0 else np.nan

        # 신호 하위집단 안에서의 사기율 범위 (하위집단 평균 대비)
        for name, msk in subs.items():
            sub = d[msk]
            bb = pd.qcut(sub[c], 5, duplicates="drop") if cont else sub[c].astype(str)
            tt = sub[L.TARGET].groupby(bb, observed=True).agg(["size", "mean"])
            tt = tt[tt["size"] >= 300]
            base = sub[L.TARGET].mean()
            r[f"in_{name}"] = (tt["mean"].max() - tt["mean"].min()) / base if len(tt) > 1 else np.nan
        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(L.OUT / "feature_group_profile.csv", index=False)

    num = out[out.kind != "범주"].copy()
    num = num[num.d_caught.notna()].sort_values("d_caught", key=abs, ascending=False)
    print("[숫자형] 표준화 평균차 d (정상 대비), KS p, 위치(놓친/잡힌)")
    print(num[["feature", "kind", "d_missed", "d_caught", "position", "ks_p_missed", "ks_p_caught"]].round(3).to_string(index=False))
    cat = out[out.tvd_caught.notna()].sort_values("tvd_caught", ascending=False) if "tvd_caught" in out else out.iloc[:0]
    print("\n[범주·bool] TVD (정상 대비 분포 차이), 카이제곱 p")
    print(cat[["feature", "kind", "tvd_missed", "tvd_caught", "position", "chi_p_missed", "chi_p_caught"]].round(4).to_string(index=False))
    print("\n[값별 사기율] 범위, chi2 p, 단조성(rho), 들쭉날쭉(zigzag, 1=단조)")
    print(out[["feature", "rate_min_%", "rate_max_%", "rate_chi_p", "monotonic_rho", "zigzag"]]
          .sort_values("rate_max_%", ascending=False).round(4).to_string(index=False))
    print("\n[신호 하위집단 안 사기율 범위 / 하위집단 평균] (0 에 가까우면 그 신호 안에서 영향 없음)")
    print(out[["feature"] + [f"in_{k}" for k in subs]].round(3).to_string(index=False))
    print("\n하위집단 평균 사기율:", {k: round(d.loc[m, L.TARGET].mean() * 100, 2) for k, m in subs.items()},
          " 크기:", {k: int(m.sum()) for k, m in subs.items()})


if __name__ == "__main__":
    main()
