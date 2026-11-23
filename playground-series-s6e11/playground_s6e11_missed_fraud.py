"""Playground S6E11 (Predicting Credit Card Fraud) 두 모델이 모두 놓친 사기 분석.

그룹 정의 (각 모델에서 예측 확률 상위 k건 = 실제 사기 수 를 '사기로 지목' 으로 본다)
  - missed : 실제 사기인데 선형(v2)·LightGBM(v3) 둘 다 지목 못함
  - caught : 실제 사기이고 둘 다 지목함
  - normal : 실제 정상
1) 피처 프로파일: 놓친 사기가 각 피처에서 정상 쪽에 가까운지, 잡힌 사기 쪽에 가까운지
   위치 = (missed − normal) / (caught − normal). 0 이면 정상과 같음, 1 이면 잡힌 사기와 같음.
2) LightGBM 기여도(SHAP, pred_contrib): 놓친 사기 각 행에서 어떤 피처가 예측을 얼마나 내렸나 (log-odds).
   OOF 와 같은 fold 로 다시 학습해 검증 fold 행에 대해 계산한다.
"""

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import playground_s6e11_error_correlation as E
import playground_s6e11_lightgbm as G
import playground_s6e11_linear_baseline as L


def groups(tr):
    d = E.load()
    k = int(d.y.sum())
    top_lin = d.p_lin.rank(ascending=False, method="first") <= k
    top_lgb = d.p_lgb.rank(ascending=False, method="first") <= k
    g = np.where(d.y == 0, "normal", np.where(top_lin & top_lgb, "caught",
                 np.where(~top_lin & ~top_lgb, "missed", "one_model")))
    out = tr.copy()
    out["group"], out["p_lgb"], out["p_lin"] = g, d.p_lgb.to_numpy(), d.p_lin.to_numpy()
    print(out.group.value_counts().to_string(), "\n")
    print("그룹별 평균 예측 확률 (%)")
    print((out.groupby("group")[["p_lin", "p_lgb"]].mean() * 100).round(2).to_string(), "\n")
    return out


def profile(d):
    rows = []
    m = {g: d[d.group == g] for g in ["missed", "caught", "normal"]}
    num = [c for c in L.numeric_cols(d) if c not in ("group", "p_lgb", "p_lin")]
    for c in num + L.BOOLEAN:
        v = {g: m[g][c].astype(float).mean() for g in m}
        rows.append({"feature": c, "level": "평균", **v})
    for c in L.CATEGORICAL:
        # 잡힌 사기에서 정상 대비 가장 많이 늘어나는 범주 하나로 비교
        share = {g: m[g][c].value_counts(normalize=True) for g in m}
        lv = (share["caught"] - share["normal"]).idxmax()
        rows.append({"feature": c, "level": lv, **{g: share[g].get(lv, 0) for g in m}})
    t = pd.DataFrame(rows)
    t["position"] = (t.missed - t.normal) / (t.caught - t.normal)
    t["gap_caught_vs_normal"] = (t.caught - t.normal).abs()
    t = t.sort_values("position")
    print("[피처 프로파일] position 0 = 정상과 같음, 1 = 잡힌 사기와 같음 (bool 은 True 비율)")
    print(t.round(3).to_string(index=False))
    return t


def shap_oof(d):
    X = G.prepare(d.drop(columns=["group", "p_lgb", "p_lin"]))
    y = d[L.TARGET].astype(int).to_numpy()
    contrib = np.zeros((len(X), X.shape[1] + 1))
    skf = StratifiedKFold(5, shuffle=True, random_state=L.SEED)
    for k, (tr_i, va_i) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[tr_i], y[tr_i])
        dva = lgb.Dataset(X.iloc[va_i], y[va_i], reference=dtr)
        mdl = lgb.train(G.PARAMS, dtr, 5000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(200, verbose=False)])
        contrib[va_i] = mdl.predict(X.iloc[va_i], num_iteration=mdl.best_iteration, pred_contrib=True)
        print(f"  fold {k} 기여도 계산 완료")
    return pd.DataFrame(contrib, columns=list(X.columns) + ["base"], index=d.index)


def shap_summary(d, sh):
    feats = [c for c in sh.columns if c != "base"]
    mean = pd.DataFrame({g: sh.loc[d.group == g, feats].mean() for g in ["missed", "caught", "normal"]})
    mean["missed−caught"] = mean.missed - mean.caught
    mean = mean.sort_values("missed−caught")
    print("\n[LightGBM 기여도 평균 (log-odds), +는 사기 쪽으로 올림]")
    print(mean.round(3).to_string())

    ms = sh.loc[d.group == "missed", feats]
    top_down = ms.idxmin(axis=1).value_counts(normalize=True) * 100
    top_up = ms.idxmax(axis=1).value_counts(normalize=True) * 100
    print("\n[놓친 사기 행별] 예측을 가장 많이 내린 피처 (행 비율 %)")
    print(top_down.round(1).head(10).to_string())
    print("\n[놓친 사기 행별] 예측을 가장 많이 올린 피처 (행 비율 %)")
    print(top_up.round(1).head(10).to_string())
    print(f"\n놓친 사기 중 '양의 기여 합' 이 0.5 미만인 행 (사기 신호가 거의 없음): "
          f"{(ms.clip(lower=0).sum(axis=1) < 0.5).mean():.1%}")
    return mean


def plot(prof, mean, path):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7.5))

    a = ax[0]
    # 잡힌 사기에서 기여도가 거의 없는 피처(사기와 무관)는 position 이 의미 없어서 제외
    keep = mean.index[mean.caught.abs() >= 0.02]
    p = prof[prof.feature.isin(keep)].sort_values("position")
    labels = [f"{f}" + (f" = {lv}" if lv != "평균" else "") for f, lv in zip(p.feature, p.level)]
    a.barh(labels, p.position, color=E.BLUE, height=0.7)
    a.axvline(0, color=E.INK2, linewidth=1)
    a.axvline(1, color=E.INK2, linewidth=1, linestyle="--")
    a.set_title("놓친 사기의 피처 위치\n0 = 정상 거래와 같음, 1 = 잡힌 사기와 같음",
                color=E.INK, fontsize=11, loc="left")
    a.set_xlabel("position")
    a.tick_params(axis="y", labelsize=8)

    b = ax[1]
    m = mean.sort_values("caught")
    yy = np.arange(len(m))
    b.barh(yy + 0.2, m.caught, height=0.38, color=E.ORANGE, label="잡힌 사기")
    b.barh(yy - 0.2, m.missed, height=0.38, color=E.BLUE, label="놓친 사기")
    b.set_yticks(yy, m.index, fontsize=8)
    b.axvline(0, color=E.INK2, linewidth=1)
    b.set_title("LightGBM 피처별 평균 기여도 (log-odds)\n+ 사기 쪽으로 올림, - 정상 쪽으로 내림",
                color=E.INK, fontsize=11, loc="left")
    b.set_xlabel("평균 기여도")
    b.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("\n그림 저장:", path)


def main():
    tr, _ = L.load()
    d = groups(tr)
    prof = profile(d)
    sh = shap_oof(d)
    mean = shap_summary(d, sh)
    pd.concat([d[["id", "group", "p_lin", "p_lgb"]], sh.add_prefix("shap_")], axis=1)[
        d.group == "missed"].to_csv(L.OUT / "missed_fraud_rows.csv", index=False)
    plot(prof, mean, L.OUT / "missed_fraud.png")


if __name__ == "__main__":
    main()
