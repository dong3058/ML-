"""Playground S6E11 (Predicting Credit Card Fraud) 선형(v2) vs LightGBM(v3) 오차 상관 + 오차 분포.

입력: output/oof_linear.csv, output/oof_lightgbm.csv (각 스크립트가 저장한 OOF 예측)
오차 정의
  - 잔차 e = y − p  (사기(y=1)면 1−p, 정상(y=0)이면 −p. 0 에 가까울수록 정확)
  - 행별 log loss  (확신하고 틀릴수록 크게 벌점)
상관은 전체 / 사기 행 / 정상 행으로 나눠 본다. 사기 비율이 1.7% 라 전체 상관은 정상 행이 좌우한다.
마지막으로 두 모델을 섞었을 때(순위 평균) AUC 가 오르는지 확인한다.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

import playground_s6e11_linear_baseline as L

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load():
    lin = pd.read_csv(L.OUT / "oof_linear.csv")
    lgb = pd.read_csv(L.OUT / "oof_lightgbm.csv")
    assert (lin["id"] == lgb["id"]).all()
    d = pd.DataFrame({"id": lin["id"], "y": lin["y"], "p_lin": lin["p"], "p_lgb": lgb["p"],
                      "loss_lin": lin["loss"], "loss_lgb": lgb["loss"]})
    d["e_lin"], d["e_lgb"] = d["y"] - d["p_lin"], d["y"] - d["p_lgb"]
    return d


def correlations(d):
    rows = []
    for name, m in [("전체", slice(None)), ("사기 (y=1)", d.y == 1), ("정상 (y=0)", d.y == 0)]:
        s = d[m]
        rows.append({
            "구간": name, "n": len(s),
            "잔차 pearson": pearsonr(s.e_lin, s.e_lgb)[0],
            "잔차 spearman": spearmanr(s.e_lin, s.e_lgb)[0],
            "logloss pearson": pearsonr(s.loss_lin, s.loss_lgb)[0],
        })
    t = pd.DataFrame(rows).round(3)
    print("[오차 상관]")
    print(t.to_string(index=False))

    # 누가 더 많이 틀렸나 + 서로 다른 행을 틀리는가
    k = int(d.y.sum())  # 실제 사기 건수만큼 상위로 뽑았을 때
    top_lin = set(d.nlargest(k, "p_lin").id)
    top_lgb = set(d.nlargest(k, "p_lgb").id)
    fraud = set(d[d.y == 1].id)
    print(f"\n[상위 {k}건(=실제 사기 수) 겹침] 두 모델 상위 집합 겹침 {len(top_lin & top_lgb) / k:.1%}")
    print(f"  잡은 사기: 선형 {len(top_lin & fraud)}, LightGBM {len(top_lgb & fraud)}, "
          f"둘 다 {len(top_lin & top_lgb & fraud)}, 선형만 {len(top_lin & fraud - top_lgb)}, "
          f"LightGBM만 {len(top_lgb & fraud - top_lin)}, 둘 다 놓침 {len(fraud - top_lin - top_lgb)}")
    better = (d.loss_lgb < d.loss_lin).groupby(d.y).mean()
    print(f"  LightGBM 쪽 log loss 가 더 작은 행 비율: 정상 {better[0]:.1%}, 사기 {better[1]:.1%}")
    return t


def distributions(d):
    print("\n[오차 분포] 잔차 |e| 분위수")
    q = [0.5, 0.9, 0.99, 0.999]
    for cls, name in [(1, "사기"), (0, "정상")]:
        s = d[d.y == cls]
        t = pd.DataFrame({m: s[f"e_{m}"].abs().quantile(q).values for m in ["lin", "lgb"]},
                         index=[f"{int(x * 1000) / 10}%" for x in q])
        t.loc["평균"] = [s.e_lin.abs().mean(), s.e_lgb.abs().mean()]
        print(f"  {name} (n={len(s)})")
        print(t.round(4).to_string())
    print("\n  log loss 합 기여: 사기 행이 전체 log loss 중 차지하는 비율 "
          f"선형 {d.loc[d.y == 1, 'loss_lin'].sum() / d.loss_lin.sum():.1%}, "
          f"LightGBM {d.loc[d.y == 1, 'loss_lgb'].sum() / d.loss_lgb.sum():.1%}")


def blend(d):
    r_lin, r_lgb = rankdata(d.p_lin) / len(d), rankdata(d.p_lgb) / len(d)
    ws = np.round(np.arange(0, 1.01, 0.1), 1)
    aucs = [roc_auc_score(d.y, w * r_lin + (1 - w) * r_lgb) for w in ws]
    print("\n[순위 평균 블렌드] 선형 가중치별 AUC")
    print("  " + "  ".join(f"{w:.1f}:{a:.5f}" for w, a in zip(ws, aucs)))
    return ws, aucs


def plot(d, ws, aucs, path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    bins = np.linspace(0, 1, 51)

    for a, cls, title in [(ax[0, 0], 1, "사기 행 (y=1): 예측 확률 분포  — 오른쪽(1)일수록 정확"),
                          (ax[0, 1], 0, "정상 행 (y=0): 예측 확률 분포  — 왼쪽(0)일수록 정확, y축 로그")]:
        s = d[d.y == cls]
        for col, c, lab in [("p_lin", BLUE, "선형 (v2)"), ("p_lgb", ORANGE, "LightGBM (v3)")]:
            a.hist(s[col], bins=bins, histtype="step", linewidth=2, color=c, label=lab)
        if cls == 0:
            a.set_yscale("log")
        a.set_title(title, color=INK, fontsize=11, loc="left")
        a.set_xlabel("예측 사기 확률 p")
        a.set_ylabel("행 수")
        a.legend(frameon=False)

    a = ax[1, 0]
    f = d[d.y == 1]
    a.scatter(f.p_lin, f.p_lgb, s=6, alpha=0.35, color=BLUE, linewidths=0)
    a.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle="--")
    r = pearsonr(f.e_lin, f.e_lgb)[0]
    a.set_title(f"사기 행: 두 모델 예측 비교 (잔차 상관 r={r:.3f})", color=INK, fontsize=11, loc="left")
    a.set_xlabel("선형 예측 p")
    a.set_ylabel("LightGBM 예측 p")
    a.text(0.03, 0.93, "대각선 위: LightGBM 이 더 잘 잡음", transform=a.transAxes, color=INK2, fontsize=9)
    a.text(0.55, 0.05, "대각선 아래: 선형이 더 잘 잡음", transform=a.transAxes, color=INK2, fontsize=9)

    a = ax[1, 1]
    a.plot(ws, aucs, color=BLUE, linewidth=2, marker="o", markersize=8)
    best = int(np.argmax(aucs))
    a.text(0.04, 0.12, f"최고 {aucs[best]:.5f} (선형 {ws[best]:.1f})\nLightGBM 단독 {aucs[0]:.5f}",
           transform=a.transAxes, color=INK, fontsize=9)
    a.set_title("순위 평균 블렌드: 선형 가중치별 OOF AUC", color=INK, fontsize=11, loc="left")
    a.set_xlabel("선형 가중치 (0 = LightGBM 만, 1 = 선형만)")
    a.set_ylabel("OOF ROC AUC")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("\n그림 저장:", path)


def main():
    d = load()
    correlations(d)
    distributions(d)
    ws, aucs = blend(d)
    plot(d, ws, aucs, L.OUT / "error_correlation.png")


if __name__ == "__main__":
    main()
