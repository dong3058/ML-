"""Playground S6E11 (Predicting Credit Card Fraud) 놓친 사기 유형 군집화.

놓친 사기(선형 v2·LightGBM v3 모두 상위 k 에 못 든 사기, playground_s6e11_missed_fraud.py 와 같은 정의)를
LightGBM SHAP 벡터(피처별 기여도, log-odds) 공간에서 KMeans 로 군집화한다.
SHAP 공간을 쓰는 이유: '어떤 신호 때문에 조금 의심받았나' 로 유형을 나누기 위해서.

군집 수: k=3..8 중 silhouette 점수가 가장 높은 k.
닮은 정상 거래: 지목되지 않은 모든 거래(예측 < 기준선)를 가장 가까운 군집 중심에 배정해
             군집마다 '같은 모양인 거래 중 실제 사기 비율' 을 계산한다.
결과: output/missed_clusters.json (아티팩트용), output/missed_clusters_rows.csv
"""

import contextlib
import io
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import playground_s6e11_linear_baseline as L
import playground_s6e11_missed_fraud as M

KEY_FEATURES = ["cvv_retry_count", "merchant_risk_score", "velocity_score", "customer_age", "amount_usd",
                "txn_count_last_24h", "hours_since_last_txn", "prior_disputes"]


def main():
    tr, _ = L.load()
    with contextlib.redirect_stdout(io.StringIO()):
        d = M.groups(tr)
    k_top = int(d[L.TARGET].sum())
    cut = float(d.p_lgb.nlargest(k_top).min())
    print("SHAP 계산 (5-fold)...", flush=True)
    cache = L.OUT / "shap_oof_v3.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        sh = pd.DataFrame(z["v"], columns=list(z["c"]), index=d.index)
    else:
        sh = M.shap_oof(d)
        np.savez(cache, v=sh.to_numpy(), c=np.array(sh.columns))
    feats = [c for c in sh.columns if c != "base"]

    miss = (d.group == "missed").to_numpy()
    S = sh.loc[miss, feats].to_numpy()

    print("\n[군집 수 선택] silhouette", flush=True)
    best, sil = None, {}
    for k in range(3, 9):
        km = KMeans(k, n_init=10, random_state=L.SEED).fit(S)
        sc = silhouette_score(S, km.labels_, sample_size=4000, random_state=L.SEED)
        print(f"  k={k}: {sc:.3f}")
        sil[k] = float(sc)
        if best is None or sc > best[0]:
            best = (sc, k, km)
    _, K, km = best
    print(f"선택 k={K}")

    # 군집 정렬: 크기 순
    order = np.argsort(-np.bincount(km.labels_))
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[x] for x in km.labels_])
    centers = km.cluster_centers_[order]

    # 닮은 거래: 지목되지 않은 모든 거래를 가장 가까운 중심에 배정
    unflag = (d.p_lgb < cut).to_numpy()
    U = sh.loc[unflag, feats].to_numpy()
    dist = ((U[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    ulab = dist.argmin(1)
    uy = d.loc[unflag, L.TARGET].to_numpy()

    flags = d[L.BOOLEAN].astype(int).sum(axis=1) + (d.cvv_retry_count > 0) + (d.auth_method == "No Authentication")
    dm = d[miss].assign(cluster=lab, n_flags=flags[miss].to_numpy())
    normal = d[d.group == "normal"]

    clusters = []
    for c in range(K):
        g = dm[dm.cluster == c]
        mean_shap = sh.loc[miss, feats][lab == c].mean()
        top = mean_shap.sort_values(ascending=False)
        look = ulab == c
        clusters.append({
            "id": c, "n": int(len(g)), "share": float(len(g) / miss.sum()),
            "p_mean": float(g.p_lgb.mean()), "p_median": float(g.p_lgb.median()),
            "p_q": np.percentile(g.p_lgb, [10, 25, 50, 75, 90]).tolist(),
            "flags_mean": float(g.n_flags.mean()),
            "flags_dist": g.n_flags.clip(upper=4).value_counts(normalize=True).reindex(range(5), fill_value=0).tolist(),
            "shap": {f: float(v) for f, v in mean_shap.items()},
            "shap_sum": float(mean_shap.sum()),
            "top_pos": [[f, float(v)] for f, v in top.head(4).items()],
            "top_neg": [[f, float(v)] for f, v in top.tail(3).items()],
            "num": {f: float(g[f].mean()) for f in KEY_FEATURES},
            "bool": {f: float(g[f].mean()) for f in L.BOOLEAN},
            "auth": g.auth_method.value_counts(normalize=True).to_dict(),
            "category": g.merchant_category.value_counts(normalize=True).head(4).to_dict(),
            "lookalike_n": int(look.sum()), "lookalike_fraud": int(uy[look].sum()),
            "lookalike_rate": float(uy[look].mean()),
        })
        print(f"\n군집 {c}: {len(g)}건, 평균 p {g.p_lgb.mean():.3f}, 신호 {g.n_flags.mean():.2f}, "
              f"닮은 거래 {look.sum()} 중 사기 {uy[look].mean():.2%}")
        print("  +", [(f, round(v, 2)) for f, v in top.head(4).items()])

    ref = {
        "normal": {"num": {f: float(normal[f].mean()) for f in KEY_FEATURES},
                   "bool": {f: float(normal[f].mean()) for f in L.BOOLEAN},
                   "auth_none": float((normal.auth_method == "No Authentication").mean())},
        "caught": {"num": {f: float(d.loc[d.group == "caught", f].mean()) for f in KEY_FEATURES},
                   "bool": {f: float(d.loc[d.group == "caught", f].mean()) for f in L.BOOLEAN},
                   "auth_none": float((d.loc[d.group == "caught", "auth_method"] == "No Authentication").mean())},
    }
    # 2D 지도용 (PCA)
    Sc = S - S.mean(0)
    _, _, vt = np.linalg.svd(Sc, full_matrices=False)
    xy = Sc @ vt[:2].T
    rng = np.random.default_rng(L.SEED)
    pick = rng.choice(len(S), size=min(2500, len(S)), replace=False)
    pca_load = {f: [float(vt[0, i]), float(vt[1, i])] for i, f in enumerate(feats)}

    out = {"K": K, "cut": cut, "n_missed": int(miss.sum()), "base": float(sh.base.mean()),
           "clusters": clusters, "ref": ref, "features": feats,
           "map": {"x": xy[pick, 0].round(3).tolist(), "y": xy[pick, 1].round(3).tolist(),
                   "c": lab[pick].tolist(), "p": d.p_lgb.to_numpy()[miss][pick].round(4).tolist()},
           "pca_load": pca_load,
           "silhouette": sil}
    (L.OUT / "missed_clusters.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    dm[["id", "cluster", "p_lgb", "n_flags"]].to_csv(L.OUT / "missed_clusters_rows.csv", index=False)
    print("\n저장: output/missed_clusters.json, output/missed_clusters_rows.csv")


if __name__ == "__main__":
    main()
