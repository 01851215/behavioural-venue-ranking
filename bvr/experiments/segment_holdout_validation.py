"""
A3 — Held-out segment validation.

Tests whether Loyalist NDCG@10 = 0.17 is circular.

Method:
  - Split users 80/20 BEFORE K-means clustering
  - Train clustering on 80% only
  - Classify held-out 20% using kmeans.predict()
  - Evaluate NDCG@10 on held-out users only

If Loyalist NDCG stays > 0.3 on held-out users, the result is robust.
If it drops to ~0.1 (near-random), the original was circular.

Saves: data/results/segment_validation_holdout.csv
"""

import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
warnings.filterwarnings("ignore")

from bvr.core.validation import (
    compute_user_features, compute_venue_features,
    build_decayed_edges, build_birank_ranking, build_popularity_ranking,
    evaluate_per_user,
)
from bvr.pipelines.london import temporal_split

DATA_DIR = Path(__file__).parent.parent.parent / "data"
YELP_DIR = Path(__file__).parent.parent.parent.parent.parent / "Desktop/Master Project/yelp_dataset"

COFFEE_FILE = Path("/Users/chris/Desktop/Master Project/yelp_dataset/coffee_interactions.csv")
SPLIT_DATE  = "2020-01-01"
N_CLUSTERS  = 4
HOLDOUT_FRAC = 0.20


def run():
    print("Loading coffee interactions...")
    df = pd.read_csv(COFFEE_FILE, dtype={"user_id": str, "business_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["stars"] = np.nan

    train, test_uv_rev, train_uv, _ = temporal_split(df, SPLIT_DATE)
    print(f"  Train: {len(train):,} | Revisit users: {len(test_uv_rev):,}")

    # Compute features on training data
    user_feat  = compute_user_features(train)
    venue_feat = compute_venue_features(train)

    # 80/20 user split BEFORE clustering
    all_users = user_feat["user_id"].unique()
    np.random.seed(42)
    holdout_mask = np.random.rand(len(all_users)) < HOLDOUT_FRAC
    train_users  = set(all_users[~holdout_mask])
    holdout_users = set(all_users[holdout_mask])

    print(f"  Train users for clustering: {len(train_users):,}")
    print(f"  Held-out users:             {len(holdout_users):,}")

    # Feature columns for clustering
    feat_cols = ["total_visits", "unique_venues", "revisit_ratio",
                 "top1_venue_share", "venue_entropy"]
    feat_cols = [c for c in feat_cols if c in user_feat.columns]

    # Fit K-means on training users only
    uf_train = user_feat[user_feat["user_id"].isin(train_users)][feat_cols].fillna(0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(uf_train.values)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(X_train)

    # Classify held-out users
    uf_holdout = user_feat[user_feat["user_id"].isin(holdout_users)].copy()
    X_holdout = scaler.transform(uf_holdout[feat_cols].fillna(0).values)
    uf_holdout["cluster"] = kmeans.predict(X_holdout)
    uf_holdout["user_id"] = uf_holdout["user_id"].values  # ensure string

    # Identify the Loyalist cluster (highest revisit_ratio)
    if "revisit_ratio" in feat_cols:
        cluster_rr = uf_holdout.groupby("cluster")["revisit_ratio"].mean()
        loyalist_cluster = int(cluster_rr.idxmax())
        print(f"  Loyalist cluster: {loyalist_cluster} (revisit_ratio={cluster_rr.max():.3f})")
    else:
        loyalist_cluster = 0

    # Build ranking
    decay_e = build_decayed_edges(train, SPLIT_DATE, lam=0.5)
    ranking = build_birank_ranking(decay_e, user_feat, venue_feat)

    # Evaluate on ALL held-out users and per-cluster
    holdout_uv_rev = {u: s for u, s in test_uv_rev.items() if u in holdout_users}
    print(f"  Held-out revisit users: {len(holdout_uv_rev):,}")

    rows = []
    for cluster_id in range(N_CLUSTERS):
        cluster_users = set(uf_holdout[uf_holdout["cluster"] == cluster_id]["user_id"])
        cluster_uv_rev = {u: s for u, s in holdout_uv_rev.items() if u in cluster_users}
        if len(cluster_uv_rev) < 3:
            continue

        agg, _, _ = evaluate_per_user(ranking, train_uv, cluster_uv_rev)
        label = "Loyalist (held-out)" if cluster_id == loyalist_cluster else f"Cluster {cluster_id} (held-out)"
        print(f"  {label:<30} n={len(cluster_uv_rev):>4}  NDCG@10={agg.get('NDCG@10',0):.4f}")
        rows.append({
            "cluster": cluster_id,
            "label": label,
            "n_users": len(cluster_uv_rev),
            "ndcg10": round(agg.get("NDCG@10", 0), 4),
            "hit10":  round(agg.get("Hit@10", 0), 4),
            "is_loyalist": cluster_id == loyalist_cluster,
        })

    # Also all held-out
    agg_all, _, _ = evaluate_per_user(ranking, train_uv, holdout_uv_rev)
    print(f"  {'All held-out':<30} n={len(holdout_uv_rev):>4}  NDCG@10={agg_all.get('NDCG@10',0):.4f}")
    rows.append({
        "cluster": -1, "label": "All held-out", "n_users": len(holdout_uv_rev),
        "ndcg10": round(agg_all.get("NDCG@10", 0), 4),
        "hit10":  round(agg_all.get("Hit@10", 0), 4), "is_loyalist": False,
    })

    df_out = pd.DataFrame(rows)
    out = Path(__file__).parent.parent.parent / "data/results/segment_validation_holdout.csv"
    df_out.to_csv(out, index=False)
    print(f"\nSaved → {out}")

    loyalist_ndcg = df_out[df_out["is_loyalist"]]["ndcg10"].values
    if len(loyalist_ndcg) > 0:
        v = loyalist_ndcg[0]
        verdict = "ROBUST — Loyalist NDCG significantly above average" if v > 0.12 \
                  else "CIRCULAR — Loyalist NDCG collapses on held-out users"
        print(f"\nVERDICT: Loyalist held-out NDCG@10 = {v:.4f} → {verdict}")


if __name__ == "__main__":
    run()
