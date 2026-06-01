"""
Ablation study: exploration prior exponent α

Tests q0[venue] = 1 / popularity^α for α ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}

α=0.0 → uniform prior (standard BiRank, no popularity penalty)
α=0.5 → inverse-sqrt (moderate penalty)
α=1.0 → inverse-popularity (aggressive penalty)
α=2.0 → inverse-square (very aggressive)

Metrics: rising-stars ρ (PRIMARY) + NDCG@10 on both London and UK FSQ.
Saves: ablation_exploration_prior.csv
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from bvr.core.validation import (
    compute_user_features, compute_venue_features,
    build_decayed_edges, build_adjacency, birank, evaluate_per_user,
)
from bvr.pipelines.london import (
    temporal_split, compute_rising_stars, spearman_rising,
)

DATA_DIR = Path(__file__).parent
ALPHAS   = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
DECAY_LAM = 0.5


def exploration_prior_alpha(user_feat, venue_feat, u2i, v2i, alpha: float):
    """Parameterised exploration prior: q0[v] = 1 / (1 + popularity)^alpha."""
    nu, nv = len(u2i), len(v2i)
    uv_map   = user_feat.set_index("user_id")["unique_venues"]
    top1_map = user_feat.set_index("user_id")["top1_venue_share"]
    pop_map  = venue_feat.set_index("business_id")["popularity_visits"]

    p0 = np.ones(nu)
    for uid, idx in u2i.items():
        uv = float(uv_map.get(uid, 1) or 1)
        t1 = float(top1_map.get(uid, 1) or 1)
        p0[idx] = np.log1p(uv) * (1 - t1)

    q0 = np.ones(nv)
    for vid, idx in v2i.items():
        pop = float(pop_map.get(vid, 1) or 1)
        if alpha == 0.0:
            q0[idx] = 1.0          # uniform — standard BiRank
        else:
            q0[idx] = 1.0 / (np.log1p(pop) ** alpha)

    return np.clip(p0, 1e-10, None), np.clip(q0, 1e-10, None)


def run_ablation(interactions_file, split_date, label):
    print(f"\n{'='*60}")
    print(f"Ablation: {label}  |  split {split_date}")
    df = pd.read_csv(interactions_file, dtype={"user_id": str, "business_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["stars"] = np.nan

    train, test_uv_rev, train_uv, test_traffic = temporal_split(df, split_date)
    user_feat  = compute_user_features(train)
    venue_feat = compute_venue_features(train)
    decay_e    = build_decayed_edges(train, split_date, lam=DECAY_LAM)

    test_full   = df[df["timestamp"] >= pd.Timestamp(split_date)].copy()
    rising_stars = compute_rising_stars(train, test_full)

    W, u2i, v2i, i2u, i2v = build_adjacency(decay_e)

    results = []
    print(f"{'α':>6} {'ρ(rising)':>12} {'p-value':>12} {'NDCG@10':>10} {'Hit@10':>10}")
    print("-" * 55)

    for alpha in ALPHAS:
        p0, q0 = exploration_prior_alpha(user_feat, venue_feat, u2i, v2i, alpha)
        _, q   = birank(W, p0=p0, q0=q0)
        ranking = {i2v[i]: float(q[i]) for i in range(len(i2v))}

        rho, pval = spearman_rising(ranking, rising_stars)
        agg, _, _ = evaluate_per_user(ranking, train_uv, test_uv_rev)

        ndcg = agg.get("NDCG@10", 0)
        hit  = agg.get("Hit@10",  0)
        sig  = "***" if pval < 0.001 else ("*" if pval < 0.05 else "ns")
        print(f"{alpha:>6.2f} {rho:>+12.4f} {pval:>12.4f} {sig:>3} {ndcg:>10.4f} {hit:>10.4f}")

        results.append({
            "dataset": label, "alpha": alpha,
            "rho_rising": rho, "p_value": pval,
            "ndcg_10": ndcg, "hit_10": hit,
        })

    return results


if __name__ == "__main__":
    all_results = []

    all_results += run_ablation(
        DATA_DIR / "london_interactions.csv", "2018-01-01", "London TripAdvisor"
    )
    all_results += run_ablation(
        DATA_DIR / "uk_fsq_interactions.csv", "2013-07-01", "UK Foursquare"
    )

    df_out = pd.DataFrame(all_results)
    df_out.to_csv(DATA_DIR / "ablation_exploration_prior.csv", index=False)
    print(f"\nSaved → ablation_exploration_prior.csv")
    print(df_out.to_string(index=False))
