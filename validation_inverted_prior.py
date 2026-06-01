"""
V4: Inverted-feature baseline — negative control

Runs BiRank with INVERTED exploration prior:
    q0[venue] = log(1 + popularity)   ← popularity-AMPLIFYING (wrong direction)

This should perform WORSE than popularity baseline on rising-stars ρ.
If it doesn't, the prior design isn't doing what we claim.

Also runs the anti-loyalty prior (q0 = repeat_user_rate * -1 clamped positive)
as a second negative control.

Saves: inverted_prior_validation.csv
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from validate_v5 import (
    compute_user_features, compute_venue_features,
    build_decayed_edges, build_adjacency, birank, evaluate_per_user,
)
from run_london_pipeline import (
    temporal_split, compute_rising_stars, spearman_rising,
)

DATA_DIR  = Path(__file__).parent
DECAY_LAM = 0.5


def run_inverted_prior(interactions_file, split_date, label, has_stars=True):
    print(f"\n=== {label} ===")
    df = pd.read_csv(interactions_file, dtype={"user_id": str, "business_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not has_stars:
        df["stars"] = np.nan

    train, test_uv_rev, train_uv, _ = temporal_split(df, split_date)
    user_feat  = compute_user_features(train)
    venue_feat = compute_venue_features(train)
    decay_e    = build_decayed_edges(train, split_date, lam=DECAY_LAM)
    test_full  = df[df["timestamp"] >= pd.Timestamp(split_date)].copy()
    rising     = compute_rising_stars(train, test_full)

    W, u2i, v2i, i2u, i2v = build_adjacency(decay_e)
    n_users, n_venues = len(u2i), len(v2i)
    pop_map = venue_feat.set_index("business_id")["popularity_visits"]
    rr_map  = venue_feat.set_index("business_id")["repeat_user_rate"]

    rows = []

    # 1. INVERTED exploration prior — popularity AMPLIFYING (negative control)
    p0 = np.ones(n_users)
    q0_inv = np.array([
        float(np.log1p(pop_map.get(i2v[i], 1) or 1))   # ← amplifies popularity
        for i in range(n_venues)
    ])
    q0_inv = np.clip(q0_inv, 1e-10, None)
    _, q = birank(W, p0=p0, q0=q0_inv)
    r_inv_pop = {i2v[i]: float(q[i]) for i in range(n_venues)}
    rho, pval = spearman_rising(r_inv_pop, rising)
    agg, _, _ = evaluate_per_user(r_inv_pop, train_uv, test_uv_rev)
    rows.append({"dataset": label, "method": "inverted_popularity_prior",
                 "rho": rho, "p_value": pval, "ndcg10": agg.get("NDCG@10", 0),
                 "interpretation": "popularity-amplifying (should be WORSE)"})
    print(f"  inverted_pop_prior:  ρ={rho:+.4f} (p={pval:.4f})  NDCG={agg.get('NDCG@10',0):.4f}")

    # 2. INVERTED loyalty prior — high revisit rate penalised (anti-loyalty)
    q0_anti_loyal = np.array([
        float(1.0 / (rr_map.get(i2v[i], 0.01) + 0.01))  # ← penalises loyal venues
        for i in range(n_venues)
    ])
    q0_anti_loyal = np.clip(q0_anti_loyal, 1e-10, None)
    _, q = birank(W, p0=p0, q0=q0_anti_loyal)
    r_anti_loyal = {i2v[i]: float(q[i]) for i in range(n_venues)}
    rho, pval = spearman_rising(r_anti_loyal, rising)
    agg, _, _ = evaluate_per_user(r_anti_loyal, train_uv, test_uv_rev)
    rows.append({"dataset": label, "method": "inverted_loyalty_prior",
                 "rho": rho, "p_value": pval, "ndcg10": agg.get("NDCG@10", 0),
                 "interpretation": "anti-loyalty (penalises high-revisit venues)"})
    print(f"  inverted_loyal_prior: ρ={rho:+.4f} (p={pval:.4f})  NDCG={agg.get('NDCG@10',0):.4f}")

    # 3. Reference: correct exploration prior (α=1.0)
    from run_london_pipeline import build_birank_explore
    r_correct = build_birank_explore(decay_e, user_feat, venue_feat)
    rho, pval = spearman_rising(r_correct, rising)
    agg, _, _ = evaluate_per_user(r_correct, train_uv, test_uv_rev)
    rows.append({"dataset": label, "method": "correct_exploration_prior",
                 "rho": rho, "p_value": pval, "ndcg10": agg.get("NDCG@10", 0),
                 "interpretation": "correct prior (should be BETTER)"})
    print(f"  correct_prior:        ρ={rho:+.4f} (p={pval:.4f})  NDCG={agg.get('NDCG@10',0):.4f}")

    return rows


if __name__ == "__main__":
    all_rows = []
    all_rows += run_inverted_prior(
        DATA_DIR / "london_interactions.csv", "2018-01-01",
        "London TripAdvisor", has_stars=True
    )
    all_rows += run_inverted_prior(
        DATA_DIR / "uk_fsq_interactions.csv", "2013-07-01",
        "UK Foursquare", has_stars=False
    )

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(DATA_DIR / "inverted_prior_validation.csv", index=False)
    print(f"\nSaved → inverted_prior_validation.csv")
    print(df_out[["dataset","method","rho","p_value"]].to_string(index=False))

    # Verdict
    print("\n=== NEGATIVE CONTROL VERDICT ===")
    for _, row in df_out.iterrows():
        if "inverted" in row["method"] and row["rho"] < 0:
            print(f"  ✓ {row['dataset']} {row['method']}: ρ={row['rho']:+.4f} — "
                  f"correctly performs WORSE (negative ρ confirms prior design works)")
        elif "inverted" in row["method"]:
            print(f"  ⚠ {row['dataset']} {row['method']}: ρ={row['rho']:+.4f} — "
                  f"UNEXPECTED positive ρ — investigate")
