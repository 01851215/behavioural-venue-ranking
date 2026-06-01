"""
V3: London-only FSQ filter — cross-source within-city replication

Filters uk_fsq_interactions.csv to London bounding box
(lat 51.3–51.7, lon –0.5 to 0.3) and reruns the full pipeline.

If hybrid wins on London-only FSQ, the cross-source replication is
within the *same city* — TripAdvisor London vs FSQ London —
removing geography as a confound.

Saves: london_fsq_validation_summary.txt, london_fsq_venue_scores.csv
"""

import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from validate_v5 import (
    compute_user_features, compute_venue_features,
    build_decayed_edges, build_popularity_ranking, evaluate_per_user,
)
from run_london_pipeline import (
    temporal_split, compute_rising_stars, spearman_rising,
    build_birank_explore, build_mf_ranking, blend_rankings,
    spearman_venue,
)

DATA_DIR  = Path(__file__).parent
SPLIT_DATE = "2013-07-01"
DECAY_LAM  = 0.5

# London bounding box
LAT_MIN, LAT_MAX =  51.3,  51.7
LON_MIN, LON_MAX = -0.5,   0.3


def bootstrap_ci(arr, n=1000):
    np.random.seed(0)
    means = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


if __name__ == "__main__":
    t0 = time.time()

    # Load venue metadata for bbox filtering
    biz = pd.read_csv(DATA_DIR / "uk_fsq_businesses.csv",
                      dtype={"business_id": str})
    london_ids = set(biz[
        (biz["lat"] >= LAT_MIN) & (biz["lat"] <= LAT_MAX) &
        (biz["lon"] >= LON_MIN) & (biz["lon"] <= LON_MAX)
    ]["business_id"])
    print(f"London-bbox venues: {len(london_ids):,}")

    # Filter interactions to London venues
    df = pd.read_csv(DATA_DIR / "uk_fsq_interactions.csv",
                     dtype={"user_id": str, "business_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["stars"]     = np.nan
    df = df[df["business_id"].isin(london_ids)].reset_index(drop=True)
    print(f"London-only interactions: {len(df):,}  "
          f"users={df['user_id'].nunique():,}  venues={df['business_id'].nunique():,}")

    if len(df) < 500:
        print("⚠ Too few interactions after London filter — stopping")
        raise SystemExit(1)

    train, test_uv_rev, train_uv, test_traffic = temporal_split(df, SPLIT_DATE)
    print(f"Train: {len(train):,}  Revisit users: {len(test_uv_rev):,}")

    if len(test_uv_rev) < 20:
        print("⚠ Too few revisit users — stopping")
        raise SystemExit(1)

    user_feat  = compute_user_features(train)
    venue_feat = compute_venue_features(train)
    decay_e    = build_decayed_edges(train, SPLIT_DATE, lam=DECAY_LAM)

    test_full   = df[df["timestamp"] >= pd.Timestamp(SPLIT_DATE)].copy()
    rising_stars = compute_rising_stars(train, test_full)

    r_explore = build_birank_explore(decay_e, user_feat, venue_feat)
    r_als     = build_mf_ranking(train, "als")
    r_hybrid  = blend_rankings(r_explore, r_als, lam=0.5)
    r_pop     = build_popularity_ranking(train)
    np.random.seed(42)
    r_random  = {v: float(np.random.random()) for v in train["business_id"].unique()}

    methods = {"hybrid_explore_als": r_hybrid,
               "baseline_popularity": r_pop,
               "baseline_random": r_random,
               "birank_explore": r_explore}

    print(f"\n{'Method':<28} {'ρ(rising)':>12} {'p-value':>10} {'NDCG@10':>10} {'95% CI':>22}")
    print("-" * 88)
    rows = []
    for name, ranking in methods.items():
        rho, pval = spearman_rising(ranking, rising_stars)
        agg, pu, _ = evaluate_per_user(ranking, train_uv, test_uv_rev)
        ndcg = agg.get("NDCG@10", 0)
        lo, hi = bootstrap_ci(pu["NDCG@10"]) if len(pu.get("NDCG@10", [])) > 10 else (0, 0)
        sig = "***" if pval < 0.001 else ("*" if pval < 0.05 else "ns")
        ci_str = f"[{lo:.4f}, {hi:.4f}]"
        print(f"{name:<28} {rho:>+12.4f} {pval:>10.4f} {sig:>3} {ndcg:>10.4f} {ci_str:>22}")
        rows.append({"method": name, "rho": rho, "p_value": pval,
                     "ndcg10": ndcg, "ci_lo": lo, "ci_hi": hi})

    winner_rho = max(rows, key=lambda r: r["rho"])
    print(f"\nWinner (ρ): {winner_rho['method']}  (ρ={winner_rho['rho']:+.4f})")
    print(f"Runtime: {time.time()-t0:.0f}s")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(DATA_DIR / "london_fsq_validation.csv", index=False)
    print(f"Saved → london_fsq_validation.csv")
