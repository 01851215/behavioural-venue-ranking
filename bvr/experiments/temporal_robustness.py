"""
Temporal split robustness for UK datasets.

Runs the hybrid (explore + ALS) vs popularity baseline and random across
multiple split dates. Confirms the winner is stable, not an artefact of
a single lucky cut.

London splits:  2016-07-01, 2017-01-01, 2017-07-01, 2018-01-01, 2018-07-01
UK FSQ splits:  2013-01-01, 2013-04-01, 2013-07-01, 2013-10-01

Saves: temporal_robustness_london.csv, temporal_robustness_uk_fsq.csv
"""

import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from bvr.core.validation import (
    compute_user_features, compute_venue_features,
    build_decayed_edges, build_popularity_ranking, evaluate_per_user,
)
from bvr.pipelines.london import (
    temporal_split, compute_rising_stars, spearman_rising,
    build_birank_explore, build_mf_ranking, blend_rankings,
)

DATA_DIR  = Path(__file__).parent
DECAY_LAM = 0.5

LONDON_SPLITS = [
    "2016-07-01", "2017-01-01", "2017-07-01", "2018-01-01", "2018-07-01"
]
FSQ_SPLITS = [
    "2013-01-01", "2013-04-01", "2013-07-01", "2013-10-01"
]


def run_split(df, split_date, label):
    t0 = time.time()
    train, test_uv_rev, train_uv, _ = temporal_split(df, split_date)

    if len(train) < 1000 or len(test_uv_rev) < 10:
        print(f"  {split_date}  SKIP — insufficient data")
        return None

    user_feat  = compute_user_features(train)
    venue_feat = compute_venue_features(train)
    decay_e    = build_decayed_edges(train, split_date, lam=DECAY_LAM)
    test_full  = df[df["timestamp"] >= pd.Timestamp(split_date)].copy()
    rising     = compute_rising_stars(train, test_full)

    # Methods
    r_explore = build_birank_explore(decay_e, user_feat, venue_feat)
    r_als     = build_mf_ranking(train, "als")
    r_hybrid  = blend_rankings(r_explore, r_als, lam=0.5)
    r_pop     = build_popularity_ranking(train)
    np.random.seed(42 + hash(split_date) % 1000)   # reproducible per split
    r_random  = {v: float(np.random.random()) for v in train["business_id"].unique()}

    row = {"dataset": label, "split_date": split_date,
           "n_train": len(train), "n_revisit_users": len(test_uv_rev)}

    for name, ranking in [("hybrid", r_hybrid), ("popularity", r_pop), ("random", r_random)]:
        rho, pval = spearman_rising(ranking, rising)
        agg, _, _ = evaluate_per_user(ranking, train_uv, test_uv_rev)
        row[f"{name}_rho"]    = round(rho,  4)
        row[f"{name}_p"]      = round(pval, 4)
        row[f"{name}_ndcg10"] = round(agg.get("NDCG@10", 0), 4)

    elapsed = time.time() - t0
    print(f"  {split_date}  hybrid ρ={row['hybrid_rho']:+.4f}  pop ρ={row['popularity_rho']:+.4f}  "
          f"NDCG@10={row['hybrid_ndcg10']:.4f}  ({elapsed:.0f}s)")
    return row


if __name__ == "__main__":

    # London
    print("\n=== London TripAdvisor ===")
    london_df = pd.read_csv(DATA_DIR / "london_interactions.csv",
                            dtype={"user_id": str, "business_id": str})
    london_df["timestamp"] = pd.to_datetime(london_df["timestamp"])
    london_df["stars"]     = london_df["stars"].astype(float)

    london_rows = [run_split(london_df, s, "London TripAdvisor") for s in LONDON_SPLITS]
    london_rows = [r for r in london_rows if r]
    df_london   = pd.DataFrame(london_rows)
    df_london.to_csv(DATA_DIR / "temporal_robustness_london.csv", index=False)
    print(f"\nSaved → temporal_robustness_london.csv")

    # UK FSQ
    print("\n=== UK Foursquare ===")
    fsq_df = pd.read_csv(DATA_DIR / "uk_fsq_interactions.csv",
                         dtype={"user_id": str, "business_id": str})
    fsq_df["timestamp"] = pd.to_datetime(fsq_df["timestamp"])
    fsq_df["stars"]     = np.nan

    fsq_rows = [run_split(fsq_df, s, "UK Foursquare") for s in FSQ_SPLITS]
    fsq_rows = [r for r in fsq_rows if r]
    df_fsq   = pd.DataFrame(fsq_rows)
    df_fsq.to_csv(DATA_DIR / "temporal_robustness_uk_fsq.csv", index=False)
    print(f"Saved → temporal_robustness_uk_fsq.csv")

    print("\n=== Summary ===")
    for df, name in [(df_london, "London"), (df_fsq, "UK FSQ")]:
        print(f"\n{name}:")
        print(df[["split_date", "n_train", "hybrid_rho", "popularity_rho", "hybrid_ndcg10"]].to_string(index=False))
