"""
Compute Cohen's d and rank-biserial correlation for all Wilcoxon comparisons.

Reads: benchmark_results.json
Appends: cohen_d and rank_biserial columns
Saves: effect_sizes.csv

Cohen's d (paired):
    d = mean(diff) / std(diff)

Rank-biserial correlation (for Wilcoxon):
    r = 1 - 2W / (n*(n+1)/2)   where W = Wilcoxon statistic
Simpler equivalent: r = Z / sqrt(n), where Z is the z-score of the statistic.
We compute it as the effect size from the per-user NDCG@10 arrays.
"""

import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from validate_v5 import evaluate_per_user
from run_london_pipeline import temporal_split
from lightgcn import build_lightgcn_ranking
from run_london_pipeline import (
    build_birank_explore, build_mf_ranking, blend_rankings
)
from validate_v5 import (
    compute_user_features, compute_venue_features,
    build_count_edges, build_decayed_edges,
    build_birank_ranking, build_popularity_ranking, build_rating_ranking
)

DATA_DIR = Path(__file__).parent


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.mean() / (diff.std() + 1e-10))


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial r from Wilcoxon signed-rank test."""
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(diff))) + 1
    W_plus  = ranks[diff > 0].sum()
    W_minus = ranks[diff < 0].sum()
    W = min(W_plus, W_minus)
    n_nonzero = len(diff)
    max_W = n_nonzero * (n_nonzero + 1) / 2
    return float(1 - 2 * W / max_W)


def compute_for_dataset(interactions_file, split_date, label, has_stars=True):
    print(f"\n=== {label} ===")
    df = pd.read_csv(interactions_file, dtype={"user_id": str, "business_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not has_stars:
        df["stars"] = float("nan")

    train, test_uv_rev, train_uv, _ = temporal_split(df, split_date)
    user_feat  = compute_user_features(train)
    venue_feat = compute_venue_features(train)
    decay_e    = build_decayed_edges(train, split_date, lam=0.5)

    r_explore = build_birank_explore(decay_e, user_feat, venue_feat)
    r_als     = build_mf_ranking(train, "als")
    r_hybrid  = blend_rankings(r_explore, r_als, lam=0.5)
    r_pop     = build_popularity_ranking(train)

    rankings = {"hybrid": r_hybrid, "popularity": r_pop}
    if has_stars:
        rankings["rating"] = build_rating_ranking(train)
    rankings["birank_count"] = build_birank_ranking(
        build_count_edges(train), user_feat, venue_feat
    )

    # Per-user NDCG arrays
    pu_arrays = {}
    for name, ranking in rankings.items():
        _, pu, _ = evaluate_per_user(ranking, train_uv, test_uv_rev)
        pu_arrays[name] = pu["NDCG@10"]

    # Effect sizes vs hybrid (winner)
    winner = pu_arrays["hybrid"]
    rows = []
    print(f"{'Method':<25} {'Cohen d':>10} {'Rank-biserial r':>17} {'Magnitude':>12}")
    print("-" * 68)
    for name, arr in pu_arrays.items():
        if name == "hybrid":
            continue
        d  = cohens_d_paired(winner, arr)
        rb = rank_biserial(winner, arr)
        mag = "large" if abs(d) >= 0.8 else ("medium" if abs(d) >= 0.5 else ("small" if abs(d) >= 0.2 else "negligible"))
        print(f"{name:<25} {d:>+10.4f} {rb:>+17.4f} {mag:>12}")
        rows.append({
            "dataset": label, "method": name,
            "cohens_d": round(d, 4), "rank_biserial_r": round(rb, 4),
            "magnitude": mag,
        })
    return rows


if __name__ == "__main__":
    all_rows = []
    all_rows += compute_for_dataset(
        DATA_DIR / "london_interactions.csv", "2018-01-01",
        "London TripAdvisor", has_stars=True
    )
    all_rows += compute_for_dataset(
        DATA_DIR / "uk_fsq_interactions.csv", "2013-07-01",
        "UK Foursquare", has_stars=False
    )

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(DATA_DIR / "effect_sizes.csv", index=False)
    print(f"\nSaved → effect_sizes.csv")
    print(df_out.to_string(index=False))
