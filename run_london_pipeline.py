"""
London Restaurant BiRank Pipeline

Applies the full behavioral ranking methodology to TripAdvisor London data.
Temporal split: 2018-01-01 (train=659K reviews, test=337K, overlap=128K users)

Reuses core functions from validate_v5.py — no code duplication.

Outputs:
  london_birank_venue_scores.csv   — ranked London restaurants
  london_user_features.csv         — user behavioral features
  london_venue_features.csv        — venue behavioral features
  london_validation_summary.txt    — NDCG@10 vs baselines
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

# Import core functions from the validated v5 pipeline (no duplication)
sys.path.insert(0, str(Path(__file__).parent))
from validate_v5 import (
    corrected_burstiness,
    compute_user_features,
    compute_venue_features,
    build_count_edges,
    build_decayed_edges,
    behavioral_priors,
    build_birank_ranking,
    build_rating_ranking,
    build_popularity_ranking,
    birank,
    ndcg_at_k,
    evaluate_per_user,
    bootstrap_ci,
    build_adjacency,
)

warnings.filterwarnings("ignore")

DATA_DIR   = Path(__file__).parent
SPLIT_DATE = "2018-01-01"
DECAY_LAM  = 0.5   # same half-life as v5_combined
K_VALUES   = (5, 10, 20)
N_BOOTSTRAP = 1000


# ============================================================================
# Data loading
# ============================================================================

def load_london_data():
    print("Loading London TripAdvisor interactions...")
    df = pd.read_csv(DATA_DIR / "london_interactions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["stars"] = df["stars"].astype(float)
    print(f"  {len(df):,} reviews  |  "
          f"{df['user_id'].nunique():,} users  |  "
          f"{df['business_id'].nunique():,} venues")
    return df


def temporal_split(df: pd.DataFrame, split_date: str):
    split = pd.Timestamp(split_date)
    train = df[df["timestamp"] < split].copy()
    test  = df[df["timestamp"] >= split].copy()

    overlap = set(train["user_id"]) & set(test["user_id"])
    print(f"\nTemporal split: {split_date}")
    print(f"  Train: {len(train):,}  |  Test: {len(test):,}  |  Overlap users: {len(overlap):,}")

    # REVISIT eval (venues seen in both train and test — ~7% of test visits)
    train_uv = train[train["user_id"].isin(overlap)].groupby("user_id")["business_id"].apply(set).to_dict()
    test_uv  = test[test["user_id"].isin(overlap)].groupby("user_id")["business_id"].apply(set).to_dict()
    # test_uv for NDCG = revisit set only (intersection)
    test_uv_revisit = {u: test_uv[u] & train_uv.get(u, set()) for u in test_uv}
    test_uv_revisit = {u: s for u, s in test_uv_revisit.items() if s}
    print(f"  Revisit rate: {sum(len(s) for s in test_uv_revisit.values())}/{len(test):,} "
          f"({sum(len(s) for s in test_uv_revisit.values())/len(test)*100:.1f}%)")
    print(f"  Users with ≥1 revisit: {len(test_uv_revisit):,}")

    # Venue-level test traffic (for Spearman eval — primary metric)
    test_traffic = test["business_id"].value_counts().to_dict()

    return train, test_uv_revisit, train_uv, test_traffic


# ============================================================================
# Evaluation
# ============================================================================

def wilcoxon_p(a, b):
    try:
        _, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return float(p)
    except Exception:
        return 1.0


def spearman_venue(ranking: dict, test_traffic: dict) -> tuple[float, float]:
    """Spearman ρ between predicted rank score and actual test traffic per venue."""
    from scipy.stats import spearmanr
    common = set(ranking) & set(test_traffic)
    if len(common) < 10:
        return np.nan, np.nan
    pred   = [ranking[v]      for v in common]
    actual = [test_traffic[v] for v in common]
    rho, p = spearmanr(pred, actual)
    return float(rho), float(p)


def evaluate_all(rankings: dict, train_uv: dict, test_uv_revisit: dict,
                 test_traffic: dict) -> dict:
    results = {}
    for name, ranking in rankings.items():
        # Venue-level Spearman (primary — appropriate for exploration data)
        rho, pval = spearman_venue(ranking, test_traffic)
        # Per-user NDCG on revisit subset (secondary — low rate but still informative)
        if test_uv_revisit:
            agg, per_user, _ = evaluate_per_user(ranking, train_uv, test_uv_revisit)
        else:
            agg, per_user = {}, {}
        results[name] = {"spearman_rho": rho, "spearman_p": pval,
                         "agg": agg, "per_user": per_user}
    return results


def print_results(results: dict, train: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("LONDON BIRANK VALIDATION RESULTS")
    lines.append(f"Split: {SPLIT_DATE}  |  Train: {len(train):,}  |  "
                 f"Venues: {train['business_id'].nunique():,}")
    lines.append("Note: Primary metric = Spearman ρ (venue traffic prediction).")
    lines.append("      TripAdvisor data has 7% revisit rate → NDCG is supplementary.")
    lines.append("=" * 80)

    # Venue-level Spearman table
    lines.append(f"\n  PRIMARY METRIC — Venue Traffic Spearman ρ (BiRank score vs test reviews/venue)")
    lines.append(f"  {'Method':<28} {'Spearman ρ':>12} {'p-value':>12}")
    lines.append("  " + "-" * 55)
    for name, data in results.items():
        rho  = data["spearman_rho"]
        pval = data["spearman_p"]
        sig  = " ***" if pval < 0.001 else (" *" if pval < 0.05 else "")
        lines.append(f"  {name:<28} {rho:>12.4f} {pval:>12.4f}{sig}")

    # NDCG table (supplementary)
    if any(r["agg"] for r in results.values()):
        lines.append(f"\n  SUPPLEMENTARY — NDCG@10 on revisit subset (7% of test visits)")
        lines.append(f"  {'Method':<28} {'NDCG@10':>10} {'Hit@10':>10}")
        lines.append("  " + "-" * 52)
        for name, data in results.items():
            agg = data["agg"]
            lines.append(f"  {name:<28} {agg.get('NDCG@10',0):>10.4f} {agg.get('Hit@10',0):>10.4f}")

    # Summary
    br = results.get("birank_decay", results.get("birank_count", {}))
    pop = results.get("baseline_popularity", {})
    rat = results.get("baseline_rating", {})
    delta_pop = br.get("spearman_rho", 0) - pop.get("spearman_rho", 0)
    delta_rat = br.get("spearman_rho", 0) - rat.get("spearman_rho", 0)
    lines.append(f"\n  BiRank (decay) vs Popularity: Δρ = {delta_pop:+.4f}")
    lines.append(f"  BiRank (decay) vs Stars:      Δρ = {delta_rat:+.4f}")

    lines.append("=" * 80)
    report = "\n".join(lines)
    print(report)
    return report


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    t0 = time.time()

    # 1. Load
    df = load_london_data()

    # 2. Split
    train, test_uv_revisit, train_uv, test_traffic = temporal_split(df, SPLIT_DATE)

    # 3. Features (computed on training data only — no leakage)
    print("\nComputing user behavioral features...")
    user_feat = compute_user_features(train)
    user_feat.to_csv(DATA_DIR / "london_user_features.csv", index=False)
    print(f"  {len(user_feat):,} users  |  saved → london_user_features.csv")

    print("\nComputing venue behavioral features...")
    venue_feat = compute_venue_features(train)
    venue_feat.to_csv(DATA_DIR / "london_venue_features.csv", index=False)
    print(f"  {len(venue_feat):,} venues  |  saved → london_venue_features.csv")

    # 4. Build graph edges
    print("\nBuilding bipartite graph edges...")
    count_edges  = build_count_edges(train)
    decayed_edges = build_decayed_edges(train, SPLIT_DATE, lam=DECAY_LAM)
    print(f"  Count edges: {len(count_edges):,}  |  Decayed edges: {len(decayed_edges):,}")

    # 5. BiRank variants
    print("\nRunning BiRank variants...")
    rankings = {}

    # BiRank with count edges + behavioral priors
    rankings["birank_count"] = build_birank_ranking(
        count_edges, user_feat, venue_feat
    )
    print("  ✓ birank_count")

    # BiRank with temporal decay + behavioral priors (best variant from v5)
    rankings["birank_decay"] = build_birank_ranking(
        decayed_edges, user_feat, venue_feat
    )
    print("  ✓ birank_decay")

    # Baselines
    rankings["baseline_rating"]     = build_rating_ranking(train)
    rankings["baseline_popularity"] = build_popularity_ranking(train)
    print("  ✓ baselines (rating, popularity)")

    # 6. Evaluate
    print("\nEvaluating all methods...")
    results = evaluate_all(rankings, train_uv, test_uv_revisit, test_traffic)

    # 7. Report
    report = print_results(results, train)
    with open(DATA_DIR / "london_validation_summary.txt", "w") as f:
        f.write(report)
    print("\nSaved → london_validation_summary.txt")

    # 8. Save venue scores (best BiRank variant)
    best_ranking = rankings.get("birank_decay", rankings["birank_count"])
    venue_scores = pd.DataFrame([
        {"business_id": vid, "birank_score": score, "rank": rank + 1}
        for rank, (vid, score) in enumerate(
            sorted(best_ranking.items(), key=lambda x: x[1], reverse=True)
        )
    ])
    # Merge in venue names
    ta_biz = pd.read_csv(DATA_DIR / "london_businesses.csv")
    ta_biz["business_id"] = ta_biz["business_id"].astype(str)
    venue_scores["business_id"] = venue_scores["business_id"].astype(str)
    venue_scores = venue_scores.merge(ta_biz[["business_id", "name"]], on="business_id", how="left")
    venue_scores.to_csv(DATA_DIR / "london_birank_venue_scores.csv", index=False)
    print(f"Saved → london_birank_venue_scores.csv  ({len(venue_scores):,} venues)")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
