"""
London Restaurant BiRank Pipeline (Phase 1: rising stars + exploration priors)

Applies behavioral ranking to TripAdvisor London data with TWO key fixes for
exploration-driven (vs loyalty-driven) data:

  1. Rising-stars evaluation — predicts venue traffic GROWTH (residual after
     controlling for popularity), not absolute popularity. This is a fairer
     test for any non-popularity-based ranker.

  2. Exploration priors — inverts the loyalty-biased priors used for Yelp coffee:
     - User prior:  log1p(unique_venues) * (1 - top1_venue_share)
                    rewards diverse explorers, not single-venue loyalists
     - Venue prior: 1 / log1p(popularity_visits)
                    INVERSE popularity, gives non-popular venues a fair shot
                    so BiRank's mutual reinforcement can find quality signals

Temporal split: 2018-01-01 (train=659K reviews, test=337K, overlap=48K users)

Outputs:
  london_birank_venue_scores.csv   — ranked London restaurants
  london_user_features.csv         — user behavioral features
  london_venue_features.csv        — venue behavioral features
  london_validation_summary.txt    — Spearman vs baselines (rising stars)
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


def compute_rising_stars(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    For each venue, compute traffic-growth residual after popularity baseline.

    Method: predict log(test_rate) from log(train_rate) using linear regression.
    Residual > 0 = venue grew faster than popularity predicts (rising star).
    Residual < 0 = venue lost relative traffic (falling).

    Returns: {business_id: residual} — the signal a NON-popularity model should
    correlate with to prove it adds value beyond popularity.
    """
    train_months = (train["timestamp"].max() - train["timestamp"].min()).days / 30.0
    test_months  = (test["timestamp"].max()  - test["timestamp"].min()).days  / 30.0

    train_rate = (train["business_id"].value_counts() / max(train_months, 1)).to_dict()
    test_rate  = (test["business_id"].value_counts()  / max(test_months,  1)).to_dict()

    common = set(train_rate) & set(test_rate)
    common = [v for v in common if train_rate[v] > 0 and test_rate[v] > 0]
    if len(common) < 30:
        return {}

    log_train = np.log1p([train_rate[v] for v in common])
    log_test  = np.log1p([test_rate[v]  for v in common])

    # Simple OLS: log(test) = a + b * log(train) + residual
    slope, intercept = np.polyfit(log_train, log_test, 1)
    pred = intercept + slope * log_train
    residuals = log_test - pred

    return dict(zip(common, residuals))


def spearman_rising(ranking: dict, residuals: dict) -> tuple[float, float]:
    """Spearman ρ between model score and rising-stars residual."""
    from scipy.stats import spearmanr
    common = set(ranking) & set(residuals)
    if len(common) < 10:
        return np.nan, np.nan
    pred   = [ranking[v]   for v in common]
    actual = [residuals[v] for v in common]
    rho, p = spearmanr(pred, actual)
    return float(rho), float(p)


def exploration_priors(user_feat: pd.DataFrame, venue_feat: pd.DataFrame,
                       u2i: dict, v2i: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Exploration-style priors (vs Loyalist-style in validate_v5).

    User prior:  log1p(unique_venues) * (1 - top1_venue_share)
                 → high for diverse explorers, ~0 for single-venue users
    Venue prior: 1 / log1p(popularity_visits)
                 → INVERSE popularity, levels playing field for less-popular venues

    The inverse-popularity venue prior is the key insight: without it, BiRank's
    mutual reinforcement just amplifies popularity. With it, a less-popular
    venue favoured by trustworthy explorers gets propagated upward.
    """
    nu, nv = len(u2i), len(v2i)

    uv_map  = user_feat.set_index("user_id")["unique_venues"]
    top1_map = user_feat.set_index("user_id")["top1_venue_share"]
    pop_map  = venue_feat.set_index("business_id")["popularity_visits"]

    p0 = np.ones(nu)
    for uid, idx in u2i.items():
        uv   = float(uv_map.get(uid, 1) or 1)
        t1   = float(top1_map.get(uid, 1) or 1)
        p0[idx] = np.log1p(uv) * (1 - t1)

    q0 = np.ones(nv)
    for vid, idx in v2i.items():
        pop = float(pop_map.get(vid, 1) or 1)
        q0[idx] = 1.0 / np.log1p(pop)

    return np.clip(p0, 1e-10, None), np.clip(q0, 1e-10, None)


def build_birank_explore(edges_df: pd.DataFrame, user_feat: pd.DataFrame,
                          venue_feat: pd.DataFrame) -> dict:
    """BiRank with exploration priors instead of Loyalist priors."""
    W, u2i, v2i, i2u, i2v = build_adjacency(edges_df)
    p0, q0 = exploration_priors(user_feat, venue_feat, u2i, v2i)
    _, q = birank(W, p0=p0, q0=q0)
    return {i2v[i]: float(q[i]) for i in range(len(i2v))}


def evaluate_all(rankings: dict, train_uv: dict, test_uv_revisit: dict,
                 test_traffic: dict, rising_stars: dict) -> dict:
    results = {}
    for name, ranking in rankings.items():
        # PRIMARY: rising-stars residual correlation (popularity-debiased)
        rho_rs, p_rs = spearman_rising(ranking, rising_stars)
        # Secondary: raw test traffic correlation (popularity-confounded)
        rho_tf, p_tf = spearman_venue(ranking, test_traffic)
        # Tertiary: per-user NDCG on revisit subset
        if test_uv_revisit:
            agg, per_user, _ = evaluate_per_user(ranking, train_uv, test_uv_revisit)
        else:
            agg, per_user = {}, {}
        results[name] = {
            "rising_rho":  rho_rs, "rising_p":  p_rs,
            "traffic_rho": rho_tf, "traffic_p": p_tf,
            "agg": agg, "per_user": per_user,
        }
    return results


def print_results(results: dict, train: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append("LONDON BIRANK VALIDATION RESULTS — Phase 1 (rising stars + exploration priors)")
    lines.append(f"Split: {SPLIT_DATE}  |  Train: {len(train):,}  |  "
                 f"Venues: {train['business_id'].nunique():,}")
    lines.append("=" * 90)

    # PRIMARY: Rising stars (popularity-debiased)
    lines.append(f"\n  PRIMARY — Rising Stars residual correlation (popularity-debiased)")
    lines.append(f"  Tests if a method identifies venues that grew BEYOND popularity baseline.")
    lines.append(f"  {'Method':<28} {'ρ (rising)':>12} {'p-value':>12}")
    lines.append("  " + "-" * 55)
    for name, data in results.items():
        rho  = data["rising_rho"]
        pval = data["rising_p"]
        sig  = " ***" if pval < 0.001 else (" *" if pval < 0.05 else "")
        lines.append(f"  {name:<28} {rho:>+12.4f} {pval:>12.4f}{sig}")

    # SECONDARY: Raw traffic (popularity-confounded)
    lines.append(f"\n  SECONDARY — Raw test-traffic Spearman ρ (popularity dominates)")
    lines.append(f"  {'Method':<28} {'ρ (traffic)':>12} {'p-value':>12}")
    lines.append("  " + "-" * 55)
    for name, data in results.items():
        rho  = data["traffic_rho"]
        pval = data["traffic_p"]
        sig  = " ***" if pval < 0.001 else (" *" if pval < 0.05 else "")
        lines.append(f"  {name:<28} {rho:>+12.4f} {pval:>12.4f}{sig}")

    # Tertiary: NDCG (very thin signal here)
    if any(r["agg"] for r in results.values()):
        lines.append(f"\n  TERTIARY — NDCG@10 on revisit subset (~2.6% of test visits)")
        lines.append(f"  {'Method':<28} {'NDCG@10':>10} {'Hit@10':>10}")
        lines.append("  " + "-" * 52)
        for name, data in results.items():
            agg = data["agg"]
            lines.append(f"  {name:<28} {agg.get('NDCG@10',0):>10.4f} {agg.get('Hit@10',0):>10.4f}")

    # Headline comparison
    pop = results.get("baseline_popularity", {})
    rat = results.get("baseline_rating", {})
    explore = results.get("birank_explore", {})
    decay   = results.get("birank_decay",   {})

    lines.append(f"\n  HEADLINE — Rising-stars ρ (does the method add value beyond popularity?)")
    lines.append(f"    Popularity baseline:    {pop.get('rising_rho', 0):+.4f}  (defines 0 by construction)")
    lines.append(f"    Star ratings:           {rat.get('rising_rho', 0):+.4f}")
    lines.append(f"    BiRank (loyalty priors): {decay.get('rising_rho', 0):+.4f}")
    lines.append(f"    BiRank (explore priors): {explore.get('rising_rho', 0):+.4f}  ← Phase 1 fix")
    lines.append("=" * 90)

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

    # BiRank with count edges + Loyalist priors (validate_v5 default)
    rankings["birank_count"] = build_birank_ranking(count_edges, user_feat, venue_feat)
    print("  ✓ birank_count       (count edges, loyalty priors)")

    # BiRank with temporal decay + Loyalist priors (best v5 variant)
    rankings["birank_decay"] = build_birank_ranking(decayed_edges, user_feat, venue_feat)
    print("  ✓ birank_decay       (decayed edges, loyalty priors)")

    # NEW: BiRank with EXPLORATION priors (Phase 1 fix)
    rankings["birank_explore"] = build_birank_explore(decayed_edges, user_feat, venue_feat)
    print("  ✓ birank_explore     (decayed edges, exploration priors — INVERSE popularity)")

    # Baselines
    rankings["baseline_rating"]     = build_rating_ranking(train)
    rankings["baseline_popularity"] = build_popularity_ranking(train)
    print("  ✓ baselines          (rating, popularity)")

    # 6. Compute rising-stars residuals (PRIMARY metric for exploration data)
    print("\nComputing rising-stars residuals...")
    test = df[df["timestamp"] >= pd.Timestamp(SPLIT_DATE)].copy()
    rising_stars = compute_rising_stars(train, test)
    print(f"  Rising-stars signal computed for {len(rising_stars):,} venues")

    # 7. Evaluate
    print("\nEvaluating all methods...")
    results = evaluate_all(rankings, train_uv, test_uv_revisit, test_traffic, rising_stars)

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
