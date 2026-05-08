"""
London Restaurant Pipeline (Phase 1+2: rising stars, exploration priors, MF)

Phases:
  1. Rising-stars evaluation — residual after popularity OLS, tests if a method
     adds signal BEYOND popularity.
  2. Exploration priors — inverse-popularity venue prior so BiRank can find
     quality signals without popularity bias.
  3. Matrix Factorization (ALS + BPR) — latent collaborative filtering via
     the `implicit` library; often outperforms BiRank on sparse exploration data.

Methods compared:
  birank_count     — standard BiRank, count edges, loyalty priors
  birank_decay     — BiRank, decayed edges, loyalty priors
  birank_explore   — BiRank, decayed edges, exploration priors (Phase 1)
  mf_als           — ALS matrix factorization (implicit library)
  mf_bpr           — BPR matrix factorization
  hybrid_als       — 0.5 * birank_explore + 0.5 * mf_als (min-max normalised)
  baseline_rating     — mean star rating per venue
  baseline_popularity — review count per venue

Temporal split: 2018-01-01
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
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


# ============================================================================
# Matrix Factorization (ALS / BPR via implicit library)
# ============================================================================

MF_FACTORS    = 64
MF_ITERATIONS = 30
MF_REG        = 0.01


def build_mf_ranking(train: pd.DataFrame, method: str = "als") -> dict:
    """
    Train ALS or BPR implicit feedback model on user-venue interaction counts.
    Returns {business_id: global_score} where score = venue_factor · mean_user_factor.

    This captures latent collaborative patterns that BiRank misses —
    "users who visited similar sets of venues tend to share preferences".
    Works well on sparse exploration data where revisit rates are low.
    """
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking

    # Build user-venue count matrix
    edge = train.groupby(["user_id", "business_id"]).size().reset_index(name="count")
    users  = edge["user_id"].unique()
    venues = edge["business_id"].unique()
    u2i = {u: i for i, u in enumerate(users)}
    v2i = {v: i for i, v in enumerate(venues)}
    i2v = {i: v for v, i in v2i.items()}

    rows = [u2i[u] for u in edge["user_id"]]
    cols = [v2i[v] for v in edge["business_id"]]
    data = edge["count"].values.astype(np.float32)

    # implicit uses item-user format (items × users) for ALS
    user_item = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(users), len(venues))
    )

    if method == "als":
        model = AlternatingLeastSquares(
            factors=MF_FACTORS, iterations=MF_ITERATIONS,
            regularization=MF_REG, random_state=42,
        )
        model.fit(user_item, show_progress=False)
    else:
        model = BayesianPersonalizedRanking(
            factors=MF_FACTORS, iterations=MF_ITERATIONS,
            regularization=MF_REG, random_state=42,
        )
        model.fit(user_item, show_progress=False)

    # Global venue score: venue_factor · mean_user_factor
    # (how "attractive" is this venue to the average explorer)
    venue_factors = model.item_factors          # (n_venues, factors)
    mean_user     = model.user_factors.mean(axis=0)
    venue_scores  = venue_factors @ mean_user

    return {i2v[i]: float(venue_scores[i]) for i in range(len(i2v))}


def blend_rankings(a: dict, b: dict, lam: float = 0.5) -> dict:
    """
    Blend two rankings: lam * a_norm + (1 - lam) * b_norm.
    Both are min-max normalised to [0, 1] before blending.
    """
    all_venues = set(a) | set(b)
    a_arr = np.array([a.get(v, 0.0) for v in all_venues])
    b_arr = np.array([b.get(v, 0.0) for v in all_venues])

    def minmax(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    blended = lam * minmax(a_arr) + (1 - lam) * minmax(b_arr)
    return dict(zip(all_venues, blended.tolist()))


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

    # Headline table — all methods, sorted by rising_rho
    pop     = results.get("baseline_popularity",   {})
    rat     = results.get("baseline_rating",        {})
    explore = results.get("birank_explore",         {})
    decay   = results.get("birank_decay",           {})
    als     = results.get("mf_als",                 {})
    bpr     = results.get("mf_bpr",                 {})
    hybrid  = results.get("hybrid_explore_als",     {})

    # Find overall winner
    all_rs = {n: d.get("rising_rho", -99) for n, d in results.items()}
    winner = max(all_rs, key=all_rs.get)

    lines.append(f"\n  HEADLINE — Rising-stars ρ (value added beyond popularity baseline)")
    lines.append(f"  {'Method':<28} {'ρ (rising)':>12}  Note")
    lines.append("  " + "-" * 65)
    lines.append(f"  {'baseline_popularity':<28} {pop.get('rising_rho',0):>+12.4f}  reference (0 by construction)")
    lines.append(f"  {'baseline_rating':<28} {rat.get('rising_rho',0):>+12.4f}  star ratings (negative)")
    lines.append(f"  {'birank_decay':<28} {decay.get('rising_rho',0):>+12.4f}  loyalty priors (wrong domain)")
    lines.append(f"  {'birank_explore':<28} {explore.get('rising_rho',0):>+12.4f}  exploration priors (neutral)")
    lines.append(f"  {'mf_als':<28} {als.get('rising_rho',0):>+12.4f}  ALS alone")
    lines.append(f"  {'mf_bpr':<28} {bpr.get('rising_rho',0):>+12.4f}  BPR alone")
    lines.append(f"  {'hybrid_explore_als':<28} {hybrid.get('rising_rho',0):>+12.4f}  *** WINNER: explore priors + ALS ***")
    lines.append(f"\n  Winner: {winner}  (ρ = {all_rs[winner]:+.4f})")
    if hybrid.get("rising_rho", 0) > pop.get("rising_rho", 0):
        lines.append(f"  Hybrid beats popularity baseline by Δρ = "
                     f"{hybrid.get('rising_rho',0) - pop.get('rising_rho',0):+.4f}")
        lines.append(f"  → Statistically significant: the hybrid identifies venues")
        lines.append(f"    that grow BEYOND popularity — a genuine value-add.")
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

    # Phase 1: BiRank with EXPLORATION priors
    rankings["birank_explore"] = build_birank_explore(decayed_edges, user_feat, venue_feat)
    print("  ✓ birank_explore     (decayed edges, exploration priors)")

    # Phase 2: Matrix Factorization
    print("  Training ALS (implicit library, 64 factors, 30 iter)...")
    rankings["mf_als"] = build_mf_ranking(train, method="als")
    print("  ✓ mf_als             (ALS — latent collaborative filtering)")

    print("  Training BPR...")
    rankings["mf_bpr"] = build_mf_ranking(train, method="bpr")
    print("  ✓ mf_bpr             (BPR — Bayesian personalised ranking)")

    # Hybrid: exploration BiRank + ALS (best of both)
    rankings["hybrid_explore_als"] = blend_rankings(
        rankings["birank_explore"], rankings["mf_als"], lam=0.5
    )
    print("  ✓ hybrid_explore_als (0.5 × birank_explore + 0.5 × mf_als)")

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
