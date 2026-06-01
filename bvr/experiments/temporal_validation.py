"""
Step 6: Temporal Validation — Behavior Prediction

Split data chronologically into training (past) and testing (future) windows.
Rank venues using training data, then evaluate whether those rankings
predict actual future visits.

Metrics:
  - NDCG@k   – do top-ranked venues match where users actually went?
  - Hit@k    – fraction of users with ≥1 future venue in top-k
  - Spearman ρ – rank correlation between predicted and actual visit counts

Baselines compared:
  - BiRank (behavior-based, with priors)
  - Rating-based (mean star rating)
  - Popularity-based (total visit count)
  - Random
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).parent

# ---- Cut-off for temporal split ----
SPLIT_DATE = pd.Timestamp("2020-01-01")


# ============================================================================
# NDCG
# ============================================================================

def dcg_at_k(relevances, k):
    """Discounted Cumulative Gain at k."""
    relevances = np.asarray(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(predicted_ranking, actual_venues, k):
    """
    NDCG@k for a single user.

    predicted_ranking : list of business_ids in predicted rank order
    actual_venues     : set of business_ids the user actually visited
    """
    # Binary relevance: 1 if the predicted venue was actually visited
    relevances = [1.0 if v in actual_venues else 0.0
                  for v in predicted_ranking[:k]]
    dcg = dcg_at_k(relevances, k)

    # Ideal: all actual venues at the top
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_at_k(predicted_ranking, actual_venues, k):
    """1 if any of the top-k predicted venues was actually visited."""
    return 1.0 if len(set(predicted_ranking[:k]) & actual_venues) > 0 else 0.0


# ============================================================================
# BUILD RANKINGS FROM TRAINING DATA
# ============================================================================

def build_birank_ranking(train_interactions, birank_fn, build_adj_fn,
                         user_feat, venue_feat):
    """
    Re-run BiRank on training-window data only, with behavioral priors.
    Returns {business_id: score} dict.
    """
    # Build edge list from training interactions
    edge_df = (
        train_interactions
        .groupby(["user_id", "business_id"])
        .size()
        .reset_index(name="weight")
    )

    W, u2i, v2i, i2u, i2v = build_adj_fn(edge_df)
    nu, nv = W.shape

    # User priors
    p0 = np.ones(nu)
    bust_map = user_feat.set_index("user_id")["burstiness_index"]
    visits_map = user_feat.set_index("user_id")["total_visits"]
    for uid, idx in u2i.items():
        b = bust_map.get(uid, 0)
        v = visits_map.get(uid, 1)
        b = b if not np.isnan(b) else 0
        p0[idx] = np.log1p(v) * (1 - b)
    p0 = np.clip(p0, 1e-10, None)

    # Venue priors
    q0 = np.ones(nv)
    rr_map = venue_feat.set_index("business_id")["repeat_user_rate"]
    cv_map = venue_feat.set_index("business_id")["stability_cv"]
    for vid, idx in v2i.items():
        rr = rr_map.get(vid, 0)
        cv = cv_map.get(vid, 1)
        rr = rr if not np.isnan(rr) else 0
        cv = cv if not np.isnan(cv) else 1
        q0[idx] = rr * (1 / (1 + cv))
    q0 = np.clip(q0, 1e-10, None)

    _, q, _ = birank_fn(W, p0=p0, q0=q0)

    return {i2v[i]: q[i] for i in range(nv)}


def build_rating_ranking(train_interactions):
    """Mean star rating from training window reviews."""
    if "stars" in train_interactions.columns:
        rated = train_interactions.dropna(subset=["stars"])
        return rated.groupby("business_id")["stars"].mean().to_dict()
    # Fallback: use existing baselines
    try:
        baselines = pd.read_csv(DATA_DIR / "coffee_baselines.csv")
        return baselines.set_index("business_id")["rating_mean"].to_dict()
    except FileNotFoundError:
        return {}


def build_popularity_ranking(train_interactions):
    """Total visit count in training window."""
    return train_interactions.groupby("business_id").size().to_dict()


# ============================================================================
# EVALUATE — TWO PARADIGMS
# ============================================================================

def evaluate_global_ranking(ranking_dict, test_user_venues, k_values=(5, 10, 20)):
    """
    PARADIGM 1: Global ranking evaluation.
    Same venue list for all users.  Good for measuring overall quality ranking.
    """
    sorted_venues = sorted(ranking_dict, key=ranking_dict.get, reverse=True)

    results = {}
    for k in k_values:
        ndcg_scores = []
        hit_scores = []
        for uid, actual in test_user_venues.items():
            ndcg_scores.append(ndcg_at_k(sorted_venues, actual, k))
            hit_scores.append(hit_at_k(sorted_venues, actual, k))

        results[f"NDCG@{k}"] = np.mean(ndcg_scores)
        results[f"Hit@{k}"] = np.mean(hit_scores)

    return results


def evaluate_per_user_reranking(ranking_dict, train_user_venues,
                                test_user_venues, k_values=(5, 10, 20)):
    """
    PARADIGM 2: Per-user candidate re-ranking.

    For each user, take only venues they visited in training as candidates,
    rank them by the method's score, then evaluate against which ones
    they revisited in the test window.

    This is the standard evaluation for bipartite quality ranking:
    it answers "among venues a user already knows, does our score
    correctly predict which ones they'll return to?"
    """
    results = {}
    for k in k_values:
        ndcg_scores = []
        hit_scores = []
        spearman_rhos = []

        for uid in test_user_venues:
            if uid not in train_user_venues:
                continue

            candidates = train_user_venues[uid]
            actual_future = test_user_venues[uid]

            # Rank candidates by method score
            scored = [(v, ranking_dict.get(v, 0)) for v in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked_candidates = [v for v, _ in scored]

            if len(ranked_candidates) < 2:
                continue

            ndcg_scores.append(ndcg_at_k(ranked_candidates, actual_future, k))
            hit_scores.append(hit_at_k(ranked_candidates, actual_future, k))

        results[f"NDCG@{k}"] = np.mean(ndcg_scores) if ndcg_scores else 0
        results[f"Hit@{k}"] = np.mean(hit_scores) if hit_scores else 0

    return results


def evaluate_venue_level(ranking_dict, test_user_venues):
    """
    Venue-level aggregate evaluation.

    Correlate each venue's ranking score with its total future visit count.
    This directly answers: "do higher-ranked venues get more future traffic?"
    """
    test_venue_counts = {}
    for uid, venues in test_user_venues.items():
        for v in venues:
            test_venue_counts[v] = test_venue_counts.get(v, 0) + 1

    common = set(ranking_dict.keys()) & set(test_venue_counts.keys())
    results = {}
    if len(common) > 10:
        pred = [ranking_dict[v] for v in common]
        actual = [test_venue_counts[v] for v in common]
        rho, p = spearmanr(pred, actual)
        results["Spearman_rho"] = rho
        results["Spearman_p"] = p

        # Top-k precision: what fraction of top-k ranked venues
        # are actually in the top-k by future visits?
        sorted_pred = sorted(common, key=lambda v: ranking_dict[v], reverse=True)
        sorted_actual = sorted(common, key=lambda v: test_venue_counts[v], reverse=True)
        for k in (10, 20, 50):
            pk = set(sorted_pred[:k])
            ak = set(sorted_actual[:k])
            results[f"TopK_Precision@{k}"] = len(pk & ak) / k if k <= len(common) else np.nan
    else:
        results["Spearman_rho"] = np.nan
        results["Spearman_p"] = np.nan

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_validation(interactions, venue_scores_full, user_feat, venue_feat,
                   edges, birank_fn, build_adj_fn):
    """Called from run_pipeline.py."""
    print(f"\n  Split date: {SPLIT_DATE.date()}")

    # ---- Chronological split ----
    train = interactions[interactions["timestamp"] < SPLIT_DATE].copy()
    test = interactions[interactions["timestamp"] >= SPLIT_DATE].copy()
    print(f"  Training:   {len(train):>10,}  interactions  "
          f"({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"  Testing:    {len(test):>10,}  interactions  "
          f"({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")

    # Users active in BOTH windows
    train_users = set(train["user_id"].dropna().unique())
    test_users = set(test["user_id"].dropna().unique())
    overlap_users = train_users & test_users
    print(f"  Users in both windows: {len(overlap_users):,}")

    if len(overlap_users) < 50:
        print("  ⚠ Too few overlapping users for reliable validation.")
        return None

    # ---- Build ground truth ----
    test_user_venues = {}
    train_user_venues = {}
    for uid in overlap_users:
        tv = set(test[test["user_id"] == uid]["business_id"].unique())
        trv = set(train[train["user_id"] == uid]["business_id"].unique())
        if len(tv) > 0:
            test_user_venues[uid] = tv
        if len(trv) > 0:
            train_user_venues[uid] = trv

    print(f"  Users with future visits: {len(test_user_venues):,}")

    # ---- Build rankings from training data ----
    print("\n  Building rankings from training data...")

    print("    → BiRank (behavioral priors)...")
    birank_ranking = build_birank_ranking(
        train, birank_fn, build_adj_fn, user_feat, venue_feat
    )

    print("    → Rating baseline...")
    rating_ranking = build_rating_ranking(train)

    print("    → Popularity baseline...")
    pop_ranking = build_popularity_ranking(train)

    # Random baseline
    all_venues = list(set(train["business_id"].unique()))
    rng = np.random.RandomState(42)
    random_ranking = {v: rng.random() for v in all_venues}

    methods = {
        "BiRank (Behavioral)": birank_ranking,
        "Rating (Stars)": rating_ranking,
        "Popularity (Visits)": pop_ranking,
        "Random": random_ranking,
    }

    k_values = (5, 10, 20)

    # ================================================================
    # EVALUATION A: Per-User Candidate Re-Ranking
    # ================================================================
    print("\n  " + "─" * 66)
    print("  EVALUATION A: PER-USER CANDIDATE RE-RANKING")
    print("  " + "─" * 66)
    print("  (Among venues user visited before, predict which they'll revisit)")

    rerank_results = []
    for name, ranking in methods.items():
        res = evaluate_per_user_reranking(
            ranking, train_user_venues, test_user_venues, k_values
        )
        res["Method"] = name
        rerank_results.append(res)
        print(f"\n  {name}:")
        for k in k_values:
            print(f"    NDCG@{k:>2d} = {res[f'NDCG@{k}']:.4f}   "
                  f"Hit@{k:>2d} = {res[f'Hit@{k}']:.4f}")

    rerank_df = pd.DataFrame(rerank_results)
    rerank_df = rerank_df[["Method"] + [c for c in rerank_df.columns if c != "Method"]]

    # ================================================================
    # EVALUATION B: Venue-Level Aggregate
    # ================================================================
    print("\n  " + "─" * 66)
    print("  EVALUATION B: VENUE-LEVEL RANKING QUALITY")
    print("  " + "─" * 66)
    print("  (Do higher-ranked venues get more future traffic?)")

    venue_results = []
    for name, ranking in methods.items():
        res = evaluate_venue_level(ranking, test_user_venues)
        res["Method"] = name
        venue_results.append(res)
        rho = res.get("Spearman_rho", np.nan)
        p = res.get("Spearman_p", np.nan)
        if not np.isnan(rho):
            print(f"\n  {name}:")
            print(f"    Spearman ρ = {rho:.4f}  (p={p:.2e})")
            for kk in (10, 20, 50):
                pk = res.get(f"TopK_Precision@{kk}", np.nan)
                if not np.isnan(pk):
                    print(f"    Top-{kk:>2d} Precision = {pk:.1%}")

    venue_df = pd.DataFrame(venue_results)
    venue_df = venue_df[["Method"] + [c for c in venue_df.columns if c != "Method"]]

    # ================================================================
    # EVALUATION C: Global Ranking (original approach, for reference)
    # ================================================================
    print("\n  " + "─" * 66)
    print("  EVALUATION C: GLOBAL RANKING (reference)")
    print("  " + "─" * 66)

    global_results = []
    for name, ranking in methods.items():
        res = evaluate_global_ranking(ranking, test_user_venues, k_values)
        res["Method"] = name
        global_results.append(res)

    global_df = pd.DataFrame(global_results)
    global_df = global_df[["Method"] + [c for c in global_df.columns if c != "Method"]]

    for _, row in global_df.iterrows():
        print(f"\n  {row['Method']}:")
        for k in k_values:
            print(f"    NDCG@{k:>2d} = {row[f'NDCG@{k}']:.4f}   "
                  f"Hit@{k:>2d} = {row[f'Hit@{k}']:.4f}")

    print("  " + "─" * 66)

    # ---- Check: BiRank should beat Random on re-ranking ----
    br_ndcg = rerank_df[rerank_df["Method"] == "BiRank (Behavioral)"]["NDCG@10"].iloc[0]
    rn_ndcg = rerank_df[rerank_df["Method"] == "Random"]["NDCG@10"].iloc[0]
    if br_ndcg > rn_ndcg:
        print(f"\n  ✓ BiRank re-ranking NDCG@10 ({br_ndcg:.4f}) > "
              f"Random ({rn_ndcg:.4f}) — model is meaningful")
    else:
        print(f"\n  ⚠ BiRank re-ranking NDCG@10 ({br_ndcg:.4f}) ≤ "
              f"Random ({rn_ndcg:.4f}) — investigate")

    # ---- Save combined results ----
    # Main results file = per-user re-ranking (primary evaluation)
    rerank_df.to_csv(DATA_DIR / "validation_results.csv", index=False)
    venue_df.to_csv(DATA_DIR / "validation_venue_level.csv", index=False)
    global_df.to_csv(DATA_DIR / "validation_global.csv", index=False)
    print(f"\n  ✓ Saved validation_results.csv  (per-user re-ranking)")
    print(f"  ✓ Saved validation_venue_level.csv  (venue aggregate)")
    print(f"  ✓ Saved validation_global.csv  (global ranking)")

    # Summary text
    with open(DATA_DIR / "validation_summary.txt", "w") as f:
        f.write("TEMPORAL VALIDATION SUMMARY\n")
        f.write(f"Split date: {SPLIT_DATE.date()}\n")
        f.write(f"Training interactions: {len(train):,}\n")
        f.write(f"Testing interactions: {len(test):,}\n")
        f.write(f"Overlapping users: {len(overlap_users):,}\n")
        f.write(f"Users with future visits: {len(test_user_venues):,}\n\n")
        f.write("=" * 60 + "\n")
        f.write("EVALUATION A: PER-USER CANDIDATE RE-RANKING\n")
        f.write("=" * 60 + "\n")
        f.write(rerank_df.to_string(index=False))
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write("EVALUATION B: VENUE-LEVEL RANKING QUALITY\n")
        f.write("=" * 60 + "\n")
        f.write(venue_df.to_string(index=False))
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write("EVALUATION C: GLOBAL RANKING (reference)\n")
        f.write("=" * 60 + "\n")
        f.write(global_df.to_string(index=False))
        f.write("\n")
    print(f"  ✓ Saved validation_summary.txt")

    return rerank_df


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("This module is designed to be called from run_pipeline.py")
    print("Running standalone with pre-built data...\n")

    from run_pipeline import (load_data, compute_user_features,
                              compute_venue_features, birank,
                              build_adjacency)

    business, interactions, edges = load_data()
    user_feat = compute_user_features(interactions)
    venue_feat = compute_venue_features(interactions)

    venue_scores = pd.read_csv(DATA_DIR / "coffee_birank_venue_scores.csv")

    run_validation(interactions, venue_scores, user_feat, venue_feat,
                   edges, birank_fn=birank, build_adj_fn=build_adjacency)
