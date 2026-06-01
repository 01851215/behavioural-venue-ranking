"""
Restaurant BiRank Validation — v5-style with significance tests.

Applies the same rigorous methodology as validate_v5.py (coffee) to
the restaurant domain:
  - Temporal split at 2020-01-01 (same as coffee for direct comparability)
  - Features recomputed from training data only (no leakage)
  - Bootstrap 95% CI + Wilcoxon signed-rank p-values
  - Per-group NDCG@10 by restaurant user archetype

Methods evaluated:
  birank_count     — BiRank with count edges + behavioral priors
  birank_decay     — BiRank with temporal decay + behavioral priors
  s_ruc_model      — existing S(R,U,C) score (loaded from restaurant_scores.csv)
  baseline_rating  — mean star rating
  baseline_popularity — review count
  baseline_ease    — EASE collaborative filtering (from existing validation)

Outputs:
  restaurant_validation_summary.txt   — thesis-ready results table
  restaurant_birank_venue_scores.csv  — BiRank rankings for 64K venues
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

from bvr.core.validation import (
    compute_user_features,
    compute_venue_features,
    build_count_edges,
    build_decayed_edges,
    build_birank_ranking,
    build_rating_ranking,
    build_popularity_ranking,
    ndcg_at_k,
    hit_at_k,
    bootstrap_ci,
    K_VALUES,
    N_BOOTSTRAP,
)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

DATA_DIR   = Path(__file__).parent
SPLIT_DATE = "2020-01-01"


# ============================================================================
# Data loading
# ============================================================================

def load_data():
    print("Loading restaurant interactions (15M rows)...")
    df = pd.read_csv(DATA_DIR / "restaurant_interactions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["stars"]     = df["stars"].astype(float)
    print(f"  {len(df):,} reviews  |  "
          f"{df.user_id.nunique():,} users  |  "
          f"{df.business_id.nunique():,} venues")
    return df


def split_data(df, split_date):
    split = pd.Timestamp(split_date)
    train = df[df["timestamp"] < split].copy()
    test  = df[df["timestamp"] >= split].copy()

    overlap = set(train.user_id) & set(test.user_id)
    print(f"\nTemporal split: {split_date}")
    print(f"  Train: {len(train):,}  |  Test: {len(test):,}  |  Overlap users: {len(overlap):,}")

    train_uv = (train[train.user_id.isin(overlap)]
                .groupby("user_id")["business_id"].apply(set).to_dict())
    test_uv_all = (test[test.user_id.isin(overlap)]
                   .groupby("user_id")["business_id"].apply(set).to_dict())

    # v5-style: test_uv keeps ALL test venues (including revisits).
    # Candidate pool = training venues; ground truth = test venues.
    # A revisit (venue in both train & test) contributes to NDCG.
    # Only keep users who have ≥1 test venue that was also in training (revisit candidates).
    test_uv = {}
    for u in overlap:
        t_venues = train_uv.get(u, set())
        te_venues = test_uv_all.get(u, set())
        if te_venues & t_venues:  # ≥1 revisit available
            test_uv[u] = te_venues  # full test set as ground truth
    train_uv = {u: train_uv[u] for u in test_uv}

    revisit_total = sum(len(test_uv[u] & train_uv[u]) for u in test_uv)
    test_total    = sum(len(test_uv[u]) for u in test_uv)
    print(f"  Evaluable users (≥1 revisit in test): {len(test_uv):,}")
    print(f"  Revisit rate: {revisit_total}/{test_total} = {revisit_total/max(test_total,1)*100:.1f}%")
    return train, train_uv, test_uv


# ============================================================================
# Evaluation (same as validate_v5)
# ============================================================================

def evaluate_per_user(ranking_dict, train_uv, test_uv):
    uid_list = []
    scores = {f"{m}@{k}": [] for m in ("NDCG", "Hit") for k in K_VALUES}

    for uid in test_uv:
        if uid not in train_uv:
            continue
        cands = list(train_uv[uid])
        actual = test_uv[uid]
        cands_ranked = sorted(cands, key=lambda v: ranking_dict.get(v, 0), reverse=True)
        if len(cands_ranked) < 2:
            continue
        uid_list.append(uid)
        for k in K_VALUES:
            scores[f"NDCG@{k}"].append(ndcg_at_k(cands_ranked, actual, k))
            scores[f"Hit@{k}"].append(hit_at_k(cands_ranked, actual, k))

    per_user = {key: np.array(vals) for key, vals in scores.items()}
    agg = {key: float(arr.mean()) for key, arr in per_user.items()}
    agg["n_users"] = len(uid_list)
    return agg, per_user


def wilcoxon_p(a, b):
    try:
        _, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return float(p)
    except Exception:
        return 1.0


def evaluate_per_group(ranking_dict, train_uv, test_uv, profiles):
    groups = profiles.set_index("user_id")["archetype"].to_dict()
    group_names = sorted(profiles["archetype"].unique())
    results = {}
    for g in group_names:
        g_users = {u for u, a in groups.items() if a == g}
        g_train = {u: v for u, v in train_uv.items() if u in g_users}
        g_test  = {u: v for u, v in test_uv.items()  if u in g_users}
        if not g_test:
            continue
        agg, _ = evaluate_per_user(ranking_dict, g_train, g_test)
        results[g] = agg
    return results


# ============================================================================
# Print + save results
# ============================================================================

def print_results(all_results, per_group, train):
    best_key = max(
        [k for k in all_results if "birank" in k or "s_ruc" in k],
        key=lambda k: all_results[k]["agg"].get("NDCG@10", 0)
    )
    best_pu = all_results[best_key]["per_user"]["NDCG@10"]

    lines = []
    lines.append("=" * 80)
    lines.append("RESTAURANT BIRANK VALIDATION — v5-style (corrected, with significance)")
    lines.append(f"Split: {SPLIT_DATE}  |  Train: {len(train):,}  |  "
                 f"Venues: {train.business_id.nunique():,}")
    lines.append("=" * 80)

    lines.append(f"\n  {'Method':<28} {'NDCG@5':>8} {'NDCG@10':>8} {'NDCG@20':>8} "
                 f"{'Hit@10':>8} {'95% CI NDCG@10':>22} {'p-value':>10}")
    lines.append("  " + "-" * 100)

    for name, data in all_results.items():
        agg = data["agg"]
        pu  = data["per_user"]
        ci_lo, ci_hi = bootstrap_ci(pu["NDCG@10"])
        p = wilcoxon_p(best_pu, pu["NDCG@10"]) if name != best_key else 1.0
        sig = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
        lines.append(
            f"  {name:<28} {agg.get('NDCG@5',0):>8.4f} {agg.get('NDCG@10',0):>8.4f} "
            f"{agg.get('NDCG@20',0):>8.4f} {agg.get('Hit@10',0):>8.4f}  "
            f"[{ci_lo:.4f}, {ci_hi:.4f}]{sig:>6} {p:>10.4f}"
        )

    # Per-group
    if per_group:
        lines.append(f"\n  PER-GROUP NDCG@10 (best BiRank/S(R,U,C) variant):")
        lines.append(f"  {'Archetype':<30} {'NDCG@10':>10} {'n_users':>10}")
        lines.append("  " + "-" * 54)
        for g, agg in sorted(per_group.items(), key=lambda x: -x[1].get("NDCG@10", 0)):
            lines.append(f"  {g:<30} {agg.get('NDCG@10',0):>10.4f} {agg.get('n_users',0):>10,}")

    # Coffee comparison
    coffee_ndcg = 0.0765
    best_ndcg   = all_results.get(best_key, {}).get("agg", {}).get("NDCG@10", 0)
    lines.append(f"\n  CROSS-DOMAIN COMPARISON:")
    lines.append(f"    Coffee BiRank (v5_combined):   NDCG@10 = {coffee_ndcg:.4f}")
    lines.append(f"    Restaurant best ({best_key}): NDCG@10 = {best_ndcg:.4f}")
    lines.append(f"    Delta: {best_ndcg - coffee_ndcg:+.4f}")

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
    df = load_data()

    # 2. Split
    train, train_uv, test_uv = split_data(df, SPLIT_DATE)

    # 3. Features on training data only
    # Restrict to overlap users for speed (1.5M → 139K): non-overlap users
    # get uniform priors in BiRank, which is acceptable since we only evaluate overlap users.
    print("\nComputing user behavioral features (overlap users only for speed)...")
    overlap_users = set(train_uv.keys()) | set(test_uv.keys())
    train_overlap = train[train["user_id"].isin(overlap_users)]
    user_feat = compute_user_features(train_overlap)
    print(f"  {len(user_feat):,} users with features (of {train.user_id.nunique():,} total)")

    print("Computing venue behavioral features...")
    venue_feat = compute_venue_features(train)
    print(f"  {len(venue_feat):,} venues with features")

    # 4. Graph edges
    print("\nBuilding bipartite graph edges...")
    count_edges   = build_count_edges(train)
    decayed_edges = build_decayed_edges(train, SPLIT_DATE, lam=0.5)
    print(f"  Count edges: {len(count_edges):,}  |  Decayed: {len(decayed_edges):,}")

    # 5. Run methods
    print("\nRunning ranking methods...")
    rankings = {}
    rankings["birank_count"]  = build_birank_ranking(count_edges,   user_feat, venue_feat)
    print("  ✓ birank_count")
    rankings["birank_decay"]  = build_birank_ranking(decayed_edges, user_feat, venue_feat)
    print("  ✓ birank_decay")

    # Load existing S(R,U,C) scores — personalised, so aggregate to global venue mean
    sruc_path = DATA_DIR / "restaurant_scores.csv"
    if sruc_path.exists():
        sruc = pd.read_csv(sruc_path)
        if "business_id" in sruc.columns and "score" in sruc.columns:
            # Aggregate: mean S(R,U,C) score per venue = global quality proxy
            venue_sruc = sruc.groupby("business_id")["score"].mean().to_dict()
            rankings["s_ruc_global"] = venue_sruc
            n_covered = sruc["business_id"].nunique()
            n_users   = sruc["user_id"].nunique()
            print(f"  ✓ s_ruc_global (mean S(R,U,C) per venue — {n_covered:,} venues, "
                  f"{n_users:,} source users)")
        else:
            print("  ⚠ restaurant_scores.csv missing expected columns — skipping")

    rankings["baseline_rating"]     = build_rating_ranking(train)
    rankings["baseline_popularity"] = build_popularity_ranking(train)
    print("  ✓ baselines (rating, popularity)")

    # LightGCN (He et al., SIGIR 2020)
    try:
        import sys; sys.path.insert(0, str(DATA_DIR))
        from lightgcn import build_lightgcn_ranking
        print("  Training LightGCN...")
        rankings["lightgcn"] = build_lightgcn_ranking(train, verbose=True)
        print("  ✓ lightgcn")
    except Exception as e:
        print(f"  lightgcn FAILED: {e}")

    # 6. Evaluate
    print(f"\nEvaluating {len(test_uv):,} users...")
    all_results = {}
    for name, ranking in rankings.items():
        agg, per_user = evaluate_per_user(ranking, train_uv, test_uv)
        all_results[name] = {"agg": agg, "per_user": per_user}
        print(f"  {name:<28} NDCG@10={agg.get('NDCG@10',0):.4f}  n={agg.get('n_users',0):,}")

    # 7. Per-group
    print("\nPer-group evaluation...")
    profiles_path = DATA_DIR / "restaurant_user_profiles.csv"
    per_group = {}
    best_ranking_name = max(all_results, key=lambda k: all_results[k]["agg"].get("NDCG@10", 0))
    if profiles_path.exists():
        profiles = pd.read_csv(profiles_path)
        per_group = evaluate_per_group(
            rankings[best_ranking_name], train_uv, test_uv, profiles
        )
        for g, agg in per_group.items():
            print(f"  {g:<30}  NDCG@10={agg.get('NDCG@10',0):.4f}  n={agg.get('n_users',0):,}")

    # 8. Report
    report = print_results(all_results, per_group, train)
    out_path = DATA_DIR / "restaurant_validation_summary.txt"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nSaved → {out_path}")

    # 9. Save BiRank scores
    best_ranking = rankings.get("birank_decay", rankings.get("birank_count"))
    venue_scores = pd.DataFrame([
        {"business_id": vid, "birank_score": score, "rank": rank + 1}
        for rank, (vid, score) in enumerate(
            sorted(best_ranking.items(), key=lambda x: x[1], reverse=True)
        )
    ])
    venue_scores.to_csv(DATA_DIR / "restaurant_birank_venue_scores.csv", index=False)
    print(f"Saved → restaurant_birank_venue_scores.csv  ({len(venue_scores):,} venues)")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
