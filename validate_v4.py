"""
Phase 5: Ablation Validation — v3 vs v4 variants.

Runs 5 model variants on the same 2020-01-01 temporal split and compares:
  v3_baseline        — existing behavioral priors, no FSQ data
  v4_fsq_volume      — adds raw FSQ check-in volume to venue prior
  v4_social_direct   — adds direct friend signals only (no FoF)
  v4_social_fof      — adds direct + friend-of-friend signals
  v4_full            — all signals at best gamma from Phase 4

Evaluations:
  A) Per-user candidate re-ranking: NDCG@5/10/20, Hit@5/10/20
  B) Venue-level aggregate:         Spearman ρ, TopK Precision@10/20/50
  C) Social-linked subgroup:        NDCG@10 restricted to venues in venue_linkage

Outputs:
  validation_v4_results.csv         — Paradigm A, all variants
  validation_v4_venue_level.csv     — Paradigm B, all variants
  ablation_summary.csv              — delta vs v3 for every metric
  validation_social_subgroup.csv    — subgroup NDCG@10
"""

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.stats import spearmanr
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).parent

SPLIT_DATE  = pd.Timestamp("2020-01-01")
K_VALUES    = (5, 10, 20)
BEST_GAMMA  = 0.2   # from Phase 4 (all gammas tie — use 0.2 as default social weight)


# ============================================================================
# CORE BIRANK (copied from run_pipeline.py — no imports to avoid side effects)
# ============================================================================

def build_adjacency(edges):
    users  = edges["user_id"].unique()
    venues = edges["business_id"].unique()
    u2i = {u: i for i, u in enumerate(users)}
    v2i = {v: i for i, v in enumerate(venues)}
    i2u = {i: u for u, i in u2i.items()}
    i2v = {i: v for v, i in v2i.items()}
    rows = [u2i[u] for u in edges["user_id"]]
    cols = [v2i[v] for v in edges["business_id"]]
    W = sparse.csr_matrix(
        (edges["weight"].values.astype(np.float64), (rows, cols)),
        shape=(len(users), len(venues))
    )
    return W, u2i, v2i, i2u, i2v


def birank(W, p0=None, q0=None, alpha=0.85, beta=0.85, max_iter=200, tol=1e-8):
    nu, nv = W.shape
    if p0 is None: p0 = np.ones(nu) / nu
    if q0 is None: q0 = np.ones(nv) / nv
    p0 = p0 / p0.sum(); q0 = q0 / q0.sum()
    rs = np.array(W.sum(axis=1)).flatten(); rs[rs == 0] = 1.0
    Su = sparse.diags(1.0 / rs) @ W
    cs = np.array(W.sum(axis=0)).flatten(); cs[cs == 0] = 1.0
    Sv = W @ sparse.diags(1.0 / cs)
    p, q = p0.copy(), q0.copy()
    for it in range(1, max_iter + 1):
        p_prev, q_prev = p.copy(), q.copy()
        p = alpha * (Su @ q)  + (1 - alpha) * p0
        q = beta  * (Sv.T @ p) + (1 - beta)  * q0
        p /= p.sum(); q /= q.sum()
        if np.abs(p - p_prev).sum() < tol and np.abs(q - q_prev).sum() < tol:
            break
    return p, q


# ============================================================================
# PRIOR BUILDERS — one per variant
# ============================================================================

def behavioral_priors(train, user_feat, venue_feat, u2i, v2i):
    """v3 baseline: pure behavioral priors."""
    nu, nv = len(u2i), len(v2i)
    bust_map   = user_feat.set_index("user_id")["burstiness_index"]
    visits_map = user_feat.set_index("user_id")["total_visits"]
    rr_map     = venue_feat.set_index("business_id")["repeat_user_rate"]
    cv_map     = venue_feat.set_index("business_id")["stability_cv"]

    p0 = np.ones(nu)
    for uid, idx in u2i.items():
        b = bust_map.get(uid, 0);   b = 0 if np.isnan(b) else b
        v = visits_map.get(uid, 1); v = 1 if np.isnan(v) else v
        p0[idx] = np.log1p(v) * (1 - b)

    q0 = np.ones(nv)
    for vid, idx in v2i.items():
        rr = rr_map.get(vid, 0); rr = 0 if np.isnan(rr) else rr
        cv = cv_map.get(vid, 1); cv = 1 if np.isnan(cv) else cv
        q0[idx] = rr * (1 / (1 + cv))

    return np.clip(p0, 1e-10, None), np.clip(q0, 1e-10, None)


def social_venue_prior(base_q0, v2i, social_df, gamma,
                       use_friend=True, use_fof=True):
    """
    Augment venue prior with social signal.
    social_boost = normalised log(1 + friend_checkin_count)
                 + normalised log(1 + fof_checkin_count) * FOF_WEIGHT
    """
    soc = social_df.copy()

    if use_friend and use_fof:
        raw = np.log1p(soc["friend_checkin_count"]) + 0.3 * np.log1p(soc["fof_checkin_count"])
    elif use_friend:
        raw = np.log1p(soc["friend_checkin_count"])
    elif use_fof:
        raw = 0.3 * np.log1p(soc["fof_checkin_count"])
    else:
        return base_q0.copy()

    max_raw = raw.max()
    soc["boost"] = raw / max(max_raw, 1e-10)
    boost_map = dict(zip(soc["yelp_business_id"], soc["boost"] * soc["mean_bridge_confidence"]))

    q0 = base_q0.copy()
    for vid, idx in v2i.items():
        behavioral = base_q0[idx]
        social_term = boost_map.get(vid, 0.0)
        q0[idx] = (1 - gamma) * behavioral + gamma * social_term

    return np.clip(q0, 1e-10, None)


def fsq_volume_prior(base_q0, v2i, checkins_linked):
    """
    v4_fsq_volume: add raw FSQ check-in count at each linked venue
    to the behavioral prior (no social graph, just visit mass).
    """
    fsq_counts = (
        checkins_linked
        .groupby("yelp_business_id")
        .size()
        .reset_index(name="fsq_count")
    )
    max_log = np.log1p(fsq_counts["fsq_count"].max())
    fsq_counts["fsq_boost"] = np.log1p(fsq_counts["fsq_count"]) / max(max_log, 1e-10)
    fsq_map = dict(zip(fsq_counts["yelp_business_id"], fsq_counts["fsq_boost"]))

    q0 = base_q0.copy()
    gamma = BEST_GAMMA
    for vid, idx in v2i.items():
        behavioral = base_q0[idx]
        fsq_term = fsq_map.get(vid, 0.0)
        q0[idx] = (1 - gamma) * behavioral + gamma * fsq_term

    return np.clip(q0, 1e-10, None)


# ============================================================================
# RANKING BUILDER — runs BiRank on training window
# ============================================================================

def build_ranking(train, user_feat, venue_feat, social_df,
                  checkins_linked, variant):
    """
    Re-run BiRank on train-window edges with variant-specific priors.
    Returns {business_id: birank_score}.
    """
    edge_df = (
        train.groupby(["user_id", "business_id"])
        .size().reset_index(name="weight")
    )
    W, u2i, v2i, i2u, i2v = build_adjacency(edge_df)

    p0_base, q0_base = behavioral_priors(train, user_feat, venue_feat, u2i, v2i)

    if variant == "v3_baseline":
        p0, q0 = p0_base, q0_base

    elif variant == "v4_fsq_volume":
        p0 = p0_base
        q0 = fsq_volume_prior(q0_base, v2i, checkins_linked)

    elif variant == "v4_social_direct":
        p0 = p0_base
        q0 = social_venue_prior(q0_base, v2i, social_df, BEST_GAMMA,
                                use_friend=True, use_fof=False)

    elif variant == "v4_social_fof":
        p0 = p0_base
        q0 = social_venue_prior(q0_base, v2i, social_df, BEST_GAMMA,
                                use_friend=True, use_fof=True)

    elif variant == "v4_full":
        p0 = p0_base
        q0 = social_venue_prior(q0_base, v2i, social_df, BEST_GAMMA,
                                use_friend=True, use_fof=True)
        # Also blend in raw FSQ volume
        q0_fsq = fsq_volume_prior(q0_base, v2i, checkins_linked)
        q0 = 0.5 * q0 + 0.5 * q0_fsq
        q0 = np.clip(q0, 1e-10, None)

    _, q = birank(W, p0=p0, q0=q0)
    return {i2v[i]: float(q[i]) for i in range(len(i2v))}


# ============================================================================
# EVALUATION FUNCTIONS (matching temporal_validation.py)
# ============================================================================

def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if len(relevances) == 0: return 0.0
    return float(np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2))))


def ndcg_at_k_user(predicted, actual, k):
    rel = [1.0 if v in actual else 0.0 for v in predicted[:k]]
    dcg  = dcg_at_k(rel, k)
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k_user(predicted, actual, k):
    return 1.0 if set(predicted[:k]) & actual else 0.0


def evaluate_reranking(ranking, train_uv, test_uv):
    results = {}
    for k in K_VALUES:
        ndcgs, hits = [], []
        for uid, actual in test_uv.items():
            if uid not in train_uv: continue
            cands = sorted(train_uv[uid], key=lambda v: ranking.get(v, 0), reverse=True)
            if len(cands) < 2: continue
            ndcgs.append(ndcg_at_k_user(cands, actual, k))
            hits.append(hit_at_k_user(cands, actual, k))
        results[f"NDCG@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        results[f"Hit@{k}"]  = float(np.mean(hits))  if hits  else 0.0
    return results


def evaluate_venue_level(ranking, test_uv):
    test_counts = defaultdict(int)
    for venues in test_uv.values():
        for v in venues:
            test_counts[v] += 1
    common = set(ranking) & set(test_counts)
    if len(common) < 10:
        return {"Spearman_rho": np.nan, "Spearman_p": np.nan}
    pred   = [ranking[v]     for v in common]
    actual = [test_counts[v] for v in common]
    rho, p = spearmanr(pred, actual)
    res = {"Spearman_rho": round(float(rho), 4), "Spearman_p": float(p)}
    sp = sorted(common, key=lambda v: ranking[v],     reverse=True)
    sa = sorted(common, key=lambda v: test_counts[v], reverse=True)
    for k in (10, 20, 50):
        overlap = len(set(sp[:k]) & set(sa[:k]))
        res[f"TopK_Precision@{k}"] = round(overlap / k, 4) if k <= len(common) else np.nan
    return res


def evaluate_social_subgroup(ranking, test_uv, train_uv, social_venue_ids):
    """
    NDCG@10 restricted to social-linked venues as candidates.
    Isolates improvement driven specifically by the social signal.
    """
    ndcgs = []
    for uid, actual in test_uv.items():
        if uid not in train_uv: continue
        # Candidates = only social-linked venues the user visited in training
        cands_all = train_uv[uid]
        cands_social = [v for v in cands_all if v in social_venue_ids]
        if len(cands_social) < 2: continue
        cands_ranked = sorted(cands_social, key=lambda v: ranking.get(v, 0), reverse=True)
        actual_social = actual & social_venue_ids
        if not actual_social: continue
        ndcgs.append(ndcg_at_k_user(cands_ranked, actual_social, 10))
    return float(np.mean(ndcgs)) if ndcgs else np.nan


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE 5: ABLATION VALIDATION")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    interactions = pd.read_csv(DATA_DIR / "coffee_interactions.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    user_feat  = pd.read_csv(DATA_DIR / "coffee_user_features_v3.csv")
    venue_feat = pd.read_csv(DATA_DIR / "coffee_venue_features_v3.csv")
    social_df  = pd.read_csv(DATA_DIR / "social_venue_signals.csv")
    checkins   = pd.read_parquet(DATA_DIR / "fsq_checkins_linked.parquet")
    linkage    = pd.read_csv(DATA_DIR / "venue_linkage.csv")
    social_venue_ids = set(linkage["yelp_business_id"])

    print(f"  Interactions: {len(interactions):,}")
    print(f"  Social-linked venues: {len(social_venue_ids):,}")

    # Temporal split
    train = interactions[interactions["timestamp"] < SPLIT_DATE].copy()
    test  = interactions[interactions["timestamp"] >= SPLIT_DATE].copy()
    print(f"\n  Train: {len(train):,}  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"  Test:  {len(test):,}  ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")

    # Ground truth
    overlap = set(train["user_id"].dropna()) & set(test["user_id"].dropna())
    print(f"  Overlapping users: {len(overlap):,}")

    train_uv, test_uv = {}, {}
    for uid in overlap:
        tv  = set(test[test["user_id"]   == uid]["business_id"].unique())
        trv = set(train[train["user_id"] == uid]["business_id"].unique())
        if tv:  test_uv[uid]  = tv
        if trv: train_uv[uid] = trv
    print(f"  Users with future visits: {len(test_uv):,}")

    # Run variants
    VARIANTS = [
        "v3_baseline",
        "v4_fsq_volume",
        "v4_social_direct",
        "v4_social_fof",
        "v4_full",
    ]

    rerank_rows, venue_rows, subgroup_rows = [], [], []

    for variant in VARIANTS:
        print(f"\n  [{variant}]")
        t0 = time.time()
        ranking = build_ranking(
            train, user_feat, venue_feat, social_df, checkins, variant
        )
        elapsed = time.time() - t0
        print(f"    Built ranking in {elapsed:.1f}s  ({len(ranking):,} venues)")

        # Paradigm A
        a = evaluate_reranking(ranking, train_uv, test_uv)
        a["variant"] = variant
        rerank_rows.append(a)

        # Paradigm B
        b = evaluate_venue_level(ranking, test_uv)
        b["variant"] = variant
        venue_rows.append(b)

        # Social subgroup
        sg = evaluate_social_subgroup(ranking, test_uv, train_uv, social_venue_ids)
        subgroup_rows.append({"variant": variant, "subgroup_NDCG@10": sg})

        print(f"    NDCG@10={a['NDCG@10']:.6f}  Hit@10={a['Hit@10']:.6f}  "
              f"ρ={b.get('Spearman_rho', float('nan')):.4f}  "
              f"SocialNDCG@10={sg:.6f}" if not np.isnan(sg) else
              f"    NDCG@10={a['NDCG@10']:.6f}  Hit@10={a['Hit@10']:.6f}  "
              f"ρ={b.get('Spearman_rho', float('nan')):.4f}  SocialNDCG@10=n/a")

    # Build result tables
    rerank_df   = pd.DataFrame(rerank_rows)
    venue_df    = pd.DataFrame(venue_rows)
    subgroup_df = pd.DataFrame(subgroup_rows)

    # Ablation delta table
    baseline_ndcg = rerank_df.loc[rerank_df["variant"] == "v3_baseline", "NDCG@10"].iloc[0]
    baseline_rho  = venue_df.loc[venue_df["variant"]   == "v3_baseline", "Spearman_rho"].iloc[0]

    ablation_rows = []
    for i, row in rerank_df.iterrows():
        v = row["variant"]
        rho = venue_df.loc[venue_df["variant"] == v, "Spearman_rho"].iloc[0]
        sg  = subgroup_df.loc[subgroup_df["variant"] == v, "subgroup_NDCG@10"].iloc[0]
        ablation_rows.append({
            "variant":             v,
            "NDCG@5":              row["NDCG@5"],
            "NDCG@10":             row["NDCG@10"],
            "NDCG@20":             row["NDCG@20"],
            "Hit@10":              row["Hit@10"],
            "Spearman_rho":        rho,
            "Social_subgroup_NDCG@10": sg,
            "delta_NDCG@10":       row["NDCG@10"] - baseline_ndcg,
            "delta_rho":           rho - baseline_rho if not np.isnan(rho) and not np.isnan(baseline_rho) else np.nan,
        })

    ablation_df = pd.DataFrame(ablation_rows)

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(ablation_df[[
        "variant", "NDCG@10", "Hit@10", "Spearman_rho",
        "Social_subgroup_NDCG@10", "delta_NDCG@10"
    ]].to_string(index=False))

    # Winner
    best_v = ablation_df.loc[ablation_df["NDCG@10"].idxmax(), "variant"]
    best_ndcg = ablation_df["NDCG@10"].max()
    print(f"\n  Best variant: {best_v}  (NDCG@10={best_ndcg:.6f})")
    if best_v != "v3_baseline":
        delta = best_ndcg - baseline_ndcg
        pct   = (delta / baseline_ndcg * 100) if baseline_ndcg > 0 else 0
        print(f"  Improvement over v3: {delta:+.6f}  ({pct:+.2f}%)")

    # Save
    rerank_df.to_csv(DATA_DIR / "validation_v4_results.csv",      index=False)
    venue_df.to_csv(DATA_DIR  / "validation_v4_venue_level.csv",  index=False)
    subgroup_df.to_csv(DATA_DIR / "validation_social_subgroup.csv", index=False)
    ablation_df.to_csv(DATA_DIR / "ablation_summary.csv",          index=False)

    print(f"\n  Saved: validation_v4_results.csv")
    print(f"  Saved: validation_v4_venue_level.csv")
    print(f"  Saved: validation_social_subgroup.csv")
    print(f"  Saved: ablation_summary.csv")


if __name__ == "__main__":
    main()
