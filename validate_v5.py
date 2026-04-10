"""
Phase 6: Comprehensive Validation v5.

Fixes applied:
  1. Feature leakage:  recompute user/venue features from TRAINING data only
  2. NDCG bug:         IDCG uses total relevant candidates, not just top-k slice
  3. Significance:     bootstrap 95% CI + Wilcoxon signed-rank on per-user scores
  4. Per-group eval:   NDCG@10 stratified by behavioral user segment

New model improvements:
  5. Temporal edge decay:    recent visits weighted exponentially more
  6. Selective social:       only high-confidence FSQ bridge venues get social boost
  7. Stronger baselines:     item-KNN (cosine), IUF-popularity

Variants tested:
  v3_baseline           — count edges + behavioral priors (reproduced with fixes)
  v5_temporal_decay     — exponentially decayed edges + behavioral priors
  v5_selective_social   — count edges + behavioral + filtered social (gamma=0.15)
  v5_combined           — decayed edges + behavioral + filtered social (gamma=0.15)
  baseline_rating       — mean star rating
  baseline_popularity   — raw visit count
  baseline_iuf          — inverse-user-frequency weighted popularity
  baseline_item_knn     — cosine item-item KNN
  baseline_random       — random

Evaluation:
  A) Per-user candidate re-ranking: NDCG@5/10/20, Hit@5/10/20
  B) Per-group NDCG@10 by user segment
  C) Significance: bootstrap 95% CI, Wilcoxon p-value vs best variant
  D) Robustness: 3 temporal splits (2019-01-01, 2019-07-01, 2020-01-01)

Outputs:
  validation_v5_results.csv       — main results with CIs and p-values
  validation_v5_per_group.csv     — per-group breakdown
  validation_v5_robustness.csv    — multi-split NDCG@10 comparison
  validation_v5_summary.txt       — human-readable report
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.stats import spearmanr, wilcoxon
from collections import defaultdict

warnings.filterwarnings("ignore")

# Flush prints immediately so we can monitor progress
sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent

# ---- Configuration ----
SPLIT_DATES    = ["2019-01-01", "2019-07-01", "2020-01-01"]
PRIMARY_SPLIT  = "2020-01-01"
K_VALUES       = (5, 10, 20)
DECAY_LAMBDA   = 0.5        # half-life ≈ 1.4 years
N_BOOTSTRAP    = 1000
SOCIAL_GAMMA   = 0.15       # social blending weight (lower than v4's 0.2)
SOCIAL_CONF_MIN = 0.3       # minimum bridge confidence for selective social
KNN_NEIGHBORS  = 30


# ============================================================================
# BIRANK ENGINE
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
        shape=(len(users), len(venues)),
    )
    return W, u2i, v2i, i2u, i2v


def birank(W, p0=None, q0=None, alpha=0.85, beta=0.85, max_iter=200, tol=1e-8):
    nu, nv = W.shape
    if p0 is None:
        p0 = np.ones(nu) / nu
    if q0 is None:
        q0 = np.ones(nv) / nv
    p0 = p0 / p0.sum()
    q0 = q0 / q0.sum()

    rs = np.array(W.sum(axis=1)).flatten()
    rs[rs == 0] = 1.0
    Su = sparse.diags(1.0 / rs) @ W

    cs = np.array(W.sum(axis=0)).flatten()
    cs[cs == 0] = 1.0
    Sv = W @ sparse.diags(1.0 / cs)

    p, q = p0.copy(), q0.copy()
    for it in range(1, max_iter + 1):
        p_prev, q_prev = p.copy(), q.copy()
        p = alpha * (Su @ q) + (1 - alpha) * p0
        q = beta * (Sv.T @ p) + (1 - beta) * q0
        p /= p.sum()
        q /= q.sum()
        if np.abs(p - p_prev).sum() < tol and np.abs(q - q_prev).sum() < tol:
            break
    return p, q


# ============================================================================
# FEATURE COMPUTATION (called on TRAINING data only — fixes leakage)
# ============================================================================

def _safe(val, default=0):
    """Return default if val is NaN."""
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


def corrected_burstiness(intervals):
    if len(intervals) < 2:
        return np.nan
    mu, sigma = intervals.mean(), intervals.std()
    denom = sigma + mu
    return 0.0 if denom == 0 else (sigma - mu) / denom


def compute_user_features(interactions):
    rows = []
    for uid, udf in interactions.groupby("user_id"):
        udf = udf.sort_values("timestamp")
        total = len(udf)
        vc = udf["business_id"].value_counts()
        unique = len(vc)
        top1 = vc.iloc[0] / total if total > 0 else 0

        if total >= 2:
            ts = udf["timestamp"].sort_values()
            intervals = ts.diff().dt.total_seconds().dropna() / 86400
            bust = corrected_burstiness(intervals.values)
            span = (ts.max() - ts.min()).total_seconds() / 86400
        else:
            bust, span = np.nan, 0.0

        probs = vc / total
        entropy = float(-np.sum(probs * np.log2(probs))) if total > 1 else 0.0
        revisit = (total - unique) / total if total > 0 else 0

        rows.append({
            "user_id": uid, "total_visits": total,
            "unique_venues": unique, "revisit_ratio": revisit,
            "top1_venue_share": top1, "burstiness_index": bust,
            "active_span_days": span, "venue_entropy": entropy,
        })
    return pd.DataFrame(rows)


def compute_venue_features(interactions):
    rows = []
    for bid, vdf in interactions.groupby("business_id"):
        total = len(vdf)
        uc = vdf["user_id"].value_counts()
        unique_users = len(uc)
        repeat_users = int((uc >= 2).sum())
        repeat_rate = repeat_users / unique_users if unique_users > 0 else 0

        if total >= 7:
            vdf_c = vdf.copy()
            vdf_c["week"] = vdf_c["timestamp"].dt.to_period("W")
            weekly = vdf_c.groupby("week").size()
            cv = float(weekly.std() / weekly.mean()) if weekly.mean() > 0 else 0.0
        else:
            cv = 0.0

        rows.append({
            "business_id": bid,
            "popularity_visits": total,
            "unique_users": unique_users,
            "repeat_user_rate": repeat_rate,
            "stability_cv": cv,
        })
    return pd.DataFrame(rows)


# ============================================================================
# EDGE BUILDERS
# ============================================================================

def build_count_edges(train):
    return train.groupby(["user_id", "business_id"]).size().reset_index(name="weight")


def build_decayed_edges(train, split_date, lam=DECAY_LAMBDA):
    """Weight each interaction by exp(-lambda * age_in_years)."""
    t = train.copy()
    age_days = (pd.Timestamp(split_date) - t["timestamp"]).dt.total_seconds() / 86400.0
    t["decay_weight"] = np.exp(-lam * age_days / 365.0)
    return (
        t.groupby(["user_id", "business_id"])
        .agg(weight=("decay_weight", "sum"))
        .reset_index()
    )


# ============================================================================
# PRIOR BUILDERS
# ============================================================================

def behavioral_priors(user_feat, venue_feat, u2i, v2i):
    nu, nv = len(u2i), len(v2i)
    bust_map   = user_feat.set_index("user_id")["burstiness_index"]
    visits_map = user_feat.set_index("user_id")["total_visits"]
    rr_map     = venue_feat.set_index("business_id")["repeat_user_rate"]
    cv_map     = venue_feat.set_index("business_id")["stability_cv"]

    p0 = np.ones(nu)
    for uid, idx in u2i.items():
        b = _safe(bust_map.get(uid, 0), 0)
        v = _safe(visits_map.get(uid, 1), 1)
        p0[idx] = np.log1p(v) * (1 - b)

    q0 = np.ones(nv)
    for vid, idx in v2i.items():
        rr = _safe(rr_map.get(vid, 0), 0)
        cv = _safe(cv_map.get(vid, 1), 1)
        q0[idx] = rr * (1 / (1 + cv))

    return np.clip(p0, 1e-10, None), np.clip(q0, 1e-10, None)


def selective_social_prior(base_q0, v2i, social_df, gamma=SOCIAL_GAMMA,
                           conf_min=SOCIAL_CONF_MIN):
    """Augment venue prior with social signal from high-confidence bridges only."""
    soc = social_df[social_df["mean_bridge_confidence"] >= conf_min].copy()
    if len(soc) == 0:
        return base_q0.copy()

    raw = np.log1p(soc["friend_checkin_count"])
    mx = raw.max()
    soc = soc.copy()
    soc["boost"] = (raw / max(mx, 1e-10)) * soc["mean_bridge_confidence"]
    boost_map = dict(zip(soc["yelp_business_id"], soc["boost"]))

    q0 = base_q0.copy()
    for vid, idx in v2i.items():
        if vid in boost_map:
            q0[idx] = (1 - gamma) * base_q0[idx] + gamma * boost_map[vid]
    return np.clip(q0, 1e-10, None)


# ============================================================================
# RANKING BUILDERS
# ============================================================================

def build_birank_ranking(edges_df, user_feat, venue_feat,
                         social_df=None, use_social=False):
    W, u2i, v2i, i2u, i2v = build_adjacency(edges_df)
    p0, q0 = behavioral_priors(user_feat, venue_feat, u2i, v2i)
    if use_social and social_df is not None:
        q0 = selective_social_prior(q0, v2i, social_df)
    _, q = birank(W, p0=p0, q0=q0)
    return {i2v[i]: float(q[i]) for i in range(len(i2v))}


def build_rating_ranking(train):
    if "stars" in train.columns:
        rated = train.dropna(subset=["stars"])
        if len(rated) > 0:
            return rated.groupby("business_id")["stars"].mean().to_dict()
    # Fallback: try loading baselines file
    bp = DATA_DIR / "coffee_baselines.csv"
    if bp.exists():
        bl = pd.read_csv(bp)
        if "rating_mean" in bl.columns:
            return bl.set_index("business_id")["rating_mean"].to_dict()
    return {}


def build_popularity_ranking(train):
    return train.groupby("business_id").size().to_dict()


def build_iuf_ranking(train):
    """Inverse-User-Frequency: visits from focused users count more."""
    uvc = train.groupby("user_id")["business_id"].nunique()
    t = train.copy()
    t["iuf"] = t["user_id"].map(lambda u: 1.0 / np.log1p(uvc.get(u, 1)))
    return t.groupby("business_id")["iuf"].sum().to_dict()


def build_item_knn_ranking(edges_df, top_k=KNN_NEIGHBORS):
    """Cosine item-item KNN: venue score = mean similarity to top-K neighbors."""
    W, u2i, v2i, i2u, i2v = build_adjacency(edges_df)
    nv = len(v2i)

    # Column-normalise for cosine similarity
    col_sq = np.array(W.power(2).sum(axis=0)).flatten()
    col_norms = np.sqrt(col_sq)
    col_norms[col_norms == 0] = 1.0
    W_n = W @ sparse.diags(1.0 / col_norms)

    # Item similarity: S = W_n.T @ W_n  (nv x nv, sparse)
    S = (W_n.T @ W_n).tocsr()
    S.setdiag(0)

    scores = {}
    for i in range(nv):
        row = S.getrow(i).toarray().flatten()
        if len(row) <= top_k:
            scores[i2v[i]] = float(row.mean()) if len(row) > 0 else 0.0
        else:
            top_vals = np.partition(row, -top_k)[-top_k:]
            scores[i2v[i]] = float(top_vals.mean())
    return scores


# ============================================================================
# CORRECTED EVALUATION METRICS
# ============================================================================

def ndcg_at_k(predicted_full, actual_set, k):
    """
    Corrected NDCG@k.

    IDCG uses min(k, total_relevant_in_candidates), not just the
    relevant items that happen to fall in the top-k slice.
    """
    rel_k = [1.0 if v in actual_set else 0.0 for v in predicted_full[:k]]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel_k))

    n_relevant = sum(1 for v in predicted_full if v in actual_set)
    n_ideal = min(k, n_relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_ideal))

    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(predicted, actual_set, k):
    return 1.0 if set(predicted[:k]) & actual_set else 0.0


# ============================================================================
# PER-USER EVALUATION (returns arrays for significance testing)
# ============================================================================

def evaluate_per_user(ranking_dict, train_uv, test_uv):
    """
    Per-user candidate re-ranking.

    For each user, rank their training-window venues by the method's score,
    then check which ones they actually revisited in the test window.

    Returns: (agg_results, per_user_arrays, evaluated_uids)
    """
    uid_list = []
    scores = {f"{m}@{k}": [] for m in ("NDCG", "Hit") for k in K_VALUES}

    for uid in test_uv:
        if uid not in train_uv:
            continue
        cands = list(train_uv[uid])
        actual = test_uv[uid]
        # Rank ALL candidates (not just top-k) for correct IDCG
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
    return agg, per_user, uid_list


# ============================================================================
# VENUE-LEVEL EVALUATION
# ============================================================================

def evaluate_venue_level(ranking_dict, test_uv):
    """Spearman correlation between predicted score and future traffic."""
    counts = defaultdict(int)
    for venues in test_uv.values():
        for v in venues:
            counts[v] += 1
    common = set(ranking_dict) & set(counts)
    if len(common) < 10:
        return {"Spearman_rho": np.nan, "Spearman_p": np.nan}
    pred   = [ranking_dict[v] for v in common]
    actual = [counts[v]       for v in common]
    rho, p = spearmanr(pred, actual)
    return {"Spearman_rho": round(float(rho), 4), "Spearman_p": float(p)}


# ============================================================================
# STATISTICAL SIGNIFICANCE
# ============================================================================

def bootstrap_ci(scores, n_boot=N_BOOTSTRAP, confidence=0.95):
    """Bootstrap 95% CI for the mean of per-user scores."""
    n = len(scores)
    rng = np.random.RandomState(42)
    means = [scores[rng.choice(n, n, replace=True)].mean() for _ in range(n_boot)]
    alpha = (1 - confidence) / 2
    return (float(np.percentile(means, 100 * alpha)),
            float(np.percentile(means, 100 * (1 - alpha))))


def wilcoxon_p(a, b):
    """Wilcoxon signed-rank p-value (two-sided). Returns 1.0 if identical."""
    diff = a - b
    if np.all(diff == 0):
        return 1.0
    try:
        _, p = wilcoxon(diff, alternative="two-sided")
        return float(p)
    except Exception:
        return 1.0


# ============================================================================
# PER-GROUP EVALUATION
# ============================================================================

def assign_user_groups(user_feat):
    """
    Rule-based assignment matching the original 4-cluster schema.
    Uses revisit_ratio, total_visits, and venue_entropy.
    """
    uf = user_feat.copy()
    uf["burstiness_index"] = uf["burstiness_index"].fillna(0)

    uf["group"] = "Casual Weekenders"  # default
    uf.loc[uf["revisit_ratio"] >= 0.25, "group"] = "Loyalists"
    uf.loc[
        (uf["total_visits"] <= 2) & (uf["revisit_ratio"] < 0.25), "group"
    ] = "Infrequent Visitors"
    uf.loc[
        (uf["total_visits"] >= 4) & (uf["revisit_ratio"] < 0.25), "group"
    ] = "Weekday Regulars"

    return uf[["user_id", "group"]]


def evaluate_per_group(ranking_dict, train_uv, test_uv, user_groups, k=10):
    gmap = dict(zip(user_groups["user_id"], user_groups["group"]))
    group_scores = defaultdict(list)

    for uid in test_uv:
        if uid not in train_uv:
            continue
        g = gmap.get(uid, "Unknown")
        cands = sorted(train_uv[uid], key=lambda v: ranking_dict.get(v, 0), reverse=True)
        if len(cands) < 2:
            continue
        group_scores[g].append(ndcg_at_k(cands, test_uv[uid], k))

    return {
        g: {
            "mean_NDCG@10": float(np.mean(s)),
            "std": float(np.std(s)),
            "n_users": len(s),
        }
        for g, s in group_scores.items()
    }


# ============================================================================
# SINGLE SPLIT RUNNER
# ============================================================================

def run_single_split(interactions, social_df, split_date, verbose=True):
    split = pd.Timestamp(split_date)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  TEMPORAL SPLIT: {split.date()}")
        print(f"{'='*70}")

    # ---- Split ----
    train = interactions[interactions["timestamp"] < split].copy()
    test  = interactions[interactions["timestamp"] >= split].copy()
    if verbose:
        print(f"  Train: {len(train):>10,} interactions  "
              f"({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
        print(f"  Test:  {len(test):>10,} interactions  "
              f"({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")

    overlap = set(train["user_id"].dropna()) & set(test["user_id"].dropna())
    if verbose:
        print(f"  Overlapping users: {len(overlap):,}")
    if len(overlap) < 50:
        print("  ⚠ Too few overlapping users — skipping this split.")
        return None

    # ---- Ground truth ----
    train_uv, test_uv = {}, {}
    for uid in overlap:
        tv  = set(test[test["user_id"]  == uid]["business_id"].unique())
        trv = set(train[train["user_id"] == uid]["business_id"].unique())
        if tv:  test_uv[uid]  = tv
        if trv: train_uv[uid] = trv
    if verbose:
        print(f"  Users with future visits: {len(test_uv):,}")

    # ---- Features from TRAINING data only (fixes leakage) ----
    if verbose:
        print("\n  Computing features from training data only...")
    t0 = time.time()
    user_feat  = compute_user_features(train)
    venue_feat = compute_venue_features(train)
    if verbose:
        print(f"  Done in {time.time()-t0:.0f}s  "
              f"({len(user_feat):,} users, {len(venue_feat):,} venues)")

    # ---- Edge variants ----
    count_edges = build_count_edges(train)
    decay_edges = build_decayed_edges(train, split)

    # ---- Build rankings ----
    if verbose:
        print("\n  Building rankings...")

    rankings = {}

    t0 = time.time()
    rankings["v3_baseline"] = build_birank_ranking(count_edges, user_feat, venue_feat)
    if verbose:
        print(f"    v3_baseline          {time.time()-t0:>5.1f}s")

    t0 = time.time()
    rankings["v5_temporal_decay"] = build_birank_ranking(
        decay_edges, user_feat, venue_feat,
    )
    if verbose:
        print(f"    v5_temporal_decay    {time.time()-t0:>5.1f}s")

    if social_df is not None and len(social_df) > 0:
        t0 = time.time()
        rankings["v5_selective_social"] = build_birank_ranking(
            count_edges, user_feat, venue_feat,
            social_df=social_df, use_social=True,
        )
        if verbose:
            print(f"    v5_selective_social  {time.time()-t0:>5.1f}s")

        t0 = time.time()
        rankings["v5_combined"] = build_birank_ranking(
            decay_edges, user_feat, venue_feat,
            social_df=social_df, use_social=True,
        )
        if verbose:
            print(f"    v5_combined          {time.time()-t0:>5.1f}s")

    # Baselines
    t0 = time.time()
    rankings["baseline_rating"]     = build_rating_ranking(train)
    rankings["baseline_popularity"] = build_popularity_ranking(train)
    rankings["baseline_iuf"]        = build_iuf_ranking(train)

    rng = np.random.RandomState(42)
    all_v = list(set(train["business_id"].unique()))
    rankings["baseline_random"] = {v: rng.random() for v in all_v}

    rankings["baseline_item_knn"] = build_item_knn_ranking(count_edges)
    if verbose:
        print(f"    baselines            {time.time()-t0:>5.1f}s")

    # ---- Evaluate ----
    if verbose:
        print("\n  Evaluating per-user re-ranking...")

    all_agg    = {}
    all_pu     = {}
    all_venue  = {}
    for name, ranking in rankings.items():
        agg, pu, _ = evaluate_per_user(ranking, train_uv, test_uv)
        all_agg[name]   = agg
        all_pu[name]    = pu
        all_venue[name] = evaluate_venue_level(ranking, test_uv)

    # ---- Significance ----
    birank_keys = [k for k in rankings if k.startswith("v")]
    best_key = max(birank_keys, key=lambda k: all_agg[k]["NDCG@10"])
    best_pu  = all_pu[best_key]["NDCG@10"]

    sig = {}
    for name in rankings:
        ci_lo, ci_hi = bootstrap_ci(all_pu[name]["NDCG@10"])
        p = wilcoxon_p(best_pu, all_pu[name]["NDCG@10"]) if name != best_key else 1.0
        sig[name] = {"CI_lo": ci_lo, "CI_hi": ci_hi, "p_vs_best": p}

    # ---- Per-group ----
    user_groups = assign_user_groups(user_feat)
    group_res = {
        name: evaluate_per_group(ranking, train_uv, test_uv, user_groups)
        for name, ranking in rankings.items()
    }

    return {
        "split_date": str(split.date()),
        "n_train": len(train), "n_test": len(test), "n_overlap": len(overlap),
        "best_variant": best_key,
        "results": all_agg,
        "venue_level": all_venue,
        "significance": sig,
        "group_results": group_res,
        "per_user": all_pu,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    wall_start = time.time()

    print("\n" + "=" * 70)
    print("  PHASE 6: COMPREHENSIVE VALIDATION v5")
    print("=" * 70)
    print()
    print("  Fixes applied:")
    print("    ✓ Feature leakage  — features computed from training data only")
    print("    ✓ NDCG correction  — IDCG uses total relevant candidates")
    print("    ✓ Significance     — bootstrap 95% CI + Wilcoxon signed-rank")
    print("    ✓ Per-group eval   — NDCG@10 by user behavioral segment")
    print()
    print("  New improvements:")
    print("    ✓ Temporal decay   — exponential edge weighting (λ=0.5)")
    print("    ✓ Selective social — high-confidence FSQ bridges only")
    print("    ✓ Item-KNN        — cosine similarity baseline")
    print("    ✓ IUF-popularity  — inverse user-frequency baseline")
    print()

    # ---- Load data ----
    print("  Loading data...")
    interactions = pd.read_csv(DATA_DIR / "coffee_interactions.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    print(f"    Interactions: {len(interactions):,}")

    social_df = None
    sp = DATA_DIR / "social_venue_signals.csv"
    if sp.exists():
        social_df = pd.read_csv(sp)
        print(f"    Social signals: {len(social_df):,} venues")
    else:
        print("    Social signals: not found — social variants disabled")

    # ---- Run all splits ----
    split_results = {}
    for sd in SPLIT_DATES:
        res = run_single_split(interactions, social_df, sd)
        if res:
            split_results[sd] = res

    primary = split_results.get(PRIMARY_SPLIT)
    if not primary:
        print("\n  ✗ Primary split failed. Aborting.")
        return

    # ==================================================================
    #  PRINT COMPREHENSIVE RESULTS
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"  RESULTS — PRIMARY SPLIT ({PRIMARY_SPLIT})")
    print(f"  Train: {primary['n_train']:,}  |  Test: {primary['n_test']:,}  "
          f"|  Overlap: {primary['n_overlap']:,}")
    print("=" * 70)

    # ---- A) Per-user re-ranking ----
    print("\n  A) PER-USER CANDIDATE RE-RANKING (corrected NDCG)")
    print("  " + "─" * 95)
    print(f"  {'Method':<26} {'NDCG@5':>8} {'NDCG@10':>8} {'NDCG@20':>8}"
          f" {'Hit@10':>8} {'ρ':>8} {'95% CI (NDCG@10)':>20} {'p-value':>10}")
    print("  " + "─" * 95)

    ordered = sorted(primary["results"],
                     key=lambda k: primary["results"][k]["NDCG@10"], reverse=True)
    for name in ordered:
        r  = primary["results"][name]
        s  = primary["significance"][name]
        vl = primary["venue_level"][name]
        ci = f"[{s['CI_lo']:.4f}, {s['CI_hi']:.4f}]"
        p  = f"{s['p_vs_best']:.2e}" if s["p_vs_best"] < 0.999 else "ref" if name == primary["best_variant"] else "ns"
        rho = f"{vl['Spearman_rho']:.4f}" if not np.isnan(vl.get("Spearman_rho", np.nan)) else "—"
        star = " ★" if name == primary["best_variant"] else ""
        print(f"  {name:<26} {r['NDCG@5']:>8.4f} {r['NDCG@10']:>8.4f} "
              f"{r['NDCG@20']:>8.4f} {r['Hit@10']:>8.4f} {rho:>8} "
              f"{ci:>20} {p:>10}{star}")

    # ---- B) Per-group ----
    groups_present = sorted(
        set(g for gr in primary["group_results"].values() for g in gr)
    )
    print(f"\n  B) PER-GROUP NDCG@10")
    print("  " + "─" * (28 + 22 * len(groups_present)))
    hdr = f"  {'Method':<26}" + "".join(f" {g:>20}" for g in groups_present)
    print(hdr)
    print("  " + "─" * (28 + 22 * len(groups_present)))
    for name in ordered:
        gr = primary["group_results"][name]
        vals = []
        for g in groups_present:
            if g in gr:
                vals.append(f"{gr[g]['mean_NDCG@10']:.4f} (n={gr[g]['n_users']:,})")
            else:
                vals.append("—")
        print(f"  {name:<26}" + "".join(f" {v:>20}" for v in vals))

    # ---- C) Robustness ----
    print(f"\n  C) ROBUSTNESS ACROSS SPLITS (NDCG@10)")
    print("  " + "─" * (28 + 16 * len(SPLIT_DATES)))
    hdr = f"  {'Method':<26}" + "".join(f" {sd:>14}" for sd in SPLIT_DATES)
    print(hdr)
    print("  " + "─" * (28 + 16 * len(SPLIT_DATES)))
    for name in ordered:
        vals = []
        for sd in SPLIT_DATES:
            if sd in split_results and name in split_results[sd]["results"]:
                vals.append(f"{split_results[sd]['results'][name]['NDCG@10']:.4f}")
            else:
                vals.append("—")
        print(f"  {name:<26}" + "".join(f" {v:>14}" for v in vals))

    # ---- D) Delta table (vs v3) ----
    v3_ndcg = primary["results"].get("v3_baseline", {}).get("NDCG@10", 0)
    print(f"\n  D) IMPROVEMENT OVER v3 BASELINE (NDCG@10 = {v3_ndcg:.4f})")
    print("  " + "─" * 55)
    print(f"  {'Variant':<26} {'NDCG@10':>8} {'Δ':>10} {'Δ%':>8} {'p-value':>10}")
    print("  " + "─" * 55)
    for name in ordered:
        r = primary["results"][name]
        delta = r["NDCG@10"] - v3_ndcg
        pct   = (delta / v3_ndcg * 100) if v3_ndcg > 0 else 0
        s     = primary["significance"][name]
        p     = f"{s['p_vs_best']:.2e}" if s["p_vs_best"] < 0.999 else "—"
        print(f"  {name:<26} {r['NDCG@10']:>8.4f} {delta:>+10.4f} {pct:>+7.2f}% {p:>10}")

    # ==================================================================
    #  SAVE OUTPUTS
    # ==================================================================

    # Main results
    rows = []
    for name in ordered:
        r = primary["results"][name]
        s = primary["significance"][name]
        vl = primary["venue_level"][name]
        rows.append({
            "method": name,
            **{k: v for k, v in r.items() if k != "n_users"},
            "n_users": r["n_users"],
            "Spearman_rho": vl.get("Spearman_rho", np.nan),
            "CI_lo": s["CI_lo"], "CI_hi": s["CI_hi"],
            "p_vs_best": s["p_vs_best"],
            "delta_NDCG@10": r["NDCG@10"] - v3_ndcg,
        })
    pd.DataFrame(rows).to_csv(DATA_DIR / "validation_v5_results.csv", index=False)

    # Per-group
    grow = []
    for name, gr in primary["group_results"].items():
        for g, vals in gr.items():
            grow.append({"method": name, "group": g, **vals})
    pd.DataFrame(grow).to_csv(DATA_DIR / "validation_v5_per_group.csv", index=False)

    # Robustness
    rrows = []
    for sd in SPLIT_DATES:
        if sd not in split_results:
            continue
        for name, r in split_results[sd]["results"].items():
            rrows.append({"split_date": sd, "method": name, "NDCG@10": r["NDCG@10"]})
    pd.DataFrame(rrows).to_csv(DATA_DIR / "validation_v5_robustness.csv", index=False)

    # Human-readable summary
    with open(DATA_DIR / "validation_v5_summary.txt", "w") as f:
        f.write("VALIDATION v5 SUMMARY\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Primary split: {PRIMARY_SPLIT}\n")
        f.write(f"Train: {primary['n_train']:,}  Test: {primary['n_test']:,}  "
                f"Overlap: {primary['n_overlap']:,}\n\n")
        f.write("FIXES APPLIED:\n")
        f.write("  1. Features recomputed from training data only (no leakage)\n")
        f.write("  2. NDCG IDCG corrected (uses total relevant candidates)\n")
        f.write("  3. Bootstrap 95% CI + Wilcoxon significance tests added\n")
        f.write("  4. Per-group evaluation added\n\n")
        f.write("IMPROVEMENTS:\n")
        f.write(f"  - Temporal decay: λ={DECAY_LAMBDA} (half-life ≈ {0.693/DECAY_LAMBDA:.1f} yr)\n")
        f.write(f"  - Selective social: confidence ≥ {SOCIAL_CONF_MIN}, γ={SOCIAL_GAMMA}\n")
        f.write(f"  - Item-KNN baseline: cosine, top-{KNN_NEIGHBORS} neighbors\n")
        f.write(f"  - IUF-popularity baseline\n\n")
        f.write(f"BEST VARIANT: {primary['best_variant']}\n")
        best_r = primary["results"][primary["best_variant"]]
        best_s = primary["significance"][primary["best_variant"]]
        f.write(f"  NDCG@10 = {best_r['NDCG@10']:.6f}  "
                f"95% CI [{best_s['CI_lo']:.4f}, {best_s['CI_hi']:.4f}]\n")
        delta = best_r["NDCG@10"] - v3_ndcg
        pct = (delta / v3_ndcg * 100) if v3_ndcg > 0 else 0
        f.write(f"  vs v3_baseline: {delta:+.6f} ({pct:+.2f}%)\n\n")

        f.write("PER-USER RE-RANKING RESULTS:\n")
        f.write(f"  {'Method':<26} {'NDCG@10':>8} {'Hit@10':>8} {'CI_lo':>8} {'CI_hi':>8} {'p':>10}\n")
        for name in ordered:
            r = primary["results"][name]
            s = primary["significance"][name]
            p = f"{s['p_vs_best']:.2e}" if s["p_vs_best"] < 0.999 else "ref"
            f.write(f"  {name:<26} {r['NDCG@10']:>8.4f} {r['Hit@10']:>8.4f} "
                    f"{s['CI_lo']:>8.4f} {s['CI_hi']:>8.4f} {p:>10}\n")

    elapsed = time.time() - wall_start
    print(f"\n  Saved: validation_v5_results.csv")
    print(f"  Saved: validation_v5_per_group.csv")
    print(f"  Saved: validation_v5_robustness.csv")
    print(f"  Saved: validation_v5_summary.txt")
    print(f"\n  Total wall time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
