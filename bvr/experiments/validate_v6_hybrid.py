"""
Phase 7: Hybrid BiRank + Matrix Factorization.

BiRank captures graph-structural quality (important users visit important venues).
ALS/BPR captures latent collaborative patterns (users who visit similar venues
share hidden preferences). Blending both should capture what each misses alone.

Approach:
  1. Train BiRank on training edges (v5_combined: decayed edges + selective social)
  2. Train ALS matrix factorization on the same user-venue interaction matrix
  3. For each venue, produce a blended score:
       hybrid_score = lambda * birank_score + (1 - lambda) * mf_score
  4. Grid-search lambda on a validation split (2019-07-01)
  5. Report final results on the test split (2020-01-01)

This uses a proper train/val/test protocol:
  - Train:      < 2019-07-01   (feature computation + model training)
  - Validation:   2019-07-01 — 2019-12-31  (lambda tuning)
  - Test:       >= 2020-01-01  (final reporting)

Additional variants tested:
  hybrid_als_XX       — BiRank + ALS at various lambda blends
  hybrid_bpr_XX       — BiRank + BPR at various lambda blends
  pure_als            — ALS alone (no BiRank)
  pure_bpr            — BPR alone (no BiRank)
  v5_combined         — BiRank alone (reproduced as reference)

Outputs:
  validation_v6_results.csv       — main results with CIs and p-values
  validation_v6_per_group.csv     — per-group breakdown
  validation_v6_lambda_tuning.csv — lambda search results on validation split
  validation_v6_summary.txt       — human-readable report
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
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent

# ---- Configuration ----
TRAIN_CUTOFF = "2019-07-01"    # Train: everything before this
VAL_CUTOFF   = "2020-01-01"    # Val: TRAIN_CUTOFF to VAL_CUTOFF; Test: after VAL_CUTOFF
K_VALUES     = (5, 10, 20)
DECAY_LAMBDA = 0.5
SOCIAL_GAMMA = 0.15
SOCIAL_CONF_MIN = 0.3
N_BOOTSTRAP  = 1000

# MF hyperparameters
MF_FACTORS    = 64
MF_ITERATIONS = 30
MF_REG        = 0.01

# Lambda search grid (weight on BiRank; 1-lambda on MF)
LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ============================================================================
# BIRANK (from validate_v5.py)
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
    if p0 is None: p0 = np.ones(nu) / nu
    if q0 is None: q0 = np.ones(nv) / nv
    p0 = p0 / p0.sum()
    q0 = q0 / q0.sum()
    rs = np.array(W.sum(axis=1)).flatten(); rs[rs == 0] = 1.0
    Su = sparse.diags(1.0 / rs) @ W
    cs = np.array(W.sum(axis=0)).flatten(); cs[cs == 0] = 1.0
    Sv = W @ sparse.diags(1.0 / cs)
    p, q = p0.copy(), q0.copy()
    for it in range(1, max_iter + 1):
        pp, qp = p.copy(), q.copy()
        p = alpha * (Su @ q) + (1 - alpha) * p0
        q = beta * (Sv.T @ p) + (1 - beta) * q0
        p /= p.sum(); q /= q.sum()
        if np.abs(p - pp).sum() < tol and np.abs(q - qp).sum() < tol:
            break
    return p, q


def _safe(val, default=0):
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


def corrected_burstiness(intervals):
    if len(intervals) < 2: return np.nan
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
        if total >= 2:
            ts = udf["timestamp"].sort_values()
            intervals = ts.diff().dt.total_seconds().dropna() / 86400
            bust = corrected_burstiness(intervals.values)
        else:
            bust = np.nan
        probs = vc / total
        entropy = float(-np.sum(probs * np.log2(probs))) if total > 1 else 0.0
        revisit = (total - unique) / total if total > 0 else 0
        rows.append({
            "user_id": uid, "total_visits": total,
            "unique_venues": unique, "revisit_ratio": revisit,
            "burstiness_index": bust, "venue_entropy": entropy,
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
            "business_id": bid, "repeat_user_rate": repeat_rate,
            "stability_cv": cv,
        })
    return pd.DataFrame(rows)


def build_decayed_edges(train, split_date, lam=DECAY_LAMBDA):
    t = train.copy()
    age_days = (pd.Timestamp(split_date) - t["timestamp"]).dt.total_seconds() / 86400.0
    t["decay_weight"] = np.exp(-lam * age_days / 365.0)
    return t.groupby(["user_id", "business_id"]).agg(weight=("decay_weight", "sum")).reset_index()


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


def selective_social_prior(base_q0, v2i, social_df, gamma=SOCIAL_GAMMA, conf_min=SOCIAL_CONF_MIN):
    soc = social_df[social_df["mean_bridge_confidence"] >= conf_min].copy()
    if len(soc) == 0: return base_q0.copy()
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


def build_birank_ranking(edges_df, user_feat, venue_feat, social_df=None):
    """Build v5_combined ranking (decayed edges + selective social)."""
    W, u2i, v2i, i2u, i2v = build_adjacency(edges_df)
    p0, q0 = behavioral_priors(user_feat, venue_feat, u2i, v2i)
    if social_df is not None:
        q0 = selective_social_prior(q0, v2i, social_df)
    _, q = birank(W, p0=p0, q0=q0)
    return {i2v[i]: float(q[i]) for i in range(len(i2v))}


# ============================================================================
# MATRIX FACTORIZATION
# ============================================================================

def build_mf_ranking(train, method="als"):
    """
    Train ALS or BPR on the user-venue interaction matrix.
    Returns {business_id: score} where score = mean of all user factors
    dotted with venue factor (global venue popularity in latent space).

    Also returns per-user-venue scoring function for personalized ranking.
    """
    # Build user-venue count matrix
    edge_df = train.groupby(["user_id", "business_id"]).size().reset_index(name="count")
    users = edge_df["user_id"].unique()
    venues = edge_df["business_id"].unique()
    u2i = {u: i for i, u in enumerate(users)}
    v2i = {v: i for i, v in enumerate(venues)}
    i2u = {i: u for u, i in u2i.items()}
    i2v = {i: v for v, i in v2i.items()}

    rows = [u2i[u] for u in edge_df["user_id"]]
    cols = [v2i[v] for v in edge_df["business_id"]]
    data = edge_df["count"].values.astype(np.float32)

    # implicit fit() takes user-item matrix (users x items)
    user_item = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(venues)))

    if method == "als":
        model = AlternatingLeastSquares(
            factors=MF_FACTORS, iterations=MF_ITERATIONS,
            regularization=MF_REG, random_state=42,
        )
    else:
        model = BayesianPersonalizedRanking(
            factors=MF_FACTORS, iterations=MF_ITERATIONS,
            regularization=MF_REG, random_state=42,
        )

    model.fit(user_item, show_progress=False)

    # Global venue score: norm of venue factor vector
    # (captures how "strong" a venue is in the latent space)
    venue_factors = model.item_factors  # shape: (n_venues, factors)
    # Use mean user vector as a "generic user" to score all venues
    mean_user = np.mean(model.user_factors, axis=0)
    venue_scores = venue_factors @ mean_user

    global_ranking = {i2v[i]: float(venue_scores[i]) for i in range(len(i2v))}

    # Also build per-user scoring for personalized re-ranking
    def user_venue_score(uid, vid):
        """Score a specific user-venue pair using learned factors."""
        if uid not in u2i or vid not in v2i:
            return 0.0
        return float(model.user_factors[u2i[uid]] @ model.item_factors[v2i[vid]])

    return global_ranking, user_venue_score, u2i, v2i


def build_hybrid_ranking(birank_scores, mf_scores, lam):
    """
    Blend BiRank and MF scores.
    hybrid = lambda * birank_norm + (1 - lambda) * mf_norm
    Both are min-max normalised to [0, 1] before blending.
    """
    all_venues = set(birank_scores) | set(mf_scores)

    # Min-max normalise each
    br_vals = np.array([birank_scores.get(v, 0) for v in all_venues])
    mf_vals = np.array([mf_scores.get(v, 0) for v in all_venues])

    br_min, br_max = br_vals.min(), br_vals.max()
    mf_min, mf_max = mf_vals.min(), mf_vals.max()

    br_range = br_max - br_min if br_max > br_min else 1.0
    mf_range = mf_max - mf_min if mf_max > mf_min else 1.0

    hybrid = {}
    for i, v in enumerate(all_venues):
        br_norm = (br_vals[i] - br_min) / br_range
        mf_norm = (mf_vals[i] - mf_min) / mf_range
        hybrid[v] = lam * br_norm + (1 - lam) * mf_norm

    return hybrid


def build_personalized_hybrid_ranking(birank_scores, mf_user_score_fn,
                                       mf_u2i, mf_v2i, lam):
    """
    For per-user re-ranking, use personalized MF scores instead of global.
    Returns a function: (uid, venue_list) -> {vid: hybrid_score}
    """
    # Normalise BiRank scores globally
    br_vals = list(birank_scores.values())
    br_min, br_max = min(br_vals), max(br_vals)
    br_range = br_max - br_min if br_max > br_min else 1.0

    def rank_user_venues(uid, venues):
        # Get BiRank normalised scores
        br_scores = {v: (birank_scores.get(v, 0) - br_min) / br_range for v in venues}

        # Get personalized MF scores
        if uid in mf_u2i:
            mf_raw = {v: mf_user_score_fn(uid, v) for v in venues}
            mf_vals = list(mf_raw.values())
            mf_min_l, mf_max_l = min(mf_vals), max(mf_vals)
            mf_range = mf_max_l - mf_min_l if mf_max_l > mf_min_l else 1.0
            mf_scores = {v: (mf_raw[v] - mf_min_l) / mf_range for v in venues}
        else:
            # User not in MF model — fall back to pure BiRank
            mf_scores = {v: 0.5 for v in venues}

        return {v: lam * br_scores[v] + (1 - lam) * mf_scores[v] for v in venues}

    return rank_user_venues


# ============================================================================
# EVALUATION (corrected, from validate_v5.py)
# ============================================================================

def ndcg_at_k(predicted_full, actual_set, k):
    rel_k = [1.0 if v in actual_set else 0.0 for v in predicted_full[:k]]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel_k))
    n_relevant = sum(1 for v in predicted_full if v in actual_set)
    n_ideal = min(k, n_relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(predicted, actual_set, k):
    return 1.0 if set(predicted[:k]) & actual_set else 0.0


def evaluate_per_user_global(ranking_dict, train_uv, test_uv):
    """Standard global-score re-ranking (same as v5)."""
    uid_list = []
    scores = {f"{m}@{k}": [] for m in ("NDCG", "Hit") for k in K_VALUES}
    for uid in test_uv:
        if uid not in train_uv: continue
        cands = sorted(train_uv[uid], key=lambda v: ranking_dict.get(v, 0), reverse=True)
        if len(cands) < 2: continue
        uid_list.append(uid)
        for k in K_VALUES:
            scores[f"NDCG@{k}"].append(ndcg_at_k(cands, test_uv[uid], k))
            scores[f"Hit@{k}"].append(hit_at_k(cands, test_uv[uid], k))
    per_user = {key: np.array(vals) for key, vals in scores.items()}
    agg = {key: float(arr.mean()) for key, arr in per_user.items()}
    agg["n_users"] = len(uid_list)
    return agg, per_user, uid_list


def evaluate_per_user_personalized(rank_fn, train_uv, test_uv):
    """Personalized re-ranking using per-user MF scores."""
    uid_list = []
    scores = {f"{m}@{k}": [] for m in ("NDCG", "Hit") for k in K_VALUES}
    for uid in test_uv:
        if uid not in train_uv: continue
        cands = list(train_uv[uid])
        if len(cands) < 2: continue
        # Get personalized scores for this user's candidates
        user_scores = rank_fn(uid, cands)
        cands_ranked = sorted(cands, key=lambda v: user_scores.get(v, 0), reverse=True)
        uid_list.append(uid)
        for k in K_VALUES:
            scores[f"NDCG@{k}"].append(ndcg_at_k(cands_ranked, test_uv[uid], k))
            scores[f"Hit@{k}"].append(hit_at_k(cands_ranked, test_uv[uid], k))
    per_user = {key: np.array(vals) for key, vals in scores.items()}
    agg = {key: float(arr.mean()) for key, arr in per_user.items()}
    agg["n_users"] = len(uid_list)
    return agg, per_user, uid_list


def bootstrap_ci(scores, n_boot=N_BOOTSTRAP, confidence=0.95):
    n = len(scores)
    rng = np.random.RandomState(42)
    means = [scores[rng.choice(n, n, replace=True)].mean() for _ in range(n_boot)]
    alpha = (1 - confidence) / 2
    return (float(np.percentile(means, 100 * alpha)),
            float(np.percentile(means, 100 * (1 - alpha))))


def wilcoxon_p(a, b):
    diff = a - b
    if np.all(diff == 0): return 1.0
    try:
        _, p = wilcoxon(diff, alternative="two-sided")
        return float(p)
    except: return 1.0


def assign_user_groups(user_feat):
    uf = user_feat.copy()
    uf["burstiness_index"] = uf["burstiness_index"].fillna(0)
    uf["group"] = "Casual Weekenders"
    uf.loc[uf["revisit_ratio"] >= 0.25, "group"] = "Loyalists"
    uf.loc[(uf["total_visits"] <= 2) & (uf["revisit_ratio"] < 0.25), "group"] = "Infrequent Visitors"
    uf.loc[(uf["total_visits"] >= 4) & (uf["revisit_ratio"] < 0.25), "group"] = "Weekday Regulars"
    return uf[["user_id", "group"]]


def evaluate_per_group(rank_fn_or_dict, train_uv, test_uv, user_groups,
                       personalized=False, k=10):
    gmap = dict(zip(user_groups["user_id"], user_groups["group"]))
    group_scores = defaultdict(list)
    for uid in test_uv:
        if uid not in train_uv: continue
        g = gmap.get(uid, "Unknown")
        cands = list(train_uv[uid])
        if len(cands) < 2: continue
        if personalized:
            user_scores = rank_fn_or_dict(uid, cands)
            cands_ranked = sorted(cands, key=lambda v: user_scores.get(v, 0), reverse=True)
        else:
            cands_ranked = sorted(cands, key=lambda v: rank_fn_or_dict.get(v, 0), reverse=True)
        group_scores[g].append(ndcg_at_k(cands_ranked, test_uv[uid], k))
    return {g: {"mean_NDCG@10": float(np.mean(s)), "std": float(np.std(s)), "n_users": len(s)}
            for g, s in group_scores.items()}


# ============================================================================
# MAIN
# ============================================================================

def main():
    wall_start = time.time()

    print("\n" + "=" * 70)
    print("  PHASE 7: HYBRID BIRANK + MATRIX FACTORIZATION")
    print("=" * 70)
    print()
    print(f"  Train:      < {TRAIN_CUTOFF}")
    print(f"  Validation:   {TRAIN_CUTOFF} — {VAL_CUTOFF}")
    print(f"  Test:       >= {VAL_CUTOFF}")
    print(f"  MF factors: {MF_FACTORS}, iterations: {MF_ITERATIONS}")
    print(f"  Lambda grid: {LAMBDA_GRID}")
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

    # ---- Split ----
    train_cutoff = pd.Timestamp(TRAIN_CUTOFF)
    val_cutoff   = pd.Timestamp(VAL_CUTOFF)

    train = interactions[interactions["timestamp"] < train_cutoff].copy()
    val   = interactions[(interactions["timestamp"] >= train_cutoff) &
                         (interactions["timestamp"] < val_cutoff)].copy()
    test  = interactions[interactions["timestamp"] >= val_cutoff].copy()

    print(f"\n    Train: {len(train):>10,}  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"    Val:   {len(val):>10,}  ({val['timestamp'].min().date()} → {val['timestamp'].max().date()})")
    print(f"    Test:  {len(test):>10,}  ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")

    # Ground truth for validation (vectorized with groupby)
    train_gb = train.dropna(subset=["user_id"]).groupby("user_id")["business_id"].apply(set).to_dict()
    val_gb   = val.dropna(subset=["user_id"]).groupby("user_id")["business_id"].apply(set).to_dict()
    test_gb  = test.dropna(subset=["user_id"]).groupby("user_id")["business_id"].apply(set).to_dict()

    val_overlap = set(train_gb.keys()) & set(val_gb.keys())
    val_train_uv = {uid: train_gb[uid] for uid in val_overlap if train_gb[uid]}
    val_test_uv  = {uid: val_gb[uid]   for uid in val_overlap if val_gb[uid]}
    print(f"    Val overlap users: {len(val_overlap):,}, with future visits: {len(val_test_uv):,}")

    # Ground truth for test
    test_overlap = set(train_gb.keys()) & set(test_gb.keys())
    test_train_uv = {uid: train_gb[uid] for uid in test_overlap if train_gb[uid]}
    test_test_uv  = {uid: test_gb[uid]  for uid in test_overlap if test_gb[uid]}
    print(f"    Test overlap users: {len(test_overlap):,}, with future visits: {len(test_test_uv):,}")

    # ---- Compute features from training data ----
    print("\n  Computing features from training data...")
    t0 = time.time()
    user_feat  = compute_user_features(train)
    venue_feat = compute_venue_features(train)
    print(f"    Done in {time.time()-t0:.0f}s")

    # ---- Build BiRank ----
    print("\n  Building BiRank (v5_combined)...")
    t0 = time.time()
    decay_edges = build_decayed_edges(train, TRAIN_CUTOFF)
    birank_scores = build_birank_ranking(decay_edges, user_feat, venue_feat, social_df)
    print(f"    Done in {time.time()-t0:.1f}s  ({len(birank_scores):,} venues)")

    # ---- Train ALS ----
    print("\n  Training ALS (factors={}, iter={})...".format(MF_FACTORS, MF_ITERATIONS))
    t0 = time.time()
    als_global, als_user_fn, als_u2i, als_v2i = build_mf_ranking(train, method="als")
    print(f"    Done in {time.time()-t0:.1f}s  ({len(als_global):,} venues)")

    # ---- Train BPR ----
    print("\n  Training BPR (factors={}, iter={})...".format(MF_FACTORS, MF_ITERATIONS))
    t0 = time.time()
    bpr_global, bpr_user_fn, bpr_u2i, bpr_v2i = build_mf_ranking(train, method="bpr")
    print(f"    Done in {time.time()-t0:.1f}s  ({len(bpr_global):,} venues)")

    # ==================================================================
    # LAMBDA TUNING ON VALIDATION SET
    # ==================================================================
    print("\n" + "=" * 70)
    print("  LAMBDA TUNING (on validation split)")
    print("=" * 70)

    tuning_rows = []

    for lam in LAMBDA_GRID:
        # ALS hybrid — personalized
        als_rank_fn = build_personalized_hybrid_ranking(
            birank_scores, als_user_fn, als_u2i, als_v2i, lam
        )
        als_res, _, _ = evaluate_per_user_personalized(als_rank_fn, val_train_uv, val_test_uv)

        # BPR hybrid — personalized
        bpr_rank_fn = build_personalized_hybrid_ranking(
            birank_scores, bpr_user_fn, bpr_u2i, bpr_v2i, lam
        )
        bpr_res, _, _ = evaluate_per_user_personalized(bpr_rank_fn, val_train_uv, val_test_uv)

        tuning_rows.append({
            "lambda": lam,
            "als_NDCG@10": als_res["NDCG@10"],
            "als_Hit@10": als_res["Hit@10"],
            "bpr_NDCG@10": bpr_res["NDCG@10"],
            "bpr_Hit@10": bpr_res["Hit@10"],
        })

        print(f"    λ={lam:.1f}  ALS: {als_res['NDCG@10']:.6f}  "
              f"BPR: {bpr_res['NDCG@10']:.6f}")

    tuning_df = pd.DataFrame(tuning_rows)

    # Find best lambda for each method
    best_als_row = tuning_df.loc[tuning_df["als_NDCG@10"].idxmax()]
    best_bpr_row = tuning_df.loc[tuning_df["bpr_NDCG@10"].idxmax()]
    best_als_lam = float(best_als_row["lambda"])
    best_bpr_lam = float(best_bpr_row["lambda"])

    print(f"\n    Best ALS λ = {best_als_lam:.1f} (val NDCG@10 = {best_als_row['als_NDCG@10']:.6f})")
    print(f"    Best BPR λ = {best_bpr_lam:.1f} (val NDCG@10 = {best_bpr_row['bpr_NDCG@10']:.6f})")

    # ==================================================================
    # FINAL EVALUATION ON TEST SET
    # ==================================================================
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION (test split >= 2020-01-01)")
    print("=" * 70)

    # Build all rankings for test evaluation
    print("\n  Building final rankings...")

    all_results = {}
    all_per_user = {}

    # v5_combined (BiRank only, lambda=1.0)
    res, pu, _ = evaluate_per_user_global(birank_scores, test_train_uv, test_test_uv)
    all_results["v5_combined (BiRank)"] = res
    all_per_user["v5_combined (BiRank)"] = pu

    # Pure ALS (lambda=0.0)
    als_pure_fn = build_personalized_hybrid_ranking(
        birank_scores, als_user_fn, als_u2i, als_v2i, 0.0
    )
    res, pu, _ = evaluate_per_user_personalized(als_pure_fn, test_train_uv, test_test_uv)
    all_results["pure_als"] = res
    all_per_user["pure_als"] = pu

    # Pure BPR (lambda=0.0)
    bpr_pure_fn = build_personalized_hybrid_ranking(
        birank_scores, bpr_user_fn, bpr_u2i, bpr_v2i, 0.0
    )
    res, pu, _ = evaluate_per_user_personalized(bpr_pure_fn, test_train_uv, test_test_uv)
    all_results["pure_bpr"] = res
    all_per_user["pure_bpr"] = pu

    # Best hybrid ALS
    als_best_fn = build_personalized_hybrid_ranking(
        birank_scores, als_user_fn, als_u2i, als_v2i, best_als_lam
    )
    res, pu, _ = evaluate_per_user_personalized(als_best_fn, test_train_uv, test_test_uv)
    all_results[f"hybrid_als (λ={best_als_lam})"] = res
    all_per_user[f"hybrid_als (λ={best_als_lam})"] = pu

    # Best hybrid BPR
    bpr_best_fn = build_personalized_hybrid_ranking(
        birank_scores, bpr_user_fn, bpr_u2i, bpr_v2i, best_bpr_lam
    )
    res, pu, _ = evaluate_per_user_personalized(bpr_best_fn, test_train_uv, test_test_uv)
    all_results[f"hybrid_bpr (λ={best_bpr_lam})"] = res
    all_per_user[f"hybrid_bpr (λ={best_bpr_lam})"] = pu

    # Also test a few fixed lambdas for the table
    for lam in [0.3, 0.5, 0.7]:
        if lam == best_als_lam: continue
        fn = build_personalized_hybrid_ranking(
            birank_scores, als_user_fn, als_u2i, als_v2i, lam
        )
        res, pu, _ = evaluate_per_user_personalized(fn, test_train_uv, test_test_uv)
        all_results[f"hybrid_als (λ={lam})"] = res
        all_per_user[f"hybrid_als (λ={lam})"] = pu

    # Random baseline
    rng = np.random.RandomState(42)
    all_v = list(set(train["business_id"].unique()))
    random_ranking = {v: rng.random() for v in all_v}
    res, pu, _ = evaluate_per_user_global(random_ranking, test_train_uv, test_test_uv)
    all_results["baseline_random"] = res
    all_per_user["baseline_random"] = pu

    # ---- Significance tests ----
    best_name = max(all_results, key=lambda k: all_results[k]["NDCG@10"])
    best_pu = all_per_user[best_name]["NDCG@10"]

    sig = {}
    for name in all_results:
        ci_lo, ci_hi = bootstrap_ci(all_per_user[name]["NDCG@10"])
        p = wilcoxon_p(best_pu, all_per_user[name]["NDCG@10"]) if name != best_name else 1.0
        sig[name] = {"CI_lo": ci_lo, "CI_hi": ci_hi, "p_vs_best": p}

    # ---- Per-group for best hybrid ----
    user_groups = assign_user_groups(user_feat)

    group_results = {}
    # BiRank groups
    group_results["v5_combined (BiRank)"] = evaluate_per_group(
        birank_scores, test_train_uv, test_test_uv, user_groups, personalized=False
    )
    # Best hybrid ALS groups
    group_results[f"hybrid_als (λ={best_als_lam})"] = evaluate_per_group(
        als_best_fn, test_train_uv, test_test_uv, user_groups, personalized=True
    )
    # Best hybrid BPR groups
    group_results[f"hybrid_bpr (λ={best_bpr_lam})"] = evaluate_per_group(
        bpr_best_fn, test_train_uv, test_test_uv, user_groups, personalized=True
    )
    # Pure ALS groups
    group_results["pure_als"] = evaluate_per_group(
        als_pure_fn, test_train_uv, test_test_uv, user_groups, personalized=True
    )

    # ==================================================================
    # PRINT RESULTS
    # ==================================================================
    birank_ndcg = all_results["v5_combined (BiRank)"]["NDCG@10"]

    print("\n  A) PER-USER CANDIDATE RE-RANKING (test split)")
    print("  " + "─" * 95)
    print(f"  {'Method':<28} {'NDCG@5':>8} {'NDCG@10':>8} {'NDCG@20':>8}"
          f" {'Hit@10':>8} {'95% CI (NDCG@10)':>20} {'p-value':>10}")
    print("  " + "─" * 95)

    ordered = sorted(all_results, key=lambda k: all_results[k]["NDCG@10"], reverse=True)
    for name in ordered:
        r = all_results[name]
        s = sig[name]
        ci = f"[{s['CI_lo']:.4f}, {s['CI_hi']:.4f}]"
        p = f"{s['p_vs_best']:.2e}" if s["p_vs_best"] < 0.999 else "ref" if name == best_name else "ns"
        star = " ★" if name == best_name else ""
        print(f"  {name:<28} {r['NDCG@5']:>8.4f} {r['NDCG@10']:>8.4f} "
              f"{r['NDCG@20']:>8.4f} {r['Hit@10']:>8.4f} "
              f"{ci:>20} {p:>10}{star}")

    # Delta table
    print(f"\n  B) IMPROVEMENT OVER BiRank (NDCG@10 = {birank_ndcg:.4f})")
    print("  " + "─" * 55)
    print(f"  {'Method':<28} {'NDCG@10':>8} {'Δ':>10} {'Δ%':>8}")
    print("  " + "─" * 55)
    for name in ordered:
        r = all_results[name]
        delta = r["NDCG@10"] - birank_ndcg
        pct = (delta / birank_ndcg * 100) if birank_ndcg > 0 else 0
        print(f"  {name:<28} {r['NDCG@10']:>8.4f} {delta:>+10.4f} {pct:>+7.2f}%")

    # Per-group
    groups_present = sorted(set(g for gr in group_results.values() for g in gr))
    print(f"\n  C) PER-GROUP NDCG@10")
    print("  " + "─" * (28 + 22 * len(groups_present)))
    hdr = f"  {'Method':<26}" + "".join(f" {g:>20}" for g in groups_present)
    print(hdr)
    print("  " + "─" * (28 + 22 * len(groups_present)))
    for name in group_results:
        gr = group_results[name]
        vals = []
        for g in groups_present:
            if g in gr:
                vals.append(f"{gr[g]['mean_NDCG@10']:.4f} (n={gr[g]['n_users']:,})")
            else:
                vals.append("—")
        print(f"  {name:<26}" + "".join(f" {v:>20}" for v in vals))

    # ==================================================================
    # SAVE
    # ==================================================================
    rows = []
    for name in ordered:
        r = all_results[name]
        s = sig[name]
        rows.append({
            "method": name,
            **{k: v for k, v in r.items() if k != "n_users"},
            "n_users": r["n_users"],
            "CI_lo": s["CI_lo"], "CI_hi": s["CI_hi"],
            "p_vs_best": s["p_vs_best"],
            "delta_NDCG@10": r["NDCG@10"] - birank_ndcg,
        })
    pd.DataFrame(rows).to_csv(DATA_DIR / "validation_v6_results.csv", index=False)

    tuning_df.to_csv(DATA_DIR / "validation_v6_lambda_tuning.csv", index=False)

    grow = []
    for name, gr in group_results.items():
        for g, vals in gr.items():
            grow.append({"method": name, "group": g, **vals})
    pd.DataFrame(grow).to_csv(DATA_DIR / "validation_v6_per_group.csv", index=False)

    # Summary
    with open(DATA_DIR / "validation_v6_summary.txt", "w") as f:
        f.write("VALIDATION v6 — HYBRID BIRANK + MATRIX FACTORIZATION\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Train: < {TRAIN_CUTOFF}  |  Val: {TRAIN_CUTOFF}—{VAL_CUTOFF}  |  Test: >= {VAL_CUTOFF}\n")
        f.write(f"MF: {MF_FACTORS} factors, {MF_ITERATIONS} iterations, reg={MF_REG}\n\n")
        f.write(f"Best ALS λ: {best_als_lam} (tuned on validation)\n")
        f.write(f"Best BPR λ: {best_bpr_lam} (tuned on validation)\n\n")
        f.write(f"BEST OVERALL: {best_name}\n")
        best_r = all_results[best_name]
        best_s = sig[best_name]
        f.write(f"  NDCG@10 = {best_r['NDCG@10']:.6f}  95% CI [{best_s['CI_lo']:.4f}, {best_s['CI_hi']:.4f}]\n")
        delta = best_r["NDCG@10"] - birank_ndcg
        pct = (delta / birank_ndcg * 100) if birank_ndcg > 0 else 0
        f.write(f"  vs BiRank: {delta:+.6f} ({pct:+.2f}%)\n\n")
        f.write("ALL RESULTS:\n")
        f.write(f"  {'Method':<28} {'NDCG@10':>8} {'Hit@10':>8} {'Δ vs BiRank':>12}\n")
        for name in ordered:
            r = all_results[name]
            d = r["NDCG@10"] - birank_ndcg
            f.write(f"  {name:<28} {r['NDCG@10']:>8.4f} {r['Hit@10']:>8.4f} {d:>+12.4f}\n")

    elapsed = time.time() - wall_start
    print(f"\n  Saved: validation_v6_results.csv")
    print(f"  Saved: validation_v6_lambda_tuning.csv")
    print(f"  Saved: validation_v6_per_group.csv")
    print(f"  Saved: validation_v6_summary.txt")
    print(f"\n  Total wall time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
