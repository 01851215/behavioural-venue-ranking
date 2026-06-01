"""
Behavioral Ranking Pipeline — Consolidated Master Script

Orchestrates the full behavioral model:
  Step 1-2: Load already-extracted Coffee & Tea data + interactions
  Step 3:   Recompute behavioral features (corrected burstiness formula)
  Step 4:   Bipartite graph (already built — reload edges)
  Step 5:   BiRank with behavioral priors injected
  Step 6:   Temporal validation (calls temporal_validation module)

Uses existing CSVs from prior pipeline runs.  Only recomputes what the
optimizations change (burstiness, priors, validation).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import sparse
import time, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).parent

# ============================================================================
# STEP 1–2: LOAD EXISTING DATA
# ============================================================================

def load_data():
    """Load pre-built CSVs (Steps 1-2 already done)."""
    print("\n" + "=" * 70)
    print("STEPS 1–2: LOADING EXISTING DATA")
    print("=" * 70)

    business = pd.read_csv(DATA_DIR / "business_coffee_v2.csv")
    print(f"  Venues:        {len(business):>8,}")

    interactions = pd.read_csv(DATA_DIR / "coffee_interactions.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    print(f"  Interactions:  {len(interactions):>8,}")

    edges = pd.read_csv(DATA_DIR / "coffee_bipartite_edges.csv")
    print(f"  Graph edges:   {len(edges):>8,}")

    return business, interactions, edges


# ============================================================================
# STEP 3: BEHAVIORAL FEATURE ENGINEERING (with corrected burstiness)
# ============================================================================

def corrected_burstiness(intervals):
    """
    Goh & Barabási burstiness:  B = (σ − μ) / (σ + μ)

    Range: [-1, +1]
      -1 = perfectly periodic (steady commuter)
       0 = Poisson random
      +1 = maximally bursty (tourist-like spikes)
    """
    if len(intervals) < 2:
        return np.nan
    mu = intervals.mean()
    sigma = intervals.std()
    denom = sigma + mu
    if denom == 0:
        return 0.0
    return (sigma - mu) / denom


def compute_user_features(interactions):
    """Recompute user-level features with corrected burstiness."""
    print("\n" + "=" * 70)
    print("STEP 3a: USER BEHAVIORAL FEATURES (corrected burstiness)")
    print("=" * 70)

    user_groups = interactions.groupby("user_id")
    rows = []

    for uid, udf in user_groups:
        udf = udf.sort_values("timestamp")
        total = len(udf)
        venue_counts = udf["business_id"].value_counts()
        unique_venues = len(venue_counts)

        # — Commitment —
        top1_share = venue_counts.iloc[0] / total if total > 0 else 0
        max_visits_single = venue_counts.iloc[0] if len(venue_counts) > 0 else 0

        # — Burstiness (corrected) —
        if total >= 2:
            ts = udf["timestamp"].sort_values()
            intervals = ts.diff().dt.total_seconds().dropna() / 86400  # days
            bust = corrected_burstiness(intervals.values)
            active_span = (ts.max() - ts.min()).total_seconds() / 86400
        else:
            bust = np.nan
            active_span = 0.0

        # — Diversity / Entropy —
        if total > 1:
            probs = venue_counts / total
            entropy = -np.sum(probs * np.log2(probs))
            unique_ratio = unique_venues / total
        else:
            entropy = 0.0
            unique_ratio = 1.0

        # — Temporal —
        revisit_ratio = (total - unique_venues) / total if total > 0 else 0

        rows.append({
            "user_id": uid,
            "total_visits": total,
            "unique_venues": unique_venues,
            "revisit_ratio": revisit_ratio,
            "top1_venue_share": top1_share,
            "max_visits_single_venue": max_visits_single,
            "burstiness_index": bust,
            "active_span_days": active_span,
            "venue_entropy": entropy,
            "unique_venue_ratio": unique_ratio,
        })

    user_feat = pd.DataFrame(rows)

    # Sanity: burstiness should be in [-1, 1]
    valid_b = user_feat["burstiness_index"].dropna()
    assert valid_b.between(-1, 1).all(), "Burstiness out of range!"
    print(f"  Users:      {len(user_feat):,}")
    print(f"  Burstiness  min={valid_b.min():.4f}  max={valid_b.max():.4f}  "
          f"mean={valid_b.mean():.4f}")
    return user_feat


def compute_venue_features(interactions):
    """Venue-level features (unchanged from v2, but consolidated here)."""
    print("\n" + "=" * 70)
    print("STEP 3b: VENUE BEHAVIORAL FEATURES")
    print("=" * 70)

    def gini(values):
        v = np.sort(values)
        n = len(v)
        if n == 0 or v.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * v)) / (n * v.sum()) - (n + 1) / n

    venue_groups = interactions.groupby("business_id")
    rows = []

    for bid, vdf in venue_groups:
        total = len(vdf)
        user_counts = vdf["user_id"].value_counts()
        unique_users = len(user_counts)

        # Gini
        gini_val = gini(user_counts.values) if unique_users > 0 else 0

        # Repeat users
        repeat_users = int((user_counts >= 2).sum())
        repeat_rate = repeat_users / unique_users if unique_users > 0 else 0
        avg_repeat = float(user_counts[user_counts >= 2].mean()) if repeat_users > 0 else 0

        # Temporal stability (weekly CV)
        if total >= 7:
            vdf = vdf.copy()
            vdf["week"] = vdf["timestamp"].dt.to_period("W")
            weekly = vdf.groupby("week").size()
            w_mean = weekly.mean()
            w_std = weekly.std()
            cv = w_std / w_mean if w_mean > 0 else np.nan
        else:
            w_mean, w_std, cv = total, 0, 0

        # Seasonal variance (quarterly)
        if total >= 4:
            vdf_copy = vdf.copy() if "week" not in vdf.columns else vdf
            vdf_copy["quarter"] = vdf_copy["timestamp"].dt.to_period("Q")
            qv = vdf_copy.groupby("quarter").size()
            seasonal_var = float(qv.var()) if len(qv) >= 2 else 0
        else:
            seasonal_var = 0

        rows.append({
            "business_id": bid,
            "popularity_visits": total,
            "unique_users": unique_users,
            "gini_user_contribution": gini_val,
            "repeat_user_count": repeat_users,
            "repeat_user_rate": repeat_rate,
            "avg_user_repeat_visits": avg_repeat,
            "weekly_visit_mean": w_mean,
            "weekly_visit_std": w_std,
            "stability_cv": cv,
            "seasonal_variance": seasonal_var,
        })

    venue_feat = pd.DataFrame(rows)
    print(f"  Venues:     {len(venue_feat):,}")
    return venue_feat


# ============================================================================
# STEP 4–5: BIRANK WITH BEHAVIORAL PRIORS
# ============================================================================

def build_adjacency(edges):
    """Build sparse adjacency matrix from edge list."""
    users = edges["user_id"].unique()
    venues = edges["business_id"].unique()
    u2i = {u: i for i, u in enumerate(users)}
    v2i = {v: i for i, v in enumerate(venues)}
    i2u = {i: u for u, i in u2i.items()}
    i2v = {i: v for v, i in v2i.items()}

    rows = [u2i[u] for u in edges["user_id"]]
    cols = [v2i[v] for v in edges["business_id"]]
    weights = edges["weight"].values.astype(np.float64)

    W = sparse.csr_matrix((weights, (rows, cols)),
                          shape=(len(users), len(venues)))
    return W, u2i, v2i, i2u, i2v


def birank(W, p0=None, q0=None, alpha=0.85, beta=0.85,
           max_iter=200, tol=1e-8):
    """
    BiRank with optional behavioral priors.

    p_{t+1} = α · S_u · q_t  +  (1−α) · p_0
    q_{t+1} = β · S_v^T · p_{t+1}  +  (1−β) · q_0
    """
    nu, nv = W.shape

    if p0 is None:
        p0 = np.ones(nu) / nu
    if q0 is None:
        q0 = np.ones(nv) / nv

    # Normalize p0, q0
    p0 = p0 / p0.sum()
    q0 = q0 / q0.sum()

    # Row-normalize  (user → venue)
    rs = np.array(W.sum(axis=1)).flatten()
    rs[rs == 0] = 1.0
    Su = sparse.diags(1.0 / rs) @ W

    # Col-normalize  (venue → user)
    cs = np.array(W.sum(axis=0)).flatten()
    cs[cs == 0] = 1.0
    Sv = W @ sparse.diags(1.0 / cs)

    p, q = p0.copy(), q0.copy()
    log = []

    t0 = time.time()
    for it in range(1, max_iter + 1):
        p_prev, q_prev = p.copy(), q.copy()
        p = alpha * (Su @ q) + (1 - alpha) * p0
        q = beta * (Sv.T @ p) + (1 - beta) * q0
        p /= p.sum()
        q /= q.sum()

        dp = np.abs(p - p_prev).sum()
        dq = np.abs(q - q_prev).sum()
        log.append((it, dp, dq))

        if dp < tol and dq < tol:
            print(f"  Converged at iteration {it}  ({time.time()-t0:.1f}s)")
            break
    else:
        print(f"  Reached max iterations ({max_iter})  ({time.time()-t0:.1f}s)")

    return p, q, log


def run_birank_with_priors(edges, user_feat, venue_feat):
    """
    Step 5: Inject behavioral priors into BiRank.

    User prior  = log(1 + total_visits) × (1 − burstiness)
       → steady, high-frequency users get higher priors

    Venue prior = repeat_user_rate × 1/(1 + stability_cv)
       → venues with loyal, steady traffic get higher priors
    """
    print("\n" + "=" * 70)
    print("STEPS 4–5: BIRANK WITH BEHAVIORAL PRIORS")
    print("=" * 70)

    W, u2i, v2i, i2u, i2v = build_adjacency(edges)
    nu, nv = W.shape
    print(f"  Graph: {nu:,} users × {nv:,} venues  |  {W.nnz:,} edges")

    # ---- User priors ----
    p0 = np.ones(nu)
    bust_map = user_feat.set_index("user_id")["burstiness_index"]
    visits_map = user_feat.set_index("user_id")["total_visits"]
    for uid, idx in u2i.items():
        b = bust_map.get(uid, 0)
        v = visits_map.get(uid, 1)
        b = b if not np.isnan(b) else 0
        p0[idx] = np.log1p(v) * (1 - b)
    p0 = np.clip(p0, 1e-10, None)

    # ---- Venue priors ----
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

    # ---- Run BiRank ----
    p, q, log = birank(W, p0=p0, q0=q0)

    # ---- Package results ----
    venue_scores = pd.DataFrame([
        {"business_id": i2v[i], "birank_score": q[i]} for i in range(nv)
    ]).sort_values("birank_score", ascending=False)
    venue_scores["rank"] = range(1, len(venue_scores) + 1)

    user_scores = pd.DataFrame([
        {"user_id": i2u[i], "birank_score": p[i]} for i in range(nu)
    ]).sort_values("birank_score", ascending=False)

    print(f"\n  Top 5 venues by BiRank:")
    for _, row in venue_scores.head(5).iterrows():
        print(f"    #{int(row['rank']):>4d}  {row['business_id'][:12]}…  "
              f"score={row['birank_score']:.6e}")

    return venue_scores, user_scores, W, u2i, v2i, i2u, i2v


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║   BEHAVIORAL RANKING MODEL — CONSOLIDATED PIPELINE              ║")
    print("╚" + "═" * 68 + "╝")

    # Steps 1–2
    business, interactions, edges = load_data()

    # Step 3
    user_feat = compute_user_features(interactions)
    venue_feat = compute_venue_features(interactions)

    # Save updated features
    user_feat.to_csv(DATA_DIR / "coffee_user_features_v3.csv", index=False)
    venue_feat.to_csv(DATA_DIR / "coffee_venue_features_v3.csv", index=False)
    print(f"\n  ✓ Saved coffee_user_features_v3.csv")
    print(f"  ✓ Saved coffee_venue_features_v3.csv")

    # Steps 4–5
    venue_scores, user_scores, W, u2i, v2i, i2u, i2v = \
        run_birank_with_priors(edges, user_feat, venue_feat)

    venue_scores.to_csv(DATA_DIR / "coffee_birank_venue_scores_v3.csv", index=False)
    user_scores.to_csv(DATA_DIR / "coffee_birank_user_scores_v3.csv", index=False)
    print(f"\n  ✓ Saved coffee_birank_venue_scores_v3.csv")
    print(f"  ✓ Saved coffee_birank_user_scores_v3.csv")

    # Step 6 — Temporal Validation
    print("\n" + "=" * 70)
    print("STEP 6: TEMPORAL VALIDATION")
    print("=" * 70)

    from temporal_validation import run_validation
    val_results = run_validation(
        interactions, venue_scores, user_feat, venue_feat,
        edges, birank_fn=birank, build_adj_fn=build_adjacency
    )

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║   PIPELINE COMPLETE                                             ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  Output files in: {DATA_DIR}")


if __name__ == "__main__":
    main()
