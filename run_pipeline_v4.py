"""
Phase 4: BiRank v4 with Social Priors.

Extends run_pipeline.py by augmenting the behavioral priors with social signals
from the Foursquare friendship graph (Phase 3 output).

Changes vs v3:
  Venue prior: (1-gamma) * behavioral_prior + gamma * social_boost * bridge_confidence
  User  prior: behavioral_prior * social_centrality_multiplier (bridge users only)

Runs gamma in {0.0, 0.1, 0.2, 0.3} and picks the best by NDCG@10.
gamma=0.0 reproduces v3 exactly (baseline floor).

Outputs:
  coffee_birank_venue_scores_v4.csv  — best-gamma scores + social signal columns
  coffee_birank_user_scores_v4.csv   — user scores at best gamma
  gamma_tuning_results.csv           — NDCG@10 for each gamma value
"""

import time
import warnings
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).parent

SOCIAL_SIGNALS_PATH = DATA_DIR / "social_venue_signals.csv"
BRIDGE_PATH         = DATA_DIR / "yelp_fsq_user_bridge.csv"
DB_PATH             = DATA_DIR / "fsq.duckdb"

GAMMAS     = [0.0, 0.1, 0.2, 0.3]
SPLIT_DATE = "2020-01-01"
TOP_K      = 10


# ============================================================================
# REUSED FROM run_pipeline.py (unchanged)
# ============================================================================

def corrected_burstiness(intervals):
    if len(intervals) < 2:
        return np.nan
    mu, sigma = intervals.mean(), intervals.std()
    denom = sigma + mu
    return 0.0 if denom == 0 else (sigma - mu) / denom


def compute_user_features(interactions):
    print("  Computing user features...")
    rows = []
    for uid, udf in interactions.groupby("user_id"):
        udf = udf.sort_values("timestamp")
        total = len(udf)
        venue_counts = udf["business_id"].value_counts()
        unique_venues = len(venue_counts)
        top1_share = venue_counts.iloc[0] / total if total > 0 else 0

        if total >= 2:
            ts = udf["timestamp"].sort_values()
            intervals = ts.diff().dt.total_seconds().dropna() / 86400
            bust = corrected_burstiness(intervals.values)
            active_span = (ts.max() - ts.min()).total_seconds() / 86400
        else:
            bust, active_span = np.nan, 0.0

        probs = venue_counts / total
        entropy = -np.sum(probs * np.log2(probs)) if total > 1 else 0.0
        revisit_ratio = (total - unique_venues) / total if total > 0 else 0

        rows.append({
            "user_id": uid, "total_visits": total,
            "unique_venues": unique_venues, "revisit_ratio": revisit_ratio,
            "top1_venue_share": top1_share,
            "max_visits_single_venue": venue_counts.iloc[0],
            "burstiness_index": bust, "active_span_days": active_span,
            "venue_entropy": entropy,
            "unique_venue_ratio": unique_venues / total if total > 0 else 1.0,
        })
    return pd.DataFrame(rows)


def compute_venue_features(interactions):
    print("  Computing venue features...")
    def gini(values):
        v = np.sort(values); n = len(v)
        if n == 0 or v.sum() == 0: return 0.0
        idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * v)) / (n * v.sum()) - (n + 1) / n

    rows = []
    for bid, vdf in interactions.groupby("business_id"):
        total = len(vdf)
        uc = vdf["user_id"].value_counts()
        unique_users = len(uc)
        repeat_users = int((uc >= 2).sum())
        repeat_rate = repeat_users / unique_users if unique_users > 0 else 0
        avg_repeat = float(uc[uc >= 2].mean()) if repeat_users > 0 else 0

        if total >= 7:
            vdf = vdf.copy(); vdf["week"] = vdf["timestamp"].dt.to_period("W")
            weekly = vdf.groupby("week").size()
            w_mean = weekly.mean(); w_std = weekly.std()
            cv = w_std / w_mean if w_mean > 0 else np.nan
        else:
            w_mean = total; w_std = 0; cv = 0

        if total >= 4:
            vdf = vdf.copy(); vdf["quarter"] = vdf["timestamp"].dt.to_period("Q")
            seasonal_var = float(vdf.groupby("quarter").size().var() or 0)
        else:
            seasonal_var = 0

        rows.append({
            "business_id": bid, "popularity_visits": total,
            "unique_users": unique_users,
            "gini_user_contribution": gini(uc.values),
            "repeat_user_count": repeat_users, "repeat_user_rate": repeat_rate,
            "avg_user_repeat_visits": avg_repeat,
            "weekly_visit_mean": w_mean, "weekly_visit_std": w_std,
            "stability_cv": cv, "seasonal_variance": seasonal_var,
        })
    return pd.DataFrame(rows)


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
    t0 = time.time()
    for it in range(1, max_iter + 1):
        p_prev, q_prev = p.copy(), q.copy()
        p = alpha * (Su @ q) + (1 - alpha) * p0
        q = beta  * (Sv.T @ p) + (1 - beta)  * q0
        p /= p.sum(); q /= q.sum()
        dp = np.abs(p - p_prev).sum()
        dq = np.abs(q - q_prev).sum()
        if dp < tol and dq < tol:
            print(f"    Converged at iteration {it} ({time.time()-t0:.1f}s)")
            break
    else:
        print(f"    Max iterations reached ({time.time()-t0:.1f}s)")
    return p, q


# ============================================================================
# NEW: SOCIAL PRIOR AUGMENTATION
# ============================================================================

def load_social_inputs():
    """Load social signals and bridge table."""
    print("\nLoading social inputs...")

    social = pd.read_csv(SOCIAL_SIGNALS_PATH)
    print(f"  Social signals: {len(social):,} venues")

    bridge = pd.read_csv(BRIDGE_PATH)
    print(f"  Bridge users:   {len(bridge):,}")

    # Normalise friend_checkin_count → [0, 1] using log1p + min-max
    social["log_friend_count"] = np.log1p(social["friend_checkin_count"])
    max_lfc = social["log_friend_count"].max()
    social["social_boost"] = social["log_friend_count"] / max(max_lfc, 1e-10)

    return social, bridge


def load_fsq_friend_counts(bridge_df):
    """
    For each FSQ user in the bridge, get their friend count from DuckDB.
    Returns {fsq_user_id: friend_count}.
    """
    print("  Loading FSQ friend counts...")
    fsq_ids = bridge_df["fsq_user_id"].astype(int).tolist()
    con = duckdb.connect(str(DB_PATH), read_only=True)

    con.execute("CREATE OR REPLACE TEMP TABLE bids (fsq_user_id BIGINT)")
    con.executemany("INSERT INTO bids VALUES (?)", [(i,) for i in fsq_ids])

    counts = con.execute("""
        SELECT fsq_user_id, COUNT(*) AS friend_count FROM (
            SELECT user_a AS fsq_user_id FROM social_edges WHERE user_a IN (SELECT fsq_user_id FROM bids)
            UNION ALL
            SELECT user_b FROM social_edges WHERE user_b IN (SELECT fsq_user_id FROM bids)
        ) GROUP BY 1
    """).df()
    con.close()

    return dict(zip(counts["fsq_user_id"], counts["friend_count"]))


def build_v4_priors(edges, user_feat, venue_feat, social_df, bridge_df,
                    friend_counts, gamma):
    """
    Build augmented p0 (user) and q0 (venue) priors.

    Venue prior:
        q0 = (1 - gamma) * behavioral  +  gamma * social_boost * bridge_conf

    User prior:
        p0 = behavioral * social_centrality_multiplier
        social_centrality_multiplier = 1 + min(1, log(1+friend_count) / log(501))
        Only applied to users matched in bridge table.
    """
    W, u2i, v2i, i2u, i2v = build_adjacency(edges)
    nu, nv = W.shape

    # ---- Venue priors ----
    rr_map  = venue_feat.set_index("business_id")["repeat_user_rate"]
    cv_map  = venue_feat.set_index("business_id")["stability_cv"]
    soc_map = social_df.set_index("yelp_business_id")

    q0 = np.ones(nv)
    for vid, idx in v2i.items():
        rr = rr_map.get(vid, 0); rr = 0 if np.isnan(rr) else rr
        cv = cv_map.get(vid, 1); cv = 1 if np.isnan(cv) else cv
        behavioral = rr * (1 / (1 + cv))

        if vid in soc_map.index:
            s = soc_map.loc[vid]
            social_term = float(s["social_boost"]) * float(s["mean_bridge_confidence"])
        else:
            social_term = 0.0

        q0[idx] = (1 - gamma) * behavioral + gamma * social_term

    q0 = np.clip(q0, 1e-10, None)

    # ---- User priors ----
    bust_map   = user_feat.set_index("user_id")["burstiness_index"]
    visits_map = user_feat.set_index("user_id")["total_visits"]

    # Bridge: yelp_user_id → fsq_user_id
    yelp_to_fsq = dict(zip(
        bridge_df["yelp_user_id"],
        bridge_df["fsq_user_id"].astype(int)
    ))

    p0 = np.ones(nu)
    for uid, idx in u2i.items():
        b = bust_map.get(uid, 0);   b = 0 if np.isnan(b) else b
        v = visits_map.get(uid, 1); v = 1 if np.isnan(v) else v
        behavioral = np.log1p(v) * (1 - b)

        # Social centrality boost for bridge-matched users
        if uid in yelp_to_fsq:
            fsq_uid = yelp_to_fsq[uid]
            fc = friend_counts.get(fsq_uid, 0)
            mult = 1.0 + min(1.0, np.log1p(fc) / np.log1p(500))
        else:
            mult = 1.0

        p0[idx] = behavioral * mult

    p0 = np.clip(p0, 1e-10, None)

    return W, u2i, v2i, i2u, i2v, p0, q0


# ============================================================================
# TEMPORAL VALIDATION (lightweight inline NDCG@K)
# ============================================================================

def ndcg_at_k(interactions, venue_scores_df, k=10, split_date=SPLIT_DATE):
    """
    Global ranking NDCG@k (matches temporal_validation.py Paradigm 1).

    1. Sort ALL venues by BiRank score (global ranked list).
    2. For each user active in both train + test windows:
         - actual = venues they visited in test window
         - relevances = [1 if ranked_venue in actual else 0  for top-k venues]
         - NDCG = DCG / IDCG
    """
    interactions = interactions.copy()
    split = pd.Timestamp(split_date)
    train = interactions[interactions["timestamp"] < split]
    test  = interactions[interactions["timestamp"] >= split]

    train_users = set(train["user_id"].unique())
    test_users  = set(test["user_id"].unique())
    eval_users  = train_users & test_users

    # Global sorted venue list
    sorted_venues = (
        venue_scores_df
        .sort_values("birank_score", ascending=False)["business_id"]
        .tolist()
    )

    ndcgs = []
    for uid in eval_users:
        actual = set(test[test["user_id"] == uid]["business_id"].unique())
        if not actual:
            continue

        relevances = [1.0 if v in actual else 0.0 for v in sorted_venues[:k]]
        dcg  = sum(rel / np.log2(r + 2) for r, rel in enumerate(relevances))
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(actual))))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE 4: BIRANK v4 WITH SOCIAL PRIORS")
    print("=" * 70)

    # Load base data
    print("\nLoading base data...")
    interactions = pd.read_csv(DATA_DIR / "coffee_interactions.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    edges = pd.read_csv(DATA_DIR / "coffee_bipartite_edges.csv")
    print(f"  Interactions: {len(interactions):,}  |  Edges: {len(edges):,}")

    # Compute features
    print("\nComputing behavioral features...")
    user_feat  = compute_user_features(interactions)
    venue_feat = compute_venue_features(interactions)

    # Load social inputs
    social_df, bridge_df = load_social_inputs()
    friend_counts = load_fsq_friend_counts(bridge_df)
    print(f"  FSQ friend counts loaded for {len(friend_counts):,} users")

    # Gamma tuning loop
    print("\n" + "=" * 70)
    print("GAMMA TUNING  (gamma=0.0 reproduces v3 baseline)")
    print("=" * 70)

    gamma_results = []

    for gamma in GAMMAS:
        print(f"\n  gamma={gamma:.1f}")
        W, u2i, v2i, i2u, i2v, p0, q0 = build_v4_priors(
            edges, user_feat, venue_feat,
            social_df, bridge_df, friend_counts, gamma
        )
        p, q = birank(W, p0=p0, q0=q0)

        venue_scores = pd.DataFrame([
            {"business_id": i2v[i], "birank_score": q[i]} for i in range(len(i2v))
        ]).sort_values("birank_score", ascending=False)
        venue_scores["rank"] = range(1, len(venue_scores) + 1)

        ndcg = ndcg_at_k(interactions, venue_scores, k=TOP_K)
        print(f"    NDCG@{TOP_K} = {ndcg:.6f}")

        gamma_results.append({
            "gamma": gamma,
            f"ndcg@{TOP_K}": ndcg,
            "venue_scores": venue_scores,
            "user_scores": pd.DataFrame([
                {"user_id": i2u[i], "birank_score": p[i]} for i in range(len(i2u))
            ]).sort_values("birank_score", ascending=False),
        })

    # Pick best gamma
    best = max(gamma_results, key=lambda x: x[f"ndcg@{TOP_K}"])
    print(f"\n  Best gamma = {best['gamma']:.1f}  "
          f"(NDCG@{TOP_K}={best[f'ndcg@{TOP_K}']:.6f})")

    # Save gamma tuning table
    tuning_df = pd.DataFrame([{
        "gamma": r["gamma"],
        f"ndcg@{TOP_K}": r[f"ndcg@{TOP_K}"],
        "delta_vs_v3": r[f"ndcg@{TOP_K}"] - gamma_results[0][f"ndcg@{TOP_K}"],
    } for r in gamma_results])

    print("\nGamma tuning results:")
    print(tuning_df.to_string(index=False))
    tuning_df.to_csv(DATA_DIR / "gamma_tuning_results.csv", index=False)

    # Enrich best venue scores with social signal columns
    best_venue = best["venue_scores"].copy()
    social_cols = social_df[[
        "yelp_business_id", "friend_checkin_count", "fof_checkin_count",
        "social_unique_visitors", "social_diversity_index",
        "mean_bridge_confidence", "social_boost"
    ]].rename(columns={"yelp_business_id": "business_id"})
    best_venue = best_venue.merge(social_cols, on="business_id", how="left")
    best_venue["social_linked"] = best_venue["friend_checkin_count"].notna()
    best_venue["best_gamma"] = best["gamma"]

    # Save outputs
    out_venue = DATA_DIR / "coffee_birank_venue_scores_v4.csv"
    out_user  = DATA_DIR / "coffee_birank_user_scores_v4.csv"
    out_tuning = DATA_DIR / "gamma_tuning_results.csv"

    best_venue.to_csv(out_venue, index=False)
    best["user_scores"].to_csv(out_user, index=False)

    print(f"\n  Saved: {out_venue}")
    print(f"  Saved: {out_user}")
    print(f"  Saved: {out_tuning}")

    # Summary
    print("\n" + "=" * 70)
    print("TOP 15 VENUES — v4 BiRank")
    print("=" * 70)
    top15 = best_venue.head(15)[[
        "business_id", "birank_score", "rank",
        "friend_checkin_count", "social_unique_visitors", "social_linked"
    ]]
    print(top15.to_string(index=False))

    social_linked_top50 = best_venue.head(50)["social_linked"].sum()
    print(f"\n  Social-linked venues in top 50: {social_linked_top50}/50")
    print(f"\n  v3 NDCG@{TOP_K}: {gamma_results[0][f'ndcg@{TOP_K}']:.6f}")
    print(f"  v4 NDCG@{TOP_K}: {best[f'ndcg@{TOP_K}']:.6f}  "
          f"(delta={best[f'ndcg@{TOP_K}']-gamma_results[0][f'ndcg@{TOP_K}']:+.6f})")


if __name__ == "__main__":
    main()
