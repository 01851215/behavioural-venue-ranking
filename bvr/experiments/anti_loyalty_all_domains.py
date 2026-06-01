"""
Phase A1 — Anti-loyalty + ALS hybrid on all 5 domains.

Tests whether the anti-loyalty prior generalises beyond London + UK FSQ
to the original Yelp domains (coffee, restaurants, hotels).

Saves: data/results/anti_loyalty_5domains.csv
"""

import sys, warnings, time, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
warnings.filterwarnings("ignore")

from bvr.core.validation import (
    compute_user_features, compute_venue_features,
    build_decayed_edges, build_adjacency, birank,
    build_popularity_ranking, build_rating_ranking,
    evaluate_per_user,
)
from bvr.pipelines.london import (
    temporal_split, compute_rising_stars, spearman_rising,
    build_birank_explore, build_mf_ranking, blend_rankings,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
REPO     = Path(__file__).parent.parent.parent
YELP_DIR = Path("/Users/chris/Desktop/Master Project/yelp_dataset")

DOMAINS = {
    "coffee": {
        "file":  YELP_DIR / "coffee_interactions.csv",
        "split": "2020-01-01",
        "has_stars": False,
    },
    "restaurant": {
        "file":  DATA_DIR / "processed/restaurant_interactions.csv",
        "split": "2020-01-01",
        "has_stars": True,
    },
    "hotel": {
        "file":  DATA_DIR / "processed/hotel_interactions.csv",
        "split": "2020-01-01",
        "has_stars": True,
    },
    "london": {
        "file":  DATA_DIR / "results/london_interactions.csv",
        "split": "2018-01-01",
        "has_stars": True,
    },
    "uk_fsq": {
        "file":  DATA_DIR / "results/uk_fsq_interactions.csv",
        "split": "2013-07-01",
        "has_stars": False,
    },
}


def anti_loyalty_hybrid(train, decay_e, user_feat, venue_feat, r_als):
    W, u2i, v2i, i2u, i2v = build_adjacency(decay_e)
    rr_map = venue_feat.set_index("business_id")["repeat_user_rate"]
    p0 = np.ones(len(u2i))
    q0 = np.clip(np.array([
        1.0 / (float(rr_map.get(i2v[i], 0.01) or 0.01) + 0.01)
        for i in range(len(v2i))
    ]), 1e-10, None)
    _, q = birank(W, p0=p0, q0=q0)
    r_anti = {i2v[i]: float(q[i]) for i in range(len(v2i))}
    return blend_rankings(r_anti, r_als, lam=0.5)


def bootstrap_ci(arr, n=1000):
    np.random.seed(42)
    m = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run_domain(name, cfg):
    if not cfg["file"].exists():
        print(f"  {name}: file not found — {cfg['file']}")
        return None

    print(f"\n{'='*50}\n{name.upper()}\n{'='*50}")
    df = pd.read_csv(cfg["file"], dtype={"user_id": str, "business_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not cfg["has_stars"] or "stars" not in df.columns:
        df["stars"] = np.nan

    train, test_uv_rev, train_uv, _ = temporal_split(df, cfg["split"])
    print(f"  Train: {len(train):,} | Revisit users: {len(test_uv_rev):,}")
    if len(test_uv_rev) < 10:
        print(f"  Too few revisit users — skip")
        return None

    uf = compute_user_features(train)
    vf = compute_venue_features(train)
    decay_e = build_decayed_edges(train, cfg["split"], lam=0.5)
    test_full = df[df["timestamp"] >= pd.Timestamp(cfg["split"])].copy()
    rising = compute_rising_stars(train, test_full)

    # Build all rankings
    r_explore = build_birank_explore(decay_e, uf, vf)
    r_als     = build_mf_ranking(train, "als")
    r_hybrid_explore = blend_rankings(r_explore, r_als, lam=0.5)
    r_hybrid_anti    = anti_loyalty_hybrid(train, decay_e, uf, vf, r_als)
    r_pop     = build_popularity_ranking(train)
    np.random.seed(42)
    r_rand    = {v: float(np.random.random()) for v in train["business_id"].unique()}

    rows = []
    for mname, ranking in [
        ("hybrid_anti_loyalty_als", r_hybrid_anti),
        ("hybrid_explore_als",       r_hybrid_explore),
        ("baseline_popularity",      r_pop),
        ("baseline_random",          r_rand),
    ]:
        rho, p = spearman_rising(ranking, rising)
        agg, pu, _ = evaluate_per_user(ranking, train_uv, test_uv_rev)
        ndcg = agg.get("NDCG@10", 0)
        lo, hi = bootstrap_ci(pu["NDCG@10"]) if len(pu.get("NDCG@10", [])) > 5 else (0, 0)
        sig = "***" if p < 0.001 else ("*" if p < 0.05 else "ns")
        print(f"  {mname:<32} ρ={rho:+.4f}{sig}  NDCG={ndcg:.4f}  CI=[{lo:.4f},{hi:.4f}]")
        rows.append({
            "domain": name, "method": mname,
            "rho": round(rho, 4), "p_value": round(p, 4),
            "ndcg10": round(ndcg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        })
    return rows


if __name__ == "__main__":
    t0 = time.time()
    all_rows = []
    for name, cfg in DOMAINS.items():
        rows = run_domain(name, cfg)
        if rows:
            all_rows.extend(rows)

    df_out = pd.DataFrame(all_rows)
    out = REPO / "data/results/anti_loyalty_5domains.csv"
    df_out.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    print(f"Total: {time.time()-t0:.0f}s")

    # Pivot for easy reading
    pivot = df_out.pivot_table(index="method", columns="domain", values="rho", aggfunc="first")
    print("\n=== Rising-stars ρ across all 5 domains ===")
    print(pivot.round(4).to_string())
