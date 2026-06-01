"""
UK Foursquare Check-in Pipeline — BiRank + MF + Hybrid Validation

Data source: Foursquare WWW2019 GB check-ins (Apr 2012 – Jan 2014)
  288,389 check-ins · 6,733 users · 70,042 venues across the whole UK

Key difference from run_london_pipeline.py:
  - No star ratings (check-in only) → rating baseline returns empty dict
  - Split date shifted to 2013-07-01 (65 / 35 of the 22-month window)
  - Venues span all of GB (not just London restaurants)

Methods compared (same as London pipeline):
  birank_count       — standard BiRank, count edges, loyalty priors
  birank_decay       — BiRank, decayed edges, loyalty priors
  birank_explore     — BiRank, decayed edges, exploration priors
  mf_als             — ALS matrix factorization
  mf_bpr             — BPR matrix factorization
  hybrid_explore_als — 0.5 * birank_explore + 0.5 * mf_als
  baseline_popularity — review count per venue (no rating baseline)

Outputs:
  uk_fsq_user_features.csv
  uk_fsq_venue_features.csv
  uk_fsq_venue_scores.csv
  uk_fsq_validation_summary.txt
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.stats import wilcoxon

from bvr.core.validation import (
    corrected_burstiness,
    compute_user_features,
    compute_venue_features,
    build_count_edges,
    build_decayed_edges,
    behavioral_priors,
    build_birank_ranking,
    build_popularity_ranking,
    birank,
    ndcg_at_k,
    evaluate_per_user,
    bootstrap_ci,
    build_adjacency,
)
# Reuse helpers from London pipeline
from bvr.pipelines.london import (
    temporal_split,
    compute_rising_stars,
    spearman_rising,
    spearman_venue,
    exploration_priors,
    build_birank_explore,
    build_mf_ranking,
    blend_rankings,
    evaluate_all,
)

warnings.filterwarnings("ignore")

DATA_DIR   = Path(__file__).parent
SPLIT_DATE = "2013-07-01"   # ~65 / 35 of Apr 2012 – Jan 2014
DECAY_LAM  = 0.5
K_VALUES   = (5, 10, 20)


# ============================================================================
# Data loading
# ============================================================================

def load_uk_fsq_data() -> pd.DataFrame:
    path = DATA_DIR / "uk_fsq_interactions.csv"
    if not path.exists():
        raise FileNotFoundError(
            "uk_fsq_interactions.csv not found. Run extract_uk_fsq.py first."
        )
    print("Loading UK Foursquare check-ins...")
    df = pd.read_csv(path, dtype={"user_id": str, "business_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["stars"] = np.nan   # no ratings in FSQ check-in data
    print(f"  {len(df):,} check-ins  |  "
          f"{df['user_id'].nunique():,} users  |  "
          f"{df['business_id'].nunique():,} venues")
    return df


# ============================================================================
# Report
# ============================================================================

def print_results(results: dict, train: pd.DataFrame) -> str:
    lines = []
    sep = "=" * 90
    lines.append(sep)
    lines.append("UK FOURSQUARE BIRANK VALIDATION — Exploration priors + MF hybrid")
    lines.append(f"Data: FSQ WWW2019 GB check-ins (Apr 2012 – Jan 2014)  |  "
                 f"Split: {SPLIT_DATE}")
    lines.append(f"Train: {len(train):,}  |  Venues: {train['business_id'].nunique():,}  |  "
                 f"Users: {train['user_id'].nunique():,}")
    lines.append("No star ratings — check-in frequency only (rating baseline omitted)")
    lines.append(sep)

    lines.append(f"\n  PRIMARY — Rising Stars residual correlation (popularity-debiased)")
    lines.append(f"  {'Method':<28} {'ρ (rising)':>12} {'p-value':>12}")
    lines.append("  " + "-" * 55)
    for name, data in results.items():
        rho  = data["rising_rho"]
        pval = data["rising_p"]
        sig  = " ***" if pval < 0.001 else (" *" if pval < 0.05 else "")
        lines.append(f"  {name:<28} {rho:>+12.4f} {pval:>12.4f}{sig}")

    lines.append(f"\n  SECONDARY — Raw test-traffic Spearman ρ")
    lines.append(f"  {'Method':<28} {'ρ (traffic)':>12} {'p-value':>12}")
    lines.append("  " + "-" * 55)
    for name, data in results.items():
        rho  = data["traffic_rho"]
        pval = data["traffic_p"]
        sig  = " ***" if pval < 0.001 else (" *" if pval < 0.05 else "")
        lines.append(f"  {name:<28} {rho:>+12.4f} {pval:>12.4f}{sig}")

    if any(r["agg"] for r in results.values()):
        lines.append(f"\n  TERTIARY — NDCG@10 on revisit subset")
        lines.append(f"  {'Method':<28} {'NDCG@10':>10} {'Hit@10':>10}")
        lines.append("  " + "-" * 52)
        for name, data in results.items():
            agg = data["agg"]
            lines.append(f"  {name:<28} {agg.get('NDCG@10', 0):>10.4f} {agg.get('Hit@10', 0):>10.4f}")

    all_rs = {n: d.get("rising_rho", -99) for n, d in results.items()}
    winner = max(all_rs, key=all_rs.get)
    pop_rho = results.get("baseline_popularity", {}).get("rising_rho", 0)

    lines.append(f"\n  HEADLINE — Rising-stars ρ (value added beyond popularity baseline)")
    lines.append(f"  {'Method':<28} {'ρ (rising)':>12}  Note")
    lines.append("  " + "-" * 65)
    for name, data in results.items():
        note = ""
        if name == "baseline_popularity":
            note = "reference (0 by construction)"
        elif name == "birank_count":
            note = "loyalty priors, count edges"
        elif name == "birank_decay":
            note = "loyalty priors, decayed edges"
        elif name == "birank_explore":
            note = "exploration priors"
        elif name == "mf_als":
            note = "ALS latent factors"
        elif name == "mf_bpr":
            note = "BPR latent factors"
        elif name == "hybrid_explore_als":
            note = "*** WINNER CANDIDATE: explore + ALS ***"
        lines.append(f"  {name:<28} {data.get('rising_rho', 0):>+12.4f}  {note}")

    lines.append(f"\n  Winner: {winner}  (ρ = {all_rs[winner]:+.4f})")
    hybrid_rho = results.get("hybrid_explore_als", {}).get("rising_rho", -99)
    if hybrid_rho > pop_rho:
        lines.append(f"  Hybrid beats popularity baseline by Δρ = {hybrid_rho - pop_rho:+.4f}")
    lines.append(sep)

    report = "\n".join(lines)
    print(report)
    return report


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    t0 = time.time()

    df = load_uk_fsq_data()

    train, test_uv_revisit, train_uv, test_traffic = temporal_split(df, SPLIT_DATE)

    print("\nComputing user behavioral features...")
    user_feat = compute_user_features(train)
    user_feat.to_csv(DATA_DIR / "uk_fsq_user_features.csv", index=False)
    print(f"  {len(user_feat):,} users  →  uk_fsq_user_features.csv")

    print("\nComputing venue behavioral features...")
    venue_feat = compute_venue_features(train)
    venue_feat.to_csv(DATA_DIR / "uk_fsq_venue_features.csv", index=False)
    print(f"  {len(venue_feat):,} venues  →  uk_fsq_venue_features.csv")

    print("\nBuilding graph edges...")
    count_edges   = build_count_edges(train)
    decayed_edges = build_decayed_edges(train, SPLIT_DATE, lam=DECAY_LAM)
    print(f"  Count: {len(count_edges):,}  |  Decayed: {len(decayed_edges):,}")

    print("\nRunning methods...")
    rankings = {}

    rankings["birank_count"]   = build_birank_ranking(count_edges,   user_feat, venue_feat)
    print("  ✓ birank_count")

    rankings["birank_decay"]   = build_birank_ranking(decayed_edges, user_feat, venue_feat)
    print("  ✓ birank_decay")

    rankings["birank_explore"] = build_birank_explore(decayed_edges, user_feat, venue_feat)
    print("  ✓ birank_explore")

    print("  Training ALS...")
    rankings["mf_als"] = build_mf_ranking(train, method="als")
    print("  ✓ mf_als")

    print("  Training BPR...")
    rankings["mf_bpr"] = build_mf_ranking(train, method="bpr")
    print("  ✓ mf_bpr")

    rankings["hybrid_explore_als"] = blend_rankings(
        rankings["birank_explore"], rankings["mf_als"], lam=0.5
    )
    print("  ✓ hybrid_explore_als")

    rankings["baseline_popularity"] = build_popularity_ranking(train)
    print("  ✓ baseline_popularity  (no rating baseline — FSQ has no stars)")

    # Anti-loyalty + ALS hybrid
    from bvr.core.validation import build_adjacency, birank as _birank
    import numpy as _np
    W_al, u2i_al, v2i_al, _, i2v_al = build_adjacency(decayed_edges)
    rr_map_al = venue_feat.set_index("business_id")["repeat_user_rate"]
    p0_al = _np.ones(len(u2i_al))
    q0_al = _np.clip(_np.array([
        1.0 / (float(rr_map_al.get(i2v_al[i], 0.01) or 0.01) + 0.01)
        for i in range(len(v2i_al))
    ]), 1e-10, None)
    _, q_al = _birank(W_al, p0=p0_al, q0=q0_al)
    r_anti = {i2v_al[i]: float(q_al[i]) for i in range(len(v2i_al))}
    rankings["hybrid_anti_loyalty_als"] = blend_rankings(r_anti, rankings["mf_als"], lam=0.5)
    print("  ✓ hybrid_anti_loyalty_als  (anti-loyalty prior + ALS)")

    print("  Training LightGCN (3 layers, 64 dim, 50 epochs, MPS)...")
    from lightgcn import build_lightgcn_ranking
    rankings["lightgcn"] = build_lightgcn_ranking(train, verbose=True)
    print("  ✓ lightgcn             (3-layer graph convolution, BPR loss)")

    print("\nComputing rising-stars residuals...")
    test_full = df[df["timestamp"] >= pd.Timestamp(SPLIT_DATE)].copy()
    rising_stars = compute_rising_stars(train, test_full)
    print(f"  Rising-stars for {len(rising_stars):,} venues")

    print("\nEvaluating...")
    results = evaluate_all(rankings, train_uv, test_uv_revisit, test_traffic, rising_stars)

    report = print_results(results, train)
    out_summary = DATA_DIR / "uk_fsq_validation_summary.txt"
    out_summary.write_text(report)
    print(f"\nSaved → {out_summary}")

    # Save venue scores (hybrid is best candidate from London results)
    best_ranking = rankings.get("hybrid_explore_als", rankings["birank_decay"])
    biz = pd.read_csv(DATA_DIR / "uk_fsq_businesses.csv", dtype={"business_id": str})
    venue_scores = pd.DataFrame([
        {"business_id": vid, "birank_score": score, "rank": rank + 1}
        for rank, (vid, score) in enumerate(
            sorted(best_ranking.items(), key=lambda x: x[1], reverse=True)
        )
    ])
    venue_scores["business_id"] = venue_scores["business_id"].astype(str)
    venue_scores = venue_scores.merge(
        biz[["business_id", "name", "lat", "lon", "category"]],
        on="business_id", how="left"
    )
    venue_scores.to_csv(DATA_DIR / "uk_fsq_venue_scores.csv", index=False)
    print(f"Saved → uk_fsq_venue_scores.csv  ({len(venue_scores):,} venues)")

    print(f"\nTotal runtime: {time.time() - t0:.0f}s")
