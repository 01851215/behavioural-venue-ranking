"""
EmergeRec — Functional Benchmark Runner

Usage:
    python -m emergerec.benchmark --datasets uk_foursquare,tripadvisor_london --models all

    from emergerec.benchmark import run
    results = run("uk_foursquare")
    print(results)

Paper target: NeurIPS Datasets & Benchmarks 2028
"""

import sys, warnings, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from emergerec.datasets.registry import DATASETS, get_dataset
from bvr.core.validation import (
    compute_user_features, compute_venue_features,
    build_count_edges, build_decayed_edges, build_adjacency, birank,
    build_birank_ranking, build_popularity_ranking, build_rating_ranking,
    evaluate_per_user, bootstrap_ci,
)
from bvr.pipelines.london import (
    temporal_split, compute_rising_stars, spearman_rising,
    build_birank_explore, build_mf_ranking, blend_rankings,
)

SUPPORTED_MODELS = [
    "baseline_popularity",
    "baseline_random",
    "birank_loyalty",
    "birank_explore",
    "hybrid_explore_als",
    "hybrid_anti_loyalty_als",
]

SUPPORTED_METRICS = ["rising_stars_rho", "ndcg@10", "hit@10"]


def _build_anti_loyalty(decay_e, venue_feat):
    from bvr.core.validation import build_adjacency, birank
    W, u2i, v2i, _, i2v = build_adjacency(decay_e)
    rr_map = venue_feat.set_index("business_id")["repeat_user_rate"]
    q0 = np.clip(np.array([
        1.0 / (float(rr_map.get(i2v[i], 0.01) or 0.01) + 0.01)
        for i in range(len(v2i))
    ]), 1e-10, None)
    _, q = birank(W, p0=np.ones(len(u2i)), q0=q0)
    return {i2v[i]: float(q[i]) for i in range(len(v2i))}


def run(
    dataset: str = "uk_foursquare",
    models:  list = None,
    metrics: list = None,
    n_bootstrap: int = 500,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the EmergeRec benchmark on one dataset.
    Returns a DataFrame: rows=methods, columns=metrics.
    """
    if models  is None: models  = SUPPORTED_MODELS
    if metrics is None: metrics = SUPPORTED_METRICS

    cfg = get_dataset(dataset)
    if verbose:
        print(f"\n{'='*60}")
        print(f"EmergeRec: {dataset}")
        print(f"  {cfg['description']}")
        print(f"  Models: {models}")
        print(f"{'='*60}")

    df = pd.read_csv(cfg["interactions"], dtype={"user_id": str, "business_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not cfg["has_stars"] or "stars" not in df.columns:
        df["stars"] = np.nan

    train, test_rev, train_uv, _ = temporal_split(df, cfg["split_date"])
    if len(test_rev) < 5:
        print("  Too few revisit users — skipping")
        return pd.DataFrame()

    user_feat  = compute_user_features(train)
    venue_feat = compute_venue_features(train)
    decay_e    = build_decayed_edges(train, cfg["split_date"], lam=0.5)
    count_e    = build_count_edges(train)
    test_full  = df[df["timestamp"] >= pd.Timestamp(cfg["split_date"])].copy()
    rising     = compute_rising_stars(train, test_full)

    r_explore = build_birank_explore(decay_e, user_feat, venue_feat)
    r_als     = build_mf_ranking(train, "als")
    np.random.seed(42)

    RANKINGS = {
        "baseline_popularity":      build_popularity_ranking(train),
        "baseline_random":          {v: float(np.random.random()) for v in train["business_id"].unique()},
        "birank_loyalty":           build_birank_ranking(decay_e, user_feat, venue_feat),
        "birank_explore":           r_explore,
        "hybrid_explore_als":       blend_rankings(r_explore, r_als, lam=0.5),
        "hybrid_anti_loyalty_als":  blend_rankings(_build_anti_loyalty(decay_e, venue_feat), r_als, lam=0.5),
    }
    if cfg["has_stars"]:
        RANKINGS["baseline_rating"] = build_rating_ranking(train)

    rows = []
    for mname in models:
        if mname not in RANKINGS:
            continue
        ranking = RANKINGS[mname]
        row = {"dataset": dataset, "method": mname}

        if "rising_stars_rho" in metrics:
            rho, p = spearman_rising(ranking, rising)
            row["rising_stars_rho"] = round(rho, 4)
            row["rho_p_value"]      = round(p, 4)
            row["rho_sig"]          = "***" if p < 0.001 else ("*" if p < 0.05 else "ns")

        if "ndcg@10" in metrics or "hit@10" in metrics:
            agg, pu, _ = evaluate_per_user(ranking, train_uv, test_rev)
            if "ndcg@10" in metrics:
                ndcg = agg.get("NDCG@10", 0)
                arr  = pu.get("NDCG@10", np.array([0]))
                lo, hi = bootstrap_ci(arr, n_boot=n_bootstrap) if len(arr) > 5 else (0, 0)
                row["ndcg@10"]    = round(ndcg, 4)
                row["ndcg_ci_lo"] = round(lo, 4)
                row["ndcg_ci_hi"] = round(hi, 4)
            if "hit@10" in metrics:
                row["hit@10"] = round(agg.get("Hit@10", 0), 4)

        rows.append(row)
        if verbose:
            rho_str  = f"ρ={row.get('rising_stars_rho',0):+.4f}{row.get('rho_sig','')}" if "rising_stars_rho" in metrics else ""
            ndcg_str = f"NDCG@10={row.get('ndcg@10',0):.4f}" if "ndcg@10" in metrics else ""
            print(f"  {mname:<30} {rho_str:<14} {ndcg_str}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="EmergeRec benchmark")
    parser.add_argument("--datasets", default="uk_foursquare,tripadvisor_london",
                        help="comma-separated or 'all'")
    parser.add_argument("--models",   default="all")
    parser.add_argument("--metrics",  default="rising_stars_rho,ndcg@10,hit@10")
    parser.add_argument("--n-bootstrap", type=int, default=500)
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.datasets == "all" else args.datasets.split(",")
    models   = SUPPORTED_MODELS      if args.models   == "all" else args.models.split(",")
    metrics  = args.metrics.split(",")

    all_results = []
    for ds in datasets:
        try:
            r = run(ds, models, metrics, args.n_bootstrap)
            if not r.empty:
                all_results.append(r)
        except FileNotFoundError as e:
            print(f"  Skipping {ds}: {e}")

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        out   = Path(__file__).parent.parent / "data/results/emergerec_leaderboard.csv"
        final.to_csv(out, index=False)
        leaderboard_dst = Path(__file__).parent / "leaderboard/results.csv"
        leaderboard_dst.parent.mkdir(exist_ok=True)
        final.to_csv(leaderboard_dst, index=False)
        print(f"\nLeaderboard saved → {out}")
        print(final[["dataset", "method", "rising_stars_rho", "ndcg@10"]].to_string(index=False))
    return all_results


if __name__ == "__main__":
    main()
