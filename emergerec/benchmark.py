"""
EmergeRec — One-Command Benchmark Runner

Usage:
    python -m emergerec.benchmark --datasets all --models all

    from emergerec import benchmark
    results = benchmark.run("uk_foursquare", models=["popularity", "hybrid_anti_loyalty"])
    print(results)

Paper target: NeurIPS Datasets & Benchmarks 2028
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SUPPORTED_DATASETS = ["yelp_coffee", "yelp_restaurants", "yelp_hotels",
                       "tripadvisor_london", "uk_foursquare"]

SUPPORTED_MODELS   = ["popularity", "random", "item_knn", "birank_loyalty",
                       "birank_explore", "hybrid_explore_als",
                       "hybrid_anti_loyalty_als", "lightgcn", "sasrec"]

SUPPORTED_METRICS  = ["ndcg@10", "hit@10", "rising_stars_rho", "lte@10"]


def run(dataset: str = "uk_foursquare",
        models: list = None,
        metrics: list = None,
        n_bootstrap: int = 1000) -> pd.DataFrame:
    """
    Run the EmergeRec benchmark on a dataset with specified models and metrics.

    Returns a DataFrame with one row per model, one column per metric.
    """
    if models is None:
        models = SUPPORTED_MODELS
    if metrics is None:
        metrics = SUPPORTED_METRICS

    print(f"EmergeRec benchmark: {dataset}")
    print(f"  Models:  {models}")
    print(f"  Metrics: {metrics}")
    print("  (Full implementation in PhD Year 1 — skeleton ready)")

    # TODO: implement full dataset loading + model training + evaluation
    # Each model imports from bvr.models.*, bvr.core.validation, temporal.models.*
    # Each metric imports from bvr.metrics.*, temporal.metrics.lte

    return pd.DataFrame({"model": models, "status": ["pending"] * len(models)})


def main():
    parser = argparse.ArgumentParser(description="EmergeRec benchmark runner")
    parser.add_argument("--datasets", default="uk_foursquare",
                        help=f"comma-separated, or 'all'. Options: {SUPPORTED_DATASETS}")
    parser.add_argument("--models", default="all",
                        help=f"comma-separated, or 'all'. Options: {SUPPORTED_MODELS}")
    parser.add_argument("--metrics", default="ndcg@10,rising_stars_rho",
                        help="comma-separated metric names")
    parser.add_argument("--n-bootstrap", type=int, default=1000)

    args = parser.parse_args()
    datasets = SUPPORTED_DATASETS if args.datasets == "all" else args.datasets.split(",")
    models   = SUPPORTED_MODELS   if args.models   == "all" else args.models.split(",")
    metrics  = args.metrics.split(",")

    all_results = []
    for ds in datasets:
        results = run(ds, models, metrics, args.n_bootstrap)
        results.insert(0, "dataset", ds)
        all_results.append(results)

    final = pd.concat(all_results, ignore_index=True)
    print(final.to_string(index=False))
    return final


if __name__ == "__main__":
    main()
