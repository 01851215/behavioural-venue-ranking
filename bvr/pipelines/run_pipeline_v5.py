"""
Pipeline v5 — BiRank + Cold-Start Injection.

Warm venues (>= best threshold): keep BiRank score unchanged.
Cold venues (< threshold, have check-in data): inject normalized pseudo-score.
Unranked venues (no BiRank, no check-in data): score=0, source="unranked".

Reads:
  coffee_birank_venue_scores_by_group.csv
  cold_start_scores.csv
  cold_start_threshold_sweep.csv
  coffee_venue_features_v2.csv

Writes:
  coffee_birank_venue_scores_v5.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from cold_start_ranker import select_best_threshold

DATA_DIR = Path(__file__).parent


def load_birank_scores() -> pd.DataFrame:
    """Aggregate by-group BiRank scores to single per-venue score."""
    by_group = pd.read_csv(DATA_DIR / "coffee_birank_venue_scores_by_group.csv")
    scores = by_group.groupby("business_id")["score"].mean().reset_index()
    scores.columns = ["business_id", "birank_score"]
    return scores


def load_review_counts() -> pd.DataFrame:
    venue_features = pd.read_csv(DATA_DIR / "coffee_venue_features_v2.csv")
    return venue_features[["business_id", "total_visits"]].rename(
        columns={"total_visits": "review_count"}
    )


def merge_scores(
    birank: pd.DataFrame,
    cold_start: pd.DataFrame,
    review_counts: pd.DataFrame,
    best_threshold: int,
) -> pd.DataFrame:
    """
    Merge BiRank + cold-start scores into unified output.

    Warm (review_count >= threshold): final_score = birank_score
    Cold (review_count <  threshold, in cold_start): final_score = pseudo_birank_score
    Unranked: final_score = 0
    """
    all_venues = birank.merge(review_counts, on="business_id", how="outer")
    all_venues["review_count"] = all_venues["review_count"].fillna(0).astype(int)

    all_venues = all_venues.merge(
        cold_start[["business_id", "pseudo_birank_score"]],
        on="business_id",
        how="left",
    )

    def assign_score(row):
        birank_val = row.get("birank_score", np.nan)
        pseudo_val = row.get("pseudo_birank_score", np.nan)
        if row["review_count"] >= best_threshold and not pd.isna(birank_val):
            return birank_val, "birank"
        elif not pd.isna(pseudo_val):
            return pseudo_val, "cold_start"
        else:
            return 0.0, "unranked"

    results = all_venues.apply(assign_score, axis=1, result_type="expand")
    all_venues["final_score"]  = results[0]
    all_venues["score_source"] = results[1]
    all_venues["cold_threshold_used"] = best_threshold

    all_venues = all_venues.sort_values("final_score", ascending=False).reset_index(drop=True)
    all_venues["rank"] = all_venues.index + 1

    return all_venues[["business_id", "final_score", "rank",
                        "score_source", "review_count", "cold_threshold_used"]]


if __name__ == "__main__":
    print("Loading BiRank scores...")
    birank = load_birank_scores()
    print(f"  {len(birank):,} venues with BiRank scores")

    print("Loading cold-start scores...")
    cold_start = pd.read_csv(DATA_DIR / "cold_start_scores.csv")
    print(f"  {len(cold_start):,} cold venues with pseudo-scores")

    print("Loading review counts...")
    review_counts = load_review_counts()

    print("Loading threshold sweep to find best threshold...")
    sweep = pd.read_csv(DATA_DIR / "cold_start_threshold_sweep.csv")
    sweep_results = sweep.to_dict("records")
    best = select_best_threshold(sweep_results, spearman_floor=0.4,
                                 ndcg_tolerance=0.01, baseline_ndcg=0.0765)
    best_threshold = best["threshold"]
    print(f"  Best threshold: {best_threshold} reviews")

    print("Merging scores...")
    unified = merge_scores(birank, cold_start, review_counts, best_threshold)

    src_counts = unified["score_source"].value_counts()
    total = len(unified)
    print(f"\nCoverage report:")
    print(f"  birank:      {src_counts.get('birank', 0):,} venues  "
          f"({src_counts.get('birank', 0)/total*100:.1f}%)")
    print(f"  cold_start:  {src_counts.get('cold_start', 0):,} venues  "
          f"({src_counts.get('cold_start', 0)/total*100:.1f}%)")
    print(f"  unranked:    {src_counts.get('unranked', 0):,} venues  "
          f"({src_counts.get('unranked', 0)/total*100:.1f}%)")

    before_pct = src_counts.get("birank", 0) / total * 100
    after_pct  = (src_counts.get("birank", 0) + src_counts.get("cold_start", 0)) / total * 100
    print(f"\n  Coverage gain: {before_pct:.1f}% -> {after_pct:.1f}%  "
          f"(+{after_pct - before_pct:.1f}%)")

    out_path = DATA_DIR / "coffee_birank_venue_scores_v5.csv"
    unified.to_csv(out_path, index=False)
    print(f"\nSaved -> coffee_birank_venue_scores_v5.csv  ({len(unified):,} venues)")
