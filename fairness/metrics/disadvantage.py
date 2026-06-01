"""
Algorithmic Disadvantage Metric

Quantifies the rank gap between venues with identical quality signals
but different popularity levels — the structural penalty paid by
less-popular (often independent) venues.

The metric answers: "If venue A and venue B have the same quality,
but venue A has 10x fewer reviews, how many rank positions does
venue B lose?"

Paper target: FAccT 2030 / RecSys 2030 Fairness Workshop
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bvr.core.validation import compute_venue_features


def long_tail_ranking_gap(
    ranking: dict,
    train_df: pd.DataFrame,
    quality_col: str = "repeat_user_rate",
    n_bins: int = 5,
) -> dict:
    """
    Compute the expected rank gap between venues of equal quality
    but different popularity.

    Method:
    1. Bin venues by popularity (review count quintiles)
    2. Within each popularity bin, measure the spread of rankings
       for venues with similar quality_col values
    3. Report mean rank gap across matched pairs (high-pop vs low-pop,
       same quality bin)

    Returns:
        dict with overall_gap, gap_by_popularity_bin, and significance
    """
    venue_feat = compute_venue_features(train_df)
    venue_feat = venue_feat.copy()
    venue_feat["rank"] = venue_feat["business_id"].map(
        {vid: rank for rank, (vid, _) in enumerate(
            sorted(ranking.items(), key=lambda x: x[1], reverse=True), 1
        )}
    )

    # Quintile bins by popularity
    venue_feat["pop_bin"] = pd.qcut(
        venue_feat["popularity_visits"], q=n_bins,
        labels=[f"Q{i+1}" for i in range(n_bins)], duplicates="drop"
    )

    # Quality bins
    venue_feat["quality_bin"] = pd.qcut(
        venue_feat[quality_col], q=n_bins,
        labels=[f"Q{i+1}" for i in range(n_bins)], duplicates="drop"
    )

    # For each quality bin, compare ranks of Q1 (least popular) vs Q5 (most popular)
    gaps = []
    for q_bin in venue_feat["quality_bin"].dropna().unique():
        sub = venue_feat[venue_feat["quality_bin"] == q_bin].dropna(subset=["pop_bin", "rank"])
        low_pop  = sub[sub["pop_bin"] == "Q1"]["rank"]
        high_pop = sub[sub["pop_bin"] == "Q5"]["rank"]
        if len(low_pop) > 0 and len(high_pop) > 0:
            # Lower rank number = better ranking; gap = less-popular has higher rank number
            gap = float(low_pop.mean() - high_pop.mean())
            gaps.append({"quality_bin": q_bin, "gap": gap})

    overall_gap = float(np.mean([g["gap"] for g in gaps])) if gaps else 0.0

    return {
        "overall_gap": round(overall_gap, 2),
        "gap_by_quality_bin": gaps,
        "interpretation": (
            f"An independent venue with the same quality as a popular venue "
            f"is ranked on average {abs(overall_gap):.0f} positions lower."
        ),
    }
