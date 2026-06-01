"""
Lead-Time-to-Emergence (LTE@k) — Novel Evaluation Metric

Measures how many days BEFORE a venue's actual emergence event
the model first ranks it in the top-k.

Higher = better: the model detected the venue's rise earlier.

This metric is missing from the RecSys literature and is
the primary novel evaluation contribution of Paper P3.

Reference: to be published in temporal/papers/p3_tgn/
"""

from datetime import datetime
from typing import Dict, List
import numpy as np


def lead_time_to_emergence(
    rankings_over_time: Dict[datetime, List[str]],
    true_emergence_dates: Dict[str, datetime],
    k: int = 10,
) -> float:
    """
    Compute mean Lead-Time-to-Emergence at k.

    Args:
        rankings_over_time: {timestamp: [venue_id_1, venue_id_2, ...]}
                            sorted by model score, at each evaluation point
        true_emergence_dates: {venue_id: emergence_timestamp}
                              ground truth emergence events
        k: cut-off rank

    Returns:
        mean LTE@k in days (higher = earlier detection = better)
    """
    lead_times = []

    for venue_id, emerge_date in true_emergence_dates.items():
        first_detected = None

        for ts in sorted(rankings_over_time.keys()):
            if ts >= emerge_date:
                break   # can't detect after the fact
            top_k = rankings_over_time[ts][:k]
            if venue_id in top_k:
                first_detected = ts
                break   # first time in top-k before emergence

        if first_detected is not None:
            days_lead = (emerge_date - first_detected).days
            lead_times.append(days_lead)

    return float(np.mean(lead_times)) if lead_times else 0.0


def define_emergence_events(
    interactions_df,
    split_date: str,
    window_days: int = 90,
    min_growth_pct: float = 0.5,
) -> Dict[str, datetime]:
    """
    Identify venue emergence events from interaction data.

    A venue "emerges" when its traffic rate in a rolling window
    exceeds its pre-split baseline by min_growth_pct.

    This is the ground-truth signal for LTE evaluation.
    """
    import pandas as pd
    df = interactions_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    split = pd.Timestamp(split_date)

    baseline = (
        df[df["timestamp"] < split]
        .groupby("business_id")
        .size()
        .rename("baseline_count")
    )

    emergence_dates = {}
    for vid, group in df[df["timestamp"] >= split].sort_values("timestamp").groupby("business_id"):
        base = baseline.get(vid, 1)
        rolling = group.set_index("timestamp").resample(f"{window_days}D").size()
        # Normalise to per-day rate
        daily = rolling / window_days
        baseline_daily = base / max((split - df[df["business_id"] == vid]["timestamp"].min()).days, 1)

        for ts, rate in daily.items():
            if rate > baseline_daily * (1 + min_growth_pct):
                emergence_dates[vid] = ts
                break

    return emergence_dates
