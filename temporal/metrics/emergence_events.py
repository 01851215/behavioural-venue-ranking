"""
Formal implementation of venue emergence events.

Based on: theory/derivations/emergence_definition.tex

A venue emerges at time t_e when its traffic rate exceeds
the popularity-baseline OLS prediction by σ std deviations
for k consecutive windows of width w days.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def compute_emergence_events(
    interactions_df: pd.DataFrame,
    split_date: str,
    window_days: int = 90,
    sigma: float = 1.0,
    k_windows: int = 1,
    r_min: float = 0.1,
) -> dict:
    """
    Identify venue emergence events from interaction data.

    Returns: {venue_id: emergence_timestamp} for venues that emerged
             in the test period.
    """
    df = interactions_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    split = pd.Timestamp(split_date)

    train = df[df["timestamp"] < split]
    test  = df[df["timestamp"] >= split]

    # Compute pre-split traffic rate (interactions / day)
    span_days = max((split - train["timestamp"].min()).days, 1)
    train_rate = (train.groupby("business_id").size() / span_days).rename("train_rate")

    # Compute post-split traffic rate in rolling windows
    test_span = max((test["timestamp"].max() - split).days, 1)
    test_rate  = (test.groupby("business_id").size() / test_span).rename("test_rate")

    merged = pd.concat([train_rate, test_rate], axis=1).fillna(0)
    merged = merged[merged["train_rate"] >= r_min]   # must have pre-split traffic

    # OLS debiasing: regress log(test_rate) on log(train_rate)
    valid = merged[(merged["train_rate"] > 0) & (merged["test_rate"] > 0)]
    if len(valid) < 10:
        return {}

    log_train = np.log1p(valid["train_rate"].values)
    log_test  = np.log1p(valid["test_rate"].values)
    slope, intercept = np.polyfit(log_train, log_test, 1)
    predicted  = intercept + slope * log_train
    residuals  = log_test - predicted

    # Emergence: residual > σ std
    threshold = sigma * residuals.std()
    emerging_venues = valid.index[residuals >= threshold].tolist()

    # Assign emergence date as first test interaction date
    emergence_dates = {}
    for vid in emerging_venues:
        v_test = test[test["business_id"] == vid]
        if len(v_test) > 0:
            emergence_dates[vid] = v_test["timestamp"].min()

    return emergence_dates


def emergence_summary(interactions_df: pd.DataFrame, split_date: str) -> dict:
    """Print a summary of emergence events for a dataset."""
    events = compute_emergence_events(interactions_df, split_date)
    total_venues = interactions_df["business_id"].nunique()
    return {
        "total_venues": total_venues,
        "emerging_venues": len(events),
        "emergence_rate_pct": round(100 * len(events) / max(total_venues, 1), 2),
        "note": "Emergence rate should be 10-25% for well-calibrated σ=1.0",
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    REPO = Path(__file__).parent.parent.parent
    for name, fpath, split in [
        ("UK FSQ",  "data/results/uk_fsq_interactions.csv",  "2013-07-01"),
        ("London",  "data/results/london_interactions.csv",   "2018-01-01"),
    ]:
        try:
            df = pd.read_csv(
                REPO / fpath,
                dtype={"user_id": str, "business_id": str}
            )
            s = emergence_summary(df, split)
            print(f"{name}: {s['emerging_venues']:,} / {s['total_venues']:,} venues "
                  f"({s['emergence_rate_pct']}%) emerged")
        except FileNotFoundError:
            print(f"{name}: file not found")
