"""
Compute anonymous venue signals from Yelp check-in JSON.

Input:  ../yelp_dataset/yelp_academic_dataset_checkin.json
Output: enriches coffee_venue_features_v2.csv with 10 new columns

Features derived purely from timestamps — no user_id required.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress

DATA_DIR   = Path(__file__).parent
CHECKIN_PATH = DATA_DIR / "../yelp_dataset/yelp_academic_dataset_checkin.json"
VENUE_FEATURES_PATH = DATA_DIR / "coffee_venue_features_v2.csv"


def compute_checkin_burstiness(timestamps) -> float:
    """CV of daily check-in counts. NaN if fewer than 2 check-ins total.
    If all check-ins fall on a single day, returns total count as a high-burstiness sentinel."""
    if len(timestamps) < 2:
        return np.nan
    series = pd.Series(timestamps)
    daily = series.dt.date.value_counts()
    if len(daily) < 2:
        # All check-ins on one day — maximally bursty; return total count as proxy
        return float(len(timestamps))
    return daily.std() / daily.mean() if daily.mean() > 0 else np.nan


def compute_peak_hour_entropy(timestamps) -> float:
    """Shannon entropy (bits) of hour-of-day distribution."""
    if not len(timestamps):
        return np.nan
    hours = pd.Series(timestamps).dt.hour
    counts = hours.value_counts(normalize=True)
    return float(-np.sum(counts * np.log2(counts)))


def compute_weekday_ratio(timestamps) -> float:
    """Fraction of check-ins falling on Mon-Fri."""
    if not len(timestamps):
        return np.nan
    days = pd.Series(timestamps).dt.dayofweek
    return float((days < 5).sum() / len(days))


def compute_temporal_stability_cv(timestamps) -> float:
    """CV of weekly check-in counts. NaN if fewer than 2 weeks."""
    if len(timestamps) < 2:
        return np.nan
    series = pd.Series(timestamps)
    weekly = series.dt.to_period("W").value_counts()
    if len(weekly) < 2:
        return np.nan
    return float(weekly.std() / weekly.mean()) if weekly.mean() > 0 else np.nan


def compute_visit_velocity_recent(timestamps, reference_date: pd.Timestamp) -> float:
    """Fraction of check-ins in last 6 months relative to reference_date."""
    if not len(timestamps):
        return np.nan
    cutoff = reference_date - pd.DateOffset(months=6)
    series = pd.Series(timestamps)
    return float((series >= cutoff).sum() / len(series))


def compute_growth_trend(timestamps) -> float:
    """Slope of linear regression on monthly check-in counts. NaN if <2 months."""
    if len(timestamps) < 2:
        return np.nan
    series = pd.Series(timestamps)
    monthly = series.dt.to_period("M").value_counts().sort_index()
    if len(monthly) < 2:
        return np.nan
    x = np.arange(len(monthly), dtype=float)
    y = monthly.values.astype(float)
    slope, _, _, _, _ = linregress(x, y)
    return float(slope)


def compute_lunch_dinner_ratio(timestamps) -> float:
    """Fraction of check-ins during lunch (11-14h) or dinner (17-21h)."""
    if not len(timestamps):
        return np.nan
    hours = pd.Series(timestamps).dt.hour
    mask = ((hours >= 11) & (hours < 14)) | ((hours >= 17) & (hours < 21))
    return float(mask.sum() / len(hours))


def compute_late_night_ratio(timestamps) -> float:
    """Fraction of check-ins between 22:00 and 02:00."""
    if not len(timestamps):
        return np.nan
    hours = pd.Series(timestamps).dt.hour
    mask = (hours >= 22) | (hours < 2)
    return float(mask.sum() / len(hours))


def compute_peak_hour_mode(timestamps) -> int:
    """Most common hour of day (0-23)."""
    if not len(timestamps):
        return np.nan
    hours = pd.Series(timestamps).dt.hour
    return int(hours.mode().iloc[0])


def extract_venue_features(timestamps, reference_date: pd.Timestamp) -> dict:
    """Compute all 10 anonymous features for a single venue."""
    ts = pd.to_datetime(timestamps)
    return {
        "total_checkins":        len(ts),
        "checkin_burstiness":    compute_checkin_burstiness(ts),
        "peak_hour_entropy":     compute_peak_hour_entropy(ts),
        "weekday_ratio":         compute_weekday_ratio(ts),
        "temporal_stability_cv": compute_temporal_stability_cv(ts),
        "visit_velocity_recent": compute_visit_velocity_recent(ts, reference_date),
        "growth_trend":          compute_growth_trend(ts),
        "lunch_dinner_ratio":    compute_lunch_dinner_ratio(ts),
        "late_night_ratio":      compute_late_night_ratio(ts),
        "peak_hour_mode":        compute_peak_hour_mode(ts),
    }


def load_checkin_data(path: Path) -> dict:
    """Returns {business_id: [pd.Timestamp, ...]}."""
    print(f"Loading check-in data from {path}...")
    checkins = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            bid = row["business_id"]
            timestamps = pd.to_datetime(row["date"].split(", "))
            checkins[bid] = list(timestamps)
    print(f"  Loaded {len(checkins):,} venues, parsing timestamps done.")
    return checkins


def compute_all_venue_features(checkins: dict) -> pd.DataFrame:
    """Compute 10 features for every venue in the check-in dict."""
    all_dates = [ts for tss in checkins.values() for ts in tss]
    reference_date = max(all_dates)
    print(f"  Reference date for velocity: {reference_date.date()}")

    rows = []
    for i, (bid, ts_list) in enumerate(checkins.items()):
        if i % 20000 == 0:
            print(f"  Processing venue {i:,}/{len(checkins):,}...")
        feats = extract_venue_features(ts_list, reference_date)
        feats["business_id"] = bid
        rows.append(feats)

    return pd.DataFrame(rows)


def enrich_venue_features(venue_features_path: Path, anon_features: pd.DataFrame) -> pd.DataFrame:
    """Merge anonymous features into existing venue features CSV."""
    existing = pd.read_csv(venue_features_path)
    anon_cols = [c for c in anon_features.columns if c != "business_id"]

    existing = existing.drop(columns=[c for c in anon_cols if c in existing.columns])

    merged = existing.merge(
        anon_features[["business_id"] + anon_cols],
        on="business_id",
        how="left",
    )
    print(f"  Venue features: {len(existing):,} rows merged with anon signals")
    print(f"  Coverage: {anon_features['business_id'].isin(existing['business_id']).sum():,} "
          f"of {len(existing):,} coffee venues have check-in data")
    return merged


if __name__ == "__main__":
    checkins = load_checkin_data(CHECKIN_PATH)

    print("Computing features for all venues...")
    anon_features = compute_all_venue_features(checkins)

    anon_out = DATA_DIR / "anonymous_venue_signals.csv"
    anon_features.to_csv(anon_out, index=False)
    print(f"Saved full anonymous signals -> {anon_out}")

    enriched = enrich_venue_features(VENUE_FEATURES_PATH, anon_features)
    enriched.to_csv(VENUE_FEATURES_PATH, index=False)
    print(f"Enriched coffee_venue_features_v2.csv - now {enriched.shape[1]} columns")
    print("Done.")
