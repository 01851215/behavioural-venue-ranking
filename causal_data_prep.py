"""
Causal data preparation for Direction 5: PSM study.

Streams Yelp review JSON to compute:
  - Treatment: consistency_score = weekday_ratio - minmax_norm(peak_hour_entropy)
  - Outcome:   future_revisit_rate (fraction of pre-2020 users who returned post-2020)
  - Confounders: total_visits, unique_users, gini_user_contribution (from venue features)

Writes: causal_venue_dataset.csv
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

DATA_DIR      = Path(__file__).parent
REVIEW_PATH   = DATA_DIR / "../yelp_dataset/yelp_academic_dataset_review.json"
COFFEE_PATH   = DATA_DIR / "business_coffee_v2.csv"
FEATURES_PATH = DATA_DIR / "coffee_venue_features_v2.csv"
SPLIT_DATE    = "2020-01-01"

CONFOUNDER_COLS = ["total_visits", "unique_users", "gini_user_contribution"]


def compute_consistency_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add consistency_score and binary treatment column.
    consistency_score = weekday_ratio - minmax_norm(peak_hour_entropy)
    Treatment = 1 if score > median, 0 if <= median, NaN if inputs missing.
    """
    df = df.copy()
    valid = df["weekday_ratio"].notna() & df["peak_hour_entropy"].notna()

    ent = df.loc[valid, "peak_hour_entropy"]
    denom = ent.max() - ent.min()
    ent_norm = (ent - ent.min()) / denom if denom > 0 else pd.Series(0.0, index=ent.index)

    df["peak_hour_entropy_norm"] = np.nan
    df.loc[valid, "peak_hour_entropy_norm"] = ent_norm

    df["consistency_score"] = np.nan
    df.loc[valid, "consistency_score"] = (
        df.loc[valid, "weekday_ratio"] - df.loc[valid, "peak_hour_entropy_norm"]
    )

    median = df.loc[valid, "consistency_score"].median()
    df["treatment"] = np.nan
    df.loc[valid & (df["consistency_score"] > median),  "treatment"] = 1.0
    df.loc[valid & (df["consistency_score"] <= median), "treatment"] = 0.0

    return df


def compute_future_revisit_rate(
    pre_users: dict,
    post_users: dict,
    coffee_ids: set,
) -> dict:
    """
    For each venue, compute fraction of pre-2020 users who returned post-2020.
    Venues with zero pre-2020 users are excluded.
    """
    rates = {}
    for bid in coffee_ids:
        pre = pre_users.get(bid, set())
        if not pre:
            continue
        post = post_users.get(bid, set())
        returning = pre & post
        rates[bid] = len(returning) / len(pre)
    return rates


def build_causal_dataset(df: pd.DataFrame, future_revisit_rates: dict) -> pd.DataFrame:
    """
    Join treatment, confounders, and outcome into one analysis-ready DataFrame.
    Drops venues missing outcome or treatment.
    """
    missing = [c for c in CONFOUNDER_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"build_causal_dataset: features DataFrame is missing confounder columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.copy()
    df["future_revisit_rate"] = df["business_id"].map(future_revisit_rates)

    keep_cols = (
        ["business_id", "consistency_score", "treatment", "future_revisit_rate"]
        + CONFOUNDER_COLS
    )
    result = df[keep_cols].copy()
    n_before = len(result)
    result = result.dropna(subset=["future_revisit_rate", "treatment"])
    n_dropped = n_before - len(result)
    if n_dropped:
        print(f"  Warning: dropped {n_dropped} venues missing treatment or revisit rate")
    return result.reset_index(drop=True)


def load_coffee_reviews(review_path: Path, coffee_ids: set, split_date: str = SPLIT_DATE):
    """
    Stream Yelp review JSON. Returns (pre_users, post_users) dicts:
      pre_users[venue_id]  = set of user_ids who visited before split_date
      post_users[venue_id] = set of user_ids who visited on/after split_date
    """
    print(f"Streaming {review_path} (5.3 GB — takes ~2 min)...")
    pre_users  = defaultdict(set)
    post_users = defaultdict(set)
    n_total = n_coffee = 0

    with open(review_path) as f:
        for line in f:
            row = json.loads(line)
            n_total += 1
            bid = row["business_id"]
            if bid not in coffee_ids:
                continue
            n_coffee += 1
            uid  = row["user_id"]
            date = row["date"]
            if date < split_date:
                pre_users[bid].add(uid)
            else:
                post_users[bid].add(uid)

            if n_total % 1_000_000 == 0:
                print(f"  {n_total/1e6:.0f}M reviews processed, {n_coffee:,} coffee...")

    print(f"  Done. {n_total:,} reviews scanned, {n_coffee:,} coffee.")
    print(f"  Venues with pre-2020 users:  {len(pre_users):,}")
    print(f"  Venues with post-2020 users: {len(post_users):,}")
    return dict(pre_users), dict(post_users)


if __name__ == "__main__":
    print("Loading coffee business IDs...")
    coffee_df = pd.read_csv(COFFEE_PATH)
    coffee_ids = set(coffee_df["business_id"])
    print(f"  {len(coffee_ids):,} coffee venues")

    print("Loading venue features...")
    features = pd.read_csv(FEATURES_PATH)

    print("Computing treatment (consistency score)...")
    features = compute_consistency_score(features)
    n_treated  = (features["treatment"] == 1).sum()
    n_control  = (features["treatment"] == 0).sum()
    n_excluded = features["treatment"].isna().sum()
    print(f"  Treated: {n_treated:,}  Control: {n_control:,}  Excluded: {n_excluded:,}")

    pre_users, post_users = load_coffee_reviews(REVIEW_PATH, coffee_ids)

    print("Computing future revisit rates...")
    rates = compute_future_revisit_rate(pre_users, post_users, coffee_ids)
    print(f"  {len(rates):,} venues with valid outcome")
    rate_series = pd.Series(list(rates.values()))
    print(f"  Rate: mean={rate_series.mean():.4f}  median={rate_series.median():.4f}  max={rate_series.max():.4f}")

    print("Building causal dataset...")
    dataset = build_causal_dataset(features, rates)
    print(f"  Final dataset: {len(dataset):,} venues  "
          f"(treated: {(dataset['treatment']==1).sum():,}, "
          f"control: {(dataset['treatment']==0).sum():,})")

    out = DATA_DIR / "causal_venue_dataset.csv"
    dataset.to_csv(out, index=False)
    print(f"Saved -> causal_venue_dataset.csv")
