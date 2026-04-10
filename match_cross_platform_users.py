"""
Phase 2: Cross-Platform User Similarity Inference.

Matches Yelp users to Foursquare users by:
  1. Inverted-index blocking  — only compare pairs sharing >= 2 venues
  2. Jaccard similarity       — |shared venues| / |union of venues|
  3. Temporal co-presence     — did visits to shared venues overlap within 72h?
  4. Combined score           — 0.6 * jaccard + 0.4 * temporal

Confidence tiers:
  high:   combined >= 0.5 and shared_venues >= 5
  medium: combined >= 0.3 and shared_venues >= 3
  low:    combined >= 0.15 and shared_venues >= 2

Output: yelp_fsq_user_bridge.csv

Runtime: ~10-30 minutes on M5 Mac.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import timedelta

DATA_DIR = Path("/Users/chris/Desktop/Yelp JSON/yelp_dataset")
LINKAGE_PATH      = DATA_DIR / "venue_linkage.csv"
CHECKINS_PATH     = DATA_DIR / "fsq_checkins_linked.parquet"
COFFEE_INT_PATH   = DATA_DIR / "coffee_interactions.csv"
RESTAURANT_INT_PATH = DATA_DIR / "restaurant_interactions.csv"
OUTPUT_PATH       = DATA_DIR / "yelp_fsq_user_bridge.csv"

# Matching parameters
MIN_SHARED_VENUES    = 2      # Minimum shared venues to consider a pair
TEMPORAL_WINDOW_H    = 72     # Hours: Yelp visit within this window of FSQ visit counts as overlap
CONFIDENCE_THRESHOLD = 0.15   # Minimum combined score to keep in output


def load_yelp_interactions(linkage_venues: set) -> dict:
    """
    Load Yelp interactions (coffee + restaurant) filtered to venues
    that exist in venue_linkage. Returns:
        {yelp_user_id: {yelp_business_id: [pd.Timestamp, ...]}}
    Only keeps users who visited at least 1 linked venue.
    """
    print("Loading Yelp interactions...")

    dfs = []
    for path, col_map in [
        (COFFEE_INT_PATH,     {"user_id": "user_id", "business_id": "business_id", "timestamp": "timestamp"}),
        (RESTAURANT_INT_PATH, {"user_id": "user_id", "business_id": "business_id", "timestamp": "timestamp"}),
    ]:
        df = pd.read_csv(path, usecols=["user_id", "business_id", "timestamp"],
                         parse_dates=["timestamp"])
        df = df.dropna(subset=["user_id"])
        df = df[df["business_id"].isin(linkage_venues)]
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Yelp interactions at linked venues: {len(combined):,}")
    print(f"  Unique Yelp users at linked venues: {combined['user_id'].nunique():,}")

    # Build {user: {venue: [timestamps]}}
    user_venue_times = defaultdict(lambda: defaultdict(list))
    for _, row in combined.iterrows():
        if pd.notna(row["timestamp"]):
            user_venue_times[row["user_id"]][row["business_id"]].append(row["timestamp"])

    return dict(user_venue_times)


def load_fsq_interactions() -> dict:
    """
    Load FSQ linked check-ins. Returns:
        {fsq_user_id: {yelp_business_id: [pd.Timestamp, ...]}}
    """
    print("Loading FSQ linked check-ins...")
    df = pd.read_parquet(CHECKINS_PATH, columns=["fsq_user_id", "yelp_business_id", "local_ts"])
    df["local_ts"] = pd.to_datetime(df["local_ts"])
    print(f"  FSQ linked check-ins: {len(df):,}")
    print(f"  Unique FSQ users: {df['fsq_user_id'].nunique():,}")

    fsq_user_venue_times = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        if pd.notna(row["local_ts"]):
            fsq_user_venue_times[row["fsq_user_id"]][row["yelp_business_id"]].append(row["local_ts"])

    return dict(fsq_user_venue_times)


def build_candidate_pairs(yelp_uvt: dict, fsq_uvt: dict) -> dict:
    """
    Inverted index: venue -> set of users.
    For each venue, cross-product Yelp users × FSQ users = candidate pairs.
    Keep only pairs with >= MIN_SHARED_VENUES shared venues.

    Returns: {(yelp_uid, fsq_uid): n_shared_venues}
    """
    print("Building candidate pairs via inverted index...")

    # venue -> set of yelp users
    venue_to_yelp = defaultdict(set)
    for uid, venues in yelp_uvt.items():
        for v in venues:
            venue_to_yelp[v].add(uid)

    # venue -> set of fsq users
    venue_to_fsq = defaultdict(set)
    for uid, venues in fsq_uvt.items():
        for v in venues:
            venue_to_fsq[v].add(uid)

    shared_venues = set(venue_to_yelp) & set(venue_to_fsq)
    print(f"  Shared venues (appear in both datasets): {len(shared_venues):,}")

    pair_count = defaultdict(int)
    for venue in shared_venues:
        for y_uid in venue_to_yelp[venue]:
            for f_uid in venue_to_fsq[venue]:
                pair_count[(y_uid, f_uid)] += 1

    candidates = {k: v for k, v in pair_count.items() if v >= MIN_SHARED_VENUES}
    print(f"  Candidate pairs (>= {MIN_SHARED_VENUES} shared venues): {len(candidates):,}")
    return candidates


def temporal_overlap_score(yelp_venue_times: dict, fsq_venue_times: dict) -> float:
    """
    For each shared venue, check if any Yelp timestamp and any FSQ timestamp
    are within TEMPORAL_WINDOW_H hours of each other.

    Returns the fraction of shared venues with at least one temporal overlap.
    If no shared venues, returns 0.0.
    """
    shared = set(yelp_venue_times) & set(fsq_venue_times)
    if not shared:
        return 0.0

    window = timedelta(hours=TEMPORAL_WINDOW_H)
    overlapping = 0

    for venue in shared:
        y_times = yelp_venue_times[venue]
        f_times = fsq_venue_times[venue]
        found = False
        for yt in y_times:
            for ft in f_times:
                if abs(yt - ft) <= window:
                    found = True
                    break
            if found:
                break
        if found:
            overlapping += 1

    return overlapping / len(shared)


def score_candidates(candidates: dict, yelp_uvt: dict, fsq_uvt: dict) -> pd.DataFrame:
    """
    For each candidate pair, compute Jaccard + temporal + combined score.
    """
    print(f"Scoring {len(candidates):,} candidate pairs...")
    rows = []
    total = len(candidates)

    for i, ((y_uid, f_uid), n_shared) in enumerate(candidates.items()):
        if i % 1000 == 0 and i > 0:
            print(f"  {i:,}/{total:,} pairs scored...")

        y_venues = set(yelp_uvt[y_uid].keys())
        f_venues = set(fsq_uvt[f_uid].keys())
        union = len(y_venues | f_venues)

        jaccard = n_shared / union if union > 0 else 0.0
        temporal = temporal_overlap_score(yelp_uvt[y_uid], fsq_uvt[f_uid])
        combined = 0.6 * jaccard + 0.4 * temporal

        if combined < CONFIDENCE_THRESHOLD:
            continue

        if combined >= 0.5 and n_shared >= 5:
            tier = "high"
        elif combined >= 0.3 and n_shared >= 3:
            tier = "medium"
        elif combined >= 0.15 and n_shared >= 2:
            tier = "low"
        else:
            tier = "noise"

        rows.append({
            "yelp_user_id":       y_uid,
            "fsq_user_id":        f_uid,
            "n_shared_venues":    n_shared,
            "n_yelp_venues":      len(y_venues),
            "n_fsq_venues":       len(f_venues),
            "jaccard_venues":     round(jaccard, 4),
            "temporal_score":     round(temporal, 4),
            "combined_similarity": round(combined, 4),
            "confidence_tier":    tier,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("  No matches found above threshold.")
        return df

    # One Yelp user → at most one FSQ user (best match wins)
    df = df[df["confidence_tier"] != "noise"]
    df = df.sort_values("combined_similarity", ascending=False)
    df = df.drop_duplicates(subset=["yelp_user_id"], keep="first")
    df = df.drop_duplicates(subset=["fsq_user_id"],  keep="first")

    return df


def print_summary(df: pd.DataFrame):
    print("\n=== Bridge Table Summary ===")
    print(f"Total matched pairs: {len(df):,}")
    if len(df) == 0:
        return

    tier_counts = df["confidence_tier"].value_counts()
    for tier in ["high", "medium", "low"]:
        n = tier_counts.get(tier, 0)
        print(f"  {tier:8s}: {n:,}")

    print(f"\nJaccard:  mean={df['jaccard_venues'].mean():.3f}  max={df['jaccard_venues'].max():.3f}")
    print(f"Temporal: mean={df['temporal_score'].mean():.3f}  max={df['temporal_score'].max():.3f}")
    print(f"Combined: mean={df['combined_similarity'].mean():.3f}  max={df['combined_similarity'].max():.3f}")

    print("\nTop 10 matches:")
    print(df.head(10)[[
        "yelp_user_id", "fsq_user_id", "n_shared_venues",
        "jaccard_venues", "temporal_score", "combined_similarity", "confidence_tier"
    ]].to_string(index=False))


def main():
    linkage = pd.read_csv(LINKAGE_PATH)
    linkage_venues = set(linkage["yelp_business_id"])
    print(f"Linked venues: {len(linkage_venues):,}\n")

    yelp_uvt = load_yelp_interactions(linkage_venues)
    fsq_uvt  = load_fsq_interactions()

    print(f"\nYelp users with linked-venue visits: {len(yelp_uvt):,}")
    print(f"FSQ  users with linked-venue visits: {len(fsq_uvt):,}\n")

    candidates = build_candidate_pairs(yelp_uvt, fsq_uvt)

    if not candidates:
        print("No candidate pairs found — datasets may not overlap in time/space.")
        return

    bridge_df = score_candidates(candidates, yelp_uvt, fsq_uvt)
    print_summary(bridge_df)

    bridge_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
