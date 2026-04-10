#!/usr/bin/env python3
"""
Step 3 — Venue Behavioral Features (Car-Centric Model)
======================================================
Computes per-restaurant behavioral features from interactions, fsq busyness,
Yelp parking attributes, and Transitland accessibility.

Outputs:
  restaurant_venue_features.csv
"""

import math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INT_FILE   = BASE / "restaurant_interactions.csv"
BIZ_FILE   = BASE / "restaurant_businesses.csv"
BUSY_FILE  = BASE / "restaurant_busyness.csv"
TRA_FILE   = BASE / "restaurant_transit.csv"
OUT_FILE   = BASE / "restaurant_venue_features.csv"

EARTH_R_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))

def gini(array):
    """Compute Gini coefficient of a 1-D array."""
    arr = np.sort(np.asarray(array, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * arr) / (n * arr.sum())) - (n + 1) / n

def main():
    print("=" * 60)
    print("VENUE BEHAVIORAL FEATURES — CAR-CENTRIC")
    print("=" * 60)

    print("▸ Loading data …")
    interactions = pd.read_csv(INT_FILE, low_memory=False)
    biz = pd.read_csv(BIZ_FILE)
    busyness = pd.read_csv(BUSY_FILE)

    # Reviews only (have user_id)
    reviews = interactions[interactions["type"] == "review"].copy()
    reviews["timestamp"] = pd.to_datetime(reviews["timestamp"])

    # All interactions for count purposes
    all_inter = interactions.copy()
    all_inter["timestamp"] = pd.to_datetime(all_inter["timestamp"])

    # ── Per-venue aggregation ──
    print("▸ Computing venue features …")
    results = []
    biz_dict = biz.set_index("business_id").to_dict("index")
    review_groups = reviews.groupby("business_id")
    all_groups = all_inter.groupby("business_id")

    for bid, info in biz_dict.items():
        if bid in all_groups.groups:
            total_interactions = len(all_groups.get_group(bid))
        else:
            total_interactions = 0

        # Review-based features
        if bid in review_groups.groups:
            rev_grp = review_groups.get_group(bid)
            user_counts = rev_grp["user_id"].value_counts()
            n_reviews = len(rev_grp)
            unique_reviewers = len(user_counts)
            repeat_users = (user_counts > 1).sum()
            repeat_user_rate = repeat_users / unique_reviewers if unique_reviewers > 0 else 0

            gini_val = gini(user_counts.values)

            rating_vals = rev_grp["stars"].dropna().values
            avg_rating = np.mean(rating_vals) if len(rating_vals) > 0 else info.get("stars", 3.0)
            rating_consistency = 1.0 / (np.std(rating_vals) + 0.1) if len(rating_vals) > 1 else 1.0

            rev_grp_ts = rev_grp.set_index("timestamp")
            weekly = rev_grp_ts.resample("W").size()
            if len(weekly) > 1 and weekly.mean() > 0:
                stability_cv = weekly.std() / weekly.mean()
            else:
                stability_cv = 1.0
        else:
            n_reviews = 0
            unique_reviewers = 0
            repeat_user_rate = 0.0
            gini_val = 0.0
            avg_rating = info.get("stars", 3.0)
            rating_consistency = 1.0
            stability_cv = 1.0

        cats = info.get("categories", "") or ""
        cat_list = [c.strip() for c in cats.split(",") if c.strip()]
        top_cats = ", ".join(cat_list[:3])

        results.append({
            "business_id":       bid,
            "popularity":        total_interactions,
            "unique_visitors":   unique_reviewers,
            "repeat_user_rate":  round(repeat_user_rate, 4),
            "gini_user_concentration": round(gini_val, 4),
            "avg_rating":        round(avg_rating, 2),
            "rating_consistency": round(rating_consistency, 3),
            "stability_cv":      round(stability_cv, 4),
            "cuisine_categories": top_cats,
            "parking_lot":       info.get("parking_lot", False),
            "parking_garage":    info.get("parking_garage", False),
            "parking_valet":     info.get("parking_valet", False),
            "parking_street":    info.get("parking_street", False),
        })

    venue_df = pd.DataFrame(results)

    # ── Merge Foursquare busyness ──
    print("▸ Merging Foursquare busyness …")
    venue_df = venue_df.merge(
        busyness[["business_id", "peak_busyness", "avg_busyness", "peak_hour"]],
        on="business_id", how="left"
    )
    venue_df["peak_busyness"] = venue_df["peak_busyness"].fillna(0)
    venue_df["avg_busyness"] = venue_df["avg_busyness"].fillna(0)
    venue_df["peak_hour"] = venue_df["peak_hour"].fillna(12)

    # ── Walking density (helps determine parking difficulty) ──
    print("▸ Computing walking density …")
    from scipy.spatial import cKDTree

    biz_lats = biz["latitude"].values
    biz_lons = biz["longitude"].values
    biz_xyz = np.array([
        [math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
         math.cos(math.radians(lat)) * math.sin(math.radians(lon)),
         math.sin(math.radians(lat))]
        for lat, lon in zip(biz_lats, biz_lons)
    ])
    tree = cKDTree(biz_xyz)
    chord_800 = 2 * math.sin(800 / (2 * EARTH_R_KM * 1000))
    counts = tree.query_ball_point(biz_xyz, r=chord_800, return_length=True)
    walking_density = pd.Series(counts - 1, index=biz["business_id"])
    venue_df["walking_density"] = venue_df["business_id"].map(walking_density).fillna(0).astype(int)

    # ── Parking Score Calculation ──
    print("▸ Computing Parking Convenience Scores …")
    density_p75 = venue_df["walking_density"].quantile(0.75)
    
    def calc_parking_score(row):
        # 1.0 (Best) = Dedicated lot or garage
        # 0.5 (Good) = Valet or Street parking in low-density area
        # 0.0 (Poor) = Street parking in high-density area, or nothing defined
        if pd.isna(row["parking_lot"]):
            return 0.5
            
        if row["parking_lot"] or row["parking_garage"]:
            return 1.0
        elif row["parking_valet"]:
            return 0.5
        elif row["parking_street"]:
            if row["walking_density"] > density_p75:
                return 0.0 # Dense area, tough street parking
            else:
                return 0.5 # Suburbs, street parking is fine
        else:
            # Nothing defined, default to 0.5 expected
            return 0.5
            
    venue_df["parking_score"] = venue_df.apply(calc_parking_score, axis=1)

    # ── Transit accessibility merge (if available) ──
    if TRA_FILE.exists():
        transit_df = pd.read_csv(
            TRA_FILE,
            usecols=["business_id", "stops_800m", "departures_day_800m", "nearest_stop_m", "transit_access_score"],
        )
        venue_df = venue_df.merge(transit_df, on="business_id", how="left")
        venue_df["stops_800m"] = venue_df["stops_800m"].fillna(0).astype(int)
        venue_df["departures_day_800m"] = venue_df["departures_day_800m"].fillna(0.0)
        venue_df["nearest_stop_m"] = venue_df["nearest_stop_m"].fillna(9999.0)
        venue_df["transit_access_score"] = venue_df["transit_access_score"].fillna(0.0)
    else:
        venue_df["stops_800m"] = 0
        venue_df["departures_day_800m"] = 0.0
        venue_df["nearest_stop_m"] = 9999.0
        venue_df["transit_access_score"] = 0.0

    venue_df.to_csv(OUT_FILE, index=False)
    print(f"\n✅ {OUT_FILE.name} saved — {len(venue_df):,} venues")

    # Summary stats
    print(f"\n  Popularity: median={venue_df['popularity'].median():.0f}, max={venue_df['popularity'].max()}")
    print(f"  Foursquare Peak Busyness: mean={venue_df['peak_busyness'].mean():.1f}")
    print(f"  Walking density: mean={venue_df['walking_density'].mean():.1f}")
    print(f"  Parking Score Distribution:\n{venue_df['parking_score'].value_counts(normalize=True).round(3)}")
    print(f"  Transit access score: mean={venue_df['transit_access_score'].mean():.3f}")

if __name__ == "__main__":
    main()
