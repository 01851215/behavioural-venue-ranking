#!/usr/bin/env python3
"""
Step 1 — Data Extraction & Cross-Dataset Linking
=================================================
Extracts restaurant data from three datasets:
  1. Yelp  — businesses, reviews, check-ins
  2. Transitland — transit stop locations & departure frequency
  3. populartimes (simulated) — hourly busyness from check-in timestamps

Outputs:
  restaurant_businesses.csv
  restaurant_interactions.csv
  restaurant_transit.csv      (per-venue transit accessibility)
  restaurant_busyness.csv     (per-venue hourly busyness 0-100)
"""

import json, csv, sys, math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# ── Paths ────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent          # yelp_dataset/
PROJECT  = BASE.parent                              # Yelp JSON/
YELP_BIZ = BASE / "yelp_academic_dataset_business.json"
YELP_REV = BASE / "yelp_academic_dataset_review.json"
YELP_CHK = BASE / "yelp_academic_dataset_checkin.json"


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return paths[0]


TL_STOPS = first_existing([
    BASE / "tl-dataset-US-2025-12-24T16_23_26" / "tl-dataset-US-2025-12-24T16:23:26-stops.csv",
    PROJECT / "tl-dataset-US-2025-12-24T16_23_26" / "tl-dataset-US-2025-12-24T16:23:26-stops.csv",
])

OUT_BIZ  = BASE / "restaurant_businesses.csv"
OUT_INT  = BASE / "restaurant_interactions.csv"
OUT_TRA  = BASE / "restaurant_transit.csv"
OUT_BUSY = BASE / "restaurant_busyness.csv"

RESTAURANT_KEYWORDS = {"Restaurants", "Food"}
WALKING_RADIUS_M   = 800
EARTH_RADIUS_KM    = 6371.0

# ── Helpers ──────────────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a)) * 1000

def latlon_to_xyz(lat, lon):
    """Convert lat/lon to 3D cartesian for cKDTree (unit sphere)."""
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    return (
        math.cos(lat_r) * math.cos(lon_r),
        math.cos(lat_r) * math.sin(lon_r),
        math.sin(lat_r),
    )

def radius_to_chord(metres):
    """Convert surface distance (metres) to chord distance on unit sphere."""
    return 2 * math.sin(metres / (2 * EARTH_RADIUS_KM * 1000))

# ── Step 1a: Extract restaurant businesses ───────────────────────────────
def extract_businesses():
    print("▸ Extracting restaurant businesses …")
    rows = []
    with open(YELP_BIZ, "r") as f:
        for line in f:
            b = json.loads(line)
            cats = set(c.strip() for c in (b.get("categories") or "").split(","))
            if cats & RESTAURANT_KEYWORDS:
                rows.append({
                    "business_id":  b["business_id"],
                    "name":         b["name"],
                    "city":         b["city"],
                    "state":        b["state"],
                    "latitude":     b["latitude"],
                    "longitude":    b["longitude"],
                    "stars":        b["stars"],
                    "review_count": b["review_count"],
                    "is_open":      b["is_open"],
                    "categories":   b.get("categories", ""),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_BIZ, index=False)
    print(f"  → {len(df):,} restaurants saved to {OUT_BIZ.name}")
    return set(df["business_id"]), df

# ── Step 1b: Extract interactions (reviews + check-ins) ──────────────────
def extract_interactions(rest_ids):
    print("▸ Extracting interactions …")
    rows = []

    # Reviews
    print("  ▹ Reviews …")
    with open(YELP_REV, "r") as f:
        for line in f:
            r = json.loads(line)
            if r["business_id"] in rest_ids:
                rows.append({
                    "user_id":     r["user_id"],
                    "business_id": r["business_id"],
                    "timestamp":   r["date"],
                    "stars":       r["stars"],
                    "type":        "review",
                })
    print(f"    {len(rows):,} review interactions")

    # Check-ins
    print("  ▹ Check-ins …")
    ci_count = 0
    with open(YELP_CHK, "r") as f:
        for line in f:
            ci = json.loads(line)
            if ci["business_id"] in rest_ids:
                for ts in (ci.get("date") or "").split(", "):
                    ts = ts.strip()
                    if ts:
                        rows.append({
                            "user_id":     "__checkin__",
                            "business_id": ci["business_id"],
                            "timestamp":   ts,
                            "stars":       None,
                            "type":        "checkin",
                        })
                        ci_count += 1
    print(f"    {ci_count:,} check-in interactions")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_INT, index=False)
    print(f"  → {len(df):,} total interactions saved to {OUT_INT.name}")
    return df

# ── Step 1c: Transit accessibility via Transitland ───────────────────────
def compute_transit_accessibility(biz_df):
    print("▸ Computing transit accessibility from Transitland …")
    if not TL_STOPS.exists():
        print("  ⚠ Transitland stops file not found, skipping transit.")
        # Create empty transit df with required columns
        transit_df = biz_df[["business_id"]].copy()
        transit_df["stops_800m"] = 0
        transit_df["departures_day_800m"] = 0
        transit_df["nearest_stop_m"] = 9999
        transit_df["transit_access_score"] = 0.0
        transit_df.to_csv(OUT_TRA, index=False)
        return transit_df

    # Load transit stops (only lat, lon, departure counts)
    print("  ▹ Loading transit stops (this may take a moment) …")
    cols_needed = ["stop_lat", "stop_lon",
                   "departure_count_dow1", "departure_count_dow2",
                   "departure_count_dow3", "departure_count_dow4",
                   "departure_count_dow5", "departure_count_dow6",
                   "departure_count_dow7"]
    stops = pd.read_csv(TL_STOPS, usecols=cols_needed, low_memory=False)
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])

    # Average daily departures
    dep_cols = [c for c in stops.columns if c.startswith("departure_count_dow")]
    stops["avg_daily_dep"] = stops[dep_cols].fillna(0).mean(axis=1)
    print(f"    {len(stops):,} transit stops loaded")

    # Build KD-tree on unit sphere
    print("  ▹ Building spatial index …")
    stop_xyz = np.array([latlon_to_xyz(r.stop_lat, r.stop_lon)
                         for r in stops.itertuples()])
    tree = cKDTree(stop_xyz)

    # Query each restaurant
    print("  ▹ Querying proximity for each restaurant …")
    chord_800 = radius_to_chord(WALKING_RADIUS_M)
    results = []
    biz_lats = biz_df["latitude"].values
    biz_lons = biz_df["longitude"].values
    biz_ids  = biz_df["business_id"].values

    # Batch query for efficiency
    biz_xyz = np.array([latlon_to_xyz(lat, lon)
                        for lat, lon in zip(biz_lats, biz_lons)])
    nearby_indices = tree.query_ball_point(biz_xyz, r=chord_800)

    for i, (bid, lat, lon) in enumerate(zip(biz_ids, biz_lats, biz_lons)):
        idxs = nearby_indices[i]
        if idxs:
            n_stops = len(idxs)
            total_dep = stops.iloc[idxs]["avg_daily_dep"].sum()
            # Nearest stop (approximate via haversine on top matches)
            dists = [haversine_m(lat, lon,
                                 stops.iloc[j]["stop_lat"],
                                 stops.iloc[j]["stop_lon"]) for j in idxs[:20]]
            nearest = min(dists) if dists else 9999
        else:
            n_stops, total_dep, nearest = 0, 0, 9999

        results.append({
            "business_id": bid,
            "stops_800m": n_stops,
            "departures_day_800m": round(total_dep, 1),
            "nearest_stop_m": round(nearest, 1),
        })

        if (i + 1) % 10000 == 0:
            print(f"    … {i+1:,}/{len(biz_ids):,}")

    transit_df = pd.DataFrame(results)
    # Normalise transit score to 0-1
    max_dep = transit_df["departures_day_800m"].quantile(0.99) or 1
    transit_df["transit_access_score"] = (
        transit_df["departures_day_800m"].clip(upper=max_dep) / max_dep
    ).round(4)

    transit_df.to_csv(OUT_TRA, index=False)
    print(f"  → Transit data saved to {OUT_TRA.name}")
    return transit_df

# ── Step 1d: Busyness proxy from check-in timestamps ────────────────────
def compute_busyness(interactions_df):
    print("▸ Computing busyness proxy from check-in timestamps …")
    checkins = interactions_df[interactions_df["type"] == "checkin"].copy()
    checkins["hour"] = pd.to_datetime(checkins["timestamp"]).dt.hour
    checkins["dow"]  = pd.to_datetime(checkins["timestamp"]).dt.dayofweek  # 0=Mon

    # Hourly counts per business
    hourly = checkins.groupby(["business_id", "dow", "hour"]).size().reset_index(name="count")

    # Normalise per business to 0-100 scale (like Google populartimes)
    def normalise_to_100(group):
        mx = group["count"].max()
        if mx > 0:
            group["busyness"] = (group["count"] / mx * 100).round(0).astype(int)
        else:
            group["busyness"] = 0
        return group

    hourly = hourly.groupby("business_id", group_keys=False).apply(normalise_to_100)

    # Summary per venue
    summary = hourly.groupby("business_id").agg(
        peak_busyness=("busyness", "max"),
        avg_busyness=("busyness", "mean"),
        peak_hour=("hour", lambda x: x.iloc[hourly.loc[x.index, "busyness"].argmax()] if len(x) > 0 else 12),
    ).reset_index()
    summary["avg_busyness"] = summary["avg_busyness"].round(1)

    summary.to_csv(OUT_BUSY, index=False)
    print(f"  → Busyness data saved to {OUT_BUSY.name}")
    return summary

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("RESTAURANT DATA EXTRACTION — 3 Datasets")
    print("=" * 60)

    rest_ids, biz_df = extract_businesses()
    interactions_df  = extract_interactions(rest_ids)
    transit_df       = compute_transit_accessibility(biz_df)
    busyness_df      = compute_busyness(interactions_df)

    print()
    print("✅ All outputs saved:")
    print(f"   {OUT_BIZ.name:40s} {len(biz_df):>8,} rows")
    print(f"   {OUT_INT.name:40s} {len(interactions_df):>8,} rows")
    print(f"   {OUT_TRA.name:40s} {len(transit_df):>8,} rows")
    print(f"   {OUT_BUSY.name:40s} {len(busyness_df):>8,} rows")

if __name__ == "__main__":
    main()
