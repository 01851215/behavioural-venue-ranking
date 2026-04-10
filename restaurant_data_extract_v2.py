#!/usr/bin/env python3
"""
Step 1 — Data Extraction & Foursquare Linking (Car-Centric Model)
=================================================
Extracts restaurant data from three datasets:
  1. Yelp  — businesses, reviews, check-ins, PARKING
  2. Foursquare — POIs & Global Checkins (WWW2019) -> replaces Populartimes

Outputs:
  restaurant_businesses.csv
  restaurant_interactions.csv
  restaurant_busyness.csv     (per-venue hourly busyness 0-100 from FSQ)
"""

import json, csv, sys, math, ast
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


FSQ_POIS = first_existing([
    BASE / "dataset_WWW2019" / "raw_POIs.txt",
    PROJECT / "dataset_WWW2019" / "raw_POIs.txt",
])
FSQ_CHKS = first_existing([
    BASE / "dataset_WWW2019" / "dataset_WWW_Checkins_anonymized.txt",
    PROJECT / "dataset_WWW2019" / "dataset_WWW_Checkins_anonymized.txt",
])

OUT_BIZ  = BASE / "restaurant_businesses.csv"
OUT_INT  = BASE / "restaurant_interactions.csv"
OUT_BUSY = BASE / "restaurant_busyness.csv"

RESTAURANT_KEYWORDS = {"Restaurants", "Food"}
EARTH_RADIUS_KM    = 6371.0

# ── Helpers ──────────────────────────────────────────────────────────────
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

# ── Step 1a: Extract restaurant businesses + Parking ─────────────────────
def extract_businesses():
    print("▸ Extracting restaurant businesses …")
    rows = []
    with open(YELP_BIZ, "r") as f:
        for line in f:
            b = json.loads(line)
            cats = set(c.strip() for c in (b.get("categories") or "").split(","))
            if cats & RESTAURANT_KEYWORDS:
                attrs = b.get("attributes") or {}
                parking_str = attrs.get("BusinessParking", "{}")
                parking = {"lot": False, "street": False, "valet": False, "garage": False}
                if parking_str and parking_str != "None":
                    try:
                        parking = ast.literal_eval(parking_str)
                    except:
                        pass
                
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
                    "parking_lot":  parking.get("lot", False) if isinstance(parking, dict) else False,
                    "parking_street": parking.get("street", False) if isinstance(parking, dict) else False,
                    "parking_valet": parking.get("valet", False) if isinstance(parking, dict) else False,
                    "parking_garage": parking.get("garage", False) if isinstance(parking, dict) else False,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_BIZ, index=False)
    print(f"  → {len(df):,} restaurants saved to {OUT_BIZ.name}")
    return set(df["business_id"]), df

# ── Step 1b: Extract interactions (reviews + check-ins) ──────────────────
def extract_interactions(rest_ids):
    print("▸ Extracting interactions (Yelp) …")
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

# ── Step 1c: Match Foursquare POIs to Yelp ───────────────────────────────
def match_foursquare_pois(biz_df):
    print("▸ Matching Foursquare POIs to Yelp restaurants …")
    if not FSQ_POIS.exists():
        print("  ⚠ Foursquare POI file not found, returning empty mapping.")
        return {}
        
    biz_lats = biz_df["latitude"].values
    biz_lons = biz_df["longitude"].values
    biz_ids = biz_df["business_id"].values
    
    biz_xyz = np.array([latlon_to_xyz(lat, lon) for lat, lon in zip(biz_lats, biz_lons)])
    tree = cKDTree(biz_xyz)
    
    fsq_mapping = {}
    chord_50 = radius_to_chord(50)  # 50m radius match
    
    matched_count = 0
    with open(FSQ_POIS, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                fsq_id = parts[0]
                try:
                    lat, lon = float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                
                xyz = latlon_to_xyz(lat, lon)
                dists, idxs = tree.query(xyz, k=1, distance_upper_bound=chord_50)
                if dists != np.inf:
                    yelp_id = biz_ids[idxs]
                    fsq_mapping[fsq_id] = yelp_id
                    matched_count += 1

    print(f"  → Matched {matched_count:,} Foursquare POIs to Yelp restaurants.")
    return fsq_mapping

# ── Step 1d: Busyness proxy from Foursquare check-ins ────────────────────
def compute_foursquare_busyness(biz_df, fsq_mapping):
    print("▸ Computing busyness proxy from Foursquare check-ins …")
    if not fsq_mapping or not FSQ_CHKS.exists():
        print("  ⚠ No FSQ mapping or check-in file found. Using default busyness.")
        busyness_df = biz_df[["business_id"]].copy()
        busyness_df["peak_busyness"] = 0
        busyness_df["avg_busyness"] = 0.0
        busyness_df["peak_hour"] = 12
        busyness_df.to_csv(OUT_BUSY, index=False)
        return busyness_df
        
    print("  ▹ Reading 1.5GB FSQ Check-ins file...")
    
    # Store counts: hourly[yelp_id][dow][hour]
    hourly = defaultdict(lambda: np.zeros((7, 24)))
    
    count = 0
    with open(FSQ_CHKS, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                fsq_id = parts[1]
                yelp_id = fsq_mapping.get(fsq_id)
                if yelp_id:
                    utc_time_str = parts[2]  # e.g. "Tue Apr 03 18:00:09 +0000 2012"
                    offset_mins = parts[3]
                    try:
                        # Fast approximation of hour and DOW parsing
                        # Datetime formats are slow. We know slicing: Time is always char 11-13
                        # We just fallback to strptime if it fails
                        dt = datetime.strptime(utc_time_str, "%a %b %d %H:%M:%S +0000 %Y")
                        offset = int(offset_mins)
                        local_time = dt + pd.Timedelta(minutes=offset)
                        hourly[yelp_id][local_time.weekday()][local_time.hour] += 1
                        count += 1
                    except Exception as e:
                        pass
    
    print(f"  → Processed {count:,} FSQ check-ins matching Yelp restaurants.")
    
    rows = []
    # yelp_ids that got at least one checkin
    for yid, mat in hourly.items():
        max_count = np.max(mat)
        if max_count > 0:
            avg_busyness = np.mean(mat / max_count * 100)
            peak_hour = np.unravel_index(mat.argmax(), mat.shape)[1]
            rows.append({
                "business_id": yid,
                "peak_busyness": 100,
                "avg_busyness": round(avg_busyness, 1),
                "peak_hour": int(peak_hour)
            })

    summary = pd.DataFrame(rows)
    print(f"  → {len(summary):,} / {len(biz_df):,} restaurants have FSQ busyness data.")
    
    if len(summary) > 0:
        busyness_df = biz_df[["business_id"]].copy()
        busyness_df = busyness_df.merge(summary, on="business_id", how="left")
        busyness_df["peak_busyness"] = busyness_df["peak_busyness"].fillna(0)
        busyness_df["avg_busyness"] = busyness_df["avg_busyness"].fillna(0.0)
        busyness_df["peak_hour"] = busyness_df["peak_hour"].fillna(12)
    else:
        busyness_df = biz_df[["business_id"]].copy()
        busyness_df["peak_busyness"] = 0
        busyness_df["avg_busyness"] = 0.0
        busyness_df["peak_hour"] = 12
        
    busyness_df.to_csv(OUT_BUSY, index=False)
    print(f"  → Busyness data saved to {OUT_BUSY.name}")
    return busyness_df

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("RESTAURANT DATA EXTRACTION — FSQ Car-Centric Integration")
    print("=" * 60)

    rest_ids, biz_df = extract_businesses()
    interactions_df  = extract_interactions(rest_ids)
    
    fsq_mapping = match_foursquare_pois(biz_df)
    busyness_df = compute_foursquare_busyness(biz_df, fsq_mapping)

    print()
    print("✅ All outputs saved:")
    print(f"   {OUT_BIZ.name:40s} {len(biz_df):>8,} rows")
    print(f"   {OUT_INT.name:40s} {len(interactions_df):>8,} rows")
    print(f"   {OUT_BUSY.name:40s} {len(busyness_df):>8,} rows")

if __name__ == "__main__":
    main()
