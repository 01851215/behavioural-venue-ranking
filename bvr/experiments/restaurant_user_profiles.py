#!/usr/bin/env python3
"""
Step 2 — User Behavioral Profiling (Car-Centric Model)
======================================================
Computes per-user behavioral features and assigns Driving Archetypes via K-Means.

Outputs:
  restaurant_user_profiles.csv
"""

import math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
INT_FILE = BASE / "restaurant_interactions.csv"
BIZ_FILE = BASE / "restaurant_businesses.csv"
OUT_FILE = BASE / "restaurant_user_profiles.csv"

EARTH_R_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))

def main():
    print("=" * 60)
    print("USER BEHAVIORAL PROFILING — CAR-CENTRIC")
    print("=" * 60)

    # Load data
    print("▸ Loading interactions …")
    interactions = pd.read_csv(INT_FILE, low_memory=False)
    # Only use reviews (they have real user_ids)
    reviews = interactions[interactions["type"] == "review"].copy()
    reviews["timestamp"] = pd.to_datetime(reviews["timestamp"])
    print(f"  {len(reviews):,} review interactions from {reviews['user_id'].nunique():,} users")

    print("▸ Loading businesses …")
    biz = pd.read_csv(BIZ_FILE)
    biz_cols = ["latitude", "longitude", "stars", "categories", "parking_lot", "parking_garage"]
    biz_info = biz.set_index("business_id")[biz_cols].to_dict("index")

    # Compute per-user features
    print("▸ Computing user features …")
    user_groups = reviews.groupby("user_id")
    results = []
    total_users = len(user_groups)

    for i, (uid, grp) in enumerate(user_groups):
        if len(grp) < 3:  # skip users with fewer than 3 reviews
            continue

        venues = grp["business_id"].values
        timestamps = grp["timestamp"].sort_values().values
        venue_counts = Counter(venues)
        n_visits = len(grp)
        n_unique = len(venue_counts)

        # ── Venue entropy (Explorer vs Loyalist) ──
        probs = np.array(list(venue_counts.values())) / n_visits
        venue_entropy = -np.sum(probs * np.log2(probs + 1e-12))

        # ── Revisit ratio ──
        revisit_ratio = (n_visits - n_unique) / n_visits if n_visits > 0 else 0

        # ── Rating features (Critic vs Casual) ──
        venue_ratings = [biz_info[v]["stars"] for v in venues if v in biz_info]
        avg_visited_rating = np.mean(venue_ratings) if venue_ratings else 3.0
        rating_std = np.std(venue_ratings) if len(venue_ratings) > 1 else 0.0

        # ── Spatial range (Preference-first vs Distance-first) ──
        locs = [(biz_info[v]["latitude"], biz_info[v]["longitude"])
                for v in set(venues) if v in biz_info and biz_info[v]["latitude"] is not None]
        if len(locs) >= 2:
            # Sample pairwise distances (max 50 pairs for speed)
            dists = []
            sample_locs = locs[:50]
            for j in range(len(sample_locs)):
                for k in range(j+1, len(sample_locs)):
                    dists.append(haversine_km(
                        sample_locs[j][0], sample_locs[j][1],
                        sample_locs[k][0], sample_locs[k][1]))
            spatial_range_km = np.percentile(dists, 90) if dists else 0.0
            # User centroid
            centroid_lat = np.mean([l[0] for l in locs])
            centroid_lon = np.mean([l[1] for l in locs])
        else:
            spatial_range_km = 0.0
            centroid_lat = locs[0][0] if locs else 0.0
            centroid_lon = locs[0][1] if locs else 0.0

        # ── Category entropy (cuisine diversity) & Nightlife Affinity ──
        cat_counter = Counter()
        nightlife_visits = 0
        easy_parking_visits = 0
        
        for v in venues:
            if v in biz_info:
                # Nightlife Affinity
                cats = str(biz_info[v].get("categories", ""))
                is_nightlife = any(kw in cats for kw in ["Bars", "Nightlife", "Lounges", "Pubs"])
                if is_nightlife:
                    nightlife_visits += 1
                
                # Parking Sensitivity (Visited venues with lot or garage)
                has_easy_parking = biz_info[v].get("parking_lot") or biz_info[v].get("parking_garage")
                if has_easy_parking:
                    easy_parking_visits += 1
                
                for c in cats.split(","):
                    c = c.strip()
                    if c:
                        cat_counter[c] += 1
                        
        if cat_counter:
            cat_total = sum(cat_counter.values())
            cat_probs = np.array(list(cat_counter.values())) / cat_total
            category_entropy = -np.sum(cat_probs * np.log2(cat_probs + 1e-12))
        else:
            category_entropy = 0.0
            
        nightlife_affinity = nightlife_visits / n_visits
        parking_sensitivity = easy_parking_visits / n_visits

        # ── Burstiness ──
        if len(timestamps) >= 3:
            intervals = np.diff(timestamps.astype("int64")) / 1e9 / 3600  # hours
            intervals = intervals[intervals > 0]
            if len(intervals) >= 2:
                mu, sigma = np.mean(intervals), np.std(intervals)
                burstiness = (sigma - mu) / (sigma + mu) if (sigma + mu) > 0 else 0
            else:
                burstiness = 0.0
        else:
            burstiness = 0.0

        results.append({
            "user_id": uid,
            "total_visits": n_visits,
            "unique_venues": n_unique,
            "venue_entropy": round(venue_entropy, 4),
            "revisit_ratio": round(revisit_ratio, 4),
            "avg_visited_rating": round(avg_visited_rating, 3),
            "rating_std": round(rating_std, 3),
            "spatial_range_km": round(spatial_range_km, 2),
            "category_entropy": round(category_entropy, 4),
            "burstiness": round(burstiness, 4),
            "nightlife_affinity": round(nightlife_affinity, 4),
            "parking_sensitivity": round(parking_sensitivity, 4),
            "centroid_lat": round(centroid_lat, 6),
            "centroid_lon": round(centroid_lon, 6),
        })

        if (i + 1) % 50000 == 0:
            print(f"  … {i+1:,}/{total_users:,}")

    user_df = pd.DataFrame(results)
    print(f"  → {len(user_df):,} user profiles computed (≥3 reviews)")

    # ── Archetype clustering (K-Means k=4) ──
    print("▸ Clustering user archetypes based on Driving Behaviors …")
    cluster_features = [
        "venue_entropy", "revisit_ratio", "total_visits", 
        "spatial_range_km", "parking_sensitivity", "nightlife_affinity"
    ]
    X = user_df[cluster_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    user_df["cluster"] = km.fit_predict(X_scaled)

    # Label assignment based on centroids (Explorer, Loyalist, Convenience Seeker, Ride-Share Candidate)
    centroids = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=cluster_features)
    
    explorer_cluster = centroids["venue_entropy"].idxmax()
    loyalist_cluster = centroids["revisit_ratio"].idxmax()
    convenience_cluster = centroids["parking_sensitivity"].idxmax()
    rideshare_cluster = centroids["nightlife_affinity"].idxmax()

    # Resolve overlapping max indices by assigning unassigned clusters if needed
    clusters = {explorer_cluster, loyalist_cluster, convenience_cluster, rideshare_cluster}
    label_map = {}
    
    if len(clusters) == 4:
        label_map = {
            explorer_cluster: "Explorer",
            loyalist_cluster: "Loyalist",
            convenience_cluster: "Convenience-Seeker",
            rideshare_cluster: "Nightlife / Ride-Share Candidate",
        }
    else:
        # Fallback to basic map if centroids overlap too much
        for c in range(4):
            if c == explorer_cluster and "Explorer" not in label_map.values():
                label_map[c] = "Explorer"
            elif c == loyalist_cluster and "Loyalist" not in label_map.values():
                label_map[c] = "Loyalist"
            elif c == convenience_cluster and "Convenience-Seeker" not in label_map.values():
                label_map[c] = "Convenience-Seeker"
            elif c == rideshare_cluster and "Nightlife / Ride-Share Candidate" not in label_map.values():
                label_map[c] = "Nightlife / Ride-Share Candidate"
            else:
                label_map[c] = "Mixed / Average"

    user_df["archetype"] = user_df["cluster"].map(label_map)

    # ── Overlay: Critic/Casual tag ──
    rating_median = user_df["avg_visited_rating"].median()
    user_df["critic_tag"] = np.where(
        user_df["avg_visited_rating"] >= rating_median + 0.3, "Critic",
        np.where(user_df["avg_visited_rating"] <= rating_median - 0.3, "Casual", "Moderate")
    )

    # ── Overlay: Driving Radiuses (Spatial Tag) ──
    # Top 30% spatial range -> Preference-First. Bottom 30% -> Distance-First.
    p70 = user_df["spatial_range_km"].quantile(0.70)
    p30 = user_df["spatial_range_km"].quantile(0.30)
    
    user_df["spatial_tag"] = np.where(
        user_df["spatial_range_km"] >= p70, "Preference-First",
        np.where(user_df["spatial_range_km"] <= p30, "Distance-First", "Balanced")
    )

    # Override Archetype slightly for output display
    print("\n  Archetype distribution:")
    for label, cnt in user_df["archetype"].value_counts().items():
        print(f"    {label:40s} {cnt:>8,} ({100*cnt/len(user_df):.1f}%)")
        
    print(f"\n  Driving Distance Tags: {user_df['spatial_tag'].value_counts().to_dict()}")

    user_df.to_csv(OUT_FILE, index=False)
    print(f"\n  → {OUT_FILE.name} saved ({len(user_df):,} rows)")


if __name__ == "__main__":
    main()
