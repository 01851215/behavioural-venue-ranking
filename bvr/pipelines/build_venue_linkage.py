"""
Phase 1B: Build venue linkage table matching Foursquare POIs to Yelp businesses.

Method: GPS bounding-box blocking (1° grid cells) + haversine exact distance
        + TF-IDF category cosine similarity.

Confidence = 0.6 * proximity_score + 0.4 * category_sim
Accept if confidence >= 0.5 and haversine <= 75m.

Output: venue_linkage.csv
  fsq_venue_id | yelp_business_id | haversine_m | category_sim | confidence | match_method

Runtime: ~20-40 minutes on M5 Mac.
"""

import json
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("/Users/chris/Desktop/Yelp JSON/yelp_dataset")
DB_PATH = DATA_DIR / "fsq.duckdb"
YELP_BUSINESS_JSON = DATA_DIR / "yelp_academic_dataset_business.json"
OUTPUT_PATH = DATA_DIR / "venue_linkage.csv"

RADIUS_M = 75
LAT_TOL = RADIUS_M / 111_000          # ~0.00067 degrees
CONFIDENCE_THRESHOLD = 0.5
MIN_CATEGORY_SIM = 0.1                 # Accept even weak category match if GPS is very close


def load_yelp_businesses():
    print("Loading Yelp businesses...")
    records = []
    with open(YELP_BUSINESS_JSON, "r") as f:
        for line in f:
            b = json.loads(line)
            lat = b.get("latitude")
            lon = b.get("longitude")
            if lat is None or lon is None:
                continue
            records.append({
                "business_id": b["business_id"],
                "name": b.get("name", ""),
                "lat": float(lat),
                "lon": float(lon),
                "categories": b.get("categories") or "",
                "city": b.get("city", ""),
                "state": b.get("state", ""),
            })
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} Yelp businesses with coordinates")
    return df


def haversine_vectorised(lat1, lon1, lat2, lon2):
    """Returns distance in metres for numpy arrays."""
    R = 6_371_000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_vectorizer(yelp_df, con):
    """Fit TF-IDF on union of Yelp categories and US FSQ categories."""
    print("Building TF-IDF vocabulary...")
    # Sample FSQ categories to fit vectoriser (avoid loading all 11M)
    fsq_cats = con.execute("""
        SELECT DISTINCT fsq_category
        FROM pois
        WHERE country_code = 'US'
          AND fsq_category IS NOT NULL
    """).df()["fsq_category"].tolist()

    yelp_cats = yelp_df["categories"].fillna("").tolist()
    all_cats = yelp_cats + fsq_cats

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-zA-Z&]+",
        lowercase=True,
        min_df=2,
        max_features=5000,
    )
    vectorizer.fit(all_cats)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
    return vectorizer


def match_venues(yelp_df, con, vectorizer):
    print("Matching venues by grid cell...")

    yelp_vecs = vectorizer.transform(yelp_df["categories"].fillna(""))

    # Add grid cell columns — use floor (not truncation) so negative lons work correctly
    # e.g. lon=-75.2 → floor=-76 (not -75), so query covers -76 to -75 correctly
    yelp_df = yelp_df.copy()
    yelp_df["grid_lat"] = np.floor(yelp_df["lat"]).astype(int)
    yelp_df["grid_lon"] = np.floor(yelp_df["lon"]).astype(int)

    matches = []
    total_cells = yelp_df.groupby(["grid_lat", "grid_lon"]).ngroups
    cell_count = 0

    for (glat, glon), chunk in yelp_df.groupby(["grid_lat", "grid_lon"]):
        cell_count += 1
        if cell_count % 50 == 0:
            print(f"  Processing cell {cell_count}/{total_cells} ({len(matches)} matches so far)...")

        cos_lat = np.cos(np.radians(glat + 0.5))
        lon_tol = LAT_TOL / max(cos_lat, 0.01)

        lat_min = glat - LAT_TOL - 0.01
        lat_max = glat + 1 + LAT_TOL + 0.01
        lon_min = glon - lon_tol - 0.01
        lon_max = glon + 1 + lon_tol + 0.01

        fsq_cands = con.execute("""
            SELECT fsq_venue_id, lat, lon, fsq_category
            FROM pois
            WHERE country_code = 'US'
              AND lat BETWEEN ? AND ?
              AND lon BETWEEN ? AND ?
              AND fsq_category IS NOT NULL
        """, [lat_min, lat_max, lon_min, lon_max]).df()

        if len(fsq_cands) == 0:
            continue

        fsq_vecs = vectorizer.transform(fsq_cands["fsq_category"].fillna(""))

        for i, yelp_row in chunk.iterrows():
            dists = haversine_vectorised(
                yelp_row["lat"], yelp_row["lon"],
                fsq_cands["lat"].values,
                fsq_cands["lon"].values,
            )

            within = dists <= RADIUS_M
            if not within.any():
                continue

            cands = fsq_cands[within].copy()
            cands["dist_m"] = dists[within]

            # Category cosine similarity
            yelp_idx = yelp_df.index.get_loc(i)
            fsq_vecs_sub = fsq_vecs[within]
            cat_sims = cosine_similarity(yelp_vecs[yelp_idx], fsq_vecs_sub).flatten()
            cands["category_sim"] = cat_sims

            # Proximity score: exponential decay, half-score at 30m
            prox = np.exp(-cands["dist_m"].values / 30.0)
            cands["confidence"] = 0.6 * prox + 0.4 * cat_sims

            best = cands.nlargest(1, "confidence").iloc[0]

            if best["confidence"] >= CONFIDENCE_THRESHOLD and best["category_sim"] >= MIN_CATEGORY_SIM:
                matches.append({
                    "fsq_venue_id": best["fsq_venue_id"],
                    "yelp_business_id": yelp_row["business_id"],
                    "yelp_name": yelp_row["name"],
                    "lat_fsq": round(best["lat"], 6),
                    "lon_fsq": round(best["lon"], 6),
                    "lat_yelp": round(yelp_row["lat"], 6),
                    "lon_yelp": round(yelp_row["lon"], 6),
                    "haversine_m": round(best["dist_m"], 1),
                    "fsq_category": best["fsq_category"],
                    "yelp_categories": yelp_row["categories"],
                    "category_sim": round(best["category_sim"], 4),
                    "confidence": round(best["confidence"], 4),
                    "match_method": "gps+category",
                })

    return pd.DataFrame(matches)


def deduplicate(df):
    """Each FSQ venue maps to at most one Yelp business (best confidence)."""
    before = len(df)
    df = df.sort_values("confidence", ascending=False)
    df = df.drop_duplicates(subset=["fsq_venue_id"], keep="first")
    df = df.drop_duplicates(subset=["yelp_business_id"], keep="first")
    print(f"  Deduplicated: {before:,} → {len(df):,} matches")
    return df


def quality_report(df):
    print("\nQuality report:")
    print(f"  Total matches:         {len(df):,}")
    print(f"  Haversine < 20m:       {(df['haversine_m'] < 20).sum():,}  ({(df['haversine_m'] < 20).mean():.1%})")
    print(f"  Haversine < 50m:       {(df['haversine_m'] < 50).sum():,}  ({(df['haversine_m'] < 50).mean():.1%})")
    print(f"  Confidence >= 0.7:     {(df['confidence'] >= 0.7).sum():,}  ({(df['confidence'] >= 0.7).mean():.1%})")
    print(f"  Category sim mean:     {df['category_sim'].mean():.3f}")
    print(f"  Category sim >= 0.3:   {(df['category_sim'] >= 0.3).sum():,}  ({(df['category_sim'] >= 0.3).mean():.1%})")
    print("\nSample matches:")
    sample = df.sample(min(10, len(df)))[
        ["yelp_name", "yelp_categories", "fsq_category", "haversine_m", "category_sim", "confidence"]
    ]
    print(sample.to_string(index=False))


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    yelp_df = load_yelp_businesses()
    vectorizer = build_vectorizer(yelp_df, con)
    matches_df = match_venues(yelp_df, con, vectorizer)
    con.close()

    if len(matches_df) == 0:
        print("No matches found — check radius or confidence threshold.")
        return

    matches_df = deduplicate(matches_df)
    quality_report(matches_df)

    matches_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
