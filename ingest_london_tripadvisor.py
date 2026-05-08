"""
Ingest TripAdvisor London restaurant reviews into the pipeline format.

Source:  Zenodo record 6583422 — "A TripAdvisor Dataset for Dyadic Context Analysis"
         https://zenodo.org/records/6583422/files/London_reviews.csv
License: CC-BY-NC 4.0 (non-commercial / academic use)

Downloads ~942 MB and converts to:
  london_interactions.csv      — user_id, business_id, timestamp, stars, type
  london_businesses.csv        — business_id, name, url, city (venue reference)

Schema mapping:
  author_id       → user_id      (stable across venues — confirmed)
  url_restaurant  → business_id  (numeric ID extracted from URL)
  date            → timestamp    ("Month DD, YYYY" → "YYYY-MM-DD 00:00:00")
  rating_review   → stars        (float 1.0–5.0)

Outputs slot directly into the same pipeline as restaurant_interactions.csv
and hotel_interactions.csv.
"""

import re
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path

DATA_DIR   = Path(__file__).parent
SOURCE_URL = "https://zenodo.org/records/6583422/files/London_reviews.csv"
RAW_PATH   = DATA_DIR / "london_reviews_raw.csv"
OUT_INTERACTIONS = DATA_DIR / "london_interactions.csv"
OUT_BUSINESSES   = DATA_DIR / "london_businesses.csv"


# ============================================================================
# Download
# ============================================================================

def download_with_progress(url: str, dest: Path) -> None:
    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"  Already downloaded: {dest.name} ({size_mb:.0f} MB) — skipping.")
        return

    print(f"Downloading {url}")
    print("  Size: ~942 MB — this will take a few minutes.")
    print("  Using curl with resume support (-C -)...")

    # curl: -L follows redirects, -C - resumes, -# shows progress bar
    result = subprocess.run(
        ["curl", "-L", "-C", "-", "-o", str(dest), url],
        check=True,
    )
    print(f"  Saved → {dest}")


# ============================================================================
# Conversion
# ============================================================================

def extract_business_id(url: str) -> str:
    """Extract numeric TripAdvisor venue ID from URL (e.g. d9994333 → 9994333)."""
    m = re.search(r"-d(\d+)-", str(url))
    return m.group(1) if m else None


def convert(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse raw CSV and return (interactions_df, businesses_df)."""
    print("Loading raw CSV...")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"  {len(df):,} rows, {df['author_id'].nunique():,} unique users")

    # Extract stable numeric business_id from URL
    print("Extracting business IDs from URLs...")
    df["business_id"] = df["url_restaurant"].apply(extract_business_id)
    missing_bid = df["business_id"].isna().sum()
    if missing_bid:
        print(f"  Warning: {missing_bid:,} rows had unparseable URLs — dropping")
        df = df.dropna(subset=["business_id"])

    # Parse date: "Month DD, YYYY" → "YYYY-MM-DD 00:00:00"
    print("Parsing dates...")
    df["timestamp"] = pd.to_datetime(
        df["date"], format="%B %d, %Y", errors="coerce"
    ).dt.strftime("%Y-%m-%d 00:00:00")
    bad_dates = df["timestamp"].isna().sum()
    if bad_dates:
        print(f"  Warning: {bad_dates:,} unparseable dates — dropping")
        df = df.dropna(subset=["timestamp"])

    # Build interactions table
    interactions = df.rename(columns={
        "author_id":     "user_id",
        "rating_review": "stars",
    })[["user_id", "business_id", "timestamp", "stars"]].copy()
    interactions["type"] = "review"
    interactions = interactions.sort_values("timestamp").reset_index(drop=True)

    # Build businesses reference table
    businesses = (
        df[["business_id", "restaurant_name", "url_restaurant", "city"]]
        .drop_duplicates(subset=["business_id"])
        .rename(columns={"restaurant_name": "name", "url_restaurant": "url"})
        .reset_index(drop=True)
    )

    return interactions, businesses


# ============================================================================
# Stats report
# ============================================================================

def print_stats(interactions: pd.DataFrame, businesses: pd.DataFrame) -> None:
    print()
    print("=" * 60)
    print("LONDON TRIPADVISOR — INGESTION REPORT")
    print("=" * 60)

    total        = len(interactions)
    n_users      = interactions["user_id"].nunique()
    n_venues     = interactions["business_id"].nunique()
    date_min     = interactions["timestamp"].min()
    date_max     = interactions["timestamp"].max()

    # User interaction density
    user_counts  = interactions.groupby("user_id").size()
    multi_venue  = (
        interactions.groupby("user_id")["business_id"].nunique()
    )
    multi_pct    = (multi_venue > 1).sum() / n_users * 100

    print(f"\n  Total reviews:         {total:>10,}")
    print(f"  Unique users:          {n_users:>10,}")
    print(f"  Unique venues:         {n_venues:>10,}")
    print(f"  Date range:            {date_min[:10]} → {date_max[:10]}")
    print(f"  Users with 2+ venues:  {(multi_venue > 1).sum():>10,}  ({multi_pct:.1f}%)")
    print(f"  Avg reviews/user:      {user_counts.mean():>10.2f}")
    print(f"  Max reviews/user:      {user_counts.max():>10,}")

    print(f"\n  Rating distribution:")
    rating_dist = interactions["stars"].value_counts().sort_index()
    for star, count in rating_dist.items():
        bar = "█" * int(count / total * 50)
        print(f"    {float(star):.0f}★  {count:>8,}  {bar}")

    print(f"\n  Top 10 venues by review count:")
    top = interactions.groupby("business_id").size().nlargest(10).reset_index()
    top.columns = ["business_id", "n_reviews"]
    top = top.merge(businesses[["business_id", "name"]], on="business_id", how="left")
    for _, r in top.iterrows():
        print(f"    {r['name'][:45]:<45}  {r['n_reviews']:>6,} reviews")

    print()
    print(f"  Yelp comparison (US coffee model):")
    print(f"    Yelp reviews:   ~630,000   |  London TripAdvisor: {total:,}")
    print(f"    Yelp users:     ~93,000    |  London users:       {n_users:,}")
    print(f"    Yelp venues:    ~8,509     |  London venues:      {n_venues:,}")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    t0 = time.time()

    # 1. Download
    download_with_progress(SOURCE_URL, RAW_PATH)

    # 2. Convert
    interactions, businesses = convert(RAW_PATH)

    # 3. Save
    interactions.to_csv(OUT_INTERACTIONS, index=False)
    businesses.to_csv(OUT_BUSINESSES, index=False)
    print(f"\nSaved:")
    print(f"  {OUT_INTERACTIONS.name}  ({len(interactions):,} rows)")
    print(f"  {OUT_BUSINESSES.name}    ({len(businesses):,} venues)")

    # 4. Stats
    print_stats(interactions, businesses)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print()
    print("Next steps:")
    print("  1. Run task1_identify_coffee_shops.py equivalent to filter venue categories")
    print("  2. Run compute_anonymous_venue_signals.py on FSQ WWW2019 UK check-ins")
    print("  3. Run run_pipeline_v5.py with london_interactions.csv as input")
