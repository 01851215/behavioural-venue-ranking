"""
Fetch London venue reference data from OpenStreetMap via Overpass API.

Pulls: cafes, restaurants, pubs, bars, hotels in London bounding box.
Outputs: london_osm_venues.csv  (name, lat, lon, category, osm_id, postcode)

This serves the same role as business_coffee_v2.csv / restaurant_businesses.csv —
a venue reference table to filter interaction data by category.

Requires: requests  (pip install requests)
"""

import time
import json
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
OUT_PATH = DATA_DIR / "london_osm_venues.csv"

# London bounding box (south, west, north, east)
LONDON_BBOX = "51.28,-0.51,51.70,0.33"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Amenity types → our category labels
CATEGORY_MAP = {
    "cafe":       "Coffee Shop",
    "restaurant": "Restaurant",
    "pub":        "Pub",
    "bar":        "Bar",
    "hotel":      "Hotel",
    "fast_food":  "Fast Food",
    "food_court": "Food Court",
}


def build_query(bbox: str) -> str:
    amenities = "|".join(CATEGORY_MAP.keys())
    return f"""
[out:json][timeout:180];
(
  node[amenity~"{amenities}"]({bbox});
  way[amenity~"{amenities}"]({bbox});
  node[tourism~"hotel|hostel|guest_house|bed_and_breakfast|motel"]({bbox});
  way[tourism~"hotel|hostel|guest_house|bed_and_breakfast|motel"]({bbox});
);
out center tags;
"""


def fetch_venues(bbox: str) -> list[dict]:
    query = build_query(bbox)
    print(f"Querying Overpass API for London ({bbox})...")
    print("  This may take 30–60 seconds...")

    headers = {"User-Agent": "BehaviouralVenueRanking/1.0 (academic research)"}
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=180, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    elements = data.get("elements", [])
    print(f"  {len(elements):,} raw elements returned")
    return elements


def parse_venues(elements: list[dict]) -> pd.DataFrame:
    rows = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()
        if not name:
            continue  # skip unnamed venues

        amenity  = tags.get("amenity", "")
        tourism  = tags.get("tourism", "")
        if tourism in ("hotel", "hostel", "guest_house", "bed_and_breakfast", "motel"):
            category = "Hotel"
            amenity  = tourism
        else:
            category = CATEGORY_MAP.get(amenity, amenity)

        # Nodes have lat/lon directly; ways have a 'center' key
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center", {})
            lat, lon = center.get("lat"), center.get("lon")

        rows.append({
            "osm_id":   el["id"],
            "osm_type": el["type"],
            "name":     name,
            "category": category,
            "amenity":  amenity,
            "lat":      lat,
            "lon":      lon,
            "postcode": tags.get("addr:postcode", ""),
            "street":   tags.get("addr:street", ""),
            "cuisine":  tags.get("cuisine", ""),
            "website":  tags.get("website", ""),
        })

    return pd.DataFrame(rows)


def print_stats(df: pd.DataFrame) -> None:
    print()
    print("=" * 50)
    print("LONDON OSM VENUES — SUMMARY")
    print("=" * 50)
    print(f"  Total venues: {len(df):,}")
    print()
    print("  By category:")
    for cat, count in df["category"].value_counts().items():
        print(f"    {cat:<20}  {count:>6,}")
    print()
    print("  Top 5 cuisines (restaurants):")
    rest = df[df["amenity"] == "restaurant"]
    for cuisine, count in rest["cuisine"].value_counts().head(5).items():
        if cuisine:
            print(f"    {cuisine:<20}  {count:>6,}")
    print("=" * 50)


if __name__ == "__main__":
    if OUT_PATH.exists():
        print(f"Already exists: {OUT_PATH.name} — delete to re-fetch.")
        df = pd.read_csv(OUT_PATH)
        print_stats(df)
    else:
        t0 = time.time()
        elements = fetch_venues(LONDON_BBOX)
        df = parse_venues(elements)
        df.to_csv(OUT_PATH, index=False)
        print(f"Saved → {OUT_PATH.name}  ({len(df):,} venues, {time.time()-t0:.0f}s)")
        print_stats(df)

    print()
    print("Next step: run ingest_london_tripadvisor.py to get review interactions,")
    print("then use build_venue_linkage.py to match TripAdvisor venues → OSM venues by GPS + name.")
