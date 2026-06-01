"""
Extract UK (GB) Foursquare WWW2019 check-ins from DuckDB into the standard
pipeline format used by run_london_pipeline.py and validate_v5.py.

Outputs (in same directory as this script):
  uk_fsq_interactions.csv   — user_id, business_id, timestamp, stars, type
  uk_fsq_businesses.csv     — business_id, name, lat, lon, category, country_code

Source: fsq.duckdb (built by ingest_foursquare.py)
  - 346,088 GB check-ins before category filter
  - 288,389 after dropping non-venue categories (home, office, road, etc.)

Run once; outputs are consumed by run_uk_fsq_pipeline.py.
"""

import duckdb
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
DB_PATH  = Path(__file__).parent.parent / "yelp_dataset" / "fsq.duckdb"

# Categories that are not public venues (no meaningful behavioral signal)
NON_VENUE_CATEGORIES = {
    "Home (private)", "Office", "Neighborhood", "Road",
    "Building", "Other Great Outdoors", "Residence",
}


def extract(db_path: Path, data_dir: Path) -> None:
    print(f"Connecting to {db_path} ...")
    con = duckdb.connect(str(db_path), read_only=True)

    # Build exclusion list for SQL
    excl = ", ".join(f"'{c}'" for c in NON_VENUE_CATEGORIES)

    print("Extracting GB check-ins (excluding non-venue categories)...")
    interactions = con.execute(f"""
        SELECT
            CAST(c.fsq_user_id AS VARCHAR)   AS user_id,
            c.fsq_venue_id                   AS business_id,
            c.utc_ts                         AS timestamp,
            NULL                             AS stars,
            'checkin'                        AS type
        FROM checkins c
        JOIN pois p ON c.fsq_venue_id = p.fsq_venue_id
        WHERE p.country_code = 'GB'
          AND p.fsq_category NOT IN ({excl})
        ORDER BY c.utc_ts
    """).df()

    print(f"  {len(interactions):,} check-ins  |  "
          f"{interactions['user_id'].nunique():,} users  |  "
          f"{interactions['business_id'].nunique():,} venues")

    print("Extracting GB venue metadata...")
    businesses = con.execute(f"""
        SELECT
            p.fsq_venue_id   AS business_id,
            p.fsq_category   AS name,
            p.lat,
            p.lon,
            p.fsq_category   AS category,
            'GB'             AS country_code
        FROM pois p
        WHERE p.country_code = 'GB'
          AND p.fsq_category NOT IN ({excl})
          AND p.fsq_venue_id IN (
              SELECT DISTINCT fsq_venue_id FROM checkins
          )
    """).df()

    print(f"  {len(businesses):,} venues with check-ins")

    out_int = data_dir / "uk_fsq_interactions.csv"
    out_biz = data_dir / "uk_fsq_businesses.csv"

    interactions.to_csv(out_int, index=False)
    businesses.to_csv(out_biz, index=False)

    print(f"\nSaved:")
    print(f"  {out_int}  ({len(interactions):,} rows)")
    print(f"  {out_biz}  ({len(businesses):,} rows)")
    con.close()


if __name__ == "__main__":
    extract(DB_PATH, DATA_DIR)
