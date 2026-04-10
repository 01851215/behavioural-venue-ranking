"""
Phase 1C: Join FSQ check-ins to venue_linkage to get yelp_business_id.

Only keeps check-ins at FSQ venues that matched a Yelp business.
Adds local time features: hour_of_day, day_of_week, is_weekend.

Output: fsq_checkins_linked.parquet

Runtime: ~2 minutes on M5 Mac.
"""

import duckdb
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/chris/Desktop/Yelp JSON/yelp_dataset")
DB_PATH = DATA_DIR / "fsq.duckdb"
LINKAGE_PATH = DATA_DIR / "venue_linkage.csv"
OUTPUT_PATH = DATA_DIR / "fsq_checkins_linked.parquet"


def main():
    if not LINKAGE_PATH.exists():
        print(f"Missing: {LINKAGE_PATH} — run build_venue_linkage.py first.")
        return

    con = duckdb.connect(str(DB_PATH))

    # Load linkage into DuckDB for fast join
    print("Loading venue linkage...")
    con.execute(f"""
        CREATE OR REPLACE TABLE venue_linkage AS
        SELECT fsq_venue_id, yelp_business_id, confidence
        FROM read_csv('{LINKAGE_PATH}', header=true)
    """)

    linkage_count = con.execute("SELECT COUNT(*) FROM venue_linkage").fetchone()[0]
    print(f"  {linkage_count:,} venue links loaded")

    print("Joining check-ins to linked venues...")
    con.execute("""
        CREATE OR REPLACE TABLE checkins_linked AS
        SELECT
            c.fsq_user_id,
            v.yelp_business_id,
            c.fsq_venue_id,
            c.utc_ts,
            c.local_ts,
            c.tz_offset_min,
            v.confidence AS venue_link_confidence,
            HOUR(c.local_ts)                          AS hour_of_day,
            DAYOFWEEK(c.local_ts)                     AS day_of_week,
            DAYOFWEEK(c.local_ts) IN (0, 6)           AS is_weekend
        FROM checkins c
        INNER JOIN venue_linkage v ON c.fsq_venue_id = v.fsq_venue_id
    """)

    total = con.execute("SELECT COUNT(*) FROM checkins_linked").fetchone()[0]
    users = con.execute("SELECT COUNT(DISTINCT fsq_user_id) FROM checkins_linked").fetchone()[0]
    venues = con.execute("SELECT COUNT(DISTINCT yelp_business_id) FROM checkins_linked").fetchone()[0]

    print(f"  Linked check-ins: {total:,}")
    print(f"  Unique FSQ users: {users:,}")
    print(f"  Unique Yelp venues: {venues:,}")

    # Time distribution
    print("\nHour of day distribution (local time):")
    hour_dist = con.execute("""
        SELECT hour_of_day, COUNT(*) AS cnt
        FROM checkins_linked
        GROUP BY 1 ORDER BY 1
    """).df()
    peak_hour = hour_dist.loc[hour_dist["cnt"].idxmax(), "hour_of_day"]
    print(f"  Peak hour: {peak_hour}:00  |  Weekend: {con.execute('SELECT AVG(is_weekend::INT) FROM checkins_linked').fetchone()[0]:.1%}")

    print("\nExporting to parquet...")
    con.execute(f"""
        COPY checkins_linked TO '{OUTPUT_PATH}'
        (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"Saved: {OUTPUT_PATH}  ({size_mb:.1f} MB)")

    con.close()


if __name__ == "__main__":
    main()
