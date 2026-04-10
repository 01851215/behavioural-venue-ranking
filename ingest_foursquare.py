"""
Phase 1A: Ingest Foursquare WWW2019 dataset into DuckDB.

Creates fsq.duckdb with three tables:
  - pois:         FSQ venue metadata (venue_id, lat, lon, category, country)
  - checkins:     22.8M check-in events with computed local timestamps
  - social_edges: Deduplicated undirected friendship graph

Runtime: ~5 minutes on M5 Mac.
"""

import duckdb
from pathlib import Path

DATA_DIR = Path("/Users/chris/Desktop/Yelp JSON/yelp_dataset")
FSQ_DIR = DATA_DIR / "dataset_WWW2019"
DB_PATH = DATA_DIR / "fsq.duckdb"


def ingest_pois(con):
    print("Ingesting POIs...")
    con.execute("""
        CREATE OR REPLACE TABLE pois AS
        SELECT
            column0 AS fsq_venue_id,
            TRY_CAST(column1 AS DOUBLE) AS lat,
            TRY_CAST(column2 AS DOUBLE) AS lon,
            column3 AS fsq_category,
            column4 AS country_code
        FROM read_csv(
            $path,
            delim='\t',
            header=false,
            columns={
                'column0': 'VARCHAR',
                'column1': 'VARCHAR',
                'column2': 'VARCHAR',
                'column3': 'VARCHAR',
                'column4': 'VARCHAR'
            }
        )
        WHERE TRY_CAST(column1 AS DOUBLE) IS NOT NULL
          AND TRY_CAST(column2 AS DOUBLE) IS NOT NULL
    """, {"path": str(FSQ_DIR / "raw_POIs.txt")})

    count = con.execute("SELECT COUNT(*) FROM pois").fetchone()[0]
    us_count = con.execute("SELECT COUNT(*) FROM pois WHERE country_code = 'US'").fetchone()[0]
    print(f"  Total POIs: {count:,}  |  US POIs: {us_count:,}")

    con.execute("CREATE INDEX IF NOT EXISTS idx_pois_country ON pois(country_code)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_pois_venue ON pois(fsq_venue_id)")


def ingest_checkins(con):
    print("Ingesting check-ins...")
    con.execute("""
        CREATE OR REPLACE TABLE checkins AS
        SELECT
            TRY_CAST(column0 AS BIGINT) AS fsq_user_id,
            column1 AS fsq_venue_id,
            try_strptime(column2, '%a %b %d %H:%M:%S +0000 %Y') AS utc_ts,
            TRY_CAST(column3 AS INTEGER) AS tz_offset_min,
            try_strptime(column2, '%a %b %d %H:%M:%S +0000 %Y')
                + (TRY_CAST(column3 AS INTEGER) * INTERVAL '1' MINUTE) AS local_ts
        FROM read_csv(
            $path,
            delim='\t',
            header=false,
            ignore_errors=true,
            columns={
                'column0': 'VARCHAR',
                'column1': 'VARCHAR',
                'column2': 'VARCHAR',
                'column3': 'VARCHAR'
            }
        )
        WHERE TRY_CAST(column0 AS BIGINT) IS NOT NULL
          AND try_strptime(column2, '%a %b %d %H:%M:%S +0000 %Y') IS NOT NULL
    """, {"path": str(FSQ_DIR / "dataset_WWW_Checkins_anonymized.txt")})

    count = con.execute("SELECT COUNT(*) FROM checkins").fetchone()[0]
    users = con.execute("SELECT COUNT(DISTINCT fsq_user_id) FROM checkins").fetchone()[0]
    venues = con.execute("SELECT COUNT(DISTINCT fsq_venue_id) FROM checkins").fetchone()[0]
    print(f"  Check-ins: {count:,}  |  Users: {users:,}  |  Venues: {venues:,}")

    con.execute("CREATE INDEX IF NOT EXISTS idx_checkins_user ON checkins(fsq_user_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_checkins_venue ON checkins(fsq_venue_id)")


def ingest_social(con):
    print("Ingesting friendship graph...")
    con.execute("""
        CREATE OR REPLACE TABLE social_edges AS
        SELECT DISTINCT
            LEAST(TRY_CAST(column0 AS BIGINT), TRY_CAST(column1 AS BIGINT))    AS user_a,
            GREATEST(TRY_CAST(column0 AS BIGINT), TRY_CAST(column1 AS BIGINT)) AS user_b
        FROM (
            SELECT column0, column1
            FROM read_csv(
                $new_path,
                delim='\t', header=false,
                columns={'column0': 'VARCHAR', 'column1': 'VARCHAR'}
            )
            UNION ALL
            SELECT column0, column1
            FROM read_csv(
                $old_path,
                delim='\t', header=false,
                columns={'column0': 'VARCHAR', 'column1': 'VARCHAR'}
            )
        )
        WHERE TRY_CAST(column0 AS BIGINT) IS NOT NULL
          AND TRY_CAST(column1 AS BIGINT) IS NOT NULL
          AND TRY_CAST(column0 AS BIGINT) != TRY_CAST(column1 AS BIGINT)
    """, {
        "new_path": str(FSQ_DIR / "dataset_WWW_friendship_new.txt"),
        "old_path": str(FSQ_DIR / "dataset_WWW_friendship_old.txt")
    })

    edges = con.execute("SELECT COUNT(*) FROM social_edges").fetchone()[0]
    nodes = con.execute(
        "SELECT COUNT(DISTINCT u) FROM (SELECT user_a AS u FROM social_edges UNION SELECT user_b FROM social_edges)"
    ).fetchone()[0]
    print(f"  Edges: {edges:,}  |  Nodes (users): {nodes:,}")

    con.execute("CREATE INDEX IF NOT EXISTS idx_social_a ON social_edges(user_a)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_social_b ON social_edges(user_b)")


def main():
    print(f"Writing database to: {DB_PATH}\n")
    con = duckdb.connect(str(DB_PATH))

    ingest_pois(con)
    ingest_checkins(con)
    ingest_social(con)

    # Summary
    print("\nDatabase tables:")
    print(con.execute("SHOW TABLES").df().to_string(index=False))

    con.close()
    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"\nDone. fsq.duckdb = {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
