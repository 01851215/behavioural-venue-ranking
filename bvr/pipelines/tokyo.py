"""
Tokyo (JP) Foursquare Pipeline

Extracts Japanese (JP) check-ins from fsq.duckdb and runs the
full anti-loyalty hybrid pipeline for geographic generalisation.

Japan has 519,409 POIs in raw_POIs.txt, making it one of the
largest non-English-speaking markets in the dataset.

Tokyo bbox: lat 35.5–35.9, lon 139.4–139.9

Outputs:
  data/processed/tokyo_fsq_interactions.csv
  data/processed/tokyo_fsq_businesses.csv
  data/results/tokyo_fsq_venue_scores.csv
  data/results/tokyo_fsq_validation_summary.txt
"""

import sys, time, warnings, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
warnings.filterwarnings("ignore")

DATA_DIR  = Path(__file__).parent.parent.parent / "data"
DB_PATH   = Path("/Users/chris/Desktop/Master Project/yelp_dataset/fsq.duckdb")

# Tokyo bounding box
TOKYO_LAT_MIN, TOKYO_LAT_MAX = 35.5, 35.9
TOKYO_LON_MIN, TOKYO_LON_MAX = 139.4, 139.9
SPLIT_DATE = "2013-07-01"

NON_VENUE = {'Home (private)', 'Office', 'Neighborhood', 'Road',
             'Building', 'Other Great Outdoors', 'Residence'}


def extract_tokyo(db_path: Path = DB_PATH) -> tuple:
    import duckdb
    excl = ", ".join(f"'{c}'" for c in NON_VENUE)
    con = duckdb.connect(str(db_path), read_only=True)

    interactions = con.execute(f"""
        SELECT
            CAST(c.fsq_user_id AS VARCHAR) AS user_id,
            c.fsq_venue_id                 AS business_id,
            c.utc_ts                       AS timestamp,
            NULL                           AS stars,
            'checkin'                      AS type
        FROM checkins c
        JOIN pois p ON c.fsq_venue_id = p.fsq_venue_id
        WHERE p.country_code = 'JP'
          AND p.lat BETWEEN {TOKYO_LAT_MIN} AND {TOKYO_LAT_MAX}
          AND p.lon BETWEEN {TOKYO_LON_MIN} AND {TOKYO_LON_MAX}
          AND p.fsq_category NOT IN ({excl})
        ORDER BY c.utc_ts
    """).df()

    businesses = con.execute(f"""
        SELECT
            p.fsq_venue_id AS business_id,
            p.fsq_category AS name,
            p.lat, p.lon,
            p.fsq_category AS category,
            'JP'           AS country_code
        FROM pois p
        WHERE p.country_code = 'JP'
          AND p.lat BETWEEN {TOKYO_LAT_MIN} AND {TOKYO_LAT_MAX}
          AND p.lon BETWEEN {TOKYO_LON_MIN} AND {TOKYO_LON_MAX}
          AND p.fsq_category NOT IN ({excl})
          AND p.fsq_venue_id IN (SELECT DISTINCT fsq_venue_id FROM checkins)
    """).df()

    con.close()
    print(f"Tokyo: {len(interactions):,} check-ins · "
          f"{interactions['user_id'].nunique():,} users · "
          f"{interactions['business_id'].nunique():,} venues")
    return interactions, businesses


if __name__ == "__main__":
    from bvr.pipelines.extract_uk_fsq import extract as _extract_base

    print("Extracting Tokyo FSQ data...")
    interactions, businesses = extract_tokyo()

    if len(interactions) < 100:
        print("Too few Tokyo check-ins in bbox — widening to all Japan")
        import duckdb
        con = duckdb.connect(str(DB_PATH), read_only=True)
        excl = ", ".join(f"'{c}'" for c in NON_VENUE)
        interactions = con.execute(f"""
            SELECT CAST(c.fsq_user_id AS VARCHAR) AS user_id,
                   c.fsq_venue_id AS business_id, c.utc_ts AS timestamp,
                   NULL AS stars, 'checkin' AS type
            FROM checkins c JOIN pois p ON c.fsq_venue_id = p.fsq_venue_id
            WHERE p.country_code = 'JP' AND p.fsq_category NOT IN ({excl})
            ORDER BY c.utc_ts LIMIT 50000
        """).df()
        businesses = con.execute(f"""
            SELECT p.fsq_venue_id AS business_id, p.fsq_category AS name,
                   p.lat, p.lon, p.fsq_category AS category, 'JP' AS country_code
            FROM pois p WHERE p.country_code = 'JP'
              AND p.fsq_category NOT IN ({excl})
              AND p.fsq_venue_id IN (SELECT DISTINCT fsq_venue_id FROM checkins)
        """).df()
        con.close()
        print(f"Japan (all): {len(interactions):,} check-ins")

    # Save
    int_out = DATA_DIR / "processed/tokyo_fsq_interactions.csv"
    biz_out = DATA_DIR / "processed/tokyo_fsq_businesses.csv"
    interactions.to_csv(int_out, index=False)
    businesses.to_csv(biz_out, index=False)
    print(f"Saved → {int_out}")
    print(f"Saved → {biz_out}")

    if len(interactions) < 50:
        print("Insufficient data for validation pipeline — extraction only")
    else:
        # Run validation
        from bvr.pipelines.london import (
            temporal_split, compute_rising_stars, spearman_rising,
            build_birank_explore, build_mf_ranking, blend_rankings,
        )
        from bvr.core.validation import (
            compute_user_features, compute_venue_features,
            build_decayed_edges, build_adjacency, birank, evaluate_per_user,
        )

        df = interactions.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["stars"] = np.nan
        train, test_rev, train_uv, _ = temporal_split(df, SPLIT_DATE)
        if len(test_rev) >= 5:
            uf = compute_user_features(train)
            vf = compute_venue_features(train)
            decay_e = build_decayed_edges(train, SPLIT_DATE, lam=0.5)
            test_full = df[df["timestamp"] >= pd.Timestamp(SPLIT_DATE)].copy()
            rising = compute_rising_stars(train, test_full)

            rr_map = vf.set_index("business_id")["repeat_user_rate"]
            W, u2i, v2i, _, i2v = build_adjacency(decay_e)
            p0 = np.ones(len(u2i))
            q0 = np.clip(np.array([1.0/(float(rr_map.get(i2v[i],0.01) or 0.01)+0.01)
                                    for i in range(len(v2i))]), 1e-10, None)
            _, q = birank(W, p0=p0, q0=q0)
            r_anti = {i2v[i]: float(q[i]) for i in range(len(v2i))}
            r_als  = build_mf_ranking(train, "als")
            r_hybrid = blend_rankings(r_anti, r_als, lam=0.5)
            r_pop    = {v: float(c) for v, c in train["business_id"].value_counts().items()}

            rho_h, p_h = spearman_rising(r_hybrid, rising)
            rho_p, p_p = spearman_rising(r_pop, rising)

            print(f"\nTokyo validation (split {SPLIT_DATE}):")
            print(f"  hybrid_anti_loyalty_als: ρ={rho_h:+.4f} (p={p_h:.4f})")
            print(f"  baseline_popularity:     ρ={rho_p:+.4f} (p={p_p:.4f})")

            result = pd.DataFrame([
                {"domain": "tokyo", "method": "hybrid_anti_loyalty_als", "rho": rho_h, "p": p_h},
                {"domain": "tokyo", "method": "baseline_popularity",     "rho": rho_p, "p": p_p},
            ])
            result.to_csv(DATA_DIR / "results/tokyo_validation.csv", index=False)
            print(f"Saved → data/results/tokyo_validation.csv")
