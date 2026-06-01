"""
Phase 3: Extract social venue signals from FSQ friendship graph.

For each Yelp venue, computes signals derived from the FSQ social graph
propagated through the user bridge table (Phase 2 output).

For each matched Yelp user u:
  - Find their FSQ identity via the bridge
  - Find their FSQ friends from the social graph
  - Find what Yelp venues those friends checked into
  - Aggregate weighted counts per venue (weighted by bridge confidence)
  - Repeat for friends-of-friends (2-hop, down-weighted x0.3)

Output columns in social_venue_signals.csv:
  yelp_business_id | friend_checkin_count | fof_checkin_count |
  social_unique_visitors | social_diversity_index | mean_bridge_confidence

Runtime: ~5-15 minutes on M5 Mac.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR      = Path("/Users/chris/Desktop/Yelp JSON/yelp_dataset")
DB_PATH       = DATA_DIR / "fsq.duckdb"
BRIDGE_PATH   = DATA_DIR / "yelp_fsq_user_bridge.csv"
CHECKINS_PATH = DATA_DIR / "fsq_checkins_linked.parquet"
OUTPUT_PATH   = DATA_DIR / "social_venue_signals.csv"

FOF_WEIGHT  = 0.3   # Friend-of-friend gets 30% of direct friend weight
FOF_CAP     = 200   # Max FoF per matched user (prevents graph explosion)


def load_bridge(min_tier="low"):
    """Load bridge table, filter to accepted tiers."""
    tiers = {"high": 3, "medium": 2, "low": 1}
    min_val = tiers[min_tier]
    df = pd.read_csv(BRIDGE_PATH)
    df = df[df["confidence_tier"].map(tiers) >= min_val]
    print(f"  Bridge pairs loaded: {len(df):,}  (tier >= {min_tier})")
    return df


def build_fsq_adjacency(con, relevant_users: set) -> dict:
    """
    Load social edges for users in or adjacent to our bridge set.
    Returns adjacency dict: {fsq_user_id: set of neighbor fsq_user_ids}
    We do two queries:
      1. Edges where user_a is in our bridge FSQ users
      2. Edges where user_b is in our bridge FSQ users
    Then union, giving us all direct friends of matched users.
    """
    users_list = list(relevant_users)
    # DuckDB parameterised IN clause via a temp table
    con.execute("CREATE OR REPLACE TEMP TABLE bridge_users (fsq_user_id BIGINT)")
    con.executemany(
        "INSERT INTO bridge_users VALUES (?)",
        [(u,) for u in users_list]
    )

    edges = con.execute("""
        SELECT user_a, user_b FROM social_edges
        WHERE user_a IN (SELECT fsq_user_id FROM bridge_users)
           OR user_b IN (SELECT fsq_user_id FROM bridge_users)
    """).df()

    print(f"  Social edges involving bridge users: {len(edges):,}")

    adj = defaultdict(set)
    for _, row in edges.iterrows():
        adj[row["user_a"]].add(row["user_b"])
        adj[row["user_b"]].add(row["user_a"])

    return dict(adj)


def build_fsq_venue_index(checkins_df: pd.DataFrame) -> dict:
    """
    Build: {fsq_user_id: {yelp_business_id: check_in_count}}
    """
    index = defaultdict(lambda: defaultdict(int))
    for _, row in checkins_df.iterrows():
        index[row["fsq_user_id"]][row["yelp_business_id"]] += 1
    return dict(index)


def compute_signals(bridge_df, adj, fsq_venue_index):
    """
    For each matched Yelp user:
      - Get their FSQ friends
      - Get what venues friends visited (weighted by bridge confidence)
      - Get what venues FoF visited (weighted by bridge_conf * FOF_WEIGHT)
    Aggregate per Yelp venue.
    """
    print(f"Computing social signals for {len(bridge_df):,} bridge users...")

    # yelp_business_id -> signals
    venue_signals = defaultdict(lambda: {
        "friend_weighted": 0.0,
        "fof_weighted":    0.0,
        "social_visitors": set(),
        "bridge_confs":    [],
    })

    yelp_to_fsq = dict(zip(bridge_df["yelp_user_id"], bridge_df["fsq_user_id"]))
    fsq_to_conf = dict(zip(bridge_df["fsq_user_id"], bridge_df["combined_similarity"]))

    for i, (yelp_uid, fsq_uid) in enumerate(yelp_to_fsq.items()):
        if i % 100 == 0:
            print(f"  Processing bridge user {i+1}/{len(yelp_to_fsq)}...")

        bridge_conf = fsq_to_conf[fsq_uid]
        friends = adj.get(fsq_uid, set())

        if not friends:
            continue

        # --- Direct friends ---
        for friend_uid in friends:
            if friend_uid not in fsq_venue_index:
                continue
            for venue_id, count in fsq_venue_index[friend_uid].items():
                venue_signals[venue_id]["friend_weighted"]  += count * bridge_conf
                venue_signals[venue_id]["social_visitors"].add(friend_uid)
                venue_signals[venue_id]["bridge_confs"].append(bridge_conf)

        # --- Friends of friends (2-hop) ---
        fof = set()
        for friend_uid in friends:
            fof |= adj.get(friend_uid, set())
        fof -= friends | {fsq_uid}          # exclude self and direct friends
        fof = list(fof)[:FOF_CAP]           # cap to control compute

        for fof_uid in fof:
            if fof_uid not in fsq_venue_index:
                continue
            for venue_id, count in fsq_venue_index[fof_uid].items():
                venue_signals[venue_id]["fof_weighted"] += count * bridge_conf * FOF_WEIGHT

    return venue_signals


def build_output(venue_signals: dict) -> pd.DataFrame:
    rows = []
    for venue_id, sig in venue_signals.items():
        n_visitors = len(sig["social_visitors"])
        rows.append({
            "yelp_business_id":        venue_id,
            "friend_checkin_count":    round(sig["friend_weighted"], 4),
            "fof_checkin_count":       round(sig["fof_weighted"], 4),
            "social_unique_visitors":  n_visitors,
            "social_diversity_index":  round(float(np.log1p(n_visitors)), 4),
            "mean_bridge_confidence":  round(
                float(np.mean(sig["bridge_confs"])) if sig["bridge_confs"] else 0.0, 4
            ),
        })

    df = pd.DataFrame(rows).sort_values("friend_checkin_count", ascending=False)
    return df


def print_summary(df: pd.DataFrame):
    print("\n=== Social Venue Signals Summary ===")
    print(f"Venues with social signals: {len(df):,}")
    print(f"\nfriend_checkin_count:")
    print(f"  mean={df['friend_checkin_count'].mean():.3f}  "
          f"max={df['friend_checkin_count'].max():.3f}  "
          f"p90={df['friend_checkin_count'].quantile(0.9):.3f}")
    print(f"fof_checkin_count:")
    print(f"  mean={df['fof_checkin_count'].mean():.3f}  "
          f"max={df['fof_checkin_count'].max():.3f}")
    print(f"social_unique_visitors:")
    print(f"  mean={df['social_unique_visitors'].mean():.1f}  "
          f"max={df['social_unique_visitors'].max()}")
    print(f"\nTop 15 venues by social signal:")
    print(df.head(15)[[
        "yelp_business_id", "friend_checkin_count", "fof_checkin_count",
        "social_unique_visitors", "social_diversity_index", "mean_bridge_confidence"
    ]].to_string(index=False))


def main():
    # Load inputs
    print("Loading bridge table...")
    bridge_df = load_bridge(min_tier="low")
    if len(bridge_df) == 0:
        print("No bridge users found — run match_cross_platform_users.py first.")
        return

    print("\nLoading FSQ check-ins...")
    checkins_df = pd.read_parquet(CHECKINS_PATH)
    print(f"  FSQ linked check-ins: {len(checkins_df):,}")

    # Build social adjacency for bridge FSQ users
    print("\nBuilding FSQ social adjacency...")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    bridge_fsq_users = set(bridge_df["fsq_user_id"].astype(int))
    adj = build_fsq_adjacency(con, bridge_fsq_users)
    con.close()

    total_friends = sum(len(v) for v in adj.values())
    print(f"  Bridge users with friends: {len(adj):,}")
    print(f"  Total friend connections:  {total_friends:,}")

    # Build FSQ venue index
    print("\nBuilding FSQ venue index...")
    fsq_venue_index = build_fsq_venue_index(checkins_df)
    print(f"  FSQ users with venue data: {len(fsq_venue_index):,}")

    # Compute signals
    print()
    venue_signals = compute_signals(bridge_df, adj, fsq_venue_index)

    # Build and save output
    output_df = build_output(venue_signals)
    print_summary(output_df)

    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({len(output_df):,} venues)")


if __name__ == "__main__":
    main()
