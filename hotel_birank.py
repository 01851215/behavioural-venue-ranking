"""
Phase 5: Hotel BiRank with Behavioral Priors

Runs BiRank on the hotel user-venue bipartite graph with hotel-specific
edge weights and venue/user behavioral priors.

Edge weights:
  recency_decay × traveler_credibility
  - recency_decay      = exp(-0.5 × age_years)   [same as coffee v5]
  - traveler_credibility = log(1 + user_n_hotel_reviews) / log(1 + max_reviews)
    (frequent hotel reviewers weighted up as more reliable signals)

Venue priors (combined behavioral score):
  0.4 × norm(review_velocity)
  + 0.3 × norm(geographic_diversity)
  + 0.2 × norm(multi_stay_rate)
  + 0.1 × norm(1 - seasonal_cv)    [stable = higher prior]

User priors:
  from hotel_user_features: hotel_frequency + hotel_city_diversity

Outputs:
  hotel_birank_venue_scores.csv
  hotel_birank_user_scores.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent
BIRANK_ITERATIONS = 200
ALPHA = 0.85   # damping factor (same as coffee model)
LAMBDA = 0.5   # decay half-life: 0.5 → ~1.4 year half-life

print("=" * 65)
print("  PHASE 5: HOTEL BIRANK")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n  Loading interactions and features...")
interactions = pd.read_csv(DATA_DIR / "hotel_interactions.csv",
                           parse_dates=["timestamp"])
venue_feat   = pd.read_csv(DATA_DIR / "hotel_venue_features.csv")
user_feat    = pd.read_csv(DATA_DIR / "hotel_user_features.csv")

# Reviews only (check-ins have no user_id)
reviews = interactions[interactions["source"] == "review"].dropna(
    subset=["user_id", "business_id"]
).copy()
reviews["timestamp"] = pd.to_datetime(reviews["timestamp"])

now = pd.Timestamp("2022-01-19")
reviews["age_years"] = (now - reviews["timestamp"]).dt.days / 365.0

print(f"    Reviews: {len(reviews):,}   Users: {reviews['user_id'].nunique():,}   "
      f"Venues: {reviews['business_id'].nunique():,}")

# ── Build edge list with weights ───────────────────────────────────────────────
print("\n  Building edge weights...")

# Traveler credibility: log-normalised review count per user
user_rev_count = reviews.groupby("user_id").size().reset_index(name="n_hotel_reviews")
max_count = user_rev_count["n_hotel_reviews"].max()
user_rev_count["credibility"] = (
    np.log1p(user_rev_count["n_hotel_reviews"]) /
    np.log1p(max_count)
)
credibility_map = dict(zip(user_rev_count["user_id"], user_rev_count["credibility"]))

# Recency decay
reviews["recency_decay"] = np.exp(-LAMBDA * reviews["age_years"])

# Combine: edge_weight = recency_decay × traveler_credibility
reviews["traveler_credibility"] = reviews["user_id"].map(credibility_map).fillna(0.1)
reviews["edge_weight"] = reviews["recency_decay"] * reviews["traveler_credibility"]

# Aggregate: one edge per user-venue pair (sum of weighted visits)
edges = (reviews.groupby(["user_id","business_id"])["edge_weight"]
         .sum().reset_index())
print(f"    Edges: {len(edges):,}")

# ── Build index maps ───────────────────────────────────────────────────────────
users  = edges["user_id"].unique()
venues = edges["business_id"].unique()
u2i = {u: i for i, u in enumerate(users)}
v2i = {v: i for i, v in enumerate(venues)}

n_u = len(users)
n_v = len(venues)
print(f"    Users: {n_u:,}   Venues: {n_v:,}")

# ── Venue priors ───────────────────────────────────────────────────────────────
def minmax(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else pd.Series(np.ones(len(s)), index=s.index)

vf = venue_feat.set_index("business_id")
v_prior = np.ones(n_v)

for vid_str, idx in v2i.items():
    if vid_str in vf.index:
        row = vf.loc[vid_str]
        score = (
            0.4 * float(row.get("review_velocity", 0)) +
            0.3 * float(row.get("geographic_diversity", 0)) +
            0.2 * float(row.get("multi_stay_rate", 0)) * 50 +   # scale up sparse signal
            0.1 * (1.0 - float(row.get("seasonal_cv", 0.5)))
        )
        v_prior[idx] = max(score, 0.01)

# Normalise
v_prior = v_prior / v_prior.sum()

# ── User priors ────────────────────────────────────────────────────────────────
uf = user_feat.set_index("user_id")
u_prior = np.ones(n_u)
for uid_str, idx in u2i.items():
    if uid_str in uf.index:
        row = uf.loc[uid_str]
        score = (
            0.6 * float(row.get("hotel_frequency", 1.0)) +
            0.4 * float(row.get("hotel_city_diversity", 0.0))
        )
        u_prior[idx] = max(score, 0.01)
u_prior = u_prior / u_prior.sum()

# ── Adjacency matrices ─────────────────────────────────────────────────────────
from scipy import sparse

rows_u = [u2i[u] for u in edges["user_id"]]
cols_v = [v2i[v] for v in edges["business_id"]]
data   = edges["edge_weight"].values.astype(np.float64)

B = sparse.csr_matrix((data, (rows_u, cols_v)), shape=(n_u, n_v))
Bt = B.T.tocsr()

# Row-normalise
def row_norm(M):
    row_sums = np.asarray(M.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sparse.diags(1.0 / row_sums)
    return D_inv @ M

B_norm  = row_norm(B)
Bt_norm = row_norm(Bt)

# ── BiRank iterations ──────────────────────────────────────────────────────────
print(f"\n  Running BiRank ({BIRANK_ITERATIONS} iterations)...")
u_scores = u_prior.copy()
v_scores = v_prior.copy()

for iteration in range(BIRANK_ITERATIONS):
    u_new = ALPHA * (B_norm  @ v_scores) + (1 - ALPHA) * u_prior
    v_new = ALPHA * (Bt_norm @ u_scores) + (1 - ALPHA) * v_prior
    u_new = u_new / u_new.sum()
    v_new = v_new / v_new.sum()
    delta = np.abs(u_new - u_scores).max() + np.abs(v_new - v_scores).max()
    u_scores = u_new
    v_scores = v_new
    if delta < 1e-8:
        print(f"    Converged at iteration {iteration+1}  (delta={delta:.2e})")
        break

# ── Save results ───────────────────────────────────────────────────────────────
i2v = {i: v for v, i in v2i.items()}
i2u = {i: u for u, i in u2i.items()}

venue_scores_df = pd.DataFrame({
    "business_id": [i2v[i] for i in range(n_v)],
    "birank_score": v_scores,
}).sort_values("birank_score", ascending=False)

# Merge venue metadata
venue_scores_df = venue_scores_df.merge(
    pd.read_csv(DATA_DIR / "hotel_businesses.csv")[
        ["business_id","name","city","state","stars","review_count","subcategory"]
    ],
    on="business_id", how="left"
)

user_scores_df = pd.DataFrame({
    "user_id":      [i2u[i] for i in range(n_u)],
    "birank_score": u_scores,
}).sort_values("birank_score", ascending=False)

venue_scores_df.to_csv(DATA_DIR / "hotel_birank_venue_scores.csv", index=False)
user_scores_df.to_csv(DATA_DIR / "hotel_birank_user_scores.csv",  index=False)

print(f"  Saved: hotel_birank_venue_scores.csv  ({len(venue_scores_df):,} venues)")
print(f"  Saved: hotel_birank_user_scores.csv   ({len(user_scores_df):,} users)")

# ── Top-20 preview ─────────────────────────────────────────────────────────────
print("\n  Top 20 hotels by BiRank:")
print(f"  {'Rank':<5} {'Name':<40} {'City':<15} {'Stars':<6} {'BiRank'}")
print("  " + "─" * 80)
for rank, (_, row) in enumerate(venue_scores_df.head(20).iterrows(), 1):
    name = str(row.get("name","?"))[:38]
    city = str(row.get("city","?"))[:13]
    stars = row.get("stars","?")
    score = row["birank_score"]
    print(f"  {rank:<5} {name:<40} {city:<15} {stars:<6} {score:.6f}")

print("\n  Phase 5 complete — ready for Phase 6 (Foursquare) & Phase 7 (Validation).\n")
