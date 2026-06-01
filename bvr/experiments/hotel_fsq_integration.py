"""
Phase 6: Foursquare Integration for Hotels

Matches FSQ hotel-type venues (Hotel, Resort, Motel, Hostel) to our
1,466 Yelp hotel businesses via GPS proximity (75m) + category similarity.
Then links FSQ check-ins and builds a BiRank variant with FSQ social signals.

Uses the same selective social filtering as v5 (confidence >= 0.3, gamma=0.15).

Outputs:
  hotel_venue_linkage.csv      — FSQ → Yelp hotel matches
  hotel_fsq_checkins.csv       — FSQ check-ins linked to Yelp hotels
  hotel_birank_fsq_scores.csv  — BiRank scores with FSQ social priors
"""

import sys
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent
DB_PATH  = DATA_DIR / "fsq.duckdb"

RADIUS_M   = 75
CONFIDENCE_THRESHOLD = 0.4   # slightly relaxed for hotels (fewer venues)
ALPHA      = 0.85
LAMBDA_DECAY = 0.5
SOCIAL_GAMMA = 0.15
BIRANK_ITER = 200

print("=" * 65)
print("  PHASE 6: FOURSQUARE INTEGRATION — HOTELS")
print("=" * 65)

# ── Load Yelp hotels ───────────────────────────────────────────────────────────
print("\n  Loading Yelp hotel businesses...")
yelp_hotels = pd.read_csv(DATA_DIR / "hotel_businesses.csv")
yelp_hotels = yelp_hotels.dropna(subset=["latitude","longitude"])
print(f"    Yelp hotels: {len(yelp_hotels):,}")

# ── Load FSQ hotel POIs ────────────────────────────────────────────────────────
print("\n  Loading FSQ hotel venues...")
con = duckdb.connect(str(DB_PATH), read_only=True)
fsq_hotels = con.execute("""
    SELECT fsq_venue_id, lat, lon, fsq_category
    FROM pois
    WHERE country_code = 'US'
      AND (lower(fsq_category) LIKE '%hotel%'
        OR lower(fsq_category) LIKE '%hostel%'
        OR lower(fsq_category) LIKE '%motel%'
        OR lower(fsq_category) LIKE '%resort%'
        OR lower(fsq_category) LIKE '%inn%'
        OR lower(fsq_category) LIKE '%lodge%')
""").df()
print(f"    FSQ hotel venues: {len(fsq_hotels):,}")

# ── GPS + Category matching ────────────────────────────────────────────────────
print("\n  Running GPS + category matching...")

LAT_TOL = RADIUS_M / 111_000   # degrees (~0.00067°)

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi   = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# TF-IDF on categories
all_cats = (list(yelp_hotels["categories"].fillna("")) +
            list(fsq_hotels["fsq_category"].fillna("")))
tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))
tfidf.fit(all_cats)

yelp_vecs = tfidf.transform(yelp_hotels["categories"].fillna(""))
fsq_vecs  = tfidf.transform(fsq_hotels["fsq_category"].fillna(""))

# Index Yelp hotels by lat bucket for blocking
yelp_hotels["lat_bucket"] = (yelp_hotels["latitude"] / LAT_TOL).astype(int)
yelp_by_bucket = yelp_hotels.groupby("lat_bucket")

matches = []
n_fsq = len(fsq_hotels)

for fsq_idx, fsq_row in fsq_hotels.iterrows():
    fsq_lat = fsq_row["lat"]
    fsq_lon = fsq_row["lon"]
    lat_bucket = int(fsq_lat / LAT_TOL)

    # Check neighbouring lat buckets
    candidates = []
    for lb in [lat_bucket - 1, lat_bucket, lat_bucket + 1]:
        if lb in yelp_by_bucket.groups:
            candidates.append(yelp_by_bucket.get_group(lb))

    if not candidates:
        continue

    cand = pd.concat(candidates, ignore_index=True)

    # Rough lon filter (cheaper)
    lon_tol = RADIUS_M / (111_000 * np.cos(np.radians(fsq_lat)) + 1e-9)
    cand = cand[np.abs(cand["longitude"] - fsq_lon) < lon_tol]
    if cand.empty:
        continue

    # Exact haversine
    dists = haversine(
        fsq_lat, fsq_lon,
        cand["latitude"].values, cand["longitude"].values
    )
    within = cand[dists <= RADIUS_M].copy()
    within["haversine_m"] = dists[dists <= RADIUS_M]
    if within.empty:
        continue

    # Category similarity
    fsq_vec = fsq_vecs[fsq_idx]
    yelp_sub_vecs = yelp_vecs[within.index]
    cat_sims = cosine_similarity(fsq_vec, yelp_sub_vecs).flatten()
    within["category_sim"] = cat_sims

    # Confidence
    prox_score = 1.0 - within["haversine_m"] / RADIUS_M
    within["confidence"] = 0.6 * prox_score + 0.4 * within["category_sim"]

    best = within.loc[within["confidence"].idxmax()]
    if best["confidence"] >= CONFIDENCE_THRESHOLD:
        matches.append({
            "fsq_venue_id":    fsq_row["fsq_venue_id"],
            "yelp_business_id": best["business_id"],
            "yelp_name":       best.get("name",""),
            "haversine_m":     best["haversine_m"],
            "fsq_category":    fsq_row["fsq_category"],
            "yelp_categories": best.get("categories",""),
            "category_sim":    best["category_sim"],
            "confidence":      best["confidence"],
        })

linkage = pd.DataFrame(matches)
print(f"    Matched: {len(linkage):,} FSQ venues → {linkage['yelp_business_id'].nunique():,} Yelp hotels")

linkage.to_csv(DATA_DIR / "hotel_venue_linkage.csv", index=False)
print(f"  Saved: hotel_venue_linkage.csv")

if linkage.empty:
    print("  ⚠ No FSQ matches found — skipping FSQ check-in and social steps.")
    print("  Phase 6 complete (no FSQ signal for hotels).\n")
    exit()

# ── Link FSQ check-ins ─────────────────────────────────────────────────────────
print("\n  Linking FSQ check-ins to matched Yelp hotels...")
fsq_ids = tuple(linkage["fsq_venue_id"].tolist())

# Fetch check-ins for matched FSQ venues
fsq_id_str = "','".join(fsq_ids)
checkins = con.execute(f"""
    SELECT fsq_user_id, fsq_venue_id, local_ts
    FROM checkins
    WHERE fsq_venue_id IN ('{fsq_id_str}')
""").df()

print(f"    FSQ check-ins at matched hotels: {len(checkins):,}")
print(f"    Unique FSQ users: {checkins['fsq_user_id'].nunique():,}")

fsq2yelp = dict(zip(linkage["fsq_venue_id"], linkage["yelp_business_id"]))
checkins["yelp_business_id"] = checkins["fsq_venue_id"].map(fsq2yelp)

checkins.to_csv(DATA_DIR / "hotel_fsq_checkins.csv", index=False)
print(f"  Saved: hotel_fsq_checkins.csv")

# ── Social signals from FSQ friends ───────────────────────────────────────────
print("\n  Extracting FSQ social venue signals...")
fsq_user_venue = checkins.groupby(["fsq_user_id","yelp_business_id"]).size().reset_index(name="n")

social_rows = con.execute("SELECT user_a, user_b FROM social_edges LIMIT 5").df()
print(f"    Social edges sample: {social_rows}")

# For each user, get friends' visited hotels
social = con.execute("SELECT user_a, user_b FROM social_edges").df()
print(f"    Total social edges: {len(social):,}")

# Build friend visit aggregation: for each yelp hotel, count distinct friend visits
user_venue_set = checkins.groupby("fsq_user_id")["yelp_business_id"].apply(set).to_dict()
friend_venue_counts = {}

for _, row in social.iterrows():
    ua, ub = row["user_a"], row["user_b"]
    venues_b = user_venue_set.get(ub, set())
    for v in venues_b:
        friend_venue_counts[(ua, v)] = friend_venue_counts.get((ua, v), 0) + 1
    venues_a = user_venue_set.get(ua, set())
    for v in venues_a:
        friend_venue_counts[(ub, v)] = friend_venue_counts.get((ub, v), 0) + 1

social_signal = pd.DataFrame([
    {"fsq_user_id": uid, "yelp_business_id": vid, "friend_visits": cnt}
    for (uid, vid), cnt in friend_venue_counts.items()
])
print(f"    Social signals: {len(social_signal):,} user-venue pairs")

con.close()

# ── Build BiRank with FSQ social priors ───────────────────────────────────────
print("\n  Building hotel BiRank with FSQ social priors...")

# Load base BiRank scores and interactions
base_scores = pd.read_csv(DATA_DIR / "hotel_birank_venue_scores.csv")
interactions = pd.read_csv(DATA_DIR / "hotel_interactions.csv", parse_dates=["timestamp"])
venue_feat   = pd.read_csv(DATA_DIR / "hotel_venue_features.csv")
user_feat    = pd.read_csv(DATA_DIR / "hotel_user_features.csv")

reviews = interactions[interactions["source"] == "review"].dropna(subset=["user_id","business_id"]).copy()
reviews["timestamp"] = pd.to_datetime(reviews["timestamp"])
now = pd.Timestamp("2022-01-19")
reviews["age_years"] = (now - reviews["timestamp"]).dt.days / 365.0

user_rev_count = reviews.groupby("user_id").size().reset_index(name="n_hotel_reviews")
max_count = user_rev_count["n_hotel_reviews"].max()
user_rev_count["credibility"] = np.log1p(user_rev_count["n_hotel_reviews"]) / np.log1p(max_count)
credibility_map = dict(zip(user_rev_count["user_id"], user_rev_count["credibility"]))

reviews["recency_decay"] = np.exp(-LAMBDA_DECAY * reviews["age_years"])
reviews["traveler_credibility"] = reviews["user_id"].map(credibility_map).fillna(0.1)
reviews["edge_weight"] = reviews["recency_decay"] * reviews["traveler_credibility"]

edges = reviews.groupby(["user_id","business_id"])["edge_weight"].sum().reset_index()

# Add social prior to venue prior
venue_social = (social_signal.groupby("yelp_business_id")["friend_visits"]
                .sum().reset_index(name="total_friend_visits"))
max_fv = venue_social["total_friend_visits"].max()
venue_social["social_prior"] = venue_social["total_friend_visits"] / max_fv

# Venue prior: base behavioral + social
vf = venue_feat.set_index("business_id")
vs = venue_social.set_index("yelp_business_id")["social_prior"].to_dict()

users  = edges["user_id"].unique()
venues = edges["business_id"].unique()
u2i = {u: i for i, u in enumerate(users)}
v2i = {v: i for i, v in enumerate(venues)}
n_u, n_v = len(users), len(venues)
i2v = {i: v for v, i in v2i.items()}
i2u = {i: u for u, i in u2i.items()}

v_prior = np.ones(n_v)
for vid, idx in v2i.items():
    beh_score = 0.0
    if vid in vf.index:
        row = vf.loc[vid]
        beh_score = (
            0.4 * float(row.get("review_velocity", 0)) +
            0.3 * float(row.get("geographic_diversity", 0)) +
            0.2 * float(row.get("multi_stay_rate", 0)) * 50 +
            0.1 * (1.0 - float(row.get("seasonal_cv", 0.5)))
        )
    soc_score = vs.get(vid, 0.0)
    v_prior[idx] = max((1 - SOCIAL_GAMMA) * beh_score + SOCIAL_GAMMA * soc_score, 0.01)
v_prior = v_prior / v_prior.sum()

uf = user_feat.set_index("user_id")
u_prior = np.ones(n_u)
for uid, idx in u2i.items():
    if uid in uf.index:
        row = uf.loc[uid]
        u_prior[idx] = max(
            0.6 * float(row.get("hotel_frequency", 1.0)) +
            0.4 * float(row.get("hotel_city_diversity", 0.0)), 0.01
        )
u_prior = u_prior / u_prior.sum()

rows_u = [u2i[u] for u in edges["user_id"]]
cols_v = [v2i[v] for v in edges["business_id"]]
data   = edges["edge_weight"].values.astype(np.float64)
B      = sparse.csr_matrix((data, (rows_u, cols_v)), shape=(n_u, n_v))
Bt     = B.T.tocsr()

def row_norm(M):
    s = np.asarray(M.sum(axis=1)).flatten()
    s[s == 0] = 1.0
    return sparse.diags(1.0/s) @ M

B_norm  = row_norm(B)
Bt_norm = row_norm(Bt)

u_scores = u_prior.copy()
v_scores = v_prior.copy()
for it in range(BIRANK_ITER):
    u_new = ALPHA * (B_norm  @ v_scores) + (1 - ALPHA) * u_prior
    v_new = ALPHA * (Bt_norm @ u_scores) + (1 - ALPHA) * v_prior
    u_new /= u_new.sum(); v_new /= v_new.sum()
    delta = np.abs(u_new - u_scores).max() + np.abs(v_new - v_scores).max()
    u_scores, v_scores = u_new, v_new
    if delta < 1e-8:
        print(f"    Converged at iteration {it+1}")
        break

venue_scores_fsq = pd.DataFrame({
    "business_id":  [i2v[i] for i in range(n_v)],
    "birank_fsq_score": v_scores,
}).sort_values("birank_fsq_score", ascending=False)

venue_scores_fsq = venue_scores_fsq.merge(
    pd.read_csv(DATA_DIR / "hotel_businesses.csv")[
        ["business_id","name","city","state","stars","review_count"]
    ],
    on="business_id", how="left"
)

venue_scores_fsq.to_csv(DATA_DIR / "hotel_birank_fsq_scores.csv", index=False)
print(f"  Saved: hotel_birank_fsq_scores.csv")
print("\n  Phase 6 complete — ready for Phase 7 (Validation).\n")
