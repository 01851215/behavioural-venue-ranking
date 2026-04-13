"""
Phase 2: Hotel Behavioral Feature Engineering

Computes hotel-appropriate behavioral features for both venues and users.
These replace the coffee-domain features (revisit rate, burstiness, entropy)
with accommodation-domain equivalents grounded in travel behavior theory.

VENUE FEATURES:
  business_leisure_ratio  — fraction of reviews on weekdays (Mon–Thu)
                            high → business hotel, low → leisure/resort
  seasonal_cv             — coefficient of variation of monthly review volume
                            low → consistent demand, high → seasonal spike
  geographic_diversity    — Shannon entropy of reviewer home states
                            high → destination-quality, draws from many places
  multi_stay_rate         — fraction of reviewers with 2+ reviews at same hotel
                            rare but very strong loyalty signal
  review_velocity         — exponentially-weighted recent review rate
                            captures current relevance
  venue_stability_cv      — CV of annual review volume (consistency over years)
  traveler_concentration  — Gini coefficient of reviewer visit frequency
                            high → niche/specialist, low → broad appeal
  revisit_rate            — fraction of reviewers who return (for compatibility)

USER FEATURES:
  total_hotel_reviews     — how many hotels reviewed (volume)
  hotel_city_diversity    — Shannon entropy of cities reviewed (multi-city = traveler)
  hotel_state_diversity   — Shannon entropy of states reviewed
  avg_rating_given        — mean stars given to hotels
  hotel_frequency         — reviews per year active
  weekday_fraction        — proportion of reviews on weekdays (Mon–Thu)
  is_road_warrior         — flag: 10+ hotels reviewed across 3+ states
  is_leisure_traveler     — flag: mostly weekend reviews, seasonal pattern

Outputs:
  hotel_venue_features.csv
  hotel_user_features.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent

# ── Load data ──────────────────────────────────────────────────────────────────
print("=" * 65)
print("  PHASE 2: HOTEL BEHAVIORAL FEATURES")
print("=" * 65)

print("\n  Loading interactions and businesses...")
interactions = pd.read_csv(DATA_DIR / "hotel_interactions.csv",
                           parse_dates=["timestamp"])
businesses   = pd.read_csv(DATA_DIR / "hotel_businesses.csv")

# Reviews only (check-ins have no user_id)
reviews = interactions[interactions["source"] == "review"].copy()
reviews = reviews.dropna(subset=["user_id", "business_id"])
reviews["timestamp"] = pd.to_datetime(reviews["timestamp"])

print(f"    Reviews: {len(reviews):,}   Users: {reviews['user_id'].nunique():,}   "
      f"Venues: {reviews['business_id'].nunique():,}")

hotel_ids = set(businesses["business_id"])

# ── Helper: Gini coefficient ───────────────────────────────────────────────────
def gini(arr):
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2 * (idx * arr).sum() / (n * arr.sum()) - (n + 1) / n)

# ── Helper: Shannon entropy ────────────────────────────────────────────────────
def entropy(counts):
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))

# ── Derive temporal columns ────────────────────────────────────────────────────
reviews["dow"]      = reviews["timestamp"].dt.dayofweek   # 0=Mon 6=Sun
reviews["month"]    = reviews["timestamp"].dt.month
reviews["year"]     = reviews["timestamp"].dt.year
reviews["is_weekday"] = reviews["dow"] < 4  # Mon-Thu = business window

now = pd.Timestamp("2022-01-19")  # dataset end

# ── Infer user home state from ALL reviews ─────────────────────────────────────
# Use the most common state in a user's review history as home state proxy
# We need to cross-reference the user's OTHER reviews (non-hotel) to infer origin
# Faster proxy: use each user's modal review state across hotel reviews as proxy
user_states = (reviews.groupby(["user_id", "business_id"])
               .first()
               .reset_index()
               .merge(businesses[["business_id","state"]], on="business_id", how="left"))

# For geographic diversity: entropy of states visited per reviewer at each hotel
hotel_reviewer_states = (reviews
    .merge(businesses[["business_id","state"]], on="business_id", how="left")
    .rename(columns={"state": "hotel_state"}))

# ============================================================================
# VENUE FEATURES
# ============================================================================
print("\n  Computing venue features...")

venue_features = []
venue_ids = reviews["business_id"].unique()

for vid in venue_ids:
    vr = reviews[reviews["business_id"] == vid]
    n  = len(vr)
    if n < 5:
        continue

    users_at_v = vr["user_id"].values
    u_counts   = pd.Series(users_at_v).value_counts()

    # 1. business_leisure_ratio
    business_leisure_ratio = float(vr["is_weekday"].mean())

    # 2. seasonal_cv — monthly review volume CV
    monthly_counts = vr["month"].value_counts()
    seasonal_cv = float(monthly_counts.std() / monthly_counts.mean()) \
                  if len(monthly_counts) > 1 else 0.0

    # 3. multi_stay_rate — users with 2+ reviews at this hotel
    multi_reviewers = (u_counts >= 2).sum()
    multi_stay_rate = float(multi_reviewers / len(u_counts))

    # 4. venue_stability_cv — annual review volume CV
    annual = vr["year"].value_counts()
    venue_stability_cv = float(annual.std() / annual.mean()) \
                         if len(annual) > 1 else 0.0

    # 5. traveler_concentration — Gini of per-user visit counts
    traveler_concentration = gini(u_counts.values)

    # 6. review_velocity — exponentially weighted recent review rate
    ages = (now - vr["timestamp"]).dt.days.values / 365.0
    weights = np.exp(-0.5 * ages)
    review_velocity = float(weights.sum())

    # 7. revisit_rate (for compatibility with coffee model framework)
    revisit_rate = multi_stay_rate

    # 8. geographic diversity — entropy over reviewer home states
    # Use hotel states as proxy (what states do this venue's reviewers visit hotels in?)
    # This is a rough proxy — Phase 3 will refine with user home inference
    geo_div = 0.0  # placeholder; computed below from user-level data

    venue_features.append({
        "business_id":             vid,
        "n_reviews":               n,
        "n_unique_users":          len(u_counts),
        "business_leisure_ratio":  business_leisure_ratio,
        "seasonal_cv":             seasonal_cv,
        "multi_stay_rate":         multi_stay_rate,
        "venue_stability_cv":      venue_stability_cv,
        "traveler_concentration":  traveler_concentration,
        "review_velocity":         review_velocity,
        "revisit_rate":            revisit_rate,
    })

venue_feat_df = pd.DataFrame(venue_features)

# Compute geographic diversity properly:
# For each hotel, get the set of states where its reviewers also reviewed hotels
# (proxy for: do visitors come from diverse places)
user_state_map = (reviews
    .merge(businesses[["business_id","state"]], on="business_id", how="left")
    .groupby("user_id")["state"]
    .agg(lambda x: x.mode()[0] if len(x) > 0 else "unknown")
    .to_dict())

geo_div_rows = []
for vid in venue_feat_df["business_id"]:
    vr = reviews[reviews["business_id"] == vid]
    states = [user_state_map.get(u, "unknown") for u in vr["user_id"].unique()]
    state_counts = pd.Series(states).value_counts().values
    geo_div_rows.append(entropy(state_counts))

venue_feat_df["geographic_diversity"] = geo_div_rows

# Merge with business metadata
venue_feat_df = venue_feat_df.merge(
    businesses[["business_id","name","city","state","stars","review_count","subcategory"]],
    on="business_id", how="left"
)

print(f"    Venues with features: {len(venue_feat_df):,}")
print(f"\n    Feature summary:")
feat_cols = [
    "business_leisure_ratio", "seasonal_cv", "multi_stay_rate",
    "venue_stability_cv", "traveler_concentration", "review_velocity",
    "geographic_diversity"
]
for col in feat_cols:
    vals = venue_feat_df[col].dropna()
    print(f"      {col:32s}  "
          f"med={vals.median():.3f}  mean={vals.mean():.3f}  "
          f"std={vals.std():.3f}")

venue_feat_df.to_csv(DATA_DIR / "hotel_venue_features.csv", index=False)
print(f"\n  Saved: hotel_venue_features.csv")

# ============================================================================
# USER FEATURES  (fully vectorized — no per-user loop)
# ============================================================================
print("\n  Computing user features (vectorized)...")

# Annotate reviews with hotel state/city
rev_geo = reviews.merge(
    businesses[["business_id","state","city"]], on="business_id", how="left"
)

# Basic aggregations
base = (reviews.groupby("user_id").agg(
    total_hotel_reviews = ("business_id", "count"),
    n_unique_hotels     = ("business_id", "nunique"),
    avg_rating_given    = ("stars", "mean"),
    weekday_fraction    = ("is_weekday", "mean"),
    ts_min              = ("timestamp", "min"),
    ts_max              = ("timestamp", "max"),
).reset_index())

base["years_active"] = (
    (base["ts_max"] - base["ts_min"]).dt.days / 365.0 + 1.0
)
base["hotel_frequency"] = base["total_hotel_reviews"] / base["years_active"]

# Unique state and city counts per user
n_states = (rev_geo.groupby("user_id")["state"]
            .nunique().reset_index(name="n_states_visited"))
n_cities = (rev_geo.groupby("user_id")["city"]
            .nunique().reset_index(name="n_cities_visited"))

# Shannon entropy of state distribution per user
def group_entropy(series):
    counts = series.value_counts().values
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))

state_entropy = (rev_geo.dropna(subset=["state"])
                 .groupby("user_id")["state"]
                 .apply(group_entropy)
                 .reset_index(name="hotel_state_diversity"))
city_entropy  = (rev_geo.dropna(subset=["city"])
                 .groupby("user_id")["city"]
                 .apply(group_entropy)
                 .reset_index(name="hotel_city_diversity"))

# Merge all
user_feat_df = (base
    .merge(n_states,      on="user_id", how="left")
    .merge(n_cities,      on="user_id", how="left")
    .merge(state_entropy, on="user_id", how="left")
    .merge(city_entropy,  on="user_id", how="left")
)
user_feat_df = user_feat_df.fillna({"n_states_visited": 1, "n_cities_visited": 1,
                                     "hotel_state_diversity": 0.0,
                                     "hotel_city_diversity": 0.0})

user_feat_df["is_road_warrior"] = (
    (user_feat_df["total_hotel_reviews"] >= 10) &
    (user_feat_df["n_states_visited"] >= 3)
).astype(int)
user_feat_df["is_leisure_traveler"] = (
    user_feat_df["weekday_fraction"] < 0.5
).astype(int)

# Drop helper columns
user_feat_df = user_feat_df.drop(columns=["ts_min","ts_max","years_active"])

# Road warrior and leisure counts
n_warriors = user_feat_df["is_road_warrior"].sum()
n_leisure  = user_feat_df["is_leisure_traveler"].sum()
print(f"    Users with features: {len(user_feat_df):,}")
print(f"    Road warriors (10+ hotels, 3+ states): {n_warriors:,}")
print(f"    Leisure travelers (mostly weekend):     {n_leisure:,}")

print(f"\n    User feature summary:")
user_feat_cols = [
    "total_hotel_reviews", "n_unique_hotels", "n_states_visited",
    "hotel_state_diversity", "hotel_city_diversity",
    "hotel_frequency", "weekday_fraction"
]
for col in user_feat_cols:
    vals = user_feat_df[col].dropna()
    print(f"      {col:32s}  "
          f"med={vals.median():.2f}  mean={vals.mean():.2f}  "
          f"std={vals.std():.2f}")

user_feat_df.to_csv(DATA_DIR / "hotel_user_features.csv", index=False)
print(f"\n  Saved: hotel_user_features.csv")
print("\n  Phase 2 complete — ready for Phase 3 (user archetypes).\n")
