"""
Phase 3: Hotel User Archetypes (K-Means Clustering)

Clusters hotel reviewers into behaviorally distinct groups using
accommodation-domain features. Expected archetypes:
  - Road Warriors: frequent multi-city business travelers
  - Leisure Travelers: weekend/holiday focused, seasonal
  - One-Time Tourists: single visit, no return pattern
  - Budget Explorers: hostel/B&B preference, high diversity

Outputs:
  hotel_user_groups.csv    — user_id → archetype mapping + features
  hotel_user_profiles.txt  — human-readable archetype profiles
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent

print("=" * 65)
print("  PHASE 3: HOTEL USER ARCHETYPES")
print("=" * 65)

# ── Load user features ─────────────────────────────────────────────────────────
print("\n  Loading user features...")
user_feat = pd.read_csv(DATA_DIR / "hotel_user_features.csv")
print(f"    Users: {len(user_feat):,}")

# ── Feature selection for clustering ──────────────────────────────────────────
cluster_features = [
    "total_hotel_reviews",    # volume of hotel experience
    "n_unique_hotels",        # diversity of venues visited
    "n_states_visited",       # geographic range
    "hotel_state_diversity",  # entropy of states (broader = more distributed)
    "hotel_city_diversity",   # entropy of cities
    "hotel_frequency",        # reviews per year (how often they travel)
    "weekday_fraction",       # business vs leisure signal
]

X_raw = user_feat[cluster_features].copy()

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X_raw)

# Cap outliers at 99th percentile before scaling
for i in range(X_imp.shape[1]):
    cap = np.percentile(X_imp[:, i], 99)
    X_imp[:, i] = np.clip(X_imp[:, i], 0, cap)

# Standardise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# ── K-Means with k=4 ──────────────────────────────────────────────────────────
# k=4 matches the coffee model structure and is interpretable
N_CLUSTERS = 4
print(f"\n  Running K-Means (k={N_CLUSTERS})...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
labels = km.fit_predict(X_scaled)
user_feat["cluster"] = labels

# ── Profile each cluster ───────────────────────────────────────────────────────
print("\n  Cluster profiles (feature means):")
profile_df = user_feat.groupby("cluster")[cluster_features + ["is_road_warrior","is_leisure_traveler"]].mean()
print(profile_df.round(3).to_string())

# ── Assign archetype names ─────────────────────────────────────────────────────
# Name clusters by their dominant characteristics
def name_cluster(row):
    # Road Warriors: high frequency, high state diversity, high weekday fraction
    if row["hotel_frequency"] > 1.5 and row["n_states_visited"] > 3 and row["weekday_fraction"] > 0.6:
        return "Road Warriors"
    # Leisure Travelers: low weekday fraction, moderate diversity
    if row["weekday_fraction"] < 0.55 and row["hotel_frequency"] < 2.0:
        return "Leisure Travelers"
    # Budget Explorers: high city diversity, low frequency
    if row["hotel_city_diversity"] > 1.0 and row["total_hotel_reviews"] > 3:
        return "Budget Explorers"
    return "One-Time Tourists"

cluster_names = {}
for c in range(N_CLUSTERS):
    row = profile_df.loc[c]
    cluster_names[c] = name_cluster(row)

# Resolve duplicate names
seen = {}
for c, name in cluster_names.items():
    if name in seen:
        # Differentiate by dominant feature
        row = profile_df.loc[c]
        if row["weekday_fraction"] > profile_df["weekday_fraction"].mean():
            cluster_names[c] = name + " (Business)"
        else:
            cluster_names[c] = name + " (Leisure)"
    seen[name] = c

user_feat["archetype"] = user_feat["cluster"].map(cluster_names)

# ── Print archetype summary ────────────────────────────────────────────────────
print("\n\n  ARCHETYPE SUMMARY")
print("=" * 65)
for c in range(N_CLUSTERS):
    name = cluster_names[c]
    members = user_feat[user_feat["cluster"] == c]
    n = len(members)
    pct = 100 * n / len(user_feat)
    row = profile_df.loc[c]
    print(f"\n  [{c}] {name}  (n={n:,}, {pct:.1f}%)")
    print(f"      Hotels reviewed:     {row['total_hotel_reviews']:.1f} avg")
    print(f"      States visited:      {row['n_states_visited']:.1f} avg")
    print(f"      Reviews/year:        {row['hotel_frequency']:.2f} avg")
    print(f"      Weekday fraction:    {row['weekday_fraction']:.1%}")
    print(f"      City diversity:      {row['hotel_city_diversity']:.3f}")
    print(f"      Road warrior flag:   {row['is_road_warrior']:.1%}")
    print(f"      Leisure flag:        {row['is_leisure_traveler']:.1%}")

# ── Save ───────────────────────────────────────────────────────────────────────
out_cols = ["user_id", "cluster", "archetype"] + cluster_features + [
    "is_road_warrior", "is_leisure_traveler"
]
user_feat[out_cols].to_csv(DATA_DIR / "hotel_user_groups.csv", index=False)
print(f"\n  Saved: hotel_user_groups.csv")

# Write human-readable profile
with open(DATA_DIR / "hotel_user_profiles.txt", "w") as f:
    f.write("HOTEL USER ARCHETYPES\n")
    f.write(f"Generated from {len(user_feat):,} hotel reviewers\n\n")
    for c in range(N_CLUSTERS):
        name = cluster_names[c]
        members = user_feat[user_feat["cluster"] == c]
        n = len(members)
        pct = 100 * n / len(user_feat)
        row = profile_df.loc[c]
        f.write(f"[{c}] {name}  (n={n:,}, {pct:.1f}%)\n")
        f.write(f"    Hotels reviewed (avg):  {row['total_hotel_reviews']:.1f}\n")
        f.write(f"    States visited (avg):   {row['n_states_visited']:.1f}\n")
        f.write(f"    Reviews/year (avg):     {row['hotel_frequency']:.2f}\n")
        f.write(f"    Weekday fraction:       {row['weekday_fraction']:.1%}\n")
        f.write(f"    City entropy:           {row['hotel_city_diversity']:.3f}\n\n")

print(f"  Saved: hotel_user_profiles.txt")
print("\n  Phase 3 complete — ready for Phase 4 (cross-domain transfer).\n")
