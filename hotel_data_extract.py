"""
Phase 1: Hotel / Accommodation Data Extraction

Filters Yelp dataset to hotels, B&Bs, resorts, hostels, and motels with
50+ reviews, builds a canonical interaction table from reviews + check-ins,
and prints exploratory statistics to confirm behavioral signals exist.

Outputs:
  hotel_businesses.csv     — 1,466 quality accommodation venues
  hotel_interactions.csv   — canonical interaction table (reviews + check-ins)
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent

# ── Category keywords ──────────────────────────────────────────────────────────
HOTEL_KEYWORDS = [
    'hotels', 'hotel', 'hostel', 'motel', 'bed & breakfast',
    'bed and breakfast', 'resort', 'lodge', 'inn',
]
MIN_REVIEWS = 50   # quality floor

# ── 1. BUSINESSES ──────────────────────────────────────────────────────────────
print("=" * 65)
print("  PHASE 1: HOTEL DATA EXTRACTION")
print("=" * 65)

print("\n  Loading businesses...")
businesses = []
with open(DATA_DIR / "yelp_academic_dataset_business.json") as f:
    for line in f:
        biz = json.loads(line)
        cats = (biz.get("categories") or "").lower()
        if any(k in cats for k in HOTEL_KEYWORDS):
            businesses.append(biz)

biz_df = pd.DataFrame(businesses)
print(f"    Accommodation businesses (all): {len(biz_df):,}")

# Apply quality floor
biz_df = biz_df[biz_df["review_count"] >= MIN_REVIEWS].copy()
print(f"    After {MIN_REVIEWS}+ review filter:  {len(biz_df):,}")

# Subcategory labels
def label_subcategory(cats):
    cats = (cats or "").lower()
    if "hostel" in cats:        return "Hostel"
    if "bed & breakfast" in cats or "bed and breakfast" in cats: return "B&B"
    if "resort" in cats:        return "Resort"
    if "motel" in cats:         return "Motel"
    if "lodge" in cats:         return "Lodge"
    if "inn" in cats:           return "Inn"
    return "Hotel"

biz_df["subcategory"] = biz_df["categories"].apply(label_subcategory)

print("\n  Subcategory breakdown:")
for sub, cnt in biz_df["subcategory"].value_counts().items():
    print(f"    {sub}: {cnt}")

print(f"\n  State spread: {biz_df['state'].nunique()} states")
for state, cnt in biz_df["state"].value_counts().head(8).items():
    print(f"    {state}: {cnt}")

print(f"\n  Star distribution:")
for stars, cnt in sorted(biz_df["stars"].value_counts().items()):
    print(f"    {stars}: {cnt}")

hotel_ids = set(biz_df["business_id"])

# Save
out_cols = [
    "business_id", "name", "address", "city", "state", "postal_code",
    "latitude", "longitude", "stars", "review_count", "is_open",
    "categories", "subcategory",
]
out_cols = [c for c in out_cols if c in biz_df.columns]
biz_df[out_cols].to_csv(DATA_DIR / "hotel_businesses.csv", index=False)
print(f"\n  Saved: hotel_businesses.csv  ({len(biz_df):,} venues)")

# ── 2. REVIEWS ─────────────────────────────────────────────────────────────────
print("\n  Loading reviews...")
reviews = []
with open(DATA_DIR / "yelp_academic_dataset_review.json") as f:
    for line in f:
        r = json.loads(line)
        if r["business_id"] in hotel_ids:
            reviews.append({
                "user_id":     r["user_id"],
                "business_id": r["business_id"],
                "stars":       r.get("stars"),
                "timestamp":   r["date"],
                "source":      "review",
                "review_id":   r.get("review_id"),
                "text_len":    len(r.get("text") or ""),
            })

rev_df = pd.DataFrame(reviews)
rev_df["timestamp"] = pd.to_datetime(rev_df["timestamp"])
print(f"    Hotel reviews: {len(rev_df):,}")
print(f"    Unique users:  {rev_df['user_id'].nunique():,}")

# ── 3. CHECK-INS ───────────────────────────────────────────────────────────────
print("\n  Loading check-ins...")
checkins = []
with open(DATA_DIR / "yelp_academic_dataset_checkin.json") as f:
    for line in f:
        c = json.loads(line)
        if c["business_id"] in hotel_ids:
            for ts in c.get("date", "").split(","):
                ts = ts.strip()
                if ts:
                    checkins.append({
                        "user_id":     None,
                        "business_id": c["business_id"],
                        "stars":       None,
                        "timestamp":   ts,
                        "source":      "checkin",
                        "review_id":   None,
                        "text_len":    0,
                    })

chk_df = pd.DataFrame(checkins)
chk_df["timestamp"] = pd.to_datetime(chk_df["timestamp"], errors="coerce")
chk_df = chk_df.dropna(subset=["timestamp"])
print(f"    Hotel check-ins: {len(chk_df):,}")
print(f"    Hotels with check-ins: {chk_df['business_id'].nunique():,}")

# ── 4. CANONICAL INTERACTION TABLE ─────────────────────────────────────────────
interactions = pd.concat([rev_df, chk_df], ignore_index=True)
interactions = interactions.sort_values("timestamp").reset_index(drop=True)

print(f"\n  Total interactions: {len(interactions):,}")
print(f"  Date range: {interactions['timestamp'].min().date()} → "
      f"{interactions['timestamp'].max().date()}")

interactions.to_csv(DATA_DIR / "hotel_interactions.csv", index=False)
print(f"  Saved: hotel_interactions.csv")

# ── 5. EDA — BEHAVIORAL SIGNALS ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  EDA: BEHAVIORAL SIGNAL CHECKS")
print("=" * 65)

# Multi-stay signal
multi = (rev_df.groupby(["user_id", "business_id"])
              .size()
              .reset_index(name="n_reviews"))
multi2 = multi[multi["n_reviews"] >= 2]
multi3 = multi[multi["n_reviews"] >= 3]
total_pairs = len(multi)
print(f"\n  Multi-stay signal:")
print(f"    User-venue pairs total:   {total_pairs:,}")
print(f"    Pairs with 2+ reviews:    {len(multi2):,}  ({100*len(multi2)/total_pairs:.1f}%)")
print(f"    Pairs with 3+ reviews:    {len(multi3):,}  ({100*len(multi3)/total_pairs:.1f}%)")

# Weekday / weekend distribution
rev_df["dow"] = rev_df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
rev_df["is_weekday"] = rev_df["dow"] < 5
total_reviews = len(rev_df)
weekday_frac = rev_df["is_weekday"].mean()
print(f"\n  Weekday / weekend split:")
print(f"    Weekday reviews: {weekday_frac:.1%}   Weekend: {1-weekday_frac:.1%}")
day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
for i, name in enumerate(day_names):
    cnt = (rev_df["dow"] == i).sum()
    print(f"      {name}: {cnt:,} ({100*cnt/total_reviews:.1f}%)")

# Seasonal distribution
rev_df["month"] = rev_df["timestamp"].dt.month
monthly = rev_df["month"].value_counts().sort_index()
monthly_cv = monthly.std() / monthly.mean()
print(f"\n  Seasonal pattern (monthly CV = {monthly_cv:.3f}):")
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
for m, name in enumerate(month_names, 1):
    cnt = monthly.get(m, 0)
    print(f"      {name}: {cnt:,} ({100*cnt/total_reviews:.1f}%)")

# Per-venue review velocity
rev_df["year"] = rev_df["timestamp"].dt.year
venue_yr = rev_df.groupby(["business_id","year"]).size().reset_index(name="cnt")
venue_stability = venue_yr.groupby("business_id")["cnt"].agg(
    ["mean","std"]
).dropna()
venue_stability["cv"] = venue_stability["std"] / venue_stability["mean"]
print(f"\n  Venue traffic stability (annual review CV):")
print(f"    Median CV: {venue_stability['cv'].median():.3f}")
print(f"    Mean CV:   {venue_stability['cv'].mean():.3f}")
print(f"    Venues with CV < 0.3 (very stable): "
      f"{(venue_stability['cv'] < 0.3).sum()}")

# User hotel diversity (how many different hotels per user)
user_venue_cnt = rev_df.groupby("user_id")["business_id"].nunique()
print(f"\n  Reviewer hotel diversity:")
print(f"    Median distinct hotels per user: {user_venue_cnt.median():.0f}")
print(f"    Users with 5+ hotels reviewed:  {(user_venue_cnt >= 5).sum():,}")
print(f"    Users with 10+ hotels reviewed: {(user_venue_cnt >= 10).sum():,}")
print(f"    (potential road warriors ≥10):  {(user_venue_cnt >= 10).sum():,}")

print("\n  Done. All signals confirmed — ready for Phase 2.\n")
