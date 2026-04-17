"""Load and prepare real venue data from project CSVs for the simulation.

v2 additions (Phase 1):
- City-matched venue pools (filter to persona's city, fallback to all)
- Discriminating candidate sets: BiRank top-5 vs Stars top-5 (non-overlapping)
- Archetype-specific sets: surface the behavioral feature each archetype cares about
"""
import os
from typing import Optional
import pandas as pd
import numpy as np
from config import BASE_DIR, CANDIDATE_POOL_SIZE


def _path(*parts):
    return os.path.join(BASE_DIR, *parts)


# ── Coffee ────────────────────────────────────────────────────────────────────

def load_coffee_venues(top_n: int = CANDIDATE_POOL_SIZE) -> pd.DataFrame:
    scores = pd.read_csv(_path("coffee_birank_venue_scores.csv"))
    businesses = pd.read_csv(_path("business_coffee_v2.csv"))
    features = pd.read_csv(_path("coffee_venue_features_v2.csv"))

    df = (
        scores.merge(businesses[["business_id", "name", "city", "state", "stars",
                                  "review_count", "categories"]], on="business_id", how="left")
              .merge(features[["business_id", "revisit_rate", "temporal_stability",
                                "gini_user_contribution", "unique_users",
                                "repeat_user_rate"]], on="business_id", how="left")
    )
    df = df.dropna(subset=["name", "stars"]).head(top_n).reset_index(drop=True)
    df["behavioral_tag"] = df.apply(_coffee_tag, axis=1)
    df["stars_rank"] = df["stars"].rank(ascending=False, method="first").astype(int)
    df["model_rank"] = df["rank"].astype(int) if "rank" in df.columns else range(1, len(df)+1)
    return df


def _coffee_tag(row) -> str:
    rr = row.get("revisit_rate", 0) or 0
    stab = row.get("temporal_stability", 0.5) or 0.5
    if rr > 0.20:
        return "High Retention — regulars keep coming back"
    elif rr > 0.05:
        return "Steady — loyal customer base"
    elif stab > 0.7:
        return "Consistent Traffic — reliable footfall all week"
    else:
        return "Broad Appeal — popular with explorers"


# ── Restaurants ───────────────────────────────────────────────────────────────

def load_restaurant_venues(top_n: int = CANDIDATE_POOL_SIZE) -> pd.DataFrame:
    businesses = pd.read_csv(_path("restaurant_businesses.csv"))
    features = pd.read_csv(_path("restaurant_venue_features.csv"))

    df = businesses.merge(
        features[["business_id", "popularity", "repeat_user_rate",
                   "gini_user_concentration", "avg_rating", "transit_access_score",
                   "cuisine_categories", "peak_busyness", "stops_800m"]],
        on="business_id", how="left"
    )
    df = df.dropna(subset=["name", "stars"])
    df["model_rank"] = df["popularity"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values("model_rank").head(top_n).reset_index(drop=True)
    df["behavioral_tag"] = df.apply(_restaurant_tag, axis=1)
    df["stars_rank"] = df["stars"].rank(ascending=False, method="first").astype(int)
    return df


def _restaurant_tag(row) -> str:
    rr = row.get("repeat_user_rate", 0) or 0
    transit = row.get("transit_access_score", 0) or 0
    busy = row.get("peak_busyness", 0) or 0
    if rr > 0.15:
        return "Local Favourite — high repeat visitor rate"
    elif transit > 0.8:
        return "Transit Accessible — easy to reach without a car"
    elif busy > 0.7:
        return "Lively & Buzzy — expect queues at peak times"
    else:
        return "Hidden Gem — underrated by stars, strong loyalists"


# ── Hotels ────────────────────────────────────────────────────────────────────

def load_hotel_venues(top_n: int = CANDIDATE_POOL_SIZE) -> pd.DataFrame:
    scores = pd.read_csv(_path("hotel_birank_venue_scores.csv"))
    features = pd.read_csv(_path("hotel_venue_features.csv"))

    df = scores.merge(
        features[["business_id", "business_leisure_ratio", "seasonal_cv",
                   "multi_stay_rate", "geographic_diversity",
                   "traveler_concentration"]],
        on="business_id", how="left"
    )
    df = df.dropna(subset=["name", "stars"])
    df = df.sort_values("birank_score", ascending=False).head(top_n).reset_index(drop=True)
    df["model_rank"] = range(1, len(df) + 1)
    df["behavioral_tag"] = df.apply(_hotel_tag, axis=1)
    df["stars_rank"] = df["stars"].rank(ascending=False, method="first").astype(int)
    return df


def _hotel_tag(row) -> str:
    bl = row.get("business_leisure_ratio", 0.5) or 0.5
    geo = row.get("geographic_diversity", 0) or 0
    ms = row.get("multi_stay_rate", 0) or 0
    if bl > 0.75:
        return "Business Hub — weekday-dominant, professional guests"
    elif bl < 0.25:
        return "Leisure Escape — weekend & holiday crowd"
    elif geo > 0.5:
        return "Destination Hotel — draws visitors from many states"
    elif ms > 0.05:
        return "Guest Favourite — unusually high return-guest rate"
    else:
        return "Reliable Stay — consistent quality across traveler types"


# ── City matching (Phase 1) ───────────────────────────────────────────────────

def _extract_city(city_str: str) -> str:
    """'Philadelphia, PA' → 'Philadelphia'"""
    return city_str.split(",")[0].strip()


def filter_by_city(df: pd.DataFrame, persona_city: str, min_venues: int = 12) -> pd.DataFrame:
    """Filter venue pool to persona's city; fall back to full pool if too few."""
    city_name = _extract_city(persona_city)
    city_col = df["city"].fillna("").str.strip()
    city_df = df[city_col.str.lower() == city_name.lower()].copy()
    if len(city_df) >= min_venues:
        return city_df.reset_index(drop=True)
    # fallback: try state match
    state = persona_city.split(",")[-1].strip() if "," in persona_city else ""
    if state and "state" in df.columns:
        state_df = df[df["state"].fillna("").str.strip().str.upper() == state.upper()].copy()
        if len(state_df) >= min_venues:
            return state_df.reset_index(drop=True)
    return df.reset_index(drop=True)


# ── Candidate set builders (Phase 1) ─────────────────────────────────────────

def build_discriminating_set(
    df: pd.DataFrame,
    n: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Phase 1 fix: BiRank top-5 vs Stars top-5 (non-overlapping).
    Forces a real tradeoff instead of easy top-vs-random.
    """
    if rng is None:
        rng = np.random.default_rng()

    df = df.copy().reset_index(drop=True)

    birank_top = set(df.nsmallest(5, "model_rank").index.tolist())
    stars_top = set(df.nsmallest(5, "stars_rank").index.tolist())

    # non-overlapping halves
    birank_only = list(birank_top - stars_top)
    stars_only = list(stars_top - birank_top)
    overlap = list(birank_top & stars_top)

    # fill to n=5 each from remaining pool
    pool = [i for i in df.index if i not in birank_top and i not in stars_top]
    rng.shuffle(pool)

    birank_side = birank_only[:5] + pool[:max(0, 5 - len(birank_only))]
    stars_side = stars_only[:5] + pool[5:max(5, 5 - len(stars_only) + 5)]
    # add overlap to birank side (they satisfy both)
    birank_side = (birank_side + overlap)[:5]
    stars_side = stars_side[:5]

    selected = list(set(birank_side + stars_side))
    if len(selected) < n:
        extra = [i for i in pool if i not in selected]
        selected += extra[:n - len(selected)]
    selected = selected[:n]

    candidate = df.loc[selected].copy()
    candidate = candidate.sample(frac=1, random_state=int(rng.integers(9999)))
    candidate = candidate.reset_index(drop=True)
    candidate["display_rank"] = range(1, len(candidate) + 1)
    return candidate


# ── Per-archetype feature emphasis (Phase 1.4) ────────────────────────────────

# Maps archetype → (venue feature column, direction: "high"/"low")
# Used to surface what each archetype actually cares about
ARCHETYPE_SIGNAL = {
    # Coffee
    "Loyalist":          ("revisit_rate",            "high"),
    "Weekday Regular":   ("temporal_stability",       "high"),
    "Casual Weekender":  ("gini_user_contribution",   "low"),   # broad loyalty = explorer appeal
    "Infrequent Visitor":("stars",                    "high"),
    # Restaurant
    "Loyalist":          ("repeat_user_rate",         "high"),
    "Explorer":          ("transit_access_score",     "high"),  # proxy for area diversity
    "Mixed / Average":   ("avg_rating",               "high"),
    "Nightlife Seeker":  ("peak_busyness",            "high"),
    # Hotel
    "One-Time Tourist (Business)": ("business_leisure_ratio", "high"),
    "Leisure Traveler":            ("geographic_diversity",    "high"),
    "One-Time Tourist":            ("stars",                   "high"),
    "Budget Explorer":             ("multi_stay_rate",         "low"),  # variety-seeking
}


def build_archetype_set(
    df: pd.DataFrame,
    archetype: str,
    n: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Build a candidate set with n/2 venues that strongly signal the archetype's
    preferred feature (high or low) vs n/2 venues that don't.
    This tests whether personas actually respond to their archetype's signal.
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_col, direction = ARCHETYPE_SIGNAL.get(archetype, ("stars", "high"))
    if signal_col not in df.columns:
        return build_discriminating_set(df, n=n, rng=rng)

    df = df.copy().reset_index(drop=True)
    df["_sig"] = pd.to_numeric(df[signal_col], errors="coerce").fillna(0)

    if direction == "high":
        top_signal = df.nlargest(n // 2, "_sig").index.tolist()
        low_signal = df.nsmallest(n // 2, "_sig").index.tolist()
    else:
        top_signal = df.nsmallest(n // 2, "_sig").index.tolist()
        low_signal = df.nlargest(n // 2, "_sig").index.tolist()

    selected = list(set(top_signal + low_signal))[:n]
    candidate = df.loc[selected].drop(columns=["_sig"]).copy()
    candidate = candidate.sample(frac=1, random_state=int(rng.integers(9999)))
    candidate = candidate.reset_index(drop=True)
    candidate["display_rank"] = range(1, len(candidate) + 1)
    return candidate


# ── Legacy: original random candidate set (kept for v1 compatibility) ─────────

def sample_candidate_set(
    df: pd.DataFrame,
    n: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """v1 method: top-5 model-ranked + 5 random decoys. Kept for compatibility."""
    if rng is None:
        rng = np.random.default_rng()
    top5 = df.head(5).copy()
    pool = df.iloc[5:].copy()
    decoys = pool.sample(min(n - 5, len(pool)), random_state=int(rng.integers(9999)))
    candidate = pd.concat([top5, decoys]).sample(frac=1, random_state=int(rng.integers(9999)))
    candidate = candidate.reset_index(drop=True)
    candidate["display_rank"] = range(1, len(candidate) + 1)
    return candidate


def get_ground_truth_rank(candidate: pd.DataFrame, model_rank_col: str = "model_rank") -> dict:
    return dict(zip(candidate.index, candidate[model_rank_col]))
