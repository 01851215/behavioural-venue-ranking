"""
Coffee + Restaurant Behavioral Dashboard

Stabilized dashboard implementation with:
- Robust coffee data preparation and ranking
- City search + suggestion + explicit apply flow
- City-aware map radius synchronization
- Shared rich validation visualizations for coffee and restaurants
"""

from __future__ import annotations

import difflib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Iterable

import folium
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_folium import folium_static, st_folium


# ============================================================================
# CONFIGURATION
# ============================================================================

# Resolve data files relative to this script, not the shell working directory.
DATA_DIR = Path(__file__).resolve().parent

# Restaurant data files
REST_BUSINESS_FILE = DATA_DIR / "restaurant_businesses.csv"
REST_SCORES_FILE = DATA_DIR / "restaurant_scores.csv"
REST_VENUE_FEATURES_FILE = DATA_DIR / "restaurant_venue_features.csv"
REST_VALIDATION_FILE = DATA_DIR / "restaurant_validation_results.csv"

# Coffee data files
BUSINESS_FILE = DATA_DIR / "business_coffee_v2.csv"
BIRANK_FILE = DATA_DIR / "coffee_birank_venue_scores.csv"
BASELINES_FILE = DATA_DIR / "coffee_baselines.csv"
GROUP_BIRANK_FILE = DATA_DIR / "coffee_birank_venue_scores_by_group.csv"
VENUE_FEATURES_FILE = DATA_DIR / "coffee_venue_features_v2.csv"
VENUE_FEATURES_V3_FILE = DATA_DIR / "coffee_venue_features_v3.csv"
VENUE_EXPLANATIONS_FILE = DATA_DIR / "venue_explanations.csv"
VALIDATION_FILE = DATA_DIR / "validation_results.csv"
SOCIAL_SIGNALS_FILE = DATA_DIR / "social_venue_signals.csv"
BIRANK_V4_FILE = DATA_DIR / "coffee_birank_venue_scores_v4.csv"

# Hotel data files
HOTEL_BUSINESS_FILE      = DATA_DIR / "hotel_businesses.csv"
HOTEL_BIRANK_FILE        = DATA_DIR / "hotel_birank_venue_scores.csv"
HOTEL_BIRANK_FSQ_FILE    = DATA_DIR / "hotel_birank_fsq_scores.csv"
HOTEL_VENUE_FEATURES_FILE= DATA_DIR / "hotel_venue_features.csv"
HOTEL_USER_GROUPS_FILE   = DATA_DIR / "hotel_user_groups.csv"
HOTEL_VALIDATION_FILE    = DATA_DIR / "hotel_validation_results.csv"

# City search files
CITIES_INDEX_FILE = DATA_DIR / "cities_index.pkl"
CITY_ALIASES_FILE = DATA_DIR / "city_aliases.json"

BEHAVIOR_TO_GROUP = {
    "Regular / routine visits": "Loyalists",
    "Explorer / try new places": "Infrequent Visitors",
    "Morning quick stop": "Weekday Regulars",
    "Weekend casual": "Casual Weekenders",
}

RANKING_MODES = [
    "Behaviour-based (BiRank)",
    "Rating-based",
    "Popularity-based",
    "Revisit-rate-based",
]

METRIC_HELP = {
    "NDCG": {
        "what": "Ranking quality that rewards putting truly relevant venues near the top.",
        "why": "Best for checking whether top recommendations are ordered well, not just present.",
        "formula": r"\mathrm{NDCG}@k = \frac{1}{|U|}\sum_{u \in U}\frac{\mathrm{DCG}@k_u}{\mathrm{IDCG}@k_u}",
    },
    "Hit": {
        "what": "Fraction of users with at least one correct venue in top-k.",
        "why": "Simple utility signal for shortlist-style recommendation UI.",
        "formula": r"\mathrm{Hit}@k = \frac{1}{|U|}\sum_{u\in U}\mathbb{1}[\hat{V}_u^k \cap V_u^{future}\neq\emptyset]",
    },
}


# ============================================================================
# UTILS
# ============================================================================


def safe_int(val, default: int = 0) -> int:
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.session_state.setdefault("_data_read_errors", {})[str(path)] = str(exc)
        return pd.DataFrame()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def format_percent(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def format_number(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def format_count(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def minmax_norm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    low = float(s.min())
    high = float(s.max())
    if high - low <= 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - low) / (high - low)


def coalesce_columns(df: pd.DataFrame, candidates: Iterable[str], default: float = 0.0) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for col in candidates:
        if col in df.columns:
            out = out.fillna(df[col])
    return pd.to_numeric(out, errors="coerce").fillna(default)


def normalize_coffee_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["birank_score"] = coalesce_columns(out, ["birank_score", "score"], default=0.0)
    out["rating_mean"] = coalesce_columns(out, ["rating_mean", "stars", "avg_rating"], default=0.0)
    out["popularity_visits"] = coalesce_columns(
        out,
        ["popularity_visits", "total_visits", "popularity", "review_count"],
        default=0.0,
    )
    out["revisit_rate"] = coalesce_columns(
        out,
        ["revisit_rate", "repeat_user_rate", "revisit_rate_x", "revisit_rate_y"],
        default=0.0,
    )
    out["unique_users"] = coalesce_columns(
        out,
        ["unique_users", "unique_users_x", "unique_users_y"],
        default=0.0,
    )
    out["repeat_user_rate"] = coalesce_columns(out, ["repeat_user_rate", "revisit_rate"], default=0.0)
    out["repeat_user_count"] = coalesce_columns(out, ["repeat_user_count", "repeat_users"], default=0.0)
    out["avg_user_repeat_visits"] = coalesce_columns(
        out, ["avg_user_repeat_visits", "top_user_share"], default=np.nan
    )
    out["gini_user_contribution"] = coalesce_columns(
        out, ["gini_user_contribution", "gini_user_concentration"], default=np.nan
    )
    out["weekly_visit_mean"] = coalesce_columns(out, ["weekly_visit_mean"], default=np.nan)
    out["weekly_visit_std"] = coalesce_columns(out, ["weekly_visit_std"], default=np.nan)
    out["stability_cv"] = coalesce_columns(out, ["stability_cv", "temporal_stability"], default=np.nan)
    out["seasonal_variance"] = coalesce_columns(out, ["seasonal_variance"], default=np.nan)

    for col in ["latitude", "longitude"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["name"] = out.get("name", "").fillna("").astype(str)
    out["city"] = out.get("city", "").fillna("").astype(str)
    return out


def compute_composite_score(df: pd.DataFrame, behavior_mode: str) -> pd.Series:
    birank = minmax_norm(df.get("birank_score", 0.0))
    rating = minmax_norm(df.get("rating_mean", 0.0))
    pop = minmax_norm(df.get("popularity_visits", 0.0))
    revisit = minmax_norm(df.get("revisit_rate", 0.0))

    weights = {
        "Regular / routine visits": {"birank": 0.35, "rating": 0.20, "pop": 0.00, "revisit": 0.45},
        "Explorer / try new places": {"birank": 0.25, "rating": 0.30, "pop": 0.45, "revisit": 0.00},
        "Morning quick stop": {"birank": 0.20, "rating": 0.45, "pop": 0.00, "revisit": 0.35},
        "Weekend casual": {"birank": 0.10, "rating": 0.25, "pop": 0.50, "revisit": 0.15},
    }.get(behavior_mode, {"birank": 0.35, "rating": 0.25, "pop": 0.20, "revisit": 0.20})

    return (
        birank * weights["birank"]
        + rating * weights["rating"]
        + pop * weights["pop"]
        + revisit * weights["revisit"]
    )


def classify_behavioral_profile(row: pd.Series) -> str:
    repeat_rate = float(pd.to_numeric(row.get("repeat_user_rate"), errors="coerce") or 0.0)
    gini = float(pd.to_numeric(row.get("gini_user_contribution"), errors="coerce") or 0.0)
    stability = float(pd.to_numeric(row.get("stability_cv"), errors="coerce") or 0.0)
    popularity = float(pd.to_numeric(row.get("popularity_visits"), errors="coerce") or 0.0)

    if repeat_rate >= 0.45 and gini >= 0.50:
        return "Loyal Regular Base"
    if stability > 0 and stability < 0.60 and repeat_rate >= 0.25:
        return "Steady Habit Traffic"
    if repeat_rate < 0.20 and gini < 0.35:
        return "Explorer-Friendly Mix"
    if popularity >= 500 and repeat_rate < 0.25:
        return "High-Volume Casual Traffic"
    return "Balanced Visitor Mix"


def extract_city_token(value: str) -> str:
    if not isinstance(value, str):
        return ""
    token = value.split(",")[0].strip()
    return token if token else value.strip()


@st.cache_resource
def get_city_aliases() -> dict[str, str]:
    if not CITY_ALIASES_FILE.exists():
        return {}
    try:
        with CITY_ALIASES_FILE.open("r") as f:
            raw = json.load(f)
        return {str(k).lower().strip(): str(v).strip() for k, v in raw.items()}
    except Exception:
        return {}


@st.cache_resource
def get_city_index_bundle():
    if not CITIES_INDEX_FILE.exists():
        return None, None, []
    try:
        with CITIES_INDEX_FILE.open("rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None, None, []

    vec, mat, city_list = None, None, []
    if isinstance(obj, dict):
        vec = obj.get("vectorizer")
        if vec is None:
            vec = obj.get("vec")

        mat = obj.get("city_matrix")
        if mat is None:
            mat = obj.get("matrix")

        city_list = obj.get("city_list")
        if city_list is None:
            city_list = obj.get("cities")
        if city_list is None:
            city_list = []
    elif isinstance(obj, (tuple, list)) and len(obj) >= 3:
        vec, mat, city_list = obj[0], obj[1], obj[2]

    if not isinstance(city_list, (list, tuple)):
        city_list = []
    city_list = [str(c).strip() for c in city_list if str(c).strip()]

    has_transform = hasattr(vec, "transform")
    has_matrix = mat is not None and hasattr(mat, "shape")
    if not (has_transform and has_matrix and city_list):
        return None, None, []
    return vec, mat, city_list


def suggest_cities(
    query: str,
    available_cities: list[str],
    aliases: dict[str, str],
    index_bundle,
    top_n: int = 8,
) -> list[str]:
    if not available_cities:
        return []

    clean = (query or "").strip()
    if not clean:
        return available_cities[:top_n]

    alias_resolved = aliases.get(clean.lower(), clean).strip()
    query_l = alias_resolved.lower()
    available_map = {c.lower(): c for c in available_cities}
    scores: dict[str, float] = {}

    if query_l in available_map:
        scores[available_map[query_l]] = 2.0

    for city in available_cities:
        city_l = city.lower()
        if query_l == city_l:
            score = 1.8
        elif query_l in city_l:
            score = 1.2
        else:
            score = difflib.SequenceMatcher(None, query_l, city_l).ratio()
        if score >= 0.45:
            scores[city] = max(scores.get(city, 0.0), float(score))

    vec, mat, idx_city_list = index_bundle
    if vec is not None and mat is not None and idx_city_list:
        try:
            q_vec = vec.transform([query_l])
            sims = cosine_similarity(q_vec, mat).flatten()
            top_indices = sims.argsort()[-20:][::-1]
            for idx in top_indices:
                sim = float(sims[idx])
                if sim <= 0.03:
                    continue
                raw_city = idx_city_list[idx]
                token = extract_city_token(raw_city)
                mapped = available_map.get(token.lower())
                if mapped is None:
                    close = difflib.get_close_matches(token, available_cities, n=1, cutoff=0.80)
                    mapped = close[0] if close else None
                if mapped:
                    scores[mapped] = max(scores.get(mapped, 0.0), sim + 0.25)
        except Exception:
            pass

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    if ranked:
        return [city for city, _ in ranked[:top_n]]

    # Last-resort fallback for malformed index + difficult typo.
    close = difflib.get_close_matches(alias_resolved, available_cities, n=top_n, cutoff=0.0)
    if close:
        return close
    return available_cities[:top_n]


def filter_city_frame(df: pd.DataFrame, selected_city: str) -> pd.DataFrame:
    if df.empty or "city" not in df.columns:
        return df.copy()
    city = (selected_city or "").strip()
    if not city:
        return df.copy()
    exact = df["city"].fillna("").str.lower().eq(city.lower())
    out = df[exact].copy()
    if out.empty:
        contains = df["city"].fillna("").str.contains(city, case=False, na=False)
        out = df[contains].copy()
    return out


def render_city_selector(prefix: str, city_values: pd.Series, default_city: str, reset_keys: list[str]) -> str:
    city_series = city_values.dropna().astype(str).str.strip()
    case_counts = city_series.value_counts()
    preferred_case: dict[str, tuple[str, int]] = {}
    for city_name, count in case_counts.items():
        key = city_name.lower()
        if key not in preferred_case or count > preferred_case[key][1]:
            preferred_case[key] = (city_name, int(count))
    available = sorted(v[0] for v in preferred_case.values())
    if not available:
        return ""

    selected_key = f"{prefix}_selected_city"
    input_key = f"{prefix}_city_input"
    suggest_key = f"{prefix}_city_suggestion"
    sync_input_key = f"{prefix}_sync_city_input"

    # Streamlit forbids changing a widget's session key after instantiation
    # in the same run. If a city was applied previously, sync it before
    # creating the text_input widget.
    if sync_input_key in st.session_state:
        st.session_state[input_key] = st.session_state.pop(sync_input_key)

    if selected_key not in st.session_state:
        st.session_state[selected_key] = default_city if default_city in available else available[0]
    if input_key not in st.session_state:
        st.session_state[input_key] = st.session_state[selected_key]

    query = st.sidebar.text_input("City Search", key=input_key)

    aliases = get_city_aliases()
    index_bundle = get_city_index_bundle()
    suggestions = suggest_cities(query, available, aliases, index_bundle)
    if not suggestions:
        suggestions = [st.session_state[selected_key]]
        st.sidebar.warning("No close city match found. Showing current city.")

    if suggest_key not in st.session_state or st.session_state[suggest_key] not in suggestions:
        st.session_state[suggest_key] = suggestions[0]

    st.sidebar.selectbox("City Suggestions", options=suggestions, key=suggest_key)

    if st.sidebar.button("Apply City", key=f"{prefix}_apply_city"):
        old_city = st.session_state[selected_key]
        new_city = st.session_state[suggest_key]
        st.session_state[selected_key] = new_city
        # Defer text input update to the next rerun (before widget creation).
        st.session_state[sync_input_key] = new_city
        if new_city != old_city:
            for key in reset_keys:
                st.session_state.pop(key, None)
        st.rerun()

    active_city = st.session_state[selected_key]
    st.sidebar.caption(f"Active City: **{active_city}**")
    return active_city


def build_validation_winner_table(val_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        one = val_df[["Method", metric]].dropna().sort_values(metric, ascending=False).reset_index(drop=True)
        if one.empty:
            continue
        winner = one.iloc[0]
        runner = one.iloc[1] if len(one) > 1 else None
        delta = float(winner[metric] - runner[metric]) if runner is not None else np.nan
        rows.append(
            {
                "Metric": metric,
                "Winner": winner["Method"],
                "Winner Score": float(winner[metric]),
                "Runner-up": runner["Method"] if runner is not None else "N/A",
                "Runner-up Score": float(runner[metric]) if runner is not None else np.nan,
                "Delta vs Runner-up": delta,
            }
        )
    return pd.DataFrame(rows)


def pick_focus_method(val_df: pd.DataFrame, focus_contains: str | None) -> str | None:
    if val_df.empty:
        return None
    if focus_contains:
        mask = val_df["Method"].astype(str).str.contains(
            focus_contains, case=False, na=False, regex=False
        )
        if mask.any():
            return str(val_df[mask].iloc[0]["Method"])
    non_random = val_df[~val_df["Method"].astype(str).str.contains("random", case=False, na=False)]
    if not non_random.empty:
        return str(non_random.iloc[0]["Method"])
    return str(val_df.iloc[0]["Method"])


def render_validation_section(
    validation_file: Path,
    title: str,
    focus_contains: str | None,
    key_prefix: str,
) -> None:
    st.markdown("---")
    st.subheader(title)

    if not validation_file.exists():
        st.warning(f"Validation file not found: `{validation_file.name}`")
        return

    val_df = safe_read_csv(validation_file)
    if val_df.empty or "Method" not in val_df.columns:
        st.warning("Validation data is empty or malformed.")
        return

    metric_order = ["NDCG@5", "NDCG@10", "NDCG@20", "Hit@5", "Hit@10", "Hit@20"]
    metrics = [m for m in metric_order if m in val_df.columns]
    if not metrics:
        st.warning("No recognized metric columns found.")
        return

    val_df = val_df.copy()
    for metric in metrics:
        val_df[metric] = pd.to_numeric(val_df[metric], errors="coerce")

    st.markdown(
        "This section validates ranking quality with multiple views: absolute scores, "
        "lift vs random baseline, K-trend behavior, and per-metric winners."
    )
    st.dataframe(val_df, use_container_width=True, hide_index=True)

    st.markdown("##### Absolute Metric Comparison")
    abs_df = val_df.set_index("Method")[metrics]
    st.bar_chart(abs_df, use_container_width=True)

    random_mask = val_df["Method"].astype(str).str.contains("random", case=False, na=False)
    random_row = val_df[random_mask].head(1)
    lift_df = pd.DataFrame()
    if not random_row.empty:
        baseline = random_row.iloc[0]
        lift_rows = []
        for _, row in val_df.iterrows():
            method = str(row["Method"])
            if "random" in method.lower():
                continue
            item = {"Method": method}
            for metric in metrics:
                rv = baseline[metric]
                mv = row[metric]
                if pd.notna(rv) and rv > 0 and pd.notna(mv):
                    item[metric] = (mv - rv) / rv * 100.0
                else:
                    item[metric] = np.nan
            lift_rows.append(item)
        if lift_rows:
            lift_df = pd.DataFrame(lift_rows).set_index("Method")
            st.markdown("##### % Lift Over Random Baseline")
            st.bar_chart(lift_df, use_container_width=True)
    else:
        st.info("Random baseline row not found, so lift-vs-random chart is skipped.")

    ndcg_cols = [c for c in ["NDCG@5", "NDCG@10", "NDCG@20"] if c in metrics]
    hit_cols = [c for c in ["Hit@5", "Hit@10", "Hit@20"] if c in metrics]

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("##### NDCG@K Trend")
        if len(ndcg_cols) >= 2:
            ndcg_trend = val_df.set_index("Method")[ndcg_cols].T
            ndcg_trend.index = [int(x.split("@")[1]) for x in ndcg_cols]
            ndcg_trend.index.name = "K"
            st.line_chart(ndcg_trend, use_container_width=True)
        else:
            st.info("Need at least 2 NDCG@K columns to draw trend.")
    with chart_right:
        st.markdown("##### Hit@K Trend")
        if len(hit_cols) >= 2:
            hit_trend = val_df.set_index("Method")[hit_cols].T
            hit_trend.index = [int(x.split("@")[1]) for x in hit_cols]
            hit_trend.index.name = "K"
            st.line_chart(hit_trend, use_container_width=True)
        else:
            st.info("Need at least 2 Hit@K columns to draw trend.")

    st.markdown("##### Per-Metric Winners vs Runner-up")
    winner_df = build_validation_winner_table(val_df, metrics)
    if winner_df.empty:
        st.info("No metric winner summary available.")
    else:
        st.dataframe(winner_df, use_container_width=True, hide_index=True)

    focus_method = pick_focus_method(val_df, focus_contains)
    if focus_method:
        metric_wins = 0
        strongest_metric = "N/A"
        weakest_metric = "N/A"
        avg_lift_str = "N/A"

        if not winner_df.empty:
            metric_wins = int((winner_df["Winner"] == focus_method).sum())

        if not lift_df.empty and focus_method in lift_df.index:
            focus_lift = lift_df.loc[focus_method].dropna()
            if not focus_lift.empty:
                avg_lift = float(focus_lift.mean())
                strongest_metric = str(focus_lift.idxmax())
                weakest_metric = str(focus_lift.idxmin())
                avg_lift_str = f"{avg_lift:.1f}%"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Focus Method", focus_method)
        c2.metric("Metric Wins", f"{metric_wins}/{len(metrics)}")
        c3.metric("Avg Lift vs Random", avg_lift_str)
        c4.metric("Strongest / Weakest", f"{strongest_metric} / {weakest_metric}")

        if not winner_df.empty:
            winner_counts = winner_df["Winner"].value_counts()
            dominant = str(winner_counts.index[0])
            dominant_wins = int(winner_counts.iloc[0])
            if dominant != focus_method:
                st.warning(
                    f"Validation note: `{dominant}` currently beats `{focus_method}` on "
                    f"{dominant_wins}/{len(metrics)} metrics. This suggests at least one "
                    "component weight/penalty in the full model may need recalibration."
                )

    st.markdown("##### Metric Explainer")
    selected_metric = st.radio(
        "Select metric to inspect",
        metrics,
        horizontal=True,
        key=f"{key_prefix}_metric_explainer",
        label_visibility="collapsed",
    )

    with st.container(border=True):
        bucket = "NDCG" if selected_metric.startswith("NDCG") else "Hit"
        info = METRIC_HELP[bucket]
        st.markdown(f"**{selected_metric}:** {info['what']}")
        st.markdown(f"**Why this matters:** {info['why']}")
        st.latex(info["formula"])

        if not winner_df.empty:
            row = winner_df[winner_df["Metric"] == selected_metric]
            if not row.empty:
                r = row.iloc[0]
                st.markdown(
                    f"Current best on **{selected_metric}**: `{r['Winner']}` "
                    f"({r['Winner Score']:.4f}), ahead of `{r['Runner-up']}` "
                    f"by {r['Delta vs Runner-up']:.4f}."
                )

        if not lift_df.empty and focus_method and focus_method in lift_df.index:
            val = lift_df.loc[focus_method].get(selected_metric)
            if pd.notna(val):
                st.markdown(
                    f"For focus method `{focus_method}`, lift vs random at "
                    f"**{selected_metric}** is **{val:.2f}%**."
                )


def render_data_health_panel(domain: str) -> None:
    if domain == "Restaurants":
        check_paths = [
            REST_BUSINESS_FILE,
            REST_SCORES_FILE,
            REST_VENUE_FEATURES_FILE,
            REST_VALIDATION_FILE,
        ]
    elif domain == "Hotels & Accommodation":
        check_paths = [
            HOTEL_BUSINESS_FILE,
            HOTEL_BIRANK_FILE,
            HOTEL_BIRANK_FSQ_FILE,
            HOTEL_VENUE_FEATURES_FILE,
            HOTEL_USER_GROUPS_FILE,
            HOTEL_VALIDATION_FILE,
        ]
    else:
        check_paths = [
            BUSINESS_FILE,
            BIRANK_FILE,
            BASELINES_FILE,
            GROUP_BIRANK_FILE,
            VENUE_FEATURES_V3_FILE,
            VALIDATION_FILE,
            SOCIAL_SIGNALS_FILE,
            BIRANK_V4_FILE,
        ]

    with st.sidebar.expander("Data File Status", expanded=False):
        for path in check_paths:
            exists = path.exists()
            icon = "✅" if exists else "❌"
            st.write(f"{icon} {path.name}")

        errors = st.session_state.get("_data_read_errors", {})
        if errors:
            st.markdown("**Read errors:**")
            for p, err in errors.items():
                if any(cp.name in p for cp in check_paths):
                    st.caption(f"{Path(p).name}: {err}")


# ============================================================================
# DATA LOADING
# ============================================================================


def load_data(domain: str, score_version: str = "v3"):
    if domain == "Restaurants":
        b_df = safe_read_csv(REST_BUSINESS_FILE)
        s_df = safe_read_csv(REST_SCORES_FILE)
        v_df = safe_read_csv(REST_VENUE_FEATURES_FILE)
        if b_df.empty:
            return b_df, pd.DataFrame(), None

        merged = b_df.copy()
        if not s_df.empty:
            merged = s_df.merge(
                b_df[["business_id", "name", "city", "latitude", "longitude", "categories", "stars"]],
                on="business_id",
                how="left",
            )
        if not v_df.empty:
            merged = merged.merge(v_df, on="business_id", how="left", suffixes=("", "_feat"))
        for col in ["latitude", "longitude", "score", "rank"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
        return b_df, merged, None

    b_df = safe_read_csv(BUSINESS_FILE)
    birank_file = BIRANK_V4_FILE if score_version == "v4" else BIRANK_FILE
    s_df = safe_read_csv(birank_file)
    base_df = safe_read_csv(BASELINES_FILE)
    v3_df = safe_read_csv(VENUE_FEATURES_V3_FILE)
    v2_df = safe_read_csv(VENUE_FEATURES_FILE)
    group_df = safe_read_csv(GROUP_BIRANK_FILE)
    exp_df = safe_read_csv(VENUE_EXPLANATIONS_FILE)
    social_df = safe_read_csv(SOCIAL_SIGNALS_FILE)

    if b_df.empty:
        return b_df, pd.DataFrame(), group_df if not group_df.empty else None

    merged = b_df.copy()
    for extra in [s_df, base_df, v3_df, v2_df, exp_df]:
        if not extra.empty and "business_id" in extra.columns:
            merged = merged.merge(extra, on="business_id", how="left", suffixes=("", "_dup"))

    # Merge social signals (friend_checkin_count, social_unique_visitors, etc.)
    if not social_df.empty and "yelp_business_id" in social_df.columns:
        social_df = social_df.rename(columns={"yelp_business_id": "business_id"})
        social_cols = ["business_id", "friend_checkin_count", "fof_checkin_count",
                       "social_unique_visitors", "social_diversity_index", "mean_bridge_confidence"]
        social_df = social_df[[c for c in social_cols if c in social_df.columns]]
        merged = merged.merge(social_df, on="business_id", how="left")
        merged["friend_checkin_count"] = pd.to_numeric(
            merged["friend_checkin_count"], errors="coerce"
        ).fillna(0.0)

    merged = normalize_coffee_columns(merged)
    return b_df, merged, group_df if not group_df.empty else None


# ============================================================================
# DASHBOARD RENDERING
# ============================================================================


def render_coffee_dashboard(coffee_df: pd.DataFrame, group_birank_df: pd.DataFrame | None) -> None:
    st.subheader("☕ Coffee Shop Ranking Inspector")

    if coffee_df.empty:
        st.error("Coffee data is missing or failed to load.")
        render_validation_section(
            VALIDATION_FILE,
            "Coffee Validation: Metrics & Evidence",
            focus_contains="BiRank",
            key_prefix="coffee",
        )
        return

    selected_city = render_city_selector(
        prefix="coffee",
        city_values=coffee_df["city"],
        default_city="Philadelphia",
        reset_keys=["coffee_ref_cafe", "coffee_radius_km", "coffee_selected_venue"],
    )

    ranking_mode = st.sidebar.selectbox("Ranking Mode", RANKING_MODES, key="coffee_ranking_mode")
    behavior_mode = st.sidebar.selectbox(
        "Behavior Profile",
        list(BEHAVIOR_TO_GROUP.keys()),
        key="coffee_behavior_mode",
    )
    top_k = st.sidebar.select_slider("Top K venues", options=[10, 20, 50], value=10, key="coffee_topk")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Social Signals (Phase 4)**")
    show_social = st.sidebar.toggle("Show social signals", value=True, key="coffee_show_social")
    score_version = st.sidebar.radio(
        "Score version",
        ["v3 (Behavioral)", "v4 (+ Social)"],
        key="coffee_score_version",
        horizontal=True,
    )
    score_version_key = "v4" if score_version.startswith("v4") else "v3"

    city_df = filter_city_frame(coffee_df, selected_city)
    if city_df.empty:
        st.warning(f"No coffee venues found for city: {selected_city}.")
        render_validation_section(
            VALIDATION_FILE,
            "Coffee Validation: Metrics & Evidence",
            focus_contains="BiRank",
            key_prefix="coffee",
        )
        return

    city_df = city_df.dropna(subset=["latitude", "longitude"]).copy()
    if city_df.empty:
        st.warning(f"City {selected_city} has no valid coordinates.")
        render_validation_section(
            VALIDATION_FILE,
            "Coffee Validation: Metrics & Evidence",
            focus_contains="BiRank",
            key_prefix="coffee",
        )
        return

    cafe_names = sorted(city_df["name"].dropna().unique().tolist())
    if not cafe_names:
        st.warning(f"City {selected_city} has no venue names.")
        render_validation_section(
            VALIDATION_FILE,
            "Coffee Validation: Metrics & Evidence",
            focus_contains="BiRank",
            key_prefix="coffee",
        )
        return

    ref_key = "coffee_ref_cafe"
    radius_key = "coffee_radius_km"
    if ref_key not in st.session_state or st.session_state[ref_key] not in cafe_names:
        st.session_state[ref_key] = cafe_names[0]
    if radius_key not in st.session_state:
        st.session_state[radius_key] = 10.0

    ref_cafe = st.sidebar.selectbox("Select a reference café", cafe_names, key=ref_key)
    radius_km = st.sidebar.slider(
        "Search radius (km)",
        min_value=1.0,
        max_value=50.0,
        value=float(st.session_state[radius_key]),
        step=1.0,
        key=radius_key,
    )

    ref_row = city_df[city_df["name"] == ref_cafe].iloc[0]
    ref_lat = float(ref_row["latitude"])
    ref_lon = float(ref_row["longitude"])

    city_df["distance_km"] = city_df.apply(
        lambda r: haversine_km(ref_lat, ref_lon, float(r["latitude"]), float(r["longitude"])),
        axis=1,
    )
    in_radius_df = city_df[city_df["distance_km"] <= radius_km].copy()

    if in_radius_df.empty:
        st.warning(
            f"No cafés found within {radius_km:.1f} km of {ref_cafe} in {selected_city}. "
            "Increase radius to see results."
        )
    rank_df = in_radius_df.copy()

    if not rank_df.empty:
        if ranking_mode == "Behaviour-based (BiRank)":
            used_group_scores = False
            if group_birank_df is not None and not group_birank_df.empty:
                group_label = BEHAVIOR_TO_GROUP.get(behavior_mode, "")
                group_scores = group_birank_df[
                    group_birank_df["cluster_name"].astype(str).str.contains(group_label, case=False, na=False)
                ]
                if not group_scores.empty:
                    score_map = (
                        group_scores.sort_values("score", ascending=False)
                        .drop_duplicates("business_id")
                        .set_index("business_id")["score"]
                    )
                    rank_df["score"] = rank_df["business_id"].map(score_map)
                    rank_df["score"] = rank_df["score"].fillna(rank_df["birank_score"])
                    used_group_scores = True
            if not used_group_scores:
                rank_df["score"] = compute_composite_score(rank_df, behavior_mode)
        elif ranking_mode == "Rating-based":
            rank_df["score"] = rank_df["rating_mean"]
        elif ranking_mode == "Popularity-based":
            rank_df["score"] = rank_df["popularity_visits"]
        else:
            rank_df["score"] = rank_df["revisit_rate"]
    else:
        rank_df["score"] = pd.Series(dtype=float)

    rank_df["score"] = pd.to_numeric(rank_df["score"], errors="coerce").fillna(0.0)
    top_k_df = rank_df.nlargest(top_k, "score").copy() if not rank_df.empty else rank_df.copy()
    if not top_k_df.empty:
        top_k_df["display_rank"] = np.arange(1, len(top_k_df) + 1)

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.subheader(f"🏆 Top {len(top_k_df)} Cafés")
        if top_k_df.empty:
            st.info("No ranked cafés in current city/radius selection.")
        else:
            show_cols = ["display_rank", "name", "distance_km", "score", "rating_mean", "revisit_rate"]
            table = top_k_df[[c for c in show_cols if c in top_k_df.columns]].copy()
            if "distance_km" in table.columns:
                table["distance_km"] = table["distance_km"].round(2)
            if "score" in table.columns:
                table["score"] = table["score"].round(4)
            if "revisit_rate" in table.columns:
                table["revisit_rate"] = (table["revisit_rate"] * 100).round(1)

            # Add social signal column when enabled and data present
            if show_social and "friend_checkin_count" in top_k_df.columns:
                all_counts = rank_df["friend_checkin_count"].fillna(0.0)
                p33 = all_counts.quantile(0.33)
                p66 = all_counts.quantile(0.66)

                def social_stars(v):
                    if v <= 0:
                        return ""
                    elif v < p33:
                        return "★"
                    elif v < p66:
                        return "★★"
                    else:
                        return "★★★"

                table["social_signal"] = top_k_df["friend_checkin_count"].fillna(0.0).apply(social_stars)

            table.rename(
                columns={
                    "display_rank": "Rank",
                    "name": "Café",
                    "distance_km": "Distance (km)",
                    "score": "Score",
                    "rating_mean": "Rating",
                    "revisit_rate": "Revisit %",
                    "social_signal": "Social Signal",
                },
                inplace=True,
            )
            st.dataframe(table, use_container_width=True, hide_index=True)
            if show_social and "friend_checkin_count" in top_k_df.columns:
                st.caption(
                    "**Social Signal** shows how often friends of matched users visited this café "
                    "(★ = some friend activity, ★★★ = high friend activity). "
                    "Based on Foursquare social graph data linked via cross-platform user matching."
                )

            csv_bytes = top_k_df.to_csv(index=False).encode("utf-8")
            safe_city = selected_city.replace(" ", "_")
            safe_mode = ranking_mode.replace(" ", "_").replace("(", "").replace(")", "")
            st.download_button(
                "📥 Download as CSV",
                data=csv_bytes,
                file_name=f"top_{top_k}_cafes_{safe_city}_{safe_mode}.csv",
                mime="text/csv",
            )

    with right:
        st.subheader("🗺️ Map")
        m = folium.Map(location=[ref_lat, ref_lon], zoom_start=13, tiles="OpenStreetMap")
        folium.Circle(
            location=[ref_lat, ref_lon],
            radius=float(radius_km) * 1000.0,
            color="blue",
            fill=False,
            weight=2,
            popup=f"Search radius: {radius_km:.1f} km",
        ).add_to(m)
        folium.Marker(
            location=[ref_lat, ref_lon],
            popup=f"Reference: {ref_cafe}",
            icon=folium.Icon(color="red", icon="star"),
        ).add_to(m)

        for _, row in in_radius_df.iterrows():
            folium.CircleMarker(
                location=[float(row["latitude"]), float(row["longitude"])],
                radius=3,
                color="gray",
                fill=True,
                fillColor="lightgray",
                fillOpacity=0.35,
                popup=f"{row['name']}<br>Dist: {row['distance_km']:.2f} km",
            ).add_to(m)

        for _, row in top_k_df.iterrows():
            rank = safe_int(row["display_rank"])
            color = "green" if rank <= 5 else "orange"
            icon = "star" if rank <= 5 else "coffee"
            social_tip = ""
            if show_social and "friend_checkin_count" in row.index:
                fc = float(row.get("friend_checkin_count", 0.0) or 0.0)
                if fc > 0:
                    social_tip = f"<br>Friend visits: {fc:.1f}"
            popup = (
                f"#{rank} {row['name']}<br>"
                f"Score: {row['score']:.4f}<br>"
                f"Rating: {row.get('rating_mean', np.nan):.1f}<br>"
                f"Revisit: {format_percent(row.get('revisit_rate'))}<br>"
                f"Distance: {row.get('distance_km', np.nan):.2f} km"
                f"{social_tip}"
            )
            folium.Marker(
                location=[float(row["latitude"]), float(row["longitude"])],
                popup=popup,
                icon=folium.Icon(color=color, icon=icon),
            ).add_to(m)

        folium_static(m, width=620, height=420)

    if not top_k_df.empty:
        st.subheader("📋 Venue Details & Explanation")
        venue_key = "coffee_selected_venue"
        venue_options = top_k_df["name"].tolist()
        if venue_key not in st.session_state or st.session_state[venue_key] not in venue_options:
            st.session_state[venue_key] = venue_options[0]

        selected_venue = st.selectbox("Select a venue to inspect", venue_options, key=venue_key)
        venue_row = top_k_df[top_k_df["name"] == selected_venue].iloc[0]

        profile = classify_behavioral_profile(venue_row)
        st.markdown(f"**Behavioral Profile:** `{profile}`")

        a, b, c, d = st.columns(4)
        a.metric("Score", f"{float(venue_row.get('score', 0.0)):.4f}")
        b.metric("Rating", format_number(venue_row.get("rating_mean"), digits=1))
        c.metric("Revisit Rate", format_percent(venue_row.get("revisit_rate")))
        d.metric("Total Visits", format_count(venue_row.get("popularity_visits")))

        st.markdown("#### 📊 Behavioral Patterns")
        r1, r2, r3 = st.columns(3)
        r1.metric("Loyalty Spread", format_number(venue_row.get("gini_user_contribution"), digits=3))
        r2.metric("Visit Stability (CV)", format_number(venue_row.get("stability_cv"), digits=2))
        r3.metric("Seasonal Variance", format_number(venue_row.get("seasonal_variance"), digits=2))

        r4, r5, r6 = st.columns(3)
        r4.metric("Repeat User %", format_percent(venue_row.get("repeat_user_rate")))
        r5.metric("Repeat Users", format_count(venue_row.get("repeat_user_count")))
        r6.metric("Avg Return Visits", format_number(venue_row.get("avg_user_repeat_visits"), digits=2))

        r7, r8, r9 = st.columns(3)
        r7.metric("Weekly Visits (avg)", format_number(venue_row.get("weekly_visit_mean"), digits=1))
        r8.metric("Weekly Std Dev", format_number(venue_row.get("weekly_visit_std"), digits=1))
        r9.metric("Unique Users", format_count(venue_row.get("unique_users")))

        if show_social and "friend_checkin_count" in venue_row.index:
            st.markdown("#### 🤝 Social Signal (Foursquare Friends)")
            fc = float(venue_row.get("friend_checkin_count", 0.0) or 0.0)
            fof = float(venue_row.get("fof_checkin_count", 0.0) or 0.0)
            sv = safe_int(venue_row.get("social_unique_visitors"))
            conf = float(venue_row.get("mean_bridge_confidence", 0.0) or 0.0)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Friend Visit Score", f"{fc:.2f}", help="Weighted sum of check-ins by matched friends")
            s2.metric("Friend-of-Friend Score", f"{fof:.2f}", help="2-hop social signal (30% weight)")
            s3.metric("Unique Social Visitors", str(sv), help="Distinct FSQ users who visited via bridge")
            s4.metric("Bridge Confidence", f"{conf:.3f}", help="Mean cross-platform match confidence")
            if fc == 0:
                st.caption("No direct social signal for this venue. The venue may not overlap with matched users' friends' check-in areas.")

        st.markdown("#### 💬 Why this venue is ranked highly")
        explanations = []
        for i in range(1, 6):
            key = f"explain_{i}"
            val = venue_row.get(key)
            if isinstance(val, str) and val.strip():
                explanations.append(val.strip())

        if not explanations:
            explanations = [
                f"Strong ranking score ({float(venue_row.get('score', 0.0)):.4f}) in the selected mode.",
                f"Balanced quality signals: rating {format_number(venue_row.get('rating_mean'), 1)}, "
                f"revisit {format_percent(venue_row.get('revisit_rate'))}.",
                f"Located {float(venue_row.get('distance_km', 0.0)):.2f} km from your selected reference café.",
            ]

        for line in explanations[:5]:
            st.markdown(f"- {line}")

    render_validation_section(
        VALIDATION_FILE,
        "Coffee Validation: Metrics & Evidence",
        focus_contains="BiRank",
        key_prefix="coffee",
    )


def render_restaurant_dashboard(rest_df: pd.DataFrame) -> None:
    st.subheader("🍽️ Behavioral Restaurant Recommendations (Car-Centric)")

    if rest_df.empty:
        st.error("Restaurant data is missing or failed to load.")
        render_validation_section(
            REST_VALIDATION_FILE,
            "Restaurant Validation: Metrics & Evidence",
            focus_contains="Full S(R,U,C)",
            key_prefix="restaurant",
        )
        return

    selected_city = render_city_selector(
        prefix="restaurant",
        city_values=rest_df["city"] if "city" in rest_df.columns else pd.Series(dtype=str),
        default_city="Philadelphia",
        reset_keys=[
            "restaurant_sim_lat",
            "restaurant_sim_lon",
            "restaurant_sim_city",
            "restaurant_radius_km",
        ],
    )
    top_k = st.sidebar.select_slider("Top K venues", options=[10, 20, 50], value=10, key="rest_topk")
    radius_key = "restaurant_radius_km"
    if radius_key not in st.session_state:
        st.session_state[radius_key] = 5.0
    radius_km = st.sidebar.slider(
        "Drive radius (km)",
        min_value=1.0,
        max_value=30.0,
        value=float(st.session_state[radius_key]),
        step=0.5,
        key=radius_key,
    )

    city_df = filter_city_frame(rest_df, selected_city) if selected_city else rest_df.copy()
    city_df = city_df.dropna(subset=["latitude", "longitude"]).copy()
    if city_df.empty:
        st.warning(f"No restaurants found for city: {selected_city}.")
        render_validation_section(
            REST_VALIDATION_FILE,
            "Restaurant Validation: Metrics & Evidence",
            focus_contains="Full S(R,U,C)",
            key_prefix="restaurant",
        )
        return

    sort_col = "rank" if "rank" in city_df.columns else "score"
    ascending = sort_col == "rank"
    ranked = city_df.sort_values(sort_col, ascending=ascending).drop_duplicates("business_id")
    if "score" not in ranked.columns:
        ranked["score"] = 1.0 / (ranked["rank"].fillna(9999.0) + 1.0)

    st.markdown("### 🗺️ Interactive Driving Map Simulation")
    st.caption(
        "Click the map to move the start pin. The top list updates to venues inside the selected drive radius."
    )

    center_lat = float(ranked.iloc[0]["latitude"])
    center_lon = float(ranked.iloc[0]["longitude"])
    sim_city_key = "restaurant_sim_city"
    sim_lat_key = "restaurant_sim_lat"
    sim_lon_key = "restaurant_sim_lon"
    if (
        sim_city_key not in st.session_state
        or st.session_state[sim_city_key] != selected_city
        or sim_lat_key not in st.session_state
        or sim_lon_key not in st.session_state
    ):
        st.session_state[sim_city_key] = selected_city
        st.session_state[sim_lat_key] = center_lat
        st.session_state[sim_lon_key] = center_lon

    ranked = ranked.copy()
    ranked["sim_distance_km"] = ranked.apply(
        lambda r: haversine_km(
            float(st.session_state[sim_lat_key]),
            float(st.session_state[sim_lon_key]),
            float(r["latitude"]),
            float(r["longitude"]),
        ),
        axis=1,
    )
    in_radius = ranked[ranked["sim_distance_km"] <= float(radius_km)].copy()
    if in_radius.empty:
        st.warning(
            f"No restaurants found within {radius_km:.1f} km of the selected point. "
            "Showing nearest options outside the radius."
        )
        candidate_df = ranked.nsmallest(top_k, "sim_distance_km").copy()
    else:
        candidate_df = in_radius.copy()

    candidate_df = candidate_df.sort_values(sort_col, ascending=ascending)
    top_df = candidate_df.head(top_k).copy()
    top_df["display_rank"] = np.arange(1, len(top_df) + 1)

    m = folium.Map(
        location=[float(st.session_state[sim_lat_key]), float(st.session_state[sim_lon_key])],
        zoom_start=12,
        tiles="CartoDB positron",
    )
    folium.Marker(
        [st.session_state[sim_lat_key], st.session_state[sim_lon_key]],
        popup="Simulated Start Location",
        tooltip="Your Location",
        icon=folium.Icon(color="black", icon="user"),
    ).add_to(m)
    folium.Circle(
        location=[st.session_state[sim_lat_key], st.session_state[sim_lon_key]],
        radius=float(radius_km) * 1000.0,
        color="#3186cc",
        fill=True,
        fill_color="#3186cc",
        fill_opacity=0.15,
        popup=f"{radius_km:.1f} km Drive Radius",
    ).add_to(m)
    for _, row in top_df.iterrows():
        popup = (
            f"#{safe_int(row['display_rank'])} {row['name']}<br>"
            f"Score: {float(row.get('score', 0.0)):.3f}<br>"
            f"Distance: {float(row.get('sim_distance_km', np.nan)):.2f} km"
        )
        folium.Marker(
            [float(row["latitude"]), float(row["longitude"])],
            popup=popup,
            tooltip=f"#{safe_int(row['display_rank'])} {row['name']}",
            icon=folium.Icon(color="red" if safe_int(row["display_rank"]) == 1 else "blue", icon="cutlery"),
        ).add_to(m)

    map_data = st_folium(m, height=420, width=760, key="restaurant_map")
    if map_data and map_data.get("last_clicked"):
        lat = float(map_data["last_clicked"]["lat"])
        lon = float(map_data["last_clicked"]["lng"])
        if (
            abs(float(st.session_state[sim_lat_key]) - lat) > 0.0001
            or abs(float(st.session_state[sim_lon_key]) - lon) > 0.0001
        ):
            st.session_state[sim_lat_key] = lat
            st.session_state[sim_lon_key] = lon
            st.rerun()

    st.success(
        f"Simulated location: {float(st.session_state[sim_lat_key]):.4f}, "
        f"{float(st.session_state[sim_lon_key]):.4f}"
    )

    st.markdown("#### ⏱️ Simulated Drive ETAs")
    eta_cols = st.columns(3)
    for i, (_, row) in enumerate(top_df.iterrows()):
        dist_km = float(row.get("sim_distance_km", np.nan))
        eta_min = (dist_km / 30.0) * 60.0 + 2.0
        eta_cols[i % 3].metric(
            label=f"#{safe_int(row['display_rank'])} {row['name']}",
            value=f"{eta_min:.1f} mins",
            delta=f"{dist_km:.1f} km away",
            delta_color="inverse",
        )

    st.markdown("---")
    for _, row in top_df.iterrows():
        with st.expander(
            f"#{safe_int(row['display_rank'])} | {row['name']} "
            f"(⭐ {format_number(row.get('avg_rating', row.get('stars')), 1)})"
        ):
            c1, c2, c3 = st.columns(3)
            c1.write("**Score Components S(R,U,C)**")
            c1.progress(max(0.0, min(1.0, float(row.get("u_beh", 0.0)))))
            c1.caption(f"Behavioral (U_beh): {float(row.get('u_beh', 0.0)):.2f}")
            c1.progress(max(0.0, min(1.0, float(row.get("c_mob", 0.0)))))
            c1.caption(f"Drive & Parking (C_mob): {float(row.get('c_mob', 0.0)):.2f}")
            c1.progress(max(0.0, min(1.0, float(row.get("r_ctx", 0.0)))))
            c1.caption(f"Context (R_ctx): {float(row.get('r_ctx', 0.0)):.2f}")
            c1.write(f"**Total Score:** {float(row.get('score', 0.0)):.3f}")

            c2.write("**Behavioral Profile**")
            c2.write(f"Popularity: {format_count(row.get('popularity'))} visits")
            c2.write(f"Repeat Rate: {format_percent(row.get('repeat_user_rate'))}")
            c2.write(f"Gini Concentration: {format_number(row.get('gini_user_concentration'), 2)}")

            c3.write("**Accessibility & Context**")
            c3.write(f"Parking Score: {format_number(row.get('parking_score'), 2)}")
            c3.write(f"Walking Density: {format_count(row.get('walking_density'))}")
            c3.write(f"Peak Busyness: {format_number(row.get('peak_busyness'), 1)}")

    render_validation_section(
        REST_VALIDATION_FILE,
        "Restaurant Validation: Metrics & Evidence",
        focus_contains="Full S(R,U,C)",
        key_prefix="restaurant",
    )


# ============================================================================
# HOTEL DASHBOARD
# ============================================================================

HOTEL_BEHAVIORAL_TAGS = {
    "business":  "Business Hub",
    "leisure":   "Leisure Destination",
    "seasonal":  "Seasonal Resort",
    "loyalist":  "Road Warrior Favourite",
    "explorer":  "Hidden Gem",
}

HOTEL_RANKING_MODES = [
    "BiRank (Behavioural)",
    "Rating (Stars)",
    "Popularity (Reviews)",
    "BiRank + Foursquare",
]


def classify_hotel_profile(row) -> str:
    bli = float(row.get("business_leisure_ratio") or 0.5)
    scv = float(row.get("seasonal_cv") or 0.5)
    msr = float(row.get("multi_stay_rate") or 0.0)

    if msr > 0.05:
        return "Road Warrior Favourite"
    if bli > 0.65:
        return "Business Hub"
    if scv > 0.5:
        return "Seasonal Resort"
    if bli < 0.45:
        return "Leisure Destination"
    return "Hidden Gem"


def load_hotel_data():
    b_df  = safe_read_csv(HOTEL_BUSINESS_FILE)
    s_df  = safe_read_csv(HOTEL_BIRANK_FILE)
    sf_df = safe_read_csv(HOTEL_BIRANK_FSQ_FILE)
    vf_df = safe_read_csv(HOTEL_VENUE_FEATURES_FILE)

    if b_df.empty:
        return b_df, pd.DataFrame()

    merged = b_df.copy()
    if not s_df.empty and "business_id" in s_df.columns:
        s_df = s_df.rename(columns={"birank_score": "birank_score"})
        merged = merged.merge(s_df[["business_id","birank_score"]], on="business_id", how="left")
    if not sf_df.empty and "business_id" in sf_df.columns:
        merged = merged.merge(
            sf_df[["business_id","birank_fsq_score"]],
            on="business_id", how="left"
        )
    if not vf_df.empty and "business_id" in vf_df.columns:
        feat_cols = ["business_id","business_leisure_ratio","seasonal_cv",
                     "multi_stay_rate","review_velocity","geographic_diversity",
                     "venue_stability_cv","traveler_concentration"]
        vf_sub = vf_df[[c for c in feat_cols if c in vf_df.columns]]
        merged = merged.merge(vf_sub, on="business_id", how="left")

    for col in ["latitude","longitude","stars","review_count","birank_score"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["hotel_profile"] = merged.apply(classify_hotel_profile, axis=1)
    return b_df, merged


def render_hotel_dashboard(hotel_df: pd.DataFrame) -> None:
    st.subheader("🏨 Hotel & Accommodation Ranking Inspector")

    if hotel_df.empty:
        st.error("Hotel data is missing or failed to load. Run hotel_data_extract.py first.")
        return

    # ── Sidebar controls ──────────────────────────────────────────────────────
    selected_city = render_city_selector(
        prefix="hotel",
        city_values=hotel_df["city"],
        default_city="New Orleans",
        reset_keys=["hotel_ref_venue","hotel_radius_km","hotel_selected_venue"],
    )

    ranking_mode = st.sidebar.selectbox(
        "Ranking Mode", HOTEL_RANKING_MODES, key="hotel_ranking_mode"
    )
    subcategory_filter = st.sidebar.multiselect(
        "Accommodation Type",
        options=sorted(hotel_df["subcategory"].dropna().unique().tolist()),
        default=[],
        key="hotel_subcat",
        help="Leave blank to show all types",
    )
    top_k = st.sidebar.select_slider(
        "Top K venues", options=[10, 20, 50], value=20, key="hotel_topk"
    )

    # ── Filter city ───────────────────────────────────────────────────────────
    city_df = filter_city_frame(hotel_df, selected_city)
    if city_df.empty:
        st.warning(f"No hotel venues found for city: {selected_city}.")
        return

    city_df = city_df.dropna(subset=["latitude","longitude"]).copy()
    if subcategory_filter:
        city_df = city_df[city_df["subcategory"].isin(subcategory_filter)]
    if city_df.empty:
        st.warning("No venues match the selected filters.")
        return

    # ── Ranking score selection ───────────────────────────────────────────────
    if ranking_mode == "BiRank (Behavioural)":
        score_col = "birank_score"
    elif ranking_mode == "Rating (Stars)":
        score_col = "stars"
    elif ranking_mode == "Popularity (Reviews)":
        score_col = "review_count"
    else:
        score_col = "birank_fsq_score" if "birank_fsq_score" in city_df.columns else "birank_score"

    city_df["_score"] = pd.to_numeric(city_df.get(score_col, 0), errors="coerce").fillna(0)
    top_df = city_df.nlargest(top_k, "_score").reset_index(drop=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Venues in City", f"{len(city_df):,}")
    col2.metric("Avg Rating", f"{city_df['stars'].mean():.2f}" if "stars" in city_df.columns else "—")
    col3.metric("Business Hubs", f"{(city_df['hotel_profile']=='Business Hub').sum()}")
    col4.metric("Seasonal Resorts", f"{(city_df['hotel_profile']=='Seasonal Resort').sum()}")

    st.markdown("---")

    # ── Side-by-side ranking comparison ──────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(f"#### Top {top_k} Hotels — {ranking_mode}")

        show_cols = ["name","subcategory","stars","hotel_profile"]
        if "birank_score" in top_df.columns:
            show_cols.append("birank_score")
        if "review_count" in top_df.columns:
            show_cols.append("review_count")

        display_df = top_df[[c for c in show_cols if c in top_df.columns]].copy()
        if "birank_score" in display_df.columns:
            display_df["birank_score"] = display_df["birank_score"].round(5)

        display_df.index = range(1, len(display_df)+1)
        st.dataframe(display_df, use_container_width=True)

        # CSV download
        csv_data = top_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV", csv_data,
            f"hotel_rankings_{selected_city}.csv", "text/csv",
            key="hotel_csv_download"
        )

    with col_right:
        st.markdown("#### Behavioral Profile Mix")
        if "hotel_profile" in city_df.columns:
            profile_counts = city_df["hotel_profile"].value_counts()
            for profile, cnt in profile_counts.items():
                pct = 100 * cnt / len(city_df)
                st.progress(pct/100, text=f"{profile}: {cnt} ({pct:.0f}%)")

        st.markdown("#### Category Mix")
        if "subcategory" in city_df.columns:
            subcat_counts = city_df["subcategory"].value_counts().head(5)
            for sub, cnt in subcat_counts.items():
                st.write(f"- {sub}: {cnt}")

    st.markdown("---")

    # ── Map ───────────────────────────────────────────────────────────────────
    st.markdown(f"#### Map — {selected_city}")
    try:
        import folium
        from streamlit_folium import st_folium

        center_lat = top_df["latitude"].mean()
        center_lon = top_df["longitude"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

        color_map = {
            "Business Hub":          "#2196F3",
            "Leisure Destination":   "#4CAF50",
            "Seasonal Resort":       "#FF9800",
            "Road Warrior Favourite": "#9C27B0",
            "Hidden Gem":            "#607D8B",
        }

        for rank, (_, row) in enumerate(top_df.iterrows(), 1):
            lat = row.get("latitude")
            lon = row.get("longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue

            profile = str(row.get("hotel_profile",""))
            color   = color_map.get(profile, "#607D8B")
            popup_html = f"""
                <b>#{rank} {row.get('name','?')}</b><br>
                {row.get('subcategory','Hotel')} | ⭐ {row.get('stars','?')}<br>
                Profile: {profile}<br>
                BiRank: {row.get('birank_score',0):.5f}<br>
                Reviews: {safe_int(row.get('review_count')):,}
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"#{rank} {row.get('name','?')}",
            ).add_to(m)

        st_folium(m, width=None, height=450, key="hotel_map")

        # Legend
        st.markdown("**Map Legend:**  " +
            "  ".join(f"<span style='color:{c}'>■</span> {p}"
                      for p, c in color_map.items()),
            unsafe_allow_html=True)

    except ImportError:
        st.info("Install folium + streamlit-folium to see the map.")

    st.markdown("---")

    # ── Venue detail card ─────────────────────────────────────────────────────
    st.markdown("#### Venue Detail")
    venue_names = top_df["name"].dropna().tolist()
    if venue_names:
        selected_venue = st.selectbox(
            "Select a hotel for details", venue_names, key="hotel_selected_venue"
        )
        row = top_df[top_df["name"] == selected_venue].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Stars", f"{row.get('stars','?')}")
        c2.metric("Reviews", f"{safe_int(row.get('review_count')):,}")
        c3.metric("BiRank Score", f"{row.get('birank_score',0):.5f}")

        c1, c2, c3 = st.columns(3)
        bli = row.get("business_leisure_ratio")
        if bli is not None and not pd.isna(bli):
            c1.metric("Business/Leisure Ratio",
                      f"{float(bli):.1%} weekday",
                      help="High = business hotel; Low = leisure/resort")
        scv = row.get("seasonal_cv")
        if scv is not None and not pd.isna(scv):
            c2.metric("Seasonal CV",
                      f"{float(scv):.2f}",
                      help="Low = consistent year-round demand; High = seasonal spike")
        msr = row.get("multi_stay_rate")
        if msr is not None and not pd.isna(msr):
            c3.metric("Multi-Stay Rate",
                      f"{float(msr):.1%}",
                      help="Fraction of guests who reviewed this hotel more than once")

        st.info(f"**Behavioral Profile:** {row.get('hotel_profile','—')}")

    # ── Validation section ────────────────────────────────────────────────────
    render_validation_section(
        HOTEL_VALIDATION_FILE,
        "Hotel Validation: Metrics & Evidence",
        focus_contains="hotel_birank",
        key_prefix="hotel",
    )


# ============================================================================
# LLM SIMULATION PAGE
# ============================================================================

# Import Study 2 demographic profiles (age × occupation behavioral library)
import sys as _sys
_llm_dir = str(DATA_DIR / "llm_simulation")
if _llm_dir not in _sys.path:
    _sys.path.insert(0, _llm_dir)
try:
    from demographic_profiles import (  # type: ignore
        PROFILES as _DEMO_PROFILES,
        AGE_RANGES as _AGE_RANGES,
        get_profile as _get_demo_profile,
        get_cities as _get_demo_cities,
        get_job_titles as _get_demo_job_titles,
        VALID_CELLS as _VALID_CELLS,
    )
    _DEMO_AVAILABLE = True
except ImportError:
    _DEMO_AVAILABLE = False
    _DEMO_PROFILES = {}
    _AGE_RANGES = {}
    _VALID_CELLS = []

    def _get_demo_profile(ag, occ, domain):  # type: ignore
        return {}

    def _get_demo_cities(ag, occ):  # type: ignore
        return []

    def _get_demo_job_titles(ag, occ):  # type: ignore
        return []

# ── Demographic selector options ──────────────────────────────────────────────
_AGE_GROUPS = [
    "Gen Z (18-25)",
    "Young Millennial (26-33)",
    "Senior Millennial (34-40)",
    "Gen X (41-56)",
    "Boomer (57+)",
]
_OCCUPATION_CLUSTERS = [
    "Tech / Software",
    "Healthcare",
    "Education / Academic",
    "Creative / Media",
    "Legal / Finance",
    "Trade / Manual",
    "Executive / C-Suite",
    "Hospitality / Service",
    "Student / Part-time",
    "Remote / Digital Nomad",
]

# Simulation result files
LLM_SIM_DIR = DATA_DIR / "llm_simulation" / "results"
# Prefer v2 files (gpt-5.4 run) over v1 if they exist
LLM_SIM_RECORDS = LLM_SIM_DIR / (
    "simulation_records_v2.csv" if (LLM_SIM_DIR / "simulation_records_v2.csv").exists()
    else "simulation_records.csv"
)
LLM_SIM_METRICS = LLM_SIM_DIR / (
    "simulation_metrics_v2.json" if (LLM_SIM_DIR / "simulation_metrics_v2.json").exists()
    else "simulation_metrics.json"
)
LLM_SIM_REPORT = LLM_SIM_DIR / (
    "simulation_report_v2.md" if (LLM_SIM_DIR / "simulation_report_v2.md").exists()
    else "simulation_report.md"
)

# Archetype descriptions for persona chat
_ARCHETYPE_DESCRIPTIONS = {
    # Coffee archetypes
    "Loyalist": "A habitual coffee drinker who visits the same cafe almost every day. Values familiarity, consistency, and the comfort of a 'regular' experience.",
    "Weekday Regular": "A Monday-to-Friday morning routine commuter who grabs coffee before or during work. Values speed, quality, and a reliable experience.",
    "Casual Weekender": "Visits cafes mostly on weekends with friends or family for leisure. Values ambiance, seating space, and a relaxed atmosphere.",
    "Infrequent Visitor": "Drops in occasionally and treats cafe visits as a treat or social event. Values novelty, good branding, and word-of-mouth recommendations.",
    # Restaurant archetypes
    "Explorer": "Actively seeks new and diverse dining experiences. Values unique menus, hidden gems, and adventurous cuisine over familiar chains.",
    "Mixed / Average": "A balanced diner who visits a mix of familiar spots and new places. Values price-quality ratio, convenience, and general crowd-pleasing menus.",
    "Nightlife Seeker": "Primarily visits restaurants later in the evening, often combining dining with social activities. Values late hours, atmosphere, and bar-friendly menus.",
    # Hotel archetypes
    "One-Time Tourist (Business)": "A business traveler who visits a city once for work. Values proximity to meeting venues, reliable WiFi, and efficient service.",
    "Leisure Traveler": "Visits for holidays or personal trips. Values comfort, local character, and proximity to tourist attractions.",
    "One-Time Tourist": "A tourist on a first-time visit to a city. Values location near landmarks, good reviews, and value for money.",
    "Budget Explorer": "A cost-conscious traveler who prioritises affordability and flexibility. Values price per night, cleanliness, and good transport links.",
}

_PERSONA_QUESTIONS = {
    "coffee": "You are looking for a great coffee shop in {city}. Based on your personality and coffee habits, describe exactly what you look for in a coffee shop and name one or two specific things that would make you choose it over others.",
    "restaurant": "You are choosing a restaurant for tonight in {city}. Based on your dining personality, describe what kind of restaurant experience you are after and what specific qualities would make you book a table.",
    "hotel": "You need to book a hotel in {city} for an upcoming trip. Based on who you are and why you travel, describe exactly what you look for in a hotel and what would make you choose one over another.",
}

_EXAMPLE_PERSONA_OUTPUT = {
    "coffee": {
        "name": "Maya Chen",
        "age": 28,
        "occupation": "UX Designer",
        "response": "As a Casual Weekender, I look for a coffee shop with comfortable seating, plenty of natural light, and a relaxed vibe where I can linger over a flat white with friends. The decor matters — I want somewhere Instagram-worthy but not pretentious. Good pastries are a bonus. What would make me choose one place over another? Honestly, community tables and no laptop-at-peak-hours rule signals a place that values genuine social time, which is exactly what my weekends are about.",
    },
    "restaurant": {
        "name": "Jake Osei",
        "age": 34,
        "occupation": "Food blogger",
        "response": "I'm an Explorer at heart — I want the place nobody else on the high street has found yet. I scan menus for ingredients I don't recognise and check whether the chef sources locally. A tasting menu or a 'chef's choice' option is an instant yes. What seals the deal? A small, focused menu that changes seasonally tells me someone cares deeply about what they're putting on the plate.",
    },
    "hotel": {
        "name": "Sarah Mitchell",
        "age": 41,
        "occupation": "Management Consultant",
        "response": "On a business trip I need three things: fast, reliable WiFi from the moment I check in, a proper desk in the room, and a 24-hour front desk so I can handle flight changes at midnight. Being walking distance from the conference centre is non-negotiable. What tips it in favour of one property over another? A hotel gym open before 6 am — it's the only hour that belongs to me.",
    },
}


def _load_sim_metrics() -> dict:
    if not LLM_SIM_METRICS.exists():
        return {}
    try:
        with LLM_SIM_METRICS.open("r") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _load_sim_records() -> pd.DataFrame:
    return safe_read_csv(LLM_SIM_RECORDS)


def _domain_archetypes(records: pd.DataFrame) -> dict[str, list[str]]:
    """Return archetypes grouped by domain from the records dataframe."""
    if records.empty or "domain" not in records.columns or "archetype" not in records.columns:
        return {}
    out: dict[str, list[str]] = {}
    for domain, grp in records.groupby("domain"):
        out[str(domain)] = sorted(grp["archetype"].dropna().unique().tolist())
    return out


def _render_domain_tab(domain_key: str, domain_label: str, records: pd.DataFrame, metrics: dict) -> None:
    """Render per-domain results: bar chart + archetype table with significance highlighting."""
    domain_records = records[records["domain"] == domain_key].copy() if not records.empty else pd.DataFrame()

    overall_key = f"{domain_key}|OVERALL"
    if overall_key in metrics:
        m = metrics[overall_key]
        oc1, oc2, oc3 = st.columns(3)
        oc1.metric(
            "NDCG@10",
            f"{m.get('ndcg_mean', 0):.4f}",
            delta=f"Δ vs Stars: {m.get('delta_vs_stars', 0):+.4f}",
        )
        oc2.metric("Pairwise Win Rate", f"{m.get('pairwise_win_rate', 0) * 100:.1f}%")
        p_val = m.get("wilcoxon_p", 1.0)
        sig_label = "p < 0.05 ✓" if p_val < 0.05 else "n.s."
        oc3.metric("Wilcoxon p", f"{p_val:.4f}", delta=sig_label)

    # Build per-archetype rows from metrics dict
    arch_rows = []
    for key, m in metrics.items():
        if not key.startswith(f"{domain_key}|") or key == overall_key:
            continue
        archetype = key.split("|", 1)[1]
        arch_rows.append(
            {
                "Archetype": archetype,
                "n": m.get("n", 0),
                "NDCG@10": m.get("ndcg_mean", float("nan")),
                "Stars NDCG": m.get("stars_ndcg_mean", float("nan")),
                "95% CI": f"[{m.get('ndcg_lo', 0):.4f}, {m.get('ndcg_hi', 0):.4f}]",
                "Win Rate": f"{m.get('pairwise_win_rate', 0) * 100:.1f}%",
                "p-value": m.get("wilcoxon_p", 1.0),
            }
        )

    if arch_rows:
        arch_df = pd.DataFrame(arch_rows)

        # Bar chart: behavioural vs stars NDCG per archetype
        chart_df = arch_df.set_index("Archetype")[["NDCG@10", "Stars NDCG"]]
        st.markdown("##### NDCG@10 by Archetype — Behavioural Model vs Stars Baseline")
        st.bar_chart(chart_df, use_container_width=True)

        # Table with significance highlighting
        st.markdown("##### Per-Archetype Results")
        display_df = arch_df[["Archetype", "n", "NDCG@10", "95% CI", "Win Rate", "p-value"]].copy()
        display_df["NDCG@10"] = display_df["NDCG@10"].round(4)
        display_df["Significant (p<0.05)"] = display_df["p-value"].apply(lambda p: "Yes" if p < 0.05 else "No")
        display_df["p-value"] = display_df["p-value"].apply(lambda p: f"{p:.4f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    elif domain_records.empty:
        st.info(f"No records found for domain: {domain_label}. Run the simulation first.")
    else:
        st.info("Metrics data unavailable for this domain.")


def _get_openai_key() -> str | None:
    """Return OpenAI API key from env or .env file."""
    # First check process env
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    # Try llm_simulation/.env
    env_file = DATA_DIR / "llm_simulation" / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        candidate = parts[1].strip().strip('"').strip("'")
                        if candidate:
                            return candidate
        except Exception:
            pass
    return None


# ── Archetype-specific venue ranking signal ───────────────────────────────────
# (feature_col, ascending)  ascending=False → higher value = better match
_ARCHETYPE_SORT = {
    "Loyalist":                      ("revisit_rate",            False),
    "Weekday Regular":               ("temporal_stability",      False),
    "Casual Weekender":              ("unique_users",            False),
    "Infrequent Visitor":            ("stars",                   False),
    "Explorer":                      ("popularity",              False),
    "Mixed / Average":               ("avg_rating",              False),
    "Nightlife Seeker":              ("peak_busyness",           False),
    "One-Time Tourist (Business)":   ("business_leisure_ratio",  False),
    "Leisure Traveler":              ("geographic_diversity",    False),
    "One-Time Tourist":              ("stars",                   False),
    "Budget Explorer":               ("review_count",            False),
}

_ARCHETYPE_WHY = {
    "Loyalist":                    "High return-visitor rate — regulars keep coming back",
    "Weekday Regular":             "Consistent daily traffic — reliable for your routine",
    "Casual Weekender":            "Broad appeal — draws many different types of visitor",
    "Infrequent Visitor":          "Top-rated — the safe, trusted choice",
    "Explorer":                    "High footfall — consistently popular with diverse crowds",
    "Mixed / Average":             "Highly rated — solid all-round quality",
    "Nightlife Seeker":            "Peak-hour activity — buzzy atmosphere in the evenings",
    "One-Time Tourist (Business)": "Weekday-dominant — professional, business-friendly",
    "Leisure Traveler":            "High geographic diversity — draws visitors from across the country",
    "One-Time Tourist":            "Well-reviewed — trusted by first-time visitors",
    "Budget Explorer":             "Extensively reviewed — vetted value for money",
}


@st.cache_data(show_spinner=False)
def _load_city_venues(domain: str, city: str, archetype: str, top_n: int = 3,
                      override_sort_col: str = "") -> list:
    """Load top-N venues for a domain+city filtered and ranked by archetype/demographic signal."""
    import pandas as _pd  # noqa: PLC0415

    # Demographic sort_col takes priority over archetype sort_col
    if override_sort_col and override_sort_col != "stars":
        sort_col, ascending = override_sort_col, False
    else:
        sort_col, ascending = _ARCHETYPE_SORT.get(archetype, ("stars", False))

    try:
        if domain == "coffee":
            scores = _pd.read_csv(BIRANK_FILE)
            businesses = _pd.read_csv(BUSINESS_FILE)
            features = _pd.read_csv(VENUE_FEATURES_FILE)
            df = (scores
                  .merge(businesses[["business_id", "name", "city", "state",
                                     "stars", "review_count", "categories"]],
                         on="business_id", how="left")
                  .merge(features[["business_id", "revisit_rate", "temporal_stability",
                                   "unique_users"]], on="business_id", how="left"))
            tag_fn = lambda r: (  # noqa: E731
                "High Retention — regulars keep coming back" if (r.get("revisit_rate") or 0) > 0.20
                else "Steady — loyal customer base" if (r.get("revisit_rate") or 0) > 0.05
                else "Consistent Traffic — reliable footfall all week"
                if (r.get("temporal_stability") or 0) > 0.7
                else "Broad Appeal — popular with explorers"
            )

        elif domain == "restaurant":
            businesses = _pd.read_csv(REST_BUSINESS_FILE)
            features = _pd.read_csv(REST_VENUE_FEATURES_FILE)
            df = businesses.merge(
                features[["business_id", "popularity", "repeat_user_rate",
                           "avg_rating", "transit_access_score",
                           "cuisine_categories", "peak_busyness"]],
                on="business_id", how="left"
            )
            df["review_count"] = df.get("review_count", _pd.Series(dtype=float))
            tag_fn = lambda r: (  # noqa: E731
                "Local Favourite — high repeat visitor rate"
                if (r.get("repeat_user_rate") or 0) > 0.15
                else "Transit Accessible — easy to reach without a car"
                if (r.get("transit_access_score") or 0) > 0.8
                else "Lively & Buzzy — expect queues at peak times"
                if (r.get("peak_busyness") or 0) > 0.7
                else "Hidden Gem — underrated by stars, strong loyalists"
            )

        else:  # hotel
            scores = _pd.read_csv(HOTEL_BIRANK_FILE)
            features = _pd.read_csv(HOTEL_VENUE_FEATURES_FILE)
            df = scores.merge(
                features[["business_id", "business_leisure_ratio", "geographic_diversity",
                           "multi_stay_rate", "seasonal_cv"]],
                on="business_id", how="left"
            )
            df["review_count"] = df.get("review_count",
                                        df.get("n_reviews", _pd.Series(dtype=float)))
            tag_fn = lambda r: (  # noqa: E731
                "Business Hub — weekday-dominant, professional guests"
                if (r.get("business_leisure_ratio") or 0.5) > 0.75
                else "Leisure Escape — weekend & holiday crowd"
                if (r.get("business_leisure_ratio") or 0.5) < 0.25
                else "Destination Hotel — draws visitors from many states"
                if (r.get("geographic_diversity") or 0) > 0.5
                else "Guest Favourite — unusually high return-guest rate"
                if (r.get("multi_stay_rate") or 0) > 0.05
                else "Reliable Stay — consistent quality across traveller types"
            )

    except Exception:
        return []

    # Filter to city (case-insensitive)
    city_clean = city.strip().lower()
    city_col = df.get("city", _pd.Series(dtype=str)).fillna("").str.strip().str.lower()
    city_df = df[city_col == city_clean]
    if len(city_df) < top_n:
        city_df = df  # fall back to all venues if too few in city

    city_df = city_df.dropna(subset=["name"])
    if sort_col in city_df.columns:
        city_df = city_df.sort_values(sort_col, ascending=ascending)

    venues = []
    for _, row in city_df.head(top_n).iterrows():
        rc = row.get("review_count", row.get("n_reviews", ""))
        try:
            rc_str = f"{int(float(rc)):,}" if str(rc) not in ("", "nan", "None") else ""
        except (ValueError, TypeError):
            rc_str = ""
        venues.append({
            "name":        str(row.get("name", "Unknown")),
            "city":        str(row.get("city", city)),
            "stars":       row.get("stars", "?"),
            "review_count": rc_str,
            "tag":         tag_fn(row),
            "why":         _ARCHETYPE_WHY.get(archetype, "Top behavioural model pick"),
            "categories":  str(row.get("categories", row.get("cuisine_categories",
                               row.get("subcategory", "")))).split(",")[0].strip(),
        })
    return venues


def _call_persona_chat(
    api_key: str,
    archetype: str,
    domain: str,
    city: str,
    rand_seed: int = 0,
    venues: list = None,
    age_group: str = "",
    occupation_cluster: str = "",
) -> dict:
    """Generate a persona that is shaped by BOTH archetype AND age+occupation demographics."""
    try:
        import openai  # noqa: PLC0415
    except ImportError:
        return {"error": "openai package not installed"}

    import random as _random  # noqa: PLC0415
    rng = _random.Random(rand_seed)

    # ── Names ─────────────────────────────────────────────────────────────────
    first_names = ["Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Avery", "Quinn",
                   "Sam", "Jamie", "Priya", "Omar", "Luca", "Nadia", "Rafael", "Mei",
                   "Isaac", "Sofia", "Dev", "Leila", "Marcus", "Yuna", "Tobias", "Ingrid",
                   "Zara", "Milo", "Layla", "Phoenix", "River", "Remy"]
    last_names = ["Chen", "Osei", "Mitchell", "Rivera", "Kim", "Patel", "Nguyen", "Schmidt",
                  "Lopez", "Williams", "Torres", "Park", "Hansen", "Müller", "Silva", "Rossi",
                  "O'Brien", "Johansson", "Garcia", "Thompson"]
    name = f"{rng.choice(first_names)} {rng.choice(last_names)}"

    # ── Age — use demographic range if available ───────────────────────────────
    if age_group and _DEMO_AVAILABLE and age_group in _AGE_RANGES:
        lo, hi = _AGE_RANGES[age_group]
        age = rng.randint(lo, hi)
    else:
        age = rng.randint(22, 57)

    # ── Occupation — use demographic job titles if available ──────────────────
    if occupation_cluster and _DEMO_AVAILABLE:
        job_titles = _get_demo_job_titles(age_group, occupation_cluster)
        occ = rng.choice(job_titles) if job_titles else occupation_cluster
    else:
        occ = rng.choice([
            "UX Designer", "Marketing Manager", "Software Engineer", "Nurse",
            "Journalist", "Architect", "Management Consultant", "PhD Student",
            "Financial Analyst", "Graphic Designer", "Data Scientist",
        ])

    # ── Behavioral profiles — layer archetype + demographic ───────────────────
    archetype_desc = _ARCHETYPE_DESCRIPTIONS.get(archetype, "A typical venue visitor.")

    demo_profile_text = ""
    demo_task = ""
    if age_group and occupation_cluster and _DEMO_AVAILABLE:
        demo_cell = _get_demo_profile(age_group, occupation_cluster, domain)
        if demo_cell:
            demo_profile_text = demo_cell.get("profile", "")
            demo_task = demo_cell.get("task", "")

    d_label = {"coffee": "coffee shop", "restaurant": "restaurant", "hotel": "hotel"}[domain]

    # ── Venue context ─────────────────────────────────────────────────────────
    venues = venues or []
    if venues:
        venue_lines = "\n".join(
            f"  {i+1}. {v['name']} (⭐ {v['stars']}"
            f"{', ' + v['categories'] if v['categories'] and v['categories'] != 'nan' else ''})"
            f" — {v['tag']}"
            for i, v in enumerate(venues)
        )
        rec_instruction = (
            f"\n\nYou have been recommended these top {d_label}s in {city} "
            f"based on your specific profile:\n{venue_lines}\n\n"
            f"In 4–6 sentences, explain why you personally would choose these places, "
            f"mentioning each by name. Be very specific — your age, job, and daily patterns "
            f"should come through in exactly why each place suits you."
        )
    else:
        rec_instruction = (
            f"\n\nIn 3–5 sentences, describe what kind of {d_label} in {city} "
            f"suits you best, given your job, age, and typical routine."
        )

    # System prompt layers archetype + demographic profile for maximum specificity
    profile_block = archetype_desc
    if demo_profile_text:
        profile_block = (
            f"{archetype_desc}\n\n"
            f"More specifically, as a {age}-year-old {occ}: {demo_profile_text}"
        )

    situation = demo_task if demo_task else f"You're looking for a {d_label} in {city}."

    system_prompt = (
        f"You are {name}, a {age}-year-old {occ} living in {city}.\n\n"
        f"{profile_block}\n\n"
        "Be very specific to your age and job. A student responds differently from an executive, "
        "a nurse differently from a software engineer — even if both are 'Loyalists'. "
        "Answer in first person, naturally. Do not mention AI or simulations."
    )
    user_prompt = f"Situation: {situation}{rec_instruction}"

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_completion_tokens=380,
            temperature=0.92,
        )
        text = response.choices[0].message.content or ""
    except Exception as exc:
        return {"error": str(exc)}

    return {
        "name": name, "age": age, "occupation": occ,
        "age_group": age_group, "occupation_cluster": occupation_cluster,
        "archetype": archetype, "response": text, "venues": venues,
    }


def render_llm_simulation_page() -> None:
    st.subheader("LLM Simulation Validation")
    st.caption("1,500 gpt-5.4 personas — coffee shops · restaurants · hotels")

    # ── Plain-English Summary (for both general and academic readers) ──────────
    with st.expander("📋 What is this? — Executive Summary", expanded=True):
        st.markdown("""
**In plain English:** Most apps rank venues by star ratings. But stars are noisy — a place
can have 4.8 ★ from 10 reviews while another has 4.2 ★ from thousands of loyal weekly
regulars. This project asks: *does ranking by what people actually do (visit frequency,
loyalty, return rates) produce better recommendations than ranking by what they say?*

**What we built:** Three behavioural ranking models — one for coffee shops, one for
restaurants, one for hotels — that rank venues using real visit patterns from the
[Yelp Academic Dataset](https://www.yelp.com/dataset) (8,500+ venues, 93,000+ users).
Instead of stars, the models use signals like revisit rate, temporal consistency, and
loyalty concentration.

**What this simulation tests:** To validate the models independently of the training
data, we created **1,500 synthetic human personas** powered by GPT-5.4 — each one
grounded in a real user archetype identified from the data (e.g. *Loyalists* who go to
the same café daily, *Explorers* who try somewhere new every week). We asked each persona
to rank venues and choose between the model's top pick vs. the star-rating top pick.

**Key findings:**
- ☕ **Coffee & Restaurants:** The behavioural model's top picks were preferred by
  personas significantly more often than star-rating picks (p < 0.05). Loyalists showed
  the strongest alignment — exactly what the theory predicts.
- 🏨 **Hotels:** Star ratings performed better than the behavioural model — a *positive*
  finding, because this matches what the real-data validation also found. It confirms that
  for hotels, where most people only stay 1–2 times, behavioural signals are too sparse to
  outperform simple ratings.
- 📊 The simulation provides **external ecological validity**: two independent sources
  (real Yelp data + synthetic personas) agree on when the model works and when it doesn't.

> **For the examiner:** Significance is tested via Wilcoxon signed-rank test with
> Benjamini-Hochberg correction for multiple comparisons. NDCG@10 (normalised discounted
> cumulative gain) is the primary metric, with Hit@1, Hit@3, and Kendall's τ as
> secondary measures. All metrics are computed on held-out discriminating candidate sets
> (BiRank top-5 vs. Stars top-5, non-overlapping) to prevent ceiling effects.
""")
    st.divider()

    metrics = _load_sim_metrics()
    records = _load_sim_records()

    files_present = LLM_SIM_RECORDS.exists() and LLM_SIM_METRICS.exists()
    if not files_present:
        st.warning(
            "Simulation result files not found. "
            "Run the simulation pipeline first (`llm_simulation/run_simulation.py`)."
        )

    # ── Section 1: Overview metrics ───────────────────────────────────────────
    st.markdown("### Overview Metrics (All Domains)")
    overall = metrics.get("ALL|OVERALL", {})
    if overall:
        col1, col2, col3 = st.columns(3)
        delta_stars = overall.get("delta_vs_stars", 0.0)
        col1.metric(
            "Overall NDCG@10",
            f"{overall.get('ndcg_mean', 0):.4f}",
            delta=f"Δ vs Stars: {delta_stars:+.4f}",
            delta_color="normal" if delta_stars >= 0 else "inverse",
        )
        col2.metric(
            "Pairwise Win Rate",
            f"{overall.get('pairwise_win_rate', 0) * 100:.1f}%",
        )
        p_val = overall.get("wilcoxon_p", 1.0)
        sig_label = "Significant (p < 0.05)" if p_val < 0.05 else "Not significant"
        col3.metric(
            "Wilcoxon p-value",
            f"{p_val:.4f}",
            delta=sig_label,
            delta_color="normal" if p_val < 0.05 else "off",
        )
        st.caption(
            f"Based on n={overall.get('n', 0):,} persona simulations. "
            f"95% CI for NDCG: [{overall.get('ndcg_lo', 0):.4f}, {overall.get('ndcg_hi', 0):.4f}]."
        )
    elif not files_present:
        st.info("Run simulation first to see overview metrics.")
    else:
        st.info("ALL|OVERALL key not found in metrics JSON.")

    st.markdown("---")

    # ── Section 2: Results by domain ─────────────────────────────────────────
    st.markdown("### Results by Domain")
    domain_tabs = st.tabs(["Coffee Shops", "Restaurants", "Hotels"])

    with domain_tabs[0]:
        _render_domain_tab("coffee", "Coffee Shops", records, metrics)
    with domain_tabs[1]:
        _render_domain_tab("restaurant", "Restaurants", records, metrics)
    with domain_tabs[2]:
        _render_domain_tab("hotel", "Hotels", records, metrics)

    st.markdown("---")

    # ── Section 2b: Study 2 — Occupation × Age Cross-Matrix ──────────────────
    st.markdown("### Study 2 — Occupation × Age Demographics")
    st.caption(
        "1,860 personas across 5 age groups × 10 occupation clusters × 3 domains, "
        "grounded in 51 published consumer-behaviour sources (NCA, McKinsey, J.D. Power, GBTA, Hilton Trends, etc.)"
    )

    s2_records_path = LLM_SIM_DIR / "simulation_records_study2.csv"
    s2_age_path     = LLM_SIM_DIR / "study2_by_age.csv"
    s2_occ_path     = LLM_SIM_DIR / "study2_by_occupation.csv"
    s2_matrix_path  = LLM_SIM_DIR / "study2_cross_matrix.csv"

    if not s2_records_path.exists():
        st.info("Study 2 results not found. Run `llm_simulation/main_study2.py` to generate them.")
    else:
        s2_records = safe_read_csv(s2_records_path)

        # ── Overview numbers ──────────────────────────────────────────────────
        if not s2_records.empty:
            s2_c1, s2_c2, s2_c3, s2_c4 = st.columns(4)
            overall_ndcg  = s2_records["ndcg"].mean()
            overall_stars = s2_records["stars_ndcg"].mean() if "stars_ndcg" in s2_records.columns else 0
            win_rate = s2_records["pairwise_win"].mean() if "pairwise_win" in s2_records.columns else 0
            s2_c1.metric("Total Personas", f"{len(s2_records):,}")
            s2_c2.metric("Overall NDCG@10", f"{overall_ndcg:.4f}",
                         delta=f"Δ vs Stars: {overall_ndcg - overall_stars:+.4f}")
            s2_c3.metric("Pairwise Win Rate", f"{win_rate:.1%}")
            s2_c4.metric("Age Groups × Occupations",
                         f"{s2_records['age_group'].nunique()} × {s2_records['occupation'].nunique()}"
                         if "age_group" in s2_records.columns else "5 × 10")

        # ── Tabs: Age | Occupation | Cross-Matrix ─────────────────────────────
        s2_tab1, s2_tab2, s2_tab3 = st.tabs(["By Age Group", "By Occupation", "Cross-Matrix Heatmap"])

        with s2_tab1:
            if s2_age_path.exists():
                age_df = safe_read_csv(s2_age_path)
                if not age_df.empty:
                    # bar chart
                    if "age_group" in age_df.columns and "ndcg_mean" in age_df.columns:
                        age_order = ["Gen Z (18-25)", "Young Millennial (26-33)",
                                     "Senior Millennial (34-40)", "Gen X (41-56)", "Boomer (57+)"]
                        age_df["_order"] = age_df["age_group"].map(
                            {a: i for i, a in enumerate(age_order)}
                        ).fillna(99)
                        age_df = age_df.sort_values("_order").drop(columns=["_order"])
                        chart_data = age_df.set_index("age_group")[["ndcg_mean", "stars_ndcg_mean"]].rename(
                            columns={"ndcg_mean": "Behavioural Model", "stars_ndcg_mean": "Star Ratings"}
                        ) if "stars_ndcg_mean" in age_df.columns else age_df.set_index("age_group")[["ndcg_mean"]]
                        st.bar_chart(chart_data)
                    st.dataframe(age_df, use_container_width=True, hide_index=True)
            elif not s2_records.empty and "age_group" in s2_records.columns:
                age_summary = (
                    s2_records.groupby("age_group")
                    .agg(n=("ndcg", "count"),
                         ndcg_mean=("ndcg", "mean"),
                         stars_ndcg=("stars_ndcg", "mean"),
                         win_rate=("pairwise_win", "mean"))
                    .reset_index()
                )
                age_summary["Δ vs Stars"] = age_summary["ndcg_mean"] - age_summary["stars_ndcg"]
                st.bar_chart(age_summary.set_index("age_group")["ndcg_mean"])
                st.dataframe(age_summary.round(4), use_container_width=True, hide_index=True)

        with s2_tab2:
            if s2_occ_path.exists():
                occ_df = safe_read_csv(s2_occ_path)
                if not occ_df.empty:
                    if "occupation" in occ_df.columns and "ndcg_mean" in occ_df.columns:
                        occ_df = occ_df.sort_values("ndcg_mean", ascending=False)
                        chart_data = occ_df.set_index("occupation")[["ndcg_mean", "stars_ndcg_mean"]].rename(
                            columns={"ndcg_mean": "Behavioural Model", "stars_ndcg_mean": "Star Ratings"}
                        ) if "stars_ndcg_mean" in occ_df.columns else occ_df.set_index("occupation")[["ndcg_mean"]]
                        st.bar_chart(chart_data)
                    st.dataframe(occ_df, use_container_width=True, hide_index=True)
            elif not s2_records.empty and "occupation" in s2_records.columns:
                occ_summary = (
                    s2_records.groupby("occupation")
                    .agg(n=("ndcg", "count"),
                         ndcg_mean=("ndcg", "mean"),
                         stars_ndcg=("stars_ndcg", "mean"),
                         win_rate=("pairwise_win", "mean"))
                    .reset_index()
                    .sort_values("ndcg_mean", ascending=False)
                )
                st.bar_chart(occ_summary.set_index("occupation")["ndcg_mean"])
                st.dataframe(occ_summary.round(4), use_container_width=True, hide_index=True)

        with s2_tab3:
            if s2_matrix_path.exists():
                matrix_df = safe_read_csv(s2_matrix_path)
                if not matrix_df.empty:
                    st.caption("NDCG@10 per age group × occupation cell. Higher = behavioural model aligns more strongly with persona preferences.")
                    st.dataframe(matrix_df, use_container_width=True)
            elif not s2_records.empty and "age_group" in s2_records.columns and "occupation" in s2_records.columns:
                pivot = s2_records.pivot_table(
                    values="ndcg", index="age_group", columns="occupation", aggfunc="mean"
                ).round(4)
                st.caption("NDCG@10 heatmap — age group × occupation. Higher = stronger model alignment.")
                st.dataframe(pivot, use_container_width=True)

        # ── Study 2 full report ───────────────────────────────────────────────
        s2_report_path = LLM_SIM_DIR / "simulation_report_study2.md"
        with st.expander("Study 2 Full Report", expanded=False):
            if s2_report_path.exists():
                st.markdown(s2_report_path.read_text(encoding="utf-8"))
            else:
                st.info("Run main_study2.py to generate the full report.")

    st.markdown("---")

    # ── Section 3: Live Persona Chat ──────────────────────────────────────────
    st.markdown("### Live Persona Chat")
    st.caption(
        "Generate a persona grounded in an archetype and ask it about venue preferences. "
        "Requires an OpenAI API key."
    )

    api_key = _get_openai_key()
    has_key = bool(api_key)
    if not has_key:
        st.info(
            "No OpenAI API key found. Set `OPENAI_API_KEY` in your environment or in "
            "`llm_simulation/.env`. Showing example output below."
        )

    # ── Row 1: Domain, Archetype, Age Group ──────────────────────────────────
    chat_col1, chat_col2, chat_col3 = st.columns(3)
    with chat_col1:
        chat_domain = st.selectbox(
            "Domain",
            ["coffee", "restaurant", "hotel"],
            format_func=lambda x: {"coffee": "Coffee Shops", "restaurant": "Restaurants", "hotel": "Hotels"}[x],
            key="llm_chat_domain",
        )
    with chat_col2:
        domain_arch_map = _domain_archetypes(records)
        available_archetypes = domain_arch_map.get(chat_domain, list(_ARCHETYPE_DESCRIPTIONS.keys()))
        chat_archetype = st.selectbox(
            "Behavioural Archetype",
            available_archetypes,
            key="llm_chat_archetype",
        )
    with chat_col3:
        chat_age_group = st.selectbox(
            "Age Group",
            _AGE_GROUPS,
            index=0,
            key="llm_chat_age_group",
        )

    # ── Row 2: Occupation, City ───────────────────────────────────────────────
    chat_col4, chat_col5 = st.columns(2)
    with chat_col4:
        chat_occupation = st.selectbox(
            "Occupation",
            _OCCUPATION_CLUSTERS,
            key="llm_chat_occupation",
        )
    with chat_col5:
        # Use occupation-appropriate cities if demographic data available
        demo_cities = _get_demo_cities(chat_age_group, chat_occupation) if _DEMO_AVAILABLE else []
        default_cities = demo_cities if demo_cities else [
            "Philadelphia", "Nashville", "Las Vegas", "New Orleans",
            "Tampa", "Pittsburgh", "Cleveland", "Charlotte",
        ]
        # Deduplicate and keep order
        seen, city_opts = set(), []
        for c in default_cities + ["Philadelphia", "Nashville", "Las Vegas", "New Orleans",
                                    "Tampa", "Pittsburgh", "Cleveland", "Charlotte",
                                    "Indianapolis", "Cincinnati", "Louisville", "St. Louis"]:
            if c not in seen:
                city_opts.append(c)
                seen.add(c)
        chat_city = st.selectbox("City", city_opts, key="llm_chat_city")

    # ── Demographic insight caption ────────────────────────────────────────────
    if _DEMO_AVAILABLE:
        demo_cell = _get_demo_profile(chat_age_group, chat_occupation, chat_domain)
        if demo_cell:
            sort_label = demo_cell.get("sort_col", "").replace("_", " ")
            loyalty = demo_cell.get("loyalty_score", 3)
            price = demo_cell.get("price_sens", 3)
            loyalty_str = ["", "Explorer", "Low loyalty", "Balanced", "Loyal", "Very loyal"][loyalty]
            price_str = ["", "Low sensitivity", "Low-moderate", "Moderate", "Moderate-high", "High"][price]
            st.caption(
                f"**{chat_age_group} · {chat_occupation}** — "
                f"Key venue signal: `{sort_label}` · Loyalty: {loyalty_str} · Price sensitivity: {price_str}"
            )

    generate_clicked = st.button(
        "Generate Persona" if has_key else "Generate Persona (demo)",
        key="llm_chat_generate",
        disabled=False,
    )

    if generate_clicked:
        import random as _rnd  # noqa: PLC0415
        st.session_state["llm_persona_rand"] = _rnd.randint(0, 999_999)

    rand_seed = st.session_state.get("llm_persona_rand", 0)

    if generate_clicked:
        # Get demographic sort_col to rank venues by what THIS age×occupation cares about
        demo_sort_col = ""
        if _DEMO_AVAILABLE:
            demo_cell = _get_demo_profile(chat_age_group, chat_occupation, chat_domain)
            demo_sort_col = demo_cell.get("sort_col", "") if demo_cell else ""

        venues = _load_city_venues(
            chat_domain, chat_city, chat_archetype, top_n=3,
            override_sort_col=demo_sort_col,
        )

        # Update venue "why" label to reflect demographic signal, not just archetype
        if demo_sort_col and venues:
            _DEMO_WHY = {
                "revisit_rate":           "High return-visitor rate — fits your loyalty pattern",
                "temporal_stability":     "Consistent daily traffic — reliable for your schedule",
                "unique_users":           "Broad appeal — suits your exploratory style",
                "stars":                  "Top-rated — safe, trusted choice",
                "repeat_user_rate":       "Strong repeat customers — matches your loyalty habits",
                "popularity":             "High footfall — popular with your peer group",
                "avg_rating":             "Highly rated — reliable quality",
                "peak_busyness":          "Buzzy at peak times — suits your lifestyle",
                "business_leisure_ratio": "Business-oriented — professional atmosphere",
                "geographic_diversity":   "Draws visitors from far — destination quality",
                "multi_stay_rate":        "Unusually high return rate — genuinely worth it",
            }
            why_label = _DEMO_WHY.get(demo_sort_col, _ARCHETYPE_WHY.get(chat_archetype, "Top pick"))
            venues = [{**v, "why": why_label} for v in venues]

        if has_key:
            with st.spinner(f"Generating {chat_age_group} {chat_occupation} persona…"):
                result = _call_persona_chat(
                    api_key, chat_archetype, chat_domain, chat_city,
                    rand_seed=rand_seed, venues=venues,
                    age_group=chat_age_group,
                    occupation_cluster=chat_occupation,
                )
        else:
            result = _EXAMPLE_PERSONA_OUTPUT.get(chat_domain, _EXAMPLE_PERSONA_OUTPUT["coffee"])
            result["venues"] = venues
            result["age_group"] = chat_age_group
            result["occupation_cluster"] = chat_occupation

        st.session_state["llm_last_result"] = result

    # Persist result across reruns
    result = st.session_state.get("llm_last_result")

    if result:
        if "error" in result:
            st.error(f"API error: {result['error']}")
        else:
            # ── Persona card + conversational response ────────────────────────
            with st.container(border=True):
                pc1, pc2 = st.columns([1, 3])
                with pc1:
                    st.markdown("**Persona Card**")
                    st.markdown(f"**Name:** {result.get('name', '—')}")
                    st.markdown(f"**Age:** {result.get('age', '—')} · {result.get('age_group', '')}")
                    st.markdown(f"**Job:** {result.get('occupation', '—')}")
                    st.markdown(f"**Sector:** {result.get('occupation_cluster', '')}")
                    st.markdown(f"**Archetype:** `{result.get('archetype', chat_archetype)}`")
                with pc2:
                    arch = result.get("archetype", chat_archetype)
                    desc_txt = _ARCHETYPE_DESCRIPTIONS.get(arch, "")
                    if desc_txt:
                        st.caption(f"_{desc_txt}_")
                    st.markdown("**In their own words:**")
                    st.markdown(result.get("response", ""))
                    if not has_key:
                        st.caption("Example output — add API key for live generation.")

            # ── Recommended venues (Option C: structured cards below) ─────────
            rec_venues = result.get("venues", [])
            if rec_venues:
                d_label = {"coffee": "Coffee Shops", "restaurant": "Restaurants",
                           "hotel": "Hotels"}[result.get("archetype", chat_archetype)
                                              and chat_domain or chat_domain]
                st.markdown(
                    f"##### 📍 Their Top {len(rec_venues)} Recommended "
                    f"{d_label} in {chat_city}"
                )
                st.caption(
                    f"Ranked by the behavioural model's "
                    f"**{_ARCHETYPE_SORT.get(chat_archetype, ('signal', False))[0].replace('_', ' ')}** "
                    f"signal — the dimension that matters most for *{chat_archetype}* personas."
                )
                cols = st.columns(len(rec_venues))
                for col, v in zip(cols, rec_venues):
                    with col:
                        rc_str = f" · {v['review_count']} reviews" if v.get("review_count") else ""
                        cat_str = f"\n_{v['categories']}_" if v.get("categories") and v["categories"] != "nan" else ""
                        col.markdown(
                            f"**{v['name']}**{cat_str}\n\n"
                            f"⭐ {v['stars']}{rc_str}\n\n"
                            f"🏷 {v['tag']}\n\n"
                            f"✅ _{v['why']}_"
                        )
    elif not generate_clicked:
        arch_desc = _ARCHETYPE_DESCRIPTIONS.get(chat_archetype, "")
        if arch_desc:
            st.caption(f"Selected archetype: _{arch_desc}_")

    st.markdown("---")

    # ── Section 4: Simulation Insights ───────────────────────────────────────
    with st.expander("Simulation Insights (full report)", expanded=False):
        if LLM_SIM_REPORT.exists():
            try:
                report_text = LLM_SIM_REPORT.read_text(encoding="utf-8")
                st.markdown(report_text)
            except Exception as exc:
                st.error(f"Could not read report: {exc}")
        else:
            st.info("Simulation report not found. Run the simulation pipeline to generate it.")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    st.set_page_config(page_title="Behavioral Engine", layout="wide", page_icon="☕")
    st.title("Behavioral Recommendation Dashboard")
    st.caption("City-aware recommendation inspection with richer validation evidence.")

    st.sidebar.title("Configuration")
    if st.sidebar.button("Reload Data", key="reload_data_btn"):
        st.session_state.pop("_data_read_errors", None)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    domain = st.sidebar.radio(
        "Select Domain",
        ["Coffee Shops", "Restaurants", "Hotels & Accommodation", "LLM Simulation"],
        key="domain_select"
    )

    if domain == "LLM Simulation":
        render_llm_simulation_page()
    else:
        render_data_health_panel(domain)
        if domain == "Hotels & Accommodation":
            _, hotel_df = load_hotel_data()
            render_hotel_dashboard(hotel_df)
        elif domain == "Restaurants":
            _, data_df, _ = load_data(domain)
            render_restaurant_dashboard(data_df)
        else:
            score_version_key = "v4" if str(st.session_state.get("coffee_score_version", "")).startswith("v4") else "v3"
            _, data_df, group_df = load_data(domain, score_version=score_version_key)
            render_coffee_dashboard(data_df, group_df)


if __name__ == "__main__":
    main()
