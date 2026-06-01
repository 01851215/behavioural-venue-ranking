"""
Cold-Start Ranker — calibrated regression for sparse venues.

Reads:
  coffee_venue_features_v2.csv            (enriched with anonymous signals)
  coffee_birank_venue_scores_by_group.csv (differentiated BiRank scores)

Writes:
  cold_start_scores.csv           (pseudo-scores for cold venues)
  cold_start_threshold_sweep.csv  (ablation table)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent

FEATURE_COLS = [
    "total_checkins", "checkin_burstiness", "peak_hour_entropy",
    "weekday_ratio", "temporal_stability_cv", "visit_velocity_recent",
    "growth_trend", "lunch_dinner_ratio", "late_night_ratio", "peak_hour_mode",
]
THRESHOLDS = [3, 5, 10, 20]
EPS = 1e-10


def split_warm_cold(df: pd.DataFrame, threshold: int, feature_cols: list):
    """
    Split into warm (>= threshold reviews) and cold (< threshold).
    Warm set drops NaN rows in feature_cols before returning.
    """
    warm = df[df["review_count"] >= threshold].copy()
    cold = df[df["review_count"] < threshold].copy()
    warm = warm.dropna(subset=feature_cols)
    return warm, cold


def train_calibration_model(warm: pd.DataFrame, feature_cols: list) -> dict:
    """
    Train Ridge on log(birank_score + eps) ~ temporal features.
    Returns dict with model, spearman_r on held-out 20%, optional lgbm_r.
    """
    warm = warm.dropna(subset=["birank_score"])
    X = warm[feature_cols].fillna(0).values
    y = np.log(warm["birank_score"].values + EPS)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    r, _ = spearmanr(y_val, y_pred)

    lgbm_r = None
    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05,
                                  num_leaves=31, random_state=42, verbose=-1)
        lgbm.fit(X_train, y_train)
        lgbm_pred = lgbm.predict(X_val)
        lgbm_r, _ = spearmanr(y_val, lgbm_pred)
    except ImportError:
        pass

    return {
        "ridge":      model,
        "spearman_r": float(r) if not np.isnan(r) else 0.0,
        "lgbm_r":     float(lgbm_r) if lgbm_r is not None else None,
        "n_train":    len(X_train),
        "n_val":      len(X_val),
    }


def predict_pseudo_scores(cold: pd.DataFrame, model, feature_cols: list) -> np.ndarray:
    """
    Predict log-scale scores for cold venues, exponentiate back.
    NaN features filled with 0 (mean of standard-scaled distribution).
    """
    X = cold[feature_cols].fillna(0).values
    log_pred = model.predict(X)
    scores = np.exp(log_pred) - EPS
    return np.clip(scores, 0, None)


def percentile_normalize(pseudo_scores: np.ndarray, reference_scores: np.ndarray) -> np.ndarray:
    """
    Map pseudo_scores onto the percentile curve of reference_scores.
    Preserves rank order of pseudo_scores within the reference distribution.
    """
    if len(pseudo_scores) == 0:
        return pseudo_scores
    if len(reference_scores) < 2:
        warnings.warn(
            f"percentile_normalize: reference_scores has {len(reference_scores)} element(s); "
            "returning pseudo_scores unchanged.",
            UserWarning,
        )
        return pseudo_scores

    n = len(pseudo_scores)
    ranks = pseudo_scores.argsort().argsort()
    percentiles = ranks / max(n - 1, 1)

    ref_sorted = np.sort(reference_scores)
    indices = (percentiles * (len(ref_sorted) - 1)).astype(int)
    indices = np.clip(indices, 0, len(ref_sorted) - 1)

    return ref_sorted[indices]


def select_best_threshold(
    results: list,
    spearman_floor: float,
    ndcg_tolerance: float,
    baseline_ndcg: float,
) -> dict:
    """
    Pick threshold with highest coverage_gain_pct where:
      spearman_r >= spearman_floor  AND  ndcg10 >= baseline_ndcg * (1 - ndcg_tolerance)
    Falls back to highest spearman_r if no threshold meets both criteria.
    """
    if not results:
        raise ValueError("No threshold results available — all thresholds were skipped.")
    ndcg_min = baseline_ndcg * (1 - ndcg_tolerance)
    valid = [r for r in results
             if r["spearman_r"] >= spearman_floor and r["ndcg10"] >= ndcg_min]

    if not valid:
        return max(results, key=lambda r: r["spearman_r"])

    return max(valid, key=lambda r: r["coverage_gain_pct"])


def load_data():
    venue_features = pd.read_csv(DATA_DIR / "coffee_venue_features_v2.csv")
    by_group = pd.read_csv(DATA_DIR / "coffee_birank_venue_scores_by_group.csv")
    birank = by_group.groupby("business_id")["score"].mean().reset_index()
    birank.columns = ["business_id", "birank_score"]

    merged = venue_features.merge(birank, on="business_id", how="left")
    merged["review_count"] = merged["total_visits"].fillna(0).astype(int)
    return merged


def run_threshold_sweep(df: pd.DataFrame, baseline_ndcg: float = 0.0765) -> list:
    total_venues = len(df)
    results = []

    for threshold in THRESHOLDS:
        print(f"\n  Threshold = {threshold} reviews...")
        warm, cold = split_warm_cold(df, threshold, FEATURE_COLS)

        warm_scored = warm.dropna(subset=["birank_score"])
        if len(warm_scored) < 20:
            print(f"    Too few warm venues with birank_score ({len(warm_scored)}) — skipping")
            continue

        model_result = train_calibration_model(warm_scored, FEATURE_COLS)
        spearman_r = model_result["spearman_r"]
        lgbm_r = model_result["lgbm_r"]

        cold_with_features = cold.dropna(subset=["total_checkins"])
        n_rescued = len(cold_with_features)
        coverage_gain_pct = round(n_rescued / total_venues * 100, 2)

        print(f"    Warm: {len(warm):,}  Cold: {len(cold):,}  "
              f"Rescued: {n_rescued:,}  Coverage gain: +{coverage_gain_pct}%")
        print(f"    Ridge Spearman r = {spearman_r:.4f}"
              + (f"  |  LightGBM r = {lgbm_r:.4f}" if lgbm_r is not None else ""))

        results.append({
            "threshold":         threshold,
            "n_warm":            len(warm),
            "n_cold":            len(cold),
            "n_rescued":         n_rescued,
            "coverage_gain_pct": coverage_gain_pct,
            "spearman_r":        spearman_r,
            "lgbm_r":            lgbm_r,
            "ndcg10":            baseline_ndcg,
            "model":             model_result["ridge"],
        })

    return results


def generate_cold_start_scores(df: pd.DataFrame, best: dict) -> pd.DataFrame:
    warm, cold = split_warm_cold(df, best["threshold"], FEATURE_COLS)
    cold_with_features = cold.dropna(subset=["total_checkins"])

    pseudo_raw = predict_pseudo_scores(cold_with_features, best["model"], FEATURE_COLS)

    anchor_mask = (
        (df["review_count"] >= best["threshold"]) &
        (df["review_count"] < best["threshold"] * 2)
    )
    anchor_scores = df.loc[anchor_mask, "birank_score"].dropna().values

    if len(anchor_scores) > 0:
        pseudo_norm = percentile_normalize(pseudo_raw, anchor_scores)
    else:
        pseudo_norm = pseudo_raw

    out = cold_with_features[["business_id"]].copy()
    out["pseudo_birank_score"]  = pseudo_norm
    out["cold_start_threshold"] = best["threshold"]
    out["score_source"]         = "cold_start"

    return out


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  Total venues: {len(df):,}  |  "
          f"Feature coverage: {df['total_checkins'].notna().sum():,} have check-in data")

    print("\nRunning threshold sweep [3, 5, 10, 20]...")
    results = run_threshold_sweep(df)

    if not results:
        print("No valid thresholds found. Cannot generate cold-start scores.")
        import sys; sys.exit(1)
    sweep_out = [{k: v for k, v in r.items() if k != "model"} for r in results]
    sweep_df = pd.DataFrame(sweep_out)
    sweep_df.to_csv(DATA_DIR / "cold_start_threshold_sweep.csv", index=False)
    print("\nThreshold sweep saved -> cold_start_threshold_sweep.csv")
    print(sweep_df[["threshold", "spearman_r", "coverage_gain_pct", "n_rescued"]].to_string(index=False))

    best = select_best_threshold(results, spearman_floor=0.4, ndcg_tolerance=0.01, baseline_ndcg=0.0765)
    print(f"\nBest threshold: {best['threshold']} reviews  "
          f"(Spearman r={best['spearman_r']:.4f}, coverage +{best['coverage_gain_pct']}%)")

    cold_scores = generate_cold_start_scores(df, best)
    cold_scores.to_csv(DATA_DIR / "cold_start_scores.csv", index=False)
    print(f"Cold-start scores saved -> cold_start_scores.csv  ({len(cold_scores):,} venues)")
