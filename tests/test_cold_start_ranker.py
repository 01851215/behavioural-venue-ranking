import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cold_start_ranker import (
    split_warm_cold,
    train_calibration_model,
    predict_pseudo_scores,
    percentile_normalize,
    select_best_threshold,
)

FEATURE_COLS = [
    "total_checkins", "checkin_burstiness", "peak_hour_entropy",
    "weekday_ratio", "temporal_stability_cv", "visit_velocity_recent",
    "growth_trend", "lunch_dinner_ratio", "late_night_ratio", "peak_hour_mode",
]


def make_venue_df(n=100, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "business_id": [f"biz_{i}" for i in range(n)],
        "birank_score": rng.exponential(0.0001, n),
        "review_count": rng.integers(0, 30, n),
    })
    for col in FEATURE_COLS:
        df[col] = rng.uniform(0, 1, n)
    return df


def test_split_warm_cold_threshold_5():
    df = make_venue_df()
    warm, cold = split_warm_cold(df, threshold=5, feature_cols=FEATURE_COLS)
    assert (warm["review_count"] >= 5).all()
    assert (cold["review_count"] < 5).all()
    assert len(warm) + len(cold) == len(df)


def test_split_warm_cold_no_nan_in_warm_features():
    df = make_venue_df()
    warm, _ = split_warm_cold(df, threshold=5, feature_cols=FEATURE_COLS)
    assert not warm[FEATURE_COLS].isnull().any().any()


def test_train_model_returns_dict_with_ridge():
    df = make_venue_df()
    warm, _ = split_warm_cold(df, threshold=5, feature_cols=FEATURE_COLS)
    result = train_calibration_model(warm, feature_cols=FEATURE_COLS)
    assert "ridge" in result
    assert "spearman_r" in result
    assert -1.0 <= result["spearman_r"] <= 1.0


def test_train_model_spearman_reasonable_on_synthetic():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "business_id": [f"b{i}" for i in range(n)],
        "review_count": rng.integers(10, 50, n),
    })
    df["total_checkins"] = rng.integers(1, 500, n)
    df["birank_score"] = df["total_checkins"] * 1e-6 + rng.normal(0, 1e-7, n)
    for col in FEATURE_COLS:
        if col != "total_checkins":
            df[col] = rng.uniform(0, 1, n)
    warm, _ = split_warm_cold(df, threshold=5, feature_cols=FEATURE_COLS)
    result = train_calibration_model(warm, feature_cols=FEATURE_COLS)
    assert result["spearman_r"] > 0.1


def test_predict_pseudo_scores_output_shape():
    df = make_venue_df()
    warm, cold = split_warm_cold(df, threshold=5, feature_cols=FEATURE_COLS)
    model_result = train_calibration_model(warm, feature_cols=FEATURE_COLS)
    scores = predict_pseudo_scores(cold, model_result["ridge"], feature_cols=FEATURE_COLS)
    assert len(scores) == len(cold)
    assert (scores >= 0).all(), "Pseudo-scores must be non-negative"


def test_percentile_normalize_maps_into_reference_range():
    ref_scores = np.array([0.001, 0.002, 0.005, 0.008, 0.01])
    pseudo = np.array([0.5, 1.0, 1.5])
    normalized = percentile_normalize(pseudo, ref_scores)
    assert normalized.min() >= ref_scores.min() * 0.5
    assert normalized.max() <= ref_scores.max() * 2.0


def test_percentile_normalize_preserves_rank_order():
    ref_scores = np.linspace(0.001, 0.01, 100)
    pseudo = np.array([1.0, 2.0, 3.0])
    normalized = percentile_normalize(pseudo, ref_scores)
    assert normalized[0] < normalized[1] < normalized[2]


def test_select_best_threshold_picks_highest_coverage_above_floor():
    results = [
        {"threshold": 3,  "spearman_r": 0.35, "coverage_gain_pct": 20.0, "ndcg10": 0.076},
        {"threshold": 5,  "spearman_r": 0.45, "coverage_gain_pct": 15.0, "ndcg10": 0.076},
        {"threshold": 10, "spearman_r": 0.55, "coverage_gain_pct": 10.0, "ndcg10": 0.077},
        {"threshold": 20, "spearman_r": 0.60, "coverage_gain_pct":  5.0, "ndcg10": 0.077},
    ]
    best = select_best_threshold(results, spearman_floor=0.4, ndcg_tolerance=0.01, baseline_ndcg=0.0765)
    assert best["threshold"] == 5


def test_select_best_threshold_excludes_ndcg_degradation():
    results = [
        {"threshold": 5,  "spearman_r": 0.50, "coverage_gain_pct": 15.0, "ndcg10": 0.060},
        {"threshold": 10, "spearman_r": 0.55, "coverage_gain_pct": 10.0, "ndcg10": 0.076},
    ]
    best = select_best_threshold(results, spearman_floor=0.4, ndcg_tolerance=0.01, baseline_ndcg=0.0765)
    assert best["threshold"] == 10
