import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_data_prep import (
    compute_consistency_score,
    compute_future_revisit_rate,
    build_causal_dataset,
)


def make_venue_df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "business_id": [f"b{i}" for i in range(n)],
        "weekday_ratio":         rng.uniform(0, 1, n),
        "peak_hour_entropy":     rng.uniform(0, 4.5, n),
        "total_visits":          rng.integers(10, 500, n).astype(float),
        "unique_users":          rng.integers(5, 200, n).astype(float),
        "gini_user_contribution": rng.uniform(0, 0.4, n),
    })


def test_consistency_score_produces_treatment_column():
    df = make_venue_df()
    result = compute_consistency_score(df)
    assert "consistency_score" in result.columns
    assert "treatment" in result.columns

def test_consistency_score_median_split_is_balanced():
    df = make_venue_df(n=100)
    result = compute_consistency_score(df)
    treated = (result["treatment"] == 1).sum()
    control = (result["treatment"] == 0).sum()
    assert abs(treated - control) <= 5, f"Imbalanced split: {treated} vs {control}"

def test_consistency_score_nan_inputs_excluded():
    df = make_venue_df(n=10)
    df.loc[0, "weekday_ratio"] = np.nan
    df.loc[1, "peak_hour_entropy"] = np.nan
    result = compute_consistency_score(df)
    assert pd.isna(result.loc[0, "treatment"])
    assert pd.isna(result.loc[1, "treatment"])

def test_consistency_score_range():
    df = make_venue_df(n=50)
    result = compute_consistency_score(df)
    valid = result["consistency_score"].dropna()
    assert valid.min() >= -1.0 - 1e-9
    assert valid.max() <= 1.0 + 1e-9

def test_consistency_score_treatment_is_binary():
    df = make_venue_df(n=20)
    result = compute_consistency_score(df)
    valid_treatment = result["treatment"].dropna()
    assert set(valid_treatment.unique()).issubset({0.0, 1.0})

def test_future_revisit_rate_full_retention():
    pre = {"b1": {"u1", "u2", "u3"}}
    post = {"b1": {"u1", "u2", "u3", "u4"}}
    rates = compute_future_revisit_rate(pre, post, {"b1"})
    assert rates["b1"] == pytest.approx(1.0)

def test_future_revisit_rate_no_retention():
    pre = {"b1": {"u1", "u2"}}
    post = {"b1": {"u3", "u4"}}
    rates = compute_future_revisit_rate(pre, post, {"b1"})
    assert rates["b1"] == pytest.approx(0.0)

def test_future_revisit_rate_partial():
    pre = {"b1": {"u1", "u2", "u3", "u4"}}
    post = {"b1": {"u1", "u2"}}
    rates = compute_future_revisit_rate(pre, post, {"b1"})
    assert rates["b1"] == pytest.approx(0.5)

def test_future_revisit_rate_excludes_zero_pre():
    pre = {"b1": {"u1"}, "b2": set()}
    post = {"b1": {"u1"}, "b2": {"u5"}}
    rates = compute_future_revisit_rate(pre, post, {"b1", "b2"})
    assert "b1" in rates
    assert "b2" not in rates

def test_future_revisit_rate_missing_post():
    pre = {"b1": {"u1", "u2"}}
    post = {}
    rates = compute_future_revisit_rate(pre, post, {"b1"})
    assert rates["b1"] == pytest.approx(0.0)

def test_build_causal_dataset_output_columns():
    df = compute_consistency_score(make_venue_df(n=10))
    rates = {f"b{i}": 0.1 * i for i in range(10)}
    result = build_causal_dataset(df, rates)
    expected = {"business_id", "consistency_score", "treatment",
                "future_revisit_rate", "total_visits", "unique_users",
                "gini_user_contribution"}
    assert expected.issubset(set(result.columns))

def test_build_causal_dataset_drops_missing_outcome():
    df = compute_consistency_score(make_venue_df(n=5))
    rates = {"b0": 0.1, "b1": 0.2}
    result = build_causal_dataset(df, rates)
    assert len(result) == 2

def test_build_causal_dataset_drops_nan_treatment():
    df = make_venue_df(n=5)
    df.loc[0, "weekday_ratio"] = np.nan
    df = compute_consistency_score(df)
    rates = {f"b{i}": 0.1 for i in range(5)}
    result = build_causal_dataset(df, rates)
    assert "b0" not in result["business_id"].values
