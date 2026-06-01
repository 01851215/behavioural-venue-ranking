import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bvr.causal.psm import (
    fit_propensity_model,
    nearest_neighbour_match,
    compute_smd,
    estimate_ate,
    mahalanobis_match,
    build_balance_table,
)

CONFOUNDER_COLS = ["total_visits", "unique_users", "gini_user_contribution"]


def make_psm_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    total_visits = rng.exponential(200, n)
    unique_users = total_visits * rng.uniform(0.1, 0.5, n)
    gini = rng.uniform(0, 0.4, n)
    logit = (total_visits / 500 - 0.5) * 2
    prob = 1 / (1 + np.exp(-logit))
    treatment = (rng.uniform(0, 1, n) < prob).astype(float)
    outcome = 0.05 * treatment + 0.001 * total_visits / 200 + rng.normal(0, 0.02, n)
    outcome = np.clip(outcome, 0, 1)
    return pd.DataFrame({
        "business_id": [f"b{i}" for i in range(n)],
        "treatment": treatment,
        "future_revisit_rate": outcome,
        "total_visits": total_visits,
        "unique_users": unique_users,
        "gini_user_contribution": gini,
    })


def test_fit_propensity_model_returns_scores_in_unit_interval():
    df = make_psm_df()
    result = fit_propensity_model(df, CONFOUNDER_COLS)
    ps = result["propensity_score"]
    assert (ps >= 0).all() and (ps <= 1).all()

def test_fit_propensity_model_adds_logit_column():
    df = make_psm_df()
    result = fit_propensity_model(df, CONFOUNDER_COLS)
    assert "logit_propensity" in result.columns
    assert result["logit_propensity"].notna().all()

def test_fit_propensity_model_preserves_row_count():
    df = make_psm_df(n=100)
    result = fit_propensity_model(df, CONFOUNDER_COLS)
    assert len(result) == 100

def test_compute_smd_perfect_balance_is_zero():
    n = 100
    df = pd.DataFrame({
        "treatment": [0] * (n // 2) + [1] * (n // 2),
        "total_visits": [100.0] * n,
        "unique_users": [50.0] * n,
        "gini_user_contribution": [0.1] * n,
    })
    smds = compute_smd(df, CONFOUNDER_COLS)
    for col, smd in smds.items():
        assert smd == pytest.approx(0.0, abs=1e-9), f"{col}: expected 0, got {smd}"

def test_compute_smd_imbalanced_produces_nonzero():
    df = pd.DataFrame({
        "treatment": [0] * 50 + [1] * 50,
        "total_visits": [100.0] * 50 + [300.0] * 50,
        "unique_users": [50.0] * 100,
        "gini_user_contribution": [0.1] * 100,
    })
    smds = compute_smd(df, CONFOUNDER_COLS)
    assert smds["total_visits"] > 0.5

def test_compute_smd_returns_all_confounders():
    df = make_psm_df(n=50)
    smds = compute_smd(df, CONFOUNDER_COLS)
    assert set(smds.keys()) == set(CONFOUNDER_COLS)

def test_nearest_neighbour_match_output_columns():
    df = fit_propensity_model(make_psm_df(), CONFOUNDER_COLS)
    pairs = nearest_neighbour_match(df)
    expected = {"treated_id", "control_id", "treated_propensity",
                "control_propensity", "treated_outcome", "control_outcome"}
    assert expected.issubset(set(pairs.columns))

def test_nearest_neighbour_match_no_duplicate_controls():
    df = fit_propensity_model(make_psm_df(), CONFOUNDER_COLS)
    pairs = nearest_neighbour_match(df)
    assert pairs["control_id"].nunique() == len(pairs), "Duplicate control venues found"

def test_nearest_neighbour_match_produces_pairs():
    df = fit_propensity_model(make_psm_df(n=100), CONFOUNDER_COLS)
    pairs = nearest_neighbour_match(df)
    assert len(pairs) > 10, "Too few matched pairs"

def test_estimate_ate_returns_required_keys():
    pairs = pd.DataFrame({
        "treated_outcome": [0.1, 0.2, 0.15, 0.3],
        "control_outcome": [0.05, 0.1, 0.08, 0.2],
    })
    result = estimate_ate(pairs, n_bootstrap=100)
    assert {"ate", "ci_lo", "ci_hi", "p_value", "n_pairs"} == set(result.keys())

def test_estimate_ate_ci_contains_ate():
    rng = np.random.default_rng(0)
    pairs = pd.DataFrame({
        "treated_outcome": rng.uniform(0.1, 0.3, 50),
        "control_outcome": rng.uniform(0.05, 0.2, 50),
    })
    result = estimate_ate(pairs, n_bootstrap=200)
    assert result["ci_lo"] <= result["ate"] <= result["ci_hi"]

def test_estimate_ate_zero_effect():
    vals = np.linspace(0.05, 0.2, 30)
    pairs = pd.DataFrame({"treated_outcome": vals, "control_outcome": vals})
    result = estimate_ate(pairs, n_bootstrap=100)
    assert abs(result["ate"]) < 1e-9

def test_mahalanobis_match_output_columns():
    df = make_psm_df(n=100)
    pairs = mahalanobis_match(df, CONFOUNDER_COLS)
    assert {"treated_id", "control_id", "treated_outcome", "control_outcome"}.issubset(
        set(pairs.columns)
    )

def test_mahalanobis_match_no_duplicate_controls():
    df = make_psm_df(n=100)
    pairs = mahalanobis_match(df, CONFOUNDER_COLS)
    assert pairs["control_id"].nunique() == len(pairs)


def test_build_balance_table_columns():
    df = fit_propensity_model(make_psm_df(), CONFOUNDER_COLS)
    pairs = nearest_neighbour_match(df)
    table = build_balance_table(df, pairs, CONFOUNDER_COLS)
    assert set(table.columns) == {"confounder", "smd_before", "smd_after", "balanced"}


def test_build_balance_table_balanced_flag():
    df = fit_propensity_model(make_psm_df(), CONFOUNDER_COLS)
    pairs = nearest_neighbour_match(df)
    table = build_balance_table(df, pairs, CONFOUNDER_COLS)
    for _, row in table.iterrows():
        assert row["balanced"] == (row["smd_after"] < 0.1)
