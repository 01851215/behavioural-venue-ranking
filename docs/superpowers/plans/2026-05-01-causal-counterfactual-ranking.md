# Causal / Counterfactual Ranking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a Propensity Score Matching study demonstrating that temporal consistency (routine weekday patterns + predictable peak hours) causally drives future revisit rates in coffee venues, independent of raw popularity.

**Architecture:** `causal_data_prep.py` streams the 5.3 GB Yelp review JSON to compute treatment (consistency_score), outcome (future_revisit_rate via 2020-01-01 temporal split), and joins confounders from existing venue features. `causal_psm.py` fits a logistic propensity model, runs 1:1 nearest-neighbour matching with caliper, estimates ATE with bootstrap CI, and runs a Mahalanobis robustness check. `causal_report.py` formats a thesis-ready text report.

**Tech Stack:** Python 3.9, pandas, numpy, scipy, scikit-learn, pytest

---

## Key Data Facts (pre-verified)

- Yelp review JSON: `../yelp_dataset/yelp_academic_dataset_review.json` (5.3 GB)
- Fields: `review_id`, `user_id`, `business_id`, `stars`, `date` (string "YYYY-MM-DD HH:MM:SS")
- Coffee business IDs: `business_coffee_v2.csv` (8,509 venues)
- Coffee reviews pre-2020: ~520,000 | post-2020: ~110,000
- Unique (user, venue) pairs pre-2020: ~504,000
- Treatment inputs in `coffee_venue_features_v2.csv`: `weekday_ratio`, `peak_hour_entropy`
- NaN in both: 80 venues (excluded from analysis → 8,429 eligible)
- Treatment balance after median split: ~4,214 treated / 4,215 control
- Confounders in `coffee_venue_features_v2.csv`: `total_visits`, `unique_users`, `gini_user_contribution`
- Baseline NDCG: 0.0765 (not used here — standalone analysis)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `causal_data_prep.py` | Stream reviews, compute treatment + outcome + confounders |
| Create | `causal_psm.py` | Propensity model, matching, ATE, Mahalanobis robustness |
| Create | `causal_report.py` | Format thesis-ready results report |
| Create | `tests/test_causal_data_prep.py` | Unit tests for treatment + outcome functions |
| Create | `tests/test_causal_psm.py` | Unit tests for PSM functions |
| Create | `causal_venue_dataset.csv` | One row per venue: treatment, outcome, confounders, propensity |
| Create | `psm_matched_pairs.csv` | Matched treated/control pairs |
| Create | `psm_balance_table.csv` | SMD before/after matching |
| Create | `causal_results.txt` | ATE, CI, p-value, balance summary, Mahalanobis comparison |

All paths relative to `/Users/chris/Desktop/Master Project/behavioural-venue-ranking/`.

---

## Task 1: Data Preparation — Tests + Implementation

**Files:**
- Create: `tests/test_causal_data_prep.py`
- Create: `causal_data_prep.py`
- Create: `causal_venue_dataset.csv`

### Step 1a: Write failing tests

- [ ] **Step 1: Create `tests/test_causal_data_prep.py`**

```python
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


# --- compute_consistency_score ---

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
    # Should be roughly 50/50 (allow ±5 for odd numbers)
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
    # weekday_ratio in [0,1], norm_entropy in [0,1] → score in [-1, 1]
    assert valid.min() >= -1.0 - 1e-9
    assert valid.max() <= 1.0 + 1e-9


def test_consistency_score_treatment_is_binary():
    df = make_venue_df(n=20)
    result = compute_consistency_score(df)
    valid_treatment = result["treatment"].dropna()
    assert set(valid_treatment.unique()).issubset({0.0, 1.0})


# --- compute_future_revisit_rate ---

def test_future_revisit_rate_full_retention():
    # All pre-2020 users also appear post-2020 → rate = 1.0
    pre = {"b1": {"u1", "u2", "u3"}}
    post = {"b1": {"u1", "u2", "u3", "u4"}}  # u4 is a new user (doesn't count)
    rates = compute_future_revisit_rate(pre, post, {"b1"})
    assert rates["b1"] == pytest.approx(1.0)


def test_future_revisit_rate_no_retention():
    pre = {"b1": {"u1", "u2"}}
    post = {"b1": {"u3", "u4"}}  # none of the pre-2020 users returned
    rates = compute_future_revisit_rate(pre, post, {"b1"})
    assert rates["b1"] == pytest.approx(0.0)


def test_future_revisit_rate_partial():
    pre = {"b1": {"u1", "u2", "u3", "u4"}}
    post = {"b1": {"u1", "u2"}}  # 2 of 4 returned
    rates = compute_future_revisit_rate(pre, post, {"b1"})
    assert rates["b1"] == pytest.approx(0.5)


def test_future_revisit_rate_excludes_zero_pre():
    # Venue with no pre-2020 users should not appear in output
    pre = {"b1": {"u1"}, "b2": set()}
    post = {"b1": {"u1"}, "b2": {"u5"}}
    rates = compute_future_revisit_rate(pre, post, {"b1", "b2"})
    assert "b1" in rates
    assert "b2" not in rates


def test_future_revisit_rate_missing_post():
    # Venue with pre users but zero post activity → rate = 0.0
    pre = {"b1": {"u1", "u2"}}
    post = {}
    rates = compute_future_revisit_rate(pre, post, {"b1"})
    assert rates["b1"] == pytest.approx(0.0)


# --- build_causal_dataset ---

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
    rates = {"b0": 0.1, "b1": 0.2}  # only 2 of 5 venues have outcomes
    result = build_causal_dataset(df, rates)
    assert len(result) == 2


def test_build_causal_dataset_drops_nan_treatment():
    df = make_venue_df(n=5)
    df.loc[0, "weekday_ratio"] = np.nan
    df = compute_consistency_score(df)
    rates = {f"b{i}": 0.1 for i in range(5)}
    result = build_causal_dataset(df, rates)
    assert "b0" not in result["business_id"].values
```

- [ ] **Step 2: Run tests — expect FAIL (ImportError)**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
python3 -m pytest tests/test_causal_data_prep.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'compute_consistency_score'`

### Step 1b: Implement causal_data_prep.py

- [ ] **Step 3: Create `causal_data_prep.py`**

```python
"""
Causal data preparation for Direction 5: PSM study.

Streams Yelp review JSON to compute:
  - Treatment: consistency_score = weekday_ratio - minmax_norm(peak_hour_entropy)
  - Outcome:   future_revisit_rate (fraction of pre-2020 users who returned post-2020)
  - Confounders: total_visits, unique_users, gini_user_contribution (from venue features)

Writes: causal_venue_dataset.csv
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

DATA_DIR     = Path(__file__).parent
REVIEW_PATH  = DATA_DIR / "../yelp_dataset/yelp_academic_dataset_review.json"
COFFEE_PATH  = DATA_DIR / "business_coffee_v2.csv"
FEATURES_PATH = DATA_DIR / "coffee_venue_features_v2.csv"
SPLIT_DATE   = "2020-01-01"

CONFOUNDER_COLS = ["total_visits", "unique_users", "gini_user_contribution"]


# ============================================================================
# Pure functions (testable without I/O)
# ============================================================================

def compute_consistency_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add consistency_score and binary treatment column.
    consistency_score = weekday_ratio - minmax_norm(peak_hour_entropy)
    Treatment = 1 if score > median, 0 if <= median, NaN if inputs missing.
    """
    df = df.copy()
    valid = df["weekday_ratio"].notna() & df["peak_hour_entropy"].notna()

    ent = df.loc[valid, "peak_hour_entropy"]
    ent_norm = (ent - ent.min()) / (ent.max() - ent.min())

    df["peak_hour_entropy_norm"] = np.nan
    df.loc[valid, "peak_hour_entropy_norm"] = ent_norm

    df["consistency_score"] = np.nan
    df.loc[valid, "consistency_score"] = (
        df.loc[valid, "weekday_ratio"] - df.loc[valid, "peak_hour_entropy_norm"]
    )

    median = df.loc[valid, "consistency_score"].median()
    df["treatment"] = np.nan
    df.loc[valid & (df["consistency_score"] > median), "treatment"]  = 1.0
    df.loc[valid & (df["consistency_score"] <= median), "treatment"] = 0.0

    return df


def compute_future_revisit_rate(
    pre_users: dict,
    post_users: dict,
    coffee_ids: set,
) -> dict:
    """
    For each venue, compute fraction of pre-2020 users who returned post-2020.
    Venues with zero pre-2020 users are excluded (no valid counterfactual).
    """
    rates = {}
    for bid in coffee_ids:
        pre = pre_users.get(bid, set())
        if not pre:
            continue
        post = post_users.get(bid, set())
        returning = pre & post
        rates[bid] = len(returning) / len(pre)
    return rates


def build_causal_dataset(df: pd.DataFrame, future_revisit_rates: dict) -> pd.DataFrame:
    """
    Join treatment, confounders, and outcome into one analysis-ready DataFrame.
    Drops venues missing outcome or treatment.
    """
    df = df.copy()
    df["future_revisit_rate"] = df["business_id"].map(future_revisit_rates)

    keep_cols = (
        ["business_id", "consistency_score", "treatment", "future_revisit_rate"]
        + CONFOUNDER_COLS
    )
    result = df[keep_cols].copy()
    result = result.dropna(subset=["future_revisit_rate", "treatment"])
    return result.reset_index(drop=True)


# ============================================================================
# I/O helpers
# ============================================================================

def load_coffee_reviews(review_path: Path, coffee_ids: set, split_date: str = SPLIT_DATE):
    """
    Stream Yelp review JSON. Returns (pre_users, post_users) dicts:
      pre_users[venue_id]  = set of user_ids who visited before split_date
      post_users[venue_id] = set of user_ids who visited on/after split_date
    """
    print(f"Streaming {review_path} (5.3 GB — takes ~2 min)...")
    pre_users  = defaultdict(set)
    post_users = defaultdict(set)
    n_total = n_coffee = 0

    with open(review_path) as f:
        for line in f:
            row = json.loads(line)
            n_total += 1
            bid = row["business_id"]
            if bid not in coffee_ids:
                continue
            n_coffee += 1
            uid  = row["user_id"]
            date = row["date"]
            if date < split_date:
                pre_users[bid].add(uid)
            else:
                post_users[bid].add(uid)

            if n_total % 1_000_000 == 0:
                print(f"  {n_total/1e6:.0f}M reviews processed, {n_coffee:,} coffee...")

    print(f"  Done. {n_total:,} reviews scanned, {n_coffee:,} coffee.")
    print(f"  Venues with pre-2020 users: {len(pre_users):,}")
    print(f"  Venues with post-2020 users: {len(post_users):,}")
    return dict(pre_users), dict(post_users)


if __name__ == "__main__":
    print("Loading coffee business IDs...")
    coffee_df = pd.read_csv(COFFEE_PATH)
    coffee_ids = set(coffee_df["business_id"])
    print(f"  {len(coffee_ids):,} coffee venues")

    print("Loading venue features...")
    features = pd.read_csv(FEATURES_PATH)

    print("Computing treatment (consistency score)...")
    features = compute_consistency_score(features)
    n_treated = (features["treatment"] == 1).sum()
    n_control = (features["treatment"] == 0).sum()
    n_excluded = features["treatment"].isna().sum()
    print(f"  Treated: {n_treated:,}  Control: {n_control:,}  Excluded (no check-in): {n_excluded:,}")

    pre_users, post_users = load_coffee_reviews(REVIEW_PATH, coffee_ids)

    print("Computing future revisit rates...")
    rates = compute_future_revisit_rate(pre_users, post_users, coffee_ids)
    print(f"  {len(rates):,} venues with valid outcome")
    rate_series = pd.Series(list(rates.values()))
    print(f"  Rate: mean={rate_series.mean():.4f}  median={rate_series.median():.4f}  "
          f"max={rate_series.max():.4f}")

    print("Building causal dataset...")
    dataset = build_causal_dataset(features, rates)
    print(f"  Final dataset: {len(dataset):,} venues  "
          f"(treated: {(dataset['treatment']==1).sum():,}, "
          f"control: {(dataset['treatment']==0).sum():,})")

    out = DATA_DIR / "causal_venue_dataset.csv"
    dataset.to_csv(out, index=False)
    print(f"Saved -> causal_venue_dataset.csv")
```

- [ ] **Step 4: Run tests — all must PASS**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
python3 -m pytest tests/test_causal_data_prep.py -v
```

Expected: 14 tests pass.

- [ ] **Step 5: Run on real data (~2 min for JSON streaming)**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
python3 causal_data_prep.py
```

Expected output:
```
Loading coffee business IDs...
  8,509 coffee venues
Loading venue features...
Computing treatment (consistency score)...
  Treated: ~4,214  Control: ~4,215  Excluded (no check-in): 80
Streaming .../yelp_academic_dataset_review.json (5.3 GB — takes ~2 min)...
  1M reviews processed, ...
  Done. ~6,990,280 reviews scanned, ~630,349 coffee.
Computing future revisit rates...
  ~8,400 venues with valid outcome
Building causal dataset...
  Final dataset: ~8,000 venues  (treated: ~4,000, control: ~4,000)
Saved -> causal_venue_dataset.csv
```

- [ ] **Step 6: Verify output**

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('causal_venue_dataset.csv')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print(df[['treatment','future_revisit_rate']].describe().round(4))
print('Treatment counts:', df['treatment'].value_counts().to_dict())
"
```

Expected: ~8,000 rows, columns include treatment + future_revisit_rate + 3 confounders, treatment is 0/1 balanced.

- [ ] **Step 7: Commit**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
git add causal_data_prep.py tests/test_causal_data_prep.py causal_venue_dataset.csv
git commit -m "feat: causal data prep — treatment, outcome, confounders"
```

---

## Task 2: PSM Analysis — Tests + Implementation

**Files:**
- Create: `tests/test_causal_psm.py`
- Create: `causal_psm.py`
- Create: `psm_matched_pairs.csv`
- Create: `psm_balance_table.csv`

### Step 2a: Write failing tests

- [ ] **Step 1: Create `tests/test_causal_psm.py`**

```python
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_psm import (
    fit_propensity_model,
    nearest_neighbour_match,
    compute_smd,
    estimate_ate,
    mahalanobis_match,
)

CONFOUNDER_COLS = ["total_visits", "unique_users", "gini_user_contribution"]


def make_psm_df(n=200, seed=42):
    """Synthetic causal dataset with known structure."""
    rng = np.random.default_rng(seed)
    # Treatment correlated with confounders
    total_visits = rng.exponential(200, n)
    unique_users = total_visits * rng.uniform(0.1, 0.5, n)
    gini = rng.uniform(0, 0.4, n)
    # Treatment more likely for high-visit venues
    logit = (total_visits / 500 - 0.5) * 2
    prob = 1 / (1 + np.exp(-logit))
    treatment = (rng.uniform(0, 1, n) < prob).astype(float)
    # Outcome: treated venues have higher revisit rate + noise
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


# --- fit_propensity_model ---

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


# --- compute_smd ---

def test_compute_smd_perfect_balance_is_zero():
    rng = np.random.default_rng(0)
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


# --- nearest_neighbour_match ---

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


# --- estimate_ate ---

def test_estimate_ate_returns_required_keys():
    pairs = pd.DataFrame({
        "treated_outcome": [0.1, 0.2, 0.15, 0.3],
        "control_outcome": [0.05, 0.1, 0.08, 0.2],
    })
    result = estimate_ate(pairs, n_bootstrap=100)
    assert {"ate", "ci_lo", "ci_hi", "p_value", "n_pairs"} == set(result.keys())


def test_estimate_ate_ci_contains_ate():
    pairs = pd.DataFrame({
        "treated_outcome": np.random.default_rng(0).uniform(0.1, 0.3, 50),
        "control_outcome": np.random.default_rng(1).uniform(0.05, 0.2, 50),
    })
    result = estimate_ate(pairs, n_bootstrap=200)
    assert result["ci_lo"] <= result["ate"] <= result["ci_hi"]


def test_estimate_ate_zero_effect():
    # Identical outcomes → ATE ≈ 0
    vals = np.linspace(0.05, 0.2, 30)
    pairs = pd.DataFrame({"treated_outcome": vals, "control_outcome": vals})
    result = estimate_ate(pairs, n_bootstrap=100)
    assert abs(result["ate"]) < 1e-9


# --- mahalanobis_match ---

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
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
python3 -m pytest tests/test_causal_psm.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'fit_propensity_model'`

### Step 2b: Implement causal_psm.py

- [ ] **Step 3: Create `causal_psm.py`**

```python
"""
Propensity Score Matching for Direction 5 causal analysis.

Reads:   causal_venue_dataset.csv
Writes:  psm_matched_pairs.csv, psm_balance_table.csv
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent
CONFOUNDER_COLS = ["total_visits", "unique_users", "gini_user_contribution"]
N_BOOTSTRAP = 1000
CALIPER_MULTIPLIER = 0.2


# ============================================================================
# Pure functions
# ============================================================================

def fit_propensity_model(df: pd.DataFrame, confounder_cols: list) -> pd.DataFrame:
    """
    Fit logistic regression P(treatment=1 | confounders).
    Returns df with added columns: propensity_score, logit_propensity.
    """
    df = df.copy()
    clean = df.dropna(subset=confounder_cols + ["treatment"])

    scaler = StandardScaler()
    X = scaler.fit_transform(clean[confounder_cols].values)
    y = clean["treatment"].values.astype(int)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    ps = model.predict_proba(X)[:, 1]
    eps = 1e-10
    logit_ps = np.log(ps / (1 - ps + eps))

    df["propensity_score"] = np.nan
    df["logit_propensity"] = np.nan
    df.loc[clean.index, "propensity_score"] = ps
    df.loc[clean.index, "logit_propensity"] = logit_ps

    return df


def compute_smd(df: pd.DataFrame, confounder_cols: list,
                treatment_col: str = "treatment") -> dict:
    """
    Standardised Mean Difference per confounder.
    SMD = |mean_treated - mean_control| / sqrt((var_treated + var_control) / 2)
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    smds = {}
    for col in confounder_cols:
        t_mean, c_mean = treated[col].mean(), control[col].mean()
        t_var,  c_var  = treated[col].var(),  control[col].var()
        pooled_std = np.sqrt((t_var + c_var) / 2)
        smds[col] = abs(t_mean - c_mean) / pooled_std if pooled_std > 0 else 0.0

    return smds


def nearest_neighbour_match(df: pd.DataFrame,
                             caliper_multiplier: float = CALIPER_MULTIPLIER) -> pd.DataFrame:
    """
    1:1 nearest-neighbour matching without replacement on logit propensity score.
    Caliper = caliper_multiplier × SD(logit_propensity).
    Returns DataFrame of matched pairs.
    """
    df = df.dropna(subset=["propensity_score", "logit_propensity",
                            "treatment", "future_revisit_rate"])

    caliper = caliper_multiplier * df["logit_propensity"].std()
    treated = df[df["treatment"] == 1].copy()
    control = df[df["treatment"] == 0].copy()

    matched_pairs = []
    used_control_idx = set()

    for _, t_row in treated.iterrows():
        candidates = control[~control.index.isin(used_control_idx)]
        if candidates.empty:
            break
        distances = (candidates["logit_propensity"] - t_row["logit_propensity"]).abs()
        min_dist = distances.min()
        if min_dist > caliper:
            continue
        best_idx = distances.idxmin()
        used_control_idx.add(best_idx)
        c_row = control.loc[best_idx]
        matched_pairs.append({
            "treated_id":          t_row["business_id"],
            "control_id":          c_row["business_id"],
            "treated_propensity":  t_row["propensity_score"],
            "control_propensity":  c_row["propensity_score"],
            "treated_outcome":     t_row["future_revisit_rate"],
            "control_outcome":     c_row["future_revisit_rate"],
        })

    return pd.DataFrame(matched_pairs)


def estimate_ate(matched_pairs: pd.DataFrame,
                 n_bootstrap: int = N_BOOTSTRAP,
                 random_state: int = 42) -> dict:
    """
    ATE = mean(treated_outcome - control_outcome) in matched sample.
    Bootstrap 95% CI (resample pairs). Paired t-test p-value.
    """
    diffs = matched_pairs["treated_outcome"] - matched_pairs["control_outcome"]
    ate = float(diffs.mean())

    rng = np.random.default_rng(random_state)
    boot_ates = [
        rng.choice(diffs.values, size=len(diffs), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    ci_lo = float(np.percentile(boot_ates, 2.5))
    ci_hi = float(np.percentile(boot_ates, 97.5))

    _, p_value = ttest_rel(
        matched_pairs["treated_outcome"].values,
        matched_pairs["control_outcome"].values,
    )

    return {
        "ate":     ate,
        "ci_lo":   ci_lo,
        "ci_hi":   ci_hi,
        "p_value": float(p_value),
        "n_pairs": len(matched_pairs),
    }


def mahalanobis_match(df: pd.DataFrame, confounder_cols: list) -> pd.DataFrame:
    """
    1:1 matching on Mahalanobis distance (standardised Euclidean approximation).
    Returns matched pairs DataFrame (same schema as nearest_neighbour_match).
    """
    clean = df.dropna(subset=confounder_cols + ["treatment", "future_revisit_rate"])
    treated = clean[clean["treatment"] == 1].reset_index(drop=True)
    control = clean[clean["treatment"] == 0].reset_index(drop=True)

    scaler = StandardScaler()
    all_X = scaler.fit_transform(clean[confounder_cols].values)
    t_mask = clean["treatment"].values == 1
    t_X = all_X[t_mask]
    c_X = all_X[~t_mask]

    distances = cdist(t_X, c_X, metric="euclidean")

    matched_pairs = []
    used_control = set()

    for i in range(len(treated)):
        dists = distances[i].copy()
        for j in used_control:
            dists[j] = np.inf
        best_j = int(np.argmin(dists))
        used_control.add(best_j)
        matched_pairs.append({
            "treated_id":      treated.loc[i, "business_id"],
            "control_id":      control.loc[best_j, "business_id"],
            "treated_outcome": treated.loc[i, "future_revisit_rate"],
            "control_outcome": control.loc[best_j, "future_revisit_rate"],
        })

    return pd.DataFrame(matched_pairs)


def build_balance_table(df_full: pd.DataFrame,
                        matched_pairs: pd.DataFrame,
                        confounder_cols: list) -> pd.DataFrame:
    """
    SMD before matching (full sample) and after (matched sample only).
    """
    smd_before = compute_smd(df_full, confounder_cols)

    # Reconstruct matched dataframe
    matched_ids = set(matched_pairs["treated_id"]) | set(matched_pairs["control_id"])
    df_matched = df_full[df_full["business_id"].isin(matched_ids)]
    smd_after = compute_smd(df_matched, confounder_cols)

    rows = []
    for col in confounder_cols:
        rows.append({
            "confounder":  col,
            "smd_before":  round(smd_before[col], 4),
            "smd_after":   round(smd_after[col], 4),
            "balanced":    smd_after[col] < 0.1,
        })

    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Loading causal dataset...")
    df = pd.read_csv(DATA_DIR / "causal_venue_dataset.csv")
    print(f"  {len(df):,} venues  "
          f"(treated: {(df['treatment']==1).sum():,}, "
          f"control: {(df['treatment']==0).sum():,})")

    print("\nFitting propensity model...")
    df = fit_propensity_model(df, CONFOUNDER_COLS)
    ps = df["propensity_score"].dropna()
    print(f"  Propensity scores: mean={ps.mean():.3f}  "
          f"min={ps.min():.3f}  max={ps.max():.3f}")

    print("\nSMD before matching:")
    smd_before = compute_smd(df.dropna(subset=CONFOUNDER_COLS + ["treatment"]),
                             CONFOUNDER_COLS)
    for col, smd in smd_before.items():
        flag = " ✓" if smd < 0.1 else " ✗ (imbalanced)"
        print(f"  {col:<30} SMD = {smd:.4f}{flag}")

    print("\n1:1 nearest-neighbour matching (caliper=0.2 × SD logit)...")
    matched_pairs = nearest_neighbour_match(df)
    print(f"  Matched pairs: {len(matched_pairs):,}")
    unmatched = (df["treatment"] == 1).sum() - len(matched_pairs)
    print(f"  Treated venues outside caliper (excluded): {unmatched:,}")

    print("\nSMD after matching:")
    balance = build_balance_table(df, matched_pairs, CONFOUNDER_COLS)
    for _, row in balance.iterrows():
        flag = " ✓" if row["balanced"] else " ✗ (still imbalanced)"
        print(f"  {row['confounder']:<30} "
              f"before={row['smd_before']:.4f}  after={row['smd_after']:.4f}{flag}")

    print("\nEstimating ATE (PSM, 1,000 bootstrap resamples)...")
    ate_result = estimate_ate(matched_pairs)
    sig = "significant" if ate_result["p_value"] < 0.05 else "not significant"
    print(f"  ATE  = {ate_result['ate']:+.6f}")
    print(f"  95% CI [{ate_result['ci_lo']:+.6f}, {ate_result['ci_hi']:+.6f}]")
    print(f"  p    = {ate_result['p_value']:.4f}  ({sig})")
    print(f"  n    = {ate_result['n_pairs']:,} matched pairs")

    print("\nMahalanobis robustness check...")
    mah_pairs = mahalanobis_match(df, CONFOUNDER_COLS)
    mah_ate = estimate_ate(mah_pairs)
    print(f"  Mahalanobis ATE = {mah_ate['ate']:+.6f}  "
          f"95% CI [{mah_ate['ci_lo']:+.6f}, {mah_ate['ci_hi']:+.6f}]  "
          f"p={mah_ate['p_value']:.4f}")
    robust = (mah_ate["ci_lo"] <= ate_result["ate"] <= mah_ate["ci_hi"] or
              ate_result["ci_lo"] <= mah_ate["ate"] <= ate_result["ci_hi"])
    print(f"  Robust: {'YES' if robust else 'NO — ATEs fall outside each other CIs'}")

    matched_pairs.to_csv(DATA_DIR / "psm_matched_pairs.csv", index=False)
    balance.to_csv(DATA_DIR / "psm_balance_table.csv", index=False)

    print("\nSaved -> psm_matched_pairs.csv, psm_balance_table.csv")
```

- [ ] **Step 4: Run tests — all must PASS**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
python3 -m pytest tests/test_causal_psm.py -v
```

Expected: 17 tests pass.

- [ ] **Step 5: Run PSM on real data**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
python3 causal_psm.py
```

Expected output (approximate values):
```
Loading causal dataset...
  ~8,000 venues  (treated: ~4,000, control: ~4,000)
Fitting propensity model...
  Propensity scores: mean=0.500  min=...  max=...
SMD before matching:
  total_visits                   SMD = X.XXXX ...
  unique_users                   SMD = X.XXXX ...
  gini_user_contribution         SMD = X.XXXX ...
1:1 nearest-neighbour matching (caliper=0.2 × SD logit)...
  Matched pairs: X,XXX
  Treated venues outside caliper: XXX
SMD after matching:
  total_visits                   before=X.XXXX  after=X.XXXX ✓/✗
  ...
Estimating ATE (PSM, 1,000 bootstrap resamples)...
  ATE  = +/-X.XXXXXX
  95% CI [...]
  p    = X.XXXX
Mahalanobis robustness check...
  Mahalanobis ATE = ...
  Robust: YES/NO
Saved -> psm_matched_pairs.csv, psm_balance_table.csv
```

- [ ] **Step 6: Verify outputs**

```bash
python3 -c "
import pandas as pd
pairs = pd.read_csv('psm_matched_pairs.csv')
balance = pd.read_csv('psm_balance_table.csv')
print('Matched pairs shape:', pairs.shape)
print(pairs.head(3))
print()
print('Balance table:')
print(balance.to_string(index=False))
assert pairs['control_id'].nunique() == len(pairs), 'Duplicate controls!'
print('No duplicate controls: OK')
"
```

- [ ] **Step 7: Commit**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
git add causal_psm.py tests/test_causal_psm.py psm_matched_pairs.csv psm_balance_table.csv
git commit -m "feat: PSM analysis — propensity matching, ATE, Mahalanobis robustness"
```

---

## Task 3: Report Generation

**Files:**
- Create: `causal_report.py`
- Create: `causal_results.txt`

- [ ] **Step 1: Create `causal_report.py`**

```python
"""
Generate thesis-ready causal analysis report.

Reads:  causal_venue_dataset.csv, psm_matched_pairs.csv, psm_balance_table.csv
Writes: causal_results.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from causal_psm import estimate_ate, mahalanobis_match, CONFOUNDER_COLS

DATA_DIR = Path(__file__).parent


def generate_report() -> str:
    dataset  = pd.read_csv(DATA_DIR / "causal_venue_dataset.csv")
    pairs    = pd.read_csv(DATA_DIR / "psm_matched_pairs.csv")
    balance  = pd.read_csv(DATA_DIR / "psm_balance_table.csv")

    ate_result = estimate_ate(pairs)
    mah_pairs  = mahalanobis_match(dataset, CONFOUNDER_COLS)
    mah_result = estimate_ate(mah_pairs)

    n_total   = len(dataset)
    n_treated = int((dataset["treatment"] == 1).sum())
    n_control = int((dataset["treatment"] == 0).sum())
    n_matched = len(pairs)
    n_excluded = n_treated - n_matched

    sig_psm = ate_result["p_value"] < 0.05
    sig_mah = mah_result["p_value"] < 0.05
    robust = (
        mah_result["ci_lo"] <= ate_result["ate"] <= mah_result["ci_hi"]
        or ate_result["ci_lo"] <= mah_result["ate"] <= ate_result["ci_hi"]
    )
    all_balanced = balance["balanced"].all()

    lines = []
    lines.append("=" * 70)
    lines.append("CAUSAL ANALYSIS REPORT — DIRECTION 5")
    lines.append("Propensity Score Matching: Temporal Consistency → Future Revisit Rate")
    lines.append("=" * 70)

    lines.append("\n--- STUDY DESIGN ---")
    lines.append("Treatment: consistency_score = weekday_ratio - minmax_norm(peak_hour_entropy)")
    lines.append("           Binary split at median (top half = treated)")
    lines.append("Outcome:   future_revisit_rate (fraction of pre-2020 users returning post-2020)")
    lines.append("Method:    1:1 nearest-neighbour PSM, caliper = 0.2 × SD(logit propensity)")
    lines.append(f"Confounders: {', '.join(CONFOUNDER_COLS)}")
    lines.append(f"Temporal split: 2020-01-01")

    lines.append("\n--- SAMPLE ---")
    lines.append(f"  Total eligible venues:    {n_total:,}")
    lines.append(f"  Treated (high consistency): {n_treated:,}")
    lines.append(f"  Control (low consistency):  {n_control:,}")
    lines.append(f"  Matched pairs:              {n_matched:,}")
    lines.append(f"  Excluded (outside caliper): {n_excluded:,}")

    lines.append("\n--- COVARIATE BALANCE (Austin 2011: SMD < 0.1 = well-balanced) ---")
    lines.append(f"  {'Confounder':<30} {'SMD Before':>12} {'SMD After':>12} {'Balanced':>10}")
    for _, row in balance.iterrows():
        flag = "✓" if row["balanced"] else "✗"
        lines.append(f"  {row['confounder']:<30} {row['smd_before']:>12.4f} "
                     f"{row['smd_after']:>12.4f} {flag:>10}")
    overall_balance = "All confounders well-balanced (SMD < 0.1)" if all_balanced \
        else "WARNING: some confounders remain imbalanced after matching"
    lines.append(f"  {overall_balance}")

    lines.append("\n--- PSM RESULTS ---")
    lines.append(f"  ATE  = {ate_result['ate']:+.6f}")
    lines.append(f"  95% CI [{ate_result['ci_lo']:+.6f}, {ate_result['ci_hi']:+.6f}]  "
                 f"(n_bootstrap=1000)")
    lines.append(f"  p    = {ate_result['p_value']:.4f}  "
                 f"({'statistically significant' if sig_psm else 'not significant'} at α=0.05)")
    lines.append(f"  n    = {ate_result['n_pairs']:,} matched pairs")

    direction = "positive" if ate_result["ate"] > 0 else "negative"
    if sig_psm:
        lines.append(f"\n  INTERPRETATION: Temporally consistent venues show a {direction} "
                     f"causal effect on future revisit rates (ATE={ate_result['ate']:+.4f}, "
                     f"p={ate_result['p_value']:.4f}).")
    else:
        lines.append(f"\n  INTERPRETATION: No statistically significant causal effect detected "
                     f"(ATE={ate_result['ate']:+.4f}, p={ate_result['p_value']:.4f}). "
                     f"The association between temporal consistency and revisit rates "
                     f"may be explained by confounders.")

    lines.append("\n--- ROBUSTNESS CHECK (Mahalanobis distance matching) ---")
    lines.append(f"  ATE  = {mah_result['ate']:+.6f}")
    lines.append(f"  95% CI [{mah_result['ci_lo']:+.6f}, {mah_result['ci_hi']:+.6f}]")
    lines.append(f"  p    = {mah_result['p_value']:.4f}  "
                 f"({'significant' if sig_mah else 'not significant'})")
    lines.append(f"  Robust to matching method: {'YES' if robust else 'NO'}")
    if robust:
        lines.append("  (PSM and Mahalanobis ATEs overlap within each other's 95% CIs)")
    else:
        lines.append("  (ATEs diverge — interpret PSM result with caution)")

    lines.append("\n--- THESIS CITATION GUIDE ---")
    lines.append("  PSM method: Rosenbaum & Rubin (1983); Austin (2011) for SMD balance check")
    lines.append("  Caliper: 0.2 × SD of logit propensity (Austin 2011 recommendation)")
    lines.append("  Bootstrap CI: 1,000 resamples of matched pairs")
    lines.append("  Mahalanobis: sensitivity analysis for matching-method dependence")
    lines.append("=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_report()
    print(report)
    out = DATA_DIR / "causal_results.txt"
    with open(out, "w") as f:
        f.write(report)
    print(f"\nSaved -> causal_results.txt")
```

- [ ] **Step 2: Run the report**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
python3 causal_report.py
```

Expected: full report prints to terminal and saves to `causal_results.txt`. Check:
- ATE value printed
- Balance table shows SMD before/after
- Robustness check reports YES or NO
- Interpretation paragraph present

- [ ] **Step 3: Commit**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
git add causal_report.py causal_results.txt
git commit -m "feat: causal report — thesis-ready ATE, balance table, robustness check"
```

---

## Task 4: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
python3 -m pytest tests/ -v
```

Expected: all 60 tests pass (29 previous + 14 causal_data_prep + 17 causal_psm).

- [ ] **Step 2: Confirm all output files exist**

```bash
ls -lh causal_venue_dataset.csv psm_matched_pairs.csv \
        psm_balance_table.csv causal_results.txt
```

- [ ] **Step 3: Print final report**

```bash
cat causal_results.txt
```

- [ ] **Step 4: Final commit**

```bash
cd "/Users/chris/Desktop/Master Project/behavioural-venue-ranking"
git add .
git commit -m "feat: complete causal PSM analysis (Direction 5)"
```
