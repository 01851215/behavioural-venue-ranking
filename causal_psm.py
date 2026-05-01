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
        if pooled_std == 0:
            # Fall back to overall sample std when within-group variance is zero
            pooled_std = df[col].std()
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
    Returns matched pairs DataFrame (same schema as nearest_neighbour_match minus propensity cols).
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
    """SMD before matching (full sample) and after (matched sample only)."""
    smd_before = compute_smd(df_full, confounder_cols)

    matched_ids = set(matched_pairs["treated_id"]) | set(matched_pairs["control_id"])
    df_matched = df_full[df_full["business_id"].isin(matched_ids)]
    smd_after = compute_smd(df_matched, confounder_cols)

    rows = []
    for col in confounder_cols:
        rows.append({
            "confounder": col,
            "smd_before": round(smd_before[col], 4),
            "smd_after":  round(smd_after[col], 4),
            "balanced":   smd_after[col] < 0.1,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Loading causal dataset...")
    df = pd.read_csv(DATA_DIR / "causal_venue_dataset.csv")
    print(f"  {len(df):,} venues  "
          f"(treated: {(df['treatment']==1).sum():,}, "
          f"control: {(df['treatment']==0).sum():,})")

    print("\nFitting propensity model...")
    df = fit_propensity_model(df, CONFOUNDER_COLS)
    ps = df["propensity_score"].dropna()
    print(f"  Propensity scores: mean={ps.mean():.3f}  min={ps.min():.3f}  max={ps.max():.3f}")

    print("\nSMD before matching:")
    clean = df.dropna(subset=CONFOUNDER_COLS + ["treatment"])
    smd_before = compute_smd(clean, CONFOUNDER_COLS)
    for col, smd in smd_before.items():
        flag = " ✓" if smd < 0.1 else " ✗ (imbalanced)"
        print(f"  {col:<30} SMD = {smd:.4f}{flag}")

    print("\n1:1 nearest-neighbour matching (caliper=0.2 × SD logit)...")
    matched_pairs = nearest_neighbour_match(df)
    print(f"  Matched pairs: {len(matched_pairs):,}")
    unmatched = int((df["treatment"] == 1).sum()) - len(matched_pairs)
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
    print(f"  Robust: {'YES' if robust else 'NO'}")

    matched_pairs.to_csv(DATA_DIR / "psm_matched_pairs.csv", index=False)
    balance.to_csv(DATA_DIR / "psm_balance_table.csv", index=False)
    print("\nSaved -> psm_matched_pairs.csv, psm_balance_table.csv")
