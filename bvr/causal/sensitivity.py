"""
Rosenbaum Bounds — Sensitivity Analysis for Causal PSM (V2)

Tests how much unobserved confounding (Γ) would be needed to
nullify the ATE = +0.0027 finding.

Γ = 1  → no unobserved confounding (baseline)
Γ = 1.5 → one subject could be 1.5× more likely to be treated
Γ = 2   → 2× difference — considered robust in social science
Γ > 2   → very strong claim

Method: Wilcoxon signed-rank test bounds under Γ assumption.
At each Γ, compute the most conservative (upper bound) p-value.
The critical Γ is where the upper-bound p-value crosses 0.05.

Reference: Rosenbaum (2002) Observational Studies, Springer.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

DATA_DIR = Path(__file__).parent


def rosenbaum_bounds(diff: np.ndarray, gammas: list) -> pd.DataFrame:
    """
    Compute Rosenbaum sensitivity bounds for Wilcoxon signed-rank test.

    For each Γ, compute the upper bound on the p-value (worst-case scenario
    where unobserved confounding tilts treatment assignment against us).

    Args:
        diff: per-pair outcome differences (treated - control)
        gammas: list of Γ values to test

    Returns:
        DataFrame with columns [gamma, p_upper, significant_at_05]
    """
    diff = diff[diff != 0]           # remove ties (standard Wilcoxon)
    n    = len(diff)
    abs_diff = np.abs(diff)

    # Rank absolute differences
    ranks = np.argsort(np.argsort(abs_diff)) + 1    # 1-based ranks

    rows = []
    for gamma in gammas:
        # Under Γ, the maximum probability that any treated unit's sign is +1
        # (i.e., treated > control) is gamma/(1+gamma)
        p_plus = gamma / (1 + gamma)

        # Upper bound on E[W+] and Var[W+] under worst-case confounding
        # W+ = sum of positive-signed ranks
        # E_max = sum of ranks * p_plus
        # Var_max = sum of ranks^2 * p_plus * (1 - p_plus)
        e_max   = np.sum(ranks * p_plus)
        var_max = np.sum(ranks**2 * p_plus * (1 - p_plus))

        # Observed W+ (one-sided: treated > control)
        w_plus  = np.sum(ranks[diff > 0])

        # Upper bound p-value: how extreme is W+ under worst-case?
        # Using normal approximation to the signed-rank distribution
        z_upper = (w_plus - e_max) / np.sqrt(var_max + 1e-10)
        p_upper = 1 - norm.cdf(z_upper)    # one-sided upper bound

        rows.append({
            "gamma":               gamma,
            "p_upper":             round(float(p_upper), 4),
            "significant_at_0.05": p_upper < 0.05,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    pairs = pd.read_csv(DATA_DIR / "psm_matched_pairs.csv")

    # Compute per-pair outcome differences
    if "treated_outcome" in pairs.columns and "control_outcome" in pairs.columns:
        diff = pairs["treated_outcome"].values - pairs["control_outcome"].values
    else:
        # Fallback: infer from dataset
        print("Column names:", pairs.columns.tolist())
        raise ValueError("Expected 'treated_outcome' and 'control_outcome' columns")

    print(f"Matched pairs: {len(diff):,}")
    print(f"Mean difference: {diff.mean():+.6f}")
    print(f"Positive pairs: {(diff > 0).sum():,} / {len(diff):,}")

    gammas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
              2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0]

    results = rosenbaum_bounds(diff, gammas)

    print("\n=== ROSENBAUM SENSITIVITY BOUNDS ===")
    print("Γ = how much more likely treated unit could be treated due to")
    print("    unobserved confounding. Γ=1 = no confounding.")
    print(f"\n{'Γ':>6} {'p (upper bound)':>18} {'Sig at 0.05':>14}")
    print("-" * 44)
    for _, row in results.iterrows():
        sig = "YES ✓" if row["significant_at_0.05"] else "NO  ✗"
        print(f"{row['gamma']:>6.2f} {row['p_upper']:>18.4f} {sig:>14}")

    # Find critical Gamma (where significance is lost)
    critical = results[~results["significant_at_0.05"]]["gamma"].min()
    robust   = results[results["gamma"] == 1.0]["significant_at_0.05"].iloc[0]

    print(f"\n{'='*44}")
    if pd.isna(critical):
        print("Result is robust at all tested Γ values (up to Γ=5.0)")
        critical_gamma = ">5.0"
    else:
        print(f"Critical Γ = {critical:.2f}")
        print(f"Interpretation: an unobserved confounder would need to make")
        print(f"treated venues {critical:.1f}× more likely to receive treatment")
        print(f"to nullify the result.")
        if critical >= 2.0:
            print("★ Robust (Γ ≥ 2.0 is the standard threshold in social science)")
        elif critical >= 1.5:
            print("~ Moderate robustness (Γ ≥ 1.5)")
        else:
            print("⚠ Fragile — small confounding could change the result")
        critical_gamma = str(critical)

    results.to_csv(DATA_DIR / "rosenbaum_bounds.csv", index=False)
    print(f"\nSaved → rosenbaum_bounds.csv")
