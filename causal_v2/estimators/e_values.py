"""
E-value Sensitivity Analysis (VanderWeele & Ding, 2017)

The E-value answers: "How strong would an unmeasured confounder
need to be (in terms of relative risk) to explain away the observed
treatment-outcome association?"

For a risk ratio RR (treatment effect), the E-value is:
    E = RR + sqrt(RR * (RR - 1))

If RR < 1, use 1/RR first.

Higher E-value = more robust to unmeasured confounding.

Reference: VanderWeele & Ding (2017). Sensitivity Analysis in
Observational Research. Ann. Intern. Med.

Paper contribution: closes the Rosenbaum-bounds fragility flagged
in the master's causal study.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def risk_ratio_from_ate(ate: float, baseline_rate: float) -> float:
    """
    Convert ATE (difference in probabilities) to relative risk (RR).
    RR = (baseline_rate + ate) / baseline_rate
    """
    if baseline_rate <= 0:
        return float("inf")
    treated_rate = baseline_rate + ate
    return treated_rate / baseline_rate


def e_value(rr: float) -> float:
    """
    Compute E-value from relative risk RR.
    If RR < 1, use 1/RR (null hypothesis direction).
    """
    if rr < 1:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1))


def e_value_ci(rr_ci_lo: float, rr_ci_hi: float) -> float:
    """
    E-value for the confidence interval boundary closest to null (RR=1).
    This is the more conservative E-value for the CI.
    """
    # Use the CI bound closer to 1
    if abs(rr_ci_lo - 1) < abs(rr_ci_hi - 1):
        return e_value(rr_ci_lo)
    else:
        return e_value(rr_ci_hi)


def compute_full_sensitivity(ate: float, ci_lo: float, ci_hi: float,
                              outcome_mean: float) -> dict:
    """
    Full E-value sensitivity report.

    Args:
        ate: observed ATE (absolute risk difference)
        ci_lo, ci_hi: 95% CI bounds
        outcome_mean: mean outcome in the control group (baseline rate)
    """
    rr      = risk_ratio_from_ate(ate, outcome_mean)
    rr_lo   = risk_ratio_from_ate(ci_lo, outcome_mean)
    rr_hi   = risk_ratio_from_ate(ci_hi, outcome_mean)

    ev_point = e_value(rr)
    ev_ci    = e_value_ci(rr_lo, rr_hi)

    interpretation = _interpret(ev_point)

    return {
        "ate":             round(ate, 6),
        "baseline_rate":   round(outcome_mean, 6),
        "risk_ratio":      round(rr, 4),
        "e_value_point":   round(ev_point, 4),
        "e_value_ci":      round(ev_ci, 4),
        "interpretation":  interpretation,
        "note": (
            f"An unmeasured confounder would need RR ≥ {ev_point:.2f} "
            f"with BOTH treatment AND outcome to explain away the point estimate. "
            f"The CI boundary requires RR ≥ {ev_ci:.2f}."
        ),
    }


def _interpret(ev: float) -> str:
    if ev >= 4.0:
        return "Very robust — confounder would need RR ≥ 4"
    elif ev >= 2.5:
        return "Robust — confounder would need RR ≥ 2.5"
    elif ev >= 1.5:
        return "Moderately robust — confounder would need RR ≥ 1.5"
    else:
        return "Fragile — small unmeasured confounding could explain result"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Load causal results
    try:
        pairs = pd.read_csv(
            Path(__file__).parent.parent.parent / "data/results/psm_matched_pairs.csv"
        )
        outcome_col = "treated_outcome" if "treated_outcome" in pairs.columns else pairs.columns[-1]
        control_col = "control_outcome" if "control_outcome" in pairs.columns else pairs.columns[-2]

        ate   = float((pairs[outcome_col] - pairs[control_col]).mean())
        ci_lo = ate - 1.96 * (pairs[outcome_col] - pairs[control_col]).std() / np.sqrt(len(pairs))
        ci_hi = ate + 1.96 * (pairs[outcome_col] - pairs[control_col]).std() / np.sqrt(len(pairs))
        baseline = float(pairs[control_col].mean())

        result = compute_full_sensitivity(ate, ci_lo, ci_hi, baseline)
        print("=== E-VALUE SENSITIVITY ANALYSIS ===")
        for k, v in result.items():
            print(f"  {k}: {v}")

        # Save
        out = Path(__file__).parent.parent.parent / "data/results/e_value_results.csv"
        pd.DataFrame([result]).to_csv(out, index=False)
        print(f"\nSaved → {out}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Run bvr/causal/psm.py first to generate psm_matched_pairs.csv")
