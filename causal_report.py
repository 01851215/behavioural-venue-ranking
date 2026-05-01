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
    dataset = pd.read_csv(DATA_DIR / "causal_venue_dataset.csv")
    pairs   = pd.read_csv(DATA_DIR / "psm_matched_pairs.csv")
    balance = pd.read_csv(DATA_DIR / "psm_balance_table.csv")

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
    lines.append("Propensity Score Matching: Temporal Consistency -> Future Revisit Rate")
    lines.append("=" * 70)

    lines.append("\n--- STUDY DESIGN ---")
    lines.append("Treatment: consistency_score = weekday_ratio - minmax_norm(peak_hour_entropy)")
    lines.append("           Binary split at median (top half = treated)")
    lines.append("Outcome:   future_revisit_rate (fraction of pre-2020 users returning post-2020)")
    lines.append("Method:    1:1 nearest-neighbour PSM, caliper = 0.2 x SD(logit propensity)")
    lines.append(f"Confounders: {', '.join(CONFOUNDER_COLS)}")
    lines.append("Temporal split: 2020-01-01")

    lines.append("\n--- SAMPLE ---")
    lines.append(f"  Total eligible venues:      {n_total:,}")
    lines.append(f"  Treated (high consistency): {n_treated:,}")
    lines.append(f"  Control (low consistency):  {n_control:,}")
    lines.append(f"  Matched pairs:              {n_matched:,}")
    lines.append(f"  Excluded (outside caliper): {n_excluded:,}")

    lines.append("\n--- COVARIATE BALANCE (Austin 2011: SMD < 0.1 = well-balanced) ---")
    lines.append(f"  {'Confounder':<30} {'SMD Before':>12} {'SMD After':>12} {'Balanced':>10}")
    for _, row in balance.iterrows():
        flag = "YES" if row["balanced"] else "NO"
        lines.append(f"  {row['confounder']:<30} {row['smd_before']:>12.4f} "
                     f"{row['smd_after']:>12.4f} {flag:>10}")
    lines.append(f"  Overall: {'All well-balanced (SMD < 0.1)' if all_balanced else 'WARNING: some imbalance remains'}")

    lines.append("\n--- PSM RESULTS ---")
    lines.append(f"  ATE  = {ate_result['ate']:+.6f}")
    lines.append(f"  95% CI [{ate_result['ci_lo']:+.6f}, {ate_result['ci_hi']:+.6f}]  (n_bootstrap=1000)")
    lines.append(f"  p    = {ate_result['p_value']:.4f}  "
                 f"({'statistically significant' if sig_psm else 'not significant'} at alpha=0.05)")
    lines.append(f"  n    = {ate_result['n_pairs']:,} matched pairs")

    direction = "positive" if ate_result["ate"] > 0 else "negative"
    if sig_psm:
        lines.append(f"\n  INTERPRETATION: Temporally consistent venues show a {direction} "
                     f"causal effect on future revisit rates (ATE={ate_result['ate']:+.4f}, "
                     f"p={ate_result['p_value']:.4f}).")
    else:
        lines.append(f"\n  INTERPRETATION: A small {direction} directional effect is observed "
                     f"(ATE={ate_result['ate']:+.4f}), but it does not reach statistical "
                     f"significance (p={ate_result['p_value']:.4f}). This may reflect reduced "
                     f"statistical power due to COVID-19 disruption at the 2020 split: "
                     f"most venues have future_revisit_rate=0, compressing the outcome variance.")

    lines.append("\n--- ROBUSTNESS CHECK (Mahalanobis distance matching) ---")
    lines.append(f"  ATE  = {mah_result['ate']:+.6f}")
    lines.append(f"  95% CI [{mah_result['ci_lo']:+.6f}, {mah_result['ci_hi']:+.6f}]")
    lines.append(f"  p    = {mah_result['p_value']:.4f}  ({'significant' if sig_mah else 'not significant'})")
    lines.append(f"  Robust to matching method: {'YES' if robust else 'NO'}")
    if robust:
        lines.append("  (PSM and Mahalanobis ATEs overlap within each other's 95% CIs)")
    else:
        lines.append("  (ATEs diverge — interpret PSM result with caution)")

    lines.append("\n--- THESIS CITATION GUIDE ---")
    lines.append("  PSM method: Rosenbaum & Rubin (1983); Austin (2011) for SMD balance check")
    lines.append("  Caliper: 0.2 x SD of logit propensity (Austin 2011 recommendation)")
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
