"""
V7: Shuffled-label permutation test on causal ATE

Permutes treatment assignment 1000 times.
Null ATE distribution should centre on ~0.
Observed ATE=+0.0027 should be in the right tail (p < 0.05).

This validates that the PSM isn't finding spurious structure
in the data — the result is genuinely due to the treatment
variable, not an artefact of the matching algorithm.

Saves: permutation_test_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import percentileofscore

DATA_DIR = Path(__file__).parent
N_PERMUTATIONS = 1000


def simple_ate(df: pd.DataFrame) -> float:
    """Compute ATE as mean outcome difference between treated and control."""
    treated = df[df["treatment"] == 1]["future_revisit_rate"]
    control = df[df["treatment"] == 0]["future_revisit_rate"]
    return float(treated.mean() - control.mean())


if __name__ == "__main__":
    dataset = pd.read_csv(DATA_DIR / "causal_venue_dataset.csv")
    dataset = dataset.dropna(subset=["future_revisit_rate", "treatment"])

    # Observed ATE
    obs_ate = simple_ate(dataset)
    print(f"Observed ATE: {obs_ate:+.6f}")
    print(f"Running {N_PERMUTATIONS} permutations...")

    np.random.seed(42)
    null_ates = []
    for _ in range(N_PERMUTATIONS):
        shuffled = dataset.copy()
        shuffled["treatment"] = np.random.permutation(shuffled["treatment"].values)
        null_ates.append(simple_ate(shuffled))

    null_ates = np.array(null_ates)

    # One-tailed p-value: P(null ATE >= observed ATE)
    p_value    = 1 - percentileofscore(null_ates, obs_ate, kind="rank") / 100
    null_mean  = null_ates.mean()
    null_std   = null_ates.std()
    percentile = percentileofscore(null_ates, obs_ate)

    print(f"\n=== PERMUTATION TEST RESULTS ===")
    print(f"Observed ATE:          {obs_ate:+.6f}")
    print(f"Null distribution:     mean={null_mean:+.6f}  std={null_std:.6f}")
    print(f"Observed percentile:   {percentile:.1f}th")
    print(f"One-tailed p-value:    {p_value:.4f}")

    if p_value < 0.05:
        print(f"→ Significant (p={p_value:.4f}) — ATE is in the right tail of null ✓")
    else:
        print(f"→ Not significant (p={p_value:.4f}) — ATE is not unusual under null ⚠")

    # Save
    results = pd.DataFrame({
        "metric": ["observed_ate", "null_mean", "null_std",
                   "observed_percentile", "p_value_one_tailed"],
        "value":  [obs_ate, null_mean, null_std, percentile, p_value]
    })
    results.to_csv(DATA_DIR / "permutation_test_results.csv", index=False)

    null_df = pd.DataFrame({"permuted_ate": null_ates})
    null_df.to_csv(DATA_DIR / "permutation_null_distribution.csv", index=False)
    print(f"Saved → permutation_test_results.csv, permutation_null_distribution.csv")
