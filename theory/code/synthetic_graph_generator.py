"""
Synthetic Bipartite Graph Generator for Theory Validation.

Generates bipartite user-venue graphs with controlled behavioral mode β ∈ [0,1]:
  β = 0: pure loyalty domain — all users revisit the same venues repeatedly
  β = 1: pure exploration domain — all users always visit new venues

Used to validate the Prior-Domain Match Theorem by checking whether
the theoretically predicted α*(β) matches the empirically optimal α.

Notebook: theory/notebooks/01_synthetic_validation.ipynb
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_synthetic_domain(
    n_users:  int = 500,
    n_venues: int = 200,
    n_events: int = 5000,
    beta:     float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic interaction dataset with controlled behavioral mode β.

    β = 0: each user always returns to the same venue (pure loyalty)
    β = 1: each user always picks a new venue uniformly at random (pure exploration)
    Intermediate β: mixture — each event is loyalty with prob (1-β), exploration with prob β

    Returns: DataFrame with columns [user_id, business_id, timestamp, stars]
    """
    rng = np.random.RandomState(random_state)

    # Assign each user a "home venue" (for loyalty behaviour)
    home_venues = rng.randint(0, n_venues, size=n_users)

    # Generate events
    events = []
    for i in range(n_events):
        user_id = rng.randint(0, n_users)

        if rng.rand() < beta:
            # Exploration: pick a new venue
            venue_id = rng.randint(0, n_venues)
        else:
            # Loyalty: return to home venue (with small noise)
            if rng.rand() < 0.9:
                venue_id = home_venues[user_id]
            else:
                venue_id = rng.randint(0, n_venues)

        # Timestamp: uniform over 2 years
        ts = pd.Timestamp("2018-01-01") + pd.Timedelta(days=rng.randint(0, 730))

        events.append({
            "user_id":     f"U{user_id:04d}",
            "business_id": f"V{venue_id:04d}",
            "timestamp":   ts,
            "stars":       float(rng.choice([3, 4, 5], p=[0.2, 0.4, 0.4])),
        })

    df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
    return df


def estimate_beta_from_data(df: pd.DataFrame, split_date: str = "2019-01-01") -> float:
    """Estimate β from data (for validation of estimate_behavioral_mode)."""
    train = df[df["timestamp"] < pd.Timestamp(split_date)]
    # β ≈ 1 - mean(repeat_user_rate)
    revisit_rates = []
    for uid, grp in train.groupby("user_id"):
        visits = grp["business_id"].value_counts()
        total  = len(grp)
        revisit = (grp["business_id"].duplicated()).sum()
        revisit_rates.append(revisit / total if total > 0 else 0)
    return float(1 - np.mean(revisit_rates)) if revisit_rates else 0.5


def run_alpha_sweep(
    df: pd.DataFrame,
    split_date: str = "2019-01-01",
    alphas: list = None,
) -> dict:
    """
    Run the exploration prior α-sweep on a synthetic dataset.
    Returns {alpha: rising_stars_rho} for each α.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from bvr.core.validation import (
        compute_user_features, compute_venue_features,
        build_decayed_edges, build_adjacency, birank, evaluate_per_user,
    )
    from bvr.pipelines.london import (
        temporal_split, compute_rising_stars, spearman_rising,
    )

    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    df = df.copy()
    train, _, train_uv, _ = temporal_split(df, split_date)
    if len(train) < 100:
        return {}

    uf = compute_user_features(train)
    vf = compute_venue_features(train)
    decay_e = build_decayed_edges(train, split_date, lam=0.5)
    test_full = df[df["timestamp"] >= pd.Timestamp(split_date)].copy()
    rising = compute_rising_stars(train, test_full)
    if not rising:
        return {}

    W, u2i, v2i, i2u, i2v = build_adjacency(decay_e)
    pop_map = vf.set_index("business_id")["popularity_visits"]

    results = {}
    for alpha in alphas:
        p0 = np.ones(len(u2i))
        q0 = np.clip(np.array([
            1.0 if alpha == 0 else 1.0 / (np.log1p(float(pop_map.get(i2v[i], 1) or 1)) ** alpha)
            for i in range(len(v2i))
        ]), 1e-10, None)
        _, q = birank(W, p0=p0, q0=q0)
        ranking = {i2v[i]: float(q[i]) for i in range(len(v2i))}
        rho, _ = spearman_rising(ranking, rising)
        results[alpha] = round(rho, 4)

    return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    sys_path = str(Path(__file__).parent.parent.parent)

    print("Generating synthetic graphs for theory validation...")
    print(f"{'β':>6} {'estimated_β':>14} {'best_α_empirical':>18} {'theory_α*(2β)':>15}")
    print("-" * 58)

    all_results = []
    for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        df = generate_synthetic_domain(beta=beta, n_events=3000, random_state=42)
        beta_est = estimate_beta_from_data(df)
        rho_by_alpha = run_alpha_sweep(df)

        if rho_by_alpha:
            best_alpha = max(rho_by_alpha, key=rho_by_alpha.get)
            theory_alpha = 2.0 * beta   # linear approximation
            print(f"{beta:>6.1f} {beta_est:>14.4f} {best_alpha:>18.2f} {theory_alpha:>15.2f}")
            all_results.append({
                "beta_true": beta, "beta_estimated": beta_est,
                "best_alpha_empirical": best_alpha,
                "theory_alpha": theory_alpha,
                "rho_by_alpha": rho_by_alpha,
            })

    out = Path(__file__).parent.parent.parent / "data/results/synthetic_theory_validation.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {out}")
