"""
Pillar 1 — Theoretical α Predictor

Implements the Prior-Domain Match Theorem (to be derived in
theory/derivations/prior_domain_match.tex).

Given a domain's behavioral mode β ∈ [0,1]:
  - β ≈ 0: loyalty-driven domain (coffee, habit-driven)
  - β ≈ 1: exploration-driven domain (tourist restaurants, check-ins)

The theorem predicts the optimal exploration prior exponent α* = f(β).
Current approximation (master's-level): α* ≈ 2β
PhD contribution: derive f(β) exactly via information-theoretic arguments.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bvr.core.validation import compute_user_features, compute_venue_features


def estimate_behavioral_mode(train_df: pd.DataFrame) -> float:
    """
    Estimate β ∈ [0, 1] from training interactions.
    β = 0: pure loyalty (everyone revisits the same venue)
    β = 1: pure exploration (everyone visits new venues only)

    β = 1 - mean(repeat_user_rate across venues)
    """
    venue_feat = compute_venue_features(train_df)
    return float(1.0 - venue_feat["repeat_user_rate"].mean())


def theoretical_optimal_alpha(beta: float) -> float:
    """
    Prior-Domain Match: predicted optimal α given domain behavioral mode β.

    Current (linear approximation — known to be wrong from synthetic validation):
        α* ≈ 2β

    Synthetic validation (theory/code/synthetic_graph_generator.py) shows
    this linear approximation fails: empirical best α is consistently 2.0
    regardless of true β, while estimated β is systematically overestimated.

    TWO PhD findings from this:
      1. The estimate_behavioral_mode() function is biased — needs correction.
         (β̂ is typically 0.577–0.989 even when true β = 0.1–0.9)
      2. The optimal α is NOT linearly related to β — the real relationship
         is non-linear and requires the information-theoretic derivation
         in theory/derivations/prior_domain_match.tex.

    TODO (PhD Year 1): Replace with closed-form derived from KL divergence
    between visit distributions. See Section 4 of the theory paper.
    """
    # Placeholder: will be replaced by formal derivation
    # Known to fail on synthetic data — motivates PhD theoretical contribution
    return 2.0 * beta   # INCORRECT: just a placeholder


def predict_and_compare(train_df: pd.DataFrame,
                        empirical_best_alpha: float) -> dict:
    """
    Compare theoretical α* against empirically found best α.
    Input empirical_best_alpha from bvr/experiments/ablations.py output.
    """
    beta = estimate_behavioral_mode(train_df)
    alpha_pred = theoretical_optimal_alpha(beta)
    error = abs(alpha_pred - empirical_best_alpha)

    return {
        "beta": round(beta, 4),
        "alpha_theoretical": round(alpha_pred, 4),
        "alpha_empirical": empirical_best_alpha,
        "absolute_error": round(error, 4),
        "match": error < 0.25,    # within one grid step of the ablation sweep
    }
