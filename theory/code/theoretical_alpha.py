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

    Current (linear approximation — to be superseded by PhD derivation):
        α* ≈ 2β

    PhD contribution: derive the exact functional form f(β) using
    KL divergence between visit distributions across domain types.
    The derivation is in theory/derivations/prior_domain_match.tex.
    """
    return 2.0 * beta


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
