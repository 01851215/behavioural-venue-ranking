"""
Doubly-Robust ATE Estimator (AIPW)

Augmented Inverse Probability Weighting — robust if EITHER
the propensity model OR the outcome model is correctly specified.

This is the third agreement check alongside PSM and Mahalanobis,
addressing the key limitation in the master's causal study.

Reference: Robins, Rotnitzky & Zhao (1994); Bang & Robins (2005)
Paper target: KDD 2029 / AISTATS 2029
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bvr.causal.psm import fit_propensity_model, CONFOUNDER_COLS


def estimate_ate_dr(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "future_revisit_rate",
    confounder_cols: list = CONFOUNDER_COLS,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Doubly-Robust (AIPW) ATE estimator.

    AIPW formula:
        ATE = E[ (T/e - (1-T)/(1-e)) * Y  +  (1 - T/e) * mu1  -  (1 - (1-T)/(1-e)) * mu0 ]

    where:
        e = propensity score P(T=1|X)
        mu1 = E[Y|T=1,X], mu0 = E[Y|T=0,X]  (outcome regression predictions)

    Doubly-robust: consistent if either e or (mu0, mu1) is correctly specified.
    """
    df = df.dropna(subset=confounder_cols + [treatment_col, outcome_col]).copy()

    # Step 1: Propensity model (reuse from PSM)
    df = fit_propensity_model(df, confounder_cols)
    e = df["propensity_score"].values.clip(0.01, 0.99)
    T = df[treatment_col].values
    Y = df[outcome_col].values

    # Step 2: Outcome regression
    scaler = StandardScaler()
    X = scaler.fit_transform(df[confounder_cols].values)
    X_with_T = np.c_[X, T]

    outcome_model = LinearRegression()
    outcome_model.fit(X_with_T, Y)

    X1 = np.c_[X, np.ones(len(X))]   # T=1
    X0 = np.c_[X, np.zeros(len(X))]  # T=0
    mu1 = outcome_model.predict(X1)
    mu0 = outcome_model.predict(X0)

    # Step 3: AIPW estimator
    aipw = (T / e - (1 - T) / (1 - e)) * Y + (1 - T / e) * mu1 - (1 - (1 - T) / (1 - e)) * mu0
    ate = float(aipw.mean())

    # Step 4: Bootstrap confidence interval
    np.random.seed(random_state)
    boot_ates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(aipw), len(aipw))
        boot_ates.append(aipw[idx].mean())

    ci_lo, ci_hi = np.percentile(boot_ates, [2.5, 97.5])
    p_value = 1 - (np.array(boot_ates) < 0).mean() if ate > 0 else (np.array(boot_ates) < 0).mean()

    return {
        "estimator": "AIPW (doubly-robust)",
        "ate": round(ate, 6),
        "ci_lo": round(float(ci_lo), 6),
        "ci_hi": round(float(ci_hi), 6),
        "p_value": round(float(p_value), 4),
        "significant_at_05": float(ci_lo) > 0 or float(ci_hi) < 0,
        "n": len(df),
        "n_bootstrap": n_bootstrap,
    }
