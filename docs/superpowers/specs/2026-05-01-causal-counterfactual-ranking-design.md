# Direction 5: Causal / Counterfactual Ranking

**Date:** 2026-05-01
**Status:** Approved — ready for implementation
**Domain:** Coffee (primary)

---

## Problem

The current BiRank pipeline is correlational: behavioral features predict revisit rates, but we cannot claim they *cause* them. Direction 5 moves from prediction to causal inference — demonstrating that temporal consistency (routine weekday patterns + predictable peak hours) causally drives future revisit rates, independent of raw venue popularity.

**Goal:** Run a Propensity Score Matching (PSM) study on coffee venues. Quantify the Average Treatment Effect (ATE) of behavioral consistency on future revisit rate, with bootstrap confidence intervals and a Mahalanobis robustness check. Produce a thesis-ready report.

This is a **standalone validation analysis** — the ranking pipeline is not modified.

---

## Architecture

```
coffee_venue_features_v2.csv  +  ../yelp_dataset/yelp_academic_dataset_review.json
              ↓
causal_data_prep.py
  → filters Yelp review JSON to coffee business IDs (from business_coffee_v2.csv)
  → temporal split at 2020-01-01 (pre-2020 = features, 2020+ = outcome)
  → computes treatment: consistency_score = weekday_ratio - minmax_norm(peak_hour_entropy)
  → computes outcome: future_revisit_rate per venue (2020+ window) from review timestamps
  → confounders: total_visits, unique_users, gini_user_contribution from coffee_venue_features_v2.csv
    (full-period proxies — stable venue characteristics, acceptable for matching)
  → outputs: causal_venue_dataset.csv
              ↓
causal_psm.py
  → fits logistic regression: P(treatment | confounders)
  → 1:1 nearest-neighbour matching on propensity score (caliper = 0.2 × SD of logit propensity)
  → covariate balance table: SMD before/after matching (threshold: SMD < 0.1)
  → ATE = mean(outcome | treated) − mean(outcome | control) in matched sample
  → bootstrap 95% CI: 1,000 resamples
  → Mahalanobis robustness check on same confounders
  → outputs: psm_matched_pairs.csv, psm_balance_table.csv
              ↓
causal_report.py
  → formats thesis-ready summary: ATE, CI, p-value, balance table, robustness check
  → outputs: causal_results.txt
```

---

## Treatment Definition

```
consistency_score = weekday_ratio - minmax_norm(peak_hour_entropy)
```

- `weekday_ratio`: already 0–1 (fraction of check-ins Mon–Fri)
- `peak_hour_entropy`: min-max normalised to 0–1 before subtracting (equal weight to both components)
- Score range: −1 (random/weekend) → +1 (routine/weekday + single peak hour)
- **Treatment = 1** if consistency_score > median, **Treatment = 0** otherwise
- Median split: maximises sample size in both groups, avoids cherry-picked threshold
- Only venues with check-in data are included (8,429 of 8,509 venues)

**Interpretation:** A treated venue has the behavioral signature of a "commuter habit" venue — people visit on predictable weekday schedules with consistent peak hours. Control venues are more random in timing or weekend-heavy.

---

## Confounders

Confounders sourced from `coffee_venue_features_v2.csv` (full-period values). Using full-period proxies is pragmatic and defensible: these venue characteristics (popularity, reach, user concentration) are structurally stable and predate the treatment period. The key leakage risk — outcome variables influencing confounders — is avoided by keeping outcome computation strictly post-2020.

| Confounder | Source column | Rationale |
|---|---|---|
| `total_visits` | `coffee_venue_features_v2.csv` | Raw popularity — main confound; popular venues attract consistent patterns AND more revisits |
| `unique_users` | `coffee_venue_features_v2.csv` | Reach/diversity; affects both temporal profiles and retention |
| `gini_user_contribution` | `coffee_venue_features_v2.csv` | Super-user concentration drives both routine patterns and high revisit rates |

**Excluded:**
- `repeat_user_rate` / `revisit_rate` — outcome proxy, including blocks the causal pathway
- Star ratings — potential mediator (quality → consistency → revisit), not a pure confounder

---

## Outcome

```
future_revisit_rate = unique_users_post / unique_users_pre
```

Where:
- `unique_users_pre` = distinct users who visited the venue before 2020-01-01
- `unique_users_post` = of those same users, how many visited again in 2020+
- Venues with `unique_users_pre = 0` are excluded (no valid counterfactual)

This measures **user retention** — what fraction of a venue's pre-2020 user base returned after 2020. It is forward-looking and cannot be caused by post-2020 behavioral patterns.

---

## Propensity Score Matching

**Model:** Logistic regression — `P(treatment=1 | total_visits_pre, unique_users_pre, gini_pre)`
All three confounders standardised (zero mean, unit variance) before fitting.

**Matching:**
- 1:1 nearest-neighbour matching without replacement
- Caliper = 0.2 × SD of logit propensity score (Rosenbaum & Rubin standard caliper)
- Unmatched treated venues (outside caliper) are excluded from ATE estimation

**Balance check:** SMD per confounder before and after matching.
- SMD < 0.1: well-balanced (cite Austin 2011)
- SMD > 0.1: report but note limitation

---

## ATE Estimation

```
ATE = mean(future_revisit_rate | treatment=1) − mean(future_revisit_rate | treatment=0)
      computed on matched sample only
```

**Inference:**
- Bootstrap 95% CI: 1,000 resamples of matched pairs (resample pairs, not individual venues)
- Two-sided paired t-test p-value on matched differences

**Robustness check:** Re-run matching using Mahalanobis distance on the same three confounders. Report Mahalanobis ATE alongside PSM ATE. If both ATEs fall within each other's bootstrap CIs, the result is robust to matching method choice. One paragraph in thesis.

---

## Outputs

| File | Description |
|---|---|
| `causal_venue_dataset.csv` | One row per venue: treatment, outcome, confounders, propensity score, consistency_score |
| `psm_matched_pairs.csv` | Matched treated/control pairs with venue IDs, propensity scores, confounder values |
| `psm_balance_table.csv` | SMD per confounder before/after matching |
| `causal_results.txt` | ATE, 95% CI, p-value, balance summary, Mahalanobis comparison, thesis-ready |

---

## Out of Scope

- Modifying the ranking pipeline (Direction 5 is validation only)
- Restaurant / Hotel domain extension (coffee only for this spec)
- Dashboard integration
- Instrumental variable estimation or difference-in-differences (future work)
