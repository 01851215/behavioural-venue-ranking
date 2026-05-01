# Direction 3: Cold-Start / Anonymous Venue Signal Ranking

**Date:** 2026-05-01
**Status:** Approved — ready for implementation
**Domain:** Coffee (primary), extensible to Restaurant / Hotel

---

## Problem

72% of Yelp check-in events have no user_id. These 13.3M timestamped visits across 131,930 venues are currently unused. Venues with fewer than N reviews are underrepresented in the BiRank bipartite graph — venues with zero reviews receive a score of 0 and are omitted from the ranked output entirely.

**Goal:** Build venue-side behavioral signals from anonymous timestamps alone, enrich the existing venue feature set, and produce a cold-start score for sparse/zero-review venues that integrates cleanly with the existing BiRank pipeline.

---

## Architecture

```
yelp_academic_dataset_checkin.json
           ↓
compute_anonymous_venue_signals.py
  → extracts 10 temporal features for all 131,930 venues
  → enriches coffee_venue_features_v2.csv with new columns
           ↓
cold_start_ranker.py
  → trains calibration regression on warm venues
    (temporal features → log BiRank score)
  → sweeps threshold [3, 5, 10, 20] reviews
  → produces pseudo-scores for cold venues
  → outputs: cold_start_scores.csv
           ↓
run_pipeline_v5.py  (extends run_pipeline_v4.py)
  → post-processing injection: merges BiRank + pseudo-scores
  → tags each venue: score_source = "birank" | "cold_start" | "unranked"
  → outputs: coffee_birank_venue_scores_v5.csv
           ↓
validate_v5_coldstart.py
  → coverage stat, calibration Spearman r, NDCG@10 with/without cold-start
```

**Hard rule:** Warm venue BiRank scores are never modified. The `score_source` column makes ablation trivial — filter to `birank` only to reproduce v5 baseline exactly.

---

## Feature Extraction

File: `compute_anonymous_venue_signals.py`
Input: `../yelp_dataset/yelp_academic_dataset_checkin.json`
Output: new columns on `coffee_venue_features_v2.csv`

| Feature | Description | Signal |
|---|---|---|
| `total_checkins` | Raw check-in count | Anonymous popularity proxy |
| `checkin_burstiness` | CV of daily visit counts | Spiky vs steady demand |
| `peak_hour_entropy` | Shannon entropy of hour-of-day distribution | Predictable routine vs random traffic |
| `weekday_ratio` | Fraction of check-ins Mon–Fri | Commuter/worker vs leisure |
| `temporal_stability_cv` | CV of weekly visit counts over time | Consistent vs volatile footfall |
| `visit_velocity_recent` | Check-ins in last 6 months / total | Momentum |
| `growth_trend` | Slope of monthly visit bins (linear fit) | Long-term trajectory |
| `lunch_dinner_ratio` | (11am–2pm + 5pm–9pm) / total | Food-occasion focus |
| `late_night_ratio` | 10pm–2am / total | Night crowd signal |
| `peak_hour_mode` | Most common hour (0–23) | Primary use-case anchor |

All 10 features are computed for every venue in the check-in JSON. Venues absent from the check-in JSON get `NaN` for all 10 columns.

---

## Cold-Start Regression

File: `cold_start_ranker.py`

**Training set:** Warm venues appearing in both check-in JSON and BiRank output (above threshold). 80/20 random train/held-out split (not temporal — we're calibrating signal quality, not predicting future venues).

**Target:** `log(birank_score + ε)` where `ε = 1e-10` — log-transform stabilises the heavy-tailed BiRank distribution.

**Models:**
- Primary: Ridge regression (interpretable, one hyperparameter, citable)
- Secondary: LightGBM (non-linear comparison — if Spearman r is significantly higher, report both)

**Threshold sweep:** [3, 5, 10, 20] reviews. For each threshold:
- Calibration Spearman r on held-out warm venues
- % of total venues rescued (coverage gain)
- NDCG@10 on warm-venue evaluation set

**Best threshold selection:** Highest coverage gain where Spearman r ≥ 0.4 and NDCG@10 within 1% of v5 baseline. The 0.4 Spearman floor is the minimum academic credibility threshold — below this the calibration is too weak to publish.

**Output schema:**
```
business_id | pseudo_birank_score | cold_start_threshold | score_source
```

---

## Injection & Merge

File: `run_pipeline_v5.py`

**Normalization:** Map pseudo-scores onto the BiRank percentile curve using venues within 2× the threshold as the calibration band (e.g., threshold=5 → use venues with 5–10 reviews as anchors). Percentile-to-percentile mapping ensures cold venues slot into the ranking at plausible positions.

**Merge logic:**
```python
if venue.review_count >= threshold:
    final_score = birank_score              # warm — unchanged
else:
    final_score = normalized_pseudo_score   # cold — injected
```

**Unranked fallback:** Venues absent from both check-in JSON and reviews get `final_score = 0`, `score_source = "unranked"`.

**Output schema** (extends `coffee_birank_venue_scores.csv`):
```
business_id | final_score | rank | score_source | review_count | cold_threshold_used
```

---

## Validation

File: `validate_v5_coldstart.py`
Reuses existing temporal train/test split from `validate_v5.py` for direct comparability.

**Four metrics:**

**1. Coverage gain**
- Before: % of venues with non-zero BiRank score
- After: % of venues with non-zero score (birank + cold-start)
- Delta: headline stat for thesis

**2. Calibration quality**
- Spearman r between pseudo-score and actual BiRank on held-out warm venues
- Reported per threshold in ablation table

**3. Ranking preservation**
- NDCG@10 on warm-venue evaluation set must stay within 1% of v5 baseline
- Confirms cold-start injection does not disturb warm venue ordering

**4. Threshold sweep ablation table**

| Threshold | Spearman r | Coverage gain | NDCG@10 |
|---|---|---|---|
| 3 reviews | — | — | — |
| 5 reviews | — | — | — |
| 10 reviews | — | — | — |
| 20 reviews | — | — | — |

---

## Files Produced

| File | Description |
|---|---|
| `compute_anonymous_venue_signals.py` | Extract 10 temporal features from check-in JSON |
| `cold_start_ranker.py` | Train calibration regression, produce pseudo-scores |
| `run_pipeline_v5.py` | Extended pipeline with injection/merge step |
| `validate_v5_coldstart.py` | Coverage, calibration, NDCG, threshold ablation |
| `coffee_venue_features_v2.csv` | Enriched with 10 new anonymous signal columns |
| `cold_start_scores.csv` | Pseudo-scores for cold venues |
| `coffee_birank_venue_scores_v5.csv` | Unified ranked output with score_source tags |

---

## Out of Scope

- Direction 5 (causal/counterfactual ranking) — separate spec
- Restaurant / Hotel domain extension — reuse same scripts with domain flag after coffee validation
- Dashboard integration — after validation passes
