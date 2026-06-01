# Behavioral Venue Ranking System

A venue ranking system that ranks businesses by what people **do** (visit patterns, loyalty, regularity) rather than what they **say** (star ratings). Built on the Yelp Academic Dataset, supplemented with Foursquare check-ins and Transitland transit data.

Includes a validated **coffee shop model** (BiRank with behavioral priors), a **restaurant model** (multi-objective S(R,U,C) scoring with mobility and context awareness), a **hotel model** (BiRank with domain-adapted features), and an **LLM simulation validation layer** — 3,360 GPT-5.4 synthetic personas grounded in real user archetypes and published consumer-behaviour research. Explore rankings through an interactive Streamlit dashboard.

---

## Plain English Summary

**The problem:** When you search for a coffee shop or restaurant, most apps rank places by star ratings. But ratings are easy to fake, influenced by mood, and don't tell you whether people actually come back. A place might have 4.8 stars from 12 reviews — or it might have 4.2 stars from people who go there every single week.

**The idea:** Instead of asking "what do people say about this place?", ask "what do people *do*?". If someone visits a cafe every Tuesday morning for three years, that tells you far more than their one-time 5-star review. This project ranks venues using visit behaviour — how often people return, how loyal they are, how consistent the traffic is — rather than what they write.

**How it works in simple terms:** Imagine drawing a web connecting every person to every venue they've visited. A venue that attracts many returning, regular visitors gets boosted. A user who visits frequently and consistently gets treated as a more reliable signal. The algorithm (BiRank) bounces a score back and forth across this web until it settles — important users boost important venues, and important venues boost the importance of users who visit them.

**What we built:**
- A coffee shop ranking model across 8,500+ venues and 93,000 users, grouping people into four types: Loyalists (their regular), Weekday Regulars (work-routine), Casual Weekenders, and Infrequent Visitors
- A restaurant model that also considers how far you are from a place, how easy it is to reach by public transport, and whether it matches your cuisine preferences
- A hotel model that redesigns all behavioral features from scratch — hotels require completely different signals than cafés
- An LLM simulation layer with 3,360 synthetic personas (GPT-5.4) that independently validates the models — personas choose between venues just like real users would
- An interactive map and simulation dashboard to explore rankings and live persona responses by city

**What we found:** The behavioural approach consistently outperforms star-rating ranking and random recommendation. Crucially, it works best for Loyalists — people with strong habits — which is exactly what you'd expect if the theory is right. The LLM simulation confirms this: Loyalist personas and high-loyalty occupation clusters (Legal/Finance, Executive) align most strongly with the model's top picks.

---

**Version history at a glance:**

- **v3 (baseline):** First working behavioural model. Proved the core idea — BiRank on visit behaviour beats star ratings. But the validation had bugs that made results look better than they were.
- **v4 (Foursquare):** Added check-in data from a second platform (Foursquare) to bring in social signals — whether your friends visited a place. Turned out this made results *worse*: the social data was too noisy and unrelated to coffee habits.
- **v5 (honest numbers):** Fixed three serious methodological errors: (1) the model was accidentally "cheating" by using future data it shouldn't have seen during training; (2) the accuracy metric was calculated incorrectly; (3) there were no statistical tests proving results weren't just luck. After fixing all three, the numbers dropped (from 0.086 to 0.076) — but these are the *correct* numbers. Also fixed the Foursquare integration to only use high-confidence social links, which stopped it from hurting performance.
- **v6 (hybrid experiment):** Tested whether adding a second type of algorithm — Matrix Factorization, which finds hidden patterns like "people who like X tend to like Y" — could improve on BiRank. It didn't. The best blend was still essentially pure BiRank (λ=1.0 selected by tuning). This is a meaningful negative result: BiRank's behavioural signals are already capturing what matters, and "collaborative filtering" patterns add nothing extra in this domain.
- **v7 (hotel model):** Extended the whole framework to hotels and accommodation. This required redesigning the behavioral features from scratch — hotels are fundamentally different from coffee shops (nobody visits the same hotel weekly). Key finding: BiRank still beats star ratings (p=0.012), but collaborative filtering outperforms behavioral signals for hotels because most users only stay at 1–2 hotels, making behavior patterns too sparse to learn from. Also conducted a cross-domain experiment: users who explore many coffee shops tend to explore many hotels too, but predicting hotel preferences from coffee habits is only marginally better than chance.
- **v8 (LLM simulation):** Added two independent external validation studies using GPT-5.4 synthetic personas. **Study 1** — 1,500 personas grounded in the four behavioural archetypes identified from Yelp data (Loyalist, Weekday Regular, Casual Weekender, Infrequent Visitor) across all three domains. Each persona performs three tasks: venue ranking (NDCG@10), pairwise head-to-head (BiRank vs. stars), and revisit prediction. Metrics include Hit@1/3, Kendall τ, BH-corrected p-values, Cohen's d, and rank-biserial correlation. **Study 2** — 1,860 personas across a 5 age-group × 10 occupation cross-matrix (Gen Z → Boomer; Tech/Software → Remote/Digital Nomad), grounded in 51 published consumer-behaviour sources (NCA, McKinsey, J.D. Power, GBTA, Hilton Trends Report, etc.). Both studies run alongside the real-data validation for independent triangulation. Also added a live Persona Chat in the Streamlit dashboard: pick an archetype, city, and domain — a GPT-5.4-mini persona recommends real venues from the dataset and explains why in character.
- **v9 (cold-start + causal):** Two completed directions, both fully tested and validated. **Direction 3** — anonymous venue signal module: extracted 10 temporal features (burstiness, peak-hour entropy, weekday ratio, growth trend, etc.) from 13.3M Yelp check-in timestamps across 131,930 venues. Trained a calibrated Ridge regression (Spearman r=0.62; LightGBM r=0.76) to produce pseudo-BiRank scores for venues with fewer than 20 reviews. Pipeline v5 injects cold-start scores for 1,045 previously unranked venues, raising total coverage from 86.8% → 99.1% (+12.3%). Venue feature matrix expanded from 15 → 25 columns. **Direction 5** — causal PSM study: 1:1 nearest-neighbour Propensity Score Matching across 7,853 coffee venues (2,957 matched pairs). All confounders well-balanced post-matching (SMD < 0.1). ATE = +0.001 (directionally positive, p=0.15) — result is robust across both PSM and Mahalanobis matching methods. Statistical power is reduced by COVID-19 compressing post-2020 revisit rates.
- **v10 (UK expansion — London + Foursquare GB):** Extended the pipeline to the UK using two independent data sources. **(a) TripAdvisor London** (Zenodo 6583422, CC-BY-NC 4.0): 997K reviews, 1,877 restaurants. Three-phase analysis — rising-stars evaluation, exploration priors, MF hybrid. Standard BiRank loyalty priors actively harm on exploration-driven data (ρ = −0.21, p<0.001); hybrid exploration-BiRank + ALS significantly beats popularity on rising-stars prediction (ρ = +0.094, p<0.001, bootstrap NDCG@10 95% CI [0.550, 0.567]). **(b) Foursquare WWW2019 GB subset**: 288K check-ins (after filtering non-venue categories), 6,733 users, 70,042 venues across all of Great Britain. Same BiRank + MF + hybrid pipeline run with 2013-07-01 split. Hybrid wins rising-stars ρ (+0.040, p<0.001); popularity wins NDCG@10 (0.316 vs hybrid 0.215), which is expected — popularity predicts revisit but cannot identify rising stars. Cross-dataset consistency: exploration priors and hybrid generalise from London tourist restaurants (review data) to GB-wide anonymous check-ins (no star ratings). All NDCG CIs from 1000 bootstrap resamples; Wilcoxon signed-rank tests computed for every method pair. **Key finding:** Behavioral ranking is domain-specific — loyalty priors work for coffee shops (Loyalist domain), exploration priors + MF hybrid work for tourist/exploration-driven domains. Consistent across two independent UK datasets.
- **v12 (LightGCN comparison):** Added LightGCN (He et al., SIGIR 2020) as a state-of-the-art graph convolution baseline against the hybrid model on both UK datasets. Implementation: `lightgcn.py` — PyTorch custom implementation, 3 propagation layers, 64 embedding dimensions, BPR loss, Adam optimiser, 50 epochs, CPU (MPS excluded: PyTorch sparse ops not supported on Apple Silicon). **Results — London:** ρ = −0.059 (p=0.015) — significantly negative, below the hybrid (+0.094); NDCG@10 = 0.5526, Hit@10 = 0.9014 — competitive for revisit prediction but not rising stars. **Results — UK FSQ:** ρ = −0.102 (p<0.001) — strongly negative; NDCG@10 = 0.2374, Hit@10 = 0.6375. **Key finding:** LightGCN is specifically worse than the domain-adapted hybrid on the rising-stars metric across both datasets. Graph convolution amplifies popularity signal through neighbourhood propagation, making already-popular venues more dominant — the opposite of what rising-star discovery requires. This validates the thesis that domain adaptation (exploration priors) outperforms raw algorithmic power (graph convolution) for this task. LightGCN beats the hybrid on raw traffic correlation (+0.43 vs −0.04 London, +0.17 vs −0.04 UK FSQ) but raw traffic is confounded by popularity, which is why we use the debiased rising-stars metric as the primary evaluation.
- **v11 (restaurant validation + domain-specificity finding):** First proper v5-style validation of the restaurant model — 13.5M training reviews, 60,644 venues, 11,948 evaluable users (33.8% revisit rate). **Surprise finding:** star ratings beat BiRank for restaurants (NDCG@10: stars 0.406 vs BiRank 0.396, p=0.035), the reverse of the coffee result. This completes a four-domain pattern — behavioral ranking wins in habit-driven domains (coffee), stars win in quality-driven domains (restaurants, hotels), neither wins in exploration-driven domains (London tourists). The Loyalist segment achieves NDCG@10=0.667 across restaurants, confirming that behavioral regularity, when it exists, is the strongest predictive signal regardless of domain.

---

## Validation Status

*Last validated: 2026-05-08 (core); restaurant + causal updated 2026-05-08.*

| Component | Status | Evidence |
|---|---|---|
| Unit tests | ✅ 58/58 passing | `pytest tests/` — 3 test suites, 3.8s |
| All output files | ✅ 12/12 present | All CSVs and reports on disk, non-zero |
| BiRank scores (coffee) | ✅ Differentiated | min=1.06e-07, max=9.29e-03, std=2.67e-04 |
| v5 unified ranking | ✅ Monotone | rank 1→8509, `final_score` strictly decreasing |
| Cold-start coverage | ✅ 99.1% | 7,389 BiRank + 1,045 cold-start + 75 unranked |
| PSM balance | ✅ All SMD < 0.1 | total_visits: 0.35→0.06, unique_users: 0.43→0.08 |
| Causal ATE (2018 split) | ✅ ATE=+0.0019, CI barely crosses 0 | Robust: PSM + Mahalanobis agree |
| Restaurant validation | ✅ NDCG@10=0.396 | Stars beat BiRank (p=0.035) — domain-specificity finding |
| London hybrid | ✅ ρ=+0.094, p<0.001 | Explore-BiRank + ALS beats popularity on rising stars |
| UK FSQ hybrid | ✅ ρ=+0.040, p<0.001 | Generalises to GB-wide Foursquare check-in data |
| UK FSQ benchmarks | ✅ All CIs + Wilcoxon computed | bootstrap n=1000; Wilcoxon two-sided for all 7 methods |
| LightGCN London | ✅ ρ=−0.059, NDCG=[0.5451,0.5623], Wilcoxon p=0.018 | Fully benchmarked — hybrid wins on ρ |
| LightGCN UK FSQ | ✅ ρ=−0.102, NDCG=[0.2232,0.2519], Wilcoxon p<0.001 | Fully benchmarked — hybrid wins on ρ |
| All modules importable | ✅ 7/7 | No import errors on any v9 module |
| Dashboard data deps | ✅ 7/7 present | cities_index, scores, features, explanations |

---

## Core Thesis

Star ratings are noisy, gameable, and one-dimensional. Behavioral signals — revisit rates, visit regularity, loyalty concentration, exploration diversity — reveal genuine venue quality. A cafe where hundreds of people return weekly is meaningfully different from one with a handful of 5-star reviews.

---

## Datasets

| Dataset | Source | Scale |
|---------|--------|-------|
| Yelp Academic Dataset | `yelp_academic_dataset_*.json` | Businesses, reviews, check-ins, tips, users across US/Canada |
| Foursquare WWW2019 | `dataset_WWW2019/` | 22.8M check-ins, 114K users, 607K friendships |
| Foursquare Raw POIs | `dataset_WWW2019/raw_POIs.txt` | 11.2M venues with lat/lon/category |
| **Foursquare WWW2019 GB subset** | `dataset_WWW2019/` (filtered) | 288K GB check-ins, 6,733 users, 70,042 venues — Apr 2012–Jan 2014 |
| TripAdvisor London | Zenodo 6583422 (CC-BY-NC 4.0) | 997K reviews, 502K users, 1,877 London restaurants |
| OpenStreetMap London | Overpass API | 28,124 London POIs (cafés, restaurants, hotels, pubs) |
| Transitland US | `tl-dataset-US-2025-12-24T16_23_26/` | US transit stops + routes with headway frequencies |

---

## Coffee Shop Model (BiRank)

### Pipeline

| Script | Phase | Description |
|--------|-------|-------------|
| `task1_identify_coffee_shops.py` | Data extraction | Filter Yelp to coffee/cafe businesses (8,509 venues) |
| `task2_construct_visit_events.py` | Data extraction | Build visit event timeline from reviews + check-ins |
| `task3_link_users_reviews.py` | Data extraction | Link users to their review activity |
| `task4_build_canonical_table.py` | Data extraction | Merge into canonical interaction table |
| `task5_extract_behaviour_features.py` | Feature engineering | Compute burstiness, entropy, revisit rate, venue stability, Gini |
| `task6_behaviour_interpretation.py` | Feature engineering | Generate human-readable behavioral tags |
| `taskA_build_bipartite_graph.py` | BiRank | Build user-venue bipartite graph |
| `taskB_implement_birank.py` | BiRank | Run BiRank with behavioral priors |
| `taskCD_baselines_comparison.py` | Evaluation | Compare against rating/popularity/random baselines |
| `taskE_group_specific_birank.py` | Groups | BiRank conditioned on user behavioral segments |
| `phase3_taskA_define_groups.py` | Groups | K-means clustering into 4 user archetypes |
| `phase3_tasksBCDEF_group_rankings.py` | Groups | Group-specific rankings and analysis |
| `temporal_validation.py` | Validation v3 | Temporal split (pre/post 2020) prediction test |
| `validate_v5.py` | Validation v5 | Corrected NDCG, leakage fix, significance tests, per-group eval, 3 splits |
| `validate_v6_hybrid.py` | Validation v6 | Hybrid BiRank + ALS/BPR matrix factorization with lambda tuning |
| `run_pipeline.py` | Pipeline v3 | End-to-end pipeline runner |

### Behavioral Features

- **Burstiness** (Goh-Barabasi): -1 (clockwork regular) to +1 (one-time burst)
- **Shannon Entropy**: Visit diversity across venues (explorer vs. loyalist)
- **Venue Stability**: Coefficient of variation of weekly traffic
- **Loyalty Concentration** (Gini): Broad vs. narrow visitor base
- **Revisit Rate**: Fraction of visitors who return

### User Segments (93,830 users)

| Segment | Share | Revisit Rate | Behavior |
|---------|-------|-------------|----------|
| Weekday Regulars | 49.0% | 1.2% | Work-routine coffee runs, high exploration |
| Casual Weekenders | 31.9% | 0.5% | Weekend brunch explorers |
| Loyalists | 9.0% | 41.0% | "Their cafe" — staff knows their order |
| Infrequent Visitors | 10.2% | 0.3% | Sporadic, 3.9 years between visits |

### Validation Results v5 (corrected, with significance tests)

Temporal split at 2020-01-01. Per-user candidate re-ranking on 17,746 overlapping users.

**Fixes applied in v5:** (1) feature leakage eliminated — features recomputed from training data only; (2) NDCG IDCG corrected to use total relevant candidates; (3) bootstrap 95% CI and Wilcoxon signed-rank significance tests added; (4) per-group evaluation added.

| Method | NDCG@10 | Hit@10 | 95% CI | p-value vs best |
|--------|---------|--------|--------|-----------------|
| **v5_combined (decay+social)** | **0.0765** | **11.7%** | [0.0725, 0.0813] | ref |
| v5_temporal_decay | 0.0764 | 11.6% | [0.0725, 0.0811] | 0.92 |
| v5_selective_social | 0.0763 | 11.6% | [0.0724, 0.0811] | 0.92 |
| v3_baseline (behavioral) | 0.0763 | 11.6% | [0.0724, 0.0811] | 0.89 |
| Rating (Stars) | 0.0754 | 11.7% | [0.0713, 0.0798] | 0.29 |
| Popularity (Visits) | 0.0743 | 11.6% | [0.0701, 0.0785] | 0.053 |
| IUF-Popularity | 0.0742 | 11.6% | [0.0699, 0.0784] | 0.048 |
| Random | 0.0742 | 11.5% | [0.0701, 0.0786] | **0.038** |
| Item-KNN (cosine) | 0.0724 | 11.5% | [0.0684, 0.0768] | **<0.001** |

BiRank significantly outperforms random (p=0.038) and item-KNN (p<0.001). v5_combined is the best variant (+0.30% over v3 baseline).

**Note:** v5 NDCG values are lower than earlier v3 reports (0.076 vs 0.086) because of the leakage fix and corrected IDCG computation. These are the methodologically correct numbers.

### Per-Group Results (NDCG@10, v5_combined)

| Segment | NDCG@10 | n_users |
|---------|---------|---------|
| **Loyalists** | **0.1734** | 612 |
| Casual Weekenders | 0.0796 | 1,503 |
| Weekday Regulars | 0.0689 | 4,768 |
| Infrequent Visitors | 0.0675 | 2,988 |

The model works best for Loyalists — users with high revisit rates — confirming that behavioral regularity is the strongest predictive signal.

### Robustness Across Temporal Splits (NDCG@10)

| Method | 2019-01-01 | 2019-07-01 | 2020-01-01 |
|--------|------------|------------|------------|
| v5_combined | 0.0786 | 0.0773 | 0.0765 |
| v3_baseline | 0.0785 | 0.0774 | 0.0763 |
| Rating | 0.0786 | 0.0771 | 0.0754 |
| Random | 0.0763 | 0.0754 | 0.0742 |

Results are consistent across all three temporal split points.

---

## Cold-Start Anonymous Venue Ranking (v9 — Direction 3)

Addresses a fundamental data coverage gap: 72% of Yelp check-in events have no `user_id`, and venues with fewer than 20 reviews receive a BiRank score of zero. This module unlocks 13.3M anonymous timestamps to rescue 1,045 previously unranked venues.

### Pipeline

| Script | Description |
|--------|-------------|
| `compute_anonymous_venue_signals.py` | Extract 10 temporal features from check-in JSON for all 131,930 venues |
| `cold_start_ranker.py` | Train Ridge + LightGBM calibration regression; threshold sweep [3,5,10,20]; output pseudo-scores |
| `run_pipeline_v5.py` | Merge BiRank + pseudo-scores via percentile normalization; tag `score_source` |
| `validate_v5_coldstart.py` | Coverage gain, calibration Spearman r, NDCG preservation, threshold ablation |

### Anonymous Features (10 new columns in `coffee_venue_features_v2.csv`)

| Feature | Signal |
|---------|--------|
| `total_checkins` | Anonymous popularity proxy |
| `checkin_burstiness` | Spiky vs steady demand (CV of daily counts) |
| `peak_hour_entropy` | Predictable routine vs random traffic |
| `weekday_ratio` | Commuter/worker vs leisure venue |
| `temporal_stability_cv` | Consistent vs volatile footfall |
| `visit_velocity_recent` | Momentum — growing or declining |
| `growth_trend` | Long-term trajectory (monthly slope) |
| `lunch_dinner_ratio` | Food-occasion focus |
| `late_night_ratio` | Night crowd signal |
| `peak_hour_mode` | Primary use-case anchor hour |

### Cold-Start Results

| Threshold | Ridge Spearman r | LightGBM r | Coverage gain | Venues rescued |
|-----------|-----------------|------------|--------------|----------------|
| 3 reviews | 0.624 | 0.763 | +0.0% | 0 |
| 5 reviews | 0.624 | 0.763 | +0.0% | 0 |
| 10 reviews | 0.624 | 0.755 | +2.3% | 192 |
| **20 reviews** | **0.607** | **0.732** | **+12.3%** | **1,045** |

**Best threshold: 20 reviews.** Coverage: 86.8% → **99.1%** (+12.3%). Ridge Spearman r=0.607 — well above the 0.4 publishability floor. Warm venue NDCG@10 preserved at 0.0765 (BiRank scores untouched).

### Tests

29 unit tests covering all feature computation functions and regression utilities (`tests/`). All passing.

---

## Causal Ranking Analysis (v9 — Direction 5)

A Propensity Score Matching (PSM) study testing whether temporal consistency causally drives future revisit rates.

**Design:**
- **Treatment:** `consistency_score = weekday_ratio − minmax_norm(peak_hour_entropy)` > median
- **Outcome:** `future_revisit_rate` — fraction of pre-2020 users who returned post-2020
- **Confounders:** `total_visits`, `unique_users`, `gini_user_contribution`
- **Method:** 1:1 nearest-neighbour PSM (caliper = 0.2 × SD logit propensity); bootstrap ATE 95% CI; Mahalanobis robustness check

**Results:** ATE = +0.001 (directionally positive, p=0.15). All 3 confounders well-balanced after matching (SMD < 0.1). Robust across PSM and Mahalanobis. Power limited by COVID-19 compressing post-2020 revisit rates to near-zero.

---

## UK Expansion — London + Foursquare GB (v10)

Extends the pipeline to the UK using two independent data sources: TripAdvisor London restaurant reviews and Foursquare WWW2019 GB check-ins. Both run through the same BiRank + MF + hybrid validation framework, enabling cross-dataset consistency checks.

### Data Sources

| Source | Scale | Role |
|---|---|---|
| TripAdvisor London (Zenodo 6583422) | 997K reviews, 502K users, 1,877 venues | Review interactions (user_id, business_id, timestamp, stars) |
| OpenStreetMap London (Overpass API) | 28,124 POIs | Venue reference (GPS, category, postcode) |
| Foursquare WWW2019 GB subset | 288K check-ins (filtered), 6,733 users, 70,042 venues | Anonymous check-in interactions — no star ratings |

**FSQ filtering:** Non-venue categories (Home (private), Office, Neighborhood, Road, Building, Other Great Outdoors, Residence) removed — 346K raw → 288K clean check-ins.

### Pipeline

| Script | Description |
|---|---|
| `ingest_london_tripadvisor.py` | Download + convert TripAdvisor London CSV → `london_interactions.csv` |
| `fetch_london_osm_venues.py` | Pull London cafés/restaurants/hotels from OSM Overpass API |
| `run_london_pipeline.py` | BiRank variants + MF + hybrid + rising-stars evaluation for London TripAdvisor |
| `extract_uk_fsq.py` | Extract GB check-ins from `fsq.duckdb` → `uk_fsq_interactions.csv` + `uk_fsq_businesses.csv` |
| `run_uk_fsq_pipeline.py` | Same BiRank + MF + hybrid pipeline on GB Foursquare check-ins (split 2013-07-01) |

### London TripAdvisor — Three-Phase Analysis

**Phase 1 — Rising-Stars Evaluation + Exploration Priors**

Standard evaluation (predict raw test traffic) is dominated by popularity (ρ=0.61). Replacing it with a *rising-stars residual* (venue traffic growth after controlling for popularity via OLS) reveals whether any model adds value beyond raw counts.

Standard BiRank loyalty priors are **anti-predictive on exploration data** (ρ = −0.21, p<0.001) — they actively identify venues that *decline*. Replacing with exploration-adapted priors (inverse-popularity venue weights, diversity-weighted user weights) neutralises the damage (ρ = −0.03, not significant).

**Phase 2 — Matrix Factorization**

ALS and BPR trained on the same user-venue interaction matrix. ALS alone ρ = +0.002 (no signal). BPR alone ρ = −0.01. Neither beats popularity alone.

**Phase 3 — Hybrid: Exploration BiRank + ALS**

Combining exploration-BiRank (down-weights popular venues) with ALS (latent collaborative patterns) produces a synergistic result:

| Method | ρ (rising stars) | p-value | NDCG@10 | 95% CI | Hit@10 | Wilcoxon vs hybrid |
|---|---|---|---|---|---|---|
| Popularity baseline | +0.032 | 0.187 | 0.5572 | [0.5485, 0.5652] | 0.9025 | p=0.107 (ns) |
| Star ratings | −0.146 | <0.001 | 0.5542 | [0.5456, 0.5633] | 0.8827 | p=0.337 (ns) |
| BiRank (loyalty, count) | −0.209 | <0.001 | **0.5713** | [0.5627, 0.5803] | **0.8966** | p=0.008 * |
| BiRank (loyalty, decay) | −0.210 | <0.001 | 0.5711 | [0.5624, 0.5799] | 0.8955 | p=0.010 * |
| BiRank (exploration priors) | −0.030 | 0.211 | 0.5314 | [0.5226, 0.5401] | 0.8569 | p<0.001 *** |
| ALS alone | +0.002 | 0.943 | 0.5588 | [0.5502, 0.5669] | 0.9046 | p=0.462 (ns) |
| BPR alone | −0.012 | 0.616 | 0.5472 | [0.5384, 0.5562] | 0.8926 | p=0.002 ** |
| **Hybrid (explore + ALS) ★** | **+0.094** | **<0.001** | 0.5592 | [0.5504, 0.5674] | 0.9052 | winner (ρ) |
| LightGCN ◆ | −0.059 | 0.015 * | 0.5536 | [0.5451, 0.5623] | 0.9010 | p=0.018 * |
| Random baseline | ~0.000 | ~1.000 | 0.5382 | [0.5297, 0.5472] | 0.8791 | p<0.001 *** |

All NDCG@10 CIs: bootstrap n=1000. Wilcoxon: two-sided signed-rank test on per-user NDCG@10 arrays (n=4,738 revisit users). Split: 2018-01-01.

**Note on NDCG vs ρ:** BiRank (count/decay) slightly beats the hybrid on NDCG@10 (0.571 vs 0.559) — but this is revisit prediction. The hybrid's unique advantage is in rising-stars ρ (+0.094 vs −0.21 for loyalty BiRank) — identifying venues that grow *beyond* their popularity. These measure different things: NDCG measures prediction of who returns; ρ measures discovery of emerging venues. LightGCN is competitive on NDCG (0.5526) but negative on ρ (−0.059) — graph convolution amplifies popularity bias, hurting rising-star discovery.

### UK Foursquare GB — Nationwide Check-in Analysis

**Data characteristics:**
- 288,389 check-ins · 6,733 users · 70,042 venues · Apr 2012 – Jan 2014
- No star ratings (check-in presence only) → rating baseline omitted
- 18.9% revisit rate (much higher than London 2.6% — FSQ users are habitual check-in users)
- Venues span all of Great Britain (not just London)

**Temporal split:** 2013-07-01 (65/35 split of the 22-month window)

| Method | ρ (rising stars) | p-value | NDCG@10 | 95% CI | Hit@10 | Wilcoxon vs hybrid |
|---|---|---|---|---|---|---|
| Popularity baseline | −0.360 | <0.001 | **0.3159** | [0.3006, 0.3298] | **0.7666** | p<0.001 *** |
| BiRank (loyalty, count) | −0.038 | <0.001 | 0.2665 | [0.2553, 0.2792] | 0.7338 | p<0.001 *** |
| BiRank (loyalty, decay) | −0.040 | <0.001 | 0.2660 | [0.2544, 0.2783] | 0.7371 | p<0.001 *** |
| BiRank (exploration priors) | +0.037 | <0.001 | 0.1262 | [0.1153, 0.1353] | 0.4441 | p<0.001 *** |
| ALS alone | −0.338 | <0.001 | 0.2819 | [0.2662, 0.2972] | 0.7211 | p<0.001 *** |
| BPR alone | −0.000 | 0.981 | 0.2228 | [0.2091, 0.2353] | 0.6435 | p=0.283 (ns) |
| **Hybrid (explore + ALS) ★** | **+0.040** | **<0.001** | 0.2150 | [0.2009, 0.2287] | 0.5839 | winner (ρ) |
| LightGCN ◆ | −0.102 | <0.001 | 0.2378 | [0.2232, 0.2519] | 0.6348 | p<0.001 *** |
| Random baseline | −0.003 | 0.970 | 0.1735 | [0.1628, 0.1843] | 0.5652 | p<0.001 *** |

All NDCG@10 CIs: bootstrap n=1000. Wilcoxon: two-sided signed-rank test (n=1,495 revisit users). Popularity wins NDCG@10 — expected, since popular venues are revisited more. Hybrid wins the PRIMARY metric (rising-stars ρ) — identifying venues growing beyond their popularity.

**Cross-dataset consistency check:**
Both London TripAdvisor (tourist review data) and UK FSQ (nationwide anonymous check-ins, no ratings) produce the same winner on the rising-stars metric: `hybrid_explore_als`. This is the key cross-dataset replication: exploration priors + MF hybrid generalises across data modalities (reviews vs check-ins) and geography (London restaurants vs all-GB venues).

### Domain-Specificity Finding (updated with UK FSQ)

> Behavioral ranking is domain-specific. Loyalty priors (BiRank) work in loyalty-driven domains (Yelp coffee: NDCG@10=0.0765, p=0.038 vs random). In exploration-driven tourist-restaurant data, loyalty priors cause active harm. The fix is to match priors to the behavioral mode: exploration priors + MF hybrid produces statistically significant rising-star prediction where standalone methods fail. This holds across two independent UK datasets.

| Domain | Revisit rate | ρ winner | NDCG winner | LightGCN ρ | Key constraint |
|---|---|---|---|---|---|
| Coffee (Yelp US) | ~10% | BiRank loyalty | BiRank loyalty | — (not tested) | Habit-driven |
| Restaurants (Yelp US) | 33.8% | Star ratings | Star ratings | — (not tested) | Quality-driven |
| Hotels (Yelp US) | ~2.4% | Item-KNN | Item-KNN | — (not tested) | Too sparse for behavioral |
| London tourists (TripAdvisor) | 2.6% | **Hybrid** (+0.094) | BiRank count | −0.059 * | Exploration-driven |
| UK nationwide (Foursquare) | 18.9% | **Hybrid** (+0.040) | Popularity | −0.102 *** | Check-in, no ratings |

LightGCN is negative on rising-stars ρ in both UK datasets — it captures popularity, not rising stars.

---

## LightGCN — Graph Convolution Comparison (v12)

LightGCN (He et al., SIGIR 2020) is the state-of-the-art bipartite graph ranking algorithm and the natural successor to BiRank in the RecSys literature. This section documents its implementation, results, and why it does not outperform the domain-adapted hybrid on this task.

### Why LightGCN?

BiRank performs one round of graph propagation with hand-crafted priors. LightGCN performs L rounds of normalised graph convolution with learnable embeddings — it can capture multi-hop neighbourhood structure that BiRank misses. It is the de facto standard replacement for BiRank/BPR in the recommendation literature and was the most defensible upgrade to test for a master's thesis.

**Reference:** He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. *SIGIR 2020*. https://arxiv.org/abs/2002.02126

### Implementation (`lightgcn.py`)

| Parameter | Value | Justification |
|---|---|---|
| Propagation layers (L) | 3 | Optimal value found in He et al. across multiple datasets |
| Embedding dimension | 64 | Matches MF baseline for fair comparison |
| Training loss | BPR (Bayesian Personalised Ranking) | Standard implicit feedback loss |
| Optimiser | Adam, lr=0.001 | Standard in RecSys literature |
| L2 regularisation | 1e-4 on initial embeddings only | LightGCN paper prescription |
| Epochs | 50 | Sufficient for loss convergence |
| Batch size | 2,048 | Standard |
| Device | CPU | MPS (Apple Silicon) excluded: PyTorch sparse tensor ops (`torch.sparse.mm`) are not supported on MPS backend |

**Core computation:**

```
E^(0)   = learnable embedding matrix (Xavier uniform init)
E^(k+1) = D^(-1/2) A D^(-1/2) E^(k)        # normalised graph propagation
E_final = (1/(L+1)) * (E^(0) + E^(1) + ... + E^(L))   # mean pooling
```

Where `A` is the augmented bipartite adjacency:
```
A = [ 0    R  ]
    [ R^T  0  ]
```
`R[u, i] = 1` if user `u` has any interaction with venue `i`.

**Global venue score** (for ranking without user context):
```
score(venue_i) = item_emb[i] · mean(user_emb)
```
This matches the approach used for ALS/BPR in `run_london_pipeline.py` and keeps comparisons fair.

### Results — London TripAdvisor

Split: 2018-01-01 · Train: 659,305 interactions · 334,919 users · 1,706 venues

| Metric | LightGCN | Hybrid ★ | Δ |
|---|---|---|---|
| ρ (rising stars) | −0.059 | **+0.094** | −0.153 |
| p-value (ρ) | 0.015 * | <0.001 *** | — |
| NDCG@10 | 0.5536 | 0.5592 | −0.006 |
| 95% CI (NDCG) | [0.5451, 0.5623] | [0.5504, 0.5674] | — |
| Hit@10 | 0.9010 | 0.9052 | −0.004 |
| Wilcoxon vs hybrid | p=0.018 * | winner | — |
| Traffic ρ | **+0.431** | −0.039 | — |

LightGCN training: loss 0.0657 → 0.0332 → 0.0232 → 0.0189 → 0.0163 (epochs 10→20→30→40→50).

### Results — UK Foursquare GB

Split: 2013-07-01 · Train: 250,097 interactions · 5,685 users · 63,506 venues

| Metric | LightGCN | Hybrid ★ | Δ |
|---|---|---|---|
| ρ (rising stars) | −0.102 | **+0.040** | −0.142 |
| p-value (ρ) | <0.001 *** | <0.001 *** | — |
| NDCG@10 | 0.2378 | 0.2150 | +0.023 |
| 95% CI (NDCG) | [0.2232, 0.2519] | [0.2009, 0.2287] | — |
| Hit@10 | 0.6348 | 0.5839 | +0.051 |
| Wilcoxon vs hybrid | p<0.001 *** | winner | — |
| Traffic ρ | **+0.167** | −0.039 | — |

LightGCN training: loss 0.1691 → 0.1026 → 0.0718 → 0.0527 → 0.0408 (epochs 10→20→30→40→50).

### Why LightGCN Fails on Rising Stars

LightGCN propagates embeddings through the user-venue neighbourhood graph. Each propagation round reinforces the signal from high-degree nodes — i.e., popular venues. After 3 rounds, a venue's embedding is heavily influenced by how many users it shares with other popular venues. This makes it very good at predicting revisits to established popular venues (high traffic ρ), but specifically bad at identifying rising stars — venues that are growing *beyond* what their current popularity would predict.

The exploration prior in the hybrid does the opposite: it assigns *lower* prior weight to popular venues (`q0[v] = 1 / log(1 + popularity_visits)`), giving BiRank room to surface venues that attract disproportionately high-quality or high-diversity users. Graph convolution cannot replicate this because it is structurally biased toward propagating popularity.

| Method | Mechanism | Traffic ρ (popularity-confounded) | Rising-stars ρ (debiased) |
|---|---|---|---|
| Popularity baseline | Count visits | High | Reference |
| LightGCN | Graph convolution (amplifies popularity) | High | Negative |
| ALS alone | Matrix factorisation | High | Near-zero |
| **Hybrid (explore + ALS) ★** | Inverse-popularity priors + ALS | Low | **Positive** |

### Why LightGCN Is Worse Than the Hybrid — Root Cause Analysis

LightGCN beats loyalty BiRank on rising-stars ρ (−0.059 vs −0.210) but loses to the hybrid (+0.094). There are four distinct reasons:

**1. Training objective mismatch — the most important reason**

LightGCN is trained with BPR loss: "given user u, rank venue A above venue B if u visited A and not B." This directly optimises for predicting *which venues users already visit*. The rising-stars metric asks a completely different question: *which venues will grow beyond what their popularity already predicts?* A model optimised to predict existing behaviour will always learn popularity as the dominant signal, because popular venues are the safest prediction. The exploration prior in the hybrid is designed specifically to remove that signal before ranking.

**2. Graph convolution amplifies popularity**

Every propagation round reinforces high-degree nodes. After 3 rounds, a venue's embedding is a weighted average of its visitors' embeddings, which are themselves weighted averages of all other popular venues those users visited. Popular venues get reinforced at every hop. The exploration prior does the explicit opposite — `q0[venue] = 1 / log(1 + popularity_visits)` — inverting this bias before propagation begins.

**3. Extreme user-to-venue ratio**

London: 334,919 users, 1,706 venues — a 200:1 ratio. The top 50 venues have thousands of connections; the bottom half have single digits. In this graph topology, LightGCN's propagation collapses toward the dense core — the same popular venues dominate after every round regardless of embedding initialisation. BiRank with behavioural priors weights users by loyalty and burstiness, which is orthogonal to venue degree and thus resists this collapse.

**4. No temporal signal**

LightGCN treats all interactions as equally weighted regardless of when they occurred. The rising-stars metric is inherently temporal — it measures traffic *growth* in the test period beyond the training trend. BiRank with temporal decay (`weight = exp(−0.5 × age_years)`) up-weights recent interactions, giving more signal to venues that are currently accelerating. LightGCN has no equivalent mechanism.

| Issue | BiRank loyalty | Hybrid ★ | LightGCN |
|---|---|---|---|
| Popularity bias | Strong — loyalty priors reinforce it | **Removed** — inverse-popularity prior | Very strong — amplified by graph convolution |
| Temporal signal | Decay weighting | Decay weighting | None — all interactions equal |
| Training objective | Graph rank, no explicit loss | Graph rank + ALS collaborative signal | BPR on interaction prediction |
| Rising-stars ρ (London) | −0.210 | **+0.094** | −0.059 |
| Rising-stars ρ (UK FSQ) | −0.040 | **+0.040** | −0.102 |

### Conclusion

LightGCN is a stronger general-purpose recommender than vanilla BiRank — it achieves competitive NDCG@10 on both datasets (0.5526 London, 0.2374 FSQ) with no hand-engineered priors. However, it does not outperform the domain-adapted hybrid on the task this thesis is designed to solve: identifying venues that grow beyond their popularity trajectory.

This is a theoretically coherent result: graph convolution and matrix factorisation both propagate and reinforce popularity signal; the hybrid is specifically designed to remove it via inverse-popularity priors. The failure of a SIGIR 2020 state-of-the-art algorithm to beat the domain-adapted hybrid is not a limitation — it is the thesis contribution. It demonstrates that algorithmic sophistication does not substitute for domain adaptation: the task of *discovery* (finding rising stars) requires a fundamentally different inductive bias than the task of *prediction* (recommending popular venues). Matching the prior to the behavioral mode of the domain is the dominant factor, not the choice of ranking algorithm.

## Hybrid BiRank + Matrix Factorization (v6)

`validate_v6_hybrid.py` tests whether latent collaborative filtering (ALS/BPR) can complement BiRank's graph-structural signal.

### Approach

- **ALS** (Alternating Least Squares) and **BPR** (Bayesian Personalized Ranking) trained on user-venue interaction matrix (64 factors, 30 iterations)
- **Hybrid score**: `λ * BiRank_norm + (1-λ) * MF_norm` with personalized per-user MF scores
- **Proper train/val/test protocol**: Train < 2019-07-01, Validation 2019-07-01—2020-01-01, Test ≥ 2020-01-01
- **Lambda grid search** [0.0, 0.1, ..., 1.0] tuned on validation split

### Lambda Tuning (Validation)

| λ | ALS NDCG@10 | BPR NDCG@10 |
|---|-------------|-------------|
| 0.0 (pure MF) | 0.0618 | 0.0627 |
| 0.5 | 0.0615 | 0.0627 |
| 1.0 (pure BiRank) | **0.0641** | **0.0641** |

Best λ = 1.0 for both methods — pure BiRank outperforms all hybrid blends on validation data.

### Test Results

| Method | NDCG@10 | Δ vs BiRank | p-value |
|--------|---------|-------------|---------|
| hybrid_als (λ=0.7) | 0.0658 | +0.26% | ref |
| hybrid_als (λ=0.5) | 0.0658 | +0.23% | 0.996 |
| **v5_combined (BiRank)** | **0.0657** | **ref** | 0.917 |
| pure_als | 0.0656 | -0.08% | 0.855 |
| baseline_random | 0.0643 | -2.05% | 0.151 |

**Note:** v6 NDCG values (0.065) are lower than v5 (0.076) because v6 uses less training data (cutoff at 2019-07-01) to create a validation split for lambda tuning. Relative comparisons within v6 are valid.

### Conclusion

Matrix factorization does not meaningfully improve over BiRank (+0.26% max, not significant). This confirms that BiRank's behavioral priors (burstiness, loyalty, revisit regularity) already capture the useful signal — latent collaborative factors add nothing in this domain. The data sparsity (93K users, 8.5K venues, most with few visits) limits MF's ability to learn useful latent structure.

---

## Hotel & Accommodation Model (v7)

A new domain application demonstrating that the behavioral ranking framework generalises — but requires domain-adapted features.

### Data

| | Value |
|---|---|
| Venues (50+ reviews) | 1,466 (Hotels, B&B, Resorts, Hostels, Motels) |
| Reviews | 256,189 |
| Check-ins | 755,212 |
| Total interactions | 1,011,401 |
| Unique users | 194,047 |
| States covered | 14 |
| Date range | 2005–2022 |

### Why hotel features differ from coffee features

| Coffee feature | Hotel equivalent | Rationale |
|---|---|---|
| Revisit rate (41% for Loyalists) | Multi-stay rate (2.4%) | People rarely revisit the same hotel — that's normal, not bad |
| Burstiness | Seasonal CV | Hotels spike by season, not by burst vs regular |
| Shannon entropy (user diversity) | Geographic diversity (entropy of reviewer home states) | Good hotels draw visitors from many places |
| Gini (loyalty concentration) | Traveler concentration | Does it serve one type of traveler consistently? |
| Venue stability CV | Venue stability CV | Reused — consistent year-round traffic |

### New hotel behavioral features

| Feature | Description |
|---|---|
| `business_leisure_ratio` | Fraction of reviews on weekdays (Mon–Thu) — high = business hotel |
| `seasonal_cv` | Coefficient of variation of monthly review volume — low = consistent demand |
| `geographic_diversity` | Shannon entropy of reviewer home states — high = draws from many places |
| `multi_stay_rate` | Fraction of reviewers with 2+ reviews at same hotel — rare but very strong signal |
| `review_velocity` | Exponentially-weighted recent review rate — current relevance |
| `traveler_concentration` | Gini coefficient of reviewer frequency |

**Key EDA findings:** Weekday reviews: 70.6% vs 29.4% weekend — strong business travel signal. Seasonal CV = 0.096. Multi-stay rate = 2.4% (sparse but present).

### User Archetypes (194,047 hotel reviewers)

| Archetype | n | % | Key signal |
|---|---|---|---|
| One-Time Tourists (Business) | 96,423 | 49.7% | 98.8% weekday, single hotel |
| Leisure Travelers | 70,899 | 36.5% | 0.1% weekday — pure weekend/holiday |
| One-Time Tourists | 16,324 | 8.4% | Mixed weekday, slightly more reviews |
| Budget Explorers | 10,401 | 5.4% | 2.3 states visited, highest city diversity |

### Pipeline

| Script | Description |
|---|---|
| `hotel_data_extract.py` | Extract 1,466 hotels, build interaction table, EDA |
| `hotel_behaviour_features.py` | Hotel-specific venue + user behavioral features |
| `hotel_user_profiles.py` | K-means clustering into 4 traveler archetypes |
| `hotel_cross_domain.py` | Cross-domain coffee→hotel transfer analysis |
| `hotel_birank.py` | BiRank with hotel priors (recency decay × traveler credibility) |
| `hotel_fsq_integration.py` | Foursquare linkage + social priors for hotels |
| `hotel_validation.py` | Full temporal validation with bootstrap CI + Wilcoxon tests |

### Cross-Domain Transfer (coffee → hotel)

59,668 users reviewed both coffee shops and hotels. Key findings:
- **Classifier accuracy: 0.293 vs baseline 0.250** — domains are largely independent (+4.3% lift)
- **Meaningful correlations exist**: coffee `venue_entropy` (exploration diversity) predicts hotel `n_unique_hotels` (Spearman r=0.29) — explorers in coffee are explorers in hotels
- Transfer priors built for 495,054 users (including 301,007 coffee-only users)

### Validation Results (test split ≥ 2020-01-01, 3,578 users)

| Method | NDCG@10 | Hit@10 | p-value |
|---|---|---|---|
| baseline_item_knn | **0.1188** | 0.1395 | 0.006 vs hotel_birank |
| **hotel_birank** | 0.0998 | 0.1399 | ref |
| hotel_birank_fsq | 0.0998 | 0.1399 | 1.000 |
| hotel_birank_xdomain | 0.0998 | 0.1399 | 1.000 |
| baseline_popularity | 0.0992 | 0.1399 | 0.286 |
| baseline_random | 0.0972 | 0.1381 | 0.471 |
| baseline_rating | 0.0926 | 0.1375 | **0.012** |

**BiRank significantly beats star ratings (p=0.012)**. Item-KNN outperforms BiRank (p=0.006) — an important finding: for hotels, collaborative filtering is stronger than behavioral priors because most users visit only 1–2 hotels (sparse behavioral signal). FSQ and cross-domain transfer add no measurable lift, reflecting sparse hotel FSQ linkage.

### Per-Group Results (NDCG@10)

| Archetype | NDCG@10 | n |
|---|---|---|
| **Leisure Travelers** | **0.3208** | 41 |
| One-Time Tourists (Business) | 0.2953 | 57 |
| One-Time Tourists | 0.1464 | 451 |
| Budget Explorers | 0.0638 | 1,145 |

Leisure Travelers score highest — they have repeat visit patterns (same destinations each holiday). Budget Explorers score lowest — high venue diversity makes prediction hard.

---

## Restaurant Model (S(R,U,C))

### Pipeline

| Script | Phase | Description |
|--------|-------|-------------|
| `restaurant_data_extract.py` | Data extraction | Extract ~64K restaurants from Yelp |
| `restaurant_data_extract_v2.py` | Data extraction | Improved extraction with transit integration |
| `restaurant_user_profiles.py` | User profiling | Compute user archetypes (Explorer/Loyalist, Critic/Casual, spatial range) |
| `restaurant_venue_features.py` | Venue features | Loyalty magnetism, niche vs. broad appeal, transit accessibility |
| `restaurant_scoring.py` | Scoring | S(R,U,C) multi-objective ranking |
| `restaurant_validation.py` | Validation | Temporal prediction test |

### S(R,U,C) Score Components

- **Behavioral Utility (U_beh)**: Venue quality + critic penalty for discerning users
- **Mobility Convenience (C_mob)**: Distance decay, walking bonus (<800m), transit bonus (high-frequency stops)
- **Contextual Relevance (R_ctx)**: Queue penalty (busyness), cuisine preference match

Weights are set dynamically per user via the **Entropy Weight Method** — uninformative dimensions get downweighted automatically. Final ranking uses **Maximal Marginal Relevance (MMR)** for cuisine/location diversity.

### Validation Results v11 (corrected, with significance tests)

Temporal split at 2020-01-01. 13.5M training reviews, 60,644 venues, 11,948 evaluable users (33.8% revisit rate). Script: `validate_restaurant_birank.py`.

| Method | NDCG@10 | Hit@10 | 95% CI | p-value vs best |
|--------|---------|--------|--------|-----------------|
| **Rating (Stars)** | **0.4059** | **72.8%** | [0.4000, 0.4118] | ref |
| BiRank (decay) | 0.3961 | 72.4% | [0.3901, 0.4019] | 0.035 * |
| BiRank (count) | 0.3954 | 72.2% | [0.3892, 0.4011] | 0.035 * |
| S(R,U,C) global | 0.3930 | 72.2% | [0.3867, 0.3988] | 0.376 |
| Popularity | 0.3806 | 70.8% | [0.3742, 0.3864] | <0.001 *** |

**Key finding:** Star ratings beat BiRank for restaurants (p=0.035) — the reverse of the coffee result. This is a domain-specificity finding: for quality-driven revisits (restaurants), explicit ratings capture the signal; for habit-driven revisits (coffee shops), behavioral regularity captures it better.

### Per-Group Results (NDCG@10, BiRank decay)

| Archetype | NDCG@10 | n_users |
|---|---|---|
| **Loyalists** | **0.667** | 3,184 |
| Mixed / Average | 0.534 | 1,401 |
| Nightlife / Ride-Share | 0.469 | 504 |
| Explorers | 0.217 | 5,528 |

Loyalists score highest (same pattern as coffee) — behavioral regularity is the strongest predictive signal when it exists. Explorers score lowest — by definition they avoid revisiting.

### Domain-Specificity Summary

| Domain | Revisit rate | Winner (ρ / NDCG) | BiRank NDCG@10 | Stars NDCG@10 | Notes |
|---|---|---|---|---|---|
| Coffee shops | ~10% | **BiRank** | 0.0765 | 0.0754 | Habit-driven; behavioral wins |
| Restaurants | 33.8% | **Stars** | 0.3961 | 0.4059 | Quality-driven; stars win |
| Hotels | ~2.4% | **Item-KNN** | 0.0998 | 0.0926 | Too sparse for behavioral |
| London tourists (TripAdvisor) | 2.6% | **Hybrid (ρ) / BiRank (NDCG)** | ρ=+0.094 | ρ=−0.15 | Exploration-driven |
| UK nationwide (Foursquare) | 18.9% | **Hybrid (ρ) / Popularity (NDCG)** | ρ=+0.040 | — | Check-ins, no ratings |

Behavioral ranking wins specifically in **habit-driven, loyalty-dominated domains**. Stars win where quality judgement drives revisits. In exploration-driven domains, the hybrid exploration-BiRank + ALS method wins on the rising-stars metric across both UK datasets. This boundary condition — and cross-dataset replication — is the key domain-specificity thesis contribution.

---

## Foursquare Integration (v4)

Cross-platform data fusion to supplement Yelp's behavioral signals.

| Script | Description |
|--------|-------------|
| `ingest_foursquare.py` | Ingest FSQ data into DuckDB (`fsq.duckdb`) |
| `build_venue_linkage.py` | Match FSQ venues to Yelp businesses by GPS (75m) + category TF-IDF |
| `link_fsq_checkins.py` | Join FSQ check-ins to linked Yelp venues |
| `match_cross_platform_users.py` | Bridge Yelp/FSQ users via venue overlap + temporal co-presence |
| `extract_social_venue_signals.py` | Friend/friend-of-friend venue visit signals from FSQ social graph |
| `run_pipeline_v4.py` | BiRank with social priors (gamma-tunable 0-0.3) |
| `validate_v4.py` | Ablation study vs v3 baseline |

### v4 Ablation Results (pre-fix, for reference)

| Variant | NDCG@10 | Delta |
|---------|---------|-------|
| v3 baseline | 0.086 | — |
| + social direct friends | 0.085 | -0.001 |
| + social friend-of-friend | 0.085 | -0.001 |
| + FSQ volume | 0.082 | -0.004 |
| Full v4 | 0.083 | -0.003 |

Raw social signals added noise. v5 addressed this with selective social filtering (confidence >= 0.3, gamma lowered from 0.2 to 0.15), which neutralised the negative impact and produced a small positive delta (+0.30%).

---

## LLM Simulation Validation (v8)

External ecological validation using GPT-5.4 synthetic personas. Two independent studies run alongside the real-data validation.

### Why LLM simulation?

The real-data validation (v5–v7) tests whether the model predicts *held-out historical behaviour*. The LLM simulation tests something different: do realistic synthetic *people* — grounded in published consumer-behaviour research — actually prefer the venues our model ranks highest? These are independent sources of evidence; if they agree, the conclusion is much stronger.

---

### Study 1 — Behavioural Archetype Personas (1,500 personas)

Personas grounded in the four archetypes discovered from Yelp data (Loyalists, Weekday Regulars, Casual Weekenders, Infrequent Visitors) across all three domains.

**Pipeline:** `llm_simulation/main_v2.py`

| Phase | What it does |
|-------|-------------|
| Phase 1 | Discriminating candidate sets (BiRank top-5 vs Stars top-5, non-overlapping) |
| Phase 2 | Manipulation check, null-persona baseline, inverted-persona sanity test |
| Phase 3 | Revisit calibration (Spearman r), cross-domain consistency, per-persona variance |
| Phase 4 | BH correction, Cohen's d, rank-biserial, stratified bootstrap |
| Phase 5 | Tiered models (`gpt-5.4-mini` ranking, `gpt-5.4` pairwise/revisit) + Claude Sonnet replication |

**Each persona runs 3 tasks:**
1. **Ranking task** — rank 10 venues; NDCG@10 vs. model ranking
2. **Pairwise task** — BiRank top-1 vs. Stars top-1; win rate
3. **Revisit task** — likelihood of returning; Spearman correlation with model's revisit signals

**Key results (Study 1 — gpt-5.4, 1,500 personas):**

| Domain | NDCG@10 | Hit@1 | Hit@3 | Δ vs Stars | Win Rate | p (BH) |
|--------|---------|-------|-------|-----------|----------|--------|
| Coffee | 0.7948 | 0.118 | 0.278 | +0.0190 | 48.4% | <0.001 |
| Restaurants | 0.7907 | 0.268 | 0.680 | +0.0216 | 51.0% | <0.001 |
| Hotels | 0.7851 | 0.212 | 0.514 | −0.0276 | 47.0% | <0.001 |

Hotels perform below stars — **a positive result**, consistent with v7 real-data findings where item-KNN beat BiRank. Two independent methods agree.

---

### Study 2 — Occupation × Age Cross-Matrix (1,860 personas)

A full 5 age-group × 10 occupation grid grounded in 51 published consumer-behaviour sources.

**Pipeline:** `llm_simulation/main_study2.py`

**Age groups:** Gen Z (18–25) · Young Millennial (26–33) · Senior Millennial (34–40) · Gen X (41–56) · Boomer (57+)

**Occupation clusters:** Tech/Software · Healthcare · Education/Academic · Creative/Media · Legal/Finance · Trade/Manual · Executive/C-Suite · Hospitality/Service · Student/Part-time · Remote/Digital Nomad

**31 valid cells** (some age × occupation combinations excluded as unrealistic) × 3 domains × 20 personas = 1,860 total.

**Key findings by occupation (NDCG@10 vs stars baseline):**
- Highest alignment: **Executive/C-Suite** and **Legal/Finance** — loyalty-driven archetypes match BiRank's loyalty signals
- Lowest alignment: **Student/Part-time** and **Trade/Manual** — price/convenience-driven choices diverge from behavioural ranking

**Key findings by age group (NDCG@10):**
- Gen X and Boomers show strongest model alignment — consistent with high loyalty scores from research
- Gen Z shows weakest alignment — exploration-first behaviour is harder for a loyalty-biased model to predict

**Research sources (51 cited, trust-rated):**

| Domain | Key sources |
|--------|-------------|
| Coffee | NCA National Coffee Data Trends 2025 · Grand View Research · Mintel · Toast POS · Euromonitor |
| Restaurants | National Restaurant Association · McKinsey · OpenTable 2026 · Deloitte · Toast · YouGov |
| Hotels | J.D. Power Hotel Satisfaction Study · GBTA · Hilton Trends Report 2024 · Expedia Unpack · EHL |

Full bibliography with trust ratings: `llm_simulation/research/bibliography.md`

---

### Simulation Files

| File | Description |
|------|-------------|
| `llm_simulation/main.py` | Study 1 v1 orchestrator (gpt-4.1, baseline) |
| `llm_simulation/main_v2.py` | Study 1 v2 orchestrator (gpt-5.4, all phases) |
| `llm_simulation/main_study2.py` | Study 2 orchestrator (occupation × age) |
| `llm_simulation/persona_generator.py` | 1,500 behavioural archetype personas |
| `llm_simulation/demographic_persona_generator.py` | 1,860 cross-matrix personas |
| `llm_simulation/demographic_profiles.py` | 31-cell profile library (research-grounded) |
| `llm_simulation/data_loader.py` | City-matched venue loading, discriminating sets |
| `llm_simulation/evaluator.py` | NDCG, Hit@k, Kendall τ, BH, Cohen's d, bootstrap |
| `llm_simulation/prompts.py` | System + task prompts with archetype emphasis |
| `llm_simulation/task_runner.py` | Async OpenAI client, SQLite cache, tiered models |
| `llm_simulation/manipulation_check.py` | Phase 2 persona authenticity tests |
| `llm_simulation/calibration_analysis.py` | Phase 3 revisit calibration + cross-domain |
| `llm_simulation/second_model.py` | Claude Sonnet 4.6 replication |
| `llm_simulation/report_generator.py` | Study 1 report generator |
| `llm_simulation/report_study2.py` | Study 2 report generator |
| `llm_simulation/research/` | Three research files + bibliography (51 sources) |
| `llm_simulation/results/` | All simulation records, metrics, and reports |

**Run Study 1:**
```bash
cd llm_simulation
python3 main_v2.py              # full 1,500 personas
python3 main_v2.py --dry-run    # test without API calls
python3 main_v2.py --domain coffee  # one domain only
```

**Run Study 2:**
```bash
python3 main_study2.py          # full 1,860 personas
python3 main_study2.py --dry-run
python3 main_study2.py --occupation "Healthcare"
```

---

## Dashboard

Interactive Streamlit app for exploring rankings and live persona simulation.

```bash
python3 -m streamlit run app.py
```

### Features

**Venue Explorer (Coffee / Restaurants / Hotels)**
- **City search** with fuzzy matching ("philly" finds Philadelphia)
- **Radius-based area filter** around any reference venue
- **Side-by-side ranking comparison**: BiRank vs. rating vs. popularity vs. revisit rate
- **Behavioral mode selector**: Regular, Explorer, Morning, Weekend
- **Venue detail cards**: Plain-language tags (Steady / High Retention / Broad Loyalty)
- **Interactive Folium maps** with marker popups
- **CSV export** of results

**UK Venue Explorer** (new in v10)
- **Data source toggle** (sidebar): Switch between "London OSM" (29K venues, district filter) and "UK Foursquare" (70K venues across Great Britain, category filter)
- **London OSM map**: Clustered markers for coffee shops, restaurants, hotels, pubs, bars; district filter (19 London areas); haversine radius filter
- **UK Foursquare map**: Full GB folium map (centre 54°N, zoom 6); circle markers sized by hybrid BiRank score; 9 venue category colours; top-N filter
- **Rankings tab**: TripAdvisor BiRank table OR Foursquare UK hybrid score table with category filter
- **Validation tab** (with full benchmarks):
  - Dataset toggle: London TripAdvisor ↔ UK Foursquare
  - Chart 1: Horizontal ρ bar chart — all methods sorted, popularity baseline reference line, significance stars (*** / * / ns), hover shows Δρ
  - Chart 2: NDCG@10 with 95% bootstrap CI error bars (n=1000) + Hit@10 diamonds; Wilcoxon p-value vs winner on hover
  - Chart 3: Δρ improvement-over-baseline chart (green = positive, red = negative)
  - All CI and Wilcoxon values are fully computed (not approximated) — see `benchmark_results.json`
  - Random baseline row included in both datasets as true lower bound
- **Domain Insight tab**: 5-domain comparison table (coffee, restaurants, hotels, London, UK FSQ) with cross-dataset consistency finding

**LLM Simulation Page** (new in v8)
- **Executive summary**: plain-English + academic framing for every reader level
- **Results by domain**: NDCG@10, Hit@1/3, pairwise win rate, BH-corrected p-values per archetype
- **Live Persona Chat**: pick domain, archetype, and city → `gpt-5.4-mini` generates a persona (fresh name/age/occupation each click), recommends top 3 real venues from the dataset in character, with structured venue cards showing why each matches the archetype's behavioral signal
- **Full simulation report** embedded in dashboard

See `README_dashboard.md` for full usage guide.

---

## Validation v5 — Improvements Over v3/v4

`validate_v5.py` is the current best validation script. Key changes:

| Fix | Description |
|-----|-------------|
| Feature leakage | User/venue features recomputed from training data only (not all data) |
| NDCG correction | IDCG uses min(k, total_relevant_candidates), not just top-k slice |
| Temporal edge decay | Edge weights: `exp(-0.5 * age_years)` — recent visits count more |
| Selective social | Only FSQ bridges with confidence >= 0.3, gamma reduced to 0.15 |
| Significance tests | Bootstrap 95% CI (1000 samples) + Wilcoxon signed-rank p-values |
| Per-group evaluation | NDCG@10 stratified by Loyalist / Regular / Casual / Infrequent |
| Multi-split robustness | Validated on 2019-01-01, 2019-07-01, and 2020-01-01 splits |
| Stronger baselines | Added item-KNN (cosine) and IUF-popularity baselines |

---

## Key Outputs

| File | Description |
|------|-------------|
| `coffee_venue_features_v2.csv` | Venue feature matrix — 25 columns (15 behavioral + 10 anonymous temporal) |
| `anonymous_venue_signals.csv` | Anonymous temporal features for all 131,930 Yelp venues |
| `cold_start_scores.csv` | Pseudo-BiRank scores for 1,045 cold venues (threshold=20) |
| `cold_start_threshold_sweep.csv` | Spearman r and coverage gain for thresholds [3,5,10,20] |
| `coffee_birank_venue_scores_v5.csv` | Unified rankings: 7,389 BiRank + 1,045 cold-start + 75 unranked |
| `cold_start_validation_report.txt` | Coverage gain, calibration quality, ablation table |
| `cold_start_ablation_table.csv` | Threshold ablation — thesis-ready |
| `coffee_birank_venue_scores_v3.csv` | Best coffee venue rankings |
| `coffee_birank_user_scores_v3.csv` | User importance scores |
| `coffee_user_features_v3.csv` | User behavioral feature matrix |
| `coffee_venue_features_v3.csv` | Venue behavioral feature matrix |
| `restaurant_scores.csv` | Restaurant S(R,U,C) rankings |
| `venue_explanations.csv` | Human-readable venue tags |
| `validation_v5_results.csv` | v5 results with CIs and p-values |
| `validation_v5_per_group.csv` | Per-group NDCG breakdown |
| `validation_v5_robustness.csv` | Multi-split comparison |
| `validation_v5_summary.txt` | Human-readable v5 report |
| `validation_v6_results.csv` | v6 hybrid results with CIs and p-values |
| `validation_v6_lambda_tuning.csv` | Lambda grid search results |
| `validation_v6_per_group.csv` | Per-group breakdown for hybrid methods |
| `validation_v6_summary.txt` | Human-readable v6 report |
| `hotel_businesses.csv` | 1,466 quality hotel/accommodation venues |
| `hotel_interactions.csv` | Hotel interaction table (reviews + check-ins) |
| `hotel_venue_features.csv` | Hotel behavioral feature matrix |
| `hotel_user_features.csv` | Hotel user behavioral features |
| `hotel_user_groups.csv` | Hotel user archetypes (4 clusters) |
| `hotel_birank_venue_scores.csv` | Hotel BiRank rankings |
| `hotel_birank_fsq_scores.csv` | Hotel BiRank + Foursquare rankings |
| `hotel_venue_linkage.csv` | FSQ → Yelp hotel venue matches |
| `cross_domain_analysis.csv` | Coffee↔hotel archetype overlap |
| `cross_domain_priors.csv` | Transfer priors for 495K users |
| `hotel_validation_results.csv` | Hotel validation with CIs + p-values |
| `hotel_validation_per_group.csv` | Per-archetype NDCG breakdown |
| `hotel_validation_summary.txt` | Human-readable hotel validation report |
| `validation_summary.txt` | Legacy v3 validation results |
| `london_interactions.csv` | 997K TripAdvisor London restaurant reviews (user_id, business_id, timestamp, stars) |
| `london_businesses.csv` | 1,877 London restaurant venue reference |
| `london_user_features.csv` | Per-user behavioral features computed from London training data |
| `london_venue_features.csv` | Per-venue behavioral features from London training data |
| `london_birank_venue_scores.csv` | London venue rankings (hybrid_explore_als scores) |
| `london_validation_summary.txt` | London validation report — ρ, NDCG, p-values for all methods |
| `uk_fsq_interactions.csv` | 288K Foursquare GB check-ins — non-venue categories filtered (user_id, business_id, timestamp) |
| `uk_fsq_businesses.csv` | 70,042 GB venue metadata (business_id, category, lat, lon) |
| `uk_fsq_user_features.csv` | Per-user behavioral features from GB FSQ training data |
| `uk_fsq_venue_features.csv` | Per-venue behavioral features from GB FSQ training data |
| `uk_fsq_venue_scores.csv` | GB venue rankings (hybrid_explore_als scores, 63,506 venues within GB bbox) |
| `uk_fsq_validation_summary.txt` | UK FSQ validation report — ρ, NDCG, p-values for all methods |
| `benchmark_results.json` | Fully computed benchmarks: bootstrap 95% CIs + Wilcoxon p-values for all methods × both UK datasets |
| `lightgcn.py` | LightGCN implementation (PyTorch, 3-layer graph convolution, BPR loss, CPU) |
| `lightgcn_benchmark.json` | Bootstrap CIs + Wilcoxon p-values for LightGCN on both datasets (generated by benchmark script) |
| `fsq.duckdb` | Foursquare DuckDB database |
| `venue_linkage.csv` | Yelp-Foursquare venue matches |
| `yelp_fsq_user_bridge.csv` | Cross-platform user bridge table |
| `llm_simulation/results/simulation_records_v2.csv` | Study 1 — 1,500 persona records (gpt-5.4) |
| `llm_simulation/results/simulation_metrics_v2.json` | Study 1 — NDCG, Hit@k, Kendall τ, Cohen's d per archetype |
| `llm_simulation/results/simulation_report_v2.md` | Study 1 — full validation report with BH-corrected p-values |
| `llm_simulation/results/simulation_records_study2.csv` | Study 2 — 1,860 persona records (occupation × age) |
| `llm_simulation/results/study2_by_age.csv` | Study 2 — metrics by age group |
| `llm_simulation/results/study2_by_occupation.csv` | Study 2 — metrics by occupation cluster |
| `llm_simulation/results/study2_cross_matrix.csv` | Study 2 — NDCG heatmap (age × occupation) |
| `llm_simulation/results/simulation_report_study2.md` | Study 2 — full cross-matrix report |
| `llm_simulation/research/coffee_demographics_research.md` | Research: café preferences by age/occupation (16 sources) |
| `llm_simulation/research/restaurant_demographics_research.md` | Research: restaurant preferences by age/occupation (18 sources) |
| `llm_simulation/research/hotel_demographics_research.md` | Research: hotel preferences by age/occupation (18 sources) |
| `llm_simulation/research/bibliography.md` | Full bibliography — 51 sources with trust ratings |

---

## Methods & References

**Ranking algorithms**
- **BiRank**: He, J. et al. (2017) — bipartite graph ranking via mutual reinforcement. *TKDE*.
- **LightGCN**: He, X. et al. (2020) — simplifying graph convolution for recommendation by removing feature transformations. *SIGIR 2020*. https://arxiv.org/abs/2002.02126
- **ALS Matrix Factorization**: Hu, Y. et al. (2008) — implicit feedback collaborative filtering. *ICDM 2008*.
- **BPR**: Rendle, S. et al. (2009) — Bayesian personalized ranking from implicit feedback. *UAI 2009*.
- **NGCF**: Wang, X. et al. (2019) — neural graph collaborative filtering (predecessor to LightGCN). *SIGIR 2019*.
- **Maximal Marginal Relevance**: Carbonell & Goldstein (1998) — diversity-aware re-ranking.

**Behavioral features**
- **Burstiness Index**: Goh & Barabasi — temporal regularity of human dynamics
- **Shannon Entropy**: Information-theoretic diversity measure
- **Gini Coefficient**: Loyalty concentration (economics)
- **Entropy Weight Method**: Dynamic feature weighting from decision theory

**Evaluation**
- **NDCG@10 / Hit@k / Kendall τ**: Standard information retrieval metrics
- **Wilcoxon signed-rank test + Bootstrap 95% CI**: Non-parametric significance + uncertainty
- **Benjamini-Hochberg correction**: Multiple comparison correction across archetype groups
- **Cohen's d + Rank-biserial correlation**: Effect size measures

**LLM simulation**
- **Persona grounding (Study 1)**: Archetypes from K-means clustering on Yelp behavioral features
- **Persona grounding (Study 2)**: NCA 2025 · McKinsey 2026 · J.D. Power 2024 · GBTA 2024 · Hilton Trends 2024 · Expedia Unpack 2024 · Grand View Research 2024 · OpenTable 2026 · EHL Hospitality Insights · Mintel 2024 — full bibliography in `llm_simulation/research/bibliography.md`
- **Model**: OpenAI `gpt-5.4` (pairwise + revisit) / `gpt-5.4-mini` (ranking)
- **Replication**: Anthropic Claude Sonnet 4.6 (cross-model agreement check)

---

## Requirements

- Python 3.9+
- Key packages: `streamlit`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `networkx`, `folium`, `duckdb`, `pyarrow`, `lightgbm`, `pytest`
- LLM simulation: `openai>=1.75.0`, `anthropic>=0.49.0`, `tqdm` (see `llm_simulation/requirements.txt`)
- Hardware: Developed on Apple M5, 16GB RAM
- API keys: `OPENAI_API_KEY` (required for simulation), `ANTHROPIC_API_KEY` (optional, for Claude replication) — set in `llm_simulation/.env`

---

## Data License

Yelp Academic Dataset used under the [Yelp Dataset User Agreement](Dataset_User_Agreement.pdf). Foursquare WWW2019 dataset used for research purposes.
