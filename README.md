# Behavioral Venue Ranking System

A venue ranking system that ranks businesses by what people **do** (visit patterns, loyalty, regularity) rather than what they **say** (star ratings). Built on the Yelp Academic Dataset, supplemented with Foursquare check-ins and Transitland transit data.

Includes a validated **coffee shop model** (BiRank with behavioral priors) and a **restaurant model** (multi-objective S(R,U,C) scoring with mobility and context awareness), plus an interactive Streamlit dashboard for exploration.

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

## Dashboard

Interactive Streamlit app for exploring rankings.

```bash
python3 -m streamlit run app.py
```

### Features

- **City search** with fuzzy matching ("philly" finds Philadelphia)
- **Radius-based area filter** around any reference venue
- **Side-by-side ranking comparison**: BiRank vs. rating vs. popularity vs. revisit rate
- **Behavioral mode selector**: Regular, Explorer, Morning, Weekend
- **Venue detail cards**: Plain-language tags (Steady / High Retention / Broad Loyalty)
- **Interactive Folium maps** with marker popups
- **CSV export** of results

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
| `validation_summary.txt` | Legacy v3 validation results |
| `fsq.duckdb` | Foursquare DuckDB database |
| `venue_linkage.csv` | Yelp-Foursquare venue matches |
| `yelp_fsq_user_bridge.csv` | Cross-platform user bridge table |

---

## Methods & References

- **BiRank**: He et al. — bipartite graph ranking via mutual reinforcement
- **Burstiness Index**: Goh & Barabasi — temporal regularity of human dynamics
- **Shannon Entropy**: Information-theoretic diversity measure
- **Gini Coefficient**: Loyalty concentration (economics)
- **NDCG / Hit Rate**: Standard information retrieval evaluation metrics
- **Entropy Weight Method**: Dynamic feature weighting from decision theory
- **Maximal Marginal Relevance**: Diversity-aware re-ranking (Carbonell & Goldstein)

---

## Requirements

- Python 3.9+
- Key packages: `streamlit`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `networkx`, `folium`, `duckdb`, `pyarrow`
- Hardware: Developed on Apple M5, 16GB RAM

---

## Data License

Yelp Academic Dataset used under the [Yelp Dataset User Agreement](Dataset_User_Agreement.pdf). Foursquare WWW2019 dataset used for research purposes.
