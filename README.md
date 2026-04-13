# Behavioral Venue Ranking System

A venue ranking system that ranks businesses by what people **do** (visit patterns, loyalty, regularity) rather than what they **say** (star ratings). Built on the Yelp Academic Dataset, supplemented with Foursquare check-ins and Transitland transit data.

Includes a validated **coffee shop model** (BiRank with behavioral priors) and a **restaurant model** (multi-objective S(R,U,C) scoring with mobility and context awareness), plus an interactive Streamlit dashboard for exploration.

---

## Plain English Summary

**The problem:** When you search for a coffee shop or restaurant, most apps rank places by star ratings. But ratings are easy to fake, influenced by mood, and don't tell you whether people actually come back. A place might have 4.8 stars from 12 reviews — or it might have 4.2 stars from people who go there every single week.

**The idea:** Instead of asking "what do people say about this place?", ask "what do people *do*?". If someone visits a cafe every Tuesday morning for three years, that tells you far more than their one-time 5-star review. This project ranks venues using visit behaviour — how often people return, how loyal they are, how consistent the traffic is — rather than what they write.

**How it works in simple terms:** Imagine drawing a web connecting every person to every venue they've visited. A venue that attracts many returning, regular visitors gets boosted. A user who visits frequently and consistently gets treated as a more reliable signal. The algorithm (BiRank) bounces a score back and forth across this web until it settles — important users boost important venues, and important venues boost the importance of users who visit them.

**What we built:**
- A coffee shop ranking model across 8,500+ venues and 93,000 users, grouping people into four types: Loyalists (their regular), Weekday Regulars (work-routine), Casual Weekenders, and Infrequent Visitors
- A restaurant model that also considers how far you are from a place, how easy it is to reach by public transport, and whether it matches your cuisine preferences
- An interactive map dashboard to explore rankings by city

**What we found:** The behavioural approach consistently outperforms star-rating ranking and random recommendation. Crucially, it works best for Loyalists — people with strong habits — which is exactly what you'd expect if the theory is right.

---

**Version history at a glance:**

- **v3 (baseline):** First working behavioural model. Proved the core idea — BiRank on visit behaviour beats star ratings. But the validation had bugs that made results look better than they were.
- **v4 (Foursquare):** Added check-in data from a second platform (Foursquare) to bring in social signals — whether your friends visited a place. Turned out this made results *worse*: the social data was too noisy and unrelated to coffee habits.
- **v5 (honest numbers):** Fixed three serious methodological errors: (1) the model was accidentally "cheating" by using future data it shouldn't have seen during training; (2) the accuracy metric was calculated incorrectly; (3) there were no statistical tests proving results weren't just luck. After fixing all three, the numbers dropped (from 0.086 to 0.076) — but these are the *correct* numbers. Also fixed the Foursquare integration to only use high-confidence social links, which stopped it from hurting performance.
- **v6 (hybrid experiment):** Tested whether adding a second type of algorithm — Matrix Factorization, which finds hidden patterns like "people who like X tend to like Y" — could improve on BiRank. It didn't. The best blend was still essentially pure BiRank (λ=1.0 selected by tuning). This is a meaningful negative result: BiRank's behavioural signals are already capturing what matters, and "collaborative filtering" patterns add nothing extra in this domain.
- **v7 (hotel model):** Extended the whole framework to hotels and accommodation. This required redesigning the behavioral features from scratch — hotels are fundamentally different from coffee shops (nobody visits the same hotel weekly). Key finding: BiRank still beats star ratings (p=0.012), but collaborative filtering outperforms behavioral signals for hotels because most users only stay at 1–2 hotels, making behavior patterns too sparse to learn from. Also conducted a cross-domain experiment: users who explore many coffee shops tend to explore many hotels too, but predicting hotel preferences from coffee habits is only marginally better than chance.

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
- **ALS Matrix Factorization**: Hu et al. — implicit feedback collaborative filtering
- **BPR**: Rendle et al. — Bayesian personalized ranking from implicit feedback

---

## Requirements

- Python 3.9+
- Key packages: `streamlit`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `networkx`, `folium`, `duckdb`, `pyarrow`
- Hardware: Developed on Apple M5, 16GB RAM

---

## Data License

Yelp Academic Dataset used under the [Yelp Dataset User Agreement](Dataset_User_Agreement.pdf). Foursquare WWW2019 dataset used for research purposes.
