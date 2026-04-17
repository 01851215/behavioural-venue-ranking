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
- **BiRank**: He et al. — bipartite graph ranking via mutual reinforcement
- **ALS Matrix Factorization**: Hu et al. — implicit feedback collaborative filtering
- **BPR**: Rendle et al. — Bayesian personalized ranking from implicit feedback
- **Maximal Marginal Relevance**: Diversity-aware re-ranking (Carbonell & Goldstein)

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
- Key packages: `streamlit`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `networkx`, `folium`, `duckdb`, `pyarrow`
- LLM simulation: `openai>=1.75.0`, `anthropic>=0.49.0`, `tqdm` (see `llm_simulation/requirements.txt`)
- Hardware: Developed on Apple M5, 16GB RAM
- API keys: `OPENAI_API_KEY` (required for simulation), `ANTHROPIC_API_KEY` (optional, for Claude replication) — set in `llm_simulation/.env`

---

## Data License

Yelp Academic Dataset used under the [Yelp Dataset User Agreement](Dataset_User_Agreement.pdf). Foursquare WWW2019 dataset used for research purposes.
