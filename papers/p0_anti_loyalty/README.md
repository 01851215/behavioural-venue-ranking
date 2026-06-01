# Paper P0 — Anti-Loyalty Priors for Venue Emergence Detection

**Target venue:** RecSys 2026 (workshop) or WSDM 2027 (long paper track)  
**Deadline:** ~June 2026  
**Status:** Draft ready (master's thesis findings + formalization)

## Title (working)
"Anti-Loyalty Priors for Popularity-Debiased Rising-Star Detection in Bipartite Interaction Graphs"

## Abstract (draft)
Venue recommendation systems face a fundamental tension: optimising for revisit prediction rewards popular venues, while identifying emerging venues requires specifically penalising them. We introduce the rising-stars metric — a popularity-debiased Spearman correlation that measures whether a model identifies venues growing beyond their popularity trajectory — and propose anti-loyalty priors that directly target this signal. On two independent UK datasets (TripAdvisor London: ρ=+0.249; Foursquare GB: ρ=+0.215), our method outperforms the best prior approach (ρ=+0.094) and state-of-the-art graph convolution (LightGCN, ρ=−0.059). We show that LightGCN's failure is structural: negative rising-stars ρ persists across L=1–5 propagation layers, confirming that popularity amplification is an inherent property of graph convolution, not a tuning issue.

## Key figures
- Fig 1: Rising-stars ρ bar chart (from bvr/dashboard/app.py validation tab)
- Fig 2: Anti-loyalty prior α sweep (from data/results/ablation_exploration_prior.csv)
- Fig 3: LightGCN layer ablation (from data/results/lightgcn_layer_ablation.csv)
- Fig 4: Temporal split robustness (from data/results/temporal_robustness_london.csv)

## Status
- [ ] LaTeX draft (main.tex)
- [ ] Figure exports
- [ ] References (.bib)
- [ ] Submission
