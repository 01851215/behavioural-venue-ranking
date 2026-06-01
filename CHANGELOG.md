# Changelog

## [0.2.0] — 2026-06-01 — PhD Restructure
Restructured from master's project into research codebase.

### Changed
- All source code moved into `bvr/` Python package (importable via `from bvr.core.validation import ...`)
- All output files consolidated under `data/results/`
- All pipeline scripts under `bvr/pipelines/`
- Dashboard moved to `bvr/dashboard/app.py`
- FastAPI service moved to `bvr/api/serve.py`

### Added
- `setup.py` — installable package (`pip install -e .`)
- `Makefile` — one-command reproducibility (`make all`)
- `CITATION.cff` — citable research software
- `LICENSE` — MIT open-source licence
- `theory/` — skeleton for Pillar 1: formal theory of behavioral priors
- `temporal/` — skeleton for Pillar 2: temporal GNN for venue emergence
- `causal_v2/` — skeleton for Pillar 3: doubly-robust causal inference
- `fairness/` — skeleton for Pillar 4: algorithmic fairness for long-tail venues
- `emergerec/` — skeleton for open benchmark / library
- `papers/` — LaTeX paper drafts (p0 through p6)
- `docs/` — Sphinx documentation structure

## [0.1.0] — 2026-05-22 — Master's Thesis Baseline
First complete version. All master's thesis results reproduced and validated.

### Key results
- BiRank v5: NDCG@10 = 0.0765 (coffee), beats random p=0.038
- Anti-loyalty + ALS hybrid: ρ = +0.249 (London), +0.215 (UK FSQ)
- LightGCN failure confirmed: ρ < 0 at L=1–5 on UK FSQ (structural)
- Causal PSM (COVID-adjusted): ATE = +0.0027, p = 0.031
- Cold-start coverage: 86.8% → 99.1%
- LLM simulation: 3,360 personas, cross-validated GPT-5.4 + Claude Sonnet 4.6
