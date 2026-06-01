.PHONY: all setup test benchmark dashboard api figures clean

# ── Setup ──────────────────────────────────────────────────────────────────
setup:
	pip install -e .
	pip install -r requirements-lock.txt

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -q

test-all:
	pytest tests/ theory/tests/ temporal/tests/ causal_v2/tests/ fairness/tests/ -q

# ── Core benchmarks ────────────────────────────────────────────────────────
benchmark-coffee:
	cd bvr && python -m experiments.validate_coffee

benchmark-london:
	cd bvr && python -m pipelines.london

benchmark-uk-fsq:
	cd bvr && python -m pipelines.uk_fsq

benchmark: benchmark-coffee benchmark-london benchmark-uk-fsq

# ── Ablations ──────────────────────────────────────────────────────────────
ablation-prior:
	python -m bvr.experiments.ablations

ablation-lightgcn:
	python -m bvr.experiments.lightgcn_ablation

temporal-robustness:
	python -m bvr.experiments.temporal_robustness

ablations: ablation-prior ablation-lightgcn temporal-robustness

# ── Causal ────────────────────────────────────────────────────────────────
causal:
	python -m bvr.causal.data_prep
	python -m bvr.causal.psm
	python -m bvr.causal.report
	python -m bvr.causal.permutation
	python -m bvr.causal.sensitivity

# ── EmergeRec benchmark ───────────────────────────────────────────────────
emergerec:
	python -m emergerec.benchmark --datasets all --models all

# ── Dashboard ────────────────────────────────────────────────────────────
dashboard:
	python -m streamlit run bvr/dashboard/app.py --server.port 8501

# ── API ──────────────────────────────────────────────────────────────────
api:
	python -m uvicorn bvr.api.serve:app --port 8000 --reload

# ── Figures ──────────────────────────────────────────────────────────────
figures:
	@echo "Running theory notebooks..."
	jupyter nbconvert --to notebook --execute theory/notebooks/*.ipynb --inplace 2>/dev/null || true
	@echo "Figures generated from notebooks."

# ── Papers ───────────────────────────────────────────────────────────────
paper-p0:
	cd papers/p0_anti_loyalty && latexmk -pdf -interaction=nonstopmode main.tex 2>/dev/null || echo "LaTeX not installed — draft only"

papers: paper-p0
	@echo "Paper builds complete."

# ── Full pipeline ────────────────────────────────────────────────────────
all: setup test benchmark ablations causal figures
	@echo "Full pipeline complete."

# ── Clean ────────────────────────────────────────────────────────────────
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
