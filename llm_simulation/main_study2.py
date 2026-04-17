"""
Study 2 — Occupation × Age Cross-Matrix LLM Simulation
=======================================================
3,000 personas across 5 age groups × 10 occupation clusters × 3 domains.
Runs alongside Study 1 (behavioural archetypes) as independent validation.

Run:
    python main_study2.py [--dry-run] [--domain coffee|restaurant|hotel]
    python main_study2.py --age-group "Gen Z (18-25)"
    python main_study2.py --occupation "Healthcare"
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

# ── env setup ─────────────────────────────────────────────────────────────────
_ENV_KEY = "OPENAI_API_KEY"
if not os.environ.get(_ENV_KEY):
    _env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_file):
        for line in open(_env_file):
            line = line.strip()
            if line.startswith(_ENV_KEY + "="):
                os.environ[_ENV_KEY] = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

import config as cfg
cfg.OPENAI_API_KEY = os.environ.get(_ENV_KEY, cfg.OPENAI_API_KEY)

from config import RESULTS_DIR, RANDOM_SEED
from data_loader import (
    load_coffee_venues, load_restaurant_venues, load_hotel_venues,
    filter_by_city, build_discriminating_set,
)
from demographic_persona_generator import DemographicPersona, generate_study2_personas
from prompts import build_pairwise_prompt, build_revisit_prompt
from task_runner import call_llm, call_llm_mini, FULL_MODEL
from evaluator import ndcg_at_k, hits_all_k, kendall_tau_score
from report_study2 import generate_study2_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)
_venues: Dict[str, pd.DataFrame] = {}


def get_venues(domain: str) -> pd.DataFrame:
    if domain not in _venues:
        _venues[domain] = {
            "coffee": load_coffee_venues,
            "restaurant": load_restaurant_venues,
            "hotel": load_hotel_venues,
        }[domain]()
    return _venues[domain]


# ── Ranking prompt builder for demographic personas ───────────────────────────

def _build_demographic_ranking_prompt(persona: DemographicPersona, candidate_df: pd.DataFrame) -> str:
    """Ranking prompt anchored to occupation + age behavioral context."""
    from prompts import _format_venues  # reuse venue formatting
    domain = persona.domain
    venue_lines = _format_venues(candidate_df, domain)
    d_label = {"coffee": "coffee shop", "restaurant": "restaurant", "hotel": "hotel"}[domain]

    return (
        f"**Your situation:** {persona.task_context}\n\n"
        f"You are choosing a {d_label}. Here are {len(candidate_df)} options:\n\n"
        f"{venue_lines}\n\n"
        f"Rank these from most to least appealing **for you specifically** — "
        f"a {persona.age}-year-old {persona.occupation_label} in {persona.city}. "
        f"Think about your real habits, schedule, and priorities as described.\n\n"
        "Return JSON:\n"
        '{"ranking": [1, 3, 7, 2, 5, 8, 4, 6, 9, 10], "reasoning": "one sentence"}\n\n'
        "Include all venue numbers exactly once."
    )


def _build_demographic_system_prompt(persona: DemographicPersona) -> str:
    return (
        f"You are {persona.name}, a {persona.age}-year-old {persona.occupation_label} "
        f"living in {persona.city}.\n\n"
        f"{persona.behavioral_profile}\n\n"
        "Respond as this specific person. Reflect real habits and priorities. "
        "Do not break character or mention AI simulations."
    )


def _parse_ranking(response: dict, n: int = 10) -> List[int]:
    raw = response.get("ranking", [])
    if not isinstance(raw, list):
        return list(range(n))
    result, seen = [], set()
    for v in raw:
        try:
            idx = int(v) - 1
            if 0 <= idx < n and idx not in seen:
                result.append(idx)
                seen.add(idx)
        except (TypeError, ValueError):
            pass
    for i in range(n):
        if i not in seen:
            result.append(i)
    return result[:n]


# ── Per-persona task ──────────────────────────────────────────────────────────

async def run_demographic_persona(
    persona: DemographicPersona,
    dry_run: bool = False,
) -> Optional[Dict]:
    all_venues = get_venues(persona.domain)
    city_venues = filter_by_city(all_venues, persona.city)

    # Use persona's archetype-specific sort column if available in data
    if persona.sort_col in city_venues.columns:
        sort_ascending = False
        city_venues = city_venues.sort_values(persona.sort_col, ascending=sort_ascending)

    system = _build_demographic_system_prompt(persona)
    temp = cfg.TEMPERATURE

    # Task 1: Ranking (gpt-5.4-mini — high volume)
    candidate = build_discriminating_set(city_venues, n=cfg.VENUES_PER_TASK, rng=rng)
    model_rank_col = "model_rank" if "model_rank" in candidate.columns else "rank"
    gt_ranks = dict(enumerate(candidate[model_rank_col].tolist()))
    stars_ranks = dict(enumerate(candidate["stars_rank"].tolist()))

    ranking_prompt = _build_demographic_ranking_prompt(persona, candidate)

    if dry_run:
        persona_ranking = list(range(len(candidate)))
    else:
        r1 = await call_llm_mini(system, ranking_prompt, temperature=temp)
        if "error" in r1:
            logger.warning("Ranking error %s: %s", persona.id, r1["error"])
            return None
        persona_ranking = _parse_ranking(r1, n=len(candidate))

    ndcg = ndcg_at_k(persona_ranking, gt_ranks, k=10)
    hits = hits_all_k(persona_ranking, gt_ranks, relevant_cutoff=3)
    tau = kendall_tau_score(persona_ranking, gt_ranks)
    stars_order = sorted(range(len(candidate)), key=lambda i: stars_ranks.get(i, 99))
    stars_ndcg = ndcg_at_k(stars_order, gt_ranks, k=10)
    stars_hits = hits_all_k(stars_order, gt_ranks, relevant_cutoff=3)

    # Task 2: Pairwise (gpt-5.4 — quality signal)
    birank_top = candidate.sort_values(model_rank_col).iloc[0].to_dict()
    stars_top = candidate.sort_values("stars_rank").iloc[0].to_dict()
    pairwise_prompt = build_pairwise_prompt(persona, birank_top, stars_top, persona.domain)

    if dry_run:
        pairwise_win = int(rng.integers(0, 2))
    else:
        r2 = await call_llm(system, pairwise_prompt, temperature=temp, model=FULL_MODEL)
        pairwise_win = 1 if str(r2.get("choice", "")).strip().upper() == "A" else 0

    # Task 3: Revisit (gpt-5.4)
    revisit_target = candidate.iloc[0].to_dict()
    model_revisit = float(
        revisit_target.get("revisit_rate",
        revisit_target.get("multi_stay_rate",
        revisit_target.get("repeat_user_rate", 0.0))) or 0.0
    )
    revisit_prompt = build_revisit_prompt(persona, revisit_target, persona.domain)

    if dry_run:
        revisit_score = float(rng.integers(0, 11))
    else:
        r3 = await call_llm(system, revisit_prompt, temperature=temp, model=FULL_MODEL)
        try:
            revisit_score = float(r3.get("revisit_score", 5))
        except (TypeError, ValueError):
            revisit_score = 5.0

    return {
        "persona_id":            persona.id,
        "domain":                persona.domain,
        "archetype":             f"{persona.age_group} | {persona.occupation_cluster}",
        "age_group":             persona.age_group,
        "occupation":            persona.occupation_cluster,
        "occupation_label":      persona.occupation_label,
        "city":                  persona.city,
        "loyalty_score":         persona.loyalty_score,
        "price_sensitivity":     persona.price_sensitivity,
        "ndcg":                  ndcg,
        "hit_at_1":              hits["hit_at_1"],
        "hit_at_3":              hits["hit_at_3"],
        "hit_at_10":             hits["hit_at_10"],
        "kendall_tau":           tau,
        "stars_ndcg":            stars_ndcg,
        "stars_hit_at_1":        stars_hits["hit_at_1"],
        "stars_hit_at_3":        stars_hits["hit_at_3"],
        "pairwise_win":          pairwise_win,
        "revisit_score":         revisit_score,
        "model_revisit_signal":  model_revisit,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(
    dry_run: bool = False,
    domain_filter: Optional[str] = None,
    age_filter: Optional[str] = None,
    occupation_filter: Optional[str] = None,
):
    personas = generate_study2_personas()

    if domain_filter:
        personas = [p for p in personas if p.domain == domain_filter]
    if age_filter:
        personas = [p for p in personas if p.age_group == age_filter]
    if occupation_filter:
        personas = [p for p in personas if p.occupation_cluster == occupation_filter]

    total = len(personas)
    domains = list({p.domain for p in personas})

    print(f"\n{'='*64}")
    print(f"  Study 2 — Occupation × Age Cross-Matrix Simulation")
    print(f"{'='*64}")
    print(f"  Model (ranking):  gpt-5.4-mini")
    print(f"  Model (pairwise): gpt-5.4")
    print(f"  Total personas:   {total:,}")
    print(f"  Age groups:       {len({p.age_group for p in personas})}")
    print(f"  Occupations:      {len({p.occupation_cluster for p in personas})}")
    print(f"  Concurrency:      {cfg.MAX_CONCURRENT}")

    if not dry_run:
        mini_calls = total
        full_calls = total * 2
        est_cost = (mini_calls * 700 / 1e6 * 0.40) + (mini_calls * 150 / 1e6 * 1.60)
        est_cost += (full_calls * 700 / 1e6 * 2.0) + (full_calls * 150 / 1e6 * 8.0)
        print(f"  Est. cost:        ~${est_cost:.2f}")
        confirm = input("\nProceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    for d in domains:
        get_venues(d)
        print(f"  Loaded {d} venues ({len(_venues[d])} rows)")

    print(f"\nRunning {total:,} demographic personas...\n")

    records: List[Dict] = []
    errors = 0
    start = time.time()
    sem = asyncio.Semaphore(cfg.MAX_CONCURRENT)

    async def bounded(p: DemographicPersona):
        async with sem:
            return await run_demographic_persona(p, dry_run=dry_run)

    tasks = [bounded(p) for p in personas]
    for coro in atqdm(asyncio.as_completed(tasks), total=total, desc="Study 2"):
        result = await coro
        if result is not None:
            records.append(result)
        else:
            errors += 1

    elapsed = time.time() - start
    print(f"\nCompleted {len(records):,}/{total:,} in {elapsed:.1f}s ({errors} errors)")

    if not records:
        print("No results — exiting.")
        return

    # Save records
    records_path = os.path.join(RESULTS_DIR, "simulation_records_study2.csv")
    with open(records_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"  Records → {records_path}")

    # Generate report
    report_path = generate_study2_report(records, RESULTS_DIR)
    print(f"  Report  → {report_path}")

    # Quick summary
    _print_summary(records)


def _print_summary(records: List[Dict]):
    from collections import defaultdict
    import statistics

    by_age: Dict[str, List[float]] = defaultdict(list)
    by_occ: Dict[str, List[float]] = defaultdict(list)

    for r in records:
        by_age[r["age_group"]].append(r["ndcg"])
        by_occ[r["occupation"]].append(r["ndcg"])

    print("\n" + "="*56)
    print("STUDY 2 SUMMARY — BY AGE GROUP")
    print("="*56)
    age_order = ["Gen Z (18-25)", "Young Millennial (26-33)",
                 "Senior Millennial (34-40)", "Gen X (41-56)", "Boomer (57+)"]
    for ag in age_order:
        vals = by_age.get(ag, [])
        if vals:
            print(f"  {ag:<28} n={len(vals):>4}  NDCG={statistics.mean(vals):.4f}")

    print("\n" + "="*56)
    print("STUDY 2 SUMMARY — BY OCCUPATION")
    print("="*56)
    for occ, vals in sorted(by_occ.items(), key=lambda x: -statistics.mean(x[1])):
        print(f"  {occ:<30} n={len(vals):>4}  NDCG={statistics.mean(vals):.4f}")

    overall = [r["ndcg"] for r in records]
    pairwise = [r["pairwise_win"] for r in records]
    print(f"\n  Overall NDCG@10:      {sum(overall)/len(overall):.4f}")
    print(f"  Pairwise win rate:    {sum(pairwise)/len(pairwise):.1%}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Study 2 — Demographic Simulation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--domain", choices=["coffee", "restaurant", "hotel"])
    parser.add_argument("--age-group")
    parser.add_argument("--occupation")
    args = parser.parse_args()
    asyncio.run(main(
        dry_run=args.dry_run,
        domain_filter=args.domain,
        age_filter=args.age_group,
        occupation_filter=args.occupation,
    ))
