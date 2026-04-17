"""
LLM Simulation — Behavioral Venue Ranking Validation
=====================================================
1500 synthetic personas (coffee / restaurant / hotel) each run 3 tasks:
  • Ranking Task    → NDCG@10 vs. model ranking
  • Pairwise Task   → BiRank top-1 vs. Stars top-1 head-to-head win rate
  • Revisit Task    → calibration of revisit likelihood predictions

Run:
    python main.py [--dry-run] [--domain coffee|restaurant|hotel]
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

# ── env / config setup ────────────────────────────────────────────────────────
# Allow running with OPENAI_API_KEY already in environment, or set it here
_ENV_KEY = "OPENAI_API_KEY"
if not os.environ.get(_ENV_KEY):
    # Try loading from .env file next to this script
    _env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_file):
        for line in open(_env_file):
            line = line.strip()
            if line.startswith(_ENV_KEY + "="):
                os.environ[_ENV_KEY] = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

from config import OPENAI_API_KEY, PERSONA_COUNTS, RANDOM_SEED, RESULTS_DIR
import config as cfg

if not os.environ.get(_ENV_KEY) and not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set. Create llm_simulation/.env with OPENAI_API_KEY=sk-...")
    sys.exit(1)

# Update the client key if loaded from .env
cfg.OPENAI_API_KEY = os.environ.get(_ENV_KEY, OPENAI_API_KEY)

from data_loader import (
    load_coffee_venues, load_restaurant_venues, load_hotel_venues,
    sample_candidate_set,
)
from persona_generator import generate_all_personas, Persona
from prompts import (
    build_system_prompt, build_ranking_prompt,
    build_pairwise_prompt, build_revisit_prompt,
)
from task_runner import call_llm
from evaluator import ndcg_at_k, hit_at_k, compute_metrics
from report_generator import save_records, save_metrics, generate_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


# ── Venue loaders (cached per domain) ─────────────────────────────────────────

_venues: dict[str, pd.DataFrame] = {}

def get_venues(domain: str) -> pd.DataFrame:
    if domain not in _venues:
        loader = {
            "coffee": load_coffee_venues,
            "restaurant": load_restaurant_venues,
            "hotel": load_hotel_venues,
        }[domain]
        _venues[domain] = loader()
    return _venues[domain]


# ── Individual tasks ──────────────────────────────────────────────────────────

def _parse_ranking(response: dict, n: int = 10) -> list[int]:
    """Extract and validate ranking list from LLM response."""
    raw = response.get("ranking", [])
    if not isinstance(raw, list):
        return list(range(n))  # fallback: identity order
    # convert to 0-indexed, clamp to valid range
    result = []
    seen = set()
    for v in raw:
        try:
            idx = int(v) - 1  # LLM returns 1-indexed
            if 0 <= idx < n and idx not in seen:
                result.append(idx)
                seen.add(idx)
        except (ValueError, TypeError):
            pass
    # fill any missing indices at end
    for i in range(n):
        if i not in seen:
            result.append(i)
    return result[:n]


async def run_persona_tasks(persona: Persona, dry_run: bool = False) -> Optional[dict]:
    venues = get_venues(persona.domain)
    system = build_system_prompt(persona)
    temp = cfg.TEMPERATURE

    # ── Task 1: Ranking ───────────────────────────────────────────────────────
    candidate = sample_candidate_set(venues, n=cfg.VENUES_PER_TASK, rng=rng)
    model_rank_col = "model_rank" if "model_rank" in candidate.columns else "rank"
    gt_ranks = dict(enumerate(candidate[model_rank_col].tolist()))
    stars_ranks = dict(enumerate(candidate["stars_rank"].tolist()))

    ranking_prompt = build_ranking_prompt(persona, candidate, persona.domain)

    if dry_run:
        persona_ranking = list(range(len(candidate)))
    else:
        r1 = await call_llm(system, ranking_prompt, temperature=temp)
        if "error" in r1:
            logger.warning("Ranking task error for %s: %s", persona.id, r1["error"])
            return None
        persona_ranking = _parse_ranking(r1, n=len(candidate))

    ndcg = ndcg_at_k(persona_ranking, gt_ranks, k=10)
    hit = hit_at_k(persona_ranking, gt_ranks, k=10)
    stars_ndcg = ndcg_at_k(list(range(len(candidate))), stars_ranks, k=10)

    # ── Task 2: Pairwise ──────────────────────────────────────────────────────
    birank_top = candidate.sort_values(model_rank_col).iloc[0].to_dict()
    stars_top = candidate.sort_values("stars_rank").iloc[0].to_dict()

    pairwise_prompt = build_pairwise_prompt(persona, birank_top, stars_top, persona.domain)

    if dry_run:
        pairwise_win = rng.integers(0, 2)
    else:
        r2 = await call_llm(system, pairwise_prompt, temperature=temp)
        if "error" in r2:
            pairwise_win = 0
        else:
            choice = str(r2.get("choice", "")).strip().upper()
            pairwise_win = 1 if choice == "A" else 0  # A = BiRank top

    # ── Task 3: Revisit ───────────────────────────────────────────────────────
    # pick a venue from the candidate set and ask about revisit likelihood
    revisit_target = candidate.iloc[0].to_dict()
    model_revisit_signal = float(revisit_target.get("revisit_rate",
                                 revisit_target.get("multi_stay_rate", 0.0)) or 0.0)

    revisit_prompt = build_revisit_prompt(persona, revisit_target, persona.domain)

    if dry_run:
        revisit_score = float(rng.integers(0, 11))
    else:
        r3 = await call_llm(system, revisit_prompt, temperature=temp)
        if "error" in r3:
            revisit_score = 5.0
        else:
            try:
                revisit_score = float(r3.get("revisit_score", 5))
            except (TypeError, ValueError):
                revisit_score = 5.0

    return {
        "persona_id": persona.id,
        "domain": persona.domain,
        "archetype": persona.archetype,
        "ndcg": ndcg,
        "hit": hit,
        "stars_ndcg": stars_ndcg,
        "pairwise_win": pairwise_win,
        "revisit_score": revisit_score,
        "model_revisit_signal": model_revisit_signal,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(dry_run: bool = False, domain_filter: Optional[str] = None):
    personas = generate_all_personas()
    if domain_filter:
        personas = [p for p in personas if p.domain == domain_filter]

    total = len(personas)
    total_api_calls = total * 3 if not dry_run else 0

    print(f"\n{'='*60}")
    print(f"  LLM Simulation — Behavioral Venue Ranking Validation")
    print(f"{'='*60}")
    print(f"  Model:         {cfg.MODEL}")
    print(f"  Personas:      {total:,}")
    print(f"  API calls:     {total_api_calls:,}" + (" (0 — dry run)" if dry_run else ""))
    print(f"  Concurrency:   {cfg.MAX_CONCURRENT}")
    if not dry_run:
        # rough cost estimate: gpt-4.1 ~$2/M input, $8/M output
        est_input_tok = total * 3 * 700
        est_output_tok = total * 3 * 150
        est_cost = (est_input_tok / 1e6 * 2.0) + (est_output_tok / 1e6 * 8.0)
        print(f"  Est. cost:     ~${est_cost:.2f}")
    print(f"{'='*60}\n")

    if not dry_run:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    records: list[dict] = []
    errors = 0
    start = time.time()

    # Load venues upfront (avoids repeated I/O inside async tasks)
    domains = ["coffee", "restaurant", "hotel"] if not domain_filter else [domain_filter]
    for d in domains:
        get_venues(d)
        print(f"  Loaded {d} venues ({len(_venues[d])} rows)")

    print(f"\nRunning tasks for {total:,} personas...\n")

    # Run with tqdm progress bar
    sem = asyncio.Semaphore(cfg.MAX_CONCURRENT)

    async def bounded(persona):
        async with sem:
            return await run_persona_tasks(persona, dry_run=dry_run)

    tasks = [bounded(p) for p in personas]
    results = []
    for coro in atqdm(asyncio.as_completed(tasks), total=total, desc="Personas"):
        result = await coro
        if result is not None:
            results.append(result)
        else:
            errors += 1

    elapsed = time.time() - start
    print(f"\nCompleted {len(results):,} / {total:,} personas in {elapsed:.1f}s  ({errors} errors)")

    if not results:
        print("No results — exiting.")
        return

    # Evaluate
    metrics = compute_metrics(results)

    # Save
    save_records(results)
    save_metrics(metrics)
    generate_report(metrics, results)

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Domain':<14} {'Archetype':<28} {'n':>5} {'NDCG@10':>8} {'Hit@10':>8} {'WinRate':>8} {'p':>7}")
    print("-"*70)
    for domain in domains:
        for arch in list(PERSONA_COUNTS.get(domain, {}).keys()) + ["OVERALL"]:
            m = metrics.get((domain, arch))
            if not m:
                continue
            sig = "*" if m["wilcoxon_p"] < 0.05 else " "
            print(
                f"{domain:<14} {arch:<28} {m['n']:>5} "
                f"{m['ndcg_mean']:>8.4f} {m['hit_mean']:>8.4f} "
                f"{m['pairwise_win_rate']:>8.1%} {m['wilcoxon_p']:>6.4f}{sig}"
            )
        print("-"*70)

    all_overall = metrics.get(("ALL", "OVERALL"), {})
    if all_overall:
        print(f"\n  Overall NDCG@10:      {all_overall['ndcg_mean']:.4f} "
              f"[{all_overall['ndcg_lo']:.4f}, {all_overall['ndcg_hi']:.4f}]")
        print(f"  vs. Stars baseline:   {all_overall['stars_ndcg_mean']:.4f} "
              f"(Δ {all_overall['delta_vs_stars']:+.4f})")
        print(f"  Pairwise win rate:    {all_overall['pairwise_win_rate']:.1%}")
        p = all_overall["wilcoxon_p"]
        print(f"  Wilcoxon p-value:     {p:.4f} {'✓ significant' if p < 0.05 else '✗ not significant'}")

    print(f"\nResults in: {RESULTS_DIR}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Simulation — Venue Ranking Validation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without making API calls (random responses)")
    parser.add_argument("--domain", choices=["coffee", "restaurant", "hotel"],
                        help="Run for one domain only")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, domain_filter=args.domain))
