"""
LLM Simulation v2 — Full Phase Execution
=========================================
Phase 1: Discriminating sets + city-matched venues + Hit@1/3 + Kendall τ
Phase 2: Manipulation check + null baseline + inverted personas
Phase 3: Revisit calibration + cross-domain + per-persona variance
Phase 4: BH correction + Cohen's d + stratified bootstrap + power analysis
Phase 5: Tiered models (mini for ranking, full for pairwise) + Claude replication
Phase 6: Results saved for Streamlit dashboard

Run:
    python main_v2.py [--dry-run] [--domain coffee|restaurant|hotel]
    python main_v2.py --skip-phases 5b    # skip Claude replication
    python main_v2.py --phases 1,2        # run specific phases only
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

from config import PERSONA_COUNTS, RANDOM_SEED, RESULTS_DIR
from data_loader import (
    load_coffee_venues, load_restaurant_venues, load_hotel_venues,
    filter_by_city, build_discriminating_set, build_archetype_set,
)
from persona_generator import generate_all_personas, Persona
from prompts import build_system_prompt, build_ranking_prompt, build_pairwise_prompt, build_revisit_prompt
from task_runner import call_llm, call_llm_mini, MINI_MODEL, FULL_MODEL
from evaluator import ndcg_at_k, hits_all_k, kendall_tau_score, compute_metrics, wilcoxon_test
from report_generator import save_records, save_metrics, generate_report

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


# ── Response parsing ──────────────────────────────────────────────────────────

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
        except (ValueError, TypeError):
            pass
    for i in range(n):
        if i not in seen:
            result.append(i)
    return result[:n]


# ── Per-persona task runner ───────────────────────────────────────────────────

async def run_persona_v2(
    persona: Persona,
    dry_run: bool = False,
    use_tiered: bool = True,
) -> Optional[Dict]:
    """Run 3 tasks per persona with Phase 1+5 improvements."""
    all_venues = get_venues(persona.domain)
    city_venues = filter_by_city(all_venues, persona.city)

    system = build_system_prompt(persona)
    temp = cfg.TEMPERATURE

    # ── Task 1: Ranking (Phase 1 — discriminating set; Phase 5 — mini model) ──
    candidate = build_discriminating_set(city_venues, n=cfg.VENUES_PER_TASK, rng=rng)
    model_rank_col = "model_rank" if "model_rank" in candidate.columns else "rank"
    gt_ranks = dict(enumerate(candidate[model_rank_col].tolist()))
    stars_ranks = dict(enumerate(candidate["stars_rank"].tolist()))

    ranking_prompt = build_ranking_prompt(persona, candidate, persona.domain)

    if dry_run:
        persona_ranking = list(range(len(candidate)))
    else:
        # Phase 5: use mini model for ranking (cheaper, still coherent)
        _call = call_llm_mini if use_tiered else call_llm
        r1 = await _call(system, ranking_prompt, temperature=temp)
        if "error" in r1:
            logger.warning("Ranking error %s: %s", persona.id, r1["error"])
            return None
        persona_ranking = _parse_ranking(r1, n=len(candidate))

    ndcg = ndcg_at_k(persona_ranking, gt_ranks, k=10)
    hits = hits_all_k(persona_ranking, gt_ranks, relevant_cutoff=3)
    tau = kendall_tau_score(persona_ranking, gt_ranks)

    # stars baseline: rank by stars ascending
    stars_order = sorted(range(len(candidate)), key=lambda i: stars_ranks.get(i, 99))
    stars_ndcg = ndcg_at_k(stars_order, gt_ranks, k=10)
    stars_hits = hits_all_k(stars_order, gt_ranks, relevant_cutoff=3)

    # ── Task 2: Pairwise (Phase 5 — full model, it's the discriminating signal) ──
    birank_top = candidate.sort_values(model_rank_col).iloc[0].to_dict()
    stars_top = candidate.sort_values("stars_rank").iloc[0].to_dict()

    pairwise_prompt = build_pairwise_prompt(persona, birank_top, stars_top, persona.domain)

    if dry_run:
        pairwise_win = int(rng.integers(0, 2))
    else:
        r2 = await call_llm(system, pairwise_prompt, temperature=temp, model=FULL_MODEL)
        pairwise_win = 1 if str(r2.get("choice", "")).strip().upper() == "A" else 0

    # ── Task 3: Revisit (full model — calibration anchor) ────────────────────
    revisit_target = candidate.iloc[0].to_dict()
    model_revisit_signal = float(
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
        "persona_id":          persona.id,
        "domain":              persona.domain,
        "archetype":           persona.archetype,
        "city":                persona.city,
        "ndcg":                ndcg,
        "hit_at_1":            hits["hit_at_1"],
        "hit_at_3":            hits["hit_at_3"],
        "hit_at_10":           hits["hit_at_10"],
        "kendall_tau":         tau,
        "stars_ndcg":          stars_ndcg,
        "stars_hit_at_1":      stars_hits["hit_at_1"],
        "stars_hit_at_3":      stars_hits["hit_at_3"],
        "pairwise_win":        pairwise_win,
        "revisit_score":       revisit_score,
        "model_revisit_signal": model_revisit_signal,
    }


# ── Phase 2: manipulation check + null + inverted ─────────────────────────────

async def run_phase2(dry_run: bool = False):
    print("\n[Phase 2] Running manipulation check...")
    try:
        from manipulation_check import (
            run_manipulation_check, analyze_manipulation_results,
            generate_null_personas, generate_inverted_personas,
            save_manipulation_report,
        )
    except ImportError as e:
        print(f"  manipulation_check.py not found: {e}")
        return

    all_personas = generate_all_personas()

    if dry_run:
        print("  [dry-run] skipping manipulation check API calls")
    else:
        results = await run_manipulation_check(all_personas, sample_n=150)
        analysis = analyze_manipulation_results(results)
        report_path = save_manipulation_report(analysis, RESULTS_DIR)
        print(f"  Manipulation check report → {report_path}")

        # Print quick verdict
        for domain, data in analysis.items():
            if not isinstance(data, dict):
                continue
            p = data.get("chi2_p_value", 1.0)
            verdict = "PASS" if p < 0.05 else "FAIL (archetypes not distinguishable)"
            print(f"  {domain}: chi2 p={p:.4f} → {verdict}")

    # Null personas baseline
    print("\n[Phase 2] Running null persona baseline (n=100)...")
    null_personas = generate_null_personas(n=100)
    for d in set(p.domain for p in null_personas):
        get_venues(d)

    null_records = []
    sem = asyncio.Semaphore(cfg.MAX_CONCURRENT)
    async def bounded_null(p):
        async with sem:
            return await run_persona_v2(p, dry_run=dry_run, use_tiered=True)

    for coro in atqdm(asyncio.as_completed([bounded_null(p) for p in null_personas]),
                      total=len(null_personas), desc="Null personas"):
        result = await coro
        if result:
            result["archetype"] = "NULL_BASELINE"
            null_records.append(result)

    _save_extra_records(null_records, "null_baseline_records.csv")

    # Inverted personas
    print("\n[Phase 2] Running inverted personas (n=100)...")
    all_personas_fresh = generate_all_personas()
    inverted = generate_inverted_personas(all_personas_fresh, n=100)
    inverted_records = []
    for coro in atqdm(asyncio.as_completed([bounded_null(p) for p in inverted]),
                      total=len(inverted), desc="Inverted personas"):
        result = await coro
        if result:
            inverted_records.append(result)

    _save_extra_records(inverted_records, "inverted_personas_records.csv")
    print(f"  Null NDCG: {np.mean([r['ndcg'] for r in null_records]):.4f}")
    print(f"  Inverted NDCG: {np.mean([r['ndcg'] for r in inverted_records]):.4f}")


# ── Phase 3: calibration + cross-domain ──────────────────────────────────────

def run_phase3():
    print("\n[Phase 3] Running calibration and cross-domain analysis...")
    try:
        from calibration_analysis import (
            compute_revisit_calibration, run_cross_domain_consistency,
            compute_per_persona_variance, save_calibration_report,
        )
    except ImportError as e:
        print(f"  calibration_analysis.py not found: {e}")
        return

    records_path = os.path.join(RESULTS_DIR, "simulation_records_v2.csv")
    if not os.path.exists(records_path):
        records_path = os.path.join(RESULTS_DIR, "simulation_records.csv")
    if not os.path.exists(records_path):
        print("  No records file found — run Phase 1 first.")
        return

    df = pd.read_csv(records_path)
    calibration = compute_revisit_calibration(df)
    cross_domain = run_cross_domain_consistency(df)
    variance = compute_per_persona_variance(df)
    save_calibration_report(calibration, cross_domain, variance, RESULTS_DIR)

    for domain, data in calibration.items():
        overall = data.get("overall", {})
        r = overall.get("spearman_r", float("nan"))
        p = overall.get("spearman_p", float("nan"))
        print(f"  {domain}: revisit Spearman r={r:.3f}, p={p:.4f}")


# ── Phase 5b: Claude replication ──────────────────────────────────────────────

async def run_phase5b(dry_run: bool = False):
    print("\n[Phase 5b] Claude Sonnet 4.6 replication (n=300)...")
    try:
        from second_model import run_claude_replication, compare_model_agreement, save_replication_report
    except ImportError as e:
        print(f"  second_model.py not found: {e}")
        return

    if dry_run:
        print("  [dry-run] skipping Claude API calls")
        return

    # Check Anthropic key
    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not ant_key:
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_file):
            for line in open(env_file):
                if line.startswith("ANTHROPIC_API_KEY="):
                    ant_key = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not ant_key:
            print("  ANTHROPIC_API_KEY not set — skipping Claude replication.")
            return

    claude_records = await run_claude_replication(n_per_archetype=25)
    openai_path = os.path.join(RESULTS_DIR, "simulation_records_v2.csv")
    claude_path = os.path.join(RESULTS_DIR, "claude_replication_records.csv")
    comparison = compare_model_agreement(openai_path, claude_path)
    save_replication_report(comparison, RESULTS_DIR)
    print(f"  Claude replication complete ({len(claude_records)} records).")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_extra_records(records: List[dict], filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(
    dry_run: bool = False,
    domain_filter: Optional[str] = None,
    phases: Optional[List[str]] = None,
    use_tiered: bool = True,
):
    run_all = not phases
    phases_set = set(phases or [])

    personas = generate_all_personas()
    if domain_filter:
        personas = [p for p in personas if p.domain == domain_filter]

    total = len(personas)
    domains = [domain_filter] if domain_filter else ["coffee", "restaurant", "hotel"]

    print(f"\n{'='*62}")
    print(f"  LLM Simulation v2 — Behavioral Venue Ranking Validation")
    print(f"{'='*62}")
    print(f"  Model (ranking):  {'gpt-5.4-mini (tiered)' if use_tiered else 'gpt-5.4 (full)'}")
    print(f"  Model (pairwise): {FULL_MODEL}")
    print(f"  Personas:         {total:,}")
    print(f"  Concurrency:      {cfg.MAX_CONCURRENT}")

    if not dry_run:
        ranking_calls = total if use_tiered else total
        full_calls = total * 2  # pairwise + revisit
        mini_cost = (ranking_calls * 700 / 1e6 * 0.40) + (ranking_calls * 150 / 1e6 * 1.60)
        full_cost = (full_calls * 700 / 1e6 * 2.0) + (full_calls * 150 / 1e6 * 8.0)
        print(f"  Est. cost:        ~${mini_cost + full_cost:.2f}")
        confirm = input("\nProceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    # ── Phase 1+4: main 1500-persona simulation ───────────────────────────────
    if run_all or "1" in phases_set or "4" in phases_set:
        print(f"\n[Phase 1+4] Running {total:,} personas with discriminating sets...")
        for d in domains:
            get_venues(d)
            print(f"  Loaded {d} venues ({len(_venues[d])} rows)")

        records: List[dict] = []
        errors = 0
        start = time.time()
        sem = asyncio.Semaphore(cfg.MAX_CONCURRENT)

        async def bounded(p: Persona):
            async with sem:
                return await run_persona_v2(p, dry_run=dry_run, use_tiered=use_tiered)

        tasks = [bounded(p) for p in personas]
        for coro in atqdm(asyncio.as_completed(tasks), total=total, desc="Personas v2"):
            result = await coro
            if result is not None:
                records.append(result)
            else:
                errors += 1

        elapsed = time.time() - start
        print(f"\nCompleted {len(records):,}/{total:,} in {elapsed:.1f}s ({errors} errors)")

        # Save v2 records
        records_path = os.path.join(RESULTS_DIR, "simulation_records_v2.csv")
        with open(records_path, "w", newline="") as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        print(f"  Records → {records_path}")

        # Compute metrics (Phase 4: BH correction, effect sizes built into evaluator)
        metrics = compute_metrics(records)
        metrics_path = os.path.join(RESULTS_DIR, "simulation_metrics_v2.json")
        with open(metrics_path, "w") as f:
            json.dump({f"{d}|{a}": v for (d, a), v in metrics.items()}, f, indent=2)

        generate_report(metrics, records, filename="simulation_report_v2.md")

        # Print summary
        _print_summary(metrics, domains, records)

    # ── Phase 2: manipulation check + null + inverted ──────────────────────────
    if run_all or "2" in phases_set:
        await run_phase2(dry_run=dry_run)

    # ── Phase 3: calibration ──────────────────────────────────────────────────
    if run_all or "3" in phases_set:
        run_phase3()

    # ── Phase 5b: Claude replication ──────────────────────────────────────────
    if (run_all or "5b" in phases_set):
        await run_phase5b(dry_run=dry_run)

    print(f"\nAll results in: {RESULTS_DIR}\n")


def _print_summary(metrics: dict, domains: List[str], records: List[dict]):
    all_archetypes = {r["archetype"] for r in records}
    print("\n" + "="*76)
    print("v2 SUMMARY (discriminating sets, Hit@1/3, Kendall τ, BH-corrected p)")
    print("="*76)
    print(f"{'Domain':<14} {'Archetype':<30} {'n':>4} {'NDCG':>7} {'Hit@1':>6} {'Hit@3':>6} {'τ':>6} {'p(BH)':>7}")
    print("-"*76)
    for domain in domains:
        for arch in list(PERSONA_COUNTS.get(domain, {}).keys()) + ["OVERALL"]:
            m = metrics.get((domain, arch))
            if not m:
                continue
            sig = "*" if (m.get("wilcoxon_p_bh") or 1.0) < 0.05 else " "
            print(
                f"{domain:<14} {arch:<30} {m['n']:>4} "
                f"{m['ndcg_mean']:>7.4f} {m['hit_at_1']:>6.3f} {m['hit_at_3']:>6.3f} "
                f"{m['kendall_tau']:>6.3f} {(m.get('wilcoxon_p_bh') or 1.0):>6.4f}{sig}"
            )
        print("-"*76)

    all_m = metrics.get(("ALL", "OVERALL"), {})
    if all_m:
        print(f"\n  Overall NDCG@10:    {all_m['ndcg_mean']:.4f} [{all_m['ndcg_lo']:.4f}, {all_m['ndcg_hi']:.4f}]")
        print(f"  Overall Hit@1:      {all_m['hit_at_1']:.4f}  Hit@3: {all_m['hit_at_3']:.4f}")
        print(f"  Overall Kendall τ:  {all_m['kendall_tau']:.4f}")
        print(f"  Cohen's d:          {all_m['cohen_d']:.4f}  Rank-biserial r: {all_m['rank_biserial']:.4f}")
        print(f"  Pairwise win rate:  {all_m['pairwise_win_rate']:.1%}")
        p_bh = all_m.get("wilcoxon_p_bh") or 1.0
        p_raw = all_m.get("wilcoxon_p") or 1.0
        print(f"  Wilcoxon p (raw):   {p_raw:.4f}  p (BH-corrected): {p_bh:.4f} "
              f"{'✓ significant' if p_bh < 0.05 else '✗ not significant'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Simulation v2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--domain", choices=["coffee", "restaurant", "hotel"])
    parser.add_argument("--phases", help="Comma-separated phases to run, e.g. '1,4'")
    parser.add_argument("--tiered", action="store_true",
                        help="Use gpt-5.4-mini for ranking, gpt-5.4 for pairwise (cost saving)")
    args = parser.parse_args()
    phase_list = args.phases.split(",") if args.phases else None
    asyncio.run(main(
        dry_run=args.dry_run,
        domain_filter=args.domain,
        phases=phase_list,
        use_tiered=args.tiered,   # default: gpt-5.4 for everything
    ))
