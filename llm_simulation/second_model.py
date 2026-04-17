"""
second_model.py — Claude Sonnet 4.6 replication run for the LLM simulation study.

Provides a second-model replication check to validate that behavioural venue
ranking results hold across different LLM providers/model families.

Usage:
    import asyncio
    from second_model import run_claude_replication, compare_model_agreement, save_replication_report
    records = asyncio.run(run_claude_replication())
"""
import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# ── env / API key setup ───────────────────────────────────────────────────────
_ANTHROPIC_KEY_NAME = "ANTHROPIC_API_KEY"
if not os.environ.get(_ANTHROPIC_KEY_NAME):
    _env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_file):
        with open(_env_file) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith(_ANTHROPIC_KEY_NAME + "="):
                    os.environ[_ANTHROPIC_KEY_NAME] = (
                        _line.split("=", 1)[1].strip().strip('"').strip("'")
                    )
                    break

import anthropic

from config import CACHE_PATH, PERSONA_COUNTS, RANDOM_SEED, RESULTS_DIR
from data_loader import (
    load_coffee_venues,
    load_hotel_venues,
    load_restaurant_venues,
    sample_candidate_set,
)
from evaluator import ndcg_at_k, hit_at_k
from persona_generator import Persona, generate_all_personas
from prompts import (
    build_pairwise_prompt,
    build_ranking_prompt,
    build_revisit_prompt,
    build_system_prompt,
)

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 1024
CLAUDE_CONCURRENCY = 10
CLAUDE_TEMPERATURE = 0.8

# Key prefix used in the shared SQLite cache so Claude results don't collide
# with OpenAI results that share the same prompt text.
_CACHE_PREFIX = "claude|"

# ── SQLite cache (shared DB, separate key namespace) ─────────────────────────

def _init_cache() -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache "
        "(key TEXT PRIMARY KEY, response TEXT, created_at REAL)"
    )
    conn.commit()
    conn.close()


def _cache_key(system: str, user: str) -> str:
    raw = f"{_CACHE_PREFIX}{CLAUDE_MODEL}|{system}|{user}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    conn = sqlite3.connect(CACHE_PATH)
    row = conn.execute(
        "SELECT response FROM cache WHERE key=?", (key,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _cache_put(key: str, response: str) -> None:
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, response, created_at) VALUES (?,?,?)",
        (key, response, time.time()),
    )
    conn.commit()
    conn.close()


_init_cache()

# ── Lazy client singleton ─────────────────────────────────────────────────────

_claude_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _claude_client
    if _claude_client is None:
        api_key = os.environ.get(_ANTHROPIC_KEY_NAME, "")
        _claude_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _claude_client


# ── JSON extraction helper ────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """Try to parse JSON from model output. Claude may wrap JSON in markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first and last fence lines
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip().startswith("```"):
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to find the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


# ── Core async Claude caller ──────────────────────────────────────────────────

async def call_claude(
    system_prompt: str,
    user_prompt: str,
    temperature: float = CLAUDE_TEMPERATURE,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Call Claude Sonnet 4.6 with SQLite caching and exponential backoff.

    Returns a parsed JSON dict. On unrecoverable error returns {"error": <reason>}.
    The cache key is prefixed with "claude|" to avoid collisions with OpenAI entries
    in the shared response_cache.db.
    """
    key = _cache_key(system_prompt, user_prompt)
    cached = _cache_get(key)
    if cached:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            pass  # corrupt cache entry — fall through and re-fetch

    client = _get_client()

    for attempt in range(retries):
        try:
            response = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text if response.content else ""
            parsed = _extract_json(text)
            if parsed is None:
                logger.warning(
                    "JSON parse failed on attempt %d. Raw text: %.200s",
                    attempt + 1,
                    text,
                )
                if attempt == retries - 1:
                    return {"error": "json_parse_failed"}
                await asyncio.sleep(1)
                continue

            _cache_put(key, json.dumps(parsed))
            return parsed

        except anthropic.RateLimitError:
            wait = 2 ** attempt + 1
            logger.warning(
                "Claude rate limit hit (attempt %d/%d), waiting %ds",
                attempt + 1,
                retries,
                wait,
            )
            await asyncio.sleep(wait)

        except anthropic.APIStatusError as exc:
            logger.warning("Claude API status error (attempt %d): %s", attempt + 1, exc)
            if attempt == retries - 1:
                return {"error": str(exc)}
            await asyncio.sleep(2 ** attempt)

        except anthropic.APIConnectionError as exc:
            logger.warning(
                "Claude API connection error (attempt %d): %s", attempt + 1, exc
            )
            if attempt == retries - 1:
                return {"error": str(exc)}
            await asyncio.sleep(2 ** attempt)

    return {"error": "max_retries_exceeded"}


# ── Venue cache (avoid repeated I/O) ─────────────────────────────────────────

_venues_cache: Dict[str, pd.DataFrame] = {}


def _get_venues(domain: str) -> pd.DataFrame:
    if domain not in _venues_cache:
        loader = {
            "coffee": load_coffee_venues,
            "restaurant": load_restaurant_venues,
            "hotel": load_hotel_venues,
        }[domain]
        _venues_cache[domain] = loader()
    return _venues_cache[domain]


# ── Ranking response parser (mirrors main.py logic) ──────────────────────────

def _parse_ranking(response: dict, n: int = 10) -> List[int]:
    raw = response.get("ranking", [])
    if not isinstance(raw, list):
        return list(range(n))
    result: List[int] = []
    seen: set = set()
    for v in raw:
        try:
            idx = int(v) - 1  # LLM returns 1-indexed
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

async def _run_persona_tasks_claude(
    persona: Persona,
    rng: np.random.Generator,
    semaphore: asyncio.Semaphore,
) -> Optional[dict]:
    """Run ranking + pairwise + revisit tasks for a single persona via Claude."""
    async with semaphore:
        venues = _get_venues(persona.domain)
        system = build_system_prompt(persona)

        # ── Task 1: Ranking ───────────────────────────────────────────────────
        candidate = sample_candidate_set(venues, n=10, rng=rng)
        model_rank_col = "model_rank" if "model_rank" in candidate.columns else "rank"
        gt_ranks = dict(enumerate(candidate[model_rank_col].tolist()))
        stars_ranks = dict(enumerate(candidate["stars_rank"].tolist()))

        ranking_prompt = build_ranking_prompt(persona, candidate, persona.domain)
        r1 = await call_claude(system, ranking_prompt, temperature=CLAUDE_TEMPERATURE)
        if "error" in r1:
            logger.warning(
                "Ranking error for persona %s: %s", persona.id, r1["error"]
            )
            return None

        persona_ranking = _parse_ranking(r1, n=len(candidate))
        ndcg = ndcg_at_k(persona_ranking, gt_ranks, k=10)
        hit = hit_at_k(persona_ranking, gt_ranks, k=10)
        stars_ndcg = ndcg_at_k(list(range(len(candidate))), stars_ranks, k=10)

        # ── Task 2: Pairwise ──────────────────────────────────────────────────
        birank_top = candidate.sort_values(model_rank_col).iloc[0].to_dict()
        stars_top = candidate.sort_values("stars_rank").iloc[0].to_dict()
        pairwise_prompt = build_pairwise_prompt(
            persona, birank_top, stars_top, persona.domain
        )
        r2 = await call_claude(system, pairwise_prompt, temperature=CLAUDE_TEMPERATURE)
        if "error" in r2:
            pairwise_win = 0
        else:
            choice = str(r2.get("choice", "")).strip().upper()
            pairwise_win = 1 if choice == "A" else 0  # A = BiRank top

        # ── Task 3: Revisit ───────────────────────────────────────────────────
        revisit_target = candidate.iloc[0].to_dict()
        model_revisit_signal = float(
            revisit_target.get(
                "revisit_rate", revisit_target.get("multi_stay_rate", 0.0)
            )
            or 0.0
        )
        revisit_prompt = build_revisit_prompt(persona, revisit_target, persona.domain)
        r3 = await call_claude(system, revisit_prompt, temperature=CLAUDE_TEMPERATURE)
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
            "model": CLAUDE_MODEL,
        }


# ── Stratified persona sampler ────────────────────────────────────────────────

def _build_stratified_sample(
    all_personas: List[Persona],
    n_per_archetype: int,
    domains: List[str],
    rng: np.random.Generator,
) -> List[Persona]:
    """
    Return n_per_archetype personas per (domain, archetype) combination.
    Personas are drawn without replacement; if fewer than n_per_archetype
    exist for a stratum all available are included.
    """
    from collections import defaultdict

    buckets: Dict[tuple, List[Persona]] = defaultdict(list)
    for p in all_personas:
        if p.domain in domains:
            buckets[(p.domain, p.archetype)].append(p)

    selected: List[Persona] = []
    for key, bucket in buckets.items():
        # Shuffle deterministically within stratum using numpy rng
        indices = rng.permutation(len(bucket)).tolist()
        take = min(n_per_archetype, len(bucket))
        selected.extend(bucket[i] for i in indices[:take])

    # Shuffle final list so domain/archetype ordering is interleaved
    final_indices = rng.permutation(len(selected)).tolist()
    return [selected[i] for i in final_indices]


# ── Main replication runner ───────────────────────────────────────────────────

async def run_claude_replication(
    n_per_archetype: int = 25,
    domains: Optional[List[str]] = None,
) -> List[dict]:
    """
    Run a stratified replication of the simulation using Claude Sonnet 4.6.

    Parameters
    ----------
    n_per_archetype : int
        Number of personas to sample per (domain, archetype) stratum.
        Default 25 gives 25 × 4 archetypes × 3 domains = 300 personas.
    domains : list of str, optional
        Domains to include. Defaults to ["coffee", "restaurant", "hotel"].

    Returns
    -------
    list of dict
        Same schema as main simulation records, plus "model" key.
        Results are also written to results/claude_replication_records.csv.
    """
    if domains is None:
        domains = ["coffee", "restaurant", "hotel"]

    rng = np.random.default_rng(RANDOM_SEED)
    all_personas = generate_all_personas()
    sample = _build_stratified_sample(all_personas, n_per_archetype, domains, rng)

    # Pre-load venue data (avoids repeated I/O inside async tasks)
    for domain in domains:
        _get_venues(domain)

    semaphore = asyncio.Semaphore(CLAUDE_CONCURRENCY)
    total = len(sample)

    logger.info(
        "Claude replication: %d personas across domains %s "
        "(concurrency=%d, model=%s)",
        total,
        domains,
        CLAUDE_CONCURRENCY,
        CLAUDE_MODEL,
    )

    tasks = [
        _run_persona_tasks_claude(persona, rng, semaphore) for persona in sample
    ]

    records: List[dict] = []
    errors = 0
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        if result is not None:
            records.append(result)
        else:
            errors += 1
        if completed % 50 == 0 or completed == total:
            logger.info(
                "Progress: %d/%d complete (%d errors)", completed, total, errors
            )

    logger.info(
        "Claude replication finished: %d records, %d errors", len(records), errors
    )

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "claude_replication_records.csv")
    if records:
        pd.DataFrame(records).to_csv(output_path, index=False)
        logger.info("Saved %d records to %s", len(records), output_path)

    return records


# ── Model agreement comparison ────────────────────────────────────────────────

def compare_model_agreement(
    openai_records_path: str,
    claude_records_path: str,
) -> Dict[str, Any]:
    """
    Compare GPT-4.1 and Claude Sonnet 4.6 results across archetypes.

    Parameters
    ----------
    openai_records_path : str
        Path to the main simulation CSV (produced by report_generator.save_records).
    claude_records_path : str
        Path to the Claude replication CSV (produced by run_claude_replication).

    Returns
    -------
    dict
        Nested structure: {domain: {archetype: {gpt41_ndcg, claude_ndcg, delta,
        agreement_kappa, spearman_r, n_shared}}}
        Also includes a top-level "overall" key with aggregate statistics.
    """
    from scipy.stats import spearmanr

    gpt_df = pd.read_csv(openai_records_path)
    claude_df = pd.read_csv(claude_records_path)

    # Normalise column names
    for df in (gpt_df, claude_df):
        if "pairwise_win" not in df.columns and "pairwise" in df.columns:
            df.rename(columns={"pairwise": "pairwise_win"}, inplace=True)

    results: Dict[str, Any] = {}
    all_gpt_ndcg: List[float] = []
    all_claude_ndcg: List[float] = []
    all_gpt_pair: List[int] = []
    all_claude_pair: List[int] = []

    domains = sorted(set(gpt_df["domain"].unique()) & set(claude_df["domain"].unique()))

    for domain in domains:
        gpt_d = gpt_df[gpt_df["domain"] == domain]
        claude_d = claude_df[claude_df["domain"] == domain]

        archetypes = sorted(
            set(gpt_d["archetype"].unique()) & set(claude_d["archetype"].unique())
        )
        results[domain] = {}

        for arch in archetypes:
            gpt_a = gpt_d[gpt_d["archetype"] == arch]
            claude_a = claude_d[claude_d["archetype"] == arch]

            gpt_ndcg_vals = gpt_a["ndcg"].dropna().tolist()
            claude_ndcg_vals = claude_a["ndcg"].dropna().tolist()
            gpt_pair_vals = gpt_a["pairwise_win"].dropna().tolist()
            claude_pair_vals = claude_a["pairwise_win"].dropna().tolist()

            gpt_ndcg_mean = float(np.mean(gpt_ndcg_vals)) if gpt_ndcg_vals else float("nan")
            claude_ndcg_mean = float(np.mean(claude_ndcg_vals)) if claude_ndcg_vals else float("nan")
            delta = claude_ndcg_mean - gpt_ndcg_mean

            # Spearman correlation on per-persona NDCG
            # Align by persona_id where both models ran the same persona
            gpt_indexed = gpt_a.set_index("persona_id")["ndcg"].dropna()
            claude_indexed = claude_a.set_index("persona_id")["ndcg"].dropna()
            shared_ids = list(set(gpt_indexed.index) & set(claude_indexed.index))
            if len(shared_ids) >= 3:
                gpt_shared = [gpt_indexed[pid] for pid in shared_ids]
                claude_shared = [claude_indexed[pid] for pid in shared_ids]
                spearman_r, spearman_p = spearmanr(gpt_shared, claude_shared)
                spearman_r = float(spearman_r)
                spearman_p = float(spearman_p)
            else:
                spearman_r = float("nan")
                spearman_p = float("nan")

            # Cohen's kappa on pairwise win/loss
            kappa = _cohens_kappa_pairwise(gpt_pair_vals, claude_pair_vals)

            results[domain][arch] = {
                "gpt41_ndcg": round(gpt_ndcg_mean, 4),
                "claude_ndcg": round(claude_ndcg_mean, 4),
                "delta": round(delta, 4),
                "agreement_kappa": round(kappa, 4) if not np.isnan(kappa) else None,
                "spearman_r": round(spearman_r, 4) if not np.isnan(spearman_r) else None,
                "spearman_p": round(spearman_p, 4) if not np.isnan(spearman_p) else None,
                "n_gpt": len(gpt_ndcg_vals),
                "n_claude": len(claude_ndcg_vals),
                "n_shared": len(shared_ids),
            }

            all_gpt_ndcg.extend(gpt_ndcg_vals)
            all_claude_ndcg.extend(claude_ndcg_vals)
            all_gpt_pair.extend(gpt_pair_vals)
            all_claude_pair.extend(claude_pair_vals)

    # Overall aggregate
    overall_gpt = float(np.mean(all_gpt_ndcg)) if all_gpt_ndcg else float("nan")
    overall_claude = float(np.mean(all_claude_ndcg)) if all_claude_ndcg else float("nan")
    overall_kappa = _cohens_kappa_pairwise(all_gpt_pair, all_claude_pair)
    if len(all_gpt_ndcg) >= 3 and len(all_claude_ndcg) >= 3:
        min_len = min(len(all_gpt_ndcg), len(all_claude_ndcg))
        from scipy.stats import spearmanr as _spearmanr
        overall_spearman_r, overall_spearman_p = _spearmanr(
            all_gpt_ndcg[:min_len], all_claude_ndcg[:min_len]
        )
    else:
        overall_spearman_r = float("nan")
        overall_spearman_p = float("nan")

    results["overall"] = {
        "gpt41_ndcg": round(overall_gpt, 4),
        "claude_ndcg": round(overall_claude, 4),
        "delta": round(overall_claude - overall_gpt, 4),
        "agreement_kappa": round(overall_kappa, 4) if not np.isnan(overall_kappa) else None,
        "spearman_r": round(float(overall_spearman_r), 4) if not np.isnan(overall_spearman_r) else None,
        "spearman_p": round(float(overall_spearman_p), 4) if not np.isnan(overall_spearman_p) else None,
        "n_gpt": len(all_gpt_ndcg),
        "n_claude": len(all_claude_ndcg),
    }

    return results


def _cohens_kappa_pairwise(
    ratings_a: List[Any], ratings_b: List[Any]
) -> float:
    """
    Compute Cohen's kappa on binary pairwise win/loss vectors.

    If the vectors have different lengths, compute kappa on marginal agreement
    (no paired structure assumed): use the shorter length and compare element-wise.
    If both vectors are empty or all-identical, return nan.
    """
    a = [int(x) for x in ratings_a]
    b = [int(x) for x in ratings_b]

    if not a or not b:
        return float("nan")

    # Align by truncating to the shorter length
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]

    if n < 2:
        return float("nan")

    # Observed agreement
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n

    # Expected agreement
    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    p_e = (p_a1 * p_b1) + ((1 - p_a1) * (1 - p_b1))

    if abs(1 - p_e) < 1e-9:
        return float("nan")

    kappa = (p_o - p_e) / (1 - p_e)
    return float(kappa)


# ── Replication report writer ─────────────────────────────────────────────────

def save_replication_report(
    comparison: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Write a markdown table summarising model agreement to
    {output_dir}/replication_report.md.

    Verdict: "REPLICATED" if kappa > 0.4 AND |NDCG delta| < 0.02 for all
    (domain, archetype) strata; otherwise "DIVERGENT — investigate".
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "replication_report.md")

    # Determine per-stratum verdict
    divergent_strata: List[str] = []
    rows: List[str] = []

    rows.append(
        "| Domain | Archetype | GPT-4.1 NDCG | Claude NDCG | Delta | "
        "Kappa | Spearman r | Verdict |"
    )
    rows.append("|--------|-----------|:---:|:---:|:---:|:---:|:---:|:---:|")

    for domain, archetypes in comparison.items():
        if domain == "overall":
            continue
        if not isinstance(archetypes, dict):
            continue
        for arch, stats in archetypes.items():
            if not isinstance(stats, dict):
                continue
            gpt_ndcg = stats.get("gpt41_ndcg")
            claude_ndcg = stats.get("claude_ndcg")
            delta = stats.get("delta")
            kappa = stats.get("agreement_kappa")
            spearman = stats.get("spearman_r")

            # Verdict logic
            kappa_ok = kappa is not None and kappa > 0.4
            delta_ok = delta is not None and abs(delta) < 0.02
            stratum_verdict = "REPLICATED" if (kappa_ok and delta_ok) else "DIVERGENT"
            if stratum_verdict == "DIVERGENT":
                divergent_strata.append(f"{domain}/{arch}")

            gpt_str = f"{gpt_ndcg:.4f}" if gpt_ndcg is not None else "n/a"
            claude_str = f"{claude_ndcg:.4f}" if claude_ndcg is not None else "n/a"
            delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
            kappa_str = f"{kappa:.3f}" if kappa is not None else "n/a"
            spearman_str = f"{spearman:.3f}" if spearman is not None else "n/a"

            rows.append(
                f"| {domain} | {arch} | {gpt_str} | {claude_str} | "
                f"{delta_str} | {kappa_str} | {spearman_str} | {stratum_verdict} |"
            )

    # Overall row
    overall = comparison.get("overall", {})
    if overall:
        ov_gpt = overall.get("gpt41_ndcg")
        ov_claude = overall.get("claude_ndcg")
        ov_delta = overall.get("delta")
        ov_kappa = overall.get("agreement_kappa")
        ov_spearman = overall.get("spearman_r")
        ov_kappa_ok = ov_kappa is not None and ov_kappa > 0.4
        ov_delta_ok = ov_delta is not None and abs(ov_delta) < 0.02
        ov_verdict = "REPLICATED" if (ov_kappa_ok and ov_delta_ok) else "DIVERGENT"

        rows.append("|--------|-----------|:---:|:---:|:---:|:---:|:---:|:---:|")
        rows.append(
            f"| **ALL** | **OVERALL** | "
            f"{ov_gpt:.4f if ov_gpt is not None else 'n/a'} | "
            f"{ov_claude:.4f if ov_claude is not None else 'n/a'} | "
            f"{ov_delta:+.4f if ov_delta is not None else 'n/a'} | "
            f"{ov_kappa:.3f if ov_kappa is not None else 'n/a'} | "
            f"{ov_spearman:.3f if ov_spearman is not None else 'n/a'} | "
            f"**{ov_verdict}** |"
        )

    # Final verdict
    if not divergent_strata:
        verdict_line = (
            "**Overall Verdict: REPLICATED** — "
            "Both models agree across all archetypes "
            "(kappa > 0.4, |NDCG delta| < 0.02 for all strata)."
        )
    else:
        strata_str = ", ".join(divergent_strata)
        verdict_line = (
            f"**Overall Verdict: DIVERGENT — investigate** — "
            f"Agreement criteria not met for: {strata_str}. "
            f"Check kappa < 0.4 or |NDCG delta| >= 0.02 in the table above."
        )

    lines = [
        "# LLM Simulation — Second-Model Replication Report",
        "",
        f"Replication model: `{CLAUDE_MODEL}` (Anthropic)  ",
        f"Reference model: `gpt-5.4` (OpenAI)  ",
        "",
        "## Agreement Table",
        "",
        *rows,
        "",
        "## Verdict",
        "",
        verdict_line,
        "",
        "### Interpretation",
        "",
        "- **Kappa > 0.4** — moderate or better agreement on pairwise win/loss choices",
        "- **|NDCG delta| < 0.02** — NDCG@10 scores are within 2 percentage points",
        "- Both criteria must hold for a stratum to be REPLICATED",
        "",
    ]

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("Replication report saved to %s", report_path)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Claude Sonnet 4.6 replication run for LLM simulation study"
    )
    parser.add_argument(
        "--n-per-archetype",
        type=int,
        default=25,
        help="Personas per (domain, archetype) stratum (default: 25)",
    )
    parser.add_argument(
        "--domain",
        choices=["coffee", "restaurant", "hotel"],
        default=None,
        help="Restrict to a single domain (default: all three)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "After running, compare against GPT-4.1 records "
            "(requires results/simulation_records.csv)"
        ),
    )
    args = parser.parse_args()

    api_key = os.environ.get(_ANTHROPIC_KEY_NAME, "")
    if not api_key:
        import sys
        print(
            f"ERROR: {_ANTHROPIC_KEY_NAME} not set. "
            "Create llm_simulation/.env with ANTHROPIC_API_KEY=sk-ant-..."
        )
        sys.exit(1)

    domains_arg: Optional[List[str]] = [args.domain] if args.domain else None

    records = asyncio.run(
        run_claude_replication(
            n_per_archetype=args.n_per_archetype,
            domains=domains_arg,
        )
    )

    print(f"\nClaude replication complete: {len(records)} records")

    if args.compare:
        gpt_path = os.path.join(RESULTS_DIR, "simulation_records.csv")
        claude_path = os.path.join(RESULTS_DIR, "claude_replication_records.csv")
        if not os.path.exists(gpt_path):
            print(f"WARNING: GPT records not found at {gpt_path} — skipping comparison.")
        else:
            comparison = compare_model_agreement(gpt_path, claude_path)
            save_replication_report(comparison, RESULTS_DIR)
            print(f"Replication report saved to {RESULTS_DIR}/replication_report.md")

            overall = comparison.get("overall", {})
            print(
                f"\nOverall: GPT-4.1 NDCG={overall.get('gpt41_ndcg', 'n/a')}, "
                f"Claude NDCG={overall.get('claude_ndcg', 'n/a')}, "
                f"kappa={overall.get('agreement_kappa', 'n/a')}"
            )
