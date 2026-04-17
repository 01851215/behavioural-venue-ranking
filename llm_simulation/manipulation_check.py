"""
Manipulation check module for the behavioural venue-ranking study.

Validates whether LLM personas actually behave like their assigned
behavioural archetype by:
  1. Asking diagnostic questions and scoring keyword-based authenticity.
  2. Running chi-squared tests per domain to confirm inter-archetype
     answer differentiation.
  3. Generating null (no archetype) and inverted (swapped archetype)
     control personas for downstream comparison.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2_contingency

from config import RESULTS_DIR
from persona_generator import Persona
from prompts import build_system_prompt
from task_runner import call_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Diagnostic questions (3 per domain)
# ---------------------------------------------------------------------------

_DIAGNOSTIC_QUESTIONS: Dict[str, List[str]] = {
    "coffee": [
        "How many different coffee shops did you visit in the past 6 months?",
        "When did you last try a café you had never been to before?",
        "What matters most when choosing a café — habit/familiarity, "
        "convenience, discovery, or safety/reliability?",
    ],
    "restaurant": [
        "How often do you eat at the same restaurant more than twice a month?",
        "When choosing dinner, how important is trying somewhere you "
        "have not been before?",
        "Would you travel 30 minutes for a restaurant you have not tried, "
        "or pick something closer and familiar?",
    ],
    "hotel": [
        "Do you typically book the same hotel chain across different cities?",
        "What is more important to you — atmosphere/experience or "
        "reliability/consistency?",
        "How many different cities did you stay in last year?",
    ],
}

# ---------------------------------------------------------------------------
# 2. Expected keyword signals per archetype
# ---------------------------------------------------------------------------

# Each entry is a list of keyword strings (lower-case).  An answer is
# considered a "signal match" if it contains at least one of the keywords.
# We require a match on at least 2 of the 3 questions.

_ARCHETYPE_SIGNALS: Dict[str, Dict[str, List[str]]] = {
    "coffee": {
        "Loyalist": [
            ["same", "usual", "routine", "regular", "every day", "daily", "one", "two", "familiar"],
            ["same", "usual", "routine", "regular", "months ago", "rarely", "never", "long"],
            ["habit", "familiarity", "routine", "familiar", "same"],
        ],
        "Weekday Regular": [
            ["two", "three", "few", "handful", "couple", "commute", "office", "work"],
            ["occasionally", "sometimes", "route", "nearby", "few weeks", "week"],
            ["convenience", "efficient", "quick", "fast", "nearby", "close", "commute"],
        ],
        "Casual Weekender": [
            ["many", "several", "various", "lots", "different", "new", "explore", "discover"],
            ["recently", "last week", "yesterday", "weekend", "just", "this week"],
            ["discovery", "ambience", "vibe", "atmosphere", "new", "interesting", "explore"],
        ],
        "Infrequent Visitor": [
            ["one", "two", "rarely", "occasionally", "seldom", "infrequent", "few"],
            ["months ago", "long time", "rare", "can't remember", "while", "rarely"],
            ["safety", "reliable", "reliability", "safe", "ratings", "stars", "crowd"],
        ],
    },
    "restaurant": {
        "Loyalist": [
            ["always", "same", "regularly", "often", "weekly", "frequently", "every week"],
            ["not very", "somewhat", "don't care", "prefer familiar", "same place", "not important"],
            ["closer", "familiar", "usual", "nearby", "regular", "always go"],
        ],
        "Explorer": [
            ["rarely", "never", "seldom", "avoid", "try not to"],
            ["very", "extremely", "crucial", "essential", "always", "priority", "love trying"],
            ["30 minutes", "travel", "go far", "worth it", "new place", "haven't tried"],
        ],
        "Mixed / Average": [
            ["sometimes", "occasionally", "few times", "depends", "varies"],
            ["somewhat", "somewhat important", "moderate", "depends", "if ratings"],
            ["depends", "proximity", "closer", "both", "balance"],
        ],
        "Nightlife Seeker": [
            ["rarely", "seldom", "not often", "usually try", "different"],
            ["not that", "atmosphere", "buzz", "vibe", "energy", "trendy"],
            ["new place", "travel", "don't mind", "ride", "uber", "location doesn't"],
        ],
    },
    "hotel": {
        "One-Time Tourist (Business)": [
            ["yes", "same chain", "loyalty", "points", "always", "usually", "typically"],
            ["reliability", "consistent", "consistency", "reliable", "professional", "clean"],
            ["many", "frequently", "often", "every month", "dozens", "regularly"],
        ],
        "Leisure Traveler": [
            ["sometimes", "occasionally", "depends", "if loved it", "special", "revisit"],
            ["atmosphere", "experience", "feel", "ambience", "special", "memorable"],
            ["few", "several", "handful", "two or three", "selected"],
        ],
        "One-Time Tourist": [
            ["no", "rarely", "not usually", "depends", "wherever", "any"],
            ["both", "balance", "somewhat", "depends", "comfortable", "well-located"],
            ["few", "one", "two", "three", "couple", "occasionally"],
        ],
        "Budget Explorer": [
            ["no", "different", "variety", "avoid chains", "mix", "airbnb", "hostel"],
            ["value", "value-for-money", "cost", "price", "budget", "cheap", "affordable"],
            ["many", "lots", "numerous", "multiple", "several", "five", "six", "seven"],
        ],
    },
}

# ---------------------------------------------------------------------------
# 3. Archetype swap map for inverted personas
# ---------------------------------------------------------------------------

_ARCHETYPE_SWAP: Dict[str, Dict[str, str]] = {
    "coffee": {
        "Loyalist": "Casual Weekender",
        "Weekday Regular": "Infrequent Visitor",
        "Casual Weekender": "Loyalist",
        "Infrequent Visitor": "Weekday Regular",
    },
    "restaurant": {
        "Loyalist": "Explorer",
        "Explorer": "Loyalist",
        "Mixed / Average": "Nightlife Seeker",
        "Nightlife Seeker": "Mixed / Average",
    },
    "hotel": {
        "One-Time Tourist (Business)": "Budget Explorer",
        "Leisure Traveler": "One-Time Tourist",
        "One-Time Tourist": "Leisure Traveler",
        "Budget Explorer": "One-Time Tourist (Business)",
    },
}

# Narrative profiles used by generate_null_personas
_NULL_BEHAVIORAL_PROFILES: Dict[str, str] = {
    "coffee": (
        "You are an ordinary person who visits coffee shops occasionally. "
        "You have no strong preferences or routines around cafés."
    ),
    "restaurant": (
        "You are an ordinary person who visits restaurants occasionally. "
        "You have no strong preferences or routines around dining out."
    ),
    "hotel": (
        "You are an ordinary person who stays in hotels occasionally. "
        "You have no strong preferences or routines around accommodation."
    ),
}

_NULL_TASK_CONTEXTS: Dict[str, str] = {
    "coffee": "You want to grab a coffee.",
    "restaurant": "You are looking for somewhere to eat.",
    "hotel": "You need to book a hotel.",
}

# Pulled from persona_generator to avoid a circular dependency on the full
# generator; only first names and occupations are needed here.
_FIRST_NAMES = [
    "Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Avery", "Quinn",
    "Skylar", "Blake", "Drew", "Reese", "Finley", "Logan", "Harper", "Emery",
    "Rowan", "Peyton", "Hayden", "Cameron", "Sage", "Dakota", "River", "Sloane",
    "Phoenix", "Kai", "Remy", "Ellis", "Noel", "Shea", "Luca", "Zara", "Milo",
    "Nadia", "Caden", "Layla", "Isaac", "Sofia", "Eli", "Priya", "Omar", "Mei",
    "Rafael", "Aisha", "Marcus", "Yuna", "Dev", "Leila", "Tobias", "Ingrid",
]
_CITIES = [
    "Philadelphia, PA", "Las Vegas, NV", "Phoenix, AZ", "Pittsburgh, PA",
    "Nashville, TN", "Charlotte, NC", "Cleveland, OH", "New Orleans, LA",
    "Tampa, FL", "Indianapolis, IN", "Louisville, KY", "St. Louis, MO",
    "Kansas City, MO", "Tucson, AZ", "Cincinnati, OH", "Reno, NV",
]
_OCCUPATIONS = [
    "software engineer", "nurse", "accountant", "teacher", "marketing manager",
    "graphic designer", "sales rep", "data analyst", "lawyer", "chef",
    "project manager", "PhD student", "small business owner", "barista",
    "consultant", "journalist", "product manager", "architect", "doctor",
    "HR specialist", "financial analyst", "social worker", "real estate agent",
]

# ---------------------------------------------------------------------------
# 4. Stratified sampler
# ---------------------------------------------------------------------------


def _stratified_sample(
    personas: List[Persona],
    sample_n: int,
    rng: Optional[random.Random] = None,
) -> List[Persona]:
    """Return up to *sample_n* personas, equal count from each archetype."""
    if rng is None:
        rng = random.Random(42)

    # Group by (domain, archetype)
    groups: Dict[Tuple[str, str], List[Persona]] = {}
    for p in personas:
        key = (p.domain, p.archetype)
        groups.setdefault(key, []).append(p)

    n_groups = len(groups)
    per_group = max(1, sample_n // n_groups)

    sampled: List[Persona] = []
    for key, group in groups.items():
        rng.shuffle(group)
        sampled.extend(group[:per_group])

    rng.shuffle(sampled)
    return sampled[:sample_n]


# ---------------------------------------------------------------------------
# 5. Prompt builder for diagnostic questions
# ---------------------------------------------------------------------------


def _build_diagnostic_user_prompt(domain: str) -> str:
    questions = _DIAGNOSTIC_QUESTIONS[domain]
    q_block = "\n".join(
        f"{i + 1}. {q}" for i, q in enumerate(questions)
    )
    return (
        "Please answer the following questions honestly, as yourself. "
        "Return your answers as a JSON object with a single key \"answers\" "
        "containing a list of exactly 3 strings — one per question, in order.\n\n"
        f"{q_block}\n\n"
        "Example format:\n"
        '{"answers": ["answer to Q1", "answer to Q2", "answer to Q3"]}'
    )


# ---------------------------------------------------------------------------
# 6. run_manipulation_check
# ---------------------------------------------------------------------------


async def run_manipulation_check(
    personas: List[Persona],
    sample_n: int = 150,
) -> List[Dict[str, Any]]:
    """
    Sample *sample_n* personas (stratified by archetype), ask 3 domain-specific
    diagnostic questions each, and return a list of result dicts.

    Each result dict has keys:
        persona_id  : str
        domain      : str
        archetype   : str
        answers     : List[str]  (3 elements; may be empty strings on parse error)
    """
    sampled = _stratified_sample(personas, sample_n)
    logger.info(
        "Running manipulation check on %d personas (requested %d).",
        len(sampled), sample_n,
    )

    async def _check_one(persona: Persona) -> Dict[str, Any]:
        system_prompt = build_system_prompt(persona)
        user_prompt = _build_diagnostic_user_prompt(persona.domain)
        try:
            response = await call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                retries=4,
            )
            raw_answers = response.get("answers", [])
            if (
                isinstance(raw_answers, list)
                and len(raw_answers) == 3
                and all(isinstance(a, str) for a in raw_answers)
            ):
                answers = raw_answers
            else:
                logger.warning(
                    "Unexpected answers shape for persona %s: %r",
                    persona.id, raw_answers,
                )
                answers = ["", "", ""]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Parse error for persona %s: %s", persona.id, exc
            )
            answers = ["", "", ""]

        return {
            "persona_id": persona.id,
            "domain": persona.domain,
            "archetype": persona.archetype,
            "answers": answers,
        }

    tasks = [_check_one(p) for p in sampled]
    results = await asyncio.gather(*tasks)
    return list(results)


# ---------------------------------------------------------------------------
# 7. Keyword-matching helpers
# ---------------------------------------------------------------------------


def _answer_matches_signals(answer: str, keyword_list: List[str]) -> bool:
    """Return True if *answer* contains at least one keyword from *keyword_list*."""
    lower = answer.lower()
    return any(kw in lower for kw in keyword_list)


def _compute_authenticity(
    archetype: str,
    domain: str,
    results_for_archetype: List[Dict[str, Any]],
) -> float:
    """
    Fraction of personas whose answers match the expected keyword signals.
    A persona's answers are considered authentic if at least 2 of the 3
    question signals are matched.
    """
    if not results_for_archetype:
        return 0.0

    signal_lists = _ARCHETYPE_SIGNALS.get(domain, {}).get(archetype)
    if signal_lists is None:
        return 0.0  # unknown archetype

    authentic_count = 0
    for record in results_for_archetype:
        answers = record["answers"]
        matched = 0
        for q_idx, kw_list in enumerate(signal_lists):
            if q_idx < len(answers) and _answer_matches_signals(
                answers[q_idx], kw_list
            ):
                matched += 1
        if matched >= 2:
            authentic_count += 1

    return authentic_count / len(results_for_archetype)


# ---------------------------------------------------------------------------
# 8. Chi-squared contingency test
# ---------------------------------------------------------------------------


def _chi2_for_domain(
    domain_results: List[Dict[str, Any]],
    domain: str,
) -> Optional[float]:
    """
    Build a contingency table: rows = archetypes, columns = questions,
    cells = number of signal matches.  Return p-value from chi2_contingency.
    Returns None if the table is degenerate.
    """
    archetypes_in_data = sorted(
        {r["archetype"] for r in domain_results}
    )
    n_questions = 3

    contingency: List[List[int]] = []
    for archetype in archetypes_in_data:
        signal_lists = _ARCHETYPE_SIGNALS.get(domain, {}).get(archetype)
        row_counts: List[int] = [0] * n_questions
        records = [r for r in domain_results if r["archetype"] == archetype]

        for record in records:
            answers = record["answers"]
            if signal_lists is None:
                continue
            for q_idx, kw_list in enumerate(signal_lists):
                if q_idx < len(answers) and _answer_matches_signals(
                    answers[q_idx], kw_list
                ):
                    row_counts[q_idx] += 1

        contingency.append(row_counts)

    if not contingency:
        return None

    table = np.array(contingency, dtype=float)

    # Drop columns that are all zero (degenerate)
    nonzero_cols = np.any(table > 0, axis=0)
    table = table[:, nonzero_cols]

    if table.shape[0] < 2 or table.shape[1] < 2:
        return None

    try:
        _, p_value, _, _ = chi2_contingency(table)
        return float(p_value)
    except ValueError as exc:
        logger.warning("chi2_contingency error for domain %s: %s", domain, exc)
        return None


# ---------------------------------------------------------------------------
# 9. analyze_manipulation_results
# ---------------------------------------------------------------------------


def analyze_manipulation_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyse manipulation check results.

    Returns a nested dict:
        {
            domain: {
                archetype: {
                    n: int,
                    authenticity_score: float,
                    sample_answers: List[str],   # up to 3 representative answers
                },
                "chi2_p_value": float | None,
            }
        }
    """
    domains = sorted({r["domain"] for r in results})
    analysis: Dict[str, Any] = {}

    for domain in domains:
        domain_results = [r for r in results if r["domain"] == domain]
        archetypes = sorted({r["archetype"] for r in domain_results})
        domain_entry: Dict[str, Any] = {}

        for archetype in archetypes:
            arch_results = [
                r for r in domain_results if r["archetype"] == archetype
            ]
            authenticity = _compute_authenticity(archetype, domain, arch_results)

            # Collect sample answers: take the first non-empty answer from up
            # to 3 personas that passed the authenticity test.
            sample_answers: List[str] = []
            for record in arch_results:
                if len(sample_answers) >= 3:
                    break
                first_nonempty = next(
                    (a for a in record["answers"] if a.strip()), None
                )
                if first_nonempty:
                    sample_answers.append(first_nonempty)

            domain_entry[archetype] = {
                "n": len(arch_results),
                "authenticity_score": round(authenticity, 4),
                "sample_answers": sample_answers,
            }

        domain_entry["chi2_p_value"] = _chi2_for_domain(domain_results, domain)
        analysis[domain] = domain_entry

    return analysis


# ---------------------------------------------------------------------------
# 10. generate_null_personas
# ---------------------------------------------------------------------------


def generate_null_personas(n: int = 100) -> List[Persona]:
    """
    Return *n* personas with NO archetype priming.
    Equal split across the 3 domains (~n//3 each).
    """
    rng = random.Random(99)
    domains = ["coffee", "restaurant", "hotel"]
    per_domain = n // len(domains)
    remainder = n % len(domains)

    personas: List[Persona] = []
    pid = 0
    for i, domain in enumerate(domains):
        count = per_domain + (1 if i < remainder else 0)
        for _ in range(count):
            pid += 1
            personas.append(
                Persona(
                    id=f"NULL-{domain[:3].upper()}-{pid:04d}",
                    name=rng.choice(_FIRST_NAMES),
                    age=rng.randint(22, 58),
                    occupation=rng.choice(_OCCUPATIONS),
                    city=rng.choice(_CITIES),
                    domain=domain,  # type: ignore[arg-type]
                    archetype="Null",
                    behavioral_profile=_NULL_BEHAVIORAL_PROFILES[domain],
                    task_context=_NULL_TASK_CONTEXTS[domain],
                )
            )

    rng.shuffle(personas)
    return personas


# ---------------------------------------------------------------------------
# 11. generate_inverted_personas
# ---------------------------------------------------------------------------


def generate_inverted_personas(
    original_personas: List[Persona],
    n: int = 100,
) -> List[Persona]:
    """
    Sample *n* personas from *original_personas* and swap their archetype
    profiles so that each persona receives a mismatched behavioural profile.

    Returns new Persona objects with swapped behavioral_profile / task_context
    and archetype set to "<original>->INVERTED".
    """
    # Import the profile/task maps from persona_generator
    from persona_generator import (
        _COFFEE_PROFILES,
        _COFFEE_TASK,
        _RESTAURANT_PROFILES,
        _RESTAURANT_TASK,
        _HOTEL_PROFILES,
        _HOTEL_TASK,
    )

    _profile_maps: Dict[str, Dict[str, str]] = {
        "coffee": _COFFEE_PROFILES,
        "restaurant": _RESTAURANT_PROFILES,
        "hotel": _HOTEL_PROFILES,
    }
    _task_maps: Dict[str, Dict[str, str]] = {
        "coffee": _COFFEE_TASK,
        "restaurant": _RESTAURANT_TASK,
        "hotel": _HOTEL_TASK,
    }

    rng = random.Random(77)
    pool = list(original_personas)
    rng.shuffle(pool)
    selected = pool[:n]

    inverted: List[Persona] = []
    for persona in selected:
        domain = persona.domain
        swap_map = _ARCHETYPE_SWAP.get(domain, {})
        target_archetype = swap_map.get(persona.archetype)

        if target_archetype is None:
            # Fallback: pick any other archetype in same domain
            domain_archetypes = list(_profile_maps[domain].keys())
            others = [a for a in domain_archetypes if a != persona.archetype]
            target_archetype = rng.choice(others) if others else persona.archetype

        inverted.append(
            Persona(
                id=f"INV-{persona.id}",
                name=persona.name,
                age=persona.age,
                occupation=persona.occupation,
                city=persona.city,
                domain=persona.domain,
                archetype=f"{persona.archetype}->INVERTED",
                behavioral_profile=_profile_maps[domain][target_archetype],
                task_context=_task_maps[domain][target_archetype],
            )
        )

    return inverted


# ---------------------------------------------------------------------------
# 12. save_manipulation_report
# ---------------------------------------------------------------------------

_PASS_THRESHOLD = 0.6   # authenticity score >= this is a pass
_P_VALUE_THRESHOLD = 0.05  # chi-squared p-value below this is a pass


def save_manipulation_report(
    analysis: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Save a markdown manipulation check report to
    *{output_dir}/manipulation_check_report.md*.

    Returns the path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "manipulation_check_report.md")

    lines: List[str] = [
        "# Manipulation Check Report",
        "",
        "Validates whether LLM personas behave in accordance with their "
        "assigned behavioural archetype.",
        "",
        f"**Pass threshold — authenticity score:** ≥ {_PASS_THRESHOLD}",
        f"**Pass threshold — chi-squared p-value:** < {_P_VALUE_THRESHOLD}",
        "",
        "---",
        "",
    ]

    overall_pass = True

    domains = [d for d in analysis if d != "chi2_p_value"]
    for domain in sorted(domains):
        domain_data = analysis[domain]
        chi2_p = domain_data.get("chi2_p_value")
        chi2_pass = chi2_p is not None and chi2_p < _P_VALUE_THRESHOLD

        chi2_str = (
            f"{chi2_p:.4f}" if chi2_p is not None else "N/A"
        )
        chi2_verdict = "PASS" if chi2_pass else "FAIL"

        lines += [
            f"## Domain: {domain.capitalize()}",
            "",
            f"**Chi-squared p-value:** {chi2_str} — **{chi2_verdict}**",
            "",
            "| Archetype | N | Authenticity Score | Verdict |",
            "|-----------|---|--------------------|---------|",
        ]

        for archetype, data in sorted(domain_data.items()):
            if archetype == "chi2_p_value":
                continue
            score = data["authenticity_score"]
            n = data["n"]
            verdict = "PASS" if score >= _PASS_THRESHOLD else "FAIL"
            if verdict == "FAIL":
                overall_pass = False
            lines.append(
                f"| {archetype} | {n} | {score:.4f} | {verdict} |"
            )

        lines += ["", "### Sample Answers", ""]
        for archetype, data in sorted(domain_data.items()):
            if archetype == "chi2_p_value":
                continue
            lines.append(f"**{archetype}**")
            sample_answers = data.get("sample_answers", [])
            if sample_answers:
                for ans in sample_answers:
                    # Truncate very long answers for readability
                    display = ans if len(ans) <= 200 else ans[:197] + "..."
                    lines.append(f"- {display}")
            else:
                lines.append("- *(no sample answers)*")
            lines.append("")

        if not chi2_pass:
            overall_pass = False

        lines += ["---", ""]

    overall_verdict = "PASS" if overall_pass else "FAIL"
    lines += [
        "## Overall Verdict",
        "",
        f"**{overall_verdict}** — all archetypes across all domains "
        f"{'achieved sufficient differentiation' if overall_pass else 'did NOT achieve sufficient differentiation'}.",
        "",
    ]

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)

    logger.info("Manipulation check report saved to %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# CLI entry point (convenience)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    from persona_generator import generate_all_personas

    parser = argparse.ArgumentParser(
        description="Run the LLM manipulation check and save a report."
    )
    parser.add_argument(
        "--sample-n", type=int, default=150,
        help="Number of personas to sample for the check (default: 150).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=RESULTS_DIR,
        help="Directory to save the report (default: RESULTS_DIR from config).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    all_personas = generate_all_personas()

    results = asyncio.run(
        run_manipulation_check(all_personas, sample_n=args.sample_n)
    )
    analysis = analyze_manipulation_results(results)
    report_path = save_manipulation_report(analysis, args.output_dir)
    print(f"Report saved to: {report_path}")
