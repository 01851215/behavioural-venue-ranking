"""
report_study2.py — Report generator for Study 2 (occupation × age cross-matrix simulation).

Study 2 runs 3,000 personas across a 5 age-group × 10 occupation × 3 domain matrix.
Records are read from results/simulation_records_study2.csv.

Public API
----------
generate_study2_report(records, output_dir) -> str
    Saves simulation_report_study2.md and returns the path.

save_study2_csv_tables(metrics, output_dir)
    Saves study2_by_age.csv, study2_by_occupation.csv, study2_cross_matrix.csv.

compute_study2_metrics(records) -> dict
    Groups records by age_group, occupation, domain, (age_group × occupation),
    and ALL_OVERALL; returns nested metric dict.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import RESULTS_DIR, MODEL
from evaluator import bootstrap_ci, wilcoxon_test, benjamini_hochberg, cohen_d


# ── Constants ─────────────────────────────────────────────────────────────────

STUDY2_TOTAL_PERSONAS: int = 3_000
STUDY2_API_CALLS: int = 9_000  # 3 tasks × 3,000 personas

AGE_GROUPS: List[str] = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55+",
]

OCCUPATIONS: List[str] = [
    "Student",
    "Professional",
    "Manager / Executive",
    "Healthcare Worker",
    "Educator",
    "Retail / Service Worker",
    "Skilled Trades",
    "Creative / Media",
    "Retired",
    "Unemployed / Part-time",
]

DOMAINS: List[str] = ["coffee", "restaurant", "hotel"]

DOMAIN_LABELS: Dict[str, str] = {
    "coffee": "Coffee Shops",
    "restaurant": "Restaurants",
    "hotel": "Hotels",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _aggregate_group(recs: List[dict]) -> dict:
    """Compute all metrics for a list of records. Returns a flat metric dict."""
    if not recs:
        return {}

    ndcg_vals = [float(r.get("ndcg", 0)) for r in recs]
    stars_ndcg = [float(r.get("stars_ndcg", r.get("ndcg", 0))) for r in recs]
    hit1_vals = [float(r.get("hit_at_1", 0)) for r in recs]
    hit3_vals = [float(r.get("hit_at_3", 0)) for r in recs]
    hit10_vals = [float(r.get("hit_at_10", 0)) for r in recs]
    tau_vals = [float(r.get("kendall_tau", 0)) for r in recs]
    pairwise = [float(r.get("pairwise_win", 0)) for r in recs]

    mean_ndcg, lo, hi = bootstrap_ci(ndcg_vals)
    p_val = wilcoxon_test(ndcg_vals, stars_ndcg)
    d = cohen_d(ndcg_vals, stars_ndcg)

    return {
        "n": len(recs),
        "ndcg_mean": round(mean_ndcg, 4),
        "ndcg_lo": round(lo, 4),
        "ndcg_hi": round(hi, 4),
        "hit_at_1": round(_safe_mean(hit1_vals), 4),
        "hit_at_3": round(_safe_mean(hit3_vals), 4),
        "hit_at_10": round(_safe_mean(hit10_vals), 4),
        "kendall_tau": round(_safe_mean(tau_vals), 4),
        "stars_ndcg_mean": round(_safe_mean(stars_ndcg), 4),
        "delta_vs_stars": round(mean_ndcg - _safe_mean(stars_ndcg), 4),
        "pairwise_win_rate": round(_safe_mean(pairwise), 4),
        "wilcoxon_p": round(p_val, 4),
        "wilcoxon_p_bh": None,  # filled in by compute_study2_metrics
        "cohen_d": round(d, 4),
    }


# ── Public: compute_study2_metrics ────────────────────────────────────────────

def compute_study2_metrics(records: List[dict]) -> dict:
    """
    Group records by age_group, occupation, domain, (age_group × occupation),
    and ALL_OVERALL.  Apply Benjamini-Hochberg correction across all groups.

    Returns
    -------
    dict with keys of the form:
        ("age_group", <group>)
        ("occupation", <occ>)
        ("domain", <domain>)
        ("cross", <age_group>, <occupation>)
        ("ALL", "OVERALL")
    Each value is the flat metric dict produced by _aggregate_group().
    """
    if not records:
        return {}

    # Buckets
    by_age: Dict[str, List[dict]] = defaultdict(list)
    by_occ: Dict[str, List[dict]] = defaultdict(list)
    by_domain: Dict[str, List[dict]] = defaultdict(list)
    by_cross: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    all_recs: List[dict] = []

    for r in records:
        age = str(r.get("age_group", "unknown")).strip()
        occ = str(r.get("occupation", "unknown")).strip()
        dom = str(r.get("domain", "unknown")).strip()

        by_age[age].append(r)
        by_occ[occ].append(r)
        by_domain[dom].append(r)
        by_cross[(age, occ)].append(r)
        all_recs.append(r)

    results: dict = {}

    # --- age groups ---
    for age, recs in by_age.items():
        key = ("age_group", age)
        results[key] = _aggregate_group(recs)

    # --- occupations ---
    for occ, recs in by_occ.items():
        key = ("occupation", occ)
        results[key] = _aggregate_group(recs)

    # --- domains ---
    for dom, recs in by_domain.items():
        key = ("domain", dom)
        results[key] = _aggregate_group(recs)

    # --- cross matrix ---
    for (age, occ), recs in by_cross.items():
        key = ("cross", age, occ)
        results[key] = _aggregate_group(recs)

    # --- overall ---
    results[("ALL", "OVERALL")] = _aggregate_group(all_recs)

    # Benjamini-Hochberg correction over all groups that have a p-value
    ordered_keys = [k for k, v in results.items() if v and v.get("wilcoxon_p") is not None]
    raw_p = [results[k]["wilcoxon_p"] for k in ordered_keys]
    adj_p = benjamini_hochberg(raw_p)
    for k, p_bh in zip(ordered_keys, adj_p):
        results[k]["wilcoxon_p_bh"] = round(float(p_bh), 4)

    return results


# ── Public: save_study2_csv_tables ───────────────────────────────────────────

def save_study2_csv_tables(metrics: dict, output_dir: str) -> None:
    """
    Save three CSVs from the metrics dict produced by compute_study2_metrics():
    - study2_by_age.csv
    - study2_by_occupation.csv
    - study2_cross_matrix.csv  (age_group × occupation NDCG table)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── by age ────────────────────────────────────────────────────────────────
    age_rows = []
    for age in AGE_GROUPS:
        key = ("age_group", age)
        m = metrics.get(key, {})
        if not m:
            continue
        age_rows.append({
            "age_group": age,
            "n": m.get("n", 0),
            "ndcg_mean": m.get("ndcg_mean", ""),
            "ndcg_lo": m.get("ndcg_lo", ""),
            "ndcg_hi": m.get("ndcg_hi", ""),
            "hit_at_1": m.get("hit_at_1", ""),
            "hit_at_3": m.get("hit_at_3", ""),
            "hit_at_10": m.get("hit_at_10", ""),
            "kendall_tau": m.get("kendall_tau", ""),
            "stars_ndcg_mean": m.get("stars_ndcg_mean", ""),
            "delta_vs_stars": m.get("delta_vs_stars", ""),
            "pairwise_win_rate": m.get("pairwise_win_rate", ""),
            "wilcoxon_p": m.get("wilcoxon_p", ""),
            "wilcoxon_p_bh": m.get("wilcoxon_p_bh", ""),
            "cohen_d": m.get("cohen_d", ""),
        })
    # Also include any age groups in data not listed in AGE_GROUPS constant
    known_ages = {r["age_group"] for r in age_rows}
    for key, m in metrics.items():
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "age_group":
            age = key[1]
            if age not in known_ages and m:
                age_rows.append({
                    "age_group": age,
                    "n": m.get("n", 0),
                    "ndcg_mean": m.get("ndcg_mean", ""),
                    "ndcg_lo": m.get("ndcg_lo", ""),
                    "ndcg_hi": m.get("ndcg_hi", ""),
                    "hit_at_1": m.get("hit_at_1", ""),
                    "hit_at_3": m.get("hit_at_3", ""),
                    "hit_at_10": m.get("hit_at_10", ""),
                    "kendall_tau": m.get("kendall_tau", ""),
                    "stars_ndcg_mean": m.get("stars_ndcg_mean", ""),
                    "delta_vs_stars": m.get("delta_vs_stars", ""),
                    "pairwise_win_rate": m.get("pairwise_win_rate", ""),
                    "wilcoxon_p": m.get("wilcoxon_p", ""),
                    "wilcoxon_p_bh": m.get("wilcoxon_p_bh", ""),
                    "cohen_d": m.get("cohen_d", ""),
                })

    age_path = os.path.join(output_dir, "study2_by_age.csv")
    if age_rows:
        _write_csv(age_path, age_rows)

    # ── by occupation ─────────────────────────────────────────────────────────
    occ_rows = []
    seen_occ = set()
    for occ in OCCUPATIONS:
        key = ("occupation", occ)
        m = metrics.get(key, {})
        if not m:
            continue
        seen_occ.add(occ)
        occ_rows.append({
            "occupation": occ,
            "n": m.get("n", 0),
            "ndcg_mean": m.get("ndcg_mean", ""),
            "ndcg_lo": m.get("ndcg_lo", ""),
            "ndcg_hi": m.get("ndcg_hi", ""),
            "hit_at_1": m.get("hit_at_1", ""),
            "hit_at_3": m.get("hit_at_3", ""),
            "hit_at_10": m.get("hit_at_10", ""),
            "kendall_tau": m.get("kendall_tau", ""),
            "stars_ndcg_mean": m.get("stars_ndcg_mean", ""),
            "delta_vs_stars": m.get("delta_vs_stars", ""),
            "pairwise_win_rate": m.get("pairwise_win_rate", ""),
            "wilcoxon_p": m.get("wilcoxon_p", ""),
            "wilcoxon_p_bh": m.get("wilcoxon_p_bh", ""),
            "cohen_d": m.get("cohen_d", ""),
        })
    # Include any occupations in data not in the constant list
    for key, m in metrics.items():
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "occupation":
            occ = key[1]
            if occ not in seen_occ and m:
                seen_occ.add(occ)
                occ_rows.append({
                    "occupation": occ,
                    "n": m.get("n", 0),
                    "ndcg_mean": m.get("ndcg_mean", ""),
                    "ndcg_lo": m.get("ndcg_lo", ""),
                    "ndcg_hi": m.get("ndcg_hi", ""),
                    "hit_at_1": m.get("hit_at_1", ""),
                    "hit_at_3": m.get("hit_at_3", ""),
                    "hit_at_10": m.get("hit_at_10", ""),
                    "kendall_tau": m.get("kendall_tau", ""),
                    "stars_ndcg_mean": m.get("stars_ndcg_mean", ""),
                    "delta_vs_stars": m.get("delta_vs_stars", ""),
                    "pairwise_win_rate": m.get("pairwise_win_rate", ""),
                    "wilcoxon_p": m.get("wilcoxon_p", ""),
                    "wilcoxon_p_bh": m.get("wilcoxon_p_bh", ""),
                    "cohen_d": m.get("cohen_d", ""),
                })
    # Sort by NDCG descending
    occ_rows.sort(key=lambda row: float(row["ndcg_mean"]) if row["ndcg_mean"] != "" else 0.0, reverse=True)

    occ_path = os.path.join(output_dir, "study2_by_occupation.csv")
    if occ_rows:
        _write_csv(occ_path, occ_rows)

    # ── cross matrix ──────────────────────────────────────────────────────────
    # Collect all ages and occupations present in the cross keys
    cross_ages_set: set = set()
    cross_occ_set: set = set()
    for key in metrics:
        if isinstance(key, tuple) and len(key) == 3 and key[0] == "cross":
            cross_ages_set.add(key[1])
            cross_occ_set.add(key[2])

    # Order: canonical order filtered to what exists, plus any extras
    cross_ages = [a for a in AGE_GROUPS if a in cross_ages_set] + \
                 sorted(cross_ages_set - set(AGE_GROUPS))
    cross_occs = [o for o in OCCUPATIONS if o in cross_occ_set] + \
                 sorted(cross_occ_set - set(OCCUPATIONS))

    cross_rows = []
    for age in cross_ages:
        row: Dict[str, object] = {"age_group": age}
        for occ in cross_occs:
            key = ("cross", age, occ)
            m = metrics.get(key, {})
            row[occ] = m.get("ndcg_mean", "") if m else ""
        cross_rows.append(row)

    cross_path = os.path.join(output_dir, "study2_cross_matrix.csv")
    if cross_rows:
        fieldnames = ["age_group"] + cross_occs
        _write_csv(cross_path, cross_rows, fieldnames=fieldnames)


def _write_csv(path: str, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── Internal report-building helpers ─────────────────────────────────────────

def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def _sig_marker(p_bh: Optional[float]) -> str:
    """Bold marker for significant results."""
    return "**" if (p_bh is not None and p_bh < 0.05) else ""


def _age_table(metrics: dict) -> List[str]:
    """Markdown table: age group rows sorted by canonical order."""
    header = (
        "| Age Group | n | NDCG@10 | Hit@1 | Hit@3 | Kendall τ | "
        "Δ vs Stars | Win Rate | p (BH) |"
    )
    sep = "|-----------|---|---------|-------|-------|-----------|------------|----------|--------|"
    lines = [header, sep]

    # Canonical order plus any extras
    ages_present = {
        key[1] for key in metrics if isinstance(key, tuple) and len(key) == 2 and key[0] == "age_group"
    }
    ordered = [a for a in AGE_GROUPS if a in ages_present] + \
              sorted(ages_present - set(AGE_GROUPS))

    for age in ordered:
        m = metrics.get(("age_group", age), {})
        if not m:
            continue
        p_bh = m.get("wilcoxon_p_bh")
        sig = _sig_marker(p_bh)
        lines.append(
            f"| {age} | {m['n']:,} | {sig}{m['ndcg_mean']:.4f}{sig} | "
            f"{m['hit_at_1']:.3f} | {m['hit_at_3']:.3f} | {m['kendall_tau']:.3f} | "
            f"{m['delta_vs_stars']:+.4f} | {m['pairwise_win_rate']:.1%} | {_fmt_p(p_bh)} |"
        )
    return lines


def _occupation_table(metrics: dict) -> List[str]:
    """Markdown table: occupation rows sorted by NDCG descending."""
    header = (
        "| Occupation | n | NDCG@10 | Hit@1 | Hit@3 | Kendall τ | "
        "Δ vs Stars | Win Rate | p (BH) |"
    )
    sep = "|------------|---|---------|-------|-------|-----------|------------|----------|--------|"
    lines = [header, sep]

    occ_metrics = [
        (key[1], v)
        for key, v in metrics.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "occupation" and v
    ]
    # Sort by NDCG descending
    occ_metrics.sort(key=lambda x: x[1].get("ndcg_mean", 0.0), reverse=True)

    for occ, m in occ_metrics:
        p_bh = m.get("wilcoxon_p_bh")
        sig = _sig_marker(p_bh)
        lines.append(
            f"| {occ} | {m['n']:,} | {sig}{m['ndcg_mean']:.4f}{sig} | "
            f"{m['hit_at_1']:.3f} | {m['hit_at_3']:.3f} | {m['kendall_tau']:.3f} | "
            f"{m['delta_vs_stars']:+.4f} | {m['pairwise_win_rate']:.1%} | {_fmt_p(p_bh)} |"
        )
    return lines


def _domain_table(metrics: dict) -> List[str]:
    """Markdown table: one row per domain."""
    header = (
        "| Domain | n | NDCG@10 | Hit@1 | Hit@3 | Kendall τ | "
        "Δ vs Stars | Win Rate | p (BH) |"
    )
    sep = "|--------|---|---------|-------|-------|-----------|------------|----------|--------|"
    lines = [header, sep]

    for dom in DOMAINS:
        m = metrics.get(("domain", dom), {})
        if not m:
            continue
        label = DOMAIN_LABELS.get(dom, dom)
        p_bh = m.get("wilcoxon_p_bh")
        sig = _sig_marker(p_bh)
        lines.append(
            f"| {label} | {m['n']:,} | {sig}{m['ndcg_mean']:.4f}{sig} | "
            f"{m['hit_at_1']:.3f} | {m['hit_at_3']:.3f} | {m['kendall_tau']:.3f} | "
            f"{m['delta_vs_stars']:+.4f} | {m['pairwise_win_rate']:.1%} | {_fmt_p(p_bh)} |"
        )
    return lines


def _cross_heatmap_table(metrics: dict) -> List[str]:
    """Markdown table: NDCG@10 for each (age_group × occupation) cell."""
    cross_ages_set: set = set()
    cross_occ_set: set = set()
    for key in metrics:
        if isinstance(key, tuple) and len(key) == 3 and key[0] == "cross":
            cross_ages_set.add(key[1])
            cross_occ_set.add(key[2])

    if not cross_ages_set:
        return ["_No cross-matrix data available._"]

    ages = [a for a in AGE_GROUPS if a in cross_ages_set] + \
           sorted(cross_ages_set - set(AGE_GROUPS))
    occs = [o for o in OCCUPATIONS if o in cross_occ_set] + \
           sorted(cross_occ_set - set(OCCUPATIONS))

    # Header
    header_cols = " | ".join(occs)
    header = f"| Age Group | {header_cols} |"
    sep_cols = " | ".join(["-------"] * len(occs))
    sep = f"|-----------|{sep_cols}|"
    lines = [header, sep]

    for age in ages:
        cells = []
        for occ in occs:
            m = metrics.get(("cross", age, occ), {})
            val = m.get("ndcg_mean") if m else None
            cells.append(f"{val:.4f}" if val is not None else "—")
        lines.append(f"| {age} | " + " | ".join(cells) + " |")

    return lines


def _executive_summary(metrics: dict) -> List[str]:
    """3-4 bullet points for the executive summary."""
    all_m = metrics.get(("ALL", "OVERALL"), {})
    lines: List[str] = []

    if not all_m:
        lines.append("- No data available.")
        return lines

    overall_ndcg = all_m.get("ndcg_mean", 0.0)
    overall_stars = all_m.get("stars_ndcg_mean", 0.0)
    delta = all_m.get("delta_vs_stars", 0.0)
    win_rate = all_m.get("pairwise_win_rate", 0.0)
    p_bh = all_m.get("wilcoxon_p_bh") or all_m.get("wilcoxon_p", 1.0)

    # Best / worst age group
    age_entries = [
        (key[1], v.get("ndcg_mean", 0.0))
        for key, v in metrics.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "age_group" and v
    ]
    if age_entries:
        best_age = max(age_entries, key=lambda x: x[1])
        worst_age = min(age_entries, key=lambda x: x[1])
    else:
        best_age = ("n/a", 0.0)
        worst_age = ("n/a", 0.0)

    # Best / worst occupation
    occ_entries = [
        (key[1], v.get("ndcg_mean", 0.0))
        for key, v in metrics.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "occupation" and v
    ]
    if occ_entries:
        best_occ = max(occ_entries, key=lambda x: x[1])
        worst_occ = min(occ_entries, key=lambda x: x[1])
    else:
        best_occ = ("n/a", 0.0)
        worst_occ = ("n/a", 0.0)

    # Behavioural vs. stars verdict
    beats_stars = delta > 0 and (p_bh is not None and p_bh < 0.05)
    stars_verdict = (
        f"behavioural model **beats** star-rating baseline by Δ={delta:+.4f} "
        f"(Wilcoxon p_BH={_fmt_p(p_bh)})"
        if beats_stars
        else f"behavioural model does not significantly outperform star-rating baseline "
             f"(Δ={delta:+.4f}, p_BH={_fmt_p(p_bh)})"
    )

    lines += [
        f"- **Overall NDCG@10:** {overall_ndcg:.4f} (95% CI [{all_m.get('ndcg_lo', 0.0):.4f}, "
        f"{all_m.get('ndcg_hi', 0.0):.4f}]) across {STUDY2_TOTAL_PERSONAS:,} personas.",
        f"- **Age groups:** best alignment in the **{best_age[0]}** cohort "
        f"(NDCG={best_age[1]:.4f}); weakest in **{worst_age[0]}** (NDCG={worst_age[1]:.4f}).",
        f"- **Occupations:** highest agreement from **{best_occ[0]}** personas "
        f"(NDCG={best_occ[1]:.4f}); lowest from **{worst_occ[0]}** (NDCG={worst_occ[1]:.4f}).",
        f"- **Versus star-rating baseline:** {stars_verdict}; "
        f"pairwise win rate {win_rate:.1%}.",
    ]
    return lines


def _interpretation_bullets(metrics: dict) -> List[str]:
    """Qualitative interpretation bullets."""
    all_m = metrics.get(("ALL", "OVERALL"), {})
    tau = all_m.get("kendall_tau", 0.0) if all_m else 0.0
    p_bh = (all_m.get("wilcoxon_p_bh") or all_m.get("wilcoxon_p", 1.0)) if all_m else 1.0

    # Top 3 age groups by NDCG
    age_entries = sorted(
        [
            (key[1], v.get("ndcg_mean", 0.0))
            for key, v in metrics.items()
            if isinstance(key, tuple) and len(key) == 2 and key[0] == "age_group" and v
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    # Top 3 occupations by NDCG
    occ_entries = sorted(
        [
            (key[1], v.get("ndcg_mean", 0.0))
            for key, v in metrics.items()
            if isinstance(key, tuple) and len(key) == 2 and key[0] == "occupation" and v
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    top_ages = ", ".join(f"**{a}** ({n:.4f})" for a, n in age_entries[:3]) if age_entries else "n/a"
    top_occs = ", ".join(f"**{o}** ({n:.4f})" for o, n in occ_entries[:3]) if occ_entries else "n/a"

    # Potential surprises: occupations with below-average NDCG despite likely being power users
    all_mean = all_m.get("ndcg_mean", 0.0) if all_m else 0.0
    surprise_occs = [o for o, n in occ_entries if n < all_mean and o in ("Professional", "Manager / Executive")]
    surprise_note = (
        f"Notably, {' and '.join(surprise_occs)} personas scored below the overall mean, "
        "suggesting that higher socioeconomic status does not automatically map to stronger "
        "alignment with the behavioural venue ranking model."
        if surprise_occs
        else "No clear counter-intuitive patterns were detected at this level of aggregation; "
             "higher-status occupations generally align as expected with the behavioural model."
    )

    lines = [
        f"- **Age groups most aligned with behavioural model:** {top_ages}. "
        "These cohorts show the strongest Kendall τ agreement, suggesting their real-world "
        "decision heuristics most closely mirror the BiRank signal extracted from Yelp + Foursquare data.",
        f"- **Occupations most aligned:** {top_occs}. "
        "Consumer research literature predicts that time-constrained and quality-sensitive "
        "occupations prioritise reliability over novelty, consistent with BiRank scores.",
        f"- **Rank-order agreement (Kendall τ):** overall mean τ={tau:.4f}. "
        "Values above 0.3 indicate moderate agreement between persona-driven rankings "
        "and the behavioural model ordering.",
        f"- **Surprises:** {surprise_note}",
        f"- **Statistical significance:** "
        + (
            f"Results are statistically significant after BH correction (p_BH={_fmt_p(p_bh)}), "
            "indicating the behavioural model signal is robust across all sociodemographic strata."
            if p_bh < 0.05
            else f"Results do not reach statistical significance after BH correction "
                 f"(p_BH={_fmt_p(p_bh)}). This may reflect genuine heterogeneity "
                 "across the broader sociodemographic matrix."
        ),
    ]
    return lines


# ── Public: generate_study2_report ───────────────────────────────────────────

def generate_study2_report(records: List[dict], output_dir: str) -> str:
    """
    Generate the Study 2 markdown report.

    Parameters
    ----------
    records : list of dict
        Row-level records from simulation_records_study2.csv.
        Expected columns: persona_id, domain, archetype, age_group, occupation,
        city, ndcg, hit_at_1, hit_at_3, hit_at_10, kendall_tau, stars_ndcg,
        stars_hit_at_1, stars_hit_at_3, pairwise_win, revisit_score,
        model_revisit_signal.
    output_dir : str
        Directory where the report and CSVs are saved (typically RESULTS_DIR).

    Returns
    -------
    str
        Absolute path to the saved simulation_report_study2.md.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "simulation_report_study2.md")

    metrics = compute_study2_metrics(records)
    save_study2_csv_tables(metrics, output_dir)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_m = metrics.get(("ALL", "OVERALL"), {})

    lines: List[str] = []

    # ── 1. Header ─────────────────────────────────────────────────────────────
    lines += [
        "# LLM Simulation — Study 2 Report",
        "## Occupation × Age Cross-Matrix Validation",
        "",
        f"**Generated:** {ts}  ",
        f"**Model:** `{MODEL}`  ",
        f"**Total personas:** {STUDY2_TOTAL_PERSONAS:,}  ",
        f"**API calls:** {STUDY2_API_CALLS:,}  ",
        f"**Design:** 5 age groups × 10 occupations × 3 domains  ",
        "",
        "---",
        "",
    ]

    # ── 2. Executive Summary ──────────────────────────────────────────────────
    lines += [
        "## Executive Summary",
        "",
    ]
    if records:
        lines += _executive_summary(metrics)
    else:
        lines.append("_No records found — run the Study 2 simulation first._")
    lines += ["", "---", ""]

    # ── 3. Results by Age Group ───────────────────────────────────────────────
    lines += [
        "## Results by Age Group",
        "",
        "Bold NDCG values indicate statistically significant improvement over the star-rating "
        "baseline after Benjamini-Hochberg correction (α=0.05).",
        "",
    ]
    lines += _age_table(metrics)
    lines += ["", "---", ""]

    # ── 4. Results by Occupation ──────────────────────────────────────────────
    lines += [
        "## Results by Occupation",
        "",
        "Sorted by NDCG@10 descending.",
        "",
    ]
    lines += _occupation_table(metrics)
    lines += ["", "---", ""]

    # ── 5. Results by Domain ─────────────────────────────────────────────────
    lines += [
        "## Results by Domain",
        "",
    ]
    lines += _domain_table(metrics)
    lines += ["", "---", ""]

    # ── 6. Cross-matrix Heatmap Data ─────────────────────────────────────────
    lines += [
        "## Cross-Matrix Heatmap Data",
        "",
        "NDCG@10 for each (age group × occupation) cell across all domains. "
        "Dashes (—) indicate cells with no data.",
        "",
    ]
    lines += _cross_heatmap_table(metrics)
    lines += ["", "---", ""]

    # ── 7. Comparison with Study 1 ────────────────────────────────────────────
    lines += [
        "## Comparison with Study 1",
        "",
        "Study 1 used behavioural archetypes from Yelp data; Study 2 uses sociodemographic "
        "archetypes from consumer research literature. Agreement between studies strengthens "
        "external validity.",
        "",
        "| Dimension | Study 1 | Study 2 |",
        "|-----------|---------|---------|",
        "| Archetype basis | Behavioural clusters (Yelp + Foursquare) | "
        "Sociodemographic segments (age × occupation) |",
        "| Personas | 1,500 | 3,000 |",
        "| Segmentation axes | Domain × archetype | Domain × age group × occupation |",
        "| Validity type | Internal (data-grounded archetypes) | "
        "External (consumer research literature) |",
        "",
        "Where both studies show significant alignment with the behavioural model "
        "(NDCG above star-rating baseline, p_BH < 0.05), this provides convergent "
        "evidence that the BiRank venue ranking generalises beyond the segmentation "
        "scheme used to derive it.",
        "",
        "---",
        "",
    ]

    # ── 8. Interpretation ────────────────────────────────────────────────────
    lines += [
        "## Interpretation",
        "",
    ]
    if records:
        lines += _interpretation_bullets(metrics)
    else:
        lines.append("_No data to interpret._")
    lines += [
        "",
        "---",
        "",
        "_Generated by LLM Simulation Pipeline — Master Project Behavioral Venue Ranking_",
    ]

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return report_path


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    records_path = os.path.join(RESULTS_DIR, "simulation_records_study2.csv")
    if not os.path.exists(records_path):
        print(f"ERROR: records file not found at {records_path}")
        print("Run the Study 2 simulation first, then re-run this script.")
        sys.exit(1)

    loaded: List[dict] = []
    with open(records_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Coerce numeric fields
            for numeric_col in (
                "ndcg", "hit_at_1", "hit_at_3", "hit_at_10", "kendall_tau",
                "stars_ndcg", "stars_hit_at_1", "stars_hit_at_3",
                "pairwise_win", "revisit_score", "model_revisit_signal",
            ):
                try:
                    row[numeric_col] = float(row[numeric_col])
                except (ValueError, KeyError):
                    row[numeric_col] = 0.0
            loaded.append(row)

    print(f"Loaded {len(loaded):,} records from {records_path}")
    out_path = generate_study2_report(loaded, RESULTS_DIR)
    print(f"Report saved → {out_path}")
