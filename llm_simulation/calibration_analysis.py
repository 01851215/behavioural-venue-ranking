"""
Phase 3 Calibration Analysis — LLM Simulation Validation Study.

Functions:
  compute_revisit_calibration    — Spearman/Pearson between revisit_score and model signal
  run_cross_domain_consistency   — Cross-domain NDCG correlation (or archetype-level pattern)
  compute_per_persona_variance   — NDCG variance and power analysis per archetype
  generate_calibration_plot_data — Scatter-plot data + optional PNG save
  save_calibration_report        — Markdown report to output_dir/calibration_report.md
"""

from __future__ import annotations

import os
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR

# ---------------------------------------------------------------------------
# Module-level paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS_CSV = os.path.join(RESULTS_DIR, "simulation_records.csv")

DOMAINS: List[str] = ["coffee", "restaurant", "hotel"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (rho, p_value).  Returns (nan, nan) when n < 3 or zero-variance."""
    if len(x) < 3:
        return float("nan"), float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (r, p_value).  Returns (nan, nan) when n < 3 or zero-variance."""
    if len(x) < 3:
        return float("nan"), float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    result = stats.pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def _normalise_revisit(series: pd.Series) -> pd.Series:
    """Normalise a 0-10 revisit_score column to 0-1."""
    return series.astype(float) / 10.0


def _recommended_n_repeats(std: float, ci_half_width: float = 0.005) -> int:
    """
    Estimate n repeats needed so that a 95% CI for the mean has half-width
    <= ci_half_width.  Uses the formula: n = (1.96 * std / ci_half_width)^2.
    Returns at least 1.
    """
    if std <= 0 or not math.isfinite(std):
        return 1
    n = math.ceil((1.96 * std / ci_half_width) ** 2)
    return max(1, n)


# ---------------------------------------------------------------------------
# 1. compute_revisit_calibration
# ---------------------------------------------------------------------------


def compute_revisit_calibration(records_df: pd.DataFrame) -> Dict:
    """
    For each domain, compute Spearman and Pearson correlations between
    normalised revisit_score (0-1) and model_revisit_signal (0-1).

    Also breaks down Spearman rho per archetype within each domain.

    Parameters
    ----------
    records_df : pd.DataFrame
        Must contain columns: domain, archetype, revisit_score,
        model_revisit_signal.

    Returns
    -------
    dict  keyed by domain, each value:
        {
          "overall": {spearman_r, spearman_p, pearson_r, pearson_p, n},
          "by_archetype": {archetype: {spearman_r, n}}
        }
    """
    df = records_df.copy()
    df["revisit_norm"] = _normalise_revisit(df["revisit_score"])
    df["model_signal"] = df["model_revisit_signal"].astype(float)

    # Drop rows where either signal is NaN
    df = df.dropna(subset=["revisit_norm", "model_signal"])

    result: Dict = {}

    for domain in DOMAINS:
        dom_df = df[df["domain"] == domain]
        if dom_df.empty:
            result[domain] = {
                "overall": {
                    "spearman_r": float("nan"),
                    "spearman_p": float("nan"),
                    "pearson_r": float("nan"),
                    "pearson_p": float("nan"),
                    "n": 0,
                },
                "by_archetype": {},
            }
            continue

        x = dom_df["revisit_norm"].values
        y = dom_df["model_signal"].values

        sp_r, sp_p = _safe_spearman(x, y)
        pe_r, pe_p = _safe_pearson(x, y)

        overall = {
            "spearman_r": round(sp_r, 4) if math.isfinite(sp_r) else sp_r,
            "spearman_p": round(sp_p, 4) if math.isfinite(sp_p) else sp_p,
            "pearson_r": round(pe_r, 4) if math.isfinite(pe_r) else pe_r,
            "pearson_p": round(pe_p, 4) if math.isfinite(pe_p) else pe_p,
            "n": int(len(dom_df)),
        }

        by_archetype: Dict = {}
        for arch, arch_df in dom_df.groupby("archetype"):
            ax = arch_df["revisit_norm"].values
            ay = arch_df["model_signal"].values
            arch_sp_r, _ = _safe_spearman(ax, ay)
            by_archetype[arch] = {
                "spearman_r": (
                    round(arch_sp_r, 4) if math.isfinite(arch_sp_r) else arch_sp_r
                ),
                "n": int(len(arch_df)),
            }

        result[domain] = {"overall": overall, "by_archetype": by_archetype}

    return result


# ---------------------------------------------------------------------------
# 2. run_cross_domain_consistency
# ---------------------------------------------------------------------------


def run_cross_domain_consistency(
    records_df: pd.DataFrame, n_shared: int = 100
) -> Dict:
    """
    Attempt to find personas that appear in both coffee and hotel domains.
    Because v1 used unique persona IDs per domain, this function falls back to
    an archetype-level cross-domain pattern analysis when no shared IDs exist.

    The archetype pattern tested:
      - Coffee Loyalists (high revisit_score) vs. non-Loyalists.
      - Do personas with higher coffee NDCG also show higher hotel NDCG
        at the archetype level?
      - Concretely: compute mean NDCG per archetype × domain, then correlate
        coffee-archetype mean NDCG with hotel-archetype mean NDCG where there
        is a conceptual mapping (high-loyalty coffee ↔ high-retention hotel).

    Parameters
    ----------
    records_df : pd.DataFrame
    n_shared   : int  (informational; used only if shared IDs are found)

    Returns
    -------
    dict:
        {
          "cross_domain_corr": float,
          "p_value": float,
          "n_shared": int,
          "archetype_pattern": dict
        }
    """
    df = records_df.copy()

    # --- Attempt 1: genuine cross-domain personas (same persona_id) ----------
    id_domains = df.groupby("persona_id")["domain"].nunique()
    shared_ids = id_domains[id_domains > 1].index.tolist()

    if len(shared_ids) >= 3:
        shared_df = df[df["persona_id"].isin(shared_ids)]
        coffee_ndcg = (
            shared_df[shared_df["domain"] == "coffee"]
            .set_index("persona_id")["ndcg"]
            .rename("coffee_ndcg")
        )
        hotel_ndcg = (
            shared_df[shared_df["domain"] == "hotel"]
            .set_index("persona_id")["ndcg"]
            .rename("hotel_ndcg")
        )
        merged = pd.concat([coffee_ndcg, hotel_ndcg], axis=1).dropna()

        if len(merged) >= 3:
            corr, p_val = _safe_pearson(
                merged["coffee_ndcg"].values, merged["hotel_ndcg"].values
            )
            archetype_pattern = _build_archetype_pattern(df)
            return {
                "cross_domain_corr": (
                    round(corr, 4) if math.isfinite(corr) else corr
                ),
                "p_value": round(p_val, 4) if math.isfinite(p_val) else p_val,
                "n_shared": int(len(merged)),
                "archetype_pattern": archetype_pattern,
            }

    # --- Fallback: archetype-level cross-domain pattern ----------------------
    archetype_pattern = _build_archetype_pattern(df)

    # Build archetype-level mean NDCG vectors aligned by loyalty proxy.
    # Loyalty ordering defined below (higher index = higher retention/loyalty).
    coffee_loyalty_order = {
        "Loyalist": 3,
        "Weekday Regular": 2,
        "Casual Weekender": 1,
        "Infrequent Visitor": 0,
    }
    hotel_retention_order = {
        "Leisure Traveler": 3,
        "Budget Explorer": 2,
        "One-Time Tourist": 1,
        "One-Time Tourist (Business)": 0,
    }

    coffee_means = (
        df[df["domain"] == "coffee"]
        .groupby("archetype")["ndcg"]
        .mean()
    )
    hotel_means = (
        df[df["domain"] == "hotel"]
        .groupby("archetype")["ndcg"]
        .mean()
    )

    # Map each archetype to a loyalty rank, then correlate rank vs mean NDCG.
    c_pairs = [
        (rank, coffee_means[arch])
        for arch, rank in coffee_loyalty_order.items()
        if arch in coffee_means.index
    ]
    h_pairs = [
        (rank, hotel_means[arch])
        for arch, rank in hotel_retention_order.items()
        if arch in hotel_means.index
    ]

    cross_corr = float("nan")
    cross_p = float("nan")
    n_arch_pairs = 0

    if len(c_pairs) >= 3 and len(h_pairs) >= 3:
        c_ranks = np.array([p[0] for p in c_pairs])
        c_ndcg = np.array([p[1] for p in c_pairs])
        h_ranks = np.array([p[0] for p in h_pairs])
        h_ndcg = np.array([p[1] for p in h_pairs])

        # Spearman: loyalty rank vs NDCG within each domain
        c_corr, c_p = _safe_spearman(c_ranks, c_ndcg)
        h_corr, h_p = _safe_spearman(h_ranks, h_ndcg)

        # Summary: mean of the two within-domain rank-NDCG correlations
        valid = [v for v in [c_corr, h_corr] if math.isfinite(v)]
        if valid:
            cross_corr = float(np.mean(valid))
            cross_p = float(np.mean([v for v in [c_p, h_p] if math.isfinite(v)]))
            n_arch_pairs = len(c_pairs) + len(h_pairs)

        archetype_pattern["coffee_loyalty_rank_ndcg_corr"] = (
            round(c_corr, 4) if math.isfinite(c_corr) else c_corr
        )
        archetype_pattern["hotel_retention_rank_ndcg_corr"] = (
            round(h_corr, 4) if math.isfinite(h_corr) else h_corr
        )

    return {
        "cross_domain_corr": (
            round(cross_corr, 4) if math.isfinite(cross_corr) else cross_corr
        ),
        "p_value": round(cross_p, 4) if math.isfinite(cross_p) else cross_p,
        "n_shared": n_arch_pairs,
        "archetype_pattern": archetype_pattern,
    }


def _build_archetype_pattern(df: pd.DataFrame) -> Dict:
    """
    Build per-domain, per-archetype mean NDCG and mean revisit_score.
    Also compute whether coffee Loyalists have higher revisit_score than
    non-Loyalists (a within-domain consistency check).
    """
    pattern: Dict = {}

    for domain in DOMAINS:
        dom = df[df["domain"] == domain]
        if dom.empty:
            continue
        domain_pattern: Dict = {}
        arch_stats = (
            dom.groupby("archetype")
            .agg(
                mean_ndcg=("ndcg", "mean"),
                mean_revisit=("revisit_score", "mean"),
                n=("ndcg", "count"),
            )
            .to_dict(orient="index")
        )
        for arch, stats_dict in arch_stats.items():
            domain_pattern[arch] = {
                "mean_ndcg": round(float(stats_dict["mean_ndcg"]), 4),
                "mean_revisit": round(float(stats_dict["mean_revisit"]), 4),
                "n": int(stats_dict["n"]),
            }
        pattern[domain] = domain_pattern

    # Coffee-specific: Loyalist revisit vs non-Loyalist
    coffee_df = df[df["domain"] == "coffee"]
    if not coffee_df.empty:
        loyalist_revisit = coffee_df[coffee_df["archetype"] == "Loyalist"][
            "revisit_score"
        ].mean()
        non_loyalist_revisit = coffee_df[coffee_df["archetype"] != "Loyalist"][
            "revisit_score"
        ].mean()
        pattern["coffee_loyalist_revisit_premium"] = round(
            float(loyalist_revisit - non_loyalist_revisit)
            if (
                math.isfinite(float(loyalist_revisit))
                and math.isfinite(float(non_loyalist_revisit))
            )
            else float("nan"),
            4,
        )

    return pattern


# ---------------------------------------------------------------------------
# 3. compute_per_persona_variance
# ---------------------------------------------------------------------------


def compute_per_persona_variance(records_df: pd.DataFrame) -> Dict:
    """
    Groups by (domain, archetype) and computes variance in NDCG across
    personas within each group.  Because v1 ran 1 trial per persona, variance
    here is the between-persona NDCG variance within an archetype.

    Also estimates how many repeated trials per persona would be needed to
    achieve a 95% CI half-width of ±0.005 on the group mean.

    Parameters
    ----------
    records_df : pd.DataFrame

    Returns
    -------
    dict:
        {domain: {archetype: {ndcg_mean, ndcg_std, recommended_n_repeats}}}
    """
    df = records_df.copy()
    result: Dict = {}

    for domain in DOMAINS:
        dom_df = df[df["domain"] == domain]
        if dom_df.empty:
            result[domain] = {}
            continue

        domain_result: Dict = {}
        for arch, arch_df in dom_df.groupby("archetype"):
            ndcg_vals = arch_df["ndcg"].astype(float).dropna().values
            n = len(ndcg_vals)
            mean_val = float(np.mean(ndcg_vals)) if n > 0 else float("nan")
            std_val = float(np.std(ndcg_vals, ddof=1)) if n > 1 else float("nan")
            rec_n = _recommended_n_repeats(std_val) if math.isfinite(std_val) else 1

            domain_result[arch] = {
                "ndcg_mean": round(mean_val, 4) if math.isfinite(mean_val) else mean_val,
                "ndcg_std": round(std_val, 4) if math.isfinite(std_val) else std_val,
                "recommended_n_repeats": rec_n,
            }

        result[domain] = domain_result

    return result


# ---------------------------------------------------------------------------
# 4. generate_calibration_plot_data
# ---------------------------------------------------------------------------


def generate_calibration_plot_data(
    calibration_results: Dict,
    records_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Build scatter-plot data for revisit calibration and optionally save a PNG.

    The calibration_results dict (from compute_revisit_calibration) is used
    only for annotating the figure with Spearman r.  The raw scatter values
    are drawn from records_df (loaded from RECORDS_CSV if not provided).

    Parameters
    ----------
    calibration_results : dict   output of compute_revisit_calibration
    records_df          : optional DataFrame; loaded from disk if None
    output_dir          : directory for PNG; defaults to RESULTS_DIR

    Returns
    -------
    dict:
        {domain: {"x": [...], "y": [...], "labels": [...]}}
    where x = model_revisit_signal, y = revisit_score (norm 0-1),
    labels = archetype name per point.
    """
    if records_df is None:
        records_df = pd.read_csv(RECORDS_CSV)

    if output_dir is None:
        output_dir = RESULTS_DIR

    df = records_df.copy()
    df["revisit_norm"] = _normalise_revisit(df["revisit_score"])
    df["model_signal"] = df["model_revisit_signal"].astype(float)

    plot_data: Dict = {}

    for domain in DOMAINS:
        dom_df = df[df["domain"] == domain].dropna(
            subset=["revisit_norm", "model_signal"]
        )
        plot_data[domain] = {
            "x": dom_df["model_signal"].tolist(),
            "y": dom_df["revisit_norm"].tolist(),
            "labels": dom_df["archetype"].tolist(),
        }

    # --- Optional matplotlib figure -----------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = {
            arch: c
            for arch, c in zip(
                df["archetype"].unique(),
                plt.cm.tab10.colors,  # type: ignore[attr-defined]
            )
        }

        for ax, domain in zip(axes, DOMAINS):
            pdata = plot_data[domain]
            x_arr = np.array(pdata["x"])
            y_arr = np.array(pdata["y"])
            label_arr = pdata["labels"]

            unique_archs = list(dict.fromkeys(label_arr))
            for arch in unique_archs:
                mask = [lb == arch for lb in label_arr]
                ax.scatter(
                    x_arr[mask],
                    y_arr[mask],
                    label=arch,
                    alpha=0.5,
                    s=15,
                    color=colors.get(arch, "grey"),
                )

            # Regression line (if enough variance)
            if len(x_arr) > 1 and np.std(x_arr) > 0 and np.std(y_arr) > 0:
                m, b, *_ = stats.linregress(x_arr, y_arr)
                x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
                ax.plot(x_line, m * x_line + b, "k--", linewidth=1, alpha=0.7)

            # Annotate Spearman r
            sp_r = (
                calibration_results.get(domain, {})
                .get("overall", {})
                .get("spearman_r", float("nan"))
            )
            sp_p = (
                calibration_results.get(domain, {})
                .get("overall", {})
                .get("spearman_p", float("nan"))
            )
            r_label = (
                f"ρ = {sp_r:.3f}"
                if math.isfinite(sp_r)
                else "ρ = n/a"
            )
            p_label = (
                f"p = {sp_p:.3f}"
                if math.isfinite(sp_p)
                else ""
            )
            ax.text(
                0.05,
                0.95,
                f"{r_label}\n{p_label}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

            domain_labels = {
                "coffee": "Coffee Shops",
                "restaurant": "Restaurants",
                "hotel": "Hotels",
            }
            ax.set_title(domain_labels.get(domain, domain), fontsize=11)
            ax.set_xlabel("Model revisit signal", fontsize=9)
            ax.set_ylabel("Persona revisit score (norm.)", fontsize=9)
            ax.legend(fontsize=6, loc="lower right", markerscale=1.2)

        fig.suptitle(
            "Revisit Calibration: LLM Persona Score vs Model Signal",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        png_path = os.path.join(output_dir, "calibration_scatter.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Calibration scatter plot saved → {png_path}")

    except ImportError:
        pass  # matplotlib not available; skip gracefully

    return plot_data


# ---------------------------------------------------------------------------
# 5. save_calibration_report
# ---------------------------------------------------------------------------


def save_calibration_report(
    calibration: Dict,
    cross_domain: Dict,
    variance: Dict,
    output_dir: Optional[str] = None,
) -> None:
    """
    Write a Markdown calibration report to {output_dir}/calibration_report.md.

    Parameters
    ----------
    calibration  : output of compute_revisit_calibration
    cross_domain : output of run_cross_domain_consistency
    variance     : output of compute_per_persona_variance
    output_dir   : directory to write report; defaults to RESULTS_DIR
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: List[str] = [
        "# Phase 3 Calibration Analysis Report",
        "",
        f"**Generated:** {ts}  ",
        f"**Source data:** `simulation_records.csv`  ",
        "",
        "---",
        "",
        "## 1. Revisit Calibration",
        "",
        "Spearman rank correlation and Pearson r between the LLM persona's "
        "self-reported revisit likelihood (normalised 0–1) and the venue's "
        "empirical revisit signal derived from the behavioural ranking model.",
        "",
        "| Domain | n | Spearman ρ | p-value | Pearson r | p-value |",
        "|--------|---|-----------|---------|-----------|---------|",
    ]

    domain_labels = {
        "coffee": "Coffee Shops",
        "restaurant": "Restaurants",
        "hotel": "Hotels",
    }

    for domain in DOMAINS:
        cal = calibration.get(domain, {}).get("overall", {})
        n = cal.get("n", 0)
        sp_r = cal.get("spearman_r", float("nan"))
        sp_p = cal.get("spearman_p", float("nan"))
        pe_r = cal.get("pearson_r", float("nan"))
        pe_p = cal.get("pearson_p", float("nan"))

        def _fmt(v: float, dec: int = 4) -> str:
            return f"{v:.{dec}f}" if math.isfinite(v) else "n/a"

        sig_sp = "**" if math.isfinite(sp_p) and sp_p < 0.05 else ""
        sig_pe = "**" if math.isfinite(pe_p) and pe_p < 0.05 else ""

        lines.append(
            f"| {domain_labels.get(domain, domain)} | {n} "
            f"| {sig_sp}{_fmt(sp_r)}{sig_sp} | {_fmt(sp_p)} "
            f"| {sig_pe}{_fmt(pe_r)}{sig_pe} | {_fmt(pe_p)} |"
        )

    lines += ["", "Bold values indicate p < 0.05.", ""]

    # Per-archetype breakdown
    lines += [
        "### Per-Archetype Revisit Calibration (Spearman ρ)",
        "",
    ]
    for domain in DOMAINS:
        by_arch = calibration.get(domain, {}).get("by_archetype", {})
        if not by_arch:
            continue
        lines += [
            f"**{domain_labels.get(domain, domain)}**",
            "",
            "| Archetype | n | Spearman ρ |",
            "|-----------|---|-----------|",
        ]
        for arch, arch_cal in sorted(by_arch.items()):
            arch_r = arch_cal.get("spearman_r", float("nan"))
            arch_n = arch_cal.get("n", 0)
            lines.append(
                f"| {arch} | {arch_n} "
                f"| {_fmt(arch_r) if math.isfinite(arch_r) else 'n/a'} |"  # type: ignore[arg-type]
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 2. Cross-Domain Consistency",
        "",
    ]

    cd_corr = cross_domain.get("cross_domain_corr", float("nan"))
    cd_p = cross_domain.get("p_value", float("nan"))
    cd_n = cross_domain.get("n_shared", 0)

    if cd_n > 0 and math.isfinite(cd_corr):
        lines += [
            f"Cross-domain NDCG correlation: **{cd_corr:.4f}** "
            f"(p = {cd_p:.4f}, n = {cd_n})",
            "",
        ]
    else:
        lines += [
            "No shared persona IDs exist across domains (v1 design: unique IDs per domain).  ",
            "Archetype-level loyalty-rank × NDCG Spearman correlation used as proxy.",
            "",
        ]

    arch_patt = cross_domain.get("archetype_pattern", {})
    c_rank_corr = arch_patt.get("coffee_loyalty_rank_ndcg_corr", float("nan"))
    h_rank_corr = arch_patt.get("hotel_retention_rank_ndcg_corr", float("nan"))
    loyalist_premium = arch_patt.get("coffee_loyalist_revisit_premium", float("nan"))

    if math.isfinite(c_rank_corr):
        lines.append(
            f"- Coffee loyalty-rank vs NDCG Spearman ρ: **{c_rank_corr:.4f}**"
        )
    if math.isfinite(h_rank_corr):
        lines.append(
            f"- Hotel retention-rank vs NDCG Spearman ρ: **{h_rank_corr:.4f}**"
        )
    if math.isfinite(loyalist_premium):
        direction = "higher" if loyalist_premium > 0 else "lower"
        lines.append(
            f"- Coffee Loyalist revisit score premium over non-Loyalists: "
            f"**{loyalist_premium:+.2f}** ({direction} as expected for high-retention archetype)"
        )
    lines.append("")

    # Archetype mean NDCG table per domain
    for domain in DOMAINS:
        domain_arch_patt = arch_patt.get(domain, {})
        if not domain_arch_patt:
            continue
        lines += [
            f"**{domain_labels.get(domain, domain)} — archetype mean NDCG**",
            "",
            "| Archetype | n | Mean NDCG | Mean Revisit Score |",
            "|-----------|---|-----------|-------------------|",
        ]
        for arch, vals in sorted(domain_arch_patt.items()):
            lines.append(
                f"| {arch} | {vals['n']} "
                f"| {vals['mean_ndcg']:.4f} "
                f"| {vals['mean_revisit']:.2f} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 3. Power Analysis — Recommended Repeats per Archetype",
        "",
        "Estimates the number of repeated trials per persona needed to achieve "
        "a ±0.005 half-width 95% confidence interval on archetype mean NDCG, "
        "based on observed between-persona NDCG standard deviation.  "
        "Formula: n = ⌈(1.96 × σ / 0.005)²⌉.",
        "",
    ]

    for domain in DOMAINS:
        var_data = variance.get(domain, {})
        if not var_data:
            continue
        lines += [
            f"**{domain_labels.get(domain, domain)}**",
            "",
            "| Archetype | Mean NDCG | Std Dev | Recommended Repeats |",
            "|-----------|-----------|---------|---------------------|",
        ]
        for arch, vdata in sorted(var_data.items()):
            mn = vdata.get("ndcg_mean", float("nan"))
            sd = vdata.get("ndcg_std", float("nan"))
            rn = vdata.get("recommended_n_repeats", 1)
            mn_s = f"{mn:.4f}" if math.isfinite(mn) else "n/a"
            sd_s = f"{sd:.4f}" if math.isfinite(sd) else "n/a"
            lines.append(f"| {arch} | {mn_s} | {sd_s} | {rn} |")
        lines.append("")

    lines += [
        "---",
        "",
        "## Interpretation",
        "",
        "- **Revisit calibration** measures whether LLM personas internalise "
        "the same behavioural signal that the ranking model is optimising.  "
        "A positive Spearman ρ indicates the LLM's stated revisit likelihood "
        "co-varies with the empirical revisit rate, providing evidence of "
        "construct validity.",
        "",
        "- **Cross-domain consistency** examines whether the loyalty-driven "
        "NDCG advantage observed in coffee also appears in the hotel domain.  "
        "A positive rank-NDCG correlation confirms that retention-oriented "
        "personas systematically prefer retention-optimised rankings.",
        "",
        "- **Power analysis** informs future study design.  Large recommended "
        "repeat counts reflect high between-persona NDCG variance within an "
        "archetype, suggesting heterogeneous sub-populations that may warrant "
        "finer-grained segmentation.",
        "",
        "---",
        "",
        "_Generated by `calibration_analysis.py` — "
        "Master Project Behavioural Venue Ranking_",
    ]

    report_path = os.path.join(output_dir, "calibration_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Calibration report saved → {report_path}")


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_all(
    csv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Load simulation records and run all calibration analyses.

    Returns
    -------
    dict with keys: calibration, cross_domain, variance, plot_data
    """
    if csv_path is None:
        csv_path = RECORDS_CSV
    if output_dir is None:
        output_dir = RESULTS_DIR

    records_df = pd.read_csv(csv_path)

    calibration = compute_revisit_calibration(records_df)
    cross_domain = run_cross_domain_consistency(records_df)
    variance = compute_per_persona_variance(records_df)
    plot_data = generate_calibration_plot_data(
        calibration, records_df=records_df, output_dir=output_dir
    )
    save_calibration_report(calibration, cross_domain, variance, output_dir)

    return {
        "calibration": calibration,
        "cross_domain": cross_domain,
        "variance": variance,
        "plot_data": plot_data,
    }


if __name__ == "__main__":
    results = run_all()

    import json

    def _json_safe(obj: object) -> object:
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        return obj

    print(json.dumps(_json_safe(results["calibration"]), indent=2))
