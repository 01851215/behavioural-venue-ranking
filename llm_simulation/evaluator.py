"""Metrics: NDCG, Hit@k, Kendall τ, bootstrap CI, Wilcoxon, BH correction,
Cohen's d, rank-biserial correlation, stratified bootstrap. (Phase 1+4)"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from collections import defaultdict
from scipy.stats import wilcoxon, kendalltau, pearsonr
from config import BOOTSTRAP_SAMPLES, RANDOM_SEED


# ── NDCG ─────────────────────────────────────────────────────────────────────

def dcg_at_k(relevances: List[float], k: int) -> float:
    arr = np.array(relevances[:k], dtype=float)
    if len(arr) == 0:
        return 0.0
    return np.sum(arr / np.log2(np.arange(1, len(arr) + 1) + 1))


def ndcg_at_k(predicted_order: List[int], ground_truth_ranks: Dict[int, int], k: int = 10) -> float:
    """relevance = 1/log2(model_rank+1); lower model_rank = more relevant."""
    max_rank = max(ground_truth_ranks.values()) if ground_truth_ranks else 10

    def rel(idx: int) -> float:
        r = ground_truth_ranks.get(idx, max_rank + 1)
        return 1.0 / np.log2(r + 1)

    pred_rel = [rel(i) for i in predicted_order[:k]]
    ideal_order = sorted(ground_truth_ranks, key=lambda x: ground_truth_ranks[x])
    ideal_rel = [rel(i) for i in ideal_order[:k]]

    actual = dcg_at_k(pred_rel, k)
    ideal = dcg_at_k(ideal_rel, k)
    return actual / ideal if ideal > 0 else 0.0


# ── Hit@k ─────────────────────────────────────────────────────────────────────

def hit_at_k(
    predicted_order: List[int],
    ground_truth_ranks: Dict[int, int],
    k: int = 10,
    relevant_cutoff: int = 3,  # venues whose model_rank <= this count as "relevant"
) -> float:
    """1 if any of the persona's top-k picks are in the model's top-{relevant_cutoff}."""
    top_model = {idx for idx, r in ground_truth_ranks.items() if r <= relevant_cutoff}
    return 1.0 if any(idx in top_model for idx in predicted_order[:k]) else 0.0


def hits_all_k(
    predicted_order: List[int],
    ground_truth_ranks: Dict[int, int],
    relevant_cutoff: int = 3,
) -> Dict[str, float]:
    """Return Hit@1, Hit@3, Hit@5, Hit@10 in one pass."""
    top_model = {idx for idx, r in ground_truth_ranks.items() if r <= relevant_cutoff}
    return {
        "hit_at_1":  1.0 if any(i in top_model for i in predicted_order[:1])  else 0.0,
        "hit_at_3":  1.0 if any(i in top_model for i in predicted_order[:3])  else 0.0,
        "hit_at_5":  1.0 if any(i in top_model for i in predicted_order[:5])  else 0.0,
        "hit_at_10": 1.0 if any(i in top_model for i in predicted_order[:10]) else 0.0,
    }


# ── Kendall τ ─────────────────────────────────────────────────────────────────

def kendall_tau_score(
    predicted_order: List[int],
    ground_truth_ranks: Dict[int, int],
) -> float:
    """Kendall τ-b between persona ranking and model ranking."""
    indices = [i for i in predicted_order if i in ground_truth_ranks]
    if len(indices) < 2:
        return 0.0
    pred_pos = list(range(len(indices)))
    model_pos = [ground_truth_ranks[i] for i in indices]
    tau, _ = kendalltau(pred_pos, model_pos)
    return float(tau) if not np.isnan(tau) else 0.0


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: List[float],
    n_samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    arr = np.array(values)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_samples)]
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(arr)), lo, hi


def bootstrap_ci_stratified(
    records: List[dict],
    strata_col: str,
    metric_col: str,
    n_samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float, float]]:
    """Stratified bootstrap: resample within strata, aggregate means."""
    strata: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        strata[r[strata_col]].append(r[metric_col])

    rng = np.random.default_rng(RANDOM_SEED)
    pool_means = []
    for _ in range(n_samples):
        stratum_means = []
        for vals in strata.values():
            arr = np.array(vals)
            stratum_means.append(rng.choice(arr, size=len(arr), replace=True).mean())
        pool_means.append(float(np.mean(stratum_means)))

    overall_mean = float(np.mean([v for vals in strata.values() for v in vals]))
    lo = float(np.percentile(pool_means, 100 * alpha / 2))
    hi = float(np.percentile(pool_means, 100 * (1 - alpha / 2)))
    return {"mean": overall_mean, "lo": lo, "hi": hi}


# ── Wilcoxon ─────────────────────────────────────────────────────────────────

def wilcoxon_test(method_scores: List[float], baseline_scores: List[float]) -> float:
    diffs = np.array(method_scores) - np.array(baseline_scores)
    if np.all(diffs == 0) or len(diffs) < 10:
        return 1.0
    try:
        _, p = wilcoxon(diffs, alternative="two-sided", zero_method="zsplit")
        return float(p)
    except ValueError:
        return 1.0


# ── Effect sizes (Phase 4) ────────────────────────────────────────────────────

def cohen_d(a: List[float], b: List[float]) -> float:
    """Cohen's d: (mean_a - mean_b) / pooled_std."""
    a_arr, b_arr = np.array(a), np.array(b)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return 0.0
    pooled_std = np.sqrt((np.var(a_arr, ddof=1) + np.var(b_arr, ddof=1)) / 2)
    return float((a_arr.mean() - b_arr.mean()) / pooled_std) if pooled_std > 0 else 0.0


def rank_biserial(a: List[float], b: List[float]) -> float:
    """Rank-biserial correlation for Wilcoxon test (effect size)."""
    a_arr, b_arr = np.array(a), np.array(b)
    n1, n2 = len(a_arr), len(b_arr)
    if n1 == 0 or n2 == 0:
        return 0.0
    u_stat = sum(
        1.0 if x > y else (0.5 if x == y else 0.0)
        for x in a_arr for y in b_arr
    )
    return float(2 * u_stat / (n1 * n2) - 1)


# ── Benjamini-Hochberg correction (Phase 4) ───────────────────────────────────

def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """Return BH-adjusted p-values (same order as input)."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    adjusted = np.minimum(sorted_p * n / (np.arange(1, n + 1)), 1.0)
    # enforce monotonicity (take running min from right)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(n)
    result[order] = adjusted
    return result.tolist()


# ── Power analysis helper (Phase 4) ──────────────────────────────────────────

def minimum_detectable_effect(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Approximate MDE for a two-sided Wilcoxon test given sample size."""
    # Asymptotic normal approximation: z_alpha/2 + z_beta
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    return float((z_alpha + z_beta) / np.sqrt(n))


# ── Aggregate metrics (updated for Phase 1+4) ────────────────────────────────

def compute_metrics(records: List[dict]) -> dict:
    """
    Expects records with fields:
        domain, archetype, ndcg, hit_at_1, hit_at_3, hit_at_10,
        kendall_tau, stars_ndcg, pairwise_win
    Also includes BH-corrected p-values and effect sizes.
    """
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in records:
        groups[(r["domain"], r["archetype"])].append(r)
        groups[(r["domain"], "OVERALL")].append(r)
        groups[("ALL", "OVERALL")].append(r)

    # Collect raw p-values for BH correction
    raw_p: List[float] = []
    group_keys = list(groups.keys())

    results = {}
    for (domain, archetype), recs in groups.items():
        ndcg_vals  = [r["ndcg"] for r in recs]
        stars_ndcg = [r.get("stars_ndcg", r["ndcg"]) for r in recs]
        hit1_vals  = [r.get("hit_at_1", 0) for r in recs]
        hit3_vals  = [r.get("hit_at_3", 0) for r in recs]
        hit10_vals = [r.get("hit_at_10", r.get("hit", 0)) for r in recs]
        tau_vals   = [r.get("kendall_tau", 0) for r in recs]
        pairwise   = [r.get("pairwise_win", 0) for r in recs]

        mean_ndcg, lo, hi = bootstrap_ci(ndcg_vals)
        p_val = wilcoxon_test(ndcg_vals, stars_ndcg)
        d = cohen_d(ndcg_vals, stars_ndcg)
        rb = rank_biserial(ndcg_vals, stars_ndcg)
        mde = minimum_detectable_effect(len(recs))

        results[(domain, archetype)] = {
            "n": len(recs),
            "ndcg_mean": round(mean_ndcg, 4),
            "ndcg_lo": round(lo, 4),
            "ndcg_hi": round(hi, 4),
            "hit_at_1": round(float(np.mean(hit1_vals)), 4),
            "hit_at_3": round(float(np.mean(hit3_vals)), 4),
            "hit_at_10": round(float(np.mean(hit10_vals)), 4),
            "kendall_tau": round(float(np.mean(tau_vals)), 4),
            "stars_ndcg_mean": round(float(np.mean(stars_ndcg)), 4),
            "delta_vs_stars": round(mean_ndcg - float(np.mean(stars_ndcg)), 4),
            "pairwise_win_rate": round(float(np.mean(pairwise)), 3),
            "wilcoxon_p": round(p_val, 4),
            "wilcoxon_p_bh": None,  # filled in below
            "cohen_d": round(d, 4),
            "rank_biserial": round(rb, 4),
            "mde_80pct": round(mde, 4),
        }
        raw_p.append(p_val)

    # BH correction across all group comparisons
    adjusted_p = benjamini_hochberg(raw_p)
    for i, key in enumerate(group_keys):
        if key in results:
            results[key]["wilcoxon_p_bh"] = round(adjusted_p[i], 4)

    return results
