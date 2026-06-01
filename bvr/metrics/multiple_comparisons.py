"""
V6: Benjamini-Hochberg correction on all pairwise comparisons

Collects all p-values from benchmark_results.json + other validation files,
applies BH correction at FDR=0.05, reports which findings survive.

Saves: bh_corrected_pvalues.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import false_discovery_control

DATA_DIR = Path(__file__).parent


def bh_correct(p_values: list) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.array(p_values, dtype=float)
    n = len(p)
    # Sort p-values, compute adjusted, unsort
    order  = np.argsort(p)
    ranked = np.arange(1, n + 1)
    adj    = np.minimum(1.0, p[order] * n / ranked)
    # Ensure monotonicity from the right
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    result       = np.empty(n)
    result[order] = adj
    return result


if __name__ == "__main__":
    rows = []

    # 1. From benchmark_results.json (Wilcoxon p-values, NDCG)
    bench_path = DATA_DIR / "benchmark_results.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        for dataset, methods in bench.items():
            for method, vals in methods.items():
                wp = vals.get("wilcoxon_vs_hybrid_p")
                if wp is not None:
                    rows.append({"source": "benchmark_ndcg",
                                 "dataset": dataset, "method": method,
                                 "test": "Wilcoxon NDCG@10 vs hybrid",
                                 "p_raw": float(wp)})

    # 2. Rising-stars ρ p-values from validation summaries
    london_rho_pvals = {
        "baseline_popularity":    0.187,
        "baseline_rating":        0.000,
        "birank_count":           0.000,
        "birank_decay":           0.000,
        "birank_explore":         0.211,
        "mf_als":                 0.943,
        "mf_bpr":                 0.616,
        "hybrid_explore_als":     0.000,
        "lightgcn":               0.015,
        "baseline_random":        1.000,
    }
    for method, p in london_rho_pvals.items():
        rows.append({"source": "rising_stars_rho",
                     "dataset": "london", "method": method,
                     "test": "Spearman ρ (rising stars) ≠ 0",
                     "p_raw": p})

    fsq_rho_pvals = {
        "baseline_popularity":    0.000,
        "birank_count":           0.000,
        "birank_decay":           0.000,
        "birank_explore":         0.000,
        "mf_als":                 0.000,
        "mf_bpr":                 0.981,
        "hybrid_explore_als":     0.000,
        "lightgcn":               0.000,
        "sasrec":                 0.000,
        "baseline_random":        0.970,
    }
    for method, p in fsq_rho_pvals.items():
        rows.append({"source": "rising_stars_rho",
                     "dataset": "uk_fsq", "method": method,
                     "test": "Spearman ρ (rising stars) ≠ 0",
                     "p_raw": p})

    # 3. Causal PSM
    rows.append({"source": "causal_psm", "dataset": "coffee",
                 "method": "PSM_ATE", "test": "ATE > 0",
                 "p_raw": 0.0309})
    rows.append({"source": "causal_psm", "dataset": "coffee",
                 "method": "Mahalanobis_ATE", "test": "ATE > 0",
                 "p_raw": 0.0355})

    df = pd.DataFrame(rows)

    # Apply BH correction
    p_adj = bh_correct(df["p_raw"].values)
    df["p_adj_BH"]         = p_adj.round(4)
    df["sig_raw_0.05"]     = df["p_raw"] < 0.05
    df["sig_adj_0.05"]     = df["p_adj_BH"] < 0.05
    df["survives_BH"]      = df["sig_adj_0.05"]

    # Summary
    n_total     = len(df)
    n_raw_sig   = df["sig_raw_0.05"].sum()
    n_adj_sig   = df["sig_adj_0.05"].sum()
    n_lost      = n_raw_sig - n_adj_sig

    print("=== BENJAMINI-HOCHBERG CORRECTION ===")
    print(f"Total tests:            {n_total}")
    print(f"Significant raw p<0.05: {n_raw_sig}")
    print(f"Significant BH-adj:     {n_adj_sig}")
    print(f"Lost significance:      {n_lost}")
    print()

    if n_lost > 0:
        lost = df[df["sig_raw_0.05"] & ~df["sig_adj_0.05"]]
        print("Lost after BH correction:")
        for _, r in lost.iterrows():
            print(f"  {r['dataset']} | {r['method']} | p_raw={r['p_raw']:.4f} → p_adj={r['p_adj_BH']:.4f}")
    else:
        print("All significant results survive BH correction ✓")

    print("\nKey thesis findings after BH:")
    key = df[df["method"].isin(["hybrid_explore_als", "PSM_ATE", "Mahalanobis_ATE"])]
    print(key[["dataset","method","p_raw","p_adj_BH","survives_BH"]].to_string(index=False))

    df.to_csv(DATA_DIR / "bh_corrected_pvalues.csv", index=False)
    print(f"\nSaved → bh_corrected_pvalues.csv")
