"""
V5: LightGCN layer ablation — L=1,2,3,4,5

If rising-stars ρ is consistently negative regardless of L,
the failure is structural (popularity amplification) not a tuning issue.

Saves: lightgcn_layer_ablation.csv
"""

import sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from bvr.models.lightgcn import build_lightgcn_ranking
from bvr.core.validation import evaluate_per_user
from bvr.pipelines.london import temporal_split, compute_rising_stars, spearman_rising

DATA_DIR = Path(__file__).parent
LAYERS   = [1, 2, 3, 4, 5]


def run_layer_ablation(interactions_file, split_date, label, has_stars=True):
    print(f"\n=== {label} ===")
    df = pd.read_csv(interactions_file, dtype={"user_id": str, "business_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not has_stars:
        df["stars"] = np.nan

    train, test_uv_rev, train_uv, _ = temporal_split(df, split_date)
    test_full   = df[df["timestamp"] >= pd.Timestamp(split_date)].copy()
    rising_stars = compute_rising_stars(train, test_full)

    print(f"{'L':>4} {'ρ(rising)':>12} {'p-value':>10} {'NDCG@10':>10} {'time':>8}")
    print("-" * 50)

    rows = []
    for L in LAYERS:
        t0 = time.time()
        ranking = build_lightgcn_ranking(train, n_layers=L, verbose=False)
        rho, pval  = spearman_rising(ranking, rising_stars)
        agg, _, _  = evaluate_per_user(ranking, train_uv, test_uv_rev)
        ndcg       = agg.get("NDCG@10", 0)
        elapsed    = time.time() - t0
        sig        = "***" if pval < 0.001 else ("*" if pval < 0.05 else "ns")
        print(f"{L:>4} {rho:>+12.4f} {pval:>10.4f} {sig:>3} {ndcg:>10.4f} {elapsed:>6.1f}s")
        rows.append({"dataset": label, "n_layers": L,
                     "rho": rho, "p_value": pval, "ndcg10": ndcg})
    return rows


if __name__ == "__main__":
    all_rows = []
    all_rows += run_layer_ablation(
        DATA_DIR / "london_interactions.csv", "2018-01-01", "London TripAdvisor"
    )
    all_rows += run_layer_ablation(
        DATA_DIR / "uk_fsq_interactions.csv", "2013-07-01", "UK Foursquare", has_stars=False
    )

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(DATA_DIR / "lightgcn_layer_ablation.csv", index=False)
    print(f"\nSaved → lightgcn_layer_ablation.csv")
    print(df_out.to_string(index=False))

    print("\n=== VERDICT ===")
    for dataset in df_out["dataset"].unique():
        sub = df_out[df_out["dataset"] == dataset]
        all_negative = (sub["rho"] < 0).all()
        best_rho     = sub["rho"].max()
        print(f"{dataset}: all_negative={all_negative}  best_ρ={best_rho:+.4f}")
        if all_negative:
            print(f"  → ρ negative at ALL L values — failure is structural, not under-tuned ✓")
        else:
            print(f"  → ρ positive at some L — LightGCN can work with right depth ⚠")
