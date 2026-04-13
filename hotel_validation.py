"""
Phase 7: Hotel Model Validation

Temporal split validation for the hotel BiRank model using the same
rigorous framework as coffee validate_v5.py:
  - Feature leakage prevention (features from training data only)
  - Corrected NDCG (IDCG uses total relevant candidates)
  - Bootstrap 95% CI (1000 samples)
  - Wilcoxon signed-rank significance tests
  - Per-group evaluation (hotel user archetypes)

Train / Test split: 2020-01-01
Methods compared:
  1. hotel_birank           — behavioral priors (Phase 5)
  2. hotel_birank_fsq       — + Foursquare social (Phase 6)
  3. hotel_birank_xdomain   — + cross-domain transfer priors (Phase 4)
  4. hotel_birank_full      — all signals combined
  5. baseline_rating        — Yelp star average
  6. baseline_popularity    — review count
  7. baseline_random        — random permutation
  8. baseline_item_knn      — cosine similarity item-KNN

Outputs:
  hotel_validation_results.csv
  hotel_validation_per_group.csv
  hotel_validation_summary.txt
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from collections import defaultdict
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent

SPLIT_DATE   = pd.Timestamp("2020-01-01")
K_VALUES     = [5, 10, 20]
N_BOOTSTRAP  = 1000
ALPHA        = 0.85
LAMBDA_DECAY = 0.5
SOCIAL_GAMMA = 0.15
BIRANK_ITER  = 200

# ── Metric helpers ─────────────────────────────────────────────────────────────
def dcg_at_k(ranked, relevant, k):
    score = 0.0
    for i, item in enumerate(ranked[:k]):
        if item in relevant:
            score += 1.0 / np.log2(i + 2)
    return score

def ndcg_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    n_rel = sum(1 for v in ranked if v in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, n_rel)))
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg

def hit_at_k(ranked, relevant, k):
    return int(any(v in relevant for v in ranked[:k]))

def bootstrap_ci(scores, n_boot=N_BOOTSTRAP, conf=0.95):
    n = len(scores)
    rng = np.random.RandomState(42)
    means = [scores[rng.choice(n, n, replace=True)].mean() for _ in range(n_boot)]
    alpha = (1 - conf) / 2
    return float(np.percentile(means, 100*alpha)), float(np.percentile(means, 100*(1-alpha)))

def wilcoxon_p(a, b):
    diff = a - b
    if np.all(diff == 0): return 1.0
    try:
        _, p = wilcoxon(diff, alternative="two-sided")
        return float(p)
    except: return 1.0

# ── BiRank helper ──────────────────────────────────────────────────────────────
def run_birank(edges, v_prior, u_prior, n_u, n_v, u2i, v2i):
    rows_u = [u2i[u] for u in edges["user_id"]]
    cols_v = [v2i[v] for v in edges["business_id"]]
    data   = edges["edge_weight"].values.astype(np.float64)
    B  = sparse.csr_matrix((data, (rows_u, cols_v)), shape=(n_u, n_v))
    Bt = B.T.tocsr()

    def row_norm(M):
        s = np.asarray(M.sum(axis=1)).flatten()
        s[s == 0] = 1.0
        return sparse.diags(1.0/s) @ M

    B_norm  = row_norm(B)
    Bt_norm = row_norm(Bt)
    u_sc = u_prior.copy()
    v_sc = v_prior.copy()
    for _ in range(BIRANK_ITER):
        u_new = ALPHA * (B_norm @ v_sc) + (1-ALPHA) * u_prior
        v_new = ALPHA * (Bt_norm @ u_sc) + (1-ALPHA) * v_prior
        u_new /= u_new.sum(); v_new /= v_new.sum()
        if np.abs(u_new-u_sc).max() + np.abs(v_new-v_sc).max() < 1e-8:
            break
        u_sc, v_sc = u_new, v_new
    i2v = {i: v for v, i in v2i.items()}
    return {i2v[i]: float(v_sc[i]) for i in range(n_v)}

# ── Feature computation (training data only) ───────────────────────────────────
def compute_features_from_train(train_reviews, businesses):
    """Recomputes all behavioral features from training data only."""

    def gini(arr):
        arr = np.sort(np.abs(arr))
        n = len(arr)
        if n == 0 or arr.sum() == 0: return 0.0
        idx = np.arange(1, n+1)
        return float(2*(idx*arr).sum()/(n*arr.sum()) - (n+1)/n)

    train_reviews = train_reviews.copy()
    train_reviews["dow"] = train_reviews["timestamp"].dt.dayofweek
    train_reviews["is_weekday"] = train_reviews["dow"] < 4
    train_reviews["month"] = train_reviews["timestamp"].dt.month
    train_reviews["year"]  = train_reviews["timestamp"].dt.year

    now = train_reviews["timestamp"].max()

    # Venue features
    venue_feats = {}
    for vid, vr in train_reviews.groupby("business_id"):
        u_counts = vr["user_id"].value_counts()
        ages = (now - vr["timestamp"]).dt.days.values / 365.0
        monthly = vr["month"].value_counts()
        annual  = vr["year"].value_counts()
        venue_feats[vid] = {
            "review_velocity":       float(np.exp(-0.5*ages).sum()),
            "multi_stay_rate":       float((u_counts >= 2).sum() / len(u_counts)),
            "seasonal_cv":           float(monthly.std()/monthly.mean()) if len(monthly)>1 else 0.0,
            "venue_stability_cv":    float(annual.std()/annual.mean()) if len(annual)>1 else 0.0,
            "business_leisure_ratio": float(vr["is_weekday"].mean()),
            "traveler_concentration": gini(u_counts.values),
        }

    # Geographic diversity (state entropy per venue)
    rev_geo = train_reviews.merge(
        businesses[["business_id","state"]], on="business_id", how="left"
    )
    user_state_map = (rev_geo.groupby("user_id")["state"]
                      .agg(lambda x: x.mode()[0] if len(x)>0 else "?")
                      .to_dict())
    for vid, vf_row in venue_feats.items():
        vr = train_reviews[train_reviews["business_id"]==vid]
        states = [user_state_map.get(u,"?") for u in vr["user_id"].unique()]
        counts = pd.Series(states).value_counts().values
        p = counts / counts.sum()
        vf_row["geographic_diversity"] = float(-(p*np.log2(p+1e-12)).sum())

    # User features
    user_rev_count = train_reviews.groupby("user_id").size().reset_index(name="n")
    max_n = user_rev_count["n"].max()
    user_credibility = {
        r["user_id"]: np.log1p(r["n"]) / np.log1p(max_n)
        for _, r in user_rev_count.iterrows()
    }
    user_frequency = train_reviews.groupby("user_id")["timestamp"].apply(
        lambda ts: len(ts) / max((ts.max()-ts.min()).days/365+1, 1)
    ).to_dict()

    return venue_feats, user_credibility, user_frequency

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(ranking_dict, train_uv, test_uv, k=10):
    ndcg_scores, hit_scores = [], []
    for uid in test_uv:
        if uid not in train_uv: continue
        cands = list(train_uv[uid])
        if len(cands) < 2: continue
        ranked = sorted(cands, key=lambda v: ranking_dict.get(v,0), reverse=True)
        ndcg_scores.append(ndcg_at_k(ranked, test_uv[uid], k))
        hit_scores.append(hit_at_k(ranked, test_uv[uid], k))
    return np.array(ndcg_scores), np.array(hit_scores)

def evaluate_per_group(ranking_dict, train_uv, test_uv, user_groups, k=10):
    gmap = dict(zip(user_groups["user_id"], user_groups["archetype"]))
    group_scores = defaultdict(list)
    for uid in test_uv:
        if uid not in train_uv: continue
        g = gmap.get(uid, "Unknown")
        cands = list(train_uv[uid])
        if len(cands) < 2: continue
        ranked = sorted(cands, key=lambda v: ranking_dict.get(v,0), reverse=True)
        group_scores[g].append(ndcg_at_k(ranked, test_uv[uid], k))
    return {g: (float(np.mean(v)), len(v)) for g, v in group_scores.items()}

# ============================================================================
# MAIN
# ============================================================================
print("=" * 70)
print("  PHASE 7: HOTEL MODEL VALIDATION")
print("=" * 70)
t_start = time.time()

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n  Loading data...")
interactions = pd.read_csv(DATA_DIR / "hotel_interactions.csv", parse_dates=["timestamp"])
businesses   = pd.read_csv(DATA_DIR / "hotel_businesses.csv")

reviews = interactions[interactions["source"]=="review"].dropna(
    subset=["user_id","business_id"]
).copy()
reviews["timestamp"] = pd.to_datetime(reviews["timestamp"])

# Cross-domain priors
xdomain_file = DATA_DIR / "cross_domain_priors.csv"
if xdomain_file.exists():
    xdomain_priors = pd.read_csv(xdomain_file)
    xdomain_weight_map = dict(zip(
        xdomain_priors["user_id"].astype(str),
        xdomain_priors["prior_weight"]
    ))
else:
    xdomain_weight_map = {}

# FSQ social scores
fsq_venue_scores = {}
fsq_file = DATA_DIR / "hotel_birank_fsq_scores.csv"
if fsq_file.exists():
    fsq_df = pd.read_csv(fsq_file)
    fsq_venue_scores = dict(zip(fsq_df["business_id"], fsq_df["birank_fsq_score"]))

# User groups
user_groups = pd.read_csv(DATA_DIR / "hotel_user_groups.csv")

print(f"    Interactions: {len(interactions):,}")
print(f"    Reviews: {len(reviews):,}")
print(f"    FSQ venue scores: {len(fsq_venue_scores):,}")
print(f"    Cross-domain priors: {len(xdomain_weight_map):,}")

# ── Train / test split ─────────────────────────────────────────────────────────
train = reviews[reviews["timestamp"] < SPLIT_DATE].copy()
test  = reviews[reviews["timestamp"] >= SPLIT_DATE].copy()
print(f"\n    Train: {len(train):,}  Test: {len(test):,}")

# Ground truth (vectorized)
train_gb = train.dropna(subset=["user_id"]).groupby("user_id")["business_id"].apply(set).to_dict()
test_gb  = test.dropna(subset=["user_id"]).groupby("user_id")["business_id"].apply(set).to_dict()
overlap  = set(train_gb.keys()) & set(test_gb.keys())
train_uv = {uid: train_gb[uid] for uid in overlap if train_gb[uid]}
test_uv  = {uid: test_gb[uid]  for uid in overlap if test_gb[uid]}
print(f"    Overlap users: {len(overlap):,}  With future visits: {len(test_uv):,}")

if len(test_uv) < 50:
    print("  ⚠ Too few test users — check data.")
    exit()

# ── Features from training data only ──────────────────────────────────────────
print("\n  Computing features from training data...")
t0 = time.time()
venue_feats, user_cred, user_freq = compute_features_from_train(train, businesses)
print(f"    Done in {time.time()-t0:.0f}s  ({len(venue_feats):,} venues)")

# ── Build edge list ────────────────────────────────────────────────────────────
now = train["timestamp"].max()
train["age_years"] = (now - train["timestamp"]).dt.days / 365.0
train["recency_decay"] = np.exp(-LAMBDA_DECAY * train["age_years"])
train["traveler_credibility"] = train["user_id"].map(user_cred).fillna(0.1)
train["edge_weight"] = train["recency_decay"] * train["traveler_credibility"]
edges = train.groupby(["user_id","business_id"])["edge_weight"].sum().reset_index()

users  = edges["user_id"].unique()
venues = edges["business_id"].unique()
u2i = {u: i for i, u in enumerate(users)}
v2i = {v: i for i, v in enumerate(venues)}
n_u, n_v = len(users), len(venues)

def make_v_prior(feats, fsq_scores=None, social_gamma=0.0):
    vp = np.ones(n_v)
    for vid, idx in v2i.items():
        f = feats.get(vid, {})
        beh = (
            0.4 * f.get("review_velocity", 0) +
            0.3 * f.get("geographic_diversity", 0) +
            0.2 * f.get("multi_stay_rate", 0) * 50 +
            0.1 * (1.0 - f.get("seasonal_cv", 0.5))
        )
        soc = fsq_scores.get(vid, 0.0) if fsq_scores else 0.0
        vp[idx] = max((1-social_gamma)*beh + social_gamma*soc, 0.01)
    return vp / vp.sum()

def make_u_prior(xdomain=False):
    up = np.ones(n_u)
    for uid, idx in u2i.items():
        base = max(
            0.6 * user_freq.get(uid, 1.0) +
            0.4 * 0.5, 0.01
        )
        xd = xdomain_weight_map.get(str(uid), 1.0) if xdomain else 1.0
        up[idx] = base * xd
    return up / up.sum()

# ── Build all rankings ─────────────────────────────────────────────────────────
print("\n  Building rankings...")
rankings = {}

t0 = time.time()
vp_base = make_v_prior(venue_feats)
up_base = make_u_prior(xdomain=False)
rankings["hotel_birank"] = run_birank(edges, vp_base, up_base, n_u, n_v, u2i, v2i)
print(f"    hotel_birank:         {time.time()-t0:.1f}s")

if fsq_venue_scores:
    t0 = time.time()
    vp_fsq = make_v_prior(venue_feats, fsq_venue_scores, SOCIAL_GAMMA)
    rankings["hotel_birank_fsq"] = run_birank(edges, vp_fsq, up_base, n_u, n_v, u2i, v2i)
    print(f"    hotel_birank_fsq:     {time.time()-t0:.1f}s")

if xdomain_weight_map:
    t0 = time.time()
    up_xd = make_u_prior(xdomain=True)
    rankings["hotel_birank_xdomain"] = run_birank(edges, vp_base, up_xd, n_u, n_v, u2i, v2i)
    print(f"    hotel_birank_xdomain: {time.time()-t0:.1f}s")

if fsq_venue_scores and xdomain_weight_map:
    t0 = time.time()
    rankings["hotel_birank_full"] = run_birank(edges, vp_fsq, up_xd, n_u, n_v, u2i, v2i)
    print(f"    hotel_birank_full:    {time.time()-t0:.1f}s")

# Baselines
biz_meta = businesses.set_index("business_id")
rankings["baseline_rating"]     = {vid: float(biz_meta.loc[vid,"stars"]) if vid in biz_meta.index else 0.0
                                     for vid in venues}
rankings["baseline_popularity"] = {vid: float(biz_meta.loc[vid,"review_count"]) if vid in biz_meta.index else 0.0
                                     for vid in venues}
rng = np.random.RandomState(42)
rankings["baseline_random"]     = {vid: float(rng.random()) for vid in venues}

# Item-KNN (cosine)
t0 = time.time()
v_list = list(venues)
vidx   = {v: i for i, v in enumerate(v_list)}
rows_e  = [u2i.get(u, 0) for u in edges["user_id"]]
cols_e  = [vidx.get(v, 0) for v in edges["business_id"]]
data_e  = edges["edge_weight"].values
UV = sparse.csr_matrix((data_e, (rows_e, cols_e)), shape=(n_u, len(v_list)))
item_vecs = UV.T.tocsr()  # items × users
sims = cosine_similarity(item_vecs, dense_output=False)
# For each user, score = mean sim of their training venues × candidate
knn_ranking = {}
for uid in test_uv:
    if uid not in u2i: continue
    ui = u2i[uid]
    user_items = set(v for v in train_uv.get(uid, []))
    if not user_items: continue
    scores = np.zeros(len(v_list))
    for uv in user_items:
        if uv in vidx:
            scores += np.asarray(sims[vidx[uv]].todense()).flatten()
    knn_ranking[uid] = {v_list[i]: scores[i] for i in range(len(v_list))}

def evaluate_knn(knn_rank, train_uv, test_uv, k=10):
    ndcg_s, hit_s = [], []
    for uid in test_uv:
        if uid not in knn_rank or uid not in train_uv: continue
        cands = list(train_uv[uid])
        ranked = sorted(cands, key=lambda v: knn_rank[uid].get(v,0), reverse=True)
        ndcg_s.append(ndcg_at_k(ranked, test_uv[uid], k))
        hit_s.append(hit_at_k(ranked, test_uv[uid], k))
    return np.array(ndcg_s), np.array(hit_s)

print(f"    baseline_item_knn:    {time.time()-t0:.1f}s")

# ── Evaluate all methods ───────────────────────────────────────────────────────
print("\n  Evaluating...")
results = {}
per_user_scores = {}

for name, ranking in rankings.items():
    ndcg, hit = evaluate(ranking, train_uv, test_uv, k=10)
    results[name] = {"NDCG@10": ndcg.mean(), "Hit@10": hit.mean(), "n": len(ndcg)}
    per_user_scores[name] = ndcg

knn_ndcg, knn_hit = evaluate_knn(knn_ranking, train_uv, test_uv, k=10)
results["baseline_item_knn"] = {"NDCG@10": knn_ndcg.mean(), "Hit@10": knn_hit.mean(), "n": len(knn_ndcg)}
per_user_scores["baseline_item_knn"] = knn_ndcg

# Compute CIs and p-values (reference = hotel_birank)
ref_name = "hotel_birank"
ref_scores = per_user_scores[ref_name]
for name in results:
    ci_lo, ci_hi = bootstrap_ci(per_user_scores[name])
    results[name]["CI_lo"] = ci_lo
    results[name]["CI_hi"] = ci_hi
    if name == ref_name:
        results[name]["p"] = None
    else:
        s1, s2 = per_user_scores[name], ref_scores
        min_len = min(len(s1), len(s2))
        results[name]["p"] = wilcoxon_p(s1[:min_len], s2[:min_len])

# ── Per-group evaluation ───────────────────────────────────────────────────────
print("\n  Per-group evaluation...")
group_results = {}
for name, ranking in rankings.items():
    group_results[name] = evaluate_per_group(ranking, train_uv, test_uv, user_groups)

# ── Print results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  RESULTS (test split >= 2020-01-01)")
print("=" * 70)

sorted_methods = sorted(results.items(), key=lambda x: -x[1]["NDCG@10"])
ref_ndcg = results[ref_name]["NDCG@10"]

print(f"\n  {'Method':<30} {'NDCG@10':>8} {'Hit@10':>7} {'95% CI':>20}  {'p-value':>10}")
print("  " + "─" * 78)
for name, res in sorted_methods:
    ci = f"[{res['CI_lo']:.4f}, {res['CI_hi']:.4f}]"
    p_str = "ref" if res["p"] is None else f"{res['p']:.3e}"
    star = " ★" if name == ref_name else ""
    print(f"  {name:<30} {res['NDCG@10']:>8.4f} {res['Hit@10']:>7.4f} {ci:>20}  {p_str:>10}{star}")

print(f"\n  Improvement over hotel_birank (NDCG@10 = {ref_ndcg:.4f}):")
print("  " + "─" * 60)
for name, res in sorted_methods:
    delta = res["NDCG@10"] - ref_ndcg
    pct   = 100 * delta / ref_ndcg
    p_str = "ref" if res["p"] is None else f"{res['p']:.3e}"
    print(f"  {name:<30} {res['NDCG@10']:>8.4f}  {delta:>+.4f}  {pct:>+.1f}%  {p_str}")

if group_results:
    print(f"\n  Per-group NDCG@10:")
    print("  " + "─" * 80)
    all_groups = sorted(set(g for gd in group_results.values() for g in gd.keys()))
    header = f"  {'Method':<30}" + "".join(f"  {g[:20]:>22}" for g in all_groups)
    print(header)
    print("  " + "─" * 80)
    for name in [ref_name] + [n for n in rankings if n != ref_name]:
        gd = group_results.get(name, {})
        row = f"  {name:<30}"
        for g in all_groups:
            if g in gd:
                ndcg_g, n_g = gd[g]
                row += f"  {ndcg_g:.4f} (n={n_g:,})"
            else:
                row += f"  {'—':>22}"
        print(row)

# ── Save ───────────────────────────────────────────────────────────────────────
rows = []
for name, res in results.items():
    rows.append({
        "method": name,
        "NDCG@10": res["NDCG@10"],
        "Hit@10":  res["Hit@10"],
        "CI_lo":   res["CI_lo"],
        "CI_hi":   res["CI_hi"],
        "p":       res["p"],
        "n_users": res["n"],
        "delta":   res["NDCG@10"] - ref_ndcg,
    })
results_df = pd.DataFrame(rows).sort_values("NDCG@10", ascending=False)
results_df.to_csv(DATA_DIR / "hotel_validation_results.csv", index=False)
print(f"\n  Saved: hotel_validation_results.csv")

# Per-group CSV
pg_rows = []
for name, gd in group_results.items():
    for g, (ndcg_g, n_g) in gd.items():
        pg_rows.append({"method": name, "group": g, "NDCG@10": ndcg_g, "n_users": n_g})
pd.DataFrame(pg_rows).to_csv(DATA_DIR / "hotel_validation_per_group.csv", index=False)
print(f"  Saved: hotel_validation_per_group.csv")

# Summary text
with open(DATA_DIR / "hotel_validation_summary.txt", "w") as f:
    f.write("HOTEL MODEL VALIDATION SUMMARY\n")
    f.write(f"Split: 2020-01-01  |  Train: {len(train):,}  |  Test: {len(test):,}  |  "
            f"Overlap: {len(test_uv):,} users\n\n")
    f.write(f"BEST METHOD: {sorted_methods[0][0]}\n")
    f.write(f"  NDCG@10 = {sorted_methods[0][1]['NDCG@10']:.6f}  "
            f"95% CI [{sorted_methods[0][1]['CI_lo']:.4f}, {sorted_methods[0][1]['CI_hi']:.4f}]\n\n")
    f.write("ALL RESULTS:\n")
    for name, res in sorted_methods:
        p_str = "ref" if res["p"] is None else f"{res['p']:.3e}"
        f.write(f"  {name:<30} NDCG@10={res['NDCG@10']:.4f}  p={p_str}\n")
print(f"  Saved: hotel_validation_summary.txt")
print(f"\n  Total wall time: {(time.time()-t_start)/60:.1f} minutes\n")
