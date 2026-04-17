#!/usr/bin/env python3
"""
Step 5 — Temporal Validation & Ablation (Car-Centric Model)
======================================================
Validates the restaurant behavioral model using a chronological train/test split.
Evaluates: NDCG@k, Hit@k, TopK Precision.
Runs an ablation study to quantify the impact of each S(R,U,C) component 
(specifically Parking and Drive ETA).

Outputs:
  restaurant_validation_results.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import math
from scipy import sparse

# ── Paths ──
BASE = Path(__file__).resolve().parent
USR_FILE = BASE / "restaurant_user_profiles.csv"
VEN_FILE = BASE / "restaurant_venue_features.csv"
INT_FILE = BASE / "restaurant_interactions.csv"
BIZ_FILE = BASE / "restaurant_businesses.csv"
OUT_FILE = BASE / "restaurant_validation_results.csv"

# Configuration
TEST_START_DATE = "2020-01-01"
EVAL_K = [5, 10, 20]

# Calibrated full-model parameters (kept aligned with scoring script defaults).
FULL_W_BEH = 0.55
FULL_W_MOB = 0.45
FULL_W_CTX = 0.00
CRITIC_LAMBDA = 0.20
BUSYNESS_TOLERANCE = 60.0
QUEUE_MIN_FACTOR = 0.75
RATING_FIT_SCALE = 5.0

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    dcg = dcg_at_k(r, k)
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg / idcg

def hit_at_k(r, k):
    return 1 if np.sum(r[:k]) > 0 else 0

def precision_at_k(r, k):
    return np.mean(r[:k])

EARTH_R_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def compute_context_score(peak_busyness, venue_rating, user_avg_rating):
    busyness = float(peak_busyness) if pd.notna(peak_busyness) else 0.0
    if busyness <= BUSYNESS_TOLERANCE:
        queue_factor = 1.0
    else:
        queue_factor = max(QUEUE_MIN_FACTOR, 1.0 - (busyness - BUSYNESS_TOLERANCE) / 200.0)

    rating_fit = 1.0 - min(1.0, abs(float(venue_rating) - float(user_avg_rating)) / RATING_FIT_SCALE)
    return max(0.0, min(1.0, queue_factor * rating_fit))

def simulate_ranking(users, venues, biz_dict, test_inter, kind="Full Model"):
    """
    Given param settings, rank candidates and evaluate against future interactions.
    Simulating the ranking logic directly in validation to allow easy ablation.
    """
    print(f"  ▹ Evaluating: {kind} ...")
    
    metrics = {f"NDCG@{k}": [] for k in EVAL_K}
    metrics.update({f"Hit@{k}": [] for k in EVAL_K})
    
    # Pre-select frequent candidates
    top_venues = venues.sort_values("popularity", ascending=False).head(2000)["business_id"].tolist()
    
    valid_users = 0
    
    for uid in test_inter.index:
        if uid not in users.index:
            continue
            
        future_venues = set(test_inter.loc[uid])
        if not future_venues:
            continue
            
        u = users.loc[uid]
        u_lat = u.centroid_lat
        u_lon = u.centroid_lon
        u_avg_rating = u.avg_visited_rating
        u_spatial_range = max(2.0, u.spatial_range_km)
        u_critic = u.critic_tag
        u_parking_sens = u.parking_sensitivity
        u_archetype = u.archetype
        
        candidates = []
        
        use_c_mob = kind not in ["No Mobility (C_mob)"]
        use_r_ctx = kind not in ["No Context (R_ctx)"]
        use_critic = kind not in ["No Critic Penalty"]
        
        for vid in top_venues:
            v_lat, v_lon = biz_dict[vid]["lat"], biz_dict[vid]["lon"]
            
            # c_mob (Car-Centric Model)
            if use_c_mob:
                dist_km = haversine_km(u_lat, u_lon, v_lat, v_lon)
                try:
                    c_dist = 1.0 - (1.0 / (1.0 + math.exp(u_spatial_range / 2.0 - dist_km)))
                except OverflowError:
                    # If u_spatial_range is huge, we get OverflowError and c_dist approaches 1.0
                    c_dist = 1.0
                c_dist = max(0.0, min(1.0, c_dist))
                
                parking_convenience = biz_dict[vid]["parking_score"]
                if u_parking_sens > 0.5:
                    parking_convenience = parking_convenience ** 1.5
                    
                is_nightlife = any(kw in str(biz_dict[vid]["categories"]) for kw in ["Bars", "Nightlife", "Lounges", "Pubs"])
                if u_archetype == "Nightlife / Ride-Share Candidate" and is_nightlife:
                    parking_convenience = 1.0
                    drive_eta_mins = (dist_km / 30.0) * 60 + 5.0
                    cost_penalty = max(0.0, 1.0 - (drive_eta_mins * 1.5) / 100.0)
                    c_dist = c_dist * cost_penalty
                    
                transit_convenience = float(biz_dict[vid].get("transit_access_score", 0.0))
                c_mob = min(1.0, c_dist * 0.55 + parking_convenience * 0.30 + transit_convenience * 0.15)
            else:
                c_mob = 1.0
                
            # r_ctx
            v_rating = biz_dict[vid]["stars"]
            if use_r_ctx:
                r_ctx = compute_context_score(biz_dict[vid]["peak_busyness"], v_rating, u_avg_rating)
            else:
                r_ctx = 1.0
                
            # u_beh
            base_beh = (biz_dict[vid]["pop"] / 1000.0) * (1.0 + biz_dict[vid].get("repeat_user_rate", 0.0))
            base_beh = min(1.0, np.log1p(base_beh) / 2.0)
            if use_critic and u_critic == "Critic" and v_rating < u_avg_rating:
                gap = float(u_avg_rating - v_rating)
                critic_pen = max(0.0, 1.0 - CRITIC_LAMBDA * (gap / 2.0))
            else:
                critic_pen = 1.0
            u_beh = base_beh * critic_pen
            
            # Simplified Weights for Validation speed
            if kind == "Random":
                score = np.random.rand()
            elif kind == "Popularity":
                score = base_beh
            elif kind == "Rating":
                score = biz_dict[vid]["stars"]
            else:
                score = FULL_W_BEH * u_beh + FULL_W_MOB * c_mob + FULL_W_CTX * r_ctx
                
            candidates.append({"id": vid, "score": score})
            
        candidates.sort(key=lambda x: x["score"], reverse=True)
        ranked_ids = [c["id"] for c in candidates]
        
        rel = [1 if vid in future_venues else 0 for vid in ranked_ids]
        
        if sum(rel) == 0:
            continue
            
        valid_users += 1
        for k in EVAL_K:
            metrics[f"NDCG@{k}"].append(ndcg_at_k(rel, k))
            metrics[f"Hit@{k}"].append(hit_at_k(rel, k))
            
        if valid_users >= 500:
            break
            
    res = {"Method": kind}
    for k in metrics:
        res[k] = round(np.mean(metrics[k]), 4) if metrics[k] else 0.0
    return res

def main():
    print("=" * 60)
    print("TEMPORAL VALIDATION & ABLATION STUDY (CAR-CENTRIC MODEL)")
    print("=" * 60)

    print(f"▸ Splitting interactions at {TEST_START_DATE} …")
    inter = pd.read_csv(INT_FILE, usecols=["user_id", "business_id", "timestamp", "type"])
    inter = inter[inter["type"] == "review"].copy()
    inter["timestamp"] = pd.to_datetime(inter["timestamp"])
    
    train = inter[inter["timestamp"] < TEST_START_DATE]
    test = inter[inter["timestamp"] >= TEST_START_DATE]
    
    print(f"  Train: {len(train):,} interactions")
    print(f"  Test:  {len(test):,} interactions")
    
    test_inter = test.groupby("user_id")["business_id"].apply(list)
    
    print("▸ Loading entity features …")
    users = pd.read_csv(USR_FILE).set_index("user_id")
    venues = pd.read_csv(VEN_FILE)
    biz = pd.read_csv(BIZ_FILE)
    
    v_dict = venues.set_index("business_id").to_dict("index")
    b_dict = biz.set_index("business_id").to_dict("index")
    
    lookup = {}
    for vid in v_dict.keys():
        if vid in b_dict:
            lookup[vid] = {
                "lat": b_dict[vid]["latitude"],
                "lon": b_dict[vid]["longitude"],
                "stars": b_dict[vid]["stars"],
                "pop": v_dict[vid].get("popularity", 0),
                "repeat_user_rate": v_dict[vid].get("repeat_user_rate", 0),
                "parking_score": v_dict[vid].get("parking_score", 0.5),
                "transit_access_score": v_dict[vid].get("transit_access_score", 0.0),
                "categories": b_dict[vid].get("categories", ""),
                "peak_busyness": v_dict[vid].get("peak_busyness", 0)
            }
            
    print("▸ Running evaluators (Ablation Study) …")
    results = []
    
    methods = [
        "Full S(R,U,C) Model",
        "No Mobility (C_mob)",
        "No Context (R_ctx)",
        "No Critic Penalty",
        "Popularity",
        "Rating",
        "Random"
    ]
    
    for m in methods:
        res = simulate_ranking(users, venues, lookup, test_inter, kind=m)
        results.append(res)

    # EASE personalized baseline (restricted to top-2000 venues, same as simulate_ranking)
    print("  ▹ Evaluating: EASE ...")
    EASE_LAMBDA = 500.0
    top_v = venues.sort_values("popularity", ascending=False).head(2000)["business_id"].tolist()
    v_list = top_v
    v2i_r = {v: i for i, v in enumerate(v_list)}
    train_users_ease = train["user_id"].unique()
    u2i_r = {u: i for i, u in enumerate(train_users_ease)}
    matched_r = train[train["business_id"].isin(v2i_r) & train["user_id"].isin(u2i_r)]
    rows_r = matched_r["user_id"].map(u2i_r).values
    cols_r = matched_r["business_id"].map(v2i_r).values
    UV_r = sparse.csr_matrix(
        (np.ones(len(rows_r)), (rows_r, cols_r)),
        shape=(len(u2i_r), len(v_list))
    )
    G_r = (UV_r.T @ UV_r).toarray().astype(np.float64)
    n_v_r = len(v_list)
    G_r += EASE_LAMBDA * np.eye(n_v_r)
    B_r = np.linalg.inv(G_r)
    diag_r = np.diag(B_r).copy()
    B_r = -(B_r / diag_r)
    np.fill_diagonal(B_r, 0)
    ease_metrics = {f"NDCG@{k}": [] for k in EVAL_K}
    ease_metrics.update({f"Hit@{k}": [] for k in EVAL_K})
    valid_ease = 0
    for uid in test_inter.index:
        if uid not in u2i_r or valid_ease >= 500:
            continue
        future = set(test_inter.loc[uid])
        if not future:
            continue
        user_vec = np.asarray(UV_r[u2i_r[uid]].todense()).flatten()
        scores_e = user_vec @ B_r
        ranked = [v_list[i] for i in np.argsort(-scores_e)]
        rel = [1 if v in future else 0 for v in ranked]
        if sum(rel) == 0:
            continue
        valid_ease += 1
        for k in EVAL_K:
            ease_metrics[f"NDCG@{k}"].append(ndcg_at_k(rel, k))
            ease_metrics[f"Hit@{k}"].append(hit_at_k(rel, k))
    ease_res = {"Method": "EASE"}
    for k in ease_metrics:
        ease_res[k] = round(np.mean(ease_metrics[k]), 4) if ease_metrics[k] else 0.0
    results.append(ease_res)

    df = pd.DataFrame(results)
    print("\n✅ Validation Results:")
    print(df.to_string(index=False))

    df.to_csv(OUT_FILE, index=False)
    print(f"\n  → Results saved to {OUT_FILE.name}")


if __name__ == "__main__":
    main()
