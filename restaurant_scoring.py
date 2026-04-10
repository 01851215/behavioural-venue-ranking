#!/usr/bin/env python3
"""
Step 4 — Multi-Objective Scoring S(R,U,C) (Car-Centric Model)
======================================================
Computes combined behavioral scores for user-restaurant pairs, using:
  - U_beh: Behavioral Utility (BiRank priors + Critic penalty)
  - C_mob: Accessibility (Drive Distance Decay + Parking Penalty + Ride-Share)
  - R_ctx: Contextual (Category match + Queue penalty + Rating fit)

Dynamically weights these components using the Entropy Weight Method (EWM)
and re-ranks the top results using Diversity-Aware Ranking (DiAL/MMR).

Outputs:
  restaurant_scores.csv (top N recommendations per user)
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = Path(__file__).resolve().parent
USR_FILE   = BASE / "restaurant_user_profiles.csv"
VEN_FILE   = BASE / "restaurant_venue_features.csv"
INT_FILE   = BASE / "restaurant_interactions.csv"
OUT_FILE   = BASE / "restaurant_scores.csv"

EARTH_R_KM = 6371.0

# Number of users to sample for this demo (to keep runtime reasonable)
N_USERS_SAMPLE = 2000
TOP_K_CANDIDATES = 100
FINAL_K = 20

# Calibrated weighting and penalties (derived from validation diagnostics)
PRIOR_WEIGHTS = np.array([0.48, 0.42, 0.10], dtype=float)  # [U_beh, C_mob, R_ctx]
ADAPTIVE_BLEND = 0.35  # 0=fixed prior, 1=fully EWM dynamic
CRITIC_LAMBDA = 0.20
BUSYNESS_TOLERANCE = 60.0
QUEUE_MIN_FACTOR = 0.75
RATING_FIT_SCALE = 5.0

def haversine_km(lat1, lon1, lat2, lon2):
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def compute_ewm_weights(scores_matrix):
    """
    Entropy Weight Method (EWM)
    scores_matrix: (N_candidates, M_objectives) positive array in [0, 1]
    Returns weights vector of length M_objectives.
    """
    n, m = scores_matrix.shape
    if n <= 1:
        return np.ones(m) / m
        
    col_sums = scores_matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1e-12
    p = scores_matrix / col_sums
    
    epsilon = 1e-12
    entropy = -np.sum(p * np.log(p + epsilon), axis=0) / np.log(n)
    
    d = 1.0 - entropy
    sum_d = np.sum(d)
    if sum_d == 0:
        return np.ones(m) / m
    return d / sum_d


def blend_weights(dynamic_w):
    """Blend EWM dynamic weights with calibrated prior to avoid noisy domination."""
    dynamic_w = np.asarray(dynamic_w, dtype=float)
    if dynamic_w.shape[0] != 3:
        return PRIOR_WEIGHTS.copy()
    blended = ADAPTIVE_BLEND * dynamic_w + (1.0 - ADAPTIVE_BLEND) * PRIOR_WEIGHTS
    total = blended.sum()
    if total <= 0:
        return PRIOR_WEIGHTS.copy()
    return blended / total


def compute_context_score(peak_busyness, venue_rating, user_avg_rating):
    """
    Context penalty with tolerance:
    - No penalty for moderate busyness.
    - Gentle penalty only above tolerance.
    """
    busyness = float(peak_busyness) if pd.notna(peak_busyness) else 0.0
    if busyness <= BUSYNESS_TOLERANCE:
        queue_factor = 1.0
    else:
        queue_factor = max(QUEUE_MIN_FACTOR, 1.0 - (busyness - BUSYNESS_TOLERANCE) / 200.0)

    rating_fit = 1.0 - min(1.0, abs(float(venue_rating) - float(user_avg_rating)) / RATING_FIT_SCALE)
    return max(0.0, min(1.0, queue_factor * rating_fit))


def compute_critic_penalty(user_tag, user_avg_rating, venue_rating):
    if user_tag == "Critic" and venue_rating < user_avg_rating:
        gap = float(user_avg_rating - venue_rating)
        return max(0.0, 1.0 - CRITIC_LAMBDA * (gap / 2.0))
    return 1.0


def mmr_diversity_rerank(candidates, lambda_factor=0.7, k=10):
    """
    Maximal Marginal Relevance (MMR) re-ranking.
    """
    if len(candidates) == 0:
        return []
        
    reranked = []
    unselected = candidates.copy()
    
    unselected.sort(key=lambda x: x["score"], reverse=True)
    reranked.append(unselected.pop(0))
    
    while len(reranked) < k and unselected:
        best_idx = -1
        best_mmr = -float('inf')
        
        for i, c in enumerate(unselected):
            max_sim = 0.0
            c_cats = set(c["categories"].split(", "))
            
            for s in reranked:
                s_cats = set(s["categories"].split(", "))
                if len(c_cats | s_cats) > 0:
                    cat_sim = len(c_cats & s_cats) / len(c_cats | s_cats)
                else:
                    cat_sim = 0.0
                    
                dist_km = haversine_km(c["lat"], c["lon"], s["lat"], s["lon"])
                spat_sim = max(0.0, 1.0 - (dist_km / 5.0))
                
                sim = 0.7 * cat_sim + 0.3 * spat_sim
                if sim > max_sim:
                    max_sim = sim
            
            mmr_score = lambda_factor * c["score"] - (1 - lambda_factor) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
                
        reranked.append(unselected.pop(best_idx))
        
    return reranked


def main():
    print("=" * 60)
    print("MULTI-OBJECTIVE SCORING (CAR-CENTRIC C_mob)")
    print("=" * 60)

    print("▸ Loading data …")
    users = pd.read_csv(USR_FILE)
    venues = pd.read_csv(VEN_FILE)
    biz = pd.read_csv(BASE / "restaurant_businesses.csv")
    
    venues = venues.merge(biz[["business_id", "latitude", "longitude"]], on="business_id", how="left")
    
    venue_dict = venues.set_index("business_id").to_dict("index")
    
    top_venues_df = venues.sort_values("popularity", ascending=False).head(5000)
    candidate_venues = top_venues_df["business_id"].tolist()
    
    print(f"▸ Sampling {N_USERS_SAMPLE} users for demo …")
    sampled_users = users.sample(n=min(N_USERS_SAMPLE, len(users)), random_state=42)
    
    results = []
    
    print("▸ Computing S(R,U,C) …")
    for idx, u in enumerate(sampled_users.itertuples()):
        u_id = u.user_id
        
        u_lat, u_lon = u.centroid_lat, u.centroid_lon
        u_avg_rating = u.avg_visited_rating
        u_spatial_range = max(2.0, u.spatial_range_km) # min 2km to prevent flatlining
        u_archetype = u.archetype
        u_critic = u.critic_tag
        u_parking_sens = u.parking_sensitivity
        
        candidates_scores = []
        
        for v_id in candidate_venues:
            v = venue_dict[v_id]
            v_lat, v_lon = v["latitude"], v["longitude"]
            
            # ── 1. C_mob (Accessibility & Parking) ──
            dist_km = haversine_km(u_lat, u_lon, v_lat, v_lon)
            
            # Drive ETA Decay (Sigmoid curve calibrated to user's accepted range p_d)
            # 1 - 1/(1 + e^(p_d/2 - dist_j))
            try:
                c_dist = 1.0 - (1.0 / (1.0 + math.exp(u_spatial_range / 2.0 - dist_km)))
            except OverflowError:
                c_dist = 1.0 # if u_spatial_range >> dist_km, math.exp() -> inf, score -> 1.0
                
            # Keep score between 0 and 1
            c_dist = max(0.0, min(1.0, c_dist))
            
            # Parking Penalty
            parking_convenience = v.get("parking_score", 0.5)
            if u_parking_sens > 0.5: # User is highly parking-sensitive
                parking_convenience = parking_convenience ** 1.5 # punish low scores heavily
                
            is_nightlife = any(kw in str(v.get("cuisine_categories", "")) for kw in ["Bars", "Nightlife", "Lounges", "Pubs"])
            
            # Ride-Hailing Swap
            if u_archetype == "Nightlife / Ride-Share Candidate" and is_nightlife:
                # Ride-share assumes no parking hassle!
                parking_convenience = 1.0
                
                # Apply ride-share cost penalty instead
                drive_eta_mins = (dist_km / 30.0) * 60 + 5.0 # assume 30km/h avg speed city
                cost_penalty = max(0.0, 1.0 - (drive_eta_mins * 1.5) / 100.0) # $1.50 per min, max budget $100
                c_dist = c_dist * cost_penalty

            # Include Transitland accessibility so C_mob uses both new datasets.
            transit_convenience = float(v.get("transit_access_score", 0.0))
            c_mob = min(1.0, c_dist * 0.55 + parking_convenience * 0.30 + transit_convenience * 0.15)
            
            # ── 2. R_ctx (Context) ──
            busyness = v.get("peak_busyness", 0.0)
            v_rating = v.get("avg_rating", 3.0)
            r_ctx = compute_context_score(busyness, v_rating, u_avg_rating)
            
            # ── 3. U_beh (Behavioral utility) ──
            base_beh = (v.get("popularity", 0) / 1000.0) * (1.0 + v.get("repeat_user_rate", 0))
            base_beh = min(1.0, np.log1p(base_beh) / 2.0)
            
            critic_pen = compute_critic_penalty(u_critic, u_avg_rating, v_rating)
                
            u_beh = base_beh * critic_pen
            
            candidates_scores.append({
                "business_id": v_id,
                "u_beh": u_beh,
                "c_mob": c_mob,
                "r_ctx": r_ctx,
                "categories": str(v.get("cuisine_categories", "")),
                "lat": v_lat,
                "lon": v_lon
            })
            
        # ── Dynamic Weighting (EWM) ──
        mat = np.array([[c["u_beh"], c["c_mob"], c["r_ctx"]] for c in candidates_scores])
        mat = mat + 1e-6
        weights = blend_weights(compute_ewm_weights(mat))
        
        for i, c in enumerate(candidates_scores):
            # S = w1*U + w2*C + w3*R
            c["score"] = weights[0] * c["u_beh"] + weights[1] * c["c_mob"] + weights[2] * c["r_ctx"]
            
        candidates_scores.sort(key=lambda x: x["score"], reverse=True)
        top_cands = candidates_scores[:TOP_K_CANDIDATES]
        
        # ── Diversity Re-ranking (DiAL) ──
        reranked = mmr_diversity_rerank(top_cands, lambda_factor=0.7, k=FINAL_K)
        
        for rank, c in enumerate(reranked):
            results.append({
                "user_id": u_id,
                "business_id": c["business_id"],
                "rank": rank + 1,
                "score": round(c["score"], 4),
                "u_beh": round(c["u_beh"], 4),
                "c_mob": round(c["c_mob"], 4),
                "r_ctx": round(c["r_ctx"], 4),
                "weight_beh": round(weights[0], 3),
                "weight_mob": round(weights[1], 3),
                "weight_ctx": round(weights[2], 3),
            })
            
        if (idx + 1) % 200 == 0:
            print(f"  … {idx+1:,}/{N_USERS_SAMPLE} users mapped")
            
    scores_df = pd.DataFrame(results)
    scores_df.to_csv(OUT_FILE, index=False)
    
    print(f"\n✅ {OUT_FILE.name} saved — {len(scores_df):,} recommendations for {scores_df['user_id'].nunique():,} users")
    
    avg_w_beh = scores_df["weight_beh"].mean()
    avg_w_mob = scores_df["weight_mob"].mean()
    avg_w_ctx = scores_df["weight_ctx"].mean()
    print(f"\n  Average EWM weights: Utility={avg_w_beh:.2f}, Mobility={avg_w_mob:.2f}, Context={avg_w_ctx:.2f}")

if __name__ == "__main__":
    main()
