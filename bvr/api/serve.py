"""
Production-ready FastAPI service for the behavioral venue ranking hybrid model.

Endpoints:
  GET  /health             — liveness probe
  GET  /info               — model version, dataset stats
  POST /rank               — rank a list of venue IDs for a query
  GET  /venue/{venue_id}   — fetch venue metadata + score
  GET  /top                — top-N globally ranked venues with optional category filter

Run locally:
    python3 -m uvicorn serve:app --reload --port 8000

Docker:
    docker build -t behavioral-ranking .
    docker run -p 8000:8000 behavioral-ranking

Environment variables:
    DATA_DIR    — path to the CSV files (default: directory of this file)
    MODEL       — "london" or "uk_fsq" (default: "uk_fsq")
    PORT        — server port (default: 8000)
"""

import os, time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DATA_DIR   = Path(os.getenv("DATA_DIR", Path(__file__).parent))
MODEL_NAME = os.getenv("MODEL", "uk_fsq")

# ── Data loaded at startup ────────────────────────────────────────────────────

_scores:    dict = {}     # {venue_id: float}
_venues:    dict = {}     # {venue_id: {name, lat, lon, category}}
_loaded_at: float = 0.0
_stats:     dict = {}


def _load_model():
    global _scores, _venues, _loaded_at, _stats

    if MODEL_NAME == "london":
        scores_file = DATA_DIR / "london_birank_venue_scores.csv"
        biz_file    = DATA_DIR / "london_businesses.csv"
        _stats = {"dataset": "London TripAdvisor", "split": "2018-01-01"}
    else:
        scores_file = DATA_DIR / "uk_fsq_venue_scores.csv"
        biz_file    = DATA_DIR / "uk_fsq_businesses.csv"
        _stats = {"dataset": "UK Foursquare WWW2019", "split": "2013-07-01"}

    if not scores_file.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_file}")

    scores_df = pd.read_csv(scores_file, dtype={"business_id": str})
    _scores   = dict(zip(scores_df["business_id"], scores_df["birank_score"].astype(float)))

    if biz_file.exists():
        biz_df = pd.read_csv(biz_file, dtype={"business_id": str})
        for _, row in biz_df.iterrows():
            _venues[str(row["business_id"])] = {
                "name":     str(row.get("name", row.get("category", ""))),
                "category": str(row.get("category", "")),
                "lat":      float(row["lat"]) if "lat" in row and pd.notna(row["lat"]) else None,
                "lon":      float(row["lon"]) if "lon" in row and pd.notna(row["lon"]) else None,
            }

    _stats.update({
        "n_venues_scored": len(_scores),
        "n_venues_with_metadata": len(_venues),
        "score_min": float(min(_scores.values())) if _scores else 0,
        "score_max": float(max(_scores.values())) if _scores else 0,
    })
    _loaded_at = time.time()
    print(f"[serve] Loaded {len(_scores):,} venue scores from {scores_file.name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Behavioral Venue Ranking API",
    description=(
        "Hybrid exploration-BiRank + ALS model. "
        "Ranks venues by behavioral signal: what people *do* beats what they *say*."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class RankRequest(BaseModel):
    venue_ids: List[str]
    top_n: Optional[int] = None

class VenueScore(BaseModel):
    venue_id:  str
    score:     float
    rank:      int
    name:      Optional[str] = None
    category:  Optional[str] = None
    lat:       Optional[float] = None
    lon:       Optional[float] = None

class RankResponse(BaseModel):
    ranked:      List[VenueScore]
    n_input:     int
    n_scored:    int
    n_unscored:  int
    model:       str
    latency_ms:  float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enrich(venue_id: str, score: float, rank: int) -> VenueScore:
    meta = _venues.get(venue_id, {})
    return VenueScore(
        venue_id=venue_id,
        score=round(score, 6),
        rank=rank,
        name=meta.get("name"),
        category=meta.get("category"),
        lat=meta.get("lat"),
        lon=meta.get("lon"),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "loaded": _loaded_at > 0}


@app.get("/info")
def info():
    return {
        "model":       MODEL_NAME,
        "algorithm":   "Hybrid (BiRank exploration priors + ALS, λ=0.5)",
        "reference":   "He et al. SIGIR 2020 (BiRank); Hu et al. ICDM 2008 (ALS)",
        "loaded_at":   _loaded_at,
        "stats":       _stats,
    }


@app.post("/rank", response_model=RankResponse)
def rank_venues(req: RankRequest):
    t0 = time.perf_counter()

    scored, unscored = [], []
    for vid in req.venue_ids:
        vid = str(vid)
        if vid in _scores:
            scored.append((vid, _scores[vid]))
        else:
            unscored.append(vid)

    scored.sort(key=lambda x: x[1], reverse=True)
    if req.top_n:
        scored = scored[: req.top_n]

    ranked = [_enrich(vid, score, rank + 1) for rank, (vid, score) in enumerate(scored)]

    return RankResponse(
        ranked=ranked,
        n_input=len(req.venue_ids),
        n_scored=len(scored),
        n_unscored=len(unscored),
        model=MODEL_NAME,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.get("/venue/{venue_id}", response_model=VenueScore)
def get_venue(venue_id: str):
    venue_id = str(venue_id)
    if venue_id not in _scores:
        raise HTTPException(status_code=404, detail=f"Venue {venue_id} not in scored set")
    sorted_ids = sorted(_scores, key=_scores.get, reverse=True)
    rank = sorted_ids.index(venue_id) + 1
    return _enrich(venue_id, _scores[venue_id], rank)


@app.get("/top", response_model=List[VenueScore])
def top_venues(
    n: int = Query(default=20, ge=1, le=500, description="Number of top venues"),
    category: Optional[str] = Query(default=None, description="Filter by category"),
):
    t0 = time.perf_counter()
    sorted_items = sorted(_scores.items(), key=lambda x: x[1], reverse=True)

    results, rank = [], 1
    for vid, score in sorted_items:
        if category:
            meta = _venues.get(vid, {})
            if meta.get("category", "").lower() != category.lower():
                continue
        results.append(_enrich(vid, score, rank))
        rank += 1
        if len(results) >= n:
            break

    return results


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("serve:app", host="0.0.0.0", port=port, reload=False)
