"""Async OpenAI caller with SQLite caching, rate limiting, and retry logic.

v2 (Phase 5): accepts model parameter → enables tiered model use.
  - gpt-5.4-mini for ranking tasks (cheap, 1500 calls)
  - gpt-5.4    for pairwise + revisit (full quality, 3000 calls)
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import sqlite3
import time
import logging
from typing import Any, Dict, Optional

import openai
from config import MODEL, OPENAI_API_KEY, CACHE_PATH, MAX_CONCURRENT

logger = logging.getLogger(__name__)

MINI_MODEL = "gpt-5.4-mini"   # Phase 5: cheaper model for high-volume ranking task
FULL_MODEL = MODEL             # gpt-5.4 for pairwise and revisit

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_clients: Dict[str, openai.AsyncOpenAI] = {}


def get_client(api_key: Optional[str] = None) -> openai.AsyncOpenAI:
    key = api_key or OPENAI_API_KEY
    if key not in _clients:
        _clients[key] = openai.AsyncOpenAI(api_key=key)
    return _clients[key]


# ── Cache ─────────────────────────────────────────────────────────────────────

def _init_cache():
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache "
        "(key TEXT PRIMARY KEY, response TEXT, created_at REAL)"
    )
    conn.commit()
    conn.close()


def _cache_key(model: str, system: str, user: str) -> str:
    return hashlib.sha256(f"{model}|{system}|{user}".encode()).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    conn = sqlite3.connect(CACHE_PATH)
    row = conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def _cache_put(key: str, response: str):
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, response, created_at) VALUES (?,?,?)",
        (key, response, time.time())
    )
    conn.commit()
    conn.close()


_init_cache()


# ── Core call ─────────────────────────────────────────────────────────────────

async def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.8,
    retries: int = 4,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Call OpenAI with caching and exponential backoff. Returns parsed JSON."""
    _model = model or FULL_MODEL
    key = _cache_key(_model, system_prompt, user_prompt)
    cached = _cache_get(key)
    if cached:
        return json.loads(cached)

    async with _semaphore:
        for attempt in range(retries):
            try:
                response = await get_client().chat.completions.create(
                    model=_model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = response.choices[0].message.content
                parsed = json.loads(text)
                _cache_put(key, json.dumps(parsed))
                return parsed

            except openai.RateLimitError:
                wait = 2 ** attempt + 1
                logger.warning("Rate limit (%s), waiting %ds", _model, wait)
                await asyncio.sleep(wait)

            except openai.APIError as e:
                if attempt == retries - 1:
                    logger.error("API error after %d retries: %s", retries, e)
                    return {"error": str(e)}
                await asyncio.sleep(2 ** attempt)

            except json.JSONDecodeError as e:
                logger.warning("JSON parse error attempt %d: %s", attempt, e)
                if attempt == retries - 1:
                    return {"error": "json_parse_failed"}
                await asyncio.sleep(1)

    return {"error": "max_retries_exceeded"}


async def call_llm_mini(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.8,
) -> Dict[str, Any]:
    """Convenience wrapper: use gpt-5.4-mini for high-volume, lower-stakes tasks."""
    return await call_llm(system_prompt, user_prompt, temperature, model=MINI_MODEL)


# ── Batch runner ──────────────────────────────────────────────────────────────

async def run_batch(
    tasks: list,
    progress_callback=None,
) -> list:
    """Run a list of (system, user, temperature[, model]) tuples concurrently."""
    results = []
    coros = []
    for t in tasks:
        sys_p, usr_p, temp = t[0], t[1], t[2]
        mdl = t[3] if len(t) > 3 else None
        coros.append(call_llm(sys_p, usr_p, temp, model=mdl))

    for i, coro in enumerate(asyncio.as_completed(coros)):
        result = await coro
        results.append(result)
        if progress_callback:
            progress_callback(i + 1, len(tasks))

    return results
