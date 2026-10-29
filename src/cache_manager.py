"""LLM response caching: LangChain InMemoryCache or SQLiteCache, registered via set_llm_cache().

* InMemoryCache: lives in process memory, lost on restart.
* SQLiteCache:   stored in data/llm_cache.db, persists across restarts.

LangChain's cache keys include the full prompt, so cached entries contain the financial
figures sent to the model. API keys are never part of a prompt or cache entry.
Streaming calls (.stream()) bypass the LLM cache; the cache applies to .invoke()/chain.run().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_core.globals import get_llm_cache, set_llm_cache

from src.config import CACHE_OPTIONS, LLM_CACHE_PATH

logger = logging.getLogger(__name__)
_state = {"kind": "none"}


def configure_cache(kind: str, db_path: Optional[Path] = None) -> str:
    """Register the selected cache globally. Returns the kind actually active.

    Falls back to the in-memory cache if SQLite cannot be initialised.
    """
    if kind not in CACHE_OPTIONS:
        raise ValueError(f"Unknown cache type '{kind}'.")
    if kind == "none":
        set_llm_cache(None)
    elif kind == "memory":
        set_llm_cache(InMemoryCache())
    else:
        path = Path(db_path or LLM_CACHE_PATH)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            set_llm_cache(SQLiteCache(database_path=str(path)))
        except Exception as exc:  # permission / disk errors
            logger.warning("SQLite cache unavailable (%s); using in-memory cache.", type(exc).__name__)
            set_llm_cache(InMemoryCache())
            kind = "memory"
    _state["kind"] = kind
    return kind


def active_cache_kind() -> str:
    """The cache type currently registered with LangChain (verified, not assumed)."""
    cache = get_llm_cache()
    if cache is None:
        return "none"
    return "sqlite" if isinstance(cache, SQLiteCache) else "memory" if isinstance(cache, InMemoryCache) else "other"


def clear_cache() -> bool:
    """Clear cached responses of the active cache. Returns True if something was cleared."""
    cache = get_llm_cache()
    if cache is None:
        return False
    try:
        cache.clear()
        return True
    except Exception as exc:
        logger.warning("Could not clear cache: %s", type(exc).__name__)
        return False
