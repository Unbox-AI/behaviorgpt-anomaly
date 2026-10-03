"""BehaviorGPT calls with retries, a disk cache and a thread pool.

Scoring one sequence takes two ranking calls per scored event, and the same
prefixes come back again and again as a sequence is edited in the app, so
every ranking is cached on disk by (catalog, history, page, filters).
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
from behaviorgpt import UnboxAIClient
from behaviorgpt._exceptions import UnboxAIError

from anomaly.catalog import FIELDS
from anomaly.events import Event, key

DEFAULT_CATALOG = "retail_catalog"
RETRYABLE = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Ranked:
    id: str
    score: float
    data: dict


class RankCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS ranks (k TEXT PRIMARY KEY, v TEXT)"
        )
        self._lock = threading.Lock()

    def get(self, k: str) -> list[Ranked] | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM ranks WHERE k = ?", (k,)).fetchone()
        if row is None:
            return None
        return [Ranked(**r) for r in json.loads(row[0])]

    def put(self, k: str, items: list[Ranked]) -> None:
        v = json.dumps([r.__dict__ for r in items])
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO ranks VALUES (?, ?)", (k, v))
            self._db.commit()


class BehaviorAPI:
    def __init__(
        self,
        *,
        catalog_id: str = DEFAULT_CATALOG,
        market: str = "us",
        api_key: str | None = None,
        cache_path: Path | None = None,
        workers: int = 8,
        retries: int = 6,
    ):
        self.catalog_id = catalog_id
        self.client = UnboxAIClient(
            market=market,
            api_key=api_key or os.environ.get("UNBOXAI_API_KEY"),
            default_catalog_id=catalog_id,
        )
        self.cache = RankCache(cache_path) if cache_path else None
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.retries = retries

    def rank(
        self,
        history: list[Event],
        *,
        limit: int = 100,
        offset: int = 0,
        filters: dict | None = None,
    ) -> list[Ranked]:
        """The items most likely to come next after `history`, best first."""
        ck = self._cache_key(history, limit, offset, filters)
        if self.cache and (hit := self.cache.get(ck)) is not None:
            return hit
        items = self._call(history, limit, offset, filters)
        if self.cache:
            self.cache.put(ck, items)
        return items

    def rank_many(self, requests: Iterable[dict]) -> list[list[Ranked]]:
        """`rank` for many requests at once, in parallel."""
        return list(self.pool.map(lambda r: self.rank(**r), requests))

    def _call(self, history, limit, offset, filters) -> list[Ranked]:
        sdk_history = [e.to_sdk() for e in history]
        for attempt in range(self.retries + 1):
            try:
                res = self.client.complete(
                    sdk_history, limit=limit, offset=offset, filters=filters
                )
                # only what the app shows: the full payload makes the cache huge
                return [
                    Ranked(i.id, float(i.score), {f: i.data.get(f) for f in FIELDS})
                    for i in res.items
                ]
            except UnboxAIError as exc:
                if exc.status_code not in RETRYABLE or attempt == self.retries:
                    raise
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    def _cache_key(self, history, limit, offset, filters) -> str:
        raw = json.dumps(
            [self.catalog_id, key(history), limit, offset, filters or {}],
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()
