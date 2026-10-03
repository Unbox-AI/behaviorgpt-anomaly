"""The local catalog snapshot written by scripts/snapshot_catalog.py."""

import gzip
import json
import random
import re
from functools import cached_property
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CATALOG_PATH = DATA_DIR / "catalog.json.gz"
CACHE_PATH = DATA_DIR / "cache.sqlite"
FIELDS = ("name", "brand", "categories", "image_url", "price", "currency", "frequency")


def parse_price(value) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


class Catalog:
    """The snapshot's items are the store: the fixed set probabilities are
    normalized over. Items the API returns that the snapshot lacks are kept
    for display but do not join the store."""

    def __init__(self, catalog_id: str, items: list[dict]):
        self.catalog_id = catalog_id
        self.items = {it["id"]: it for it in items}
        self.ids = sorted(self.items)
        self.store = frozenset(self.ids)

    @classmethod
    def load(cls, path: Path = CATALOG_PATH) -> "Catalog":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run `uv run python scripts/snapshot_catalog.py`"
            )
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["catalog_id"], data["items"])

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self.store

    def get(self, item_id: str) -> dict:
        return self.items.get(item_id) or {"id": item_id, "name": item_id}

    def update(self, item_id: str, data: dict) -> None:
        """Remember an item the API returned that the snapshot lacks."""
        if item_id not in self.items:
            self.items[item_id] = {f: data.get(f) for f in FIELDS} | {"id": item_id}

    def price(self, item_id: str) -> float | None:
        return parse_price(self.get(item_id).get("price"))

    def category(self, item_id: str) -> str:
        return self.get(item_id).get("categories") or "Uncategorized"

    def reference_sample(self, size: int, seed: int = 0) -> list[str]:
        """A fixed uniform sample of ids, used to estimate the catalog's tail."""
        return random.Random(seed).sample(self.ids, min(size, len(self.ids)))

    def sample(self, rng: random.Random, *, where=None) -> str:
        pool = (
            self.ids if where is None else [i for i in self.ids if where(self.items[i])]
        )
        return rng.choice(pool)

    @cached_property
    def _search_index(self) -> list[tuple[str, str]]:
        return [
            (
                i,
                " ".join(
                    str(it.get(f) or "") for f in ("name", "brand", "categories")
                ).lower(),
            )
            for i, it in self.items.items()
        ]

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Plain substring search over names, for the sequence builder."""
        terms = [t for t in re.split(r"\W+", query.lower()) if t]
        if not terms:
            return []
        hits = [
            (sum(text.count(t) for t in terms), i)
            for i, text in self._search_index
            if all(t in text for t in terms)
        ]
        hits.sort(key=lambda h: (-h[0], -(self.items[h[1]].get("frequency") or 0)))
        return [self.items[i] for _, i in hits[:limit]]
