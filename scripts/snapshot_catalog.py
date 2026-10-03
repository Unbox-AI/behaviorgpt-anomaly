"""Save a local copy of the catalog's items: ids, names, images, prices.

The API has no "list the catalog" call, and any ranking pages at most 10 000
results deep. So this pages through several rankings that order the catalog
differently and keeps the union. `retail_catalog` is larger than that union:
each ranking still adds ~9 000 new items. The snapshot is therefore the
app's store, the set of items probabilities are normalized over (see
anomaly/scoring.py), and the item picker's search index.

Runs add to an existing snapshot, so new orderings can be added later.

    uv run python scripts/snapshot_catalog.py
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly.api import DEFAULT_CATALOG, BehaviorAPI  # noqa: E402
from anomaly.catalog import CACHE_PATH, CATALOG_PATH, FIELDS  # noqa: E402
from anomaly.events import Event  # noqa: E402

MAX_DEPTH = 10_000
PAGE = 100

# Histories that pull different corners of a retail catalog to the top.
ORDERINGS = [
    [],
    [Event("search", query="books novels")],
    [Event("search", query="kitchen cookware")],
    [Event("search", query="phone case charger")],
    [Event("search", query="toys games for kids")],
    [Event("search", query="power tools garden")],
    [Event("search", query="clothing shoes fashion")],
    [Event("search", query="beauty skincare makeup")],
    [Event("search", query="sports outdoors fitness")],
    [Event("search", query="laptop headphones electronics")],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=CATALOG_PATH)
    args = parser.parse_args()

    load_dotenv()
    api = BehaviorAPI(catalog_id=args.catalog, workers=12, cache_path=CACHE_PATH)
    items: dict[str, dict] = {}
    if args.out.exists():
        with gzip.open(args.out, "rt", encoding="utf-8") as f:
            items = {it["id"]: it for it in json.load(f)["items"]}
    done = set()
    if args.out.with_suffix(".orderings").exists():
        done = set(args.out.with_suffix(".orderings").read_text().splitlines())
    for history in ORDERINGS:
        label = history[0].query if history else "(popularity)"
        if label in done:
            continue
        pages = api.rank_many(
            {"history": history, "limit": PAGE, "offset": off}
            for off in range(0, MAX_DEPTH, PAGE)
        )
        before = len(items)
        for page in pages:
            for r in page:
                items.setdefault(
                    r.id, {f: r.data.get(f) for f in FIELDS} | {"id": r.id}
                )
        print(f"{label:>30}: +{len(items) - before:5d} new, {len(items)} total")
        done.add(label)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.out, "wt", encoding="utf-8") as f:
            json.dump({"catalog_id": args.catalog, "items": list(items.values())}, f)
        args.out.with_suffix(".orderings").write_text("\n".join(sorted(done)))
    print(f"{len(items)} items in {args.out}")


if __name__ == "__main__":
    main()
