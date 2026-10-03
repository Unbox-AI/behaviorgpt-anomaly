from anomaly.api import Ranked
from anomaly.calibration import Calibration
from anomaly.catalog import Catalog
from anomaly.detector import Detector, plan
from anomaly.events import Event

ITEMS = [
    {
        "id": f"i{n}",
        "name": f"Item {n}",
        "categories": "Toys" if n < 50 else "Tools",
        "price": str(10 + n),
    }
    for n in range(200)
]


class FakeAPI:
    """Ranks items by closeness to the last item touched: i0 likes i1, i2..."""

    def rank(self, history, *, limit=100, offset=0, filters=None):
        last = next((int(e.item[1:]) for e in reversed(history) if e.item), 0)
        scored = [
            Ranked(it["id"], 2.0 - abs(int(it["id"][1:]) - last) / 100, it)
            for it in ITEMS
        ]
        if filters:
            scored = [r for r in scored if r.id in set(filters["id"])]
        scored.sort(key=lambda r: -r.score)
        return scored[offset : offset + limit]

    def rank_many(self, requests):
        return [self.rank(**r) for r in requests]


def detector(**cal) -> Detector:
    return Detector(FakeAPI(), Catalog("test", ITEMS), Calibration(**cal))


def test_plan_scores_first_touch_only():
    events = [
        Event("search", query="toys"),
        Event("view", item="i1"),
        Event("add_to_cart", item="i1"),
        Event("add_to_cart", item="i2"),
        Event("remove_from_cart", item="i2"),
        Event("order"),
    ]
    assert plan(events) == ["context", "scored", "funnel", "scored", "funnel", "order"]


def test_coherent_session_scores_lower_than_jumps():
    det = detector(tau={"browse": 0.05, "search": 0.05}, barrier=3.0)
    smooth = [Event("view", item=f"i{n}") for n in range(1, 8)]
    jumpy = [Event("view", item=f"i{n}") for n in (1, 190, 3, 170, 60, 199, 5)]
    a, b = det.score(smooth), det.score(jumpy)
    assert a["summary"]["z"] < b["summary"]["z"]
    assert not a["summary"]["flagged"]
    assert b["summary"]["flagged"]
    assert b["contributors"][0]["index"] in (1, 3, 5)


def test_order_after_flag_is_held():
    det = detector(tau={"browse": 0.05, "search": 0.05}, barrier=1.0)
    events = (
        [Event("view", item="i1")]
        + [Event("add_to_cart", item=f"i{n}") for n in (199, 120, 180)]
        + [Event("order")]
    )
    report = det.score(events)
    assert report["summary"]["flagged"]
    assert "hold it for review" in report["steps"][-1]["reasons"][0]
