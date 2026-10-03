"""Simulated shoppers: normal journeys and four fraud patterns.

Normal shoppers have one or two intents. For each they search, click
results with the usual position bias (the first result gets the most
clicks), sometimes follow a recommendation from the item they are on and go
back to the results, put one of the things they viewed in the cart, and
sometimes check out. Some click a trending item on the way. The results and
recommendations come from the model, as a storefront would show them, but
the choices are made by the persona, not by the model.

Others, after their first search, carry on one action at a time with
`next_action`, as the app's "next action" button does: mostly views of an
item sampled from the model's own distribution, sometimes carting what they
just viewed, checking out, or searching for something new. The model
learned that distribution from real behavior, so a shopper who follows it
is normal by definition; leaving them out would make the barrier treat
everything but top results as suspicious.

The fraud patterns are the ones a store sees:

- account takeover: the owner's normal session, then someone else at the
  keyboard carting popular, expensive, resellable items from categories
  the owner never looked at;
- takeover via search: the same, but each item is found by searching for it
  and carting a top result. Every step then looks like normal shopping, so
  this is the pattern a behavior model alone does not catch; it is kept to
  show that limit;
- card testing: cheap, unrelated items carted straight away and ordered, to
  check that a stolen card works;
- scraper: a bot walking the catalog item by item;
- click fraud: searches whose results are ignored for unrelated clicks.
"""

import random
from dataclasses import dataclass, field

from anomaly.api import BehaviorAPI
from anomaly.calibration import Calibration
from anomaly.catalog import Catalog
from anomaly.detector import mode_of
from anomaly.events import Event
from anomaly.scoring import HEAD
from anomaly.suggest import sample_next

PERSONAS: dict[str, list[str]] = {
    "Runner": ["running shoes", "running socks", "fitness tracker", "water bottle"],
    "Gamer": ["ps5 games", "gaming headset", "controller", "gaming chair"],
    "Home cook": ["air fryer", "knife set", "cutting board", "non stick pan"],
    "Home office": ["laptop stand", "wireless mouse", "monitor", "desk lamp"],
    "New phone": ["iphone case", "screen protector", "wireless charger"],
    "Smart home": ["smart plug", "security camera", "smart speaker", "robot vacuum"],
    "DIY": ["cordless drill", "screwdriver set", "tool box", "tape measure"],
    "Music": ["bluetooth speaker", "headphones", "guitar strings"],
    "Reader": ["fantasy novel", "book light", "kindle"],
    "Gardener": ["garden hose", "pruning shears", "raised garden bed"],
}

# how often `next_action` searches, carts the item just viewed, or checks out
SEARCH_P, CART_P, ORDER_P = 0.15, 0.3, 0.25

KINDS = {
    "normal": "Normal shopper",
    "account_takeover": "Account takeover",
    "search_takeover": "Takeover via search",
    "card_testing": "Card testing",
    "scraper": "Scraper bot",
    "click_fraud": "Click fraud",
}

# what someone with a stolen account searches for: resellable, high value
RESALE_QUERIES = [
    "playstation 5 console",
    "macbook pro",
    "iphone 15 pro max",
    "gaming laptop",
    "apple watch",
    "airpods pro",
    "nintendo switch",
    "dyson vacuum",
]


@dataclass
class Scenario:
    kind: str
    label: str
    events: list[Event]
    # index of the first fraudulent event, None for a normal session
    anomaly_start: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "events": [e.to_dict() for e in self.events],
            "anomaly_start": self.anomaly_start,
            "notes": self.notes,
        }


def open_cart(events: list[Event]) -> set[str]:
    """Items in the cart since the last order."""
    cart: set[str] = set()
    for e in events:
        if e.type == "add_to_cart":
            cart.add(e.item)
        elif e.type == "remove_from_cart":
            cart.discard(e.item)
        elif e.type == "order":
            cart.clear()
    return cart


def position_biased(rng: random.Random, ids: list[str], skip: set[str]) -> str | None:
    pool = [i for i in ids if i not in skip]
    if not pool:
        return None
    weights = [1.0 / (r + 1) ** 0.9 for r in range(len(pool))]
    return rng.choices(pool, weights=weights, k=1)[0]


class Simulator:
    def __init__(self, api: BehaviorAPI, catalog: Catalog, tau: dict | None = None):
        self.api = api
        self.catalog = catalog
        # temperatures for followers' sampling: the last calibration's
        self.tau = tau or Calibration.load().tau
        priced = [(p, i) for i in catalog.ids if (p := catalog.price(i))]
        priced.sort()
        threshold = priced[int(0.9 * len(priced))][0] if priced else 500.0

        def popularity(i: str) -> int:
            return catalog.get(i).get("frequency") or 0

        expensive = [i for p, i in priced if p >= threshold]
        self.resellable = sorted(expensive, key=popularity, reverse=True)[:2000]
        self.cheap = [i for p, i in priced if p <= 15]

    def make(self, kind: str, seed: int, persona: str | None = None) -> Scenario:
        rng = random.Random(f"{kind}:{seed}")
        persona = persona or rng.choice(sorted(PERSONAS))
        match kind:
            case "normal":
                return self._normal(rng, persona)
            case "account_takeover":
                return self._takeover(rng, persona)
            case "search_takeover":
                return self._search_takeover(rng, persona)
            case "card_testing":
                return self._card_testing(rng)
            case "scraper":
                return self._scraper(rng)
            case "click_fraud":
                return self._click_fraud(rng, persona)
        raise ValueError(f"unknown scenario kind {kind!r}")

    def _normal(self, rng: random.Random, persona: str) -> Scenario:
        queries = self._queries(rng, persona, rng.randint(1, 3))
        label = persona
        if rng.random() < 0.3:
            other = rng.choice([p for p in PERSONAS if p != persona])
            queries += self._queries(rng, other, 1)
            label = f"{persona} + {other.lower()}"
        if rng.random() < 0.35:
            events = self._journey(rng, queries[:1])
            for _ in range(rng.randint(3, 12)):
                events.append(self.next_action(rng, events, persona=persona))
            label = f"{persona}, one action at a time"
        else:
            events = self._journey(rng, queries)
        if rng.random() < 0.2:
            trending = [r.id for r in self.api.rank([], limit=50)]
            at = rng.randint(1, len(events))
            touched = {e.item for e in events if e.item}
            if item := position_biased(rng, trending, touched):
                events.insert(at, Event("view", item=item))
        if rng.random() < 0.6 and open_cart(events):
            events.append(Event("order"))
        return Scenario("normal", f"{label} (normal)", events)

    def next_action(
        self,
        rng: random.Random,
        events: list[Event],
        *,
        temperature: float = 1.0,
        diversity: float = 0.0,
        persona: str | None = None,
    ) -> Event:
        """What a shopper does next: usually views an item sampled from the
        model's distribution for this history; sometimes carts the item just
        viewed, checks out, or searches for something new."""
        last = events[-1] if events else None
        if last is None or (last.type != "search" and rng.random() < SEARCH_P):
            return Event("search", query=self.new_query(rng, events, persona))
        if last.type == "view" and rng.random() < CART_P:
            if last.item not in open_cart(events):
                return Event("add_to_cart", item=last.item)
        if last.type == "add_to_cart" and rng.random() < ORDER_P:
            return Event("order")
        item = self.next_likely(rng, events, temperature, diversity)
        if item is None:
            return Event("search", query=self.new_query(rng, events, persona))
        return Event("view", item=item)

    def next_likely(
        self,
        rng: random.Random,
        events: list[Event],
        temperature: float = 1.0,
        diversity: float = 0.0,
    ) -> str | None:
        """An item sampled from the model's distribution for `events`."""
        ranked = self.api.rank(events, limit=HEAD)
        for r in ranked:
            self.catalog.update(r.id, r.data)
        touched = {e.item for e in events if e.item}
        pick = sample_next(
            [r for r in ranked if r.id not in touched],
            [self.catalog.get(i).get("name") for i in touched],
            tau=self.tau[mode_of(events)],
            temperature=temperature,
            diversity=diversity,
            rng=rng,
        )
        return pick.id if pick else None

    def new_query(
        self, rng: random.Random, events: list[Event], persona: str | None = None
    ) -> str:
        """A query the session has not run: the persona's if it has one left,
        any persona's otherwise."""
        used = {e.query for e in events if e.type == "search"}
        own = [q for q in PERSONAS.get(persona, []) if q not in used]
        pool = own or [q for qs in PERSONAS.values() for q in qs if q not in used]
        return rng.choice(pool or [q for qs in PERSONAS.values() for q in qs])

    def _queries(self, rng: random.Random, persona: str, k: int) -> list[str]:
        return rng.sample(PERSONAS[persona], k=min(k, len(PERSONAS[persona])))

    def _journey(self, rng: random.Random, queries: list[str]) -> list[Event]:
        events: list[Event] = []
        touched: set[str] = set()
        for q in queries:
            events.append(Event("search", query=q))
            results = [r.id for r in self.api.rank(events, limit=20)]
            viewed = []
            for _ in range(rng.choice([1, 2, 2, 3])):
                item = position_biased(rng, results, touched)
                if item is None:
                    break
                events.append(Event("view", item=item))
                touched.add(item)
                viewed.append(item)
                if rng.random() < 0.35:
                    recs = [r.id for r in self.api.rank(events, limit=10)]
                    if rec := position_biased(rng, recs, touched):
                        events.append(Event("view", item=rec))
                        touched.add(rec)
                        viewed.append(rec)
            if viewed and rng.random() < 0.55:
                events.append(Event("add_to_cart", item=rng.choice(viewed)))
        return events

    def unrelated(
        self, rng: random.Random, events: list[Event], pool: list[str] | None = None
    ) -> str:
        """A random item from a category the session has not touched."""
        pool = pool or self.catalog.ids
        seen = {self.catalog.category(e.item) for e in events if e.item}
        touched = {e.item for e in events if e.item}
        for _ in range(500):
            item = rng.choice(pool)
            if self.catalog.category(item) not in seen and item not in touched:
                return item
        return rng.choice(pool)

    def _takeover(self, rng: random.Random, persona: str) -> Scenario:
        events = self._journey(rng, self._queries(rng, persona, rng.randint(1, 2)))
        start = len(events)
        for _ in range(rng.randint(3, 5)):
            events.append(
                Event("add_to_cart", item=self.unrelated(rng, events, self.resellable))
            )
        events.append(Event("order"))
        return Scenario(
            "account_takeover",
            f"{persona}, then account takeover",
            events,
            start,
            [
                f"Steps 1–{start} are the account owner. From step {start + 1}, "
                "someone else carts popular, expensive items and checks out."
            ],
        )

    def _search_takeover(self, rng: random.Random, persona: str) -> Scenario:
        """The hard case: each fraudulent step is a search and a top result,
        which is exactly what a normal shopper does."""
        events = self._journey(rng, self._queries(rng, persona, rng.randint(1, 2)))
        start = len(events)
        touched = {e.item for e in events if e.item}
        for q in rng.sample(RESALE_QUERIES, k=rng.randint(2, 3)):
            events.append(Event("search", query=q))
            results = [r.id for r in self.api.rank(events, limit=3)]
            if item := next((r for r in results if r not in touched), None):
                events.append(Event("add_to_cart", item=item))
                touched.add(item)
        events.append(Event("order"))
        return Scenario(
            "search_takeover",
            f"{persona}, then takeover via search",
            events,
            start,
            [
                f"Steps 1–{start} are the account owner. From step {start + 1}, "
                "someone else searches for expensive goods and carts top results.",
                "A known blind spot: every step is a search and its top result, "
                "which is what normal shoppers do. Within one session this cannot "
                "be told apart from an owner who decided to buy a laptop; catch it "
                "with account signals (new device, new shipping address).",
            ],
        )

    def _card_testing(self, rng: random.Random) -> Scenario:
        events: list[Event] = []
        for _ in range(rng.randint(5, 8)):
            events.append(
                Event("add_to_cart", item=self.unrelated(rng, events, self.cheap))
            )
        events.append(Event("order"))
        return Scenario(
            "card_testing",
            "Card testing",
            events,
            0,
            ["Cheap, unrelated items carted straight away, then ordered."],
        )

    def _scraper(self, rng: random.Random) -> Scenario:
        events = [
            Event("view", item=rng.choice(self.catalog.ids))
            for _ in range(rng.randint(8, 14))
        ]
        return Scenario(
            "scraper",
            "Scraper bot",
            events,
            0,
            ["Views of random catalog items, one after another."],
        )

    def _click_fraud(self, rng: random.Random, persona: str) -> Scenario:
        events: list[Event] = []
        for q in self._queries(rng, persona, 3):
            events.append(Event("search", query=q))
            for _ in range(2):
                events.append(Event("view", item=self.unrelated(rng, events)))
        return Scenario(
            "click_fraud",
            f"Click fraud on {persona.lower()} searches",
            events,
            1,
            ["Each search is followed by clicks on items unrelated to it."],
        )
