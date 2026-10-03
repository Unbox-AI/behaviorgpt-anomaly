"""Score a shopper's sequence, flag it when it crosses the risk barrier, and
say which actions pushed it there.

Not every event is a choice the model can be surprised by:

- a search sets the context for what follows; its query is not scored;
- the first time the shopper touches an item (view, or add to cart without
  a view) is a choice among the whole store, and is scored;
- later events on an item already touched (add to cart after viewing,
  removing it, ordering the cart) are steps down the funnel for a choice
  already made. The model ranks items already seen low on purpose, so
  scoring these would read normal checkout as suspicious.

A click can come from two places. After a search, shoppers follow a
recommendation from the item they are on, or go back to the results list.
Each choice after a search is therefore read in both contexts (the full
history, and the history plus the last query again) and scored in the one
that explains it better. Calibration is done the same way, so this does
not make sessions look more normal than normal shoppers do.

Each scored step becomes a standardized surprise u (excess surprisal, scaled
by normal shoppers' spread). Two statistics are built from it:

- the session z-score, sum(u) / sqrt(n): length-normalized, so a 5-step and
  a 30-step session are judged on the same scale;
- a one-sided CUSUM, max(0, S + u - k): it accumulates evidence while the
  behavior stays off, forgets it while it is normal, and raises the flag at
  the step it crosses the barrier. That catches a session that starts
  normal and turns, which an average would dilute.
"""

import math
from dataclasses import dataclass, field

from anomaly.api import BehaviorAPI
from anomaly.calibration import Calibration
from anomaly.catalog import Catalog
from anomaly.events import Event
from anomaly.scoring import HEAD, Observation, Prediction, predict

REF_SIZE = 99  # one call holds 100 ids: the sample plus the chosen item


@dataclass
class Step:
    index: int
    event: dict
    status: str  # "scored", "context", "funnel", "order" or "unknown"
    item: dict | None = None
    mode: str | None = None
    context: str | None = None  # "history", or "results" for a return to a search
    logp: float | None = None
    surprisal: float | None = None
    entropy: float | None = None
    excess: float | None = None
    u: float | None = None
    rank: float | None = None
    rank_pct: float | None = None
    expected: list[dict] = field(default_factory=list)
    cusum: float = 0.0
    risk: float = 0.0
    reasons: list[str] = field(default_factory=list)


def plan(events: list[Event]) -> list[str]:
    """The status of each event: which ones are scored choices."""
    touched: set[str] = set()
    statuses = []
    for ev in events:
        if ev.type == "search":
            statuses.append("context")
        elif ev.type == "order":
            statuses.append("order")
        elif ev.item in touched or ev.type == "remove_from_cart":
            statuses.append("funnel")
        else:
            statuses.append("scored")
        if ev.item:
            touched.add(ev.item)
    return statuses


def contexts(prefix: list[Event]) -> list[tuple[str, list[Event]]]:
    """The histories a choice after `prefix` may have been made from."""
    out = [("history", prefix)]
    last_search = next((e for e in reversed(prefix) if e.type == "search"), None)
    if last_search is not None and prefix[-1].type != "search":
        out.append(("results", prefix + [last_search]))
    return out


def mode_of(history: list[Event]) -> str:
    return "search" if history and history[-1].type == "search" else "browse"


def best(
    observations: list[Observation], cal: Calibration
) -> tuple[Observation, Prediction] | None:
    """The context that explains the choice best, and its prediction."""
    preds = [(o, predict(o, cal.tau[o.mode], cal.eps)) for o in observations]
    return max(preds, key=lambda p: p[1].logp, default=None)


class Detector:
    def __init__(
        self,
        api: BehaviorAPI | None,
        catalog: Catalog,
        calibration: Calibration,
        *,
        ref_seed: int = 0,
    ):
        self.api = api
        self.catalog = catalog
        self.cal = calibration
        self.refs = catalog.reference_sample(REF_SIZE, seed=ref_seed)
        self._ref_set = set(self.refs)

    def observe(self, events: list[Event]) -> dict[int, list[Observation]]:
        """Call the model for every scored step and context, in parallel."""
        jobs = []  # (step, context name, history)
        for i, status in enumerate(plan(events)):
            if status == "scored":
                jobs += [(i, name, hist) for name, hist in contexts(events[:i])]
        requests = []
        for i, _, hist in jobs:
            ids = [events[i].item] + [r for r in self.refs if r != events[i].item]
            requests.append({"history": hist, "limit": HEAD})
            requests.append(
                {"history": hist, "limit": len(ids), "filters": {"id": ids}}
            )
        results = self.api.rank_many(requests)

        out: dict[int, list[Observation]] = {i: [] for i, _, _ in jobs}
        for n, (i, name, hist) in enumerate(jobs):
            head, ref = results[2 * n], results[2 * n + 1]
            for r in head + ref:
                self.catalog.update(r.id, r.data)
            target = events[i].item
            score = next((r.score for r in ref + head if r.id == target), None)
            if score is None:
                continue
            head_ids = [r.id for r in head]
            in_head = set(head_ids)
            out[i].append(
                Observation(
                    mode=mode_of(hist),
                    head_ids=head_ids,
                    head_scores=[r.score for r in head],
                    tail_scores=[
                        r.score
                        for r in ref
                        if r.id in self._ref_set and r.id not in in_head
                    ],
                    target_score=score,
                    n_items=len(self.catalog)
                    + sum(h not in self.catalog for h in head_ids),
                    context=name,
                    query=hist[-1].query if name == "results" else None,
                )
            )
        return out

    def score(
        self,
        events: list[Event],
        observations: dict[int, list[Observation]] | None = None,
    ) -> dict:
        if observations is None:
            observations = self.observe(events)
        cal = self.cal
        steps: list[Step] = []
        cusum = 0.0
        flagged_at: int | None = None
        touched_at: dict[str, int] = {}

        for i, (ev, status) in enumerate(zip(events, plan(events), strict=True)):
            step = Step(index=i, event=ev.to_dict(), status=status)
            if ev.item:
                step.item = self.catalog.get(ev.item)

            if status == "scored":
                chosen = best(observations.get(i) or [], cal)
                if chosen is None:
                    step.status = "unknown"
                    step.reasons.append("The model does not know this item.")
                else:
                    self._fill_scored(step, *chosen, events, i)
                    cusum = max(0.0, cusum + step.u - cal.cusum_k)
            elif status == "context":
                step.reasons.append("Sets the context for the next steps.")
            elif status == "funnel":
                if ev.type == "remove_from_cart":
                    step.reasons.append("Removes an item from the cart.")
                else:
                    step.reasons.append(
                        f"Same item as step {touched_at[ev.item] + 1}: "
                        "a step down the funnel."
                    )
            elif status == "order":
                step.reasons.append(
                    "Order placed after the session crossed the barrier: "
                    "hold it for review."
                    if flagged_at is not None
                    else "Order placed below the barrier."
                )

            if ev.item and ev.item not in touched_at:
                touched_at[ev.item] = i
            step.cusum = cusum
            step.risk = cusum / cal.barrier
            if flagged_at is None and cusum >= cal.barrier:
                flagged_at = i
            steps.append(step)

        return self._summarize(steps, flagged_at)

    def _fill_scored(
        self,
        step: Step,
        obs: Observation,
        pred: Prediction,
        events: list[Event],
        i: int,
    ) -> None:
        cal = self.cal
        lo, hi = cal.clip
        step.mode = obs.mode
        step.context = obs.context
        step.logp = pred.logp
        step.surprisal = pred.surprisal
        step.entropy = pred.entropy
        step.excess = pred.surprisal - pred.entropy
        u = (step.excess - cal.loc[obs.mode]) / cal.scale[obs.mode]
        step.u = min(max(u, lo), hi)
        step.rank = pred.rank
        step.rank_pct = pred.rank / obs.n_items
        step.expected = [
            self.catalog.get(item_id) | {"p": p}
            for item_id, p in zip(obs.head_ids[:3], pred.head_probs[:3], strict=False)
        ]
        step.reasons = self._reasons(step, obs, events, i)

    def _reasons(
        self, step: Step, obs: Observation, events: list[Event], i: int
    ) -> list[str]:
        ev = events[i]
        reasons = [
            f"Ranked #{step.rank:,.0f} of {obs.n_items:,} for this history "
            f"(p = {math.exp(step.logp):.2g})."
        ]
        if obs.context == "results" and step.u < 1:
            reasons.append(f"Read as a return to the results for “{obs.query}”.")
        if obs.mode == "search" and step.rank > HEAD:
            query = obs.query or events[i - 1].query
            reasons.append(f"Not among the top {HEAD} results for “{query}”.")

        prior = [e.item for e in events[:i] if e.item]
        cat = self.catalog.get(ev.item).get("categories")
        prior_cats = {
            c for it in prior if (c := self.catalog.get(it).get("categories"))
        }
        after_search = i > 0 and events[i - 1].type == "search"
        if cat and prior_cats and len(prior) >= 2 and cat not in prior_cats:
            seen = ", ".join(sorted(prior_cats)[:3])
            how = (
                "right after a search"
                if after_search
                else "with no search leading to it"
            )
            reasons.append(f"Jumps to {cat} {how} (session so far: {seen}).")

        price = self.catalog.price(ev.item)
        prior_prices = sorted(p for it in prior if (p := self.catalog.price(it)))
        if price and prior_prices:
            median = prior_prices[len(prior_prices) // 2]
            if price >= 3 * median:
                reasons.append(
                    f"Price ${price:,.0f} is {price / median:.0f}× the session median."
                )

        if ev.type == "add_to_cart":
            reasons.append("Added to cart without viewing it first.")
        return reasons

    def _summarize(self, steps: list[Step], flagged_at: int | None) -> dict:
        cal = self.cal
        scored = [s for s in steps if s.status == "scored"]
        n = len(scored)
        log_prob = sum(s.logp for s in scored)
        z = sum(s.u for s in scored) / math.sqrt(n) if n else 0.0
        positive = sum(max(s.u, 0.0) for s in scored)
        drivers = sorted(scored, key=lambda s: s.u, reverse=True)[:3]
        contributors = [
            {
                "index": s.index,
                "item": s.item,
                "event": s.event,
                "u": s.u,
                "share": max(s.u, 0.0) / positive if positive else 0.0,
                "reasons": s.reasons,
            }
            for s in drivers
            if s.u > 0.5
        ]
        return {
            "steps": [s.__dict__ for s in steps],
            "summary": {
                "n_events": len(steps),
                "n_scored": n,
                "log_prob": log_prob,
                "mean_surprisal": -log_prob / n if n else None,
                "perplexity": math.exp(-log_prob / n) if n else None,
                "mean_excess": sum(s.excess for s in scored) / n if n else None,
                "z": z,
                "percentile": cal.percentile(z),
                "peak_risk": max((s.risk for s in steps), default=0.0),
                "flagged": flagged_at is not None,
                "flagged_at": flagged_at,
            },
            "contributors": contributors,
        }
