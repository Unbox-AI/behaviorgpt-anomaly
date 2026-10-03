"""How likely is each action, given everything the shopper did before it?

At every step the model ranks the catalog for the history so far. The API
returns raw scores, not probabilities, so each step's distribution is
estimated from two calls:

- the head: the model's top `HEAD` items and their exact scores;
- the tail: the scores of a fixed uniform sample of the store, plus the item
  the shopper actually chose (an `id` filter scores any item).

The store is the catalog snapshot (anomaly/catalog.py) plus the head. The
API cannot list all of `retail_catalog`, so probabilities are relative to
this fixed set of items; that is what makes every step and every session
comparable. With a temperature `tau`, item i gets logit s_i / tau. The
partition function is the exact sum over the head plus the tail sum,
estimated from the sample and scaled to the tail's size. That gives, per
step:

- log p(item | history), and its surprisal -log p, in nats;
- the entropy of the predicted distribution: how many items were plausible.
  After "socks" almost everything is a surprise; after viewing one sock
  very little is. Surprisal minus entropy ("excess surprisal") is zero on
  average when the shopper behaves like the model expects, whatever the
  context, which is what makes steps comparable;
- the item's rank in the store.

A small uniform mixture `eps` bounds the surprisal of any one step, so a
single odd click cannot dominate a session.
"""

import math
from dataclasses import dataclass

import numpy as np

HEAD = 100


@dataclass(frozen=True)
class Observation:
    """What the model said at one step, before any temperature is applied."""

    mode: str  # "search" if the history ends with a search, else "browse"
    head_ids: list[str]
    head_scores: list[float]
    tail_scores: list[float]  # sampled items outside the head
    target_score: float
    n_items: int
    context: str = "history"  # or "results": the history plus the last query again
    query: str | None = None


@dataclass(frozen=True)
class Prediction:
    logp: float
    surprisal: float
    entropy: float
    rank: float  # 1 = the model's top pick
    head_probs: list[float]


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def predict(obs: Observation, tau: float, eps: float) -> Prediction:
    head = np.asarray(obs.head_scores, dtype=float) / tau
    tail = np.asarray(obs.tail_scores, dtype=float) / tau
    target = obs.target_score / tau
    n_head = len(head)
    n_tail = max(obs.n_items - n_head, 0)

    parts = [_logsumexp(head)]
    if n_tail and len(tail):
        parts.append(math.log(n_tail) + _logsumexp(tail) - math.log(len(tail)))
    log_z = _logsumexp(np.asarray(parts))

    def smooth(logits: np.ndarray) -> np.ndarray:
        return (1 - eps) * np.exp(logits - log_z) + eps / obs.n_items

    p_head = smooth(head)
    p_target = float(smooth(np.asarray([target]))[0])

    entropy = -float(np.sum(p_head * np.log(p_head)))
    if n_tail and len(tail):
        p_tail = smooth(tail)
        entropy -= n_tail * float(np.mean(p_tail * np.log(p_tail)))

    min_head = float(np.min(head)) if n_head else math.inf
    if target >= min_head:
        rank = 1.0 + float(np.sum(head > target))
    else:
        above = float(np.sum(tail > target)) + 0.5
        rank = n_head + 1.0 + n_tail * above / (len(tail) + 1)

    return Prediction(
        logp=math.log(p_target),
        surprisal=-math.log(p_target),
        entropy=entropy,
        rank=rank,
        head_probs=p_head.tolist(),
    )


def fit_tau(
    observations: list[Observation],
    eps: float,
    grid: np.ndarray | None = None,
) -> float:
    """The temperature that makes `observations` most likely (maximum likelihood)."""
    if grid is None:
        grid = np.geomspace(0.002, 0.5, 120)
    if not observations:
        return float(np.median(grid))
    loglik = [sum(predict(o, float(t), eps).logp for o in observations) for t in grid]
    return float(grid[int(np.argmax(loglik))])


def robust_location_scale(values: list[float]) -> tuple[float, float]:
    """Median and MAD-based standard deviation, so outliers in the
    calibration set do not move the barrier."""
    x = np.asarray(values, dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, max(mad, 1e-3)
