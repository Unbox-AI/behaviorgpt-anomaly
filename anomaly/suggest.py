"""Sample a plausible next item from the model, with variety.

The model's top items are often variants of one product: after a pack of
guitar strings, most of the top hundred are other strings. Taking the top
item, or a position-biased pick from the top ten, then produces the same
thing again and again. Generation therefore follows the usual practice for
sampling from a sequence model:

- near-duplicates are collapsed: of a family of variants only the best
  scored is kept, and families the session already touched are skipped;
- a repetition penalty, as in maximal marginal relevance: each candidate's
  score drops by `diversity` times its title similarity to the items the
  session already touched, so another pack of strings gives way to a cable
  or a tuner;
- the rest are sampled from the model's own softmax, p ∝ exp(s' / (tau · T)),
  where tau is the calibrated temperature and T > 1 spreads the picks;
- nucleus (top-p) sampling drops the tail that only adds noise.
"""

import math
import random
import re

from anomaly.api import Ranked

STOPWORDS = {"for", "with", "and", "the", "of", "in", "to", "a", "an", "-", "&"}


def title_tokens(name: str | None, n: int = 8) -> frozenset[str]:
    words = [w for w in re.split(r"[^\w]+", (name or "").lower()) if w]
    return frozenset(w for w in words[:n] if w not in STOPWORDS and not w.isdigit())


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def near_duplicate(
    a: frozenset[str], b: frozenset[str], threshold: float = 0.5
) -> bool:
    return similarity(a, b) >= threshold


def sample_next(
    ranked: list[Ranked],
    touched_names: list[str],
    *,
    tau: float,
    temperature: float,
    diversity: float = 0.1,
    top_p: float = 0.9,
    rng: random.Random,
) -> Ranked | None:
    """One item from `ranked` (best first), avoiding variants of what the
    session has already touched and of better-scored candidates."""
    touched = [title_tokens(n) for n in touched_names]
    seen = list(touched)
    names = {(n or "").strip().lower() for n in touched_names}
    pool: list[tuple[Ranked, float]] = []
    for r in ranked:
        name = (r.data.get("name") or "").strip().lower()
        tokens = title_tokens(name)
        # the same listing under another id is never a new choice; variants
        # are skipped only when variety is asked for
        if name in names or (
            diversity > 0 and any(near_duplicate(tokens, s) for s in seen)
        ):
            continue
        names.add(name)
        seen.append(tokens)
        penalty = diversity * max((similarity(tokens, s) for s in touched), default=0)
        pool.append((r, r.score - penalty))
    if not pool:
        return None

    pool.sort(key=lambda p: -p[1])
    t = tau * max(temperature, 1e-3)
    top = pool[0][1]
    weights = [math.exp((score - top) / t) for _, score in pool]
    total = sum(weights)

    kept, mass = [], 0.0
    for (r, _), w in zip(pool, weights, strict=True):
        kept.append((r, w))
        mass += w / total
        if mass >= top_p:
            break
    return rng.choices([r for r, _ in kept], weights=[w for _, w in kept], k=1)[0]
