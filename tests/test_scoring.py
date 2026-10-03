import math

import numpy as np
import pytest

from anomaly.scoring import Observation, fit_tau, predict


def exact(scores: np.ndarray, target: int, tau: float, eps: float):
    logits = scores / tau
    p = np.exp(logits - logits.max())
    p = (1 - eps) * p / p.sum() + eps / len(scores)
    entropy = -float(np.sum(p * np.log(p)))
    rank = 1 + int(np.sum(scores > scores[target]))
    return math.log(p[target]), entropy, rank


def observation(scores: np.ndarray, target: int, head: int) -> Observation:
    order = np.argsort(-scores)
    head_idx, tail_idx = order[:head], order[head:]
    return Observation(
        mode="browse",
        head_ids=[str(i) for i in head_idx],
        head_scores=scores[head_idx].tolist(),
        tail_scores=scores[tail_idx].tolist(),
        target_score=float(scores[target]),
        n_items=len(scores),
    )


@pytest.mark.parametrize("target", [0, 3, 250, 999])
def test_full_tail_sample_matches_exact_softmax(target):
    rng = np.random.default_rng(0)
    scores = rng.normal(1.2, 0.05, 1000)
    tau, eps = 0.02, 0.02
    pred = predict(observation(scores, target, head=100), tau, eps)
    logp, entropy, rank = exact(scores, target, tau, eps)
    assert pred.logp == pytest.approx(logp, abs=1e-9)
    assert pred.entropy == pytest.approx(entropy, rel=1e-9)
    assert pred.rank == pytest.approx(rank, abs=1.0)


def test_sampled_tail_is_close_to_exact():
    rng = np.random.default_rng(1)
    scores = rng.normal(1.2, 0.05, 20_000)
    tau, eps = 0.02, 0.02
    target = int(np.argsort(-scores)[5000])
    full = observation(scores, target, head=100)
    sample = rng.choice(full.tail_scores, 99, replace=False).tolist()
    pred = predict(Observation(**{**full.__dict__, "tail_scores": sample}), tau, eps)
    logp, entropy, rank = exact(scores, target, tau, eps)
    assert pred.logp == pytest.approx(logp, abs=0.5)
    assert pred.entropy == pytest.approx(entropy, rel=0.05)
    assert pred.rank == pytest.approx(rank, rel=0.3)


def test_uniform_scores_have_no_excess_surprisal():
    scores = np.full(500, 1.0)
    pred = predict(observation(scores, 42, head=100), 0.02, 0.02)
    assert pred.logp == pytest.approx(-math.log(500))
    assert pred.surprisal - pred.entropy == pytest.approx(0.0, abs=1e-9)


def test_eps_bounds_surprisal():
    scores = np.linspace(2.0, 0.0, 1000)
    pred = predict(observation(scores, 999, head=100), 0.001, 0.02)
    assert pred.surprisal <= -math.log(0.02 / 1000) + 1e-9


def test_fit_tau_recovers_sampling_temperature():
    rng = np.random.default_rng(2)
    true_tau, eps = 0.03, 0.0
    obs = []
    for _ in range(300):
        scores = rng.normal(1.2, 0.05, 400)
        p = np.exp(scores / true_tau - (scores / true_tau).max())
        target = int(rng.choice(len(scores), p=p / p.sum()))
        obs.append(observation(scores, target, head=400))
    assert fit_tau(obs, eps) == pytest.approx(true_tau, rel=0.2)
