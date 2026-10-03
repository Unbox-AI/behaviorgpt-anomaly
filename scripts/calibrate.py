"""Fit the risk barrier on simulated shoppers and measure how well it works.

1. Simulate normal sessions and sessions of each fraud pattern.
2. Ask the model about every scored step (cached in data/cache.sqlite).
3. On a fitting half of the normal sessions: fit the softmax temperature
   per mode by maximum likelihood, the location and scale of excess
   surprisal, and the CUSUM barrier that flags `--fpr` of normal sessions.
4. On the held-out normal sessions and the fraud sessions: report the false
   alarm rate, the detection rate and delay per pattern, and the ROC AUC.
5. Save data/calibration.json, and data/examples.json: representative
   scored sessions the app can show without an API key.

    uv run python scripts/calibrate.py
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly.api import BehaviorAPI  # noqa: E402
from anomaly.calibration import Calibration  # noqa: E402
from anomaly.catalog import CACHE_PATH, DATA_DIR, Catalog  # noqa: E402
from anomaly.detector import Detector, best  # noqa: E402
from anomaly.scoring import fit_tau, robust_location_scale  # noqa: E402
from anomaly.simulate import KINDS, Simulator  # noqa: E402

EXAMPLES_PATH = DATA_DIR / "examples.json"
MODES = ("browse", "search")


def auc(neg: list[float], pos: list[float]) -> float:
    """Probability that a random fraud session outscores a random normal one."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fit(cal: Calibration, rows) -> None:
    """Temperature, then location and scale of excess surprisal, per mode.

    Which context explains a step best depends on the temperature, so the
    two are fitted in turn until they settle."""
    steps = [ol for _, obs in rows for ol in obs.values() if ol]
    for _ in range(4):
        chosen = [best(ol, cal)[0] for ol in steps]
        for mode in MODES:
            cal.tau[mode] = fit_tau([o for o in chosen if o.mode == mode], cal.eps)
    chosen = [best(ol, cal) for ol in steps]
    for mode in MODES:
        excess = [p.surprisal - p.entropy for o, p in chosen if o.mode == mode]
        cal.loc[mode], cal.scale[mode] = robust_location_scale(excess)
        print(
            f"{mode:>6}: {len(excess)} steps, tau={cal.tau[mode]:.4f}, "
            f"excess loc={cal.loc[mode]:.2f} scale={cal.scale[mode]:.2f}"
        )


def representative(rows, key):
    """The row whose `key` is the median of its group: not the best case."""
    rows = sorted(rows, key=key)
    return rows[len(rows) // 2] if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--normal", type=int, default=500)
    parser.add_argument("--fraud", type=int, default=40, help="sessions per pattern")
    parser.add_argument("--fpr", type=float, default=0.02, help="target false alarms")
    parser.add_argument("--sessions", type=int, default=6, help="sessions in flight")
    args = parser.parse_args()

    load_dotenv()
    catalog = Catalog.load()
    api = BehaviorAPI(catalog_id=catalog.catalog_id, cache_path=CACHE_PATH, workers=12)
    sim = Simulator(api, catalog)
    detector = Detector(api, catalog, Calibration())

    jobs = [("normal", s) for s in range(args.normal)] + [
        (kind, s) for kind in KINDS if kind != "normal" for s in range(args.fraud)
    ]

    def run(job):
        scenario = sim.make(*job)
        return scenario, detector.observe(scenario.events)

    done = []
    with ThreadPoolExecutor(args.sessions) as ex:
        for n, result in enumerate(ex.map(run, jobs), 1):
            done.append(result)
            if n % 25 == 0 or n == len(jobs):
                print(f"observed {n}/{len(jobs)} sessions", flush=True)

    normal = [d for d in done if d[0].kind == "normal"]
    fraud = [d for d in done if d[0].kind != "normal"]
    random.Random(0).shuffle(normal)
    fit_rows, held_rows = normal[: len(normal) // 2], normal[len(normal) // 2 :]

    cal = Calibration(target_fpr=args.fpr)
    fit(cal, fit_rows)

    def scored(rows):
        detector.cal = cal
        return [(sc, obs, detector.score(sc.events, obs)) for sc, obs in rows]

    # the barrier only scales risk, so peaks can be read with any barrier
    peaks = [max(s["cusum"] for s in r["steps"]) for *_, r in scored(fit_rows)]
    cal.barrier = max(float(np.quantile(peaks, 1 - args.fpr)), 1.0)
    cal.normal_z = sorted(r["summary"]["z"] for *_, r in scored(fit_rows))

    held, frauds = scored(held_rows), scored(fraud)
    held_z = [r["summary"]["z"] for *_, r in held]
    held_peak = [r["summary"]["peak_risk"] for *_, r in held]
    report = {
        "normal_sessions": {"fit": len(fit_rows), "held_out": len(held_rows)},
        "false_alarm_rate": float(np.mean([r["summary"]["flagged"] for *_, r in held])),
        "patterns": {},
    }
    for kind, label in KINDS.items():
        rows = [(sc, r) for sc, _, r in frauds if sc.kind == kind]
        if kind == "normal" or not rows:
            continue
        delays = [
            r["summary"]["flagged_at"] - sc.anomaly_start
            for sc, r in rows
            if r["summary"]["flagged"] and sc.anomaly_start is not None
        ]
        report["patterns"][kind] = {
            "label": label,
            "sessions": len(rows),
            "detection_rate": float(
                np.mean([r["summary"]["flagged"] for _, r in rows])
            ),
            "median_delay_steps": float(np.median(delays)) if delays else None,
            "auc_session_z": auc(held_z, [r["summary"]["z"] for _, r in rows]),
            "auc_peak_risk": auc(
                held_peak, [r["summary"]["peak_risk"] for _, r in rows]
            ),
        }
    cal.report = report
    cal.save()
    print(json.dumps({"barrier": cal.barrier, **report}, indent=2))

    picks = []
    long_normal = [row for row in held if len(row[0].events) >= 7] or held
    with_order = [row for row in long_normal if row[0].events[-1].type == "order"]
    two_intents = [row for row in long_normal if "+" in row[0].label]
    for group in (with_order or long_normal, two_intents or long_normal):
        row = representative(group, key=lambda row: row[2]["summary"]["z"])
        if row and row not in picks:
            picks.append(row)
    for kind in KINDS:
        rows = [row for row in frauds if row[0].kind == kind]
        if row := representative(rows, key=lambda row: row[2]["summary"]["peak_risk"]):
            picks.append(row)

    examples, counts = [], {}
    for sc, obs, _ in picks:
        counts[sc.kind] = counts.get(sc.kind, 0) + 1
        examples.append(
            {
                "id": f"{sc.kind}-{counts[sc.kind]}",
                **sc.to_dict(),
                "observations": {
                    str(i): [asdict(o) for o in ol] for i, ol in obs.items()
                },
            }
        )
    used = {e["item"] for ex in examples for e in ex["events"] if "item" in e}
    used |= {
        i
        for ex in examples
        for ol in ex["observations"].values()
        for o in ol
        for i in o["head_ids"][:3]
    }
    EXAMPLES_PATH.write_text(
        json.dumps(
            {"examples": examples, "items": {i: catalog.get(i) for i in sorted(used)}}
        )
    )
    print(f"wrote {len(examples)} examples to {EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
