"""The web app: a JSON API over the detector, and the static front end.

uv run uvicorn anomaly.server:app --reload
"""

import json
import os
import random
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from anomaly.api import BehaviorAPI
from anomaly.calibration import Calibration
from anomaly.catalog import CACHE_PATH, DATA_DIR, Catalog
from anomaly.detector import Detector
from anomaly.events import Event
from anomaly.scoring import Observation
from anomaly.simulate import KINDS, PERSONAS, Simulator

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
EXAMPLES_PATH = DATA_DIR / "examples.json"
MAX_EVENTS = 40


class State:
    catalog: Catalog
    calibration: Calibration
    detector: Detector | None
    simulator: Simulator | None
    examples: list[dict]


state = State()


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv()
    state.catalog = Catalog.load()
    state.calibration = Calibration.load()
    state.examples = []
    if EXAMPLES_PATH.exists():
        data = json.loads(EXAMPLES_PATH.read_text())
        state.examples = data["examples"]
        for item_id, item in data["items"].items():
            state.catalog.update(item_id, item)

    api = None
    if os.environ.get("UNBOXAI_API_KEY"):
        api = BehaviorAPI(catalog_id=state.catalog.catalog_id, cache_path=CACHE_PATH)
    state.detector = Detector(api, state.catalog, state.calibration)
    state.simulator = Simulator(api, state.catalog) if api else None
    yield


app = FastAPI(title="BehaviorGPT anomaly detection", lifespan=lifespan)


class EventIn(BaseModel):
    type: str
    item: str | None = None
    query: str | None = None


class ScoreIn(BaseModel):
    events: list[EventIn] = Field(max_length=MAX_EVENTS)
    example_id: str | None = None


class SimulateIn(BaseModel):
    kind: str
    seed: int | None = None
    persona: str | None = None


class NextIn(BaseModel):
    events: list[EventIn] = Field(max_length=MAX_EVENTS)
    odd: bool = False
    type: str = "view"  # the event type of an odd click
    # for the item of a view: 1 samples from the model's calibrated
    # distribution, which the detector reads as normal; higher spreads picks,
    # and the detector notices
    temperature: float = Field(default=1.0, ge=0.5, le=10.0)
    # how hard more of the same is pushed down (anomaly/suggest.py); by
    # default only when the temperature asks for variety
    diversity: float | None = Field(default=None, ge=0.0, le=1.0)


def _events(raw: list[EventIn]) -> list[Event]:
    try:
        return [Event.from_dict(e.model_dump()) for e in raw]
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


def _require_api() -> None:
    if state.simulator is None:
        raise HTTPException(
            503,
            "No UNBOXAI_API_KEY is set, so only the bundled examples can be "
            "scored. Add a key to .env to score your own sequences.",
        )


def _stored_observations(example_id: str, events: list[Event]):
    ex = next((e for e in state.examples if e["id"] == example_id), None)
    if ex is None or [Event.from_dict(e) for e in ex["events"]] != events:
        return None
    return {
        int(i): [Observation(**o) for o in ol] for i, ol in ex["observations"].items()
    }


@app.get("/api/meta")
def meta() -> dict:
    cal = state.calibration
    return {
        "catalog_id": state.catalog.catalog_id,
        "store_size": len(state.catalog),
        "live": state.simulator is not None,
        "kinds": KINDS,
        "personas": sorted(PERSONAS),
        "calibration": {
            "tau": cal.tau,
            "barrier": cal.barrier,
            "target_fpr": cal.target_fpr,
            "cusum_k": cal.cusum_k,
            "report": cal.report,
            "calibrated": bool(cal.normal_z),
        },
    }


@app.get("/api/examples")
def examples() -> list[dict]:
    return [
        {k: ex[k] for k in ("id", "kind", "label", "events", "anomaly_start", "notes")}
        for ex in state.examples
    ]


@app.post("/api/score")
async def score(body: ScoreIn) -> dict:
    events = _events(body.events)
    observations = None
    if body.example_id:
        observations = _stored_observations(body.example_id, events)
    if observations is None:
        _require_api()
    return await run_in_threadpool(state.detector.score, events, observations)


@app.post("/api/simulate")
async def simulate(body: SimulateIn) -> dict:
    _require_api()
    if body.kind not in KINDS:
        raise HTTPException(422, f"kind must be one of {sorted(KINDS)}")
    if body.persona and body.persona not in PERSONAS:
        raise HTTPException(422, f"persona must be one of {sorted(PERSONAS)}")
    seed = body.seed if body.seed is not None else random.randrange(10**6)
    scenario = await run_in_threadpool(
        state.simulator.make, body.kind, seed, body.persona
    )
    return scenario.to_dict() | {"seed": seed}


@app.post("/api/next")
async def next_event(body: NextIn) -> dict:
    """What the shopper does next (see Simulator.next_action), or an odd
    click: a random item from a category the session has not touched."""
    _require_api()
    events = _events(body.events)
    rng = random.Random()
    if body.odd:
        return {"type": body.type, "item": state.simulator.unrelated(rng, events)}
    diversity = body.diversity
    if diversity is None:
        diversity = 0.25 if body.temperature > 1 else 0.0
    event = await run_in_threadpool(
        state.simulator.next_action,
        rng,
        events,
        temperature=body.temperature,
        diversity=diversity,
    )
    return event.to_dict()


@app.get("/api/items/search")
def search_items(q: str, limit: int = 12) -> list[dict]:
    return state.catalog.search(q, limit=min(limit, 50))


@app.get("/api/items/{item_id}")
def get_item(item_id: str) -> dict:
    return state.catalog.get(item_id)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
