import pytest
from fastapi.testclient import TestClient

from anomaly.detector import contexts
from anomaly.events import Event
from anomaly.server import EXAMPLES_PATH, app


@pytest.fixture
def offline(monkeypatch):
    # an empty key wins over .env, so the app runs on bundled examples only
    monkeypatch.setenv("UNBOXAI_API_KEY", "")
    with TestClient(app) as client:
        yield client


needs_examples = pytest.mark.skipif(
    not EXAMPLES_PATH.exists(), reason="run scripts/calibrate.py first"
)


@needs_examples
def test_examples_score_without_an_api_key(offline):
    assert offline.get("/api/meta").json()["live"] is False
    for ex in offline.get("/api/examples").json():
        res = offline.post(
            "/api/score", json={"events": ex["events"], "example_id": ex["id"]}
        )
        assert res.status_code == 200
        summary = res.json()["summary"]
        # takeover via search is the documented blind spot
        if ex["kind"] != "search_takeover":
            assert summary["flagged"] == (ex["kind"] != "normal")


@needs_examples
def test_edited_sequence_needs_an_api_key(offline):
    ex = offline.get("/api/examples").json()[0]
    res = offline.post(
        "/api/score", json={"events": ex["events"][:-1], "example_id": ex["id"]}
    )
    assert res.status_code == 503


def test_invalid_event_is_rejected(offline):
    res = offline.post("/api/score", json={"events": [{"type": "view"}]})
    assert res.status_code == 422


def test_choices_after_a_search_are_read_in_both_contexts():
    search = Event("search", query="socks")
    view = Event("view", item="a")
    assert [name for name, _ in contexts([search])] == ["history"]
    both = contexts([search, view])
    assert [name for name, _ in both] == ["history", "results"]
    assert both[1][1] == [search, view, search]
    assert [name for name, _ in contexts([view])] == ["history"]
