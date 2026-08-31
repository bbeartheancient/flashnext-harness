import pytest
from fastapi.testclient import TestClient

from fabric.log_store import LogStore, get_log_store
from fabric.main import app
from fabric.tools import log_recent, ship_status


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_DB_PATH", str(tmp_path / "fabric.db"))
    yield tmp_path / "fabric.db"


def test_log_roundtrip(tmp_db):
    store = LogStore(str(tmp_db))
    store.append("hello", source="op", level="info")
    store.append("warn!", source="fabric", level="warn")
    rows = store.recent(limit=10)
    assert len(rows) == 2
    assert rows[0]["text"] == "warn!"
    assert rows[1]["text"] == "hello"
    assert [r["text"] for r in store.recent(level="info")] == ["hello"]
    assert [r["text"] for r in store.recent(source="fabric")] == ["warn!"]


def test_get_log_store_env_swap(tmp_db, monkeypatch):
    first = get_log_store()
    other = str(tmp_db.parent / "other.db")
    monkeypatch.setenv("FABRIC_DB_PATH", other)
    second = get_log_store()
    assert first.path != second.path
    assert second.path == other


def test_ship_status_structure():
    s = ship_status()
    assert isinstance(s["gpus"], list)
    assert isinstance(s["slots"], list) and len(s["slots"]) >= 5
    assert set(s.keys()) >= {"ts", "gpus", "slots", "disk", "memory", "load"}
    assert len(s["load"]) == 3
    assert "free_gb" in s["disk"]
    assert "mem_available_gb" in s["memory"]


def test_miniapp_trace_roundtrip(tmp_db, monkeypatch):
    from fabric.log_store import reset_log_store

    reset_log_store()
    store = get_log_store()
    store.trace("make a 7-day habit tracker", "compose",
                choice_kind="kernel", choice_id="week",
                spec_id="app_habit",
                spec={"id": "app_habit", "title": "7-day habit tracker",
                      "kernel": "week", "fields": [{"name": "habit"}]},
                gates={"pass": True}, passed=True)
    rows = store.traces(passed=True)
    assert rows[0]["query"].startswith("make a 7-day")
    assert rows[0]["spec"]["kernel"] == "week"
    examples = store.sft_examples()
    assert examples[0]["spec"]["kernel"] == "week"
    reset_log_store()


def test_log_recent_tool(tmp_db):
    get_log_store().append("via-tool")
    rows = log_recent(limit=5)
    assert rows[-1]["text"] == "via-tool"


client = TestClient(app)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_log_endpoints(tmp_db):
    r = client.post("/log", json={"text": "ep-test", "source": "unit"})
    assert r.status_code == 200
    r = client.get("/log/recent?limit=5&source=unit")
    texts = [e["text"] for e in r.json()["entries"]]
    assert "ep-test" in texts
    assert client.post("/log", json={"text": "  "}).status_code == 400


def test_ship_status_endpoint():
    r = client.get("/ship_status")
    assert r.status_code == 200
    assert "gpus" in r.json()
