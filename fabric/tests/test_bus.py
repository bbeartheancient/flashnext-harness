"""In-process sensor bus (MQTT-shaped topics)."""

from fastapi.testclient import TestClient

from fabric import bus
from fabric.main import app
from fabric.tools import TOOLS


client = TestClient(app)


def test_publish_query_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_DB_PATH", str(tmp_path / "bus.db"))
    bus._db["c"] = None
    bus._db["path"] = None
    pub = bus.publish("ship/gpu/card1", {"power_w": 42.0, "temp_c": 51.0})
    assert pub["topic"] == "ship/gpu/card1"
    rows = bus.query(prefix="ship/gpu/", limit=10)
    assert rows
    assert rows[0]["payload"]["power_w"] == 42.0
    assert bus.query(prefix="ship/cctv/") == []


def test_publish_gpus_and_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_DB_PATH", str(tmp_path / "bus.db"))
    bus._db["c"] = None
    bus._db["path"] = None
    bus.publish_gpus()
    r = client.get("/api/sensors?prefix=ship/gpu/&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["broker"]["record"] == "sqlite"
    assert body["broker"]["mqtt_host"] is None or isinstance(body["broker"]["mqtt_host"], str)


def test_sensor_query_registered():
    assert "sensor_query" in {t.__name__ for t in TOOLS}
    assert "terrain_survey" in {t.__name__ for t in TOOLS}
