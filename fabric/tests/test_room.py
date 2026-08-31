"""Fabric room-IR endpoints (host MT path)."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fabric.main import app

client = TestClient(app)


def test_room_ir_endpoint():
    r = client.post("/api/room/ir", json={
        "src": [2.0, 1.5, 2.0],
        "lst": [6.0, 1.5, 4.0],
        "box": [8.0, 3.0, 6.0],
        "n_rays": 128,
        "max_bounce": 0,
        "order": 3,
    })
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body
    assert body["n_arrivals"] >= 1
    assert body["analysis"]["peak_delay_s"] > 0
    assert body["figure"]["type"] == "xy"
    last = client.get("/api/room/last")
    assert last.status_code == 200
    assert last.json()["figure"]["type"] == "xy"


def test_room_ir_rejects_outside():
    r = client.post("/api/room/ir", json={
        "src": [-1, 1, 1], "lst": [2, 1, 2],
    })
    assert r.status_code == 200
    assert "error" in r.json()


def test_room_last_empty_404():
    from fabric import room

    room._last["figure"] = None
    room._last["result"] = None
    assert client.get("/api/room/last").status_code == 404
