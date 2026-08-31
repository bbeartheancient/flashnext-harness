from fastapi.testclient import TestClient

from fabric.departments import catalog, pane_order, by_ext
from fabric.main import app
from fabric.tools import TOOLS


client = TestClient(app)


def test_catalog_eight_departments():
    cat = catalog()
    exts = [d["ext"] for d in cat["departments"]]
    assert exts == [100, 200, 400, 500, 600, 700, 800, 900]
    assert 300 not in exts
    wired = {d["ext"] for d in cat["departments"] if d["wired"]}
    assert {100, 400} <= wired


def test_science_pane_registry():
    sci = by_ext(400)
    assert sci["id"] == "science"
    assert sci["name"] == "Sciences"
    assert "sensors" in sci["panes"] and "lab" in sci["panes"]
    assert by_ext(300)["ext"] == 400
    order = pane_order(400)
    assert order[0] == "console"
    assert "chat" in order and "ship" in order and "log" in order
    assert order[-1] == "blank"
    assert pane_order(300) == order


def test_departments_endpoint():
    r = client.get("/api/departments")
    assert r.status_code == 200
    body = r.json()
    assert len(body["departments"]) == 8
    assert body["aliases"]["300"] == 400
    r = client.get("/api/departments/400")
    assert r.json()["name"] == "Sciences"
    assert "geo" in r.json()["pane_order"]
    r300 = client.get("/api/departments/300")
    assert r300.status_code == 200
    assert r300.json()["ext"] == 400
    assert client.get("/api/departments/999").status_code == 404


def test_room_impulse_registered():
    assert "room_impulse" in {t.__name__ for t in TOOLS}
