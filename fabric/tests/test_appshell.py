"""App Mode shell backend: blank drafts, clone (save-as), tile i/o
wrapper + wire routing validation."""

from fastapi.testclient import TestClient

from fabric.main import app
from fabric.microapps import get, reset_registry

client = TestClient(app)


def _rm(spec_id):
    if not get(spec_id):
        return
    from fabric.microapps import _root

    (_root() / f"{spec_id}.json").unlink(missing_ok=True)
    reset_registry()


def test_new_app_blank_draft():
    r = client.post("/api/library/new", json={"title": "wire test rig"})
    assert r.status_code == 200, r.text
    data = r.json()
    spec = data["spec"]
    assert spec["id"] and spec["title"] == "wire test rig"
    assert spec["kernel"] == "signal"
    assert "io" in data and "ins" in data["io"]
    # draft is loadable from the library store
    lib = client.get("/api/library").json()
    assert any(a["id"] == spec["id"] for a in lib["apps"])
    _rm(spec["id"])


def test_clone_app_save_as():
    r = client.post("/api/library/new", json={"title": "clone source"})
    src = r.json()["spec"]
    io = {"ins": [{"name": "sig", "topic": "ship/vllm/toks"}],
          "controls": [{"type": "fader", "name": "trim", "label": "t",
                        "min": 0, "max": 1, "default": 0.5}],
          "tiles": [{"name": "tile0", "kind": "series", "aux": 3}]}
    upd = client.post(f"/api/library/app/{src['id']}/io",
                      json=io)
    assert upd.status_code == 200, upd.text
    r2 = client.post(f"/api/library/app/{src['id']}/clone",
                     json={"title": "cloned copy"})
    assert r2.status_code == 200, r2.text
    clone = r2.json()["spec"]
    assert clone["id"] != src["id"]
    assert clone["title"] == "cloned copy"
    full = get(clone["id"])
    assert full["io"]["ins"][0]["name"] == "sig"
    assert full["io"]["tiles"][0]["aux"] == 3
    _rm(src["id"])
    _rm(clone["id"])


def test_wire_validation_wrapper_endpoints():
    from fabric.microfx import validate_io

    io = {
        "ins": [{"name": "sig", "topic": "bus/x"}],
        "outs": [{"name": "trace", "kind": "series"}],
        "controls": [{"type": "xy", "name": "pad"},
                     {"type": "fader", "name": "trim"}],
        "tiles": [{"name": "t0", "kind": "series"},
                  {"name": "t1", "kind": "points3d", "aux": 4}],
        "wires": [
            {"from": "in:sig", "to": "tile:t0:bus_in"},
            {"from": "tile:t0:bus_out", "to": "tile:t1:bus_in"},
            {"from": "ctl:trim", "to": "tile:t0:aux0"},
            {"from": "tile:t1:aux1", "to": "out:trace"},
            {"from": "ctl:pad_x", "to": "tile:t1:aux3"},
        ],
    }
    assert validate_io(io) is None


def test_wire_validation_rejects_unknown_ports():
    from fabric.microfx import validate_io

    base = {"tiles": [{"name": "t0", "kind": "series", "aux": 2}]}
    bad = [
        {"from": "in:nope", "to": "tile:t0:bus_in"},
        {"from": "tile:t0:aux7", "to": "out:x"},      # beyond aux cap
        {"from": "tile:ghost:bus_in", "to": "out:x"}, # no such tile
        {"from": "tile:t0:bus_nope", "to": "out:x"},  # bad wrapper port
        {"from": "tile:t0:bus_in", "to": "tile:t0:bus_in"},
    ]
    for w in bad:
        io = {**base, "wires": [w]}
        assert validate_io(io) is not None, w


def test_tile_aux_bounds():
    from fabric.microfx import validate_io

    ok = {"tiles": [{"name": "t", "kind": "lcd", "aux": 8}]}
    assert validate_io(ok) is None
    for aux in (-1, 9, "two"):
        bad = {"tiles": [{"name": "t", "kind": "lcd", "aux": aux}]}
        assert validate_io(bad) is not None, aux
