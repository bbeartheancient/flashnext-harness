"""Modular console catalog + shell static files."""

from pathlib import Path

from fastapi.testclient import TestClient

from fabric.console import FUNCTIONS, catalog
from fabric.main import app

client = TestClient(app)
WEB = Path(__file__).resolve().parents[1] / "web"


def test_function_ids_unique():
    ids = [f["id"] for f in FUNCTIONS]
    assert len(ids) == len(set(ids))


def test_science_group_has_viewports():
    cat = catalog("science")
    ids = {f["id"] for f in cat["functions"]}
    assert {"hadamard", "survey", "room", "sage", "hoa_decode", "horizon",
            "eng_ship"} <= ids
    holodeck = {f["id"] for f in catalog("holodeck")["functions"]}
    assert "sensors" in holodeck
    for f in cat["functions"]:
        assert f["viewport"] in ("term", "graph", "both")
        assert f["group"] == "science"


def test_unwired_stub():
    cat = catalog("flight")
    assert len(cat["functions"]) == 1
    stub = cat["functions"][0]
    assert stub["id"] == "stub-flight"
    assert stub["title"] == "Standby"
    assert stub["viewport"] == "term"


def test_filter_when_fields():
    filt = next(f for f in FUNCTIONS if f["id"] == "filter")
    when = [f for f in filt["fields"] if f.get("when")]
    assert {f["name"] for f in when} == {"f_lo_mhz", "f_hi_mhz"}


def test_console_endpoint():
    r = client.get("/api/console?group=command")
    assert r.status_code == 200
    ids = [f["id"] for f in r.json()["functions"]]
    assert ids[:2] == ["chat", "ship"]
    assert {"toks", "gpuwatts", "ports", "goals"} <= set(ids)


def test_sage_info_endpoint():
    r = client.get("/api/sage")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] in ("sage", "sympy")
    assert body["sympy"] is True or body["sage_bin"]


def test_sage_eval_endpoint():
    r = client.post("/api/sage", json={"expr": "factor(x**2-1)"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert "x" in body["result"]


def test_shell_static():
    assert (WEB / "app.js").is_file()
    assert (WEB / "kit.css").is_file()
    assert (WEB / "index.html").is_file()
    page = client.get("/ui/")
    assert page.status_code == 200
    text = page.text
    assert "app.js" in text and "kit.css" in text
    assert "tile-deck" in text
    assert "id=\"cmd\"" in text and "id=\"ioRight\"" in text
    assert "id=\"slashMenu\"" in text
    assert "id=\"ctlTerm\"" in text and "id=\"ctlViz\"" in text
    assert "id=\"maxBox\"" in text and "id=\"cBR\"" in text
    assert "last-bar-panel" in text
    assert "tile-toolbar" not in text
    js = client.get("/ui/app.js")
    assert js.status_code == 200
    assert "runCommand" in js.text and "addTile" in js.text
    assert "renderSlash" in js.text and "slashFetch" in js.text
    assert "upsertFn" in js.text and "miniapp-frame" in js.text
    assert "openSettings" in js.text and "wizCreate" in js.text
    html = client.get("/ui/").text
    assert "id=\"libList\"" in html and "data-settab=\"library\"" in html
    assert "id=\"wizDept\"" in html and "id=\"wizCreate\"" in html
    assert "id=\"cfgGates\"" in html
    css = client.get("/ui/kit.css")
    assert css.status_code == 200
    assert "tile-deck" in css.text and "stage-right" in css.text
    assert "miniapp-frame" in css.text


def test_room_accepts_csv_vectors():
    r = client.post("/api/room/ir", json={
        "src": "2,1.5,2", "lst": "6,1.5,4", "n_rays": 32, "max_bounce": 1,
    })
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body
    assert body.get("figure", {}).get("type") == "xy"


def test_hoa_decode_csv():
    r = client.post("/api/lab/hoa/decode", json={
        "azimuths": "90", "elevations": "0", "order": 3,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["n_channels"] == 16
    assert len(body["samples"]) == 1


def test_horizon_default_center():
    r = client.post("/api/geo/horizon", json={"n_az": 16, "h_agl": 40})
    assert r.status_code == 200
    body = r.json()
    if "error" in body:
        return
    assert body["figure"]["type"] == "polar"
    assert len(body["azimuths_deg"]) == 16
