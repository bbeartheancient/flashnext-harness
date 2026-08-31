from fabric.intent import resolve
from fastapi.testclient import TestClient

from fabric.main import app

client = TestClient(app)


def test_hadamard_autorun():
    r = resolve("hadamard 8")
    assert r["match"]["id"] == "hadamard"
    assert r["fields"]["n"] == 8
    assert r["autorun"] is True
    assert r["department"] == 400


def test_h_paren():
    r = resolve("H(12) sylvester")
    assert r["match"]["id"] == "hadamard"
    assert r["fields"]["n"] == 12
    assert r["fields"]["method"] == "sylvester"


def test_yagi():
    r = resolve("yagi 145")
    assert r["match"]["id"] == "antenna"
    assert r["fields"]["kind"] == "yagi"
    assert r["fields"]["f_mhz"] == 145


def test_sage_factor():
    r = resolve("factor(x**2-1)")
    assert r["match"]["id"] == "sage"
    assert "x**2" in r["fields"]["expr"]


def test_status_is_ship():
    r = resolve("computer, status")
    assert r["match"]["id"] in ("ship", "eng_ship", "chat")
    # prefer the calculator over a chat dump
    assert r["match"]["id"] != "chat" or r["fallback"] == "chat"


def test_help_index():
    r = resolve("help")
    assert r["fallback"] == "help"
    assert any(c["id"] == "hadamard" for c in r["candidates"])


def test_unknown_falls_to_chat():
    r = resolve("what is the weather in walsall")
    assert r["match"]["id"] == "chat"
    assert r["autorun"] is True
    assert "walsall" in r["fields"]["message"]


def test_term_implies_chat():
    r = resolve("hadamard 8", prefer_chat=True)
    assert r["match"]["id"] == "chat"
    assert r["autorun"] is True
    assert "hadamard 8" in r["fields"]["message"]
    r2 = resolve("/status", prefer_chat=True)
    assert r2.get("match", {}).get("id") == "ship" or r2.get("url") == "/ship_status" or r2.get("command") == "status"


def test_dashboard_opens_wizard(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_MICROAPP_DIR", str(tmp_path))
    from fabric.microapps import reset_registry
    reset_registry()
    r = resolve("make a dashboard")
    assert r.get("wizard") is True
    assert r["autorun"] is False
    reset_registry()


def test_make_app_composes(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_MICROAPP_DIR", str(tmp_path))
    from fabric.microapps import reset_registry
    reset_registry()
    r = resolve("make a 7-day habit tracker")
    assert r["match"]["id"] == "compose"
    assert r["autorun"] is True
    assert "habit" in r["fields"]["query"]
    reset_registry()


def test_make_yagi_stays_antenna():
    r = resolve("make a yagi antenna plot at 145")
    assert r["match"]["id"] == "antenna"


def test_neosho_is_place():
    r = resolve("show me the city of neosho, mo")
    assert r["match"]["id"] == "place"
    assert r["autorun"] is True
    assert "neosho" in r["fields"]["q"].lower()


def test_survey_from_to():
    r = resolve("survey from 52.445,-2.598 to 52.445,-2.655")
    assert r["match"]["id"] == "survey"
    assert r["fields"]["tx_lat"] == 52.445
    assert r["fields"]["rx_lon"] == -2.655


def test_resolve_endpoint():
    r = client.get("/api/resolve", params={"q": "hadamard 8"})
    assert r.status_code == 200
    body = r.json()
    assert body["match"]["id"] == "hadamard"
    assert body["fields"]["n"] == 8


def test_shell_has_command_bar():
    html = client.get("/ui/").text
    assert "id=\"cmd\"" in html
    assert "id=\"cmdIn\"" in html
    assert "id=\"boxCol\"" in html
    assert "id=\"ioRight\"" in html
    assert "id=\"ctlTerm\"" in html and "id=\"ctlViz\"" in html
