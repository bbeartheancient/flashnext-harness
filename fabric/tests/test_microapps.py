"""MiniApp compose / catalog / gates."""

from fabric.intent import resolve
from fabric.microapps import (
    apply_route, compose, extract_place, kit, looks_like_generate,
    looks_like_place, match_route, propose, reset_registry,
)
from fabric.tools import generate_microapp

from fastapi.testclient import TestClient

from fabric.main import app


client = TestClient(app)


def test_kit_has_templates_io_libraries():
    k = kit()
    ids = {t["id"] for t in k["templates"]}
    assert {"html", "canvas", "text", "square", "wide", "solo"} <= ids
    io = {p["id"] for p in k["io"]}
    assert io == {"command", "left", "right", "center", "term"}
    lib = {x["id"] for x in k["libraries"]}
    assert {"hadamard_build", "sage_eval", "antenna_pattern", "show_place"} <= lib
    assert "intention" in k["eval"]
    assert {i["id"] for i in k["inputs"]} >= {"text", "parameters", "place", "video"}
    assert {o["id"] for o in k["outputs"]} >= {"heatmap", "html", "3d", "media"}


def test_kit_endpoint():
    r = client.get("/api/microapps/kit")
    assert r.status_code == 200
    assert "templates" in r.json()


def test_looks_like_generate():
    assert looks_like_generate("make a 7-day habit tracker")
    assert looks_like_generate("build a bmi calculator")
    assert not looks_like_generate("hadamard 8")
    assert not looks_like_generate("what is the weather in walsall")


def test_habit_tracker_compose(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_MICROAPP_DIR", str(tmp_path))
    reset_registry()
    out = compose("make a 7-day habit tracker")
    assert "error" not in out
    spec = out["spec"]
    assert spec["generated"] is True
    assert spec["kernel"] == "week"
    assert spec["id"].startswith("app_")
    gates = out["gates"]
    assert gates["intention"] and gates["static"] and gates["dynamic"]
    assert gates["left_io"] and gates["pass"]
    names = [f["name"] for f in spec["fields"]]
    assert "goal" in names
    # MicroFX week has goal param, not select-type day field
    html = out["figure"]["html"]
    assert "MicroFX.runProgram" in html  # EEL2 runtime hosts the app
    assert "loop" in html  # EEL2 program present
    assert "https://" not in html
    r = client.get("/api/console?group=command")
    ids = [f["id"] for f in r.json()["functions"]]
    assert spec["id"] in ids
    hit = resolve("habit tracker")
    assert hit["match"]["id"] == spec["id"]
    run = client.post(spec["url"], json={"reset": ""})
    assert run.status_code == 200
    assert run.json()["figure"]["type"] == "html"
    reset_registry()


def test_bmi_and_remote_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_MICROAPP_DIR", str(tmp_path))
    reset_registry()
    out = generate_microapp("build a bmi calculator")
    assert out["spec"]["kernel"] == "bmi"
    assert any(f["name"] == "weight" for f in out["spec"]["fields"])
    bad = compose("make a tracker", html='<script src="https://evil.example/x.js"></script>')
    assert bad.get("error")
    reset_registry()


def test_compose_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_MICROAPP_DIR", str(tmp_path))
    reset_registry()
    r = client.post("/api/microapps/compose",
                    json={"query": "make a stopwatch timer"})
    assert r.status_code == 200
    body = r.json()
    assert body["spec"]["kernel"] == "timer"
    assert body["gates"]["dynamic"] is True
    assert body["gates"]["pass"] is True
    assert body["gates"]["left_io"] is True
    reset_registry()


def test_route_place_heatmap():
    rec = match_route("science", "place", "heatmap")
    assert rec and rec.get("bind") == "place"
    out = apply_route("show me the city of neosho, mo",
                      {"dept": "science", "inputs": ["place"],
                       "outputs": ["heatmap"], "viewports": 1})
    assert out.get("bind") == "place"
    assert out["spec"]["id"] == "place"
    assert out["fields"].get("q", "").lower().startswith("neosho")


def test_route_missing_combo():
    out = apply_route("hello", {"dept": "science", "inputs": ["video"],
                                "outputs": ["html"], "viewports": 1})
    assert out.get("wizard") is True
    assert out.get("clarify") is True


def test_route_stub_video(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_MICROAPP_DIR", str(tmp_path))
    reset_registry()
    out = apply_route("camera 1", {"dept": "flight", "inputs": ["video"],
                                   "outputs": ["media"], "viewports": 4})
    assert out.get("stub") is True
    assert out["spec"]["span"] == 2
    reset_registry()


def test_place_query_shape():
    assert looks_like_place("show me the city of neosho, mo")
    assert extract_place("show me the city of neosho, mo").lower().startswith("neosho")
    assert not looks_like_place("computer, status")
    p = propose("show me the city of neosho, mo")
    assert p["auto"] and p["auto"]["id"] == "place"


def test_shell_clarifies(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_MICROAPP_DIR", str(tmp_path))
    reset_registry()
    out = compose("make a dashboard")
    assert out.get("clarify") is True
    assert out["choices"]
    assert "spec" not in out or out.get("gates", {}).get("pass") is not True
    reset_registry()


def test_improvement_loop_traces(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRIC_MICROAPP_DIR", str(tmp_path))
    monkeypatch.setenv("FABRIC_DB_PATH", str(tmp_path / "loop.db"))
    from fabric.log_store import reset_log_store

    reset_log_store()
    reset_registry()
    ok = compose("make a 7-day habit tracker")
    assert ok["gates"]["pass"] is True
    weak_html = compose(
        "qqqzzz nnnmmm",
        html="<div>no interaction</div>",
    )
    assert weak_html.get("clarify") is True
    assert weak_html["gates"]["pass"] is False
    assert "dynamic" in weak_html["gates"]["reasons"]
    r = client.post("/api/microapps/trace", json={
        "query": "make a dashboard",
        "action": "choose",
        "choice": {"kind": "fn", "id": "place", "label": "terrain tile / map"},
    })
    assert r.status_code == 200
    traces = client.get("/api/microapps/traces").json()["traces"]
    actions = {t["action"] for t in traces}
    assert "compose" in actions and "choose" in actions
    sft = client.get("/api/microapps/traces", params={"sft": True}).json()
    assert any("habit" in (e["query"] + e["spec"].get("kernel", ""))
               for e in sft["examples"])
    reset_registry()
    reset_log_store()


def test_meal_planner_kernel_routing_and_compose():
    from fabric.microapps import _KERNELS, pick_kernel, validate

    assert pick_kernel("make a weekly meal planner with grocery list") == "meals"
    spec = {**_KERNELS["meals"]("Meal Planner"),
            "id": "app_meals", "title": "Meal Planner", "kernel": "meals"}
    assert validate(spec) is None
    assert "ingredient" in spec["principle"].lower()
    assert "Grocery" in spec["html"]
    assert 'localStorage' in spec["html"]
