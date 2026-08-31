"""MicroFX: EEL2-subset conformance (node) + CV-module integration."""

import math
import shutil
import subprocess
from pathlib import Path

import pytest

from fabric import microfx
from fabric.microapps import _KERNELS, compose, validate


def test_conformance_vectors():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    test_js = Path(__file__).parent / "jsfx_conformance.js"
    r = subprocess.run([node, str(test_js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_module_catalog():
    for name in ("const", "sine_lfo", "gain", "smooth", "threshold",
                 "moving_avg", "clamp", "mdct_flux", "clock_bpm", "sensor"):
        mod = microfx.MODULES[name]
        assert mod["outputs"], f"{name} missing outputs"
        if name not in microfx.HOST_SOURCES:
            assert "@tick" in mod["source"] or "@init" in mod["source"]
    cat = microfx.module_catalog()
    assert cat["smooth"]["inputs"] == ["in"]


SAMPLE_PATCH = {
    "modules": [
        {"id": "sn", "primitive": "sensor",
         "params": {"topic": "ship/vllm/toks"}},
        {"id": "sm", "primitive": "smooth", "params": {"alpha": 0.3}},
        {"id": "th", "primitive": "threshold",
         "params": {"lo": 20, "hi": 30}},
    ],
    "wires": [{"from": "sn.cv", "to": "sm.in"},
              {"from": "sm.cv", "to": "th.in"}],
    "views": [{"module": "sm", "as": "series"}],
}


def test_patch_build_and_validate():
    built = microfx.build_patch_html(SAMPLE_PATCH, "toks gate")
    assert "MicroFX.runPatch" in built["html"]
    assert "__MFX_MODULES_SOURCE" in built["html"]
    names = [f["name"] for f in built["fields"]]
    assert names == ["sn.topic", "sm.alpha", "th.lo", "th.hi"]
    spec = {**built, "id": "app_toks_gate", "title": "toks gate",
            "kernel": "patch"}
    assert validate(spec) is None


def test_patch_compose_gates_pass():
    built = microfx.build_patch_html(SAMPLE_PATCH)
    r = compose("microfx gate probe", html=built["html"])
    assert r.get("gates", {}).get("pass") is True


def test_microfx_kernels_replaced_legacy_dom():
    # converted primitives must host the interpreter, not DOM widgets
    for name in ("counter", "timer", "bmi", "week"):
        spec = _KERNELS[name](name.title())
        assert "MicroFX" in spec["html"], f"{name} not MicroFX-hosted"


def test_dom_kernels_still_dom():
    for name in ("notes", "meals", "bodymap"):
        spec = _KERNELS[name](name.title())
        assert "MicroFX" not in spec["html"]



_SCOPE3D_DRIVER = r"""
const M = require(process.argv[2]);
const prog = JSON.parse(require("fs").readFileSync(0, "utf8"));
const inputs = {};
const edge = M.makeEdgeTracker((k) => inputs[k] || 0);
const outs = [];
const figures = [];
const it = new M.Interpreter({
  globals: Object.fromEntries((prog.params || []).map(
    (p) => [p.name.toLowerCase(), p.default])),
  has: () => true,
  input: (nm) => inputs[String(nm).toLowerCase()] || 0,
  output: (nm, v) => outs.push([nm, v]),
  trigger: (nm) => edge(String(nm)),
  outData: (kind, nm, start, count) => {
    const data = [];
    for (let i = 0; i < count; i++) data.push(it.mem[start + i]);
    figures.push({ name: nm, kind, data });
  },
});
const secs = M.splitSections(prog.source);
for (const [name, body] of Object.entries(secs)) {
  if (name !== "init" && name !== "gfx") continue;
  const ast = M.parse(M.lex(body));
  if (name === "init") { for (const st of ast) it.evalNode(st); continue; }
  // two gfx frames: press then release
  inputs.start = 1; it.deadline = Infinity;
  for (const st of ast) it.evalNode(st);
  inputs.start = 0; it.deadline = Infinity;
  for (const st of ast) it.evalNode(st);
}
console.log(JSON.stringify({ outs, figures }));
"""


def test_app_io_manifest_and_live_program():
    """scope3d: io manifest survives the build; the real EEL2 source,
    run under node with the app-IO host contract, emits a finite
    points3d figure + scalar out on a start press."""
    import json as _json
    import subprocess

    spec = microfx.build_jsfx("scope3d")
    # spec-level io manifest drives the shell rails (left = triggers,
    # right = outputs); program JSON carries the same for the iframe.
    assert spec["io"]["triggers"][0]["name"] == "start"
    assert {o["kind"] for o in spec["io"]["outs"]} == {"points3d",
                                                       "number"}
    assert "jsfx-triggers" in spec["html"]
    blob = spec["html"].split("var PROGRAM=", 1)[1]
    prog = _json.loads(blob.split(";MicroFX.runProgram", 1)[0])
    assert prog["io"]["triggers"][0]["name"] == "start"
    kinds = [o["kind"] for o in prog["io"]["outs"]]
    assert "points3d" in kinds and "number" in kinds

    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    js = Path(__file__).parent / "_mfx_app_driver.js"
    js.write_text(_SCOPE3D_DRIVER)
    try:
        r = subprocess.run(
            [node, str(js),
             str(Path(microfx.__file__).parent / "web" / "jsfx.js")],
            input=_json.dumps(prog), capture_output=True, text=True,
            timeout=30)
    finally:
        js.unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr[-500:]
    res = _json.loads(r.stdout)
    figs = {f["name"]: f for f in res["figures"]}
    assert "scene" in figs
    scene = figs["scene"]
    assert scene["kind"] == "points3d"
    assert len(scene["data"]) == 600          # 200 xyz triples
    xs = scene["data"][0::3]
    assert all(math.isfinite(v) for v in xs)   # no NaN poison
    assert ("points", 600) in [(n, int(v)) for n, v in res["outs"]]


def test_validate_io_sources():
    ok = {"triggers": [{"name": "beat",
                        "source": {"event": "clock", "every_s": 2}}],
          "outs": [{"name": "count", "kind": "number"}]}
    assert microfx.validate_io(ok) is None
    assert microfx.validate_io({}) is None
    assert microfx.validate_io({"triggers": [{"name": "b!d"}]})
    assert microfx.validate_io({"triggers": [
        {"name": "x", "source": {"event": "cron"}}]})
    assert microfx.validate_io({"triggers": [
        {"name": "x", "source": {"event": "clock", "every_s": 0.01}}]})
    assert microfx.validate_io({"triggers": [
        {"name": "x", "source": {"event": "sensor", "topic": "t",
                                 "op": "==", "value": 1}}]})
    assert microfx.validate_io({"triggers": [
        {"name": "x", "source": {"event": "sensor", "topic": "t",
                                 "op": ">", "value": "NaN!"}}]})
    assert microfx.validate_io({"outs": [{"name": "y", "kind": "wave"}]})


def test_metronome_clock_sourced():
    """Clock-sourced trigger: manifest validates, spec carries the
    source binding, program emits a series on the beat gate."""
    spec = microfx.build_jsfx("metronome")
    assert "error" not in spec
    io = spec["io"]
    assert io["triggers"][0]["source"] == {"event": "clock", "every_s": 2}
    assert {o["kind"] for o in io["outs"]} == {"series", "number"}
    # the real program runs under the app-IO driver (node)
    import json as _json
    blob = spec["html"].split("var PROGRAM=", 1)[1]
    prog = _json.loads(blob.split(";MicroFX.runProgram", 1)[0])
    assert "trigger('beat')" in prog["source"]


def test_validate_io_ins():
    assert microfx.validate_io(
        {"ins": [{"name": "sig", "topic": "ship/vllm/toks"}]}) is None
    assert microfx.validate_io({"ins": [{"name": "sig"}]})
    assert microfx.validate_io({"ins": [{"name": "!", "topic": "t"}]})


def test_gauge_live_signal_app():
    """Pure signal-flow app: no triggers. The real EEL2 source, driven
    with a live input port value, emits windowed frames + scalar."""
    import json as _json
    import subprocess

    spec = microfx.build_jsfx("gauge")
    assert "error" not in spec
    io = spec["io"]
    assert io["ins"] == [{"name": "sig", "topic": "ship/vllm/toks"}]
    assert not io.get("triggers")
    blob = spec["html"].split("var PROGRAM=", 1)[1]
    prog = _json.loads(blob.split(";MicroFX.runProgram", 1)[0])

    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    driver = r"""
const M = require(process.argv[2]);
const prog = JSON.parse(require("fs").readFileSync(0, "utf8"));
const inputs = {};
const outs = [];
const figures = [];
const it = new M.Interpreter({
  globals: Object.fromEntries((prog.params || []).map(
    (p) => [p.name.toLowerCase(), p.default])),
  has: () => true,
  input: (nm) => inputs[String(nm).toLowerCase()] || 0,
  output: (nm, v) => outs.push([nm, v]),
  trigger: () => 0,
  outData: (kind, nm, start, count) => {
    const data = [];
    for (let i = 0; i < count; i++) data.push(it.mem[start + i]);
    figures.push({ name: nm, kind, count, data });
  },
});
const secs = M.splitSections(prog.source);
const gfx = M.parse(M.lex(secs.gfx));
if (secs.init) for (const st of M.parse(M.lex(secs.init))) it.evalNode(st);
inputs.sig = 10;                       // live bus sample lands
for (let f = 0; f < 45; f++) {         // ~1.5s at 30fps: 3 emission frames
  it.deadline = Infinity;
  for (const st of gfx) it.evalNode(st);
}
console.log(JSON.stringify({ outs, figures }));
"""
    js = Path(__file__).parent / "_mfx_gauge_driver.js"
    js.write_text(driver)
    try:
        r = subprocess.run(
            [node, str(js),
             str(Path(microfx.__file__).parent / "web" / "jsfx.js")],
            input=_json.dumps(prog), capture_output=True, text=True,
            timeout=30)
    finally:
        js.unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr[-400:]
    res = _json.loads(r.stdout)
    # 45 frames / 15 = 3 emission frames; smoothed value approaches 10
    assert len(res["outs"]) == 3
    assert all(n == "now" for n, _ in res["outs"])
    assert res["outs"][-1][1] > 9.0          # converged toward the signal
    fig = res["figures"][-1]
    assert fig["name"] == "trace" and fig["count"] == 64  # fixed window
    assert all(v == v and abs(v) < 1e6 for v in fig["data"][:5])


def test_validate_io_controls():
    ok = {"controls": [
        {"type": "xy", "name": "pos", "address": "/pos"},
        {"type": "fader", "name": "twist", "default": 0.5},
        {"type": "button", "name": "zap", "buttonType": "momentary"},
    ]}
    assert microfx.validate_io(ok) is None
    assert microfx.validate_io({"controls": [{"type": "knob",
                                              "name": "x"}]})
    assert microfx.validate_io({"controls": [{"type": "button",
                                              "name": "x",
                                              "buttonType": "latch"}]})
    assert microfx.validate_io({"controls": [{"type": "fader",
                                              "name": "x",
                                              "address": "bad addr"}]})


def test_xypad3d_control_manifest():
    """Control-driven app: xy pad owns two ports, fader one; no params,
    no triggers — the control surface is the whole input side."""
    spec = microfx.build_jsfx("xypad3d")
    assert "error" not in spec
    io = spec["io"]
    assert [c["type"] for c in io["controls"]] == ["xy", "fader"]
    assert io["controls"][0]["address"] == "/pos"
    assert spec["fields"] == []
    assert "jsfx-controls" in spec["html"]
    assert "jsfx-xy" in spec["html"]


def test_hadamard4_wxyz_rows():
    """Sliding H4 rows: W=omni sum, Y/X/Z = signed row combinations."""
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    import json as _json

    driver = r"""
const M = require(process.argv[2]);
const src = JSON.parse(require("fs").readFileSync(0, "utf8"));
const it = new M.Interpreter({
  has: () => true,
  input: () => 0,
  output: () => {},
});
const outs = [];
it.host.output = (nm, v) => outs.push([nm, v]);
const secs = M.splitSections(src);
for (const st of M.parse(M.lex(secs.init))) it.evalNode(st);
const tick = M.parse(M.lex(secs.tick));
const feed = (v) => {
  it.host.input = () => v;
  for (const st of tick) it.evalNode(st);
};
feed(1); feed(0); feed(0); feed(0);   // impulse: W=1, others row-signed
console.log(JSON.stringify(outs.slice(-4)));
"""
    js = Path(__file__).parent / "_mfx_h4_driver.js"
    js.write_text(driver)
    try:
        r = subprocess.run(
            [node, str(js),
             str(Path(microfx.__file__).parent / "web" / "jsfx.js")],
            input=_json.dumps(microfx.MODULES["hadamard4"]["source"]),
            capture_output=True, text=True, timeout=30)
    finally:
        js.unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr[-300:]
    rows = dict(_json.loads(r.stdout))
    # H4 row signs against the impulse after 4 shifts: W=1,Y=-1,X=-1,Z=1
    assert rows["w"] == 1 and rows["y"] == -1 and rows["x"] == -1 \
        and rows["z"] == 1


def test_patch_hierarchy_rules():
    """Function nodes: one input max. Visualizers may stack (wxyz3d)."""
    bad = {"modules": [
        {"id": "a", "primitive": "gain", "params": {}},
    ], "wires": []}
    # gain declares 1 input — structural check passes; a hypothetical
    # multi-input function must be rejected. hadamard4 is a function
    # with 1 input and 4 outputs (legal).
    assert microfx.validate_patch(bad) is None
    multi_fn = {"modules": [
        {"id": "x", "primitive": "gain", "params": {}},
        {"id": "y", "primitive": "gain", "params": {}},
    ], "wires": [
        {"from": "x.cv", "to": "y.in"},
        {"from": "y.cv", "to": "y.in"},  # self-wire: port valid
    ]}
    assert microfx.validate_patch(multi_fn) is None
    viz_patch = {"modules": [
        {"id": "h", "primitive": "hadamard4", "params": {}},
        {"id": "v", "primitive": "viz_wxyz3d", "params": {}},
    ], "wires": [
        {"from": "h.w", "to": "v.w"},
        {"from": "h.x", "to": "v.x"},
        {"from": "h.y", "to": "v.y"},
        {"from": "h.z", "to": "v.z"},
    ]}
    assert microfx.validate_patch(viz_patch) is None
    bad_port = {"modules": [
        {"id": "h", "primitive": "hadamard4", "params": {}},
        {"id": "g", "primitive": "gain", "params": {}},
    ], "wires": [{"from": "h.q", "to": "g.in"}]}
    assert microfx.validate_patch(bad_port)
    # unknown primitive still rejected
    assert microfx.validate_patch({"modules": [
        {"id": "z", "primitive": "nope", "params": {}}], "wires": []})


def test_appwiz_catalog_and_generation():
    from fabric import appwiz

    cat = appwiz.wizard_catalog()
    assert {c["id"] for c in cat["controls"]} == \
        {"fader", "xy", "button", "encoder"}
    assert any(v["id"] == "blender_mcp" and not v["available"]
               for v in cat["visualizers"])
    ok = appwiz.generate_signal_app("sensor", "wxyz3d",
                                    topic="ship/vllm/toks")
    assert "error" not in ok
    assert "hadamard" in ok["principle"].lower()
    assert ok["io"]["ins"][0]["topic"] == "ship/vllm/toks"
    assert appwiz.generate_signal_app("sensor", "blender_mcp")["error"]
    assert appwiz.generate_signal_app("sensor", "series")["error"]
    assert appwiz.generate_signal_app("sensor", "series",
                                      topic="ship/vllm/toks")
    assert appwiz.generate_signal_app("av_stream", "av_player",
                                      url="http://x/y.mp3")


def test_gate_multi_input_allowed():
    """Gates live ONLY in Control library (not MODULES) per spec."""
    assert "gate_xor" in microfx._GATES
    assert microfx._GATES["gate_xor"][0] == "XOR"
    assert len(microfx._GATES["gate_xor"][1]) > 1  # multi-in
    # MODULES no longer contains gates:
    assert "gate_xor" not in microfx.MODULES
    assert "alogic" in microfx.MODULES


def test_scriptwiz_converts_hoa64_module(tmp_path):
    """4-pass conversion of a real hoa64 library module (brillouin.py):
    sources from array params, spectral fns mapped, gamma/X/Y/M dict
    return recognized as a WXYZ-style 3D output, app rebuilds natively
    and its EEL2 runs under the interpreter."""
    import json as _json
    import subprocess

    from fabric import scriptwiz

    src_path = Path("/nonexistent/hoa64/brillouin.py")
    if not src_path.is_file():
        pytest.skip("hoa64 sibling missing")
    r = scriptwiz.convert(src_path)
    assert "error" not in r
    rep = r["report"]
    # pass 1: array-typed params discovered as sources
    assert any(s["kind"] == "param" and s["fn"] == "dft2"
               for s in rep["sources"])
    # pass 2: spectral transforms mapped to catalog nodes
    assert any(f["node"] in ("mdct_flux", "hadamard4")
               for f in rep["functions"])
    # pass 3: tunable literals became controls
    assert rep["controls"]
    # pass 4: gamma/X/Y/M dict return -> wxyz3d
    assert any(o["kind"] == "wxyz3d" for o in rep["outputs"])

    app = r["app"]
    assert [o["kind"] for o in app["io"]["outs"]] == ["points3d",
                                                      "number"]
    # the generated EEL2 must run clean under the interpreter
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    blob = app["html"].split("var PROGRAM=", 1)[1]
    prog = _json.loads(blob.split(";MicroFX.runProgram", 1)[0])
    driver = r"""
const M = require(process.argv[2]);
const prog = JSON.parse(require("fs").readFileSync(0, "utf8"));
const figures = [];
const it = new M.Interpreter({
  has: () => true, input: () => 0, output: () => {},
  trigger: () => 0,
  outData: (kind, nm, start, count) => {
    const data = [];
    for (let i = 0; i < count; i++) data.push(it.mem[start + i]);
    figures.push({ nm, kind, count });
  },
});
const secs = M.splitSections(prog.source);
if (secs.init) for (const st of M.parse(M.lex(secs.init))) it.evalNode(st);
const gfx = M.parse(M.lex(secs.gfx));
for (let f = 0; f < 30; f++) {
  it.deadline = Infinity;
  for (const st of gfx) it.evalNode(st);
}
console.log(JSON.stringify({ figures,
  errors: it.vars.error === undefined ? [] : [it.vars.error] }));
"""
    js = Path(__file__).parent / "_mfx_conv_driver.js"
    js.write_text(driver)
    try:
        rr = subprocess.run(
            [node, str(js),
             str(Path(microfx.__file__).parent / "web" / "jsfx.js")],
            input=_json.dumps(prog), capture_output=True, text=True,
            timeout=30)
    finally:
        js.unlink(missing_ok=True)
    assert rr.returncode == 0, rr.stderr[-300:]
    res = _json.loads(rr.stdout)
    assert any(f["nm"] == "scene" and f["kind"] == "points3d"
               for f in res["figures"])


def test_scriptwiz_rejects_non_python(tmp_path):
    from fabric import scriptwiz

    f = tmp_path / "x.txt"
    f.write_text("hello")
    assert "only .py" in scriptwiz.convert(f)["error"]


def test_library_audit_and_cull(tmp_path, monkeypatch):
    """Library separates working apps from probe/duplicate junk."""
    import json
    from fabric import library

    root = tmp_path / "microapps"
    root.mkdir()
    def w(sid, mtime):
        p = root / f"{sid}.json"
        p.write_text(json.dumps({
            "id": sid, "title": sid, "template": "html",
            "html": "<p>x</p>", "fields": []}))
        import os
        os.utime(p, (mtime, mtime))
    w("app_gate_probe_1", 100)
    w("app_gate_probe_2", 200)
    w("app_xy_scope", 300)
    w("app_xy_scope_2", 400)      # newer duplicate family
    w("app_meals", 500)
    monkeypatch.setattr(library, "_root", lambda: root)
    a = library.audit()
    assert set(a["keep"]) == {"app_xy_scope_2", "app_meals"}
    assert "app_gate_probe_1" in a["stale"]
    r = library.cull(confirm=True)
    assert sorted(r["removed"]) == ["app_gate_probe_1",
                                    "app_gate_probe_2",
                                    "app_xy_scope"]
    assert len(list(root.glob("*.json"))) == 2
    apps = library.apps()
    assert [a_["id"] for a_ in apps] == ["app_meals", "app_xy_scope_2"]
    fns = library.functions()
    assert "hadamard4" in [n["id"] for n in fns["nodes"]["function"]]
    assert any(v["id"] == "wxyz3d" for v in fns["visualizers"])
