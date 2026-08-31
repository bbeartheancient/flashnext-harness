"""AppWizard — signal-path assembly from the four component categories.

Hierarchy (operator spec):
  SOURCES/DESTINATIONS  datastreams, files, sensor outputs, a/v streams
                        (both inputs and outputs live here)
  FUNCTIONS             modulate the sources; 1 source input, N outputs
  CONTROLS              modulate the functions (control standard)
  VISUALIZERS           preconfigured routing for common visualization
                        types; may stack multiple i/o paths (sinks)

A SIGNAL PATH = source -> function chain -> visualizer, plus a CONTROL
path (faders/pads) modulating the chain. The wizard generates the app
program from identified sources — synthetic or real (bus topics, a/v
urls). Canvas rule: vizUI shows controls, labels, and visualizer
outputs; signal wires appear only in edit mode.
"""

from __future__ import annotations

import json
import time

from .microfx import MODULES, jsfx_runtime, validate_io

# ---- catalog --------------------------------------------------------

SOURCES = [
    {"id": "sensor", "label": "Sensor bus topic", "kind": "real",
     "needs": "topic", "hint": "e.g. ship/vllm/toks"},
    {"id": "av_stream", "label": "A/V stream / file URL", "kind": "real",
     "needs": "url", "hint": "http(s) media url"},
    {"id": "sine", "label": "Sine oscillator", "kind": "synthetic"},
    {"id": "clock", "label": "Clock / pulse train", "kind": "synthetic"},
    {"id": "const", "label": "Constant level", "kind": "synthetic"},
]

FUNCTIONS = sorted(
    [n for n, m in MODULES.items() if m.get("category") == "function"])

CONTROLS = [
    {"id": "fader", "label": "Fader"},
    {"id": "xy", "label": "XY pad"},
    {"id": "button", "label": "Button"},
    {"id": "encoder", "label": "Encoder"},
]

VISUALIZERS = [
    {"id": "series", "label": "Chart / plot (rolling)", "available": True},
    {"id": "wxyz3d", "label": "3D WXYZ energy scope", "available": True},
    {"id": "xyscope", "label": "XY vectorscope", "available": True},
    {"id": "av_player", "label": "A/V player", "available": True},
    {"id": "dual_scope", "label": "Dual scope (series + 3D tiles)",
     "available": True},
    {"id": "gates", "label": "Gate chain chart", "available": True},
    # framebuffer display presets (shell-level figure kinds: lcd /
    # video / fb-frame seeds into the viewport framebuffer)
    {"id": "lcd_matrix", "label": "LCD matrix framebuffer",
     "available": False},
    {"id": "video_viewport", "label": "Video framebuffer",
     "available": False},
    {"id": "blender_mcp", "label": "Blender MCP scene",
     "available": False},
]

# ---- recipes: valid signal paths ------------------------------------
# key (source, visualizer) -> {functions, controls, title}

RECIPES: dict[tuple[str, str], dict] = {
    ("sensor", "series"): {
        "title": "Live bus chart",
        "functions": ["smooth", "moving_avg"],
        "controls": ["fader"],
    },
    ("sensor", "wxyz3d"): {
        "title": "Hadamard WXYZ 3D scope (bus-fed)",
        "functions": ["hadamard4"],
        "controls": ["fader"],
    },
    ("sine", "xyscope"): {
        "title": "Lissajous vectorscope (synthetic)",
        "functions": [],
        "controls": ["xy"],
    },
    ("sine", "series"): {
        "title": "Oscillator chart (synthetic)",
        "functions": [],
        "controls": ["fader"],
    },
    ("clock", "series"): {
        "title": "Beat counter chart",
        "functions": ["accum"],
        "controls": ["fader"],
    },
    ("av_stream", "av_player"): {
        "title": "A/V player",
        "functions": [],
        "controls": ["fader"],
    },
    ("sensor", "dual_scope"): {
        "title": "Dual scope (series + 3D, muxed bus)",
        "functions": ["hadamard4", "smooth"],
        "controls": ["fader"],
    },
    ("clock", "gates"): {
        "title": "Gate chain (XOR -> counter)",
        "functions": ["gate_xor", "accum"],
        "controls": ["fader"],
    },
}


def wizard_catalog() -> dict:
    """Four-category library for the AppWizard UI + LLM assembly."""
    return {
        "sources": SOURCES,
        "functions": [{"id": f, "label": MODULES[f]["title"]}
                      for f in FUNCTIONS],
        "controls": CONTROLS,
        "visualizers": VISUALIZERS,
        "recipes": [{"source": s, "visualizer": v,
                     "title": r["title"],
                     "functions": r["functions"],
                     "controls": r["controls"]}
                    for (s, v), r in sorted(RECIPES.items())],
    }


# ---- program builders ----------------------------------------------

def _package(name: str, title: str, principle: str, params: list,
             io: dict, source: str) -> dict:
    program = {"id": f"{name}-{int(time.time())}", "title": title,
               "params": params, "source": source, "io": io}
    inner = ("<div id=mfx></div>"
             "<script>" + jsfx_runtime() + "</script>"
             "<script>var PROGRAM=" + json.dumps(program) + ";"
             "MicroFX.runProgram(document.getElementById('mfx'),"
             "PROGRAM,{fps:30});</script>")
    html = ("<style>body{background:#0d0b12;color:#eba75a;"
            "font-family:Antonio,sans-serif;margin:0;padding:8px}"
            "#mfx canvas{width:100%;height:auto}"
            ".jsfx-controls{display:flex;gap:10px;align-items:center;"
            "flex-wrap:wrap;margin-top:6px}"
            ".jsfx-control{color:#baa4e5;font-size:11px;display:flex;"
            "flex-direction:column;gap:2px}"
            ".jsfx-control-btn{background:#ff7700;color:#000;border:0;"
            "border-radius:8px;padding:6px 16px;cursor:pointer;"
            "touch-action:none}"
            ".jsfx-control-btn.on{background:#fcc19f}"
            ".jsfx-xy{touch-action:none;border-radius:6px}</style>"
            "<h1 style='font-size:16px'>" + title + "</h1>"
            "<p class=principle style='font-size:11px;color:#baa4e5'>"
            + principle + "</p>" + inner)
    return {
        "html": html,
        "fields": [{"name": p["name"], "type": "number",
                    "value": p.get("default", 0)} for p in params],
        "io": io,
        "principle": principle,
        "template": "html",
        "domain": "tools",
        "group": "command",
        "span": 1,
        "viewport": "both",
        "kernel": "signal",
    }


def _sensor_series(title: str, topic: str) -> dict:
    io = {
        "ins": [{"name": "sig", "topic": topic}],
        "controls": [{"type": "fader", "name": "alpha",
                      "label": "smoothing", "min": 0.01, "max": 1,
                      "default": 0.2, "step": 0.01}],
        "outs": [{"name": "trace", "kind": "series"},
                 {"name": "now", "kind": "number"}],
    }
    src = ("@init\ny = 0; i = 0; fc = 0;\n\n"
           "@gfx\n"
           "y += alpha * (input('sig') - y);\n"
           "fc += 1;\n"
           "fc % 15 == 0 ? (\n"
           "  i = 0;\n"
           "  loop(63, mem[3000 + i] = mem[3001 + i]; i += 1;);\n"
           "  mem[3063] = y;\n"
           "  out_series('trace', 3000, 64);\n"
           "  output('now', y);\n"
           ");")
    return _package("sensor-series", title,
                    f"Signal path: bus {topic} -> smooth -> chart. "
                    "Control path: smoothing fader.",
                    [], io, src)


def _sensor_wxyz3d(title: str, topic: str) -> dict:
    io = {
        "ins": [{"name": "sig", "topic": topic}],
        "controls": [{"type": "fader", "name": "zoom",
                      "label": "zoom", "min": 0.1, "max": 20,
                      "default": 4, "step": 0.1}],
        "outs": [{"name": "scene", "kind": "points3d"},
                 {"name": "w", "kind": "number"}],
    }
    # hadamard4 sliding rows feed the WXYZ scope: row sums draw 3D
    # lissajous patterns from the stream (W omni, Y up/down,
    # X left/right, Z front/back)
    src = ("@init\ns0 = 0; s1 = 0; s2 = 0; fc = 0; i = 0;\n\n"
           "@gfx\n"
           "gfx_clear(0.05, 0.04, 0.08, 1);\n"
           "v = input('sig') * zoom;\n"
           "w = v + s0 + s1 + s2;\n"
           "yy = v - s0 + s1 - s2;\n"
           "xx = v + s0 - s1 - s2;\n"
           "zz = v - s0 - s1 + s2;\n"
           "s2 = s1; s1 = s0; s0 = v;\n"
           "fc += 1;\n"
           "fc % 6 == 0 ? (\n"
           "  i = 0;\n"
           "  loop(200,\n"
           "    t = i * 0.0314159;\n"
           "    mem[i * 3] = xx * sin(t + w * 0.05);\n"
           "    mem[i * 3 + 1] = yy * sin(t * 1.5 + w * 0.05);\n"
           "    mem[i * 3 + 2] = zz * sin(t * 0.5);\n"
           "    i += 1;\n"
           "  );\n"
           "  out_points3d('scene', 0, 600);\n"
           "  output('w', w);\n"
           ");")
    return _package("hadamard-wxyz3d", title,
                    "Signal path: bus -> Hadamard-4 rows (W/Y/X/Z) -> "
                    "3D energy scope. Control path: zoom fader.",
                    [], io, src)


def _sine_xyscope(title: str) -> dict:
    io = {
        "controls": [
            {"type": "xy", "name": "ratio", "address": "/ratio"},
            {"type": "fader", "name": "phase", "label": "phase",
             "min": 0, "max": 6.28, "default": 0.5, "step": 0.01},
        ],
        "outs": [{"name": "scene", "kind": "points3d"},
                 {"name": "freqs", "kind": "number"}],
    }
    src = ("@init\nfc = 0; i = 0;\n\n"
           "@gfx\n"
           "gfx_clear(0.05, 0.04, 0.08, 1);\n"
           "fc += 1;\n"
           "fc % 6 == 0 ? (\n"
           "  fx = 1 + input('ratio_x') * 7;\n"
           "  fy = 1 + input('ratio_y') * 7;\n"
           "  ph = input('phase');\n"
           "  i = 0;\n"
           "  loop(200,\n"
           "    t = i * 0.0314159;\n"
           "    mem[i * 3] = sin(fx * t + ph);\n"
           "    mem[i * 3 + 1] = sin(fy * t);\n"
           "    mem[i * 3 + 2] = 0;\n"
           "    i += 1;\n"
           "  );\n"
           "  out_points3d('scene', 0, 600);\n"
           "  output('freqs', floor(fx) + floor(fy) / 10);\n"
           ");")
    return _package("sine-xyscope", title,
                    "Signal path: sine pair -> XY vectorscope. "
                    "Control path: ratio pad + phase fader.",
                    [], io, src)


def _sine_series(title: str) -> dict:
    io = {
        "controls": [{"type": "fader", "name": "rate", "label": "rate",
                      "min": 0.1, "max": 8, "default": 1, "step": 0.1}],
        "outs": [{"name": "trace", "kind": "series"},
                 {"name": "v", "kind": "number"}],
    }
    src = ("@init\nph = 0; i = 0; fc = 0;\n\n"
           "@gfx\n"
           "dt = 1 / 30;\n"
           "ph += rate * dt;\n"
           "v = sin(ph * 6.2832);\n"
           "fc += 1;\n"
           "fc % 5 == 0 ? (\n"
           "  i = 0;\n"
           "  loop(63, mem[3000 + i] = mem[3001 + i]; i += 1;);\n"
           "  mem[3063] = v;\n"
           "  out_series('trace', 3000, 64);\n"
           "  output('v', v);\n"
           ");")
    return _package("sine-series", title,
                    "Signal path: sine -> chart. Control path: "
                    "rate fader.", [], io, src)


def _clock_series(title: str) -> dict:
    io = {
        "controls": [{"type": "fader", "name": "bpm", "min": 20,
                      "max": 300, "default": 60, "step": 1}],
        "outs": [{"name": "beats", "kind": "series"},
                 {"name": "count", "kind": "number"}],
    }
    src = ("@init\ncnt = 0; acc = 0; i = 0; fc = 0;\n\n"
           "@gfx\n"
           "dt = 1 / 30;\n"
           "acc += dt;\n"
           "period = 60 / max(1, bpm);\n"
           "beat = 0;\n"
           "acc >= period ? (acc -= period; beat = 1; cnt += 1;);\n"
           "fc += 1;\n"
           "(beat == 1 || fc % 15 == 0) ? (\n"
           "  i = 0;\n"
           "  loop(63, mem[3000 + i] = mem[3001 + i]; i += 1;);\n"
           "  mem[3063] = beat;\n"
           "  out_series('beats', 3000, 64);\n"
           "  output('count', cnt);\n"
           ");")
    return _package("clock-series", title,
                    "Signal path: clock -> edge counter -> chart. "
                    "Control path: bpm fader.", [], io, src)


def _av_player(title: str, url: str) -> dict:
    safe = (url or "").replace('"', "%22").replace("<", "%3C") \
        .replace(">", "%3E")
    html = ("<style>body{background:#0d0b12;color:#eba75a;"
            "font-family:Antonio,sans-serif;margin:0;padding:8px}"
            "audio{width:100%;margin-top:10px}</style>"
            "<h1 style='font-size:16px'>" + title + "</h1>"
            "<p class=principle style='font-size:11px;color:#baa4e5'>"
            "Destination: A/V stream -> device output.</p>"
            "<audio controls src=\"" + safe + "\"></audio>")
    return {
        "html": html,
        "fields": [{"name": "url", "type": "text", "value": url or ""}],
        "io": {},
        "principle": "A/V destination: stream -> player -> device.",
        "template": "html",
        "domain": "media",
        "group": "command",
        "span": 1,
        "viewport": "both",
        "kernel": "signal",
    }


def _sensor_dual(title: str, topic: str) -> dict:
    """Tile subfunctions: ONE app, TWO inner chains (series chart +
    hadamard 3D scope) sharing the muxed bus input -> two tiles."""
    io = {
        "ins": [{"name": "sig", "topic": topic}],
        "controls": [{"type": "fader", "name": "zoom",
                      "label": "zoom", "min": 0.1, "max": 20,
                      "default": 4, "step": 0.1}],
        "outs": [{"name": "trace", "kind": "series"},
                 {"name": "scene", "kind": "points3d"},
                 {"name": "now", "kind": "number"}],
        "tiles": [{"name": "trace", "kind": "series",
                   "functions": ["smooth"]},
                  {"name": "scene", "kind": "points3d",
                   "functions": ["hadamard4"]}],
    }
    src = ("@init\ny = 0; i = 0; fc = 0; s0 = 0; s1 = 0; s2 = 0;\n\n"
           "@gfx\n"
           "v = input('sig') * zoom;\n"
           "y += alpha_default * (v - y);\n"
           "w = v + s0 + s1 + s2;\n"
           "yy = v - s0 + s1 - s2;\n"
           "xx = v + s0 - s1 - s2;\n"
           "zz = v - s0 - s1 + s2;\n"
           "s2 = s1; s1 = s0; s0 = v;\n"
           "fc += 1;\n"
           "fc % 15 == 0 ? (\n"
           "  i = 0;\n"
           "  loop(63, mem[3000 + i] = mem[3001 + i]; i += 1;);\n"
           "  mem[3063] = y;\n"
           "  out_series('trace', 3000, 64);\n"
           "  output('now', y);\n"
           ");\n"
           "fc % 6 == 0 ? (\n"
           "  i = 0;\n"
           "  loop(200,\n"
           "    t = i * 0.0314159;\n"
           "    mem[i * 3] = xx * sin(t + w * 0.05);\n"
           "    mem[i * 3 + 1] = yy * sin(t * 1.5 + w * 0.05);\n"
           "    mem[i * 3 + 2] = zz * sin(t * 0.5);\n"
           "    i += 1;\n"
           "  );\n"
           "  out_points3d('scene', 0, 600);\n"
           ");")
    # alpha_default: use a literal 0.2 (zoom is the control)
    src = src.replace("alpha_default", "0.2")
    return _package("sensor-dual", title,
                    "Muxed bus input feeds two tile subfunctions: "
                    "smooth -> chart tile, Hadamard-4 -> 3D tile. "
                    "Control path: zoom fader.", [], io, src)


def _clock_gates(title: str) -> dict:
    io = {
        "controls": [{"type": "fader", "name": "bpm", "min": 20,
                      "max": 300, "default": 60, "step": 1},
                     {"type": "fader", "name": "gate_b", "label": "b",
                      "min": 0, "max": 1, "default": 1, "step": 1}],
        "outs": [{"name": "beats", "kind": "series"},
                 {"name": "count", "kind": "number"}],
        "tiles": [{"name": "beats", "kind": "series",
                   "functions": ["gate_xor", "accum"]}],
    }
    src = ("@init\ncnt = 0; acc = 0; i = 0; fc = 0; ph = 0; prev = 0;"
           "\n\n"
           "@gfx\n"
           "dt = 1 / 30;\n"
           "acc += dt;\n"
           "period = 60 / max(1, bpm);\n"
           "a = 0;\n"
           "acc >= period ? (acc -= period; a = 1;);\n"
           "ph += dt * 2;\n"
           "b = sin(ph) > 0 ? 1 : 0;\n"
           # XOR of the two square waves drives the counter (real gates)
           "x = (a > 0.5) != (gate_b > 0.5 && b > 0.5) ? 1 : 0;\n"
           "x > prev ? cnt += 1 : 0;\n"
           "prev = x;\n"
           "fc += 1;\n"
           "fc % 15 == 0 || x == 1 ? (\n"
           "  i = 0;\n"
           "  loop(63, mem[3000 + i] = mem[3001 + i]; i += 1;);\n"
           "  mem[3063] = x;\n"
           "  out_series('beats', 3000, 64);\n"
           "  output('count', cnt);\n"
           ");")
    return _package("clock-gates", title,
                    "Apps can be built entirely from logic gates: XOR "
                    "of two clocks -> edge counter -> chart. Control "
                    "path: bpm + gate-b faders.", [], io, src)


_BUILDERS = {
    ("sensor", "dual_scope"): _sensor_dual,
    ("clock", "gates"): _clock_gates,
    ("sensor", "series"): _sensor_series,
    ("sensor", "wxyz3d"): _sensor_wxyz3d,
    ("sine", "xyscope"): _sine_xyscope,
    ("sine", "series"): _sine_series,
    ("clock", "series"): _clock_series,
    ("av_stream", "av_player"): _av_player,
}


def generate_signal_app(source: str, visualizer: str,
                        title: str | None = None,
                        topic: str | None = None,
                        url: str | None = None) -> dict:
    """Generate an app from a signal-path selection. Returns a spec
    dict (same shape as microfx.build_jsfx) or {"error": ...}."""
    source = (source or "").strip()
    visualizer = (visualizer or "").strip()
    if source not in {s["id"] for s in SOURCES}:
        return {"error": f"unknown source '{source}'"}
    viz = next((v for v in VISUALIZERS if v["id"] == visualizer), None)
    if viz is None:
        return {"error": f"unknown visualizer '{visualizer}'"}
    if not viz.get("available"):
        return {"error": f"visualizer '{visualizer}' is not wired yet"}
    recipe = RECIPES.get((source, visualizer))
    if recipe is None:
        valid = [f"{s}->{v}" for (s, v) in sorted(RECIPES)]
        return {"error": f"no signal path {source}->{visualizer}; "
                         f"valid: {', '.join(valid)}"}
    if source == "sensor" and not (topic or "").strip():
        return {"error": "sensor source needs a bus topic"}
    if source == "av_stream" and not (url or "").strip():
        return {"error": "av_stream source needs a url"}
    t = title or recipe["title"]
    if source == "av_stream":
        return _av_player(t, url or "")
    builder = _BUILDERS[(source, visualizer)]
    if source == "sensor":
        return builder(t, topic.strip())
    return builder(t)
