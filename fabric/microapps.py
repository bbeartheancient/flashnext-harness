"""MiniApp registry — compose, gate, persist.

Generated functions are catalog entries whose center tile is a MiniAppBench
MiniApp: HTML as world state, CSS as salience, JS as causal/temporal logic
(arxiv:2603.09652). They bind to the locked LCARS I/O (command / left /
right / center) and to the existing tool/library inventory. No arbitrary
host Python, no remote scripts, no CircuitJS (D19 stays concept).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

GENERATE_RE = re.compile(
    r"\b(?:make|build|create|generate|spin up|new)\b.{0,48}\b(?:"
    r"app|mini-?app|micro-?app|function|tracker|calculator|sim(?:ulator)?|"
    r"game|timer|counter|widget|dashboard|visuali[sz]er|converter)\b"
    r"|\binteractive\b.{0,40}\b(?:app|tool|sim|tracker|game)",
    re.I,
)

PLACE_RE = re.compile(
    r"\b(?:show(?:\s+me)?|map(?:\s+of)?|look at|where is|"
    r"terrain(?:\s+(?:of|at|for))?|city of|town of|dem of)\b",
    re.I,
)

_PLACE_SKIP = (
    "status", "hadamard", "yagi", "filter", "log", "habit", "bmi",
    "timer", "sage", "chat",
)

_PLACE_STRIP = re.compile(
    r"^(?:the\s+)?(?:city|town|map|terrain|dem)(?:\s+of)?\s+",
    re.I,
)

_REMOTE = re.compile(
    r"""<script[^>]+src\s*=\s*['"]https?://"""
    r"""|\bfetch\s*\(\s*['"]https?://"""
    r"""|<iframe\b"""
    r"""|document\.cookie""",
    re.I,
)

_WORD = re.compile(r"[a-z0-9]+")

# Display templates the wall already knows how to host.
TEMPLATES = [
    {"id": "html", "tile": "iframe", "span": 1,
     "when": "interactive MiniApp (MiniAppBench default)"},
    {"id": "canvas", "tile": "canvas", "span": 1,
     "when": "bound library returns heatmap/polar/xy/matrix/bars"},
    {"id": "text", "tile": "pre", "span": 1,
     "when": "CAS / chat / logs"},
    {"id": "square", "tile": "canvas|iframe", "span": 1,
     "when": "one cell on the 4×4 wall"},
    {"id": "wide", "tile": "canvas|iframe", "span": 2,
     "when": "xy / time series"},
    {"id": "solo", "tile": "iframe", "span": 4,
     "when": "one MiniApp fills the wall"},
]

# Locked chrome — do not invent a new rail.
IO_POSITIONS = [
    {"id": "command", "where": "bottom command bar, aligned with Communications",
     "kind": "query; procedurally generates the box"},
    {"id": "left", "where": "150px left rail (viz max / split box)",
     "kind": "number|text fields + Apply"},
    {"id": "right", "where": "viz right rail",
     "kind": "select options only"},
    {"id": "center", "where": "4×4 video wall tile",
     "kind": "output (canvas or sandboxed iframe)"},
    {"id": "term", "where": "terminal viewport",
     "kind": "log line; never the only home for a MiniApp"},
]

# Host libraries a MiniApp may bind. Names match fabric.tools.TOOLS.
LIBRARIES = [
    {"id": "ship_status", "group": "command", "url": "/ship_status"},
    {"id": "log_recent", "group": "command", "url": "/log/recent"},
    {"id": "hadamard_build", "group": "science", "url": "/api/lab/hadamard"},
    {"id": "hoa_encode", "group": "science", "url": "/api/lab/hoa"},
    {"id": "hoa_decode", "group": "science", "url": "/api/lab/hoa/decode"},
    {"id": "hoa_rotate", "group": "science", "url": "/api/lab/hoa/rotate"},
    {"id": "orbital_probe", "group": "science", "url": "/api/lab/orbital"},
    {"id": "antenna_pattern", "group": "science", "url": "/api/lab/antenna"},
    {"id": "filter_response", "group": "science", "url": "/api/lab/filter"},
    {"id": "link_budget", "group": "science", "url": "/api/lab/link"},
    {"id": "fdtd_probe", "group": "science", "url": "/api/lab/fdtd"},
    {"id": "materials_design", "group": "science", "url": "/api/lab/materials"},
    {"id": "actual_size_scales", "group": "science", "url": "/api/lab/scales"},
    {"id": "crown_psf", "group": "science", "url": "/api/lab/crown"},
    {"id": "sage_eval", "group": "science", "url": "/api/sage"},
    {"id": "show_place", "group": "science", "url": "/api/geo/place",
     "when": "named city / map / DEM slice"},
    {"id": "terrain_set_origin", "group": "science", "url": "/api/geo/origin"},
    {"id": "terrain_survey", "group": "science", "url": "/api/geo/survey"},
    {"id": "terrain_los", "group": "science", "url": "/api/geo/los/latlon"},
    {"id": "terrain_horizon", "group": "science", "url": "/api/geo/horizon"},
    {"id": "room_impulse", "group": "science", "url": "/api/room/ir"},
    {"id": "sensor_query", "group": "science", "url": "/api/sensors"},
]

DOMAINS = ("science", "games", "tools", "visualization", "humanities", "lifestyle")

# Wizard enums. Compose may only emit a row from ROUTES (or a labeled stub).
INPUTS = [
    {"id": "text", "wired": True, "label": "text"},
    {"id": "parameters", "wired": True, "label": "parameters"},
    {"id": "place", "wired": True, "label": "place"},
    {"id": "mqtt", "wired": True, "label": "mqtt"},
    {"id": "video", "wired": False, "label": "video"},
]
OUTPUTS = [
    {"id": "heatmap", "wired": True, "tile": "canvas"},
    {"id": "polar", "wired": True, "tile": "canvas"},
    {"id": "xy", "wired": True, "tile": "canvas"},
    {"id": "bars", "wired": True, "tile": "canvas"},
    {"id": "text", "wired": True, "tile": "pre"},
    {"id": "html", "wired": True, "tile": "iframe"},
    {"id": "webpage", "wired": True, "tile": "iframe"},
    {"id": "mqtt", "wired": True, "tile": "pre"},
    {"id": "3d", "wired": True, "tile": "canvas",
     "note": "DEM/orbital slice — no new 3D engine"},
    {"id": "media", "wired": False, "tile": "standby"},
]
VIEWPORTS = [
    {"id": 1, "span": 1, "label": "1 cell"},
    {"id": 4, "span": 2, "label": "4 cells"},
    {"id": 16, "span": 4, "label": "16 cells"},
]
ROUTES = [
    {"dept": "science", "input": "place", "output": "heatmap", "bind": "place"},
    {"dept": "science", "input": "place", "output": "3d", "bind": "place"},
    {"dept": "science", "input": "place", "output": "polar", "bind": "place",
     "fields": {"view": "horizon"}},
    {"dept": "science", "input": "parameters", "output": "heatmap",
     "bind": "hadamard",
     "alts": (("orbital", ("orbit", "psi", "hydrogen")),
              ("materials", ("material", "cloth")),
              ("crown", ("crown", "psf")))},
    {"dept": "science", "input": "parameters", "output": "3d", "bind": "orbital"},
    {"dept": "science", "input": "parameters", "output": "polar", "bind": "antenna"},
    {"dept": "science", "input": "parameters", "output": "xy", "bind": "survey"},
    {"dept": "science", "input": "parameters", "output": "text", "bind": "link"},
    {"dept": "science", "input": "text", "output": "text", "bind": "sage"},
    {"dept": "science", "input": "mqtt", "output": "text", "bind": "sensors"},
    {"dept": "science", "input": "mqtt", "output": "bars", "bind": "eng_ship"},
    {"dept": "command", "input": "text", "output": "html", "kernel": "auto"},
    {"dept": "command", "input": "text", "output": "webpage", "kernel": "auto"},
    {"dept": "command", "input": "text", "output": "text", "bind": "chat"},
    {"dept": "command", "input": "mqtt", "output": "bars", "bind": "ship"},
    {"dept": "command", "input": "mqtt", "output": "text", "bind": "log"},
    {"dept": "command", "input": "parameters", "output": "html", "kernel": "auto"},
    {"dept": "flight", "input": "video", "output": "media", "stub": True},
    {"dept": "holodeck", "input": "video", "output": "media", "stub": True},
    {"dept": "operations", "input": "text", "output": "media", "stub": True},
    {"dept": "security", "input": "video", "output": "media", "stub": True},
    {"dept": "communications", "input": "text", "output": "text", "bind": "chat"},
    {"dept": "medical", "input": "text", "output": "html", "kernel": "auto"},
]

_MINI_CSS = """
:root{--ink:#fcc19f;--hi:#eb943a;--ok:#99f;--bg:#000;--btn:#baa4e5}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
  font:15px Antonio,sans-serif;letter-spacing:.04em}
#stage{display:flex;flex-direction:column;gap:8px;height:100%;padding:10px;box-sizing:border-box}
h1{margin:0;color:var(--hi);font-size:1rem;text-transform:uppercase;letter-spacing:.16em}
.principle{margin:0;color:var(--ok);font-size:.78rem}
button,.day{background:var(--btn);color:#000;border:0;border-radius:12px 0;
  padding:.45rem .7rem;font:inherit;text-transform:uppercase;cursor:default}
button.on,.day.on{background:var(--hi)}
#row{display:flex;flex-wrap:wrap;gap:6px}
#habit{margin:0;color:var(--hi);letter-spacing:.14em;text-transform:uppercase}
#out,#vals{margin:0;flex:1;overflow:auto;white-space:pre-wrap}
input,textarea{background:#111;color:var(--ink);border:1px solid #333;font:inherit;padding:.3rem}
"""


def looks_like_generate(q: str) -> bool:
    return bool(GENERATE_RE.search(q or ""))


def looks_like_place(q: str) -> bool:
    q = q or ""
    if not PLACE_RE.search(q):
        return False
    low = q.lower()
    return not any(w in low for w in _PLACE_SKIP)


def extract_place(q: str) -> str:
    q = (q or "").strip()
    m = re.search(
        r"(?:city of|town of|map of|terrain of|dem of|"
        r"show(?:\s+me)?(?:\s+the)?|look at|where is)\s+(.+)$",
        q, re.I)
    raw = m.group(1).strip() if m else q
    raw = _PLACE_STRIP.sub("", raw).strip(" .,")
    return raw or q


def kit() -> dict:
    return {
        "templates": TEMPLATES,
        "io": IO_POSITIONS,
        "libraries": LIBRARIES,
        "domains": list(DOMAINS),
        "eval": ["intention", "static", "dynamic", "left_io"],
        "rules": [
            "HTML+JS is the world model; do not caption a principle — instantiate it.",
            "Left rail = number/text; right rail = selects; center = tile; command = query.",
            "Bind an existing library instead of reimplementing hoa64/RTU/sage.",
            "Named places bind show_place (Terrarium DEM + heatmap/horizon).",
            "If Intention is weak, return clarify choices — do not invent a caption tile.",
            "No remote scripts, no host exec, no CircuitJS, no micromag/gerzon search.",
        ],
        "kernels": ["week", "bmi", "timer", "notes", "counter", "calc"],
        "inputs": INPUTS,
        "outputs": OUTPUTS,
        "viewports": VIEWPORTS,
        "routes": ROUTES,
        "departments": _dept_summaries(),
    }


def _dept_summaries() -> list[dict]:
    from . import departments

    return [{"id": d["id"], "name": d["name"], "ext": d["ext"],
             "wired": bool(d.get("wired")), "note": d.get("note")}
            for d in departments.DEPARTMENTS]


def viewport_span(n) -> int:
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 1
    if n >= 16:
        return 4
    if n >= 4:
        return 2
    return 1


def match_route(dept: str, inp: str, out: str, query: str = "") -> dict | None:
    dept = (dept or "command").lower()
    inp = (inp or "text").lower()
    out = (out or "html").lower()
    low = (query or "").lower()
    for raw in ROUTES:
        if raw["dept"] != dept or raw["input"] != inp or raw["output"] != out:
            continue
        rec = dict(raw)
        for alt_id, words in rec.pop("alts", ()) or ():
            if any(w in low for w in words):
                rec["bind"] = alt_id
                break
        return rec
    return None


def allowed_outputs(dept: str, inp: str) -> list[str]:
    dept = (dept or "").lower()
    inp = (inp or "").lower()
    return [r["output"] for r in ROUTES if r["dept"] == dept and r["input"] == inp]


GENERATION_DOCS = [
    {"title": "Patch format",
     "text": "A MicroFX patch is JSON: {modules:[{id,primitive,params}], "
             "wires:[{from:'mod.out', to:'mod.in'}], views:[{module, "
             "output:'cv', as:'series'|'number'}]}. Modules evaluate at "
             "30 Hz control rate; wires copy output values into input "
             "bags each tick."},
    {"title": "Module catalog",
     "text": "const(value), clock_bpm(bpm)->trig, sine_lfo(rate_hz,amp,"
             "offset)->cv, gain(in,factor), bias(in,add), smooth(in,alpha) "
             "one-pole, threshold(in,lo,hi)->gate with hysteresis, "
             "moving_avg(in,n), clamp(in,lo,hi), mdct_flux(in)->flux "
             "(64-sample window, spectral energy delta via mdct). "
             "sensor(topic) is host-fed from /api/sensors."},
    {"title": "EEL2 tick language",
     "text": "Module bodies are EEL2: undeclared doubles, ?: blocks in "
             "( ), loop(count,body)/while(cond)(body), mem[1M] scratch, "
             "math builtins, input('name')/output('name',val), store/"
             "load for persistent state. Params bind as lowercase "
             "globals. @init runs once then @tick every frame."},
    {"title": "AppWizard signal paths",
     "text": "Assembly hierarchy: SOURCES/DESTINATIONS (bus topics, "
             "a/v urls, synthetic oscillators/clocks), FUNCTIONS (1 "
             "source input, N outputs; smooth, moving_avg, gain, "
             "threshold, hadamard4 WXYZ rows, accum, mdct_flux...), "
             "CONTROLS (fader/xy/button/encoder modulating functions), "
             "VISUALIZERS (sinks; may stack multiple inputs - series "
             "chart, wxyz3d 3D energy scope, xyscope, av_player; "
             "blender_mcp planned). Two signal paths: SIGNAL (left "
             "rail -> function chain -> right rail) and CONTROL "
             "(modulates). Generate via POST /api/wizard/generate "
             "{source, visualizer, topic?, url?}; catalog via GET "
             "/api/wizard/catalog. Valid recipes: sensor->series, "
             "sensor->wxyz3d, sine->xyscope, sine->series, "
             "clock->series, av_stream->av_player. Hadamard-4 rows map "
             "W=omni, Y=up/down, X=left/right, Z=front/back; row sums "
             "draw 3D lissajous patterns from the stream."},
    {"title": "Control standard (TouchOSC-derived)",
     "text": "io.controls declare user-facing signal sources rendered "
             "on the app's control surface: {type:'fader',name,min,max,"
             "default} -> port name (normalized 0..1 scaled to min..max); "
             "{type:'button',name,buttonType:'momentary'|'toggle'} -> "
             "0/1 port (momentary held while pressed; pairs with "
             "trigger('name') edge gates); {type:'xy',name} -> TWO "
             "ports name_x/name_y (0..1); {type:'encoder',name,min,max,"
             "default} -> relative drag, clamped. Optional OSC-style "
             "address:'/path'. Controls are live signals - same "
             "currency as bus inputs and triggers; apps read them with "
             "input('name') every frame. Reference patterns: "
             "~/TouchOSC examples (master_fader_mix, bpm_tap, "
             "fader_incdec) + hexler.net scripting-api."},
    {"title": "App-level IO (live signals)",
     "text": "Standalone apps declare an io block: io:{ins:[{name,"
             "topic}], triggers:[{name,label,source}], outs:[{name,"
             "kind}]}. UNIFORM SIGNAL MODEL: every port is a live "
             "captured signal sampled relative to the current frame - "
             "never one-shot requests, never batch renders. ins are "
             "continuously-sampled bus inputs (shell feeds the latest "
             "topic value; app reads input('name') every frame). "
             "triggers gate on their port via trigger('name') (rising "
             "edge); sources: 'manual' button (one-shot for plain "
             "triggers, GATE toggle for clock sources - the clock runs "
             "persistently and passes through while the gate is open), "
             "{event:'clock',every_s}, {event:'sensor',topic,op,value} "
             "threshold crossing, {event:'app',app,out}. outs emit the "
             "current frame: output('name',v) scalars; out_series("
             "'trace',start,count) fixed-length shifting windows "
             "(ring-buffer, always a complete frame); out_points3d("
             "'scene',start,count) scene snapshots for the 3D viewport. "
             "Figures route to the shell viewport matrix (auto-route "
             "toggle on the right rail). "
             "Framebuffer presets: viewport tiles are framebuffers at "
             "their fraction of full resolution - figure kinds "
             "{type:'lcd',grid} render 1-bit green-on-black matrices, "
             "{type:'video',src} seeds a media file into the tile, "
             "and apps can push raw frames as ImageBitmap via the "
             "app-figure channel (kind 'fb-frame'). Clicking a "
             "visualizer tile expands it to fill the display area "
             "(solo; click again restores)."},
    {"title": "mdct/imdct spec",
     "text": "mdct(start,size): size samples in place -> size/2 coeffs "
             "at start (1/sqrt2 normalized). imdct(start,size): size/2 "
             "coeffs -> sqrt2-compensated windowed samples; caller "
             "overlap-adds frames shifted by half window. Sizes: 64|128|"
             "256|512|1024|2048|4096. Never cross a 65536-item boundary."},
    {"title": "Wiring rules",
     "text": "Every wire endpoint is 'module.port'. Source modules have "
             "no inputs; sinks are views. Prefer few small modules over "
             "one big body. Visualize at most two series per patch. "
             "Params surface as left-rail fields automatically - never "
             "hand-build sliders."},
    {"title": "Gate rules",
     "text": "validate_patch rejects unknown primitives, duplicate ids, "
             "malformed wires. Generated HTML must stay under 40k chars "
             "with no remote scripts. Patches must produce finite values "
             "within 150 ticks (no NaN/loop-deadline)."},
]


def _ensure_generation_shard() -> None:
    try:
        from .dept_memory import seed_docs

        seed_docs("Microapps", GENERATION_DOCS)
    except Exception:
        pass


def generation_reference(query: str, k: int = 3) -> str:
    """Bounded authoring-reference block for procedural generation.

    Curated docs ranked by query-term overlap; memvid recall appended
    when it adds something the curated set missed."""
    qtoks = {t for t in re.findall(r"[a-z0-9]+", (query or "").lower())
             if len(t) > 2}
    _ensure_generation_shard()

    def score(text: str) -> int:
        hay = set(re.findall(r"[a-z0-9]+", text.lower()))
        return len(qtoks & hay)

    ranked = sorted(GENERATION_DOCS,
                    key=lambda d: -score(d.get("title", "") + " "
                                         + d.get("text", "")))
    picked = ranked[: max(1, min(k, len(ranked)))]
    lines = []
    used = 0
    for doc in picked:
        line = "- " + doc["title"] + ": " + \
            " ".join(doc["text"].split())[:350]
        if used + len(line) > 900:
            break
        lines.append(line)
        used += len(line)
    try:
        from .dept_memory import search as dm_search

        out = dm_search("Microapps", query or "MicroFX EEL2", k=2)
        seen = {d["title"] for d in picked}
        for hit in out.get("hits") or []:
            t = (hit.get("title") or "").strip()
            if t and t not in seen:
                lines.append("- " + t + ": "
                             + " ".join((hit.get("text") or "").split())[:250])
                break
    except Exception:
        pass
    return "\n".join(lines)


def skill_prompt() -> str:
    ref = generation_reference("MicroFX EEL2 primitives authoring")
    block = "\nAuthoring reference (MicroFX plugin primitives):\n" + ref \
        if ref else ""
    return (
        "MiniApp skill loaded. If the operator wants a new interactive tool, "
        "tracker, calculator, game, or visualizer that is not already a catalog "
        "function, call generate_microapp with the original query. Do not write "
        "HTML unless they pasted some — the tool composes from templates. "
        "Named places (city/map/terrain of X) belong to show_place, not a new MiniApp. "
        "If the tool returns clarify=true, list the choices in one line and stop. "
        "I/O map: command line generates the box; left = number/text; right = selects; "
        "center = sandboxed MiniApp tile. After a spec returns, one sentence: "
        "title + that it is on the wall. Do not dump markup in the spoken reply."
        + block
    )


def _root() -> Path:
    env = os.environ.get("FABRIC_MICROAPP_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    return here / "data" / "microapps"


def reset_registry() -> None:
    """Test helper: next list/get re-reads FABRIC_MICROAPP_DIR."""
    _cache.clear()


_cache: dict[str, dict] = {}


def _load_all() -> dict[str, dict]:
    d = _root()
    found: dict[str, dict] = {}
    if d.is_dir():
        for p in d.glob("*.json"):
            try:
                spec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(spec, dict) and spec.get("id"):
                found[spec["id"]] = spec
    _cache.clear()
    _cache.update(found)
    return _cache


def _ensure_loaded() -> dict[str, dict]:
    if _cache:
        return _cache
    return _load_all()


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (title or "app").lower()).strip("_")
    return (s or "app")[:40]


def _unique_id(title: str) -> str:
    base = "app_" + _slug(title)
    have = set(_ensure_loaded())
    from . import console

    have.update(f["id"] for f in console.FUNCTIONS)
    if base not in have:
        return base
    for i in range(2, 50):
        cand = f"{base}_{i}"
        if cand not in have:
            return cand
    return base + "_x"


def infer_title(query: str, explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()[:48]
    m = re.search(
        r"(?:make|build|create|generate|spin up|new)\s+(?:an?\s+)?(.+?)$",
        query.strip(), re.I)
    if m:
        t = re.sub(r"^(interactive\s+)", "", m.group(1).strip(), flags=re.I)
        t = re.sub(r"[\.!?]\s*$", "", t)
        return t[:48] or "MiniApp"
    return (query.strip()[:48] or "MiniApp")


def infer_domain(query: str) -> str:
    low = query.lower()
    if any(w in low for w in ("game", "puzzle", "score", "play")):
        return "games"
    if any(w in low for w in ("habit", "todo", "timer", "bmi", "diet", "week",
                              "day", "lifestyle")):
        return "lifestyle"
    if any(w in low for w in ("plot", "chart", "graph", "visual")):
        return "visualization"
    if any(w in low for w in ("history", "poem", "latin", "literature")):
        return "humanities"
    if any(w in low for w in ("hadamard", "hoa", "orbit", "antenna", "filter",
                              "sage", "terrain", "fdtd")):
        return "science"
    return "tools"


def propose(query: str) -> dict:
    """Rank kernels + libraries for a query. Auto if one winner, else clarify."""
    q = (query or "").strip()
    low = q.lower()
    toks = set(_WORD.findall(low))
    stop = {"a", "an", "the", "to", "for", "of", "and", "or", "me", "show",
            "make", "build", "create", "new", "app", "city", "town"}
    toks -= stop
    choices: list[dict] = []
    if looks_like_place(q):
        choices.append({
            "kind": "fn", "id": "place", "label": "terrain tile / map",
            "score": 8, "auto": True,
            "fields": {"q": extract_place(q), "view": "terrain"},
        })
    k = pick_kernel(q)
    if k != "shell":
        choices.append({
            "kind": "kernel", "id": k, "label": f"{k} miniapp",
            "score": 7, "auto": True,
        })
    for lib in LIBRARIES:
        hay = set(_WORD.findall(lib["id"].replace("_", " ")))
        extra = set(_WORD.findall((lib.get("when") or "").lower()))
        n = len((hay | extra) & toks)
        if n:
            choices.append({
                "kind": "fn", "id": lib["id"],
                "label": lib["id"].replace("_", " "),
                "score": n + 2, "auto": False,
                "url": lib["url"],
            })
    choices.sort(key=lambda c: -c["score"])
    # Dedup by id
    seen = set()
    uniq = []
    for c in choices:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        uniq.append(c)
    auto = uniq[0] if uniq and uniq[0].get("auto") and (
        len(uniq) == 1 or uniq[0]["score"] >= (uniq[1]["score"] if len(uniq) > 1 else 0) + 2
    ) else None
    return {
        "query": q,
        "choices": uniq[:8],
        "auto": auto,
        "need_clarify": auto is None and bool(uniq),
    }


def pick_kernel(query: str) -> str:
    low = query.lower()
    if any(w in low for w in ("meal", "dinner", "grocery", "pantry",
                              "food plan", "meal planner")):
        return "meals"
    if low.strip() in _MFX_PRIMITIVES:
        return low.strip()
    if any(w in low for w in ("lissajous", "scope", "3d curve",
                              "3d plot")):
        return "scope3d"
    if any(w in low for w in ("random walk", "walker", "drunkard")):
        return "walk"
    if any(w in low for w in ("first aid", "symptom", "body map",
                              "injury report")):
        return "bodymap"
    if any(w in low for w in ("habit", "week", "7-day", "7 day", "seven day",
                              "daily tracker")):
        return "week"
    if "bmi" in low or "body mass" in low:
        return "bmi"
    if any(w in low for w in ("timer", "stopwatch", "countdown")):
        return "timer"
    if any(w in low for w in ("todo", "notes", "checklist", "list")):
        return "notes"
    if any(w in low for w in ("tally", "clicker", "counter")):
        return "counter"
    if any(w in low for w in ("calc", "convert", "converter", "calculator")):
        return "calc"
    return "shell"


def _wrap(title: str, principle: str, inner: str, script: str) -> str:
    t = _esc(title)
    p = _esc(principle)
    return (
        "<!DOCTYPE html><html><head><meta charset=utf-8>"
        f"<style>{_MINI_CSS}</style></head>"
        f"<body><div id=stage><h1>{t}</h1>"
        f"<p class=principle>{p}</p>{inner}</div>"
        f"<script>{script}</script></body></html>"
    )


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _kernel_week(title: str) -> dict:
    principle = "A week has seven days. Left rail names the habit; right rail marks a day."
    inner = '<p id=habit></p><div id=row></div><p id=out>0 / 7</p>'
    script = """
const DAYS=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const on=Array(7).fill(false);
const row=document.getElementById('row');
const out=document.getElementById('out');
const hab=document.getElementById('habit');
DAYS.forEach((d)=>{
  const b=document.createElement('span');
  b.textContent=d;
  b.className='day';
  row.appendChild(b);
});
function paint(){
  [...row.children].forEach((el,i)=>el.classList.toggle('on',on[i]));
  out.textContent=on.filter(Boolean).length+' / 7';
}
function apply(f){
  if(f.habit)hab.textContent=String(f.habit);
  const i=DAYS.indexOf(String(f.day||''));
  const mark=String(f.mark||'done');
  if(mark==='reset_week'){on.fill(false);}
  else if(i>=0){on[i]=mark!=='clear';}
  paint();
}
window.addEventListener('message',(e)=>{
  if(e.data&&e.data.type==='io')apply(e.data.fields||{});
});
"""
    return {
        "html": _wrap(title, principle, inner, script),
        "fields": [
            {"name": "habit", "type": "text", "value": "habit"},
            {"name": "day", "type": "select", "value": "Mon",
             "options": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
            {"name": "mark", "type": "select", "value": "done",
             "options": ["done", "clear", "reset_week"]},
        ],
        "principle": principle,
        "template": "html",
        "domain": "lifestyle",
        "group": "command",
        "span": 1,
        "viewport": "both",
    }


def _kernel_bmi(title: str) -> dict:
    principle = "BMI = mass_kg / height_m² (height in metres)."
    inner = '<p id=out>BMI —</p>'
    script = """
const out=document.getElementById('out');
function update(f){
  const m=Number(f.mass_kg), h=Number(f.height_m);
  if(!(h>0)||!(m>0)){out.textContent='BMI —';return;}
  const bmi=m/(h*h);
  out.textContent='BMI '+bmi.toFixed(1);
}
update({mass_kg:70,height_m:1.75});
window.addEventListener('message',(e)=>{
  if(e.data&&e.data.type==='io')update(e.data.fields||{});
});
"""
    return {
        "html": _wrap(title, principle, inner, script),
        "fields": [
            {"name": "mass_kg", "type": "number", "value": 70},
            {"name": "height_m", "type": "number", "value": 1.75},
        ],
        "principle": principle,
        "template": "html",
        "domain": "lifestyle",
        "group": "command",
        "span": 1,
        "viewport": "both",
    }


def _kernel_timer(title: str) -> dict:
    principle = "Elapsed time is the integral of 1 over seconds. Left = duration; right = run."
    inner = '<p id=out>0.0 s</p>'
    script = """
let t0=0, acc=0, run=false, h=null, cap=0;
const out=document.getElementById('out');
function now(){return performance.now();}
function show(){
  const s=acc+(run?now()-t0:0);
  out.textContent=(s/1000).toFixed(1)+' s'+(cap?(' / '+cap+' s'):'');
  if(cap && s/1000>=cap && run){run=false;clearInterval(h);acc=cap*1000;}
}
function apply(f){
  if(f.duration_s!=null && f.duration_s!=='')cap=Number(f.duration_s)||0;
  const a=String(f.action||'');
  if(a==='start' && !run){t0=now();run=true;h=setInterval(show,100);}
  else if(a==='stop' && run){acc+=now()-t0;run=false;clearInterval(h);}
  else if(a==='reset'){acc=0;t0=now();run=false;clearInterval(h);}
  show();
}
window.addEventListener('message',(e)=>{
  if(e.data&&e.data.type==='io')apply(e.data.fields||{});
});
show();
"""
    return {
        "html": _wrap(title, principle, inner, script),
        "fields": [
            {"name": "duration_s", "type": "number", "value": 60},
            {"name": "action", "type": "select", "value": "start",
             "options": ["start", "stop", "reset"]},
        ],
        "principle": principle,
        "template": "html",
        "domain": "tools",
        "group": "command",
        "span": 1,
        "viewport": "both",
    }


def _kernel_notes(title: str) -> dict:
    principle = "A list is ordered items; adding appends, it does not replace."
    inner = '<div id=row></div>'
    script = """
const items=[];
const row=document.getElementById('row');
function render(){
  row.innerHTML='';
  items.forEach((t)=>{
    const p=document.createElement('p');
    p.textContent='· '+t;
    row.appendChild(p);
  });
}
window.addEventListener('message',(e)=>{
  if(!e.data||e.data.type==='io')return;
  const n=(e.data.fields||{}).note;
  if(n){items.push(String(n));render();}
});
"""
    return {
        "html": _wrap(title, principle, inner, script),
        "fields": [{"name": "note", "type": "text", "value": ""}],
        "principle": principle,
        "template": "html",
        "domain": "tools",
        "group": "command",
        "span": 1,
        "viewport": "both",
    }


def _kernel_counter(title: str) -> dict:
    principle = "A tally increments by one per event."
    inner = '<div id=row><button id=up>+</button><button id=dn>−</button></div><p id=out>0</p>'
    script = """
let n=0;
const out=document.getElementById('out');
function render(){out.textContent=String(n);}
document.getElementById('up').addEventListener('click',()=>{n+=1;render();});
document.getElementById('dn').addEventListener('click',()=>{n-=1;render();});
window.addEventListener('message',(e)=>{
  if(!e.data||e.data.type==='io')return;
  const f=e.data.fields||{};
  if(f.n!=null&&f.n!==''){n=Number(f.n)||0;render();}
});
"""
    return {
        "html": _wrap(title, principle, inner, script),
        "fields": [{"name": "n", "type": "number", "value": 0}],
        "principle": principle,
        "template": "html",
        "domain": "tools",
        "group": "command",
        "span": 1,
        "viewport": "both",
    }


def _kernel_calc(title: str) -> dict:
    principle = "A binary operator maps (a, b) to one number."
    inner = '<p id=out>—</p>'
    script = """
const out=document.getElementById('out');
function update(f){
  const a=Number(f.a), b=Number(f.b), op=f.op||'+';
  if(!Number.isFinite(a)||!Number.isFinite(b)){out.textContent='—';return;}
  let r=null;
  if(op==='+')r=a+b;
  else if(op==='-')r=a-b;
  else if(op==='*')r=a*b;
  else if(op==='/' )r=(b===0?'undef':a/b);
  out.textContent=a+' '+op+' '+b+' = '+r;
}
update({a:1,b:1,op:'+'});
window.addEventListener('message',(e)=>{
  if(e.data&&e.data.type==='io')update(e.data.fields||{});
});
"""
    return {
        "html": _wrap(title, principle, inner, script),
        "fields": [
            {"name": "a", "type": "number", "value": 1},
            {"name": "b", "type": "number", "value": 1},
            {"name": "op", "type": "select", "value": "+",
             "options": ["+", "-", "*", "/"]},
        ],
        "principle": principle,
        "template": "html",
        "domain": "tools",
        "group": "command",
        "span": 1,
        "viewport": "both",
    }


def _kernel_shell(title: str, query: str) -> dict:
    principle = "Inputs on the border change the world state in the center."
    inner = '<dl id=vals></dl>'
    script = f"""
const vals=document.getElementById('vals');
const q={json.dumps(query)};
function update(f){{
  vals.innerHTML='';
  const dt=document.createElement('dt'); dt.textContent='query';
  const dd=document.createElement('dd'); dd.textContent=q;
  vals.appendChild(dt); vals.appendChild(dd);
  Object.entries(f||{{}}).forEach(([k,v])=>{{
    const a=document.createElement('dt'); a.textContent=k;
    const b=document.createElement('dd'); b.textContent=String(v);
    vals.appendChild(a); vals.appendChild(b);
  }});
}}
update({{}});
window.addEventListener('message',(e)=>{{
  if(e.data&&e.data.type==='io')update(e.data.fields||{{}});
}});
"""
    return {
        "html": _wrap(title, principle, inner, script),
        "fields": [{"name": "note", "type": "text", "value": ""}],
        "principle": principle,
        "template": "html",
        "domain": infer_domain(query),
        "group": "command",
        "span": 1,
        "viewport": "both",
    }


def _kernel_meals(title: str) -> dict:
    principle = ("Seven dinners feed a week; every ingredient you lack "
                 "belongs on one list.")
    inner = ('<div id=plan></div><p class=sub>Grocery</p>'
             '<div id=list></div><input id=add placeholder="add item">'
             '<button id=go>+</button>')
    script = """
const DAYS=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const meals=JSON.parse(localStorage.getItem('meals')||'null')
  || DAYS.map(()=> '');
let items=JSON.parse(localStorage.getItem('grocery')||'[]');
const plan=document.getElementById('plan');
const list=document.getElementById('list');
function save(){localStorage.setItem('meals',JSON.stringify(meals));
  localStorage.setItem('grocery',JSON.stringify(items));}
function renderPlan(){
  plan.innerHTML='';
  DAYS.forEach((day,i)=>{
    const row=document.createElement('div');row.className='mrow';
    const lab=document.createElement('span');lab.textContent=day;
    const inp=document.createElement('input');inp.value=meals[i];
    inp.placeholder='-';
    inp.addEventListener('input',()=>{meals[i]=inp.value;save();});
    row.appendChild(lab);row.appendChild(inp);plan.appendChild(row);
  });}
function renderList(){
  list.innerHTML='';
  items.forEach((it,i)=>{
    const row=document.createElement('div');
    row.className='gitem'+(it.done?' done':'');
    const box=document.createElement('input');box.type='checkbox';
    box.checked=!!it.done;
    box.addEventListener('change',()=>{items[i].done=box.checked;save();renderList();});
    const span=document.createElement('span');span.textContent=it.text;
    const del=document.createElement('button');del.textContent='x';
    del.addEventListener('click',()=>{items.splice(i,1);save();renderList();});
    row.appendChild(box);row.appendChild(span);row.appendChild(del);
    list.appendChild(row);
  });}
document.getElementById('go').addEventListener('click',()=>{
  const add=document.getElementById('add');
  if(add.value.trim()){items.push({text:add.value.trim(),done:false});
    add.value='';save();renderList();}
});
renderPlan();renderList();
window.addEventListener('message',(e)=>{
  if(!e.data||e.data.type==='io')return;
  const f=e.data.fields||{};
  if(f.item){items.push({text:String(f.item),done:false});save();renderList();}
});
"""
    return {
        "html": _wrap(title, principle, inner, script),
        "fields": [{"name": "item", "type": "text", "value": ""}],
        "principle": principle,
        "template": "html",
        "domain": "tools",
        "group": "operations",
        "span": 1,
        "viewport": "both",
    }


def _kernel_bodymap(title: str) -> dict:
    principle = ("Point at what hurts; the body map turns touch into a "
                 "symptom list for the medical officer.")
    inner = ('<svg id=body viewBox="0 0 100 190" class=figure></svg>'
             '<p class=sub>Selected regions</p><div id=sel></div>'
             '<input id=sym placeholder="describe symptom">'
             '<button id=go>+</button><div id=list></div>')
    script = """
const REGIONS=[
 ['head',50,10,14],['neck',50,22,8],['chest',50,38,20],
 ['abdomen',50,62,18],['pelvis',50,82,16],
 ['arm-L',30,45,9],['arm-R',70,45,9],
 ['hand-L',26,72,7],['hand-R',74,72,7],
 ['leg-L',42,120,11],['leg-R',58,120,11],
 ['foot-L',40,168,8],['foot-R',60,168,8]];
const svg=document.getElementById('body');
const NS='http://www.w3.org/2000/svg';
let picked=JSON.parse(localStorage.getItem('bodymap')||'[]');
let symptoms=JSON.parse(localStorage.getItem('syms')||'[]');
function save(){localStorage.setItem('bodymap',JSON.stringify(picked));
 localStorage.setItem('syms',JSON.stringify(symptoms));}
function draw(){
 svg.innerHTML='';
 REGIONS.forEach(([name,cx,cy,r])=>{
  const c=document.createElementNS(NS,'circle');
  c.setAttribute('cx',cx);c.setAttribute('cy',cy);c.setAttribute('r',r);
  c.setAttribute('class',picked.includes(name)?'on':'off');
  c.addEventListener('click',()=>{
    const i=picked.indexOf(name);
    if(i>=0)picked.splice(i,1);else picked.push(name);
    save();draw();renderSel();});
  const t=document.createElementNS(NS,'text');
  t.setAttribute('x',cx+r+2);t.setAttribute('y',cy+3);
  t.textContent=name;
  svg.appendChild(c);svg.appendChild(t);});
}
function renderSel(){
 document.getElementById('sel').textContent=picked.join(', ')||'(none)';
 const list=document.getElementById('list');list.innerHTML='';
 symptoms.forEach((s,i)=>{const d=document.createElement('div');
  d.className='srow';d.textContent='- '+s;
  const b=document.createElement('button');b.textContent='x';
  b.addEventListener('click',()=>{symptoms.splice(i,1);save();renderSel2();});
  d.appendChild(b);list.appendChild(d);});}
function renderSel2(){renderSel();}
document.getElementById('go').addEventListener('click',()=>{
 const el=document.getElementById('sym');
 if(el.value.trim()){symptoms.push(el.value.trim());el.value='';save();renderSel();}});
draw();renderSel();
window.addEventListener('message',(e)=>{
 if(!e.data||e.data.type==='io')return;
 const f=e.data.fields||{};
 if(f.symptom){symptoms.push(String(f.symptom));save();renderSel();}});
"""
    return {
        "html": _wrap(title, principle, inner, script),
        "fields": [{"name": "symptom", "type": "text", "value": ""}],
        "principle": principle,
        "template": "html",
        "domain": "tools",
        "group": "medical",
        "span": 1,
        "viewport": "both",
    }


# MicroFX CV-modules live in fabric/microfx.py; converted primitives
# become kernels hosting the shared patch runtime.
# Converted app-primitives (counter/timer/bmi/week) are EEL2 programs
# hosted through build_jsfx; the rest are CV modules for patches.
try:
    from .microfx import (PRIMITIVE_APPS as _MFX_PRIMITIVES,
                          MODULES as _MFX_MODULES,
                          build_jsfx as _mfx_build,
                          build_patch_html as _mfx_patch_build)

    def _kernel_microfx(name):
        def _build(title=None):
            return _mfx_build(name, title)
        return _build

    def _patch_kernel(_name):
        def _build(title=None):
            return {"error": "patches instantiate via microfx_patch tool"}
        return _build
except ImportError:  # pragma: no cover
    _MFX_PRIMITIVES = {}
    _kernel_microfx = None
    _patch_kernel = None

_KERNELS = {
    "week": _kernel_week,
    "bmi": _kernel_bmi,
    "timer": _kernel_timer,
    "notes": _kernel_notes,
    "counter": _kernel_counter,
    "calc": _kernel_calc,
    "meals": _kernel_meals,
    "bodymap": _kernel_bodymap,
}
# MicroFX replaces the legacy DOM builds for converted primitives
for _pn in (_MFX_PRIMITIVES or {}):
    _KERNELS[_pn] = _kernel_microfx(_pn)
_KERNELS.setdefault("patch", _patch_kernel("patch"))


def _esc_check(html: str) -> str | None:
    if not html or not html.strip():
        return "empty html"
    if len(html) > 65536:
        return "html too large"
    if _REMOTE.search(html):
        return "remote script or iframe is not allowed"
    return None


def validate(spec: dict) -> str | None:
    if not spec.get("id") or not re.fullmatch(r"[a-z][a-z0-9_]{1,47}", spec["id"] or ""):
        return "bad id"
    if not spec.get("title"):
        return "missing title"
    if spec.get("template") not in {t["id"] for t in TEMPLATES}:
        return "unknown template"
    err = _esc_check(spec.get("html") or "")
    if err:
        return err
    fields = spec.get("fields") or []
    if not isinstance(fields, list):
        return "fields must be a list"
    for f in fields:
        if not isinstance(f, dict) or not f.get("name"):
            return "field missing name"
        if f.get("type") not in ("number", "text", "select"):
            return f"bad field type {f.get('type')}"
        if f["type"] == "select" and not f.get("options"):
            return "select missing options"
    return None


def evaluate(spec: dict, query: str) -> dict:
    """MiniAppEval-lite: Intention / Static / Dynamic / left-rail I/O."""
    qtoks = set(_WORD.findall((query or "").lower()))
    hay = " ".join([
        spec.get("title") or "",
        spec.get("principle") or "",
        spec.get("id") or "",
        spec.get("kernel") or "",
    ]).lower()
    htoks = set(_WORD.findall(hay))
    overlap = qtoks & htoks
    stop = {"a", "an", "the", "to", "for", "of", "and", "or", "new", "make",
            "build", "create", "generate", "app", "miniapp", "microapp"}
    overlap -= stop
    kernel = spec.get("kernel")
    intention = bool(overlap) or kernel in _KERNELS
    static_err = validate(spec)
    html = spec.get("html") or ""
    dynamic = bool(re.search(r"addEventListener|onclick|oninput|onmessage", html, re.I))
    left_io = any(f.get("type") in ("number", "text")
                  for f in (spec.get("fields") or []))
    reasons = []
    if not intention:
        reasons.append("intention")
    if static_err:
        reasons.append("static")
    if not dynamic:
        reasons.append("dynamic")
    if not left_io:
        reasons.append("left_io")
    return {
        "intention": intention,
        "static": static_err is None,
        "dynamic": dynamic,
        "left_io": left_io,
        "pass": not reasons,
        "reasons": reasons,
        "overlap": sorted(overlap)[:8],
        "static_error": static_err,
    }


def record_trace(query: str, action: str, choice: dict | None = None,
                 spec: dict | None = None, gates: dict | None = None) -> None:
    try:
        from .log_store import get_log_store

        pub = None
        if spec:
            pub = {k: spec[k] for k in
                   ("id", "title", "kernel", "template", "fields",
                    "principle", "group") if k in spec}
            if "id" not in (pub or {}) and spec.get("id"):
                pub = public_spec(spec)
                pub.pop("url", None)
                pub.pop("method", None)
        get_log_store().trace(
            query=query or "",
            action=action,
            choice_kind=(choice or {}).get("kind"),
            choice_id=(choice or {}).get("id"),
            spec_id=(spec or {}).get("id") or (pub or {}).get("id"),
            spec=pub,
            gates=gates,
            passed=bool(gates and gates.get("pass")),
        )
    except Exception:
        pass


def _fallback_choices(query: str) -> list[dict]:
    prop = propose(query)
    return prop["choices"] or [
        {"kind": "kernel", "id": "notes", "label": "notes list"},
        {"kind": "fn", "id": "place", "label": "terrain tile / map"},
        {"kind": "fn", "id": "hadamard", "label": "hadamard"},
        {"kind": "fn", "id": "sage", "label": "sage / CAS"},
        {"kind": "fn", "id": "horizon", "label": "horizon fan"},
    ]


def _clarify(query: str, gates: dict | None = None, draft: dict | None = None,
             hint: str | None = None, choices: list | None = None) -> dict:
    record_trace(query, "clarify", spec=draft, gates=gates)
    out = {
        "clarify": True,
        "wizard": True,
        "query": query,
        "choices": choices or _fallback_choices(query),
        "hint": hint or "no prerouted path — pick department, input, output, viewports",
        "kit": kit(),
    }
    if gates:
        out["gates"] = gates
    if draft:
        try:
            out["draft"] = public_spec(draft) if draft.get("html") else draft
        except Exception:
            out["draft"] = {"id": draft.get("id"), "title": draft.get("title")}
    return out


def public_spec(spec: dict) -> dict:
    return {
        "id": spec["id"],
        "title": spec["title"],
        "group": spec.get("group") or "command",
        "method": "POST",
        "url": f"/api/microapps/{spec['id']}/run",
        "viewport": spec.get("viewport") or "both",
        "fields": spec.get("fields") or [],
        "io": spec.get("io") or {},
        "generated": True,
        "domain": spec.get("domain"),
        "template": spec.get("template"),
        "span": spec.get("span") or 1,
        "principle": spec.get("principle"),
        "kernel": spec.get("kernel"),
    }


def save(spec: dict) -> None:
    d = _root()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{spec['id']}.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    _cache[spec["id"]] = spec


def list_public() -> list[dict]:
    return [public_spec(s) for s in _load_all().values()]


def get(app_id: str) -> dict | None:
    return _ensure_loaded().get(app_id) or _load_all().get(app_id)


def apply_route(query: str, route: dict, title: str | None = None) -> dict:
    """Map wizard enums onto a prerouted bind, kernel, or labeled stub."""
    query = (query or "").strip() or "new miniapp"
    dept = (route.get("dept") or "command").lower()
    ins = route.get("inputs") or [route.get("input") or "text"]
    outs = route.get("outputs") or [route.get("output") or "html"]
    if isinstance(ins, str):
        ins = [ins]
    if isinstance(outs, str):
        outs = [outs]
    inp, out = (ins[0] if ins else "text"), (outs[0] if outs else "html")
    vp = route.get("viewports") or route.get("span") or 1
    span = viewport_span(vp)
    rec = match_route(dept, inp, out, query)
    if rec is None:
        allowed = allowed_outputs(dept, inp)
        return {
            "wizard": True,
            "clarify": True,
            "query": query,
            "route": {"dept": dept, "inputs": ins, "outputs": outs,
                      "viewports": vp},
            "allowed_outputs": allowed,
            "hint": ("no prerouted path for "
                     f"{dept} · {inp} → {out} — pick another output"),
            "choices": [{"kind": "output", "id": o, "label": o} for o in allowed]
                       or [{"kind": "fn", "id": "place", "label": "terrain tile / map"}],
        }
    record_trace(query, "route",
                 choice={"kind": "route",
                         "id": f"{dept}.{inp}.{out}.{span}"},
                 spec={"id": rec.get("bind") or rec.get("kernel") or "stub",
                       "title": infer_title(query, title)},
                 gates={"pass": not rec.get("stub"), "intention": True,
                        "static": True, "dynamic": True, "left_io": True,
                        "reasons": []})
    if rec.get("stub"):
        return _stub_app(query, title, dept, inp, out, span)
    if rec.get("kernel"):
        kn = rec["kernel"]
        if kn == "auto":
            kn = pick_kernel(query)
            if kn == "shell":
                kn = "notes"
        return compose(query, title=title, kernel=kn, span=span)
    bind_id = rec.get("bind")
    extra = dict(rec.get("fields") or {})
    return _bind_app(bind_id, query, span, extra)


def _bind_app(fn_id: str, query: str, span: int, extra: dict | None = None) -> dict:
    from . import console

    fn = next((f for f in console.FUNCTIONS if f["id"] == fn_id), None)
    if fn is None:
        return {"error": f"unknown bind {fn_id}"}
    spec = {k: fn[k] for k in
            ("id", "title", "group", "method", "url", "viewport", "fields")
            if k in fn}
    spec["span"] = span
    spec["bind"] = fn_id
    fields = dict(extra or {})
    if fn_id == "place" and query:
        fields.setdefault("q", extract_place(query) if looks_like_place(query) else query)
    gates = {"intention": True, "static": True, "dynamic": True,
             "left_io": True, "pass": True, "reasons": []}
    return {
        "bind": fn_id,
        "spec": spec,
        "fields": fields,
        "span": span,
        "gates": gates,
        "query": query,
    }


def _stub_app(query: str, title: str | None, dept: str, inp: str, out: str,
              span: int) -> dict:
    from . import departments

    d = next((x for x in departments.DEPARTMENTS if x["id"] == dept), None)
    note = (d or {}).get("note") or f"{dept} · {inp} → {out} not wired"
    title_s = infer_title(query, title) or f"{dept} standby"
    principle = note
    inner = "<p id=out></p>"
    script = (
        "const out=document.getElementById('out');"
        f"out.textContent={json.dumps(note)};"
        "window.addEventListener('message',(e)=>{"
        "if(e.data&&e.data.type==='io'&&e.data.fields&&e.data.fields.note)"
        "out.textContent=String(e.data.fields.note);"
        "});"
    )
    built = {
        "html": _wrap(title_s, principle, inner, script),
        "fields": [{"name": "note", "type": "text", "value": note}],
        "principle": principle,
        "template": "html",
        "domain": "tools",
        "group": dept if dept in ("command", "science") else "command",
        "span": span,
        "viewport": "both",
        "stub": True,
    }
    spec = {
        "id": _unique_id(title_s),
        "title": title_s,
        "query": query,
        "kernel": "stub",
        **built,
    }
    err = validate(spec)
    if err:
        spec["template"] = "html"
    save(spec)
    gates = evaluate(spec, query)
    record_trace(query, "stub", spec=spec, gates=gates)
    pub = public_spec(spec)
    return {
        "id": spec["id"],
        "spec": pub,
        "figure": {"type": "html", "html": spec["html"],
                   "caption": spec["title"], "sandbox": True, "span": span},
        "gates": gates,
        "stub": True,
        "hint": note,
    }


def update_io(app_id: str, io: dict) -> dict:
    """Edit an app's io manifest (sidebar add/edit in build mode):
    merge, validate, persist. Returns the public spec or error."""
    from .microfx import validate_io

    spec = get(app_id)
    if not spec:
        return {"error": "no such app"}
    merged = dict(spec.get("io") or {})
    for key in ("ins", "outs", "controls", "tiles", "triggers", "wires"):
        if key in io:
            merged[key] = io[key]
    err = validate_io(merged)
    if err:
        return {"error": err}
    spec["io"] = merged
    err = validate(spec)
    if err:
        return {"error": err}
    save(spec)
    return {"spec": public_spec(spec), "io": merged}


def rename_app(app_id: str, title: str) -> dict:
    spec = get(app_id)
    if not spec:
        return {"error": "no such app"}
    title = (title or "").strip()
    if not title:
        return {"error": "empty title"}
    spec["title"] = title
    err = validate(spec)
    if err:
        return {"error": err}
    save(spec)
    return {"spec": public_spec(spec)}


def new_app(title: str | None = None) -> dict:
    """Blank app draft for the build wizard: valid, saved, io empty —
    the shell assembles ins/outs/controls/tiles/wires in edit mode."""
    from .appwiz import _package

    title = (title or "").strip() or "untitled app"
    built = _package(
        "blank", title,
        "Blank canvas — assemble inputs, functions, viewports in "
        "edit mode.",
        [],
        {"ins": [], "outs": [], "controls": [], "tiles": []},
        "@init\nn = 0;\n\n@gfx\n")
    spec = {
        "id": _unique_id(title),
        "title": title,
        "query": "blank canvas",
        "kernel": "signal",
        **built,
    }
    err = validate(spec)
    if err:
        return {"error": err}
    save(spec)
    return {"spec": public_spec(spec), "io": spec.get("io") or {}}


def clone_app(app_id: str, title: str | None = None) -> dict:
    """Save App As: deep-copy a stored spec under a new id/title."""
    src = get(app_id)
    if not src:
        return {"error": "no such app"}
    title = (title or "").strip() or f"{src.get('title') or 'app'} copy"
    spec = json.loads(json.dumps(
        {k: v for k, v in src.items() if not k.startswith("_")}))
    spec["id"] = _unique_id(title)
    spec["title"] = title
    err = validate(spec)
    if err:
        return {"error": err}
    save(spec)
    return {"spec": public_spec(spec)}


def compose_script(body: dict) -> dict:
    """scriptwiz: convert a python file (4 passes) into a native
    MicroFX app. body {path, title?}. Same persist/return shape."""
    from . import scriptwiz

    out = scriptwiz.convert(body.get("path") or "",
                            title=body.get("title") or None)
    if out.get("error"):
        return out
    app = out["app"]
    spec = {
        "id": _unique_id(Path(body["path"]).stem + " converted"),
        "title": app["title"],
        "query": body.get("path"),
        "kernel": "signal",
        **app,
    }
    err = validate(spec)
    if err:
        return {"error": err}
    save(spec)
    record_trace(body["path"], "compose-script",
                 choice={"kind": "scriptwiz",
                         "id": Path(body["path"]).name},
                 spec=spec, gates={"pass": True, "reasons": []})
    pub = public_spec(spec)
    return {
        "id": spec["id"],
        "spec": pub,
        "report": out["report"],
        "figure": {"type": "html", "html": spec["html"],
                   "caption": spec["title"], "sandbox": True,
                   "span": spec.get("span") or 1},
        "gates": {"pass": True},
    }


def compose_signal(body: dict) -> dict:
    """AppWizard: generate an app from a signal-path selection
    {source, visualizer, topic?, url?, title?}. Same persist/return
    shape as compose()."""
    from . import appwiz

    built = appwiz.generate_signal_app(
        body.get("source") or "", body.get("visualizer") or "",
        title=body.get("title") or None,
        topic=body.get("topic") or None,
        url=body.get("url") or None)
    if built.get("error"):
        return built
    query = body.get("query") or (built.get("principle") or "signal app")
    spec = {
        "id": _unique_id(body.get("title")
                         or f"{body.get('source')}-{body.get('visualizer')}"),
        "title": built["title"] if "title" in built
        else (body.get("title") or "signal app"),
        "query": query,
        "kernel": "signal",
        **built,
    }
    err = validate(spec)
    if err:
        return {"error": err}
    save(spec)
    record_trace(query, "compose-signal",
                 choice={"kind": "signal",
                         "id": f"{body.get('source')}->"
                               f"{body.get('visualizer')}"},
                 spec=spec, gates={"pass": True, "reasons": []})
    try:
        from .log_store import get_log_store

        get_log_store().append(f"signal app {spec['id']}",
                               source="fabric")
    except Exception:
        pass
    pub = public_spec(spec)
    return {
        "id": spec["id"],
        "spec": pub,
        "figure": {"type": "html", "html": spec["html"],
                   "caption": spec["title"], "sandbox": True,
                   "span": spec.get("span") or 1},
        "gates": {"pass": True},
    }


def compose(query: str, html: str | None = None, title: str | None = None,
            kernel: str | None = None, bind: str | None = None,
            route: dict | None = None, span: int | None = None) -> dict:
    """Build + persist a MiniApp for a free-text query. Fast path: kernels."""
    query = (query or "").strip()
    if not query and not route:
        return {"error": "empty query"}
    if route:
        return apply_route(query or "new miniapp", route, title=title)
    if bind in ("place", "show_place"):
        from . import geo

        out = geo.show_place(extract_place(query) if looks_like_place(query) else query)
        record_trace(query, "bind",
                     choice={"kind": "fn", "id": "place"},
                     spec={"id": "place", "title": "Place", "kernel": None,
                           "fields": [], "principle": "DEM at a named place"},
                     gates={"intention": True, "static": True, "dynamic": True,
                            "left_io": True, "pass": not out.get("clarify"),
                            "reasons": []})
        return out
    title_s = infer_title(query, title)
    if not (html and html.strip()):
        kn = kernel or pick_kernel(query)
        prior = get("app_" + _slug(title_s))
        if prior and prior.get("kernel") == kn and kn in _KERNELS:
            want = {f["name"] for f in _KERNELS[kn](title_s)["fields"]}
            have = {f["name"] for f in (prior.get("fields") or [])}
            if want == have:
                gates = evaluate(prior, query)
                pub = public_spec(prior)
                record_trace(query, "reuse", spec=pub, gates=gates)
                if not gates.get("pass"):
                    return _clarify(query, gates=gates, draft=prior,
                                    hint="cached MiniApp failed gates — rebind")
                return {
                    "id": prior["id"],
                    "spec": pub,
                    "figure": {
                        "type": "html",
                        "html": prior["html"],
                        "caption": prior["title"],
                        "sandbox": True,
                        "span": prior.get("span") or 1,
                    },
                    "gates": gates,
                    "kit": "html tile · left I/O · right selects",
                    "reused": True,
                }
    if html and html.strip():
        err = _esc_check(html)
        if err:
            return {"error": err}
        spec = {
            "id": _unique_id(title_s),
            "title": title_s,
            "html": html.strip(),
            "fields": [{"name": "note", "type": "text", "value": ""}],
            "principle": "Operator-supplied MiniApp.",
            "template": "html",
            "domain": infer_domain(query),
            "group": "command",
            "span": 1,
            "viewport": "both",
            "kernel": "custom",
            "query": query,
        }
    else:
        kn = kernel or pick_kernel(query)
        if kn == "shell" and not kernel:
            return _clarify(
                query,
                hint="no kernel matched — pick a library or MiniApp kernel (viz BL)",
            )
        if kn == "shell":
            built = _kernel_shell(title_s, query)
        else:
            if kn not in _KERNELS:
                return {"error": f"unknown kernel {kn}"}
            built = _KERNELS[kn](title_s)
        spec = {
            "id": _unique_id(title_s),
            "title": title_s,
            "query": query,
            "kernel": kn,
            **built,
        }
        if kn != "shell":
            spec["domain"] = built.get("domain") or infer_domain(query)
    if span:
        spec["span"] = int(span)
    err = validate(spec)
    if err:
        return {"error": err}
    forced = bool(kernel) or bool(html and html.strip())
    gates = evaluate(spec, query)
    if not gates.get("pass") and not forced:
        prop = propose(query)
        auto = prop.get("auto")
        if auto and auto.get("kind") == "kernel" and auto["id"] in _KERNELS \
                and auto["id"] != spec.get("kernel"):
            rebuilt = _KERNELS[auto["id"]](title_s)
            spec = {"id": spec["id"], "title": title_s, "query": query,
                    "kernel": auto["id"], **rebuilt}
            gates = evaluate(spec, query)
            record_trace(query, "retry", choice=auto, spec=spec, gates=gates)
        if not gates.get("pass"):
            return _clarify(
                query, gates=gates, draft=spec,
                hint="MiniApp failed Intention/Static/Dynamic/left-I/O — pick a bind",
                choices=prop.get("choices") or _fallback_choices(query),
            )
    save(spec)
    record_trace(query, "compose",
                 choice={"kind": "kernel", "id": spec.get("kernel")},
                 spec=spec, gates=gates)
    try:
        from .log_store import get_log_store

        get_log_store().append(f"miniapp {spec['id']}", source="fabric")
    except Exception:
        pass
    pub = public_spec(spec)
    out = {
        "id": spec["id"],
        "spec": pub,
        "figure": {
            "type": "html",
            "html": spec["html"],
            "caption": spec["title"],
            "sandbox": True,
            "span": spec.get("span") or 1,
        },
        "gates": gates,
        "kit": "html tile · left I/O · right selects",
        "span": spec.get("span") or 1,
    }
    if not gates.get("pass"):
        out.update(_clarify(
            query, gates=gates, draft=spec,
            hint="weak gates — rebind from the configurator",
        ))
        out["spec"] = pub
        out["figure"] = {
            "type": "html", "html": spec["html"],
            "caption": spec["title"], "sandbox": True,
            "span": spec.get("span") or 1,
        }
        out["id"] = spec["id"]
    return out


def run(app_id: str, fields: dict | None = None) -> dict:
    spec = get(app_id)
    if spec is None:
        return {"error": f"unknown miniapp {app_id}"}
    return {
        "id": spec["id"],
        "spec": public_spec(spec),
        "fields": fields or {},
        "figure": {
            "type": "html",
            "html": spec["html"],
            "caption": spec["title"],
            "sandbox": True,
            "span": spec.get("span") or 1,
        },
        "principle": spec.get("principle"),
        "gates": evaluate(spec, spec.get("query") or spec["title"]),
    }
