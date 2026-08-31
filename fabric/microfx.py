"""MicroFX — CV-module primitives (modular-synth model).

Modules are plugin programs with TYPED inputs/outputs and an EEL2 body
that runs at control rate (@tick). Patches wire module outputs to module
inputs; designated outputs may be visualized into the viz deck.

Types: number (CV), trigger (short pulse), series (rolling buffer).
Source modules (host-fed): sensor(topic), clock_bpm(bpm).
Pure modules run in-process via the EEL2 interpreter in web/jsfx.js.

Writable n-gram PLE verbs (Step 3, docs/27b-writable-ngram-plan.md):
  ngram_lookup(mem_start, token, layer[, pos])  — MV2 row -> mem[160],
    frame-cached (0 until the fetch lands; read every frame)
  ngram_store(token, layer, mem_start[, pos])   — mem[160] -> MV2 (WAL)
  ngram_neighbors(mem_start, token, layer[, k]) — reserved, returns 0
    (vec index disabled until an embedder is chosen)
Host bridge: fetch to /api/ngram/* (fabric/ngram_api.py); hosts
without the bridge (headless, patch runner) no-op to 0.
"""

from __future__ import annotations

import json
import os
import time

_JSFX_SRC: str | None = None


def jsfx_runtime() -> str:
    global _JSFX_SRC
    if _JSFX_SRC is None:
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "web", "jsfx.js")
        try:
            with open(src) as f:
                _JSFX_SRC = f.read()
        except OSError:
            _JSFX_SRC = ""
    return _JSFX_SRC


# name: {title, params[{name,min,max,default}], inputs[], outputs[],
#        source (@init/@tick)}
# NODE RULE (Rack v2 engine model): a node has AT MOST ONE input port
# and any number of output ports. Fan-out from one output is free;
# multiple cables into one input SUM (stackable inputs, Engine.cpp
# Engine_stepFrameCables). Streams are the only currency: a node is
# anything that transforms its input stream (runtime, codec, variable).
#
# HIERARCHY (operator, 2026-08-25): catalog entries carry a category —
#   source      main sources flow into functions (synthetic or real:
#               sensor topics, a/v streams, files, ports)
#   function    1 source input, N outputs; modulated by controls
#   control     modulates functions and variables (control standard)
#   visualizer  SINK nodes; may stack MULTIPLE inputs (wxyz3d takes
#               W/X/Y/Z) — the one-input rule does not bind sinks
# Two signal paths: SIGNAL (left rail in -> function blocks -> right
# rail out) and CONTROL (modulates the signal). Canvas shows controls,
# labels, and visualizer outputs; wires appear only in edit mode.
MODULES: dict[str, dict] = {
    "const": {
        "category": "source",
        "title": "Constant",
        "params": [{"name": "value", "min": -1000, "max": 1000,
                    "default": 1}],
        "inputs": [], "outputs": ["cv"],
        "source": "@tick\noutput('cv', value);",
    },
    "clock_bpm": {
        "category": "source",
        "title": "Clock",
        "params": [{"name": "bpm", "min": 1, "max": 600, "default": 60}],
        "inputs": [], "outputs": ["trig"],
        "host": True,
        "source": "",  # host-fed
    },
    "sine_lfo": {
        "category": "source",
        "title": "Sine LFO",
        "params": [{"name": "rate_hz", "min": 0.01, "max": 15,
                    "default": 1},
                   {"name": "amp", "min": 0, "max": 1000, "default": 1},
                   {"name": "offset", "min": -1000, "max": 1000,
                    "default": 0}],
        "inputs": [], "outputs": ["cv"],
        "source": """@init
phase = 0;

@tick
dt = 1 / 30;
phase += rate_hz * dt;
cv = offset + amp * sin(phase * 6.2832);
output('cv', cv);""",
    },
    "gain": {
        "category": "function",
        "title": "Gain",
        "params": [{"name": "factor", "min": -100, "max": 100,
                    "default": 1}],
        "inputs": ["in"], "outputs": ["cv"],
        "source": "@tick\noutput('cv', input('in') * factor);",
    },
    "bias": {
        "category": "function",
        "title": "Bias",
        "params": [{"name": "add", "min": -1000, "max": 1000,
                    "default": 0}],
        "inputs": ["in"], "outputs": ["cv"],
        "source": "@tick\noutput('cv', input('in') + add);",
    },
    "smooth": {
        "category": "function",
        "title": "Smooth (one-pole)",
        "params": [{"name": "alpha", "min": 0.001, "max": 1,
                    "default": 0.2}],
        "inputs": ["in"], "outputs": ["cv"],
        "source": """@init
y = 0;

@tick
y += alpha * (input('in') - y);
output('cv', y);""",
    },
    "threshold": {
        "category": "function",
        "title": "Threshold (hysteresis)",
        "params": [{"name": "lo", "min": -1e6, "max": 1e6, "default": 40},
                   {"name": "hi", "min": -1e6, "max": 1e6, "default": 50}],
        "inputs": ["in"], "outputs": ["gate"],
        "source": """@init
state = 0;

@tick
v = input('in');
v > hi ? state = 1 : v < lo ? state = 0 : 0;
output('gate', state);""",
    },
    "moving_avg": {
        "category": "function",
        "title": "Moving average",
        "params": [{"name": "n", "min": 2, "max": 128, "default": 16}],
        "inputs": ["in"], "outputs": ["cv"],
        "source": """@init
buf = 0; idx = 0; filled = 0; acc = 0;

@tick
v = input('in');
old = buf[idx];
buf[idx] = v;
acc += v - old;
idx += 1;
idx >= n ? idx = 0;
filled < n ? filled += 1;
output('cv', filled ? acc / filled : v);""",
    },
    "clamp": {
        "category": "function",
        "title": "Clamp",
        "params": [{"name": "lo", "min": -1e6, "max": 1e6, "default": 0},
                   {"name": "hi", "min": -1e6, "max": 1e6, "default": 100}],
        "inputs": ["in"], "outputs": ["cv"],
        "source": """@tick
output('cv', min(hi, max(lo, input('in'))));""",
    },
    "mdct_flux": {
        "category": "function",
        "title": "Spectral flux (mdct)",
        "params": [],
        "inputs": ["in"], "outputs": ["flux"],
        "source": """@init
buf = 0; idx = 0; prev_e = 0;

@tick
buf[idx] = input('in');
idx += 1;
idx >= 64 ? (
  idx = 0;
  mdct(0, 64);
  e = 0; k = 0;
  loop(32, e += buf[k] * buf[k]; k += 1;);
  flux = abs(e - prev_e);
  prev_e = e;
);
output('flux', flux);""",
    },
    "sensor": {
        "category": "source",
        "title": "Sensor source",
        "params": [{"name": "topic", "min": 0, "max": 0, "default": 0}],
        "inputs": [], "outputs": ["cv"],
        "host": True,
        "source": "",
    },
    "tap": {
        "category": "source",
        "title": "Tap pulse",
        "params": [],
        "inputs": [], "outputs": ["trig"],
        "host": True,
        "source": "",  # fed by canvas pointerdown (bus ui.tap)
    },
    "toggle": {
        "category": "function",
        "title": "Toggle state",
        "params": [{"name": "initial", "min": 0, "max": 1,
                    "default": 0}],
        "inputs": ["trig"], "outputs": ["state"],
        "source": """@init
state = initial;

@tick
input('trig') > 0 ? state = 1 - state;
output('state', state);""",
    },
    "accum": {
        "category": "function",
        "title": "Edge counter",
        # ONE input per node (Rack-style): counts rising edges of the
        # input signal; wrap=0 means never wrap.
        "params": [{"name": "per_tick", "min": -100, "max": 100,
                    "default": 1},
                   {"name": "wrap", "min": 0, "max": 1e6,
                    "default": 0}],
        "inputs": ["in"], "outputs": ["acc"],
        "source": """@init
acc = 0; prev = 0;

@tick
v = input('in') > 0;
v > prev ? acc += per_tick : 0;
prev = v;
wrap > 0 && acc >= wrap ? acc = 0;
output('acc', acc);""",
    },
    "hadamard4": {
        "category": "function",
        # Sliding 4-point FWHT (H4 rows x [v, s0, s1, s2]). The four row
        # sums are the ACN/WXYZ spherical-harmonic feeds: ROW0 = W
        # (omni pressure), ROW1 = Y (up/down), ROW2 = X (left/right),
        # ROW3 = Z (front/back). Feeds viz_wxyz3d; row sums draw 3D
        # lissajous patterns from the matrix data.
        "title": "Hadamard-4 (WXYZ)",
        "params": [],
        "inputs": ["in"], "outputs": ["w", "y", "x", "z"],
        "source": """@init
s0 = 0; s1 = 0; s2 = 0;

@tick
v = input('in');
output('w', v + s0 + s1 + s2);
output('y', v - s0 + s1 - s2);
output('x', v + s0 - s1 - s2);
output('z', v - s0 - s1 + s2);
s2 = s1;
s1 = s0;
s0 = v;""",
    },
    # ---- visualizers: sink nodes, may stack multiple inputs ----
    "viz_series": {
        "category": "visualizer",
        "title": "Chart (rolling)",
        "params": [],
        "inputs": ["in"], "outputs": [],
        "source": "",  # rendered from views[] by the patch runner
    },
    "viz_xy": {
        "category": "visualizer",
        "title": "XY vectorscope",
        "params": [],
        "inputs": ["x", "y"], "outputs": [],
        "source": "",
    },
    "viz_wxyz3d": {
        "category": "visualizer",
        # 3D energy visualization: W/X/Y/Z row sums as a 3D lissajous
        # (spherical-harmonic component frame).
        "title": "3D WXYZ scope",
        "params": [],
        "inputs": ["w", "x", "y", "z"], "outputs": [],
        "source": "",
    },
}

# ---- logic gates ----------------------------------------------------
# Boolean gates are sanctioned MULTI-INPUT functions (operator rule
# revision): the one-input rule binds dataflow functions, not gates.
# Boolean convention: input > 0.5 is true; outputs are 1/0.
# Analog families (DL/TDL/NL/MOS/CML/QCA) are threshold/transfer-curve
# variants; quantum gates act on bipolar (±1) streams where NOT is a
# sign flip (|0>=+1, |1>=-1).

_GATES = {
    "gate_buffer": ("Buffer", ["in"], "@tick\noutput('q', input('in') > 0.5 ? 1 : 0);"),
    "gate_not": ("NOT (inverter)", ["in"],
                 "@tick\noutput('q', input('in') > 0.5 ? 0 : 1);"),
    "gate_and": ("AND", ["a", "b"],
                 "@tick\noutput('q', input('a') > 0.5 && input('b') > 0.5 ? 1 : 0);"),
    "gate_or": ("OR", ["a", "b"],
                "@tick\noutput('q', input('a') > 0.5 || input('b') > 0.5 ? 1 : 0);"),
    "gate_nand": ("NAND", ["a", "b"],
                  "@tick\noutput('q', input('a') > 0.5 && input('b') > 0.5 ? 0 : 1);"),
    "gate_nor": ("NOR", ["a", "b"],
                 "@tick\noutput('q', input('a') > 0.5 || input('b') > 0.5 ? 0 : 1);"),
    "gate_xor": ("XOR", ["a", "b"],
                 "@tick\na = input('a') > 0.5; b = input('b') > 0.5;\n"
                 "output('q', a != b ? 1 : 0);"),
    "gate_xnor": ("XNOR", ["a", "b"],
                  "@tick\na = input('a') > 0.5; b = input('b') > 0.5;\n"
                  "output('q', a == b ? 1 : 0);"),
    "gate_imply": ("IMPLY (a->b)", ["a", "b"],
                   "@tick\na = input('a') > 0.5; b = input('b') > 0.5;\n"
                   "output('q', a && b || !a ? 1 : 0);"),
    "gate_nimply": ("NIMPLY (a !-> b)", ["a", "b"],
                    "@tick\na = input('a') > 0.5; b = input('b') > 0.5;\n"
                    "output('q', a && !b ? 1 : 0);"),
}
# Gates removed from MODULES per operator spec (2026-08-26):
# gates live ONLY in the Control library. Definitions kept for control ref.
# (Injection loops removed; _GATES / _QGATES definitions preserved above.)

MODULES["alogic"] = {
    "category": "function",
    "multi_in": True,
    # Analog logic families: distinct threshold/transfer curves on one
    # stream input + a control threshold. QCA is the majority cell
    # maj(in, cell_b, cell_c) with the cells as control params.
    "title": "Analog logic family",
    "params": [
        {"name": "family", "min": 0, "max": 5, "default": 3, "step": 1},
        {"name": "thresh", "min": 0, "max": 1, "default": 0.5,
         "step": 0.01},
        {"name": "cell_b", "min": 0, "max": 1, "default": 1, "step": 1},
        {"name": "cell_c", "min": 0, "max": 1, "default": 1, "step": 1},
    ],
    "inputs": ["in"], "outputs": ["q"],
    "source": """@init
st = 0;

@tick
v = input('in');
t = thresh;
family == 0 ? q = v > 0.6;                       // DL: diode drop
family == 1 ? (                                   // TDL: tunnel latch
  v > t + 0.1 ? st = 1 : v < t - 0.1 ? st = 0;
  q = st;
);
family == 2 ? (                                   // NL: neon strike
  v > 0.75 ? st = 1 : v < 0.4 ? st = 0;
  q = st;
);
family == 3 ? q = v > t;                          // MOS: sharp
family == 4 ? (                                   // CML: small swing
  d = v - t;
  d > 0.25 ? d = 0.25 : d < -0.25 ? d = -0.25;
  q = 0.5 + d;
);
family == 5 ? (                                   // QCA: majority cell
  a = v > t; b = cell_b > 0.5; c = cell_c > 0.5;
  q = a + b + c >= 2;
);
q = q ? 1 : 0;
output('q', q);""",
}

_QGATES = {
    # Quantum gates on bipolar (±1) streams: |0>=+1, |1>=-1, so the
    # computational NOT (Pauli X) is a sign flip. Y = iXZ is identical
    # to -in on real streams (kept distinct for graph semantics);
    # S/T phase rotations are identity on real amplitudes (metadata
    # nodes); H is the 1/sqrt(2) basis swap.
    "pauli_x": ("Pauli X (NOT)", ["q"], "output('out', -input('q'));"),
    "pauli_y": ("Pauli Y", ["q"], "output('out', -input('q'));"),
    "pauli_z": ("Pauli Z (phase)", ["q"], "output('out', -input('q'));"),
    "hadamard_gate": ("Hadamard gate", ["q"],
                      "output('out', input('q') * 0.7071067811865476);"),
    "phase_s": ("Phase S (pi/2)", ["q"], "output('out', input('q'));"),
    "phase_t": ("Phase T (pi/4)", ["q"], "output('out', input('q'));"),
}
# Quantum gates (Pauli/H/CNOT/SWAP/Toffoli) removed from MODULES.
# Definitions kept; exposed via Controls library only.

# cnot / swap_gate / toffoli: gate functions removed from MODULES
# per spec; kept as definitions for Control-library exposure only.
# (Previously injected below; removed 2026-08-26.)
HOST_SOURCES = {"clock_bpm", "sensor", "tap"}

# Converted app-primitives: single-purpose EEL2 programs hosted through
# build_jsfx (plugin-format conversions of the old DOM kernels).
#
# App-level IO contract (one level above module CV): an optional "io"
# block declares named TRIGGERS, live INPUT signals, and typed OUTS.
#
# UNIFORM SIGNAL MODEL: every port — in or out, time- or data-based —
# is a LIVE CAPTURED SIGNAL sampled relative to the current frame.
# Inputs are never one-shot requests: trigger ports carry rising-edge
# pulses, ins ports carry continuously-sampled values (the shell feeds
# them each poll; apps read input('name') every frame). Outputs never
# batch-render: scalars are per-emission samples, out_series emits the
# CURRENT WINDOW (fixed-length, shifting), out_points3d emits the
# current scene frame. You always measure against the latest frame.
#
# TRIGGERS gate on their port; SOURCES write it:
#   "manual"                                      (default; left-rail button)
#   {"event":"clock","every_s":N}                 (persistent clock; the
#       manual button toggles the GATE — open passes ticks through)
#   {"event":"sensor","topic":T,"op":O,"value":V} (threshold crossing;
#       op in {">",">=","<","<="}, fires on the rising transition)
#   {"event":"app","app":ID,"out":NAME}           (another app's output)
# INS (live data inputs): [{"name":N,"topic":T}] — the shell samples
# topic T from the sensor bus and writes the latest value into the port
# continuously; no trigger involved.
# CONTROLS (TouchOSC-derived control standard, hexler.net
# scripting-api): user-facing signal SOURCES rendered on the app's
# control surface. Every control owns one or more io ports (live
# signals, same currency as everything else):
#   {"type":"fader",  "name":N, "min":0, "max":1, "default":d}
#       -> port N (normalized 0..1 scaled to min..max)
#   {"type":"button", "name":N, "buttonType":"momentary"|"toggle"}
#       -> port N (0/1; momentary = held while pressed; pairs with
#          trigger('N') rising-edge gates)
#   {"type":"xy",     "name":N}
#       -> ports N_x, N_y (normalized 0..1; one control, two signals)
#   {"type":"encoder","name":N, "min":0, "max":1, "default":d}
#       -> port N (relative drag accumulates, clamped)
# Optional OSC-style "address" ("/name") for external surfaces.
# OUTS: scalars via output(name, v); data-plane frames sliced from
# mem[]: out_series(name, mem_start, count) -> rolling chart tile,
# out_points3d(name, mem_start, count) -> 3D viewport tile. Figures
# route to the shell's tile deck (viewport matrix) under the app title.

_TRIG_OPS = {">", ">=", "<", "<="}


def _validate_trigger_source(name: str, trig: dict) -> str | None:
    src = trig.get("source", "manual")
    if src == "manual":
        return None
    if not isinstance(src, dict):
        return f"trigger '{name}': source must be 'manual' or an object"
    ev = src.get("event")
    if ev == "clock":
        try:
            if not 0.1 <= float(src.get("every_s", 0)) <= 86400:
                raise ValueError
        except (TypeError, ValueError):
            return f"trigger '{name}': clock source needs every_s (0.1..86400)"
        return None
    if ev == "sensor":
        if not str(src.get("topic") or "").strip():
            return f"trigger '{name}': sensor source needs topic"
        if src.get("op", ">") not in _TRIG_OPS:
            return f"trigger '{name}': sensor op must be one of {sorted(_TRIG_OPS)}"
        try:
            float(src.get("value"))
        except (TypeError, ValueError):
            return f"trigger '{name}': sensor source needs numeric value"
        return None
    if ev == "app":
        if not str(src.get("app") or "").strip() \
                or not str(src.get("out") or "").strip():
            return f"trigger '{name}': app source needs app id and out name"
        return None
    return f"trigger '{name}': unknown source event '{ev}'"


_CONTROL_TYPES = {"fader", "button", "xy", "encoder", "ext"}
_EXT_INPUTS = {"mouse_xy", "keyboard", "gamepad_axes", "gamepad_buttons"}


def validate_io(io: dict | None) -> str | None:
    """Validate an app-level io manifest; returns error string or None."""
    if not io:
        return None
    for ctl in io.get("controls") or []:
        name = str(ctl.get("name") or "").strip()
        if not name or not name.replace("_", "").isalnum():
            return f"bad control name '{name}'"
        if ctl.get("type") not in _CONTROL_TYPES:
            return f"control '{name}': unknown type '{ctl.get('type')}'"
        if ctl.get("type") == "ext":
            src = str(ctl.get("input") or "")
            if src not in _EXT_INPUTS:
                return (f"control '{name}': ext input must be one of "
                        f"{sorted(_EXT_INPUTS)}")
            if src == "keyboard" and not str(ctl.get("key") or "").strip():
                return f"control '{name}': keyboard ext needs a key"
        if ctl.get("type") == "button" and \
                ctl.get("buttonType", "momentary") not in \
                ("momentary", "toggle"):
            return f"control '{name}': buttonType must be momentary|toggle"
        addr = ctl.get("address", "/" + name)
        if not str(addr).startswith("/") or " " in str(addr):
            return f"control '{name}': address must be OSC-style ('/path')"
        # inner function chain (operator spec: controls hold a chain array)
        ctl_funcs = ctl.get("functions")
        if ctl_funcs is not None:
            if not isinstance(ctl_funcs, list):
                return f"control '{name}': functions must be array"
            for fid in ctl_funcs:
                if not isinstance(fid, str) or not fid:
                    return f"control '{name}': bad function id in functions"
    for sig in io.get("ins") or []:
        name = str(sig.get("name") or "").strip()
        if not name or not name.replace("_", "").isalnum():
            return f"bad input signal name '{name}'"
        if not str(sig.get("topic") or "").strip():
            return f"input signal '{name}': live inputs need a bus topic"
    for trig in io.get("triggers") or []:
        name = str(trig.get("name") or "").strip()
        if not name or not name.replace("_", "").isalnum():
            return f"bad trigger name '{name}'"
        err = _validate_trigger_source(name, trig)
        if err:
            return err
    for out in io.get("outs") or []:
        name = str(out.get("name") or "").strip()
        if not name or not name.replace("_", "").isalnum():
            return f"bad out name '{name}'"
        if out.get("kind", "number") not in ("number", "series",
                                             "points3d"):
            return f"out '{name}': unknown kind '{out.get('kind')}'"
        loop = str(out.get("loop") or "").strip()
        if loop and not loop.replace("_", "").isalnum():
            return f"out '{name}': bad loopback input '{loop}'"
    # tile subfunctions: max 16 tiles per app; each names its inner
    # signal path. Standardized tile i/o wrapper:
    #   BUS  bus_in -> [tile] -> bus_out; tiles process SEQUENTIALLY
    #        (row-major) so changes accumulate per tile — the shell
    #        chains bus_out(k) into bus_in(k+1) unless explicitly
    #        rewired, feeds the head from sidebar ins and taps the
    #        tail into sidebar outs.
    #   AUX  free i/o channels (default 2, cap 8 — optimum is system-
    #        dependent): direction is assigned by patching (modulator,
    #        source, or direct tile->tile wire).
    tiles = io.get("tiles") or []
    if len(tiles) > 16:
        return "max 16 tiles per app"
    seen_t = set()
    for t in tiles:
        name = str(t.get("name") or "").strip()
        if not name or not name.replace("_", "").isalnum():
            return f"bad tile name '{name}'"
        if name in seen_t:
            return f"duplicate tile name '{name}'"
        seen_t.add(name)
        if t.get("kind") not in ("series", "points3d", "number",
                                 "lcd", "video"):
            return f"tile '{name}': unknown kind '{t.get('kind')}'"
        aux = t.get("aux", 2)
        if not isinstance(aux, int) or not 0 <= aux <= 8:
            return f"tile '{name}': aux channels must be 0..8"
    # patch wires: {from,to} between bus ports (in:/out:), control
    # ports (ctl:) and the tile wrapper points — bus_in/bus_out plus
    # free aux channels (aux0..aux7). Endpoints must resolve against
    # the manifest — wires are the routing truth.
    ctl_names = {str(c.get("name") or "").lower()
                 for c in io.get("controls") or []}
    in_names = {str(s.get("name") or "").lower()
                for s in io.get("ins") or []}
    out_names = {str(o.get("name") or "").lower()
                 for o in io.get("outs") or []}
    tile_aux = {str(t.get("name") or "").lower(): t.get("aux", 2)
                for t in tiles}
    for w in io.get("wires") or []:
        if not isinstance(w, dict):
            return "wire: expected {from,to}"
        for end in ("from", "to"):
            ep = str(w.get(end) or "").strip().lower()
            parts = ep.split(":")
            ok = False
            if len(parts) == 2 and parts[0] in ("in", "out", "ctl"):
                ns, nm = parts
                pool = {"in": in_names, "out": out_names,
                        "ctl": ctl_names}[ns]
                ok = nm in pool
                if not ok and ns == "ctl" and "_" in nm:
                    base, sub = nm.rsplit("_", 1)
                    ok = sub in ("x", "y") and base in pool
            elif len(parts) == 3 and parts[0] == "tile":
                tname, port = parts[1], parts[2]
                if tname in seen_t and port.replace("_", "").isalnum():
                    if port in ("bus_in", "bus_out"):
                        ok = True
                    elif port.startswith("aux"):
                        k = port[3:]
                        ok = k.isdigit() \
                            and int(k) < int(tile_aux.get(tname, 2))
            if not ok:
                return f"wire endpoint '{ep}': unknown port"
        if w.get("from") == w.get("to"):
            return "wire: from == to"
    if len(io.get("wires") or []) > 64:
        return "max 64 wires per app"
    return None


PRIMITIVE_APPS = {
    "counter": {"title": "Counter", "params": [], "source": "@init\nn = 0; n = load(1);\n\n@gfx\nhalf = gfx_w / 2;\ngfx_clear(0.05, 0.04, 0.08, 1);\nmouse_cap && mouse_y < 40 ? (\n  mouse_x < half ? n -= 1 : n += 1;\n  store(1, n);\n);"},
    "timer": {"title": "Timer",
              "params": [{"name": "minutes", "label": "minutes",
                          "min": 1, "max": 60, "default": 5, "step": 1}],
              "source": "@init\nrunning = 0; start_t = 0; left = 0;\n\n@gfx\nleft = running ? max(0, minutes * 60 - (time_precise() - start_t)) : left;\ngfx_clear(0.05, 0.04, 0.08, 1);\ngfx_setfont(26);\ngfx_drawnumber(floor(left / 60), gfx_w / 2 - 26, gfx_h / 2 - 8);"},
    "bmi": {"title": "BMI",
            "params": [{"name": "height", "min": 120, "max": 220,
                        "default": 175},
                       {"name": "weight", "min": 30, "max": 200,
                        "default": 75}],
            "source": "@slider\nh_m = height / 100;\n\n@gfx\nbmi = weight / (h_m * h_m + 0.0001);\ngfx_clear(0.05, 0.04, 0.08, 1);\ngfx_setfont(30);\ngfx_drawnumber(floor(bmi * 10) / 10, gfx_w / 2 - 28, gfx_h * 0.42);"},
    "week": {"title": "Week tracker",
             "params": [{"name": "goal", "label": "goal days",
                         "min": 1, "max": 7, "default": 7, "step": 1}],
             "source": "@init\ndays = 0; days = load(7);\n\n@gfx\ngfx_clear(0.05, 0.04, 0.08, 1);\ncnt = 0; i0 = 0;\nbw = (gfx_w - 20) / 7;\nloop(7,\n  on = (days & pow(2, i0 | 0)) != 0;\n  on ? cnt += 1;\n  i0 += 1;\n);\ngfx_drawnumber(cnt, gfx_w / 2 - 24, gfx_h - 46);"},
    "scope3d": {
        "title": "Lissajous scope (3D)",
        "params": [
            {"name": "fx", "min": 1, "max": 8, "default": 3, "step": 1},
            {"name": "fy", "min": 1, "max": 8, "default": 4, "step": 1},
            {"name": "phase", "min": 0, "max": 6.28, "default": 0.5,
             "step": 0.01},
        ],
        "io": {
            "triggers": [{"name": "start", "label": "Render"}],
            "outs": [{"name": "scene", "kind": "points3d"},
                     {"name": "points", "kind": "number"}],
        },
        "source": (
            "@init\nn = 600;\n\n"
            "@gfx\n"
            "trigger('start') ? (\n"
            "  i = 0;\n"
            "  loop(200,\n"
            "    t = i * 0.0314159;\n"
            "    mem[i * 3] = sin(fx * t + phase);\n"
            "    mem[i * 3 + 1] = sin(fy * t);\n"
            "    mem[i * 3 + 2] = sin((fx + fy) * 0.5 * t) * 0.5;\n"
            "    i += 1;\n"
            "  );\n"
            "  out_points3d('scene', 0, n);\n"
            "  output('points', n);\n"
            ");"),
    },
    "walk": {
        "title": "Random walk",
        "params": [
            {"name": "step_size", "label": "step size",
             "min": 0.1, "max": 10, "default": 1, "step": 0.1},
            {"name": "drift", "min": -2, "max": 2, "default": 0,
             "step": 0.1},
        ],
        "io": {
            "triggers": [{"name": "step", "label": "Step"}],
            "outs": [{"name": "trace", "kind": "series"},
                     {"name": "pos", "kind": "number"}],
        },
        "source": (
            "@init\npos = 0; cnt = 0; i = 0;\n\n"
            "@gfx\n"
            "trigger('step') ? (\n"
            "  pos += (rand(1) * 2 - 1) * step_size + drift;\n"
            "  cnt += 1;\n"
            # fixed 128-wide shifting window (relative frame)
            "  i = 0;\n"
            "  loop(127, mem[1000 + i] = mem[1001 + i]; i += 1;);\n"
            "  mem[1127] = pos;\n"
            "  out_series('trace', 1000, 128);\n"
            "  output('pos', pos);\n"
            ");"),
    },
    "xypad3d": {
        "title": "XY scope (control-driven 3D)",
        "params": [],
        "io": {
            # control standard demo: the xy pad IS the signal source —
            # pos_x/pos_y drive the lissajous ratios live, twist sets
            # phase. No triggers, no params: pure control surface.
            "controls": [
                {"type": "xy", "name": "pos", "address": "/pos"},
                {"type": "fader", "name": "twist", "address": "/twist",
                 "default": 0.3},
            ],
            "outs": [{"name": "scene", "kind": "points3d"},
                     {"name": "ratio", "kind": "number"}],
        },
        "source": (
            "@init\nfc = 0; i = 0;\n\n"
            "@gfx\n"
            "gfx_clear(0.05, 0.04, 0.08, 1);\n"
            "fc += 1;\n"
            "fc % 6 == 0 ? (\n"
            "  fx = 1 + input('pos_x') * 7;\n"
            "  fy = 1 + input('pos_y') * 7;\n"
            "  ph = input('twist') * 6.2832;\n"
            "  i = 0;\n"
            "  loop(200,\n"
            "    t = i * 0.0314159;\n"
            "    mem[i * 3] = sin(fx * t + ph);\n"
            "    mem[i * 3 + 1] = sin(fy * t);\n"
            "    mem[i * 3 + 2] = sin((fx + fy) * 0.5 * t) * 0.5;\n"
            "    i += 1;\n"
            "  );\n"
            "  out_points3d('scene', 0, 600);\n"
            "  output('ratio', floor(fx) + floor(fy) / 10);\n"
            ");"),
    },
    "metronome": {
        "title": "Metronome (clock-sourced)",
        "params": [
            {"name": "accent", "min": 0, "max": 1, "default": 1,
             "step": 1},
        ],
        "io": {
            # trigger source = clock: the gate fires on the SIGNAL
            # (every_s); the manual button toggles the gate open/closed.
            "triggers": [{"name": "beat", "label": "Beat",
                          "source": {"event": "clock", "every_s": 2}}],
            "outs": [{"name": "beats", "kind": "series"},
                     {"name": "count", "kind": "number"}],
        },
        "source": (
            "@init\ncnt = 0; i = 0;\n\n"
            "@gfx\n"
            "trigger('beat') ? (\n"
            "  cnt += 1;\n"
            # fixed 64-wide shifting window: the chart is always a
            # complete relative frame, never a growing buffer
            "  i = 0;\n"
            "  loop(63, mem[2000 + i] = mem[2001 + i]; i += 1;);\n"
            "  mem[2063] = cnt % (2 + accent * 2) == 0 ? 2 : 1;\n"
            "  out_series('beats', 2000, 64);\n"
            "  output('count', cnt);\n"
            ");"),
    },
    "gauge": {
        "title": "Live gauge (bus-fed)",
        "params": [
            {"name": "alpha", "label": "smoothing", "min": 0.01,
             "max": 1, "default": 0.2, "step": 0.01},
        ],
        "io": {
            # live data input: no trigger at all — the shell samples the
            # bus topic continuously and the app emits the current frame
            "ins": [{"name": "sig", "topic": "ship/vllm/toks"}],
            "outs": [{"name": "trace", "kind": "series"},
                     {"name": "now", "kind": "number"}],
        },
        "source": (
            "@init\ny = 0; i = 0; fc = 0;\n\n"
            "@gfx\n"
            "y += alpha * (input('sig') - y);\n"
            "fc += 1;\n"
            "fc % 15 == 0 ? (\n"
            "  i = 0;\n"
            "  loop(63, mem[3000 + i] = mem[3001 + i]; i += 1;);\n"
            "  mem[3063] = y;\n"
            "  out_series('trace', 3000, 64);\n"
            "  output('now', y);\n"
            ");"),
    },
}


def build_jsfx(name: str, title: str | None = None) -> dict:
    prim = PRIMITIVE_APPS.get(name)
    if prim is None:
        raise KeyError(name)
    err = validate_io(prim.get("io"))
    if err:
        return {"error": err}
    t = title or prim["title"]
    program = {"id": f"{name}-{int(time.time())}", "title": t,
               "params": prim["params"], "source": prim["source"]}
    if prim.get("io"):
        program["io"] = prim["io"]
    principle = ("Plugin primitive: declared parameters drive an EEL2 "
                 "program; @gfx renders every frame."
                 + (" Triggers gate on io ports; figures route to the "
                    "viewport matrix." if prim.get("io") else ""))
    inner = ("<div id=mfx></div>"
             "<script>" + jsfx_runtime() + "</script>"
             "<script>var PROGRAM=" + json.dumps(program) + ";"
             "MicroFX.runProgram(document.getElementById('mfx'),"
             "PROGRAM,{fps:30});</script>")
    html = ("<style>body{background:#0d0b12;color:#eba75a;"
            "font-family:Antonio,sans-serif;margin:0;padding:8px}"
            "#mfx canvas{width:100%;height:auto}"
            ".jsfx-triggers{margin-top:6px;display:flex;gap:6px}"
            ".jsfx-triggers button{background:#ff7700;color:#000;"
            "border:0;border-radius:10px 10px 10px 3px;padding:4px 18px;"
            "font-family:Antonio,sans-serif;font-size:14px;"
            "letter-spacing:.06em;cursor:pointer}"
            ".jsfx-triggers button:active{background:#eba75a}"
            ".jsfx-controls{display:flex;gap:10px;align-items:center;"
            "flex-wrap:wrap;margin-top:6px}"
            ".jsfx-control{color:#baa4e5;font-size:11px;display:flex;"
            "flex-direction:column;gap:2px}"
            ".jsfx-control-btn{background:#ff7700;color:#000;border:0;"
            "border-radius:8px;padding:6px 16px;font-family:Antonio,"
            "sans-serif;font-size:13px;letter-spacing:.05em;"
            "cursor:pointer;touch-action:none}"
            ".jsfx-control-btn.on{background:#fcc19f}"
            ".jsfx-xy{touch-action:none;border-radius:6px}</style>"
            "<h1 style='font-size:16px'>" + (t or "") + "</h1>"
            "<p class=principle>" + principle + "</p>"
            + inner)
    return {
        "html": html,
        "fields": [{"name": prm["name"], "type": "number",
                    "value": prm.get("default", 0)}
                   for prm in prim["params"]],
        "io": prim.get("io") or {},
        "principle": principle,
        "template": "html",
        "domain": "tools",
        "group": "command",
        "span": 1,
        "viewport": "both",
        "kernel": name,
    }


def module_catalog() -> dict:
    return {name: {
        "title": mod["title"],
        "params": mod["params"],
        "inputs": mod["inputs"],
        "outputs": mod["outputs"],
        "host": bool(mod.get("host")),
    } for name, mod in MODULES.items()}


def validate_patch(patch: dict) -> str | None:
    mods = patch.get("modules") or []
    if not isinstance(mods, list) or not mods:
        return "patch needs modules[]"
    ids = set()
    for mod in mods[:32]:
        name = (mod.get("primitive") or "").strip()
        if name not in MODULES:
            return f"unknown primitive '{name}'"
        mid = str(mod.get("id") or "")
        if not mid or mid in ids:
            return f"bad/duplicate module id '{mid}'"
        ids.add(mid)
    # NODE RULE: one input port per node, N outputs; fan-out free,
    # multiple cables into one input sum (Rack stackable inputs).
    for wire in patch.get("wires") or []:
        frm, to = str(wire.get("from", "")), str(wire.get("to", ""))
        if "." not in frm or "." not in to:
            return f"wire endpoints must be 'module.port': {frm}->{to}"
        src_id, src_port = frm.split(".", 1)
        dst_id, dst_port = to.split(".", 1)
        if src_id == "ui":
            pass  # ui.tap and friends: virtual sources
        elif src_id in ids:
            outs = MODULES.get(
                (next((m for m in mods if m.get("id") == src_id),
                      {}) or {}).get("primitive") or ""
            ) or {}
            outs = outs.get("outputs") or []
            if src_port not in outs:
                return (f"wire source '{frm}' is not a declared output "
                        f"({', '.join(outs)})")
        else:
            return f"wire references unknown module: {frm}->{to}"
        if dst_id not in ids:
            return f"wire references unknown module: {frm}->{to}"
        dst_prim = (next((m for m in mods if m.get("id") == dst_id),
                         {}) or {}).get("primitive") or ""
        ins = (MODULES.get(dst_prim) or {}).get("inputs") or []
        if dst_port not in ins:
            return (f"wire destination '{to}' is not the node's input "
                    f"port ({', '.join(ins) or 'none'})")
        if not ins:
            return f"node '{dst_id}' ({dst_prim}) takes no input"
    # hierarchy rule: FUNCTION nodes declare at most one input unless
    # flagged multi_in (boolean/analog/quantum gates, per operator
    # rule revision); VISUALIZERS may stack multiple (sinks)
    for mod in mods:
        prim = MODULES.get(mod.get("primitive") or "") or {}
        if prim.get("category") == "function" \
                and not prim.get("multi_in") \
                and len(prim.get("inputs") or []) > 1:
            return (f"function node '{mod.get('id')}' "
                    f"({mod.get('primitive')}) must have one input")
    return None


def build_patch_html(patch: dict, title: str | None = None) -> dict:
    """Kernel-style builder: hosts the patch runner + optional viz tiles."""
    err = validate_patch(patch)
    if err:
        return {"error": err}
    t = title or "MicroFX patch"
    program = json.dumps({"patch": patch}, default=str)
    src_map = {name: mod["source"] for name, mod in MODULES.items()}
    prm_map = {name: mod["params"] for name, mod in MODULES.items()}
    inner = ("<div id=mfx class=mfx-patch></div>"
             "<script>window.__MFX_MODULES_SOURCE=" +
             json.dumps(src_map) + ";"
             "window.__MFX_MODULES_PARAMS=" + json.dumps(prm_map) + ";"
             "</script>"
             "<script>" + jsfx_runtime() + "</script>"
             "<script>var PATCH=" + program + ";"
             "MicroFX.runPatch(document.getElementById('mfx'),PATCH,"
             "{fps:30});</script>")
    principle = ("Modular patch: typed modules, wired outputs to inputs; "
                 "views render designated signals.")
    esc = (lambda x: (x or "").replace("&", "&amp;")
           .replace("<", "&lt;").replace(">", "&gt;"))
    html = ("<!DOCTYPE html><html><head><meta charset=utf-8>"
            "<style>body{background:#0d0b12;color:#eba75a;"
            "font-family:Antonio,sans-serif;margin:0;padding:8px}"
            ".principle{font-size:11px;color:#baa4e5}"
            ".mfx-patch .view{background:#000;border-radius:8px;"
            "margin-bottom:6px}"
            ".mfx-patch canvas{width:100%;height:auto;display:block}"
            ".jsfx-params label{display:block;font-size:11px;"
            "color:#baa4e5;margin-top:4px}"
            ".jsfx-params input{width:100%}</style></head><body>"
            "<h1 style='font-size:16px;margin:4px 0'>" + esc(t) + "</h1>"
            "<p class=principle>" + esc(principle) + "</p>"
            + inner + "</body></html>")
    fields = []
    for mod in patch.get("modules") or []:
        for prm in (MODULES.get(mod.get("primitive")) or {}).get(
                "params") or []:
            fields.append({"name": f"{mod['id']}.{prm['name']}",
                           "type": "number",
                           "value": prm.get("default", 0)})
    return {
        "html": html,
        "fields": fields,
        "principle": principle,
        "template": "html",
        "domain": "tools",
        "group": "command",
        "span": 2,
        "viewport": "both",
        "kernel": "patch",
    }
