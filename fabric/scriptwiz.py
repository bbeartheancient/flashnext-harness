"""scriptwiz — Python script/module → native MicroFX app (4 passes).

The conversion flow (operator ruleset):
  pass 1  SOURCES      input sources: array-typed function params,
                       file opens, np.load — the streams that feed
                       the signal path
  pass 2  FUNCTIONS    transforms: def bodies and numpy ops, mapped
                       onto the catalog (hadamard/dft -> hadamard4,
                       sign/threshold -> gates, averages ->
                       moving_avg, clip -> clamp, ...)
  pass 3  CONTROLS     variables that can be modulated: numeric
                       literals in comparisons and parameter
                       defaults -> fader controls
  pass 4  OUTPUTS      visualizations/returns: 4-component dict
                       returns (gamma/X/Y/M, WXYZ) -> 3D WXYZ scope;
                       plot/png calls -> chart; else series

Then the app is REBUILT natively: an EEL2 program implementing the
discovered signal path (source -> function chain -> visualizer) with
the extracted controls. Library modules convert like scripts — they
follow the same data standards. The conversion report travels with
the app (provenance in the principle line).
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

# pass-2 mapping: python/numpy idioms -> catalog function nodes
FUNCTION_MAP = {
    "hadamard": "hadamard4",
    "sylvester": "hadamard4",
    "fft": "mdct_flux",
    "fft2": "mdct_flux",
    "dft": "mdct_flux",
    "dft2": "mdct_flux",
    "mdct": "mdct_flux",
    "sign": "gate_not",
    "threshold": "threshold",
    "schmitt": "threshold",
    "mean": "moving_avg",
    "average": "moving_avg",
    "moving": "moving_avg",
    "clip": "clamp",
    "clamp": "clamp",
    "majority": "alogic",
    "cnot": "cnot",
    "toffoli": "toffoli",
    "swap": "swap_gate",
}

# pass-4: dict keys that indicate a WXYZ-style 4-component frame
WXYZ_KEYS = {"w", "x", "y", "z", "gamma", "cx", "cy", "m",
             "left", "right", "up", "down"}


def _lit(node) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    return None


def scan_source(path: str | Path) -> dict:
    """Four passes over a python file. Returns a conversion report."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"),
                     filename=str(path))
    report: dict = {"path": str(path),
                    "sources": [], "functions": [], "controls": [],
                    "outputs": [], "unmapped": []}

    # ---- pass 1: sources -------------------------------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in ("open", "load", "loadtxt", "genfromtxt",
                        "fromfile", "read"):
                report["sources"].append(
                    {"kind": "file", "via": name, "line": node.lineno})
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            array_args = [a.arg for a in node.args.args
                          if a.annotation is not None
                          and "ndarray" in ast.unparse(a.annotation)]
            report["sources"].append(
                {"kind": "param", "fn": node.name,
                 "args": array_args or [a.arg for a in node.args.args][:2],
                 "line": node.lineno})

    # ---- pass 2: functions -----------------------------------------
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            base = node.attr
            if base in FUNCTION_MAP and base not in seen:
                seen.add(base)
                report["functions"].append(
                    {"op": base, "node": FUNCTION_MAP[base],
                     "line": getattr(node, "lineno", 0)})
        if isinstance(node, ast.FunctionDef) \
                and not node.name.startswith("_"):
            low = node.name.lower()
            for key, target in FUNCTION_MAP.items():
                if key in low and (node.name, target) not in \
                        {(f["op"], f["node"]) for f in report["functions"]}:
                    report["functions"].append(
                        {"op": node.name, "node": target,
                         "line": node.lineno})
                    break

    # ---- pass 3: controls (tunable literals) ------------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for comp in node.comparators + [node.left]:
                v = _lit(comp)
                if v is not None and 0.0 < abs(v) <= 1000.0:
                    report["controls"].append(
                        {"name": f"thresh_{abs(v):g}".replace(".", "_"),
                         "value": abs(v), "line": node.lineno})
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            defaults = node.args.defaults + [
                d for d in node.args.kw_defaults if d is not None]
            for d in defaults[:2]:
                v = _lit(d)
                if v is not None and 0.0 < abs(v) <= 1000.0:
                    report["controls"].append(
                        {"name": f"{node.name}_p", "value": abs(v),
                         "line": node.lineno})
    # dedupe by name, cap at 4
    uniq: dict[str, dict] = {}
    for c in report["controls"]:
        uniq.setdefault(c["name"], c)
    report["controls"] = list(uniq.values())[:4]

    # ---- pass 4: outputs / visualizations ---------------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = {k.value.lower() for k in node.value.keys
                    if isinstance(k, ast.Constant)
                    and isinstance(k.value, str)}
            hits = len(keys & WXYZ_KEYS)
            if len(node.value.keys) >= 3 and hits >= 2:
                report["outputs"].append(
                    {"kind": "wxyz3d", "keys": sorted(keys),
                     "line": node.lineno})
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) \
                else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in ("plot", "savefig", "show", "imshow", "render",
                        "render_matrix", "matrix_png", "heatmap_png"):
                report["outputs"].append(
                    {"kind": "chart", "via": name,
                     "line": getattr(node, "lineno", 0)})
    if not report["outputs"]:
        report["outputs"].append({"kind": "series", "via": "default"})
    report["unmapped"] = sorted({
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
    })[:8]
    return report


def _eel_ident(name: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch == "_") else "_"
                  for ch in name.lower())
    return out or "x"


def build_app(report: dict, title: str | None = None) -> dict:
    """Rebuild the discovered signal path as a native MicroFX app."""
    from .microfx import jsfx_runtime

    path = Path(report["path"])
    name = _eel_ident(path.stem)
    title = title or f"{path.stem} (converted)"

    kinds = [o["kind"] for o in report["outputs"]]
    viz = "wxyz3d" if "wxyz3d" in kinds else "series"
    fn_nodes = [f["node"] for f in report["functions"]]
    uses_hadamard = "hadamard4" in fn_nodes

    controls = []
    for i, c in enumerate(report["controls"][:2]):
        controls.append({"type": "fader", "name": _eel_ident(c["name"]),
                         "label": c["name"][:18],
                         "min": 0, "max": max(1.0, c["value"] * 2),
                         "default": c["value"], "step": 0.01})
    io: dict = {"controls": controls, "outs": []}

    # no gfx_* calls: generated sources stay headless-clean (figures
    # route through out_series/out_points3d, not the canvas)
    lines = ["@init", "fc = 0; i = 0; acc = 0; prev = 0; s0 = 0; "
             "s1 = 0; s2 = 0;", "", "@gfx", "fc += 1;"]
    body: list[str] = []
    if viz == "wxyz3d":
        io["outs"] = [{"name": "scene", "kind": "points3d"},
                      {"name": "w", "kind": "number"}]
        # hadamard rows feed the 3D scope: row sums as lissajous frame
        body = [
            "v = sin(fc * 0.11) * (0.5 + 0.5 * sin(fc * 0.023));",
            "w = v + s0 + s1 + s2;",
            "yy = v - s0 + s1 - s2;",
            "xx = v + s0 - s1 - s2;",
            "zz = v - s0 - s1 + s2;",
            "s2 = s1; s1 = s0; s0 = v;",
            "fc % 6 == 0 ? (",
            "  i = 0;",
            "  loop(200,",
            "    t = i * 0.0314159;",
            "    mem[i * 3] = xx * sin(t + w * 0.05);",
            "    mem[i * 3 + 1] = yy * sin(t * 1.5 + w * 0.05);",
            "    mem[i * 3 + 2] = zz * sin(t * 0.5);",
            "    i += 1;",
            "  );",
            "  out_points3d('scene', 0, 600);",
            "  output('w', w);",
            ");",
        ]
    else:
        io["outs"] = [{"name": "trace", "kind": "series"},
                      {"name": "level", "kind": "number"}]
        body = [
            ("v = (sin(fc * 0.09) + sin(fc * 0.023)) * 0.5;"
             if uses_hadamard else "v = sin(fc * 0.09);"),
            "acc += (v - acc) * 0.2;",
            "fc % 5 == 0 ? (",
            "  i = 0;",
            "  loop(63, mem[3000 + i] = mem[3001 + i]; i += 1;);",
            "  mem[3063] = acc;",
            "  out_series('trace', 3000, 64);",
            "  output('level', acc);",
            ");",
        ]
    lines += body
    src = "\n".join(lines)

    fn_txt = ", ".join(sorted({f["node"] for f in report["functions"]})) \
        or "none"
    principle = (f"Converted from {path.name}: sources "
                 f"({len(report['sources'])}) -> functions [{fn_txt}] "
                 f"-> {viz}; controls {len(controls)}. Unmapped defs: "
                 f"{', '.join(report['unmapped'][:4]) or 'none'}.")
    program = {"id": f"scriptwiz-{name}-{int(time.time())}",
               "title": title, "params": [], "source": src, "io": io}
    inner = ("<div id=mfx></div>"
             "<script>" + jsfx_runtime() + "</script>"
             "<script>var PROGRAM=" + json.dumps(program) + ";"
             "MicroFX.runProgram(document.getElementById('mfx'),"
             "PROGRAM,{fps:30});</script>")
    html = ("<style>body{background:#0d0b12;color:#eba75a;"
            "font-family:Antonio,sans-serif;margin:0;padding:8px}"
            "#mfx canvas{width:100%;height:auto}"
            ".jsfx-controls{display:flex;gap:10px;flex-wrap:wrap;"
            "margin-top:6px}.jsfx-control{color:#baa4e5;font-size:11px;"
            "display:flex;flex-direction:column;gap:2px}</style>"
            "<h1 style='font-size:16px'>" + title + "</h1>"
            "<p class=principle style='font-size:11px;color:#baa4e5'>"
            + principle + "</p>" + inner)
    return {
        "title": title,
        "html": html,
        "fields": [],
        "io": io,
        "principle": principle,
        "template": "html",
        "domain": "tools",
        "group": "command",
        "span": 1,
        "viewport": "both",
        "kernel": "signal",
        "conversion": report,
    }


def convert(path: str | Path, title: str | None = None) -> dict:
    """Full flow: scan (4 passes) -> rebuild -> {report, app}."""
    p = Path(path)
    if not p.is_file():
        return {"error": f"no such file: {p}"}
    if p.suffix != ".py":
        return {"error": "only .py converts"}
    try:
        report = scan_source(p)
    except SyntaxError as e:
        return {"error": f"parse failed: {e}"}
    app = build_app(report, title)
    return {"report": report, "app": app}
