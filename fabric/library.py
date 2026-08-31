"""App library — working apps, function catalog, stale culling.

The library separates WHAT CAN BE USED from WHAT WAS BUILT:
  functions()  the catalog under the new standards (sources /
               functions / controls / visualizers, from microfx +
               appwiz) — available in build/edit mode
  apps()       constructed applications (validated saved specs),
               loadable into viewports
  audit/cull   the store accumulates e2e probes and older duplicates;
               audit identifies stale entries, cull removes them
               (confirm-gated)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import appwiz
from .microapps import public_spec, validate

_PROBE_PAT = re.compile(r"(_probe(_[a-z0-9]+)?|_e2e|_test)\b|_probe_\w*\d")


def _root() -> Path:
    from .microapps import _root as mroot

    return mroot()


def _base_name(app_id: str) -> str:
    """Normalize an id to its family name for duplicate detection."""
    s = re.sub(r"[_ ]?\d{10,}$", "", app_id)          # timestamps
    s = re.sub(r"(_\d+)+$", "", s)                     # trailing counters
    s = _PROBE_PAT.sub("", s)
    return s.strip("_") or app_id


def _load_all() -> list[dict]:
    root = _root()
    out = []
    for p in sorted(root.glob("*.json")):
        try:
            spec = json.loads(p.read_text(encoding="utf-8"))
            spec["_mtime"] = p.stat().st_mtime
            spec["_path"] = str(p)
            out.append(spec)
        except Exception:  # noqa: BLE001
            out.append({"id": p.stem, "_path": str(p), "_broken": True})
    return out


def audit() -> dict:
    """Classify the store: keep newest per family, flag probes/broken."""
    specs = _load_all()
    stale: list[str] = []
    families: dict[str, list[dict]] = {}
    for spec in specs:
        sid = spec.get("id") or ""
        if spec.get("_broken"):
            stale.append(sid)
            continue
        if _PROBE_PAT.search(sid):
            stale.append(sid)
            continue
        err = validate({k: v for k, v in spec.items()
                        if not k.startswith("_")})
        if err:
            stale.append(sid)
            continue
        families.setdefault(_base_name(sid), []).append(spec)
    keep: list[str] = []
    for _base, group in families.items():
        group.sort(key=lambda s: s.get("_mtime", 0), reverse=True)
        keep.append(group[0]["id"])
        stale.extend(s["id"] for s in group[1:])
    return {"total": len(specs), "keep": sorted(keep),
            "stale": sorted(set(stale) - set(keep))}


def cull(confirm: bool = False) -> dict:
    """Delete stale entries (audit list). Requires confirm=True."""
    a = audit()
    if not confirm:
        return {"would_delete": a["stale"], "keep": a["keep"],
                "note": "call with confirm=true to delete"}
    removed = []
    for sid in a["stale"]:
        for p in _root().glob(f"{sid}.json"):
            try:
                p.unlink()
                removed.append(sid)
            except OSError:
                pass
    from .microapps import _cache

    for sid in removed:
        _cache.pop(sid, None)
    return {"removed": removed, "kept": a["keep"]}


def apps() -> list[dict]:
    """Working applications (newest per family), loadable specs."""
    a = audit()
    keep = set(a["keep"])
    out = []
    for spec in _load_all():
        sid = spec.get("id") or ""
        if sid not in keep:
            continue
        pub = public_spec({k: v for k, v in spec.items()
                           if not k.startswith("_")})
        pub["kernel_kind"] = (
            "signal" if spec.get("kernel") == "signal"
            else "patch" if spec.get("kernel") == "patch"
            else "microfx" if spec.get("kernel") in
            ("scope3d", "walk", "metronome", "gauge", "xypad3d",
             "counter", "timer", "bmi", "week")
            else "dom")
        out.append(pub)
    out.sort(key=lambda s: s["title"].lower())
    return out


def functions() -> dict:
    """Function + control library for build/edit mode."""
    from .microfx import MODULES

    cat: dict[str, list[dict]] = {"source": [], "function": [],
                                  "control": [], "visualizer": []}
    for name, mod in MODULES.items():
        entry = {"id": name, "title": mod["title"],
                 "inputs": mod.get("inputs") or [],
                 "outputs": mod.get("outputs") or [],
                 "multi_in": bool(mod.get("multi_in")),
                 "params": [p["name"] for p in mod.get("params") or []]}
        cat.setdefault(mod.get("category") or "function",
                       []).append(entry)
    return {
        "nodes": cat,
        "controls": appwiz.CONTROLS,
        "sources": appwiz.SOURCES,
        "visualizers": appwiz.VISUALIZERS,
    }
