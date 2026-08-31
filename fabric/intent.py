"""Resolve a free-text command into a console function + field overrides.

The LCARS shell treats the command line as the menu: user input
procedurally generates the box input-column and MiniApp tiles.
Menus (full function rails) stay off the default path.
"""

from __future__ import annotations

import re

from . import console

# Extra trigger phrases beyond id/title. Keep this the ABI for the
# command line so the catalog file does not grow a synonym column.
_ALIASES = {
    "chat": ("computer", "ask", "say"),
    "ship": ("status", "ship status", "power", "gpu"),
    "eng_ship": ("engineering status",),
    "log": ("log", "journal", "recent log"),
    "hadamard": ("hadamard", "h(", "sylvester", "walsh matrix"),
    "hoa_encode": ("hoa encode", "encode hoa", "ambisonic", "hoa field"),
    "hoa_decode": ("hoa decode", "decode hoa"),
    "hoa_rotate": ("hoa rotate", "rotate hoa", "yaw"),
    "orbital": ("orbital", "psi", "hydrogen", "2p"),
    "antenna": ("antenna", "yagi", "dipole", "monopole", "helix", "patch",
                "loop"),
    "filter": ("filter", "lpf", "hpf", "bpf", "bsf", "s21"),
    "link": ("link budget", "friis", "fspl"),
    "survey": ("survey", "terrain", "path profile", "deygout"),
    "los": ("los", "line of sight", "line-of-sight"),
    "place": ("city of", "map of", "show me", "where is", "neosho"),
    "horizon": ("horizon", "coverage fan"),
    "room": ("room ir", "room-ir", "impulse", "rt60", "sabine"),
    "fdtd": ("fdtd", "yee"),
    "materials": ("materials", "cloth", "metamaterial", "touchpad"),
    "scales": ("actual size", "press 1980", "scales"),
    "crown": ("crown", "psf"),
    "sage": ("sage", "cas", "sympy", "factor", "integrate", "simplify"),
    "sensors": ("sensor", "mqtt", "hwmon"),
    "compose": ("make a", "build a", "create a", "miniapp", "microapp",
                "new app", "new function"),
}

_GROUP_EXT = {"command": 100, "science": 400}

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(q: str) -> set[str]:
    return set(_WORD.findall(q.lower()))


def _score(fn: dict, q: str, toks: set[str]) -> int:
    low = q.lower()
    score = 0
    ident = fn["id"].lower().replace("_", " ")
    title = fn["title"].lower()
    if re.search(rf"\b{re.escape(fn['id'])}\b", low, re.I):
        score += 6
    if title and title in low:
        score += 5
    for alias in _ALIASES.get(fn["id"], ()):
        if alias in low:
            score += 4
            break
    title_toks = set(_WORD.findall(title)) | set(_WORD.findall(ident))
    # digits and 1-2 char fragments ("2", "x") must not score app titles
    # or ids — "factor(x**2-1)" once matched app_..._optimizer_2
    overlap = {t for t in (title_toks & toks)
               if len(t) >= 3 and not t.isdigit()}
    overlap -= {"a", "an", "the", "to", "for", "of", "and", "or", "me",
                "show", "in", "at", "on", "city", "town"}
    score += len(overlap)
    if fn.get("generated") and overlap:
        score += 3 + len(overlap)
    return score


def _coerce(field: dict, raw: str):
    if field["type"] == "number":
        try:
            n = float(raw)
            return int(n) if n.is_integer() else n
        except ValueError:
            return None
    return raw


def _extract(fn: dict, q: str) -> dict:
    fields: dict = {}
    low = q.lower()
    for f in fn.get("fields") or []:
        for opt in f.get("options") or []:
            if re.search(rf"\b{re.escape(str(opt))}\b", low):
                fields[f["name"]] = opt
    for m in re.finditer(r"(\w+)\s*=\s*([^\s,]+)", q):
        name = m.group(1)
        spec = next((f for f in fn.get("fields") or [] if f["name"] == name), None)
        if spec:
            val = _coerce(spec, m.group(2))
            if val is not None:
                fields[name] = val
    nums = re.findall(r"-?\d+\.?\d*", q)
    if fn["id"] in ("survey", "los"):
        geo = re.search(
            r"from\s+(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s+to\s+"
            r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)",
            q, re.I)
        if geo:
            fields["tx_lat"] = float(geo.group(1))
            fields["tx_lon"] = float(geo.group(2))
            fields["rx_lat"] = float(geo.group(3))
            fields["rx_lon"] = float(geo.group(4))
            nums = []
    nfields = [f for f in fn.get("fields") or []
               if f["type"] == "number" and f["name"] not in fields]
    for spec, raw in zip(nfields, nums):
        val = _coerce(spec, raw)
        if val is not None:
            fields[spec["name"]] = val
    if fn["id"] == "sage":
        expr = re.sub(r"^(sage|cas|sympy)\b[:\s]*", "", q, flags=re.I).strip()
        if expr:
            fields["expr"] = expr
    if fn["id"] == "chat":
        fields["message"] = q
    if fn["id"] == "compose":
        fields["query"] = q
    if fn["id"] == "place":
        from .microapps import extract_place, looks_like_place

        fields["q"] = extract_place(q) if looks_like_place(q) else q
    return fields


def _chat_match(q: str) -> dict:
    chat = next(f for f in console.FUNCTIONS if f["id"] == "chat")
    return {
        "query": q,
        "match": {k: chat[k] for k in
                  ("id", "title", "group", "method", "url", "viewport",
                   "fields") if k in chat},
        "fields": {"message": q},
        "autorun": True,
        "candidates": [],
        "fallback": "chat",
        "department": 100,
    }


def resolve(q: str, prefer_chat: bool = False) -> dict:
    q = (q or "").strip()
    if not q:
        return {"query": q, "match": None, "fields": {}, "autorun": False,
                "candidates": [], "fallback": None, "department": None}
    if q.startswith("/"):
        from . import commands

        return commands.resolve(q)
    low = q.lower()
    if prefer_chat and low not in ("help", "index", "directory", "0", "menu", "?"):
        return _chat_match(q)
    if low in ("help", "index", "directory", "0", "menu", "?"):
        titles = [{"id": f["id"], "title": f["title"], "group": f["group"]}
                  for f in console.FUNCTIONS]
        return {"query": q, "match": None, "fields": {}, "autorun": False,
                "candidates": titles, "fallback": "help", "department": None,
                "hint": "type a function name; the box generates its I/O"}
    toks = _tokens(q)
    from .microapps import (
        extract_place, looks_like_generate, looks_like_place, propose,
    )

    if looks_like_place(q):
        place_fn = next(f for f in console.FUNCTIONS if f["id"] == "place")
        return {
            "query": q,
            "match": {k: place_fn[k] for k in
                      ("id", "title", "group", "method", "url", "viewport",
                       "fields") if k in place_fn},
            "fields": {"q": extract_place(q), "view": "terrain"},
            "autorun": True,
            "candidates": [],
            "fallback": None,
            "department": 400,
        }

    catalog = console.catalog()["functions"]
    ranked = []
    for fn in catalog:
        if fn["id"] == "compose":
            continue
        s = _score(fn, q, toks)
        if s:
            ranked.append((s, fn))
    ranked.sort(key=lambda t: t[0], reverse=True)
    compose_fn = next(f for f in console.FUNCTIONS if f["id"] == "compose")
    best_existing = ranked[0][0] if ranked else 0
    if looks_like_generate(q) and best_existing < 5:
        from .microapps import pick_kernel

        kn = pick_kernel(q)
        if kn != "shell":
            return {
                "query": q,
                "match": {k: compose_fn[k] for k in
                          ("id", "title", "group", "method", "url", "viewport",
                           "fields") if k in compose_fn},
                "fields": {"query": q},
                "autorun": True,
                "candidates": [{"id": f["id"], "title": f["title"], "score": s}
                               for s, f in ranked[:3]],
                "fallback": None,
                "department": 100,
            }
        return {
            "query": q,
            "match": {k: compose_fn[k] for k in
                      ("id", "title", "group", "method", "url", "viewport",
                       "fields") if k in compose_fn},
            "fields": {"query": q},
            "autorun": False,
            "wizard": True,
            "candidates": [{"id": f["id"], "title": f["title"], "score": s}
                           for s, f in ranked[:3]],
            "fallback": None,
            "department": 100,
            "hint": "set department, input, output, viewports",
        }
    if not ranked:
        prop = propose(q)
        if prop.get("auto") and prop["auto"]["kind"] == "fn":
            fn = next((f for f in catalog if f["id"] == prop["auto"]["id"]
                       or f["id"] == prop["auto"]["id"].replace("show_", "")), None)
            if fn:
                return {
                    "query": q, "match": {k: fn[k] for k in
                        ("id", "title", "group", "method", "url", "viewport",
                         "fields") if k in fn},
                    "fields": prop["auto"].get("fields") or {},
                    "autorun": True, "candidates": prop["choices"][:3],
                    "fallback": None,
                    "department": _GROUP_EXT.get(fn.get("group")),
                }
        if prop.get("need_clarify") or prop.get("choices"):
            return {
                "query": q,
                "match": {k: compose_fn[k] for k in
                          ("id", "title", "group", "method", "url", "viewport",
                           "fields") if k in compose_fn},
                "fields": {"query": q},
                "autorun": False,
                "wizard": True,
                "clarify": {"choices": prop["choices"],
                            "hint": "set department, input, output, viewports"},
                "candidates": prop["choices"][:3],
                "fallback": None,
                "department": 100,
            }
        return _chat_match(q)
    best, fn = ranked[0][0], ranked[0][1]
    close = [f for s, f in ranked if s >= best - 1][:3]
    # Prefer the more specific id when scores tie (hoa_rotate vs antenna "yaw")
    if len(close) > 1 and ranked[1][0] == best:
        # still take best unless the gap is zero and query names another id
        for s, other in ranked[1:]:
            if s == best and other["id"] in low.replace(" ", "_"):
                fn = other
                break
    fields = _extract(fn, q)
    autorun = (not fn.get("fields")) or bool(fields)
    if fn["id"] == "chat":
        autorun = True
        fields.setdefault("message", q)
    if fn.get("generated") and looks_like_generate(q):
        autorun = True
    # Weak name-only match: generate the form, do not fire yet.
    if best < 4 and not fields:
        autorun = False
    if best < 3 and fn["id"] != "chat":
        out = _chat_match(q)
        out["candidates"] = [{"id": f["id"], "title": f["title"], "score": s}
                             for s, f in ranked[:3]]
        return out
    return {
        "query": q,
        "match": {k: fn[k] for k in
                  ("id", "title", "group", "method", "url", "viewport",
                   "fields") if k in fn},
        "fields": fields,
        "autorun": autorun,
        "candidates": [{"id": f["id"], "title": f["title"], "score": s}
                       for s, f in ranked[:3]],
        "fallback": None,
        "department": _GROUP_EXT.get(fn.get("group")),
    }
