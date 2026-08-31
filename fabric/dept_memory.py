"""Department memory shards — memvid .mv2, one file per department.

Each department keeps its own single-file memory (persona + knowledge
scope + curated facts). The duty agent gets retrieval tools; /chat
injects a bounded context block when a department is pinned. Shards
live under fabric/data/memory/ (gitignored).

SDK: memvid-sdk (MV2 single-file format — WAL, lex+vec indexes).
Import failures degrade to honest error dicts.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .departments import DEPARTMENTS

_DIR = Path(os.environ.get(
    "FABRIC_MEMORY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "data", "memory")))

_lock = threading.Lock()
_handles: dict[str, object] = {}

# Persona + knowledge scope per department (mainUIDdefault.md scopes).
DEPT_SEEDS: dict[str, str] = {
    "Command": (
        "COMMAND is the operator's line to the ship computer. Scope: "
        "chat routing, tok/s + GPU power telemetry, backend port status, "
        "coding agents and goals, admin settings and credentials "
        "(harness.json, media-keys.env stay masked). Tone: terse status "
        "reports."),
    "Navigation": (
        "NAVIGATION explores maps and the solar system. Owns the 3D "
        "solar-system globe (Keplerian ephemeris via fabric/ephem.py), "
        "earth terrain navigation, and the geolocated news feed overlay. "
        "Positions are astrometric; gate against JPL Horizons."),
    "Sciences": (
        "SCIENCES holds the labs: hoa64 Hadamard/HOA/room-IR/RF "
        "calculators (fabric/lab.py), RTU terrain propagation "
        "(fabric/geo.py), Sage/SymPy CAS, orbital probes. Visualizer "
        "kernel kit lives in fabric/web. Locked rule: heavy search "
        "(micromag/gerzon) stays host-side."),
    "Medical": (
        "MEDICAL runs the First Aid body-map wizard (microapps bodymap "
        "kernel). Always answer with practical first-aid guidance plus a "
        "clear 'not a doctor' disclaimer; escalate emergencies to real "
        "services. Supplementary first-aid LLM is parked pending model "
        "availability."),
    "Security": (
        "SECURITY watches the LAN: Orbi RBR50 attached-device map "
        "(fabric/net.py, SOAP jwt_local session), port sweep, CCTV feeds "
        "(gated on D05 hardware). Device data is observational only; no "
        "blocking actions without operator confirm."),
    "Operations": (
        "OPERATIONS is finance + personal organization: Firefly III net "
        "worth (read-only, FIREFLY_TOKEN), stock tickers (yahoo keyless), "
        "calendar/reminders, meal planner microapp. Trading stays parked "
        "per operator ruling."),
    "Media": (
        "MEDIA covers the library and studio plumbing: AI Backlot job "
        "spine (fabric/backlot.py), Jellyfin/*arr stack on .43, Synology "
        "NAS on .44. Submits to Backlot are confirm-gated; never restart "
        "ComfyUI or duty vLLM from this lane."),
    "Holodeck": (
        "HOLODECK is experimental: sensor-bus explorer over MQTT topics, "
        "3D render outputs via Backlot mesh jobs, Quest 2 VR link status "
        "(spatial-xr client). Edge-device feeds stay gated until "
        "hardware lands."),
}


def _slug(dept: str) -> str:
    return (dept or "").strip().lower().replace(" ", "_") or "command"


def _resolve_name(dept: str) -> str:
    """Accept ext number, id-ish slug, or display name."""
    d = (dept or "").strip()
    for row in DEPARTMENTS:
        if d in (str(row.get("ext")), row["id"], row["name"].lower(),
                 _slug(row["name"])):
            return row["name"]
    return d.title() if d else "Command"


def _path(name: str) -> Path:
    return _DIR / f"{_slug(name)}.mv2"


def _open(name: str):
    """Open-or-create a shard; cached handle per department."""
    import memvid_sdk as m  # lazy import

    path = _path(name)
    h = _handles.get(path.name)
    if h is not None:
        return h
    _DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        h = m.create(str(path))
        seed_text = DEPT_SEEDS.get(name)
        if seed_text:
            try:
                h.put(title=f"{name} charter", text=seed_text,
                      uri=f"mv2://charter/{_slug(name)}",
                      tags=["charter"])
                h.commit()
            except Exception:
                pass
    else:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = m.use("basic", str(path))
    _handles[path.name] = h
    return h


def available() -> bool:
    try:
        import memvid_sdk  # noqa: F401
        return True
    except ImportError:
        return False


def add(dept: str, text: str, title: str | None = None,
        tags: dict | None = None) -> dict:
    """Append one memory card to a department shard."""
    name = _resolve_name(dept)
    text = (text or "").strip()
    if not text:
        return {"error": "empty text"}
    if not available():
        return {"error": "memvid-sdk not installed"}
    try:
        with _lock:
            h = _open(name)
            h.put(title=(title or text[:48]).strip()[:120], text=text[:8000],
                  uri=f"mv2://{_slug(name)}/{int(time.time()*1000)}",
                  tags=[str(k) for k in (tags or {})][:6] or None)
            h.commit()
        return {"ok": True, "dept": name}
    except Exception as e:  # noqa: BLE001
        return {"error": f"shard write failed: {e}"}


def search(dept: str, query: str, k: int = 4) -> dict:
    """Lex/vec recall from a department shard."""
    name = _resolve_name(dept)
    query = (query or "").strip()
    if not query:
        return {"error": "empty query"}
    if not available():
        return {"error": "memvid-sdk not installed"}
    try:
        with _lock:
            h = _open(name)
            res = h.find(query, k=max(1, min(int(k), 10)),
                         snippet_chars=480)
        hits = []
        items = res.get("hits") if isinstance(res, dict) else             (getattr(res, "hits", None) or [])
        for hit in items:
            get = (lambda k, o=hit: o.get(k)) if isinstance(hit, dict) else \
                (lambda k, o=hit: getattr(o, k, None))
            snip = str(get("snippet") or "")
            for marker in (" extractous_metadata:", " uri: mv2://",
                           " labels:", " tags:"):
                cut = snip.find(marker)
                if cut > 0:
                    snip = snip[:cut]
            hits.append({"title": (get("title") or ""),
                         "text": snip[:500], "score": get("score")})
        return {"dept": name, "hits": hits}
    except Exception as e:  # noqa: BLE001
        return {"error": f"shard read failed: {e}"}


def context_for(dept: str, query: str,
                budget_chars: int = 1200) -> str:
    """Bounded recall block for system-prompt injection ('' if none)."""
    out = search(dept, query, k=4)
    hits = out.get("hits") or []
    lines: list[str] = []
    used = 0
    for h in hits:
        line = f"- {(h.get('title') or '').strip()}: {h.get('text', '')}"
        line = " ".join(line.split())[:400]
        if used + len(line) > budget_chars:
            break
        lines.append(line)
        used += len(line)
    charter = DEPT_SEEDS.get(dept if dept in DEPT_SEEDS
                             else _resolve_name(dept))
    block = ""
    if charter:
        block += charter + "\n"
    if lines:
        block += "Department memory:\n" + "\n".join(lines)
    return block.strip()


def status() -> dict:
    rows = []
    for row in DEPARTMENTS:
        p = _path(row["name"])
        rows.append({"dept": row["name"], "ext": row["ext"],
                     "shard": p.name, "exists": p.exists(),
                     "bytes": p.stat().st_size if p.exists() else 0})
    return {"dir": str(_DIR), "sdk": available(), "shards": rows}


def seed_docs(shard: str, docs: list[dict]) -> dict:
    """Create/populate a knowledge shard from {title,text} docs.
    Seeds only when the shard file does not yet exist."""
    name = _resolve_name(shard)
    path = _path(name)
    if path.exists():
        return {"ok": True, "seeded": False, "reason": "exists"}
    if not available():
        return {"error": "memvid-sdk not installed"}
    try:
        with _lock:
            h = _open(name)
            for doc in docs:
                h.put(title=doc.get("title", "")[:120],
                      text=(doc.get("text") or "")[:8000],
                      uri=f"mv2://{_slug(name)}/{_slug(doc.get('title',''))}",
                      tags=["guide"])
            h.commit()
        return {"ok": True, "seeded": True, "docs": len(docs)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"seed failed: {e}"}
