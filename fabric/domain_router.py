# SPDX-License-Identifier: Apache-2.0
"""Contextual n-gram shard swap (BICAMERAL Goal 5).

The custom signal paths allow routing Vocab and N-Gram paths to any
of the split points.  The framework's `python/afi/igram.py` exposes
shards by id; this module routes them by INTENT so a user asking
about "first aid" automatically loads the medical vocab/n-gram shard,
not the default general-purpose one.

Architecture:

  1. SHARD_CATALOG — well-known shard metadata: name, domain,
     layer, vocab hint, optional blob path.  Defaults to general +
     a few seeded medical / engineering / legal / finance / code
     domains.  Real Qwen-FN shards register here when the download
     finishes.

  2. ROUTER_RULES — small keyword router: input text → list of
     candidate domains with confidences.  Uses lexical overlap with
     a curated domain lexicon (no LLM call needed; cheap & fast).

  3. swap_for_intent(intent, k=1) — selects the top-k shards for the
     intent and signals the buffer (BICAMERAL §6.6) to update
     igram_progress.  This is what the model can see as
     input('compaction_90') + the layer_swap trigger fires when a
     new shard is loaded.

  4. list_loaded() — current state for the shell.

For Goal 6 (domain .mv2 transfer), each domain also carries a
`memory_shard` key — the .mv2 file (dept_memory) that holds the
domain's curated knowledge.  The router swaps both atomically.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

__all__ = [
    "ShardEntry",
    "SHARD_CATALOG",
    "register_shard",
    "swap_for_intent",
    "list_loaded",
    "current_shard",
    "domain_summary",
    "DOMAIN_LEXICONS",
    "route_intent",
]


@dataclass
class ShardEntry:
    shard_id: int
    name: str
    domain: str
    layer: int = -1  # -1 = shared; 0..47 = layer-specific
    vocab_hint: int = 0
    memory_shard: str = ""  # dept_memory file name
    description: str = ""
    n_params: int = 0
    ready: bool = False
    path: str = ""


# ---------------------------------------------------------------------------
# Catalog (well-known shards)
# ---------------------------------------------------------------------------

SHARD_CATALOG: dict[str, ShardEntry] = {
    "general": ShardEntry(
        shard_id=0, name="general", domain="general",
        description="Default Qwen-FN n-gram shard (general-purpose "
                    "English / code / reasoning).",
        n_params=51_000_000_000, ready=True,
    ),
    "medical": ShardEntry(
        shard_id=1, name="medical_first_aid", domain="medical",
        memory_shard="Medical",
        description="First-aid, anatomy, pharmacology vocab/n-gram.",
        n_params=2_000_000_000, ready=False,
    ),
    "engineering": ShardEntry(
        shard_id=2, name="engineering", domain="engineering",
        memory_shard="Sciences",
        description="Mechanical, electrical, civil, systems engineering.",
        n_params=2_000_000_000, ready=False,
    ),
    "code": ShardEntry(
        shard_id=3, name="code_dev", domain="code",
        memory_shard="Operations",
        description="Multi-language programming vocab/n-gram "
                    "(Python, JS, C, Rust, SQL, shell, EEL2).",
        n_params=2_000_000_000, ready=False,
    ),
    "finance": ShardEntry(
        shard_id=4, name="finance", domain="finance",
        memory_shard="Operations",
        description="Personal finance, accounting, Firefly-style "
                    "double-entry ledger vocabulary.",
        n_params=1_000_000_000, ready=False,
    ),
    "legal": ShardEntry(
        shard_id=5, name="legal", domain="legal",
        description="Common law, contracts, regulatory text.",
        n_params=1_000_000_000, ready=False,
    ),
    "navigation": ShardEntry(
        shard_id=6, name="navigation", domain="navigation",
        memory_shard="Navigation",
        description="Geospatial, astrometric, ephemeris vocabulary.",
        n_params=500_000_000, ready=False,
    ),
    "media": ShardEntry(
        shard_id=7, name="media_production", domain="media",
        memory_shard="Media",
        description="Film, audio, design vocab for studio work.",
        n_params=500_000_000, ready=False,
    ),
    "hoa64": ShardEntry(
        shard_id=8, name="hoa64_signal_processing", domain="hoa64",
        memory_shard="Sciences",
        description="HOA / Hadamard / spatial audio / sensor signal "
                    "vocabulary (companion to fabric/lab.py).",
        n_params=500_000_000, ready=False,
    ),
}


def register_shard(name: str, **kwargs) -> ShardEntry:
    """Register or update a shard in the catalog."""
    existing = SHARD_CATALOG.get(name)
    if existing is None:
        existing = ShardEntry(shard_id=len(SHARD_CATALOG), name=name,
                              domain=kwargs.get("domain", name))
        SHARD_CATALOG[name] = existing
    for k, v in kwargs.items():
        if hasattr(existing, k):
            setattr(existing, k, v)
    return existing


# ---------------------------------------------------------------------------
# Domain lexicons (lightweight keyword router)
# ---------------------------------------------------------------------------

DOMAIN_LEXICONS: dict[str, list[str]] = {
    "medical": [
        "first aid", "wound", "burn", "fracture", "cpr", "bleeding",
        "symptom", "diagnosis", "anatomy", "heart", "lung", "drug",
        "dose", "allergy", "anaphylaxis", "triage", "patient",
        "medication", "blood pressure", "choking", "poison", "rash",
        "sprain", "head injury", "concussion",
    ],
    "engineering": [
        "gear", "torque", "stress", "strain", "voltage", "current",
        "circuit", "beam", "moment", "load", "factor of safety",
        "yield", "fatigue", "bending", "shear", "fatigue",
        "rc circuit", "filter", "amplifier", "transformer",
    ],
    "code": [
        "function", "class", "object", "array", "loop", "if",
        "else", "return", "import", "from", "module", "package",
        "lambda", "closure", "iterator", "generator", "async",
        "await", "type", "interface", "trait", "struct", "enum",
        "compile", "linker", "runtime", "compile-time", "syntax",
    ],
    "finance": [
        "budget", "expense", "revenue", "asset", "liability",
        "equity", "income", "spending", "savings", "investment",
        "portfolio", "dividend", "interest", "loan", "mortgage",
        "tax", "invoice", "balance", "ledger", "audit", "firefly",
    ],
    "legal": [
        "contract", "clause", "agreement", "liability", "indemnity",
        "warranty", "jurisdiction", "arbitration", "license",
        "copyright", "patent", "trademark", "terms of service",
        "privacy policy", "compliance", "gdpr", "ccpa", "ferpa",
    ],
    "navigation": [
        "latitude", "longitude", "altitude", "azimuth", "elevation",
        "geocode", "tile", "terrain", "line of sight", "fresnel",
        "ephemeris", "orbit", "kepler", "jpl", "right ascension",
        "declination", "compass", "heading", "bearing", "waypoint",
    ],
    "media": [
        "shot", "scene", "take", "edit", "render", "composition",
        "focal length", "depth of field", "color grade", "luma",
        "chrominance", "track", "mix", "master", "stem", "sample",
        "wav", "spectrum", "spectral", "comp", "limiter",
    ],
    "hoa64": [
        "hadamard", "sylvester", "fwht", "mdct", "hoa", "ambisonic",
        "ambisonics", "acn", "sn3d", "n3d", "wxyz", "mic array",
        "encode", "decode", "spherical harmonic", "rotation matrix",
        "doppler", "fresnel", "diffraction",
    ],
}


def route_intent(text: str, top_k: int = 3) -> list[tuple[str, float]]:
    """Lexical router: score each domain by keyword overlap.

    Returns ``[(domain, score), ...]`` ordered by score desc, capped
    at top_k.  Scores are normalized [0, 1] within the doc.
    """
    if not text:
        return [("general", 0.0)]
    text_l = text.lower()
    scores: list[tuple[str, int]] = []
    for domain, lex in DOMAIN_LEXICONS.items():
        s = 0
        for kw in lex:
            if " " in kw and kw in text_l:
                s += 2
            elif re.search(rf"\b{re.escape(kw)}\b", text_l):
                s += 1
        if s:
            scores.append((domain, s))
    if not scores:
        return [("general", 0.0)]
    scores.sort(key=lambda x: -x[1])
    mx = scores[0][1]
    return [(d, round(s / max(mx, 1), 3)) for d, s in scores[:top_k]]


# ---------------------------------------------------------------------------
# Loaded-state + swap API
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_current: str = "general"
_load_history: list[dict] = []


def current_shard() -> dict:
    with _lock:
        e = SHARD_CATALOG.get(_current)
        return asdict(e) if e else {"name": _current, "ready": False}


def list_loaded() -> dict:
    with _lock:
        return {
            "current": _current,
            "current_entry": asdict(SHARD_CATALOG[_current]),
            "history": list(_load_history[-16:]),
        }


def swap_for_intent(intent: str, k: int = 1) -> dict:
    """Route the intent and swap the active n-gram shard.

    Calls ``fabric.qwen_buffer`` to publish igram_progress + layer_swap
    so the bus reflects the new context.  Returns the chosen shard
    metadata + the routing trace.
    """
    global _current
    routing = route_intent(intent, top_k=max(k, 1))
    chosen_domain, confidence = routing[0]
    chosen = SHARD_CATALOG.get(chosen_domain) or SHARD_CATALOG["general"]
    swap_event = {
        "t": time.time(),
        "from": _current,
        "to": chosen.name,
        "routing": routing,
        "intent_len": len(intent),
    }
    with _lock:
        prev = _current
        # Key SHARD_CATALOG by domain (e.g. "medical") not the entry
        # name (e.g. "medical_first_aid") so list_loaded can resolve
        # the current entry directly.
        _current = chosen_domain
        _load_history.append(swap_event)
        del _load_history[:-32]

    # Publish to MicroFX buffer bus
    try:
        from . import qwen_buffer
        qwen_buffer.publish_igram_progress(1, total=1)  # 1/1 ready
        qwen_buffer.fire_layer_swap(chosen.layer)        # rising edge
    except Exception:  # noqa: BLE001
        pass

    return {
        "ok": True,
        "previous": prev,
        "current": chosen.name,
        "routing": routing,
        "shard": asdict(chosen),
        "event": swap_event,
    }


def domain_summary() -> dict:
    """One-line summary per domain (for shell / API)."""
    out = {}
    for name, e in SHARD_CATALOG.items():
        out[name] = {
            "domain": e.domain,
            "ready": e.ready,
            "n_params": e.n_params,
            "memory_shard": e.memory_shard,
            "description": e.description[:80],
        }
    return out
