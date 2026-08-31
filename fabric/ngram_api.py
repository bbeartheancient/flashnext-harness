"""Writable n-gram PLE API — /api/ngram/* endpoints (Step 3 of
docs/27b-writable-ngram-plan.md).

Thin HTTP bridge over python/afi/bicameral/writable_ngram.py so the
EEL2 runtime in jsfx.js (browser iframes) can read/write the MV2
n-gram shards:

  ngram_lookup(mem_start, token, layer[, pos])  -> GET  /api/ngram/lookup
  ngram_store(token, layer, mem_start[, pos])   -> POST /api/ngram/store
  session flush at turn end                     -> POST /api/ngram/flush

The EEL2 side is frame-cached (lookup returns 0 on the first frame
and fills mem[] once the fetch lands) — same live-signal philosophy
as the sensor ins: you always read against the latest frame.

ngram_neighbors is intentionally NOT exposed: the vec index stays
disabled (memvid "unknown embedder" poisons the lex index in-process —
see writable_ngram.py landmines) until an embedder is chosen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_PYTHON = Path(__file__).resolve().parents[1] / "python"


def _wn():
    """Import writable_ngram with the bicameral dir on sys.path."""
    bic = _REPO_PYTHON / "afi" / "bicameral"
    if bic.is_dir() and str(bic) not in sys.path:
        sys.path.insert(0, str(bic))
    import writable_ngram as wn  # noqa: PLC0415
    return wn


def lookup(layer: int, token: int, pos: int = 0) -> dict:
    """One n-gram row: {found, vector?} with vector as 160 floats."""
    wn = _wn()
    store = wn.open_store(int(layer))
    row = store._find_one(int(token), int(pos))
    if row is None:
        return {"found": False}
    return {"found": True,
            "vector": [float(x) for x in row.vector.astype(np.float32)],
            "write_count": row.write_count}


def lookup_batch(layer: int, tokens: list[int], pos: int = 0) -> dict:
    """Batch lookup: {vectors: {token: [160 floats]}} (hits only)."""
    wn = _wn()
    store = wn.open_store(int(layer))
    vecs = {}
    for t in dict.fromkeys(int(t) for t in tokens[:4096]):
        row = store._find_one(t, int(pos))
        if row is not None:
            vecs[str(t)] = [float(x) for x in row.vector.astype(np.float32)]
    return {"found": len(vecs), "vectors": vecs}


def store_rows(layer: int, rows: list[dict], pos: int = 0) -> dict:
    """Write rows [{token, vector(160)}]; WAL append, flush separately."""
    wn = _wn()
    store = wn.open_store(int(layer))
    triples = []
    for r in rows[:2048]:
        if r.get("token") is None or r.get("vector") is None:
            return {"error": "each row needs token + vector"}
        vec = np.asarray(r["vector"], dtype=np.float16)
        if vec.shape != (wn._NGRAM_DIM,):
            return {"error": f"token {r.get('token')}: vector must be "
                             f"{wn._NGRAM_DIM} floats"}
        triples.append((int(r["token"]), int(pos), vec))
    meta = {}
    if rows and rows[0].get("graph_id"):
        meta["graph_id"] = str(rows[0]["graph_id"])
    store.store(triples, metadata=meta)
    return {"stored": len(triples), "layer": int(layer)}


def flush() -> dict:
    wn = _wn()
    wn.session_commit()
    return {"committed": True}


def stats() -> dict:
    wn = _wn()
    with wn._lock:
        return {"open_layers": sorted(wn._handles.keys()),
                "shards": {L: st.stats()
                           for L, st in wn._handles.items()}}
