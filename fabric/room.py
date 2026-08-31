"""Room-IR calculator surface — acoustic rays vs a shoebox, HOA-encoded.

Wraps python/afi/room_ir.py for fabric tools and the LCARS `room` pane.
"""

from __future__ import annotations

import os
import sys
import threading

_lock = threading.Lock()
_last = {"title": None, "figure": None, "result": None}


def _ensure_afi():
    repo_python = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "python")
    if os.path.isdir(repo_python) and repo_python not in sys.path:
        sys.path.insert(0, repo_python)


def last_room() -> dict:
    with _lock:
        return dict(_last)


def room_impulse(src: list[float], lst: list[float],
                 box: list[float] = None, n_rays: int = 1024,
                 max_bounce: int = 3, absorption: float = 0.2,
                 fs: float = 48000.0, duration_s: float = 0.2,
                 order: int = 3, furniture: list = None) -> dict:
    """Trace a shoebox IR and encode arrivals as SN3D HOA."""
    _ensure_afi()
    from afi.room_ir import room_impulse as _ri

    if box is None:
        box = [8.0, 3.0, 6.0]
    if not src or not lst or len(src) < 3 or len(lst) < 3:
        return {"error": "src and lst need [x, y, z] metres inside the box"}
    out = _ri(src, lst, box=box, n_rays=n_rays, max_bounce=max_bounce,
              absorption=absorption, fs=fs, duration_s=duration_s, order=order,
              furniture=furniture)
    with _lock:
        _last["title"] = out.get("figure", {}).get("caption")
        _last["figure"] = out.get("figure")
        _last["result"] = {k: v for k, v in out.items() if k != "arrivals"}
        _last["result"]["n_arrivals"] = out.get("n_arrivals")
    return out
