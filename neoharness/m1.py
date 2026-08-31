"""sys.path bridge into the M1Multitronic tree.

Per M1Multitronic/AGENTS.md: "Don't vendor hoa64 or sibling trees; import
them." This module puts the M1 n-gram / spectral code on sys.path so
NeoHarness can import:

- ``afi.fwht``, ``afi.spectral``, ``afi.mdct``     (numpy-only host package)
- ``writable_ngram``                                (python/afi/bicameral)
- ``qwen27b_ple``                                   (python/afi/bicameral)

Override the location with the ``M1_ROOT`` env var.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_M1_ROOT = Path.home() / "M1Multitronic"

# Subdirectories of M1_ROOT that carry importable modules, in priority order.
_SUBDIRS = ("python", "python/afi/bicameral", "scripts")

_added: list[str] = []


def m1_root() -> Path:
    return Path(os.environ.get("M1_ROOT", str(_DEFAULT_M1_ROOT))).expanduser()


def ensure_m1_path() -> Path:
    """Insert the M1 import dirs on sys.path (idempotent).

    Returns the M1 root actually used. Raises FileNotFoundError with a clear
    message if the root does not exist.
    """
    root = m1_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"M1Multitronic root not found at {root} "
            "(set M1_ROOT to its location)"
        )
    for sub in _SUBDIRS:
        p = str(root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
            _added.append(p)
    return root


def import_m1(name: str):
    """Import a module from the M1 tree with a helpful error on failure."""
    ensure_m1_path()
    try:
        return __import__(name)
    except ImportError as exc:  # pragma: no cover - depends on host tree
        raise ImportError(
            f"failed to import {name!r} from M1Multitronic at {m1_root()}: {exc}"
        ) from exc
