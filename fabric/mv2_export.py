# SPDX-License-Identifier: Apache-2.0
"""Domain .mv2 shard export / import (BICAMERAL Goal 6).

Storing domain knowledge sets as .mv2 shards creates an easy way to
transfer knowledge between systems.  This module wraps the existing
dept_memory.py with:

  1. export_domain(domain, out_path)  — copy + sign a domain's
     .mv2 shard for transfer.
  2. import_domain(src_path, domain=)  — load a transferred shard
     into the local system.  Honors a simple manifest.json that
     includes the originating system, domain, and a sha256 of the
     .mv2 file (the recipient verifies before installing).
  3. list_transferable() — domains that are export-ready (ready=True
     in fabric.domain_router.SHARD_CATALOG and have a non-empty
     .mv2 file on disk).

The manifest format (BICAMERAL Goal 6 explicit goal: "easy to transfer
between systems"):

    {
        "domain": "medical",
        "shard": "Medical.mv2",
        "origin": "fabric-host-001",
        "exported_at": "2026-08-26T14:00:00Z",
        "n_docs": 124,
        "sha256": "<hex>",
        "size_bytes": 12345,
        "qwen_flash_shard": "medical_first_aid"
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .departments import DEPARTMENTS
from .domain_router import SHARD_CATALOG

logger = logging.getLogger(__name__)

__all__ = [
    "TransferManifest",
    "export_domain",
    "import_domain",
    "list_transferable",
    "verify_manifest",
    "EXPORT_DIR",
]

EXPORT_DIR = Path(os.environ.get(
    "FABRIC_EXPORT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "data", "exports")))
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()


@dataclass
class TransferManifest:
    domain: str
    shard: str           # .mv2 file name
    origin: str
    exported_at: str
    n_docs: int
    sha256: str
    size_bytes: int
    qwen_flash_shard: str = ""
    note: str = ""


def _memory_dir() -> Path:
    from .dept_memory import _DIR
    return _DIR


def _origin_name() -> str:
    return os.environ.get("FABRIC_HOST_ID", "fabric-unknown")


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _n_docs(p: Path) -> int:
    """Best-effort doc count from an .mv2 file.  memvid SDK has no
    cheap count API; we scan the WAL header for a marker."""
    try:
        with p.open("rb") as f:
            data = f.read(64 * 1024)
        # look for "uri: mv2://" patterns
        return data.count(b"mv2://")
    except OSError:
        return 0


def list_transferable() -> list[dict]:
    """List domains with a non-empty .mv2 ready to export."""
    out: list[dict] = []
    mem_dir = _memory_dir()
    # DEPARTMENTS is a list of dicts (each with 'id' / 'name')
    items = DEPARTMENTS if isinstance(DEPARTMENTS, list) else \
        list(DEPARTMENTS.items())
    for entry in items:
        if isinstance(entry, dict):
            dept_name = entry.get("id") or entry.get("name")
        else:
            dept_name = entry  # tuple form
        if not dept_name:
            continue
        mv2 = mem_dir / f"{dept_name}.mv2"
        if not mv2.is_file():
            continue
        size = mv2.stat().st_size
        if size == 0:
            continue
        # find a matching domain-router entry by memory_shard field
        domain_match = None
        for sname, sentry in SHARD_CATALOG.items():
            if sentry.memory_shard == dept_name:
                domain_match = sname
                break
        out.append({
            "domain": dept_name,
            "dept_memory_file": str(mv2.relative_to(mem_dir.parent)),
            "size_bytes": size,
            "qwen_flash_shard": domain_match or "",
            "ready": size > 0,
        })
    return out


def export_domain(domain: str, out_dir: Path | str | None = None,
                  origin: str | None = None,
                  note: str = "") -> dict:
    """Export a domain's .mv2 to a portable package.

    Writes ``out_dir/<domain>-<timestamp>.mv2`` and a sibling
    ``manifest.json``.  Returns the manifest as a dict.
    """
    with _lock:
        mem_dir = _memory_dir()
        mv2 = mem_dir / f"{domain}.mv2"
        if not mv2.is_file() or mv2.stat().st_size == 0:
            return {"error": f"domain '{domain}' has no exportable .mv2 "
                              f"({mv2} missing or empty)"}
        out = Path(out_dir) if out_dir else EXPORT_DIR
        out.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        target = out / f"{domain}-{ts}.mv2"
        shutil.copy2(mv2, target)
        sha = _hash_file(target)
        manifest = TransferManifest(
            domain=domain,
            shard=target.name,
            origin=origin or _origin_name(),
            exported_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            n_docs=_n_docs(target),
            sha256=sha,
            size_bytes=target.stat().st_size,
            qwen_flash_shard=next(
                (n for n, e in SHARD_CATALOG.items()
                 if e.memory_shard == domain), ""),
            note=note,
        )
        mpath = target.with_suffix(".manifest.json")
        mpath.write_text(json.dumps(asdict(manifest), indent=2))
        return {
            "ok": True,
            "manifest": asdict(manifest),
            "path": str(target),
            "manifest_path": str(mpath),
        }


def verify_manifest(manifest: dict, shard_path: Path | str) -> bool:
    """Check the .mv2 against the manifest's sha256."""
    sp = Path(shard_path)
    if not sp.is_file():
        return False
    return _hash_file(sp) == manifest.get("sha256")


def import_domain(src_path: str | Path,
                  manifest: dict | None = None,
                  target_dir: Path | str | None = None) -> dict:
    """Install a transferred .mv2 into the local memory directory.

    The manifest is required for the sha256 check; if not given, we
    look for ``<src_path>.manifest.json`` next to the file.
    """
    src = Path(src_path)
    if not src.is_file():
        return {"error": f"source not found: {src}"}
    if manifest is None:
        mfile = src.with_suffix(".manifest.json")
        if mfile.is_file():
            manifest = json.loads(mfile.read_text())
        else:
            return {"error": "no manifest given and no sibling "
                              ".manifest.json found"}
    if not verify_manifest(manifest, src):
        return {"error": "sha256 mismatch — shard corrupted in transit"}
    tgt_dir = Path(target_dir) if target_dir else _memory_dir()
    tgt_dir.mkdir(parents=True, exist_ok=True)
    # The installed file uses the domain name (overwrites any existing)
    target_file = tgt_dir / f"{manifest['domain']}.mv2"
    shutil.copy2(src, target_file)
    return {
        "ok": True,
        "installed": str(target_file),
        "manifest": manifest,
        "n_docs": manifest.get("n_docs", 0),
        "size_bytes": target_file.stat().st_size,
    }
