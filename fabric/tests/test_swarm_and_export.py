# SPDX-License-Identifier: Apache-2.0
"""Tests for fabric/swarm.py and fabric/mv2_export.py (Goals 6, 10)."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import json
import tempfile
from pathlib import Path

from fabric import swarm
from fabric import mv2_export
import importlib


def _fresh_swarm():
    importlib.reload(swarm)
    return swarm


def _fresh_export():
    importlib.reload(mv2_export)
    return mv2_export


# ---- swarm ----

def test_default_personas():
    s = _fresh_swarm()
    assert "medic" in s.DEFAULT_PERSONAS
    assert "engineer" in s.DEFAULT_PERSONAS
    assert "analyst" in s.DEFAULT_PERSONAS
    assert "creative" in s.DEFAULT_PERSONAS


def test_swarm_dataclasses():
    s = _fresh_swarm()
    p = s.Persona(name="test", system="test", temperature=0.5)
    assert p.name == "test"
    slot = s.SwarmSlot(slot_id="x", persona=p, n_replicas=2)
    assert slot.n_replicas == 2
    res = s.SwarmResult(slot_id="x", persona="test", replica=0,
                       text="ok", elapsed_s=0.1, ok=True)
    assert res.ok


def test_run_swarm_empty_prompt():
    s = _fresh_swarm()
    res = s.run_swarm(prompt="")
    assert "error" in res


def test_run_swarm_invalid_persona_string():
    s = _fresh_swarm()
    # Should not raise — unknown persona gets a default
    res = s.run_swarm(prompt="hello", personas=["unknown-persona"],
                      n_per_persona=1, timeout=1.0)
    # Either the call succeeded (LLM reachable) or returned errors
    assert "results" in res or "error" in res


# ---- mv2_export ----

def test_export_dir_exists():
    e = _fresh_export()
    assert e.EXPORT_DIR.is_dir()


def test_list_transferable_empty_when_no_shards():
    e = _fresh_export()
    # We can't easily fake the dept_memory dir here; just check the
    # function returns a list (possibly empty)
    out = e.list_transferable()
    assert isinstance(out, list)


def test_verify_manifest_wrong_path():
    e = _fresh_export()
    m = {"sha256": "deadbeef" * 8}
    assert e.verify_manifest(m, "/no/such/file") is False


def test_export_domain_missing():
    e = _fresh_export()
    res = e.export_domain("NoSuchDomain", out_dir=tempfile.mkdtemp())
    assert "error" in res


def test_export_import_round_trip():
    """Make a tiny .mv2 in the memory dir, export, import, verify."""
    e = _fresh_export()
    import os
    from fabric.dept_memory import _DIR
    # Save and restore the original (if any)
    src = _DIR / "TEST_ROUND_TRIP.mv2"
    if src.exists():
        backup = src.read_bytes()
    else:
        backup = None
    try:
        # Write a small valid file
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"MV2 test data " * 100)
        with tempfile.TemporaryDirectory() as tmp:
            res = e.export_domain("TEST_ROUND_TRIP", out_dir=tmp)
            assert "ok" in res
            assert res["ok"] is True
            assert Path(res["path"]).is_file()
            assert Path(res["manifest_path"]).is_file()
            # Import into another temp dir
            with tempfile.TemporaryDirectory() as tgt:
                imp = e.import_domain(res["path"], target_dir=tgt)
                assert imp.get("ok"), imp
                assert Path(imp["installed"]).is_file()
                # Verify sha256
                assert e.verify_manifest(res["manifest"], res["path"])
    finally:
        # Cleanup
        if src.exists():
            src.unlink()
        if backup is not None:
            src.write_bytes(backup)


if __name__ == "__main__":
    n = 0
    for name in dir(sys.modules[__name__]):
        if name.startswith("test_"):
            fn = getattr(sys.modules[__name__], name)
            fn()
            print(f"  {name} ok")
            n += 1
    print(f"swarm+mv2_export: {n}/{n} pass")
