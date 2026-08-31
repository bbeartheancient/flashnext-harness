# SPDX-License-Identifier: Apache-2.0
"""Tests for fabric.qwen_buffer (BICAMERAL_FRAMEWORK.md §6.6)."""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def buf():
    from fabric import qwen_buffer
    importlib.reload(qwen_buffer)
    return qwen_buffer


def test_publish_compaction_clamps(buf):
    buf.publish_compaction(0.42)
    snap = buf.snapshot()
    assert snap["bus"]["compaction_90"] == 0.42
    # Negative / NaN dropped
    buf.publish_compaction(-0.5)
    assert buf.snapshot()["bus"]["compaction_90"] == 0.42
    buf.publish_compaction(float("nan"))
    assert buf.snapshot()["bus"]["compaction_90"] == 0.42


def test_publish_qsa_keep(buf):
    buf.publish_qsa_keep(512, budget=2048)
    assert buf.snapshot()["bus"]["qsa_keep_frac"] == 0.25
    buf.publish_qsa_keep(1024, budget=2048)
    assert buf.snapshot()["bus"]["qsa_keep_frac"] == 0.5
    # Saturate at 1.0
    buf.publish_qsa_keep(99999, budget=2048)
    assert buf.snapshot()["bus"]["qsa_keep_frac"] == 1.0


def test_publish_igram_progress(buf):
    buf.publish_igram_progress(7, total=10)
    assert buf.snapshot()["bus"]["igram_block_load"] == 0.7


def test_fire_layer_swap_appends_event(buf):
    buf.fire_layer_swap(2)
    snap = buf.snapshot()
    assert snap["bus"]["layer_idx"] == 2
    assert snap["events"][-1]["name"] == "layer_swap"
    assert snap["events"][-1]["layer"] == 2


def test_fire_mv2_tap(buf):
    buf.fire_mv2_tap("test-tag")
    snap = buf.snapshot()
    assert snap["events"][-1]["name"] == "mv2_tap"
    assert snap["events"][-1]["note"] == "test-tag"


def test_event_ring_caps_at_max(buf):
    # BUFFER_MODULES shouldn't fire events; only fire_* should.
    for i in range(300):
        buf.fire_layer_swap(i)
    snap = buf.snapshot()
    # snapshot only returns last 16
    assert len(snap["events"]) == 16
    # bus state still tracks the most recent
    assert snap["bus"]["layer_idx"] == 299


def test_buffer_modules_catalog_present(buf):
    assert "compaction_90" in buf.BUFFER_MODULES
    assert "layer_swap" in buf.BUFFER_MODULES
    assert "mv2_tap" in buf.BUFFER_MODULES
    # All buffer modules are 'source' (host-fed)
    for name, mod in buf.BUFFER_MODULES.items():
        assert mod["category"] == "source", name
        assert mod.get("host") is True, name


# ---- Module-level fallback runners (so the file works without pytest) ----

def _run_all():
    """Run every test_ function with a fresh module instance."""
    from fabric import qwen_buffer
    importlib.reload(qwen_buffer)
    buf = qwen_buffer
    fns = [(n, getattr(sys.modules[__name__], n)) for n in dir(sys.modules[__name__])
           if n.startswith("test_")]
    failed = []
    for n, fn in fns:
        try:
            fn(buf)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{n}: {e!r}")
    return failed


import sys

if __name__ == "__main__":
    fails = _run_all()
    if fails:
        for f in fails:
            print("FAIL", f)
        sys.exit(1)
    print(f"qwen_buffer: {sum(1 for n in dir(sys.modules[__name__]) if n.startswith('test_'))}/{sum(1 for n in dir(sys.modules[__name__]) if n.startswith('test_'))} pass")
