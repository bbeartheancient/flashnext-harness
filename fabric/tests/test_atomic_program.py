# SPDX-License-Identifier: Apache-2.0
"""Tests for fabric/atomic_program.py (BICAMERAL Goals 2, 4)."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import json
from fabric import atomic_program
import importlib


def _fresh():
    importlib.reload(atomic_program)
    return atomic_program


def test_basic_program():
    ap = _fresh()
    p = ap.AtomicProgram(
        name="t",
        description="test",
        blocks=[
            ap.AtomicBlock(id="a", primitive="const", label="Const"),
            ap.AtomicBlock(id="b", primitive="smooth", label="Smooth"),
            ap.AtomicBlock(id="c", primitive="viz_series", label="Chart"),
        ],
        wires=[
            ap.Wire(src="a.cv", dst="b.in"),
            ap.Wire(src="b.cv", dst="c.in"),
        ],
    )
    assert p.compute_hash()
    assert p.validate() is None


def test_validate_missing_block():
    ap = _fresh()
    p = ap.AtomicProgram(
        name="bad", description="",
        blocks=[ap.AtomicBlock(id="a", primitive="const")],
        wires=[ap.Wire(src="missing.cv", dst="a.cv")],
    )
    err = p.validate()
    assert err and "not found" in err


def test_to_patch():
    ap = _fresh()
    p = ap.AtomicProgram(
        name="t",
        blocks=[ap.AtomicBlock(id="a", primitive="const", params={"value": 5.0})],
        wires=[],
    )
    patch = ap.compile_program(p, "microfx")
    assert "modules" in patch
    assert patch["modules"][0]["primitive"] == "const"
    assert patch["modules"][0]["params"][0]["value"] == 5.0


def test_to_mermaid():
    ap = _fresh()
    p = ap.AtomicProgram(
        name="m",
        blocks=[
            ap.AtomicBlock(id="a", primitive="const"),
            ap.AtomicBlock(id="b", primitive="viz_series"),
        ],
        wires=[ap.Wire(src="a.cv", dst="b.in")],
    )
    mermaid = ap.compile_program(p, "mermaid")
    assert "flowchart" in mermaid
    assert "a" in mermaid and "b" in mermaid
    assert "a" in mermaid  # both block ids present


def test_teacher_registry_seeded():
    ap = _fresh()
    examples = ap.get_teacher_examples(domain="signal_processing")
    assert len(examples) >= 1
    assert all("hash" in e for e in examples)
    assert all("mermaid" in e for e in examples)


def test_register_teacher_example():
    ap = _fresh()
    p = ap.AtomicProgram(
        name="user-x", description="",
        blocks=[ap.AtomicBlock(id="a", primitive="const")],
    )
    key = ap.register_teacher_example(p, difficulty="intermediate",
                                      domain="user-domain")
    assert key == p.compute_hash()
    found = ap.get_teacher_examples(domain="user-domain")
    assert any(f["hash"] == key for f in found)


def test_describe_program():
    ap = _fresh()
    p = ap.AtomicProgram(
        name="desc-test", description="a description",
        blocks=[ap.AtomicBlock(id="a", primitive="const")],
    )
    text = ap.describe_program(p)
    assert "desc-test" in text
    assert "a description" in text


if __name__ == "__main__":
    n = 0
    for name in dir(sys.modules[__name__]):
        if name.startswith("test_"):
            fn = getattr(sys.modules[__name__], name)
            fn()
            print(f"  {name} ok")
            n += 1
    print(f"atomic_program: {n}/{n} pass")
