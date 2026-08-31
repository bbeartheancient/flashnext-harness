# SPDX-License-Identifier: Apache-2.0
"""Tests for fabric/teach_paths.py (BICAMERAL Goal 4)."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from fabric import teach_paths
import importlib


def _fresh():
    importlib.reload(teach_paths)
    return teach_paths


def test_examples_seeded():
    tp = _fresh()
    assert len(tp.TEACHING_EXAMPLES) >= 4
    assert all(hasattr(e, "assembly_mermaid") for e in tp.TEACHING_EXAMPLES)
    assert all(e.assembly_mermaid.startswith("flowchart")
               for e in tp.TEACHING_EXAMPLES)


def test_templates_seeded():
    tp = _fresh()
    assert len(tp.PATH_TEMPLATES) >= 3
    assert "smooth_then_chart" in tp.PATH_TEMPLATES
    assert "gated_clock_chart" in tp.PATH_TEMPLATES


def test_build_prompt_has_examples():
    tp = _fresh()
    prompt = tp.build_prompt(domain="all")  # include all examples
    assert "flowchart" in prompt
    assert "smooth" in prompt.lower()
    assert "clock" in prompt.lower()


def test_prompt_for_intent_routes_medical():
    tp = _fresh()
    # The route picks 'medical' for wound/bleeding keywords; if no medical
    # teaching example exists yet, the prompt falls back to general.
    prompt = tp.prompt_for_intent("I have a wound that is bleeding heavily")
    assert isinstance(prompt, str) and len(prompt) > 100
    # Routing must pick medical (keyword match in domain_router)
    from fabric import domain_router
    routes = domain_router.route_intent("I have a wound that is bleeding heavily")
    assert routes[0][0] == "medical", routes


def test_prompt_for_intent_routes_code():
    tp = _fresh()
    # code intent routes to 'code' domain
    prompt = tp.prompt_for_intent("write a python function that returns a lambda")
    assert isinstance(prompt, str) and len(prompt) > 100


def test_prompt_for_intent_unknown_falls_back():
    tp = _fresh()
    prompt = tp.prompt_for_intent("hello world")
    assert "flowchart" in prompt


def test_register_example():
    tp = _fresh()
    before = len(tp.TEACHING_EXAMPLES)
    ex = tp.TeachingExample(
        id="test-ex", domain="test", difficulty="test",
        description="test", assembly_mermaid="flowchart TD\n A-->B",
        inputs=["x"], outputs=["y"], blocks_used=["const"],
        tags=["test"])
    tp.register_example(ex)
    assert len(tp.TEACHING_EXAMPLES) == before + 1
    assert any(e.id == "test-ex" for e in tp.TEACHING_EXAMPLES)


if __name__ == "__main__":
    n = 0
    for name in dir(sys.modules[__name__]):
        if name.startswith("test_"):
            fn = getattr(sys.modules[__name__], name)
            fn()
            print(f"  {name} ok")
            n += 1
    print(f"teach_paths: {n}/{n} pass")
