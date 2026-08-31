# SPDX-License-Identifier: Apache-2.0
"""Tests for fabric/domain_router.py (BICAMERAL Goal 5)."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from fabric import domain_router
import importlib


def _fresh():
    importlib.reload(domain_router)
    return domain_router


def test_route_medical():
    d = _fresh()
    routes = d.route_intent("I have a small burn on my hand, what first "
                          "aid should I use?")
    assert routes[0][0] == "medical", routes


def test_route_engineering():
    d = _fresh()
    routes = d.route_intent("calculate the torque on this shaft and the "
                          "factor of safety")
    assert routes[0][0] == "engineering", routes


def test_route_code():
    d = _fresh()
    routes = d.route_intent("how do I write a python generator function "
                          "with async await")
    assert routes[0][0] == "code", routes


def test_route_finance():
    d = _fresh()
    routes = d.route_intent("what's the balance on my firefly ledger")
    assert routes[0][0] == "finance", routes


def test_route_falls_back_to_general():
    d = _fresh()
    routes = d.route_intent("hello there")
    assert routes[0][0] == "general"


def test_swap_for_intent():
    d = _fresh()
    res = d.swap_for_intent("patient is showing anaphylaxis symptoms")
    assert res["ok"]
    assert res["routing"][0][0] == "medical"
    assert res["current"] == "medical_first_aid"


def test_swap_keeps_history():
    d = _fresh()
    d.swap_for_intent("I have a fracture")
    d.swap_for_intent("the gear ratio is 3:1")
    state = d.list_loaded()
    assert len(state["history"]) >= 2
    assert state["current"] == "engineering"


def test_catalog_defaults_present():
    d = _fresh()
    cats = d.domain_summary()
    assert "general" in cats
    assert "medical" in cats
    assert "code" in cats
    # Each entry has the expected fields
    for name, e in cats.items():
        assert "domain" in e
        assert "ready" in e


def test_register_shard():
    d = _fresh()
    d.register_shard("custom", domain="custom",
                     description="user domain", ready=True)
    assert "custom" in d.SHARD_CATALOG


if __name__ == "__main__":
    n = 0
    for name in dir(sys.modules[__name__]):
        if name.startswith("test_"):
            fn = getattr(sys.modules[__name__], name)
            fn()
            print(f"  {name} ok")
            n += 1
    print(f"domain_router: {n}/{n} pass")
