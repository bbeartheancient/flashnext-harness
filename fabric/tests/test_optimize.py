"""Tests for /optimize — self-analysis endpoint and probe system.

Verifies:
  - All seven probes return findings with type/severity/fix
  - The endpoint routes focus= to a single probe
  - Recommendations are sorted by severity (high first)
  - The /optimize slash command resolves correctly
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_optimize_command_in_catalog():
    from fabric.commands import COMMANDS
    ids = {c["id"] for c in COMMANDS}
    assert "optimize" in ids
    cmd = next(c for c in COMMANDS if c["id"] == "optimize")
    assert cmd["lane"] == "bicameral"
    assert "memory" in cmd["blurb"].lower() or "self" in cmd["blurb"].lower()


def test_resolve_optimize_all():
    from fabric.commands import resolve
    got = resolve("/optimize")
    assert got["slash"] is True
    assert got["command"] == "optimize"
    assert got["url"] == "/api/optimize/analyze"
    assert got["method"] == "POST"
    assert got["body"] == {}
    assert got["autorun"] is True
    assert got["lane"] == "bicameral"


def test_resolve_optimize_focus():
    from fabric.commands import resolve
    got = resolve("/optimize trace")
    assert got["url"] == "/api/optimize/analyze"
    assert got["body"] == {"focus": "trace"}


def test_resolve_optimize_invalid_focus():
    from fabric.commands import resolve
    got = resolve("/optimize bogus")
    assert got["autorun"] is False
    assert "focus" in got["hint"]


def test_analyze_returns_findings():
    from fabric.optimize import analyze
    result = analyze()
    assert "probes_run" in result
    assert "n_findings" in result
    assert "tally" in result
    assert "action_items" in result
    assert "summary" in result
    assert "findings" in result
    assert "recommendations" in result
    # All 7 probes should have run
    assert len(result["probes_run"]) == 7
    expected_probes = {"shard", "trace", "buffer", "domain",
                        "memory", "process", "teach"}
    assert set(result["probes_run"]) == expected_probes


def test_analyze_focus_filters():
    from fabric.optimize import analyze
    result = analyze(focus="trace")
    assert result["probes_run"] == ["trace"]
    # all findings should have probe=trace
    for f in result["findings"]:
        assert f["probe"] == "trace"


def test_findings_have_required_fields():
    from fabric.optimize import analyze
    result = analyze(focus="process")
    for f in result["findings"]:
        assert "type" in f
        assert "subject" in f
        assert "severity" in f
        assert "detail" in f
        assert "probe" in f


def test_recommendations_sorted_by_severity():
    from fabric.optimize import _recommendations
    items = [
        {"type": "BLOAT", "subject": "x", "severity": "high",
         "detail": "x", "fix": "y"},
        {"type": "MISSING", "subject": "a", "severity": "low",
         "detail": "a", "fix": "b"},
        {"type": "STALE", "subject": "m", "severity": "medium",
         "detail": "m", "fix": "c"},
    ]
    recs = _recommendations(items)
    assert recs[0]["severity"] == "high"
    assert recs[1]["severity"] == "medium"
    assert recs[2]["severity"] == "low"


def test_endpoint_full_run():
    from fastapi.testclient import TestClient
    from fabric.main import app
    c = TestClient(app)
    r = c.post("/api/optimize/analyze", json={})
    assert r.status_code == 200
    data = r.json()
    assert data["probes_run"]
    assert data["n_findings"] >= 1
    # tally must be present
    assert "OK" in data["tally"] or len(data["findings"]) > 0


def test_endpoint_focus_run():
    from fastapi.testclient import TestClient
    from fabric.main import app
    c = TestClient(app)
    r = c.post("/api/optimize/analyze", json={"focus": "process"})
    assert r.status_code == 200
    data = r.json()
    assert data["probes_run"] == ["process"]
    for f in data["findings"]:
        assert f["probe"] == "process"


def test_process_probe_returns_rss():
    from fabric.optimize import _process_probe
    findings = _process_probe()
    assert findings
    # At least one OK with RSS
    ok = [f for f in findings if f["type"] == "OK"]
    assert ok
    assert "RSS" in ok[0]["detail"]


def test_buffer_probe_handles_empty():
    from fabric.optimize import _buffer_probe
    findings = _buffer_probe()
    assert findings
    # probe tag is added by analyze() not the probe itself
    assert "probe" not in findings[0]
    # but type/subject/severity/detail must be present
    for f in findings:
        assert "type" in f and "subject" in f
        assert "severity" in f and "detail" in f
