"""Tests for the new slash commands and /atomize command.

Verifies the operator ABI: each new command resolves to the right
URL/method/body, and the /atomize endpoint compiles a DMA trace into
both AtomicProgram and teaching-example forms.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_commands_catalog_has_new():
    from fabric.commands import COMMANDS
    ids = {c["id"] for c in COMMANDS}
    expected = {
        "omp", "bicameral", "swarm", "trace", "improve", "atomize",
        "domain", "shard", "atomic", "teach", "roundtrip", "buffer",
    }
    missing = expected - ids
    assert not missing, f"missing commands: {missing}"


def test_resolve_omp():
    from fabric.commands import resolve
    got = resolve("/omp write a hello world python")
    assert got["slash"] is True
    assert got["command"] == "omp"
    assert got["url"] == "/api/omp/run"
    assert got["method"] == "POST"
    assert got["body"]["message"] == "write a hello world python"
    assert got["autorun"] is True
    assert got["lane"] == "omp"


def test_resolve_omp_no_args():
    from fabric.commands import resolve
    got = resolve("/omp")
    assert got["autorun"] is False
    assert "OMP" in got["hint"]


def test_resolve_bicameral():
    from fabric.commands import resolve
    got = resolve("/bicameral 90 0.3")
    assert got["url"] == "/api/bicameral/run"
    assert got["body"]["n_frames"] == 90
    assert got["body"]["alpha"] == 0.3
    assert got["lane"] == "bicameral"


def test_resolve_swarm():
    from fabric.commands import resolve
    got = resolve("/swarm design a circuit")
    assert got["url"] == "/api/swarm/run"
    assert got["body"]["prompt"] == "design a circuit"
    assert got["lane"] == "swarm"


def test_resolve_trace():
    from fabric.commands import resolve
    g1 = resolve("/trace start")
    assert g1["url"] == "/api/trace/start" and g1["method"] == "POST"
    g2 = resolve("/trace stop")
    assert g2["url"] == "/api/trace/stop" and g2["method"] == "POST"
    g3 = resolve("/trace snapshot")
    assert g3["url"] == "/api/trace/snapshot" and g3["method"] == "GET"
    g4 = resolve("/trace")
    assert g4["url"] == "/api/trace/snapshot"  # default = snapshot


def test_resolve_atomize():
    from fabric.commands import resolve
    # default
    g = resolve("/atomize")
    assert g["autorun"] is False
    # source + dest
    g2 = resolve("/atomize live atomic")
    assert g2["url"] == "/api/atomize/run"
    assert g2["body"] == {"source": "live", "dest": "atomic"}
    g3 = resolve("/atomize recent ngram")
    assert g3["body"] == {"source": "recent", "dest": "ngram"}
    g4 = resolve("/atomize snapshot both")
    assert g4["body"] == {"source": "snapshot", "dest": "both"}
    # invalid source
    g5 = resolve("/atomize bogus atomic")
    assert g5["autorun"] is False
    assert "source" in g5["hint"]


def test_resolve_domain():
    from fabric.commands import resolve
    g1 = resolve("/domain")
    assert g1["url"] == "/api/domain/state" and g1["method"] == "GET"
    g2 = resolve("/domain medical emergency")
    assert g2["url"] == "/api/domain/route"
    assert g2["body"]["intent"] == "medical emergency"


def test_resolve_shard():
    from fabric.commands import resolve
    g1 = resolve("/shard")
    assert g1["url"] == "/api/shard/list"
    g2 = resolve("/shard list")
    assert g2["url"] == "/api/shard/list"
    g3 = resolve("/shard export medical_first_aid")
    assert g3["url"] == "/api/shard/export"
    assert g3["body"]["domain"] == "medical_first_aid"
    g4 = resolve("/shard import /tmp/foo.mv2")
    assert g4["url"] == "/api/shard/import"
    assert g4["body"]["src_path"] == "/tmp/foo.mv2"


def test_resolve_atomic():
    from fabric.commands import resolve
    g1 = resolve("/atomic")
    assert g1["url"] == "/api/atomic/examples"
    g2 = resolve("/atomic gate clock chart")
    assert g2["url"] == "/api/atomic/from_description"
    assert g2["body"]["description"] == "gate clock chart"


def test_resolve_teach():
    from fabric.commands import resolve
    g1 = resolve("/teach")
    assert g1["url"] == "/api/teach/examples"
    g2 = resolve("/teach oscilloscope with phase")
    assert g2["url"] == "/api/teach/prompt"
    assert g2["body"]["intent"] == "oscilloscope with phase"


def test_resolve_roundtrip():
    from fabric.commands import resolve
    g1 = resolve("/roundtrip")
    assert g1["autorun"] is False
    g2 = resolve("/roundtrip my_app")
    assert g2["url"] == "/api/roundtrip/apply"
    assert g2["body"]["app_id"] == "my_app"


def test_resolve_buffer():
    from fabric.commands import resolve
    g1 = resolve("/buffer")
    assert g1["url"] == "/api/buffer/state" and g1["method"] == "GET"
    g2 = resolve("/buffer state")
    assert g2["url"] == "/api/buffer/state"
    g3 = resolve("/buffer trigger layer_swap 2")
    assert g3["url"] == "/api/buffer/trigger"
    assert g3["body"] == {"name": "layer_swap", "layer": 2}
    g4 = resolve("/buffer publish compaction_90 0.42")
    assert g4["url"] == "/api/buffer/publish"
    assert g4["body"] == {"name": "compaction_90", "value": 0.42}


def test_resolve_improve():
    from fabric.commands import resolve
    g = resolve("/improve a short story about a dog")
    assert g["url"] == "/api/improve/run"
    assert g["body"]["prompt"] == "a short story about a dog"
    assert g["body"]["max_iterations"] == 3
    assert g["body"]["compact_rate"] == 0.5


def test_atomize_no_frames():
    """Endpoint should 404 when there are no trace frames."""
    from fastapi.testclient import TestClient
    from fabric.main import app
    from fabric import dma_trace
    dma_trace.reset_trace()
    c = TestClient(app)
    r = c.post("/api/atomize/run", json={"source": "recent", "dest": "atomic"})
    assert r.status_code == 404


def test_atomize_atomic_destination():
    """A real DMA trace → compiled AtomicProgram."""
    from fastapi.testclient import TestClient
    from fabric.main import app
    from fabric import dma_trace
    dma_trace.reset_trace()
    dma_trace.start_trace()
    dma_trace.record(
        module_id="src1", module_type="sine_lfo",
        in_ports={"freq": 1.0}, out_ports={"sig": 0.1},
        latency_us=10.0, tag="chain_a:src1",
    )
    dma_trace.record(
        module_id="gate1", module_type="threshold",
        in_ports={"in": 0.1}, out_ports={"out": 0.0},
        latency_us=20.0, tag="chain_a:gate1",
    )
    dma_trace.stop_trace()
    c = TestClient(app)
    r = c.post("/api/atomize/run",
               json={"source": "recent", "dest": "atomic"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["frames"] == 2
    assert len(data["atomic_programs"]) == 1
    p = data["atomic_programs"][0]
    assert p["blocks"] == 2
    assert p["wires"] == 1
    assert p["compiled"]
    dma_trace.reset_trace()


def test_atomize_ngram_destination():
    """A real DMA trace → teaching examples."""
    from fastapi.testclient import TestClient
    from fabric.main import app
    from fabric import dma_trace
    dma_trace.reset_trace()
    dma_trace.start_trace()
    dma_trace.record(
        module_id="src1", module_type="sine_lfo",
        in_ports={"freq": 1.0}, out_ports={"sig": 0.1},
        latency_us=10.0, tag="chain_b:src1",
    )
    dma_trace.record(
        module_id="filter1", module_type="moving_avg",
        in_ports={"in": 0.1}, out_ports={"out": 0.0},
        latency_us=20.0, tag="chain_b:filter1",
    )
    dma_trace.stop_trace()
    c = TestClient(app)
    r = c.post("/api/atomize/run",
               json={"source": "recent", "dest": "ngram"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["frames"] == 2
    assert len(data["ngram_examples"]) == 1
    e = data["ngram_examples"][0]
    assert "chain_b" in e["domain"]
    assert "src1" in e["assembly_mermaid"]
    dma_trace.reset_trace()


def test_atomize_both_destinations():
    from fastapi.testclient import TestClient
    from fabric.main import app
    from fabric import dma_trace
    dma_trace.reset_trace()
    dma_trace.start_trace()
    dma_trace.record(
        module_id="a", module_type="clock_bpm",
        in_ports={}, out_ports={"ticks": 1.0},
        latency_us=10.0, tag="c1:a",
    )
    dma_trace.record(
        module_id="b", module_type="moving_avg",
        in_ports={"in": 1.0}, out_ports={"out": 0.0},
        latency_us=20.0, tag="c1:b",
    )
    dma_trace.record(
        module_id="c", module_type="chart",
        in_ports={"sig": 0.1}, out_ports={},
        latency_us=30.0, tag="c1:c",
    )
    dma_trace.stop_trace()
    c = TestClient(app)
    r = c.post("/api/atomize/run",
               json={"source": "recent", "dest": "both"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "atomic_programs" in data
    assert "ngram_examples" in data
    dma_trace.reset_trace()


def test_atomize_live_no_recording():
    from fastapi.testclient import TestClient
    from fabric.main import app
    from fabric import dma_trace
    dma_trace.stop_trace()
    dma_trace.reset_trace()
    c = TestClient(app)
    r = c.post("/api/atomize/run",
               json={"source": "live", "dest": "atomic"})
    assert r.status_code == 409
