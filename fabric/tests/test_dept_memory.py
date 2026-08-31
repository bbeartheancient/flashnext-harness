"""Department memory shards (memvid .mv2) vs a temp dir."""

import pytest

from fabric import dept_memory as dm


@pytest.fixture()
def tmp_shards(monkeypatch, tmp_path):
    monkeypatch.setenv("FABRIC_MEMORY_DIR", str(tmp_path / "mem"))
    dm._DIR = tmp_path / "mem"
    dm._handles.clear()
    yield tmp_path
    dm._handles.clear()


def test_add_search_roundtrip(tmp_shards):
    r = dm.add("Medical", "Operator is allergic to penicillin.",
               title="Allergy")
    assert r.get("ok") is True
    out = dm.search("Medical", "penicillin allergy")
    assert out["dept"] == "Medical"
    assert any("penicillin" in h["text"].lower() for h in out["hits"])


def test_charter_seeded_on_create(tmp_shards):
    dm.add("Security", "note one", title="n1")  # forces create + seed
    out = dm.search("Security", "Orbi LAN device map")
    assert any("charter" in (h["title"] or "").lower()
               or "orbi" in (h["text"] or "").lower() for h in out["hits"])


def test_ext_and_slug_resolution(tmp_shards):
    r1 = dm.add("500", "med note A", title="a")
    r2 = dm.add("medical", "med note B", title="b")
    assert r1["dept"] == r2["dept"] == "Medical"


def test_context_for_bounded(tmp_shards):
    for i in range(6):
        dm.add("Command", f"fact {i} " + "x" * 300, title=f"f{i}")
    block = dm.context_for("Command", "fact", budget_chars=800)
    assert len(block) <= 1400  # charter + bounded hits
    assert "COMMAND" in block


def test_empty_inputs():
    assert "error" in dm.add("Command", "")
    assert "error" in dm.search("Command", "")
