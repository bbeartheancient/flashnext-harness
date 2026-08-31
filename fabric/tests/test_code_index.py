"""code_index: vocab -> file:line over a scratch tree."""

import os
import sys

import pytest

from fabric import code_index as ci


@pytest.fixture()
def scratch(monkeypatch, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        "WIDGET_LIMIT = 1\n"
        "\n"
        "class Widget:\n"
        "    def spin(self):\n"
        "        return 0\n"
        "\n"
        "def make_widget():\n"
        "    return Widget()\n"
    )
    (tmp_path / "docs" / "plan.md").write_text(
        "# Ship Plan\n"
        "\n"
        "## Phase One\n"
        "text\n"
    )
    (tmp_path / "vendorjunk.py").write_text("def noise(): pass\n")
    monkeypatch.setenv("FABRIC_PI_CWD", str(tmp_path))
    ci.reset_cache()
    yield tmp_path
    ci.reset_cache()


def test_finds_def_with_line(scratch):
    hits = ci.find("make_widget")
    assert len(hits) == 1
    h = hits[0]
    assert h["file"] == os.path.join("pkg", "mod.py")
    assert h["line"] == 7
    assert h["kind"] == "def"


def test_class_const_and_method(scratch):
    hits = ci.find("widget")
    kinds: dict[str, list[int]] = {}
    for h in hits:
        kinds.setdefault(h["kind"], []).append(h["line"])
    assert 3 in kinds.get("class", [])
    assert 1 in kinds.get("const", [])
    assert sorted(kinds.get("def", [])) == [7]
    spin = ci.find("spin")
    assert len(spin) == 1 and spin[0]["line"] == 4


def test_markdown_headings(scratch):
    hits = ci.find("phase one")
    assert len(hits) == 1
    assert hits[0]["file"] == os.path.join("docs", "plan.md")
    assert hits[0]["line"] == 3
    assert hits[0]["kind"] == "md"


def test_case_insensitive_and_substring(scratch):
    assert ci.find("MAKE_WIDGET")
    assert ci.find("widget")  # substring of make_widget + class


def test_staleness_rebuild(scratch):
    mod = scratch / "pkg" / "mod.py"
    assert not ci.find("later_thing")
    mod.write_text("def later_thing():\n    return 1\n")
    # TTL forces a rescan within _TTL_S; shrink it by forcing
    sys.modules["fabric.code_index"].reset_cache()
    hits = ci.find("later_thing")
    assert len(hits) == 1 and hits[0]["line"] == 1


def test_no_hits_empty_query(scratch):
    assert ci.find("") == []
    assert ci.find("zzz_no_such_symbol_zzz") == []


def test_stats_shape(scratch):
    s = ci.stats()
    assert s["files"] >= 2 and s["symbols"] >= 5


def test_tool_word_hints_on_phrase_miss(scratch):
    from fabric.tools import code_index

    out = code_index("widget factory maker")
    assert out["n"] == 0
    assert out.get("word_hits"), "expected word-level fallback hints"
    assert any("make_widget" in h for h in out["word_hits"])
    assert "phrase" in out["note"]
