"""Tests for tools.py — core tools, sandbox, shell allowlist."""

from __future__ import annotations

import pytest

from neoharness.tools import ToolError, make_core_tools


@pytest.fixture()
def reg(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("import os\nprint('hi')\n")
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    return make_core_tools(tmp_path)


def test_read_file(reg):
    out = reg.call("read_file", {"path": "notes.txt"})
    assert "beta" in out
    out2 = reg.call("read_file", {"path": "notes.txt", "offset": 1, "limit": 1})
    assert out2.strip() == "beta"


def test_write_file_and_append(reg):
    reg.call("write_file", {"path": "out/x.txt", "content": "hello"})
    reg.call("write_file", {"path": "out/x.txt", "content": " world", "append": True})
    assert reg.call("read_file", {"path": "out/x.txt"}) == "hello world"


def test_list_dir(reg):
    out = reg.call("list_dir", {"path": "."})
    assert "src/" in out and "notes.txt" in out


def test_grep(reg):
    out = reg.call("grep", {"pattern": "import"})
    assert "src/a.py:1: import os" in out
    assert reg.call("grep", {"pattern": "zzz"}) == "(no matches)"


def test_sandbox_escape_rejected(reg):
    with pytest.raises(ToolError, match="escapes"):
        reg.call("read_file", {"path": "../outside.txt"})
    with pytest.raises(ToolError, match="escapes"):
        reg.call("write_file", {"path": "/etc/evil", "content": "x"})
    with pytest.raises(ToolError, match="escapes"):
        reg.call("list_dir", {"path": "../../.."})


def test_shell_allowlist(reg, monkeypatch):
    monkeypatch.setenv("NEOH_SHELL_ALLOW", "echo,pwd")
    out = reg.call("run_shell", {"command": "echo hi"})
    assert "hi" in out and "(exit 0)" in out
    with pytest.raises(ToolError, match="not in NEOH_SHELL_ALLOW"):
        reg.call("run_shell", {"command": "rm -rf ."})


def test_shell_timeout_is_tool_error(reg, monkeypatch):
    monkeypatch.setenv("NEOH_SHELL_ALLOW", "sleep")
    monkeypatch.setattr("neoharness.tools.SHELL_TIMEOUT_S", 0.2)
    with pytest.raises(ToolError, match="timed out"):
        reg.call("run_shell", {"command": "sleep 5"})


def test_unknown_tool_and_bad_args(reg):
    with pytest.raises(ToolError, match="unknown tool"):
        reg.call("nope", {})
    with pytest.raises(ToolError, match="bad args"):
        reg.call("read_file", {"bogus": 1})


def test_output_cap(reg, monkeypatch):
    monkeypatch.setattr("neoharness.tools.TOOL_MAX_CHARS", 100)
    big = "\n".join(f"line {i} xxxxxxxxxxxxxxxxxxxx" for i in range(50))
    reg.call("write_file", {"path": "big.txt", "content": big})
    out = reg.call("read_file", {"path": "big.txt"})
    assert "truncated" in out
