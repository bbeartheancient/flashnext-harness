"""Tests for __main__.py — the CLI runner."""

from __future__ import annotations

import pytest

from neoharness.__main__ import build_parser, main


class StubLLM:
    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, **kw):
        return self.script.pop(0) if self.script else "<done>fallback</done>"


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("NEOH_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "notes.txt").write_text("alpha beta gamma\n")
    return tmp_path


def test_cli_solves_goal_and_streams_steps(ws, capsys):
    llm = StubLLM([
        '<tool>read_file {"path": "notes.txt"}</tool>',
        '<tool>write_file {"path": "out.txt", "content": "summary"}</tool>',
        "<action>commit[goal done | done]</action>\n<done>wrote out.txt</done>",
    ])
    rc = main(["summarize notes", "--workspace", str(ws)], llm=llm)
    out = capsys.readouterr().out
    assert rc == 0
    assert "> t0 [tool] read_file" in out
    assert "solved: True" in out
    assert "skill:" in out
    assert "### Harness state (final)" in out
    assert (ws / "out.txt").read_text() == "summary"


def test_cli_quiet_suppresses_steps(ws, capsys):
    llm = StubLLM([
        '<tool>list_dir {}</tool>',
        "<done>listed</done>",
    ])
    rc = main(["list the workspace", "--workspace", str(ws), "--quiet"], llm=llm)
    out = capsys.readouterr().out
    assert rc == 0
    assert "[tool] list_dir" not in out
    assert "solved: True" in out


def test_cli_failure_exit_code(ws, capsys):
    llm = StubLLM(['<tool>list_dir {}</tool>'] * 40)
    rc = main(["never finishes", "--workspace", str(ws), "--max-steps", "3"],
              llm=llm)
    out = capsys.readouterr().out
    assert rc == 1
    assert "solved: False" in out


def test_parser_subgoal_repeatable():
    args = build_parser().parse_args(
        ["goal", "--subgoal", "a", "--subgoal", "b", "--latent"])
    assert args.subgoal == ["a", "b"]
    assert args.latent is True


def test_cli_latent_flag_sets_env(ws, monkeypatch):
    import os
    monkeypatch.delenv("NEOH_LATENT", raising=False)
    llm = StubLLM(["<done>ok</done>"])
    main(["g", "--workspace", str(ws), "--latent", "--quiet"], llm=llm)
    assert os.environ["NEOH_LATENT"] == "1"
