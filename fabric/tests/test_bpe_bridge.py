# SPDX-License-Identifier: Apache-2.0
"""Tests for fabric/bpe_bridge.py — injection, reply application, banner."""

from __future__ import annotations

import pytest

from fabric import bpe_bridge


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NEOH_STATE_DIR", str(tmp_path / "neoh"))
    monkeypatch.setenv("FABRIC_OPENCODE_BPE", "1")
    bpe_bridge._state_cache.clear()
    bpe_bridge._pending.clear()
    bpe_bridge._compacted.clear()
    yield


def _body(system="You are OpenCode.", user="continue the task"):
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}


def test_enabled_gate(monkeypatch):
    assert bpe_bridge.enabled()
    monkeypatch.setenv("FABRIC_OPENCODE_BPE", "0")
    assert not bpe_bridge.enabled()


def test_inject_appends_to_system_message():
    out = bpe_bridge.inject(_body(), "s1")
    sysmsg = out["messages"][0]
    assert sysmsg["role"] == "system"
    assert "You are OpenCode." in sysmsg["content"]
    assert "### Harness (persistent state" in sysmsg["content"]
    assert "### Harness state (H_t)" in sysmsg["content"]
    # copy semantics
    assert "### Harness" not in _body()["messages"][0]["content"]


def test_inject_without_system_inserts_one():
    out = bpe_bridge.inject({"messages": [{"role": "user", "content": "hi"}]}, "s1")
    assert out["messages"][0]["role"] == "system"


def test_inject_content_parts_list():
    body = {"messages": [{"role": "system",
                          "content": [{"type": "text", "text": "sys"}]},
                         {"role": "user", "content": "go"}]}
    out = bpe_bridge.inject(body, "s1")
    texts = [p.get("text", "") for p in out["messages"][0]["content"]]
    assert any("Harness state" in t for t in texts)


def test_apply_reply_commit_and_note_persist():
    n = bpe_bridge.apply_reply(
        "Doing work. <action>commit[write solver | attempted]</action> "
        "more text <action>note[prior | verify before done]</action>", "s1")
    assert n == 2
    st = bpe_bridge._state("s1")
    assert st.progress.entries[0]["subgoal"] == "write solver"
    assert st.experience.note_buffer[0]["kind"] == "prior"
    # and it renders into the next injection
    out = bpe_bridge.inject(_body(), "s1")
    assert "write solver" in out["messages"][0]["content"]


def test_track_recall_answers_ride_next_inject():
    bpe_bridge.apply_reply("<action>track[world]</action>", "s1")
    out = bpe_bridge.inject(_body(), "s1")
    assert "### Harness answers" in out["messages"][0]["content"]
    assert "track[world]" in out["messages"][0]["content"]
    # consumed: second inject has no answers block
    out2 = bpe_bridge.inject(_body(), "s1")
    assert "### Harness answers" not in out2["messages"][0]["content"]


def test_compaction_banner_once():
    bpe_bridge.mark_compaction("s1")
    out = bpe_bridge.inject(_body(), "s1")
    assert "was just compacted" in out["messages"][0]["content"]
    out2 = bpe_bridge.inject(_body(), "s1")
    assert "was just compacted" not in out2["messages"][0]["content"]


def test_sessions_isolated():
    bpe_bridge.apply_reply("<action>commit[a | done]</action>", "s1")
    out = bpe_bridge.inject(_body(), "s2")
    assert "commit" in out["messages"][0]["content"]  # cheatsheet
    assert "[done] a" not in out["messages"][0]["content"]


def test_malformed_actions_ignored():
    assert bpe_bridge.apply_reply("no tags at all", "s1") == 0
    assert bpe_bridge.apply_reply("<action>commit[ x | bogus ]</action>", "s1") == 1
    st = bpe_bridge._state("s1")
    assert st.progress.entries == []  # invalid status rejected by the store


def test_bare_action_fallback():
    n = bpe_bridge.apply_reply(
        "Let me record that.\n\ncommit[verify bridge | attempted]\n\nREADY", "s1")
    assert n == 1
    st = bpe_bridge._state("s1")
    assert st.progress.entries[0]["subgoal"] == "verify bridge"


def test_bare_fallback_does_not_override_tagged():
    n = bpe_bridge.apply_reply(
        "commit[fake | done]\n<action>commit[real | done]</action>", "s1")
    st = bpe_bridge._state("s1")
    subs = [e["subgoal"] for e in st.progress.entries]
    assert "real" in subs and "fake" not in subs


def test_bare_fallback_ignores_prose_brackets():
    assert bpe_bridge.apply_reply("see section commit[tee] notes", "s1") == 0
