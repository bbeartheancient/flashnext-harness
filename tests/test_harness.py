"""Tests for harness.py — end-to-end loop with a stub LLM and toy env."""

from __future__ import annotations

import json

from neoharness.actions import HARNESS_VERBS
from neoharness.bpe import BPEState
from neoharness.harness import Harness


class StubLLM:
    def __init__(self, script):
        self.script = list(script)
        self.messages_seen = []

    def chat(self, messages, **kw):
        self.messages_seen.append(messages)
        if not self.script:
            return "<action>commit[done | done]</action>"
        return self.script.pop(0)


class ToyEnv:
    """One-room world: find the mug, clean it, place it."""

    def __init__(self):
        self.mug_clean = False
        self.placed = False

    def reset(self, task):
        return "You are in the kitchen. You see a countertop."

    def step(self, verb, arg):
        if verb.startswith("clean"):
            self.mug_clean = True
            return "You clean the mug in the sink."
        if verb.startswith("place"):
            if not self.mug_clean:
                return "Nothing happens. The mug is dirty."
            self.placed = True
            return "You place the clean mug in the cabinet."
        return f"You do: {verb} {arg}".strip()

    def solved(self):
        return self.placed


def test_episode_mixed_actions(tmp_path):
    llm = StubLLM([
        "<action>track[world]</action>",
        "<action>recall[mug cleaning]</action>",
        "<action>commit[find mug | done]</action>",
        "<action>note[prior | mugs live on countertops]</action>",
        "<action>clean mug 1 in sink 1</action>",
        "<action>place mug 1 in cabinet 1</action>",
    ])
    h = Harness(llm, state=BPEState(), trajectory_dir=tmp_path)
    ep = h.run_episode("clean and place the mug", ToyEnv())
    assert ep.solved
    assert ep.env_calls == 2 and ep.harness_calls == 4 and ep.invalid_calls == 0
    cost = ep.cost
    assert cost["harness_fraction"] == 4 / 6
    # progress and notes landed in BPE state
    assert h.state.progress.entries[0]["status"] == "done"
    assert h.state.experience.note_buffer[0]["kind"] == "prior"
    # harness state was rendered into the prompt
    assert "### Harness state (H_t)" in llm.messages_seen[0][0]["content"]
    # trajectory JSONL was written
    files = list(tmp_path.glob("ep-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().split("\n")
    header = json.loads(lines[0])
    assert header["task"] == "clean and place the mug"
    assert header["cost"]["solved"] is True
    assert len(lines) == 1 + len(ep.steps)


def test_episode_env_error_and_invalid_action(tmp_path):
    llm = StubLLM([
        "I forgot the tags entirely",
        "<action>teleport to mars</action>",
        "<action>clean mug 1 in sink 1</action>",
        "<action>place mug 1 in cabinet 1</action>",
    ])
    h = Harness(llm, state=BPEState(), trajectory_dir=tmp_path)
    ep = h.run_episode("clean and place", ToyEnv())
    assert ep.invalid_calls == 1
    assert ep.solved


def test_on_write_mirrors_to_buffer(tmp_path):
    class FakeBuffer:
        def __init__(self):
            self.stores = []

        def store_axis(self, axis, text, persist=False, mode="merge"):
            self.stores.append((axis, persist, text))
            return {"ok": True}

    llm = StubLLM([
        "<action>commit[step one | done]</action>",
        "<action>clean mug 1 in sink 1</action>",
        "<action>place mug 1 in cabinet 1</action>",
    ])
    buf = FakeBuffer()
    h = Harness(llm, state=BPEState(), buffer=buf, trajectory_dir=tmp_path)
    ep = h.run_episode("clean and place", ToyEnv())
    assert ep.solved
    assert buf.stores and buf.stores[0][0] == "y"  # progress axis
    assert buf.stores[0][1] is False  # live turns never persist
    assert "step one" in buf.stores[0][2]


def test_consolidate_epoch(tmp_path):
    class FakeBuffer:
        def __init__(self):
            self.stores = []

        def store_axis(self, axis, text, persist=False, mode="merge"):
            self.stores.append((axis, persist))

    llm = StubLLM([])
    buf = FakeBuffer()
    h = Harness(llm, state=BPEState(), buffer=buf, trajectory_dir=tmp_path)
    h.state.experience.note("knives are in the kitchen drawers", kind="prior")
    r = h.consolidate(trajectory_summary="found knives fast")
    assert len(r.added) == 2
    assert not h.state.experience.note_buffer
    # consolidation is the persist=True lane, Z axis
    assert buf.stores == [("z", True)]


def test_protocol_mentions_all_verbs():
    from neoharness.harness import PROTOCOL_HELP

    for verb in HARNESS_VERBS:
        assert verb in PROTOCOL_HELP
