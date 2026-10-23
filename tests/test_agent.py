"""Tests for agent.py — tool loop, BPE interleave, nudge, delegate, done."""

from __future__ import annotations

import json
import threading

from neoharness.agent import Agent
from neoharness.bpe import BPEState
from neoharness.harness import Harness
from neoharness.tools import make_core_tools


class StubLLM:
    # Thread-safe: the 1:4:1 delegate fans four subagents out concurrently,
    # so the scripted list is popped under a lock.
    def __init__(self, script):
        self.script = list(script)
        self._lock = threading.Lock()

    def chat(self, messages, **kw):
        with self._lock:
            return self.script.pop(0) if self.script else "<done>finished</done>"


def make_agent(tmp_path, script, **kw):
    llm = StubLLM(script)
    h = Harness(llm, state=BPEState(), trajectory_dir=tmp_path / "traj")
    return Agent(llm, harness=h, tools=make_core_tools(tmp_path), **kw)


def test_tool_call_executes(tmp_path):
    agent = make_agent(tmp_path, [
        '<tool>write_file {"path": "a.txt", "content": "hello"}</tool>',
        '<tool>read_file {"path": "a.txt"}</tool>',
        "<done>wrote and verified a.txt</done>",
    ])
    ep = agent.run("create a.txt with hello")
    assert ep.solved
    assert ep.env_calls == 2  # two tool calls
    assert (tmp_path / "a.txt").read_text() == "hello"
    assert "hello" in ep.steps[1].observation
    assert ep.steps[-1].detail["done"] == "wrote and verified a.txt"


def test_tool_error_is_fed_back_not_raised(tmp_path):
    agent = make_agent(tmp_path, [
        '<tool>read_file {"path": "../escape"}</tool>',
        '<tool>nope {}</tool>',
        "<done>ok</done>",
    ])
    ep = agent.run("probe errors")
    assert "escapes" in ep.steps[0].observation
    assert "unknown tool" in ep.steps[1].observation
    assert ep.solved


def test_bpe_and_tool_interleaved_same_turn(tmp_path):
    agent = make_agent(tmp_path, [
        '<action>recall[mugs]</action>\n<tool>list_dir {}</tool>',
        "<done>ok</done>",
    ])
    agent.harness.state.experience.add("prior", "mugs", "mugs on countertops")
    ep = agent.run("find mugs")
    assert ep.harness_calls == 1 and ep.env_calls == 1
    assert "countertops" in ep.steps[0].observation
    assert "tool list_dir" in ep.steps[0].observation


def test_plain_text_turn_pushes_back_then_recovers(tmp_path):
    agent = make_agent(tmp_path, [
        "Let me think about this at length with no tags.",
        '<tool>list_dir {}</tool>',
        "<done>ok</done>",
    ])
    ep = agent.run("do a thing")
    assert ep.solved
    # the pushback feedback was appended after the bare turn
    assert "protocol error" in ep.steps[0].observation
    assert ep.env_calls == 1


def test_consecutive_plain_turns_end_loop(tmp_path):
    agent = make_agent(tmp_path, ["no tags one", "no tags two", "<done>late</done>"])
    ep = agent.run("what is the answer")
    assert not ep.solved
    assert len(ep.steps) == 2  # gave up after MAX_PLAIN_TURNS bare turns
    assert ep.steps[0].kind == "answer"


def test_memory_nudge_fires(tmp_path):
    script = ['<tool>list_dir {}</tool>'] * 8 + ["<done>ok</done>"]
    llm = StubLLM(script)
    h = Harness(llm, state=BPEState(), trajectory_dir=tmp_path / "traj")
    seen = []

    class SpyLLM(StubLLM):
        def chat(self, messages, **kw):
            seen.extend(m["content"] for m in messages if m["role"] == "user")
            return super().chat(messages, **kw)

    llm = SpyLLM(script)
    h = Harness(llm, state=BPEState(), trajectory_dir=tmp_path / "traj")
    agent = Agent(llm, harness=h, tools=make_core_tools(tmp_path), nudge_every=4)
    ep = agent.run("long task")
    assert ep.solved
    assert any("harness nudge" in c for c in seen)


def test_delegate_expands_1_4_1(tmp_path):
    """The turn gate at phase 0 expands one goal into 4 workers and contracts to 1."""
    (tmp_path / "data.txt").write_text("payload")
    script = [
        # turn 0: the agent's own reply asks to delegate
        '<tool>delegate {"goal": "get the payload", "context": "use read_file"}</tool>',
        # then the EXPAND stage enumerates four independent sub-goals,
        # then the four workers run concurrently (order under the lock varies)
        "1. read data.txt\n2. read data.txt again\n3. read it a third time\n4. read it once more",
        '<tool>read_file {"path": "data.txt"}</tool>',
        "<done>worker done</done>",
    ]
    agent = make_agent(tmp_path, script, fanout=4, turn_len=8)
    ep = agent.run("get the payload via a subagent")
    obs = ep.steps[0].observation
    assert "[1:4:1 expansion] 4 workers -> all 4 carried verbatim" in obs
    for i in (1, 2, 3, 4):                      # all four report, none orphaned
        assert f"[worker {i}]" in obs
    assert len(agent._last_expansion) == 4


def test_gate_closed_inside_a_turn(tmp_path):
    """Inside a turn the seam is closed: delegation is refused, not queued."""
    agent = make_agent(tmp_path, [
        '<tool>list_dir {"path": "."}</tool>',       # turn 0 -> phase 0/4
        '<tool>delegate {"goal": "anything"}</tool>',  # turn 1 -> phase 1/4, closed
        "<done>stopped</done>",
    ], fanout=2, turn_len=4)
    ep = agent.run("work then delegate")
    assert ep.solved
    assert "gate closed" in ep.steps[1].observation
    assert "[1:4:1 expansion]" not in ep.steps[1].observation


def test_gate_reopens_on_the_turn_boundary(tmp_path):
    """turn_len=1 makes every step a boundary, so the gate is always open."""
    agent = make_agent(tmp_path, [
        '<tool>delegate {"goal": "anything"}</tool>',
        "<done>stopped</done>",
    ], fanout=2, turn_len=1)
    ep = agent.run("delegate immediately")
    assert "[1:4:1 expansion] 2 workers" in ep.steps[0].observation


def test_qwen_xml_tool_call_grammar(tmp_path):
    # Flash-Next trained form: qwen3_xml inside <tool_call> wrappers.
    agent = make_agent(tmp_path, [
        '<tool_call>\n<function=write_file>\n<parameter=path>\nx.txt\n</parameter>\n'
        '<parameter=content>\nxml body\n</parameter>\n</function>\n</tool_call>',
        "<done>ok</done>",
    ])
    ep = agent.run("use the qwen xml grammar")
    assert ep.solved and ep.env_calls == 1
    assert (tmp_path / "x.txt").read_text() == "xml body"


def test_qwen_xml_degenerate_tool_name_form(tmp_path):
    # observed live: <tool_call><tool=list_dir></tool></tool_call>
    agent = make_agent(tmp_path, [
        '<tool_call>\n<tool=list_dir>\n</tool>\n</tool_call>',
        "<done>ok</done>",
    ])
    ep = agent.run("list with degenerate form")
    assert ep.solved and ep.env_calls == 1


def test_native_trained_tool_grammar(tmp_path):
    agent = make_agent(tmp_path, [
        '<tool>\n{"name": "write_file", "arguments": {"path": "n.txt", "content": "native"}}\n</tool>',
        "<done>ok</done>",
    ])
    ep = agent.run("use the native grammar")
    assert ep.solved and ep.env_calls == 1
    assert (tmp_path / "n.txt").read_text() == "native"


def test_content_block_supplies_verbatim_body(tmp_path):
    body = 'import json\nprint("unescaped "quotes" and \\backslashes\\")\n'
    agent = make_agent(tmp_path, [
        '<tool>write_file {"path": "s.py"}</tool>\n<content>' + body + "</content>",
        "<done>ok</done>",
    ])
    ep = agent.run("write a script")
    assert ep.solved
    assert (tmp_path / "s.py").read_text() == body


def test_trajectory_includes_tool_steps(tmp_path):
    agent = make_agent(tmp_path, [
        '<tool>list_dir {}</tool>',
        "<done>ok</done>",
    ])
    ep = agent.run("list things")
    files = list((tmp_path / "traj").glob("ep-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().split("\n")
    assert json.loads(lines[0])["cost"]["solved"] is True
    assert len(lines) == 1 + len(ep.steps)


def test_agent_without_bpe_omits_harness_state(tmp_path):
    llm = StubLLM(['<tool>list_dir {}</tool>', "<done>ok</done>"])
    agent = Agent(llm, harness=Harness(llm, state=BPEState(),
                                       trajectory_dir=tmp_path / "t"),
                  tools=make_core_tools(tmp_path), use_bpe=False)
    prompt = agent._system_prompt()
    assert "### Belief" not in prompt and "track[" not in prompt
    ep = agent.run("list things")
    assert ep.solved and ep.harness_calls == 0


def test_xyz_buffer_rate_override(monkeypatch, tmp_path):
    import sys
    from pathlib import Path

    import numpy as np

    from neoharness import vectors
    from neoharness.vectors import BufferWorker, XYZBuffer

    seen = {}

    def spy_gate(arr, rate=None, gate=None):
        seen["rate"] = rate
        return True, 0.99

    monkeypatch.setattr(vectors, "passes_gate", spy_gate)
    monkeypatch.setattr(
        vectors, "capture_hidden",
        lambda text, base_url=None: np.zeros((4, 5120), dtype=np.float16))
    fake = str(Path(__file__).parent / "fake_buffer_worker.py")
    buf = XYZBuffer(worker=BufferWorker(cmd=[sys.executable, fake],
                                        spawn_timeout=30, request_timeout=10),
                    enabled=True, rate=0.75)
    buf.store_axis("x", "state text")
    assert seen["rate"] == 0.75
    buf.worker._proc.kill()
