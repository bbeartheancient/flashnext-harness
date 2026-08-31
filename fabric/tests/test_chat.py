"""Chat endpoint with TestModel (no duty vLLM required) + prompt-side parser."""

import asyncio
import json

from fabric.agent import parse_tool_call, _strip_think, _dangling_tool, DutyAgent

from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from fabric import main as mainmod
from fabric.main import app
from fabric.tools import TOOLS


client = TestClient(app)


def test_parse_tool_xml():
    assert parse_tool_call('<tool name="ship_status">{}</tool>') == ("ship_status", {})
    name, args = parse_tool_call(
        'thinking\n<tool name="terrain_los">{"tx": [1, 2, 3], "rx": [4, 5, 6]}</tool>')
    assert name == "terrain_los" and args["tx"] == [1, 2, 3]
    assert parse_tool_call("All slots are up.") is None
    name, args = parse_tool_call("<function=log_recent><arg_key>limit</arg_key><arg_value>5</arg_value></function>")
    assert name == "log_recent" and args["limit"] == 5
    assert _strip_think("foo</think>\nbar") == "bar"
    assert _strip_think("<think>reasoning cut mid") == ""
    assert _strip_think("prose <think>cut") == "prose"
    assert _strip_think("plain answer") == "plain answer"


def test_dangling_tool_detection():
    assert _dangling_tool('<tool name="read">{"path":"fabric/ha')
    assert _dangling_tool("<function=read>{")
    assert not _dangling_tool('<tool name="read">{}</tool>')
    assert not _dangling_tool("plain final text")
    assert not _dangling_tool("")


def test_duty_agent_repairs_truncated_tool_call():
    agent = DutyAgent(model_name="m", base="http://x/v1", api_key="k",
                      orchestrator="local")
    seen = []
    seq = [
        '<tool name="ship_status">{"arg": 1',  # truncated, no </tool>
        "All slots reported.",
    ]

    def fake_complete(messages):
        seen.append(messages[-1]["content"])
        return seq.pop(0)

    agent._complete = fake_complete
    result = asyncio.run(agent.run("status"))
    assert result.output == "All slots reported."
    assert result._tools == []
    assert len(seen) == 2
    assert "cut off" in seen[1]
    assert seen[0] == "status"


def test_duty_agent_emits_events(monkeypatch):
    import fabric.agent as amod

    monkeypatch.setattr(amod, "_dispatch",
                        lambda name, args: {"slots": "up"})
    agent = DutyAgent(model_name="m", base="http://x/v1", api_key="k",
                      orchestrator="local")
    seq = ['<tool name="ship_status">{"arg": 1}</tool>', "All up."]

    def fake_complete(messages):
        return seq.pop(0)

    agent._complete = fake_complete
    evs: list[dict] = []
    res = asyncio.run(agent.run("status", on_event=evs.append))
    kinds = [e["t"] for e in evs]
    assert kinds == ["llm", "tool", "result", "llm"]
    assert evs[0]["round"] == 1 and evs[3]["round"] == 2
    assert evs[1]["name"] == "ship_status" and "arg" in evs[1]["args"]
    assert evs[2]["ok"] is True and evs[2]["ms"] >= 0
    assert res.output == "All up."
    assert res._tools == ["ship_status"]


def test_chat_stream_progress_events():
    class _Result:
        output = "done"
        last_result = None

        def all_messages(self):
            return []

    class _Agent:
        orchestrator = "local"

        async def run(self, message, instructions=None, on_event=None):
            if on_event:
                on_event({"t": "llm", "round": 1})
                on_event({"t": "tool", "name": "ship_status", "args": "{}"})
                on_event({"t": "result", "name": "ship_status", "ok": True,
                          "ms": 12, "preview": "{}"})
            return _Result()

    prev = mainmod._agent
    mainmod._agent = _Agent()
    try:
        with client.stream("POST", "/chat/stream",
                           json={"message": "status"}) as r:
            assert r.status_code == 200
            assert "event-stream" in r.headers.get("content-type", "")
            rows = [line for line in r.iter_lines()
                    if line.startswith("data: ")]
    finally:
        mainmod._agent = prev
    evs = [json.loads(line[len("data: "):]) for line in rows]
    assert [e["t"] for e in evs] == ["llm", "tool", "result", "final"]
    assert evs[-1]["reply"] == "done"
    assert evs[-1]["tools"] == []
    assert evs[-1]["orchestrator"] == "local"
    assert any(e["t"] == "tool" and e["name"] == "ship_status" for e in evs)


def test_chat_stream_error_event():
    class _Agent:
        orchestrator = "local"

        async def run(self, message, instructions=None, on_event=None):
            raise RuntimeError("orchestrator exploded")

    prev = mainmod._agent
    mainmod._agent = _Agent()
    try:
        with client.stream("POST", "/chat/stream",
                           json={"message": "boom"}) as r:
            assert r.status_code == 200
            rows = [line for line in r.iter_lines()
                    if line.startswith("data: ")]
    finally:
        mainmod._agent = prev
    evs = [json.loads(line[len("data: "):]) for line in rows]
    assert len(evs) == 1 and evs[0]["t"] == "error"
    assert "exploded" in evs[0]["detail"]


def test_round_cap_env_and_marker(monkeypatch):
    import fabric.agent as amod

    monkeypatch.setenv("FABRIC_AGENT_MAX_ROUNDS", "4")
    monkeypatch.setattr(amod, "_dispatch", lambda n, a: {"ok": True})
    agent = DutyAgent(model_name="m", base="http://x/v1", api_key="k",
                      orchestrator="local")

    def fake_complete(messages):
        return '<tool name="ship_status">{}</tool>'

    agent._complete = fake_complete
    evs: list[dict] = []
    res = asyncio.run(agent.run("go", on_event=evs.append))
    assert res._tools == ["ship_status"] * 4
    assert evs[-1]["t"] == "capped" and evs[-1]["rounds"] == 4
    assert "4-leg tool limit" in res.output


def test_fit_history_elides_then_keeps_tail():
    import fabric.agent as amod

    big_tool = ("Tool grep returned:\n" + "x" * 20_000
                + "\nGive the final answer, or call another tool.")
    msgs = [{"role": "system", "content": "S" * 500},
            {"role": "user", "content": "start"}]
    for _ in range(6):
        msgs.append({"role": "assistant", "content": "<tool> x</tool> " * 60})
        msgs.append({"role": "user", "content": big_tool})
    msgs.append({"role": "assistant", "content": "final answer"})
    out = amod._fit_history(msgs, 45_000)
    assert amod._history_chars(out) <= 45_000
    assert out[0]["role"] == "system"
    assert out[-1]["content"] == "final answer"
    assert any("[earlier tool output elided]" in m["content"] for m in out)
    tight = amod._fit_history(msgs, 8_000)
    assert amod._history_chars(tight) <= 8_000
    assert tight[0]["role"] == "system"
    assert tight[-1]["content"] == "final answer"


def test_fit_history_noop_under_budget():
    import fabric.agent as amod

    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"}]
    assert amod._fit_history(msgs, 8_000) == msgs


def test_fit_history_keeps_tool_block_exemplars():
    """Long reasoning-style assistant turns must keep their <tool> block
    when elided — head-truncation dropped every exemplar and the model
    stopped calling tools once the run crossed the fit budget."""
    import fabric.agent as amod

    think = "let me reason about this at great length " * 80
    msgs = [{"role": "system", "content": "S" * 500},
            {"role": "user", "content": "start"}]
    for _ in range(6):
        msgs.append({"role": "assistant", "content":
                     think + '<tool name="ship_status">{}</tool>'})
        msgs.append({"role": "user", "content":
                     "Tool ship_status returned:\n" + "y" * 20_000
                     + "\nGive the final answer, or call another tool."})
    msgs.append({"role": "assistant", "content": "final answer"})
    out = amod._fit_history(msgs, 45_000)
    assert amod._history_chars(out) <= 45_000
    elided = [m for m in out if "[earlier assistant turn elided]"
              in (m.get("content") or "")]
    assert elided, "expected some assistant turns to be elided"
    for m in elided:
        assert '<tool name="ship_status">{}</tool>' in m["content"]
        # no dangling opener: every kept block is complete
        assert m["content"].count("<tool ") == m["content"].count("</tool>")


def test_ctx_error_retry(monkeypatch):
    import fabric.agent as amod

    monkeypatch.setenv("FABRIC_CTX_CHAR_BUDGET", "20000")
    monkeypatch.setattr(amod, "_dispatch", lambda n, a: {"ok": True})
    agent = DutyAgent(model_name="m", base="http://x/v1", api_key="k",
                      orchestrator="local")
    calls = {"n": 0}

    def fake_complete(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(
                "This model's maximum context length is 32768 tokens")
        return "recovered"

    agent._complete = fake_complete
    evs: list[dict] = []
    res = asyncio.run(agent.run("hello", on_event=evs.append))
    assert calls["n"] == 2
    assert res.output == "recovered"
    ctx_events = [e for e in evs if e["t"] == "ctx"]
    assert len(ctx_events) == 1 and ctx_events[0]["retry"] is True


def test_chat_empty_rejected():
    assert client.post("/chat", json={"message": "  "}).status_code == 400


def test_chat_ship_status_tool(monkeypatch):
    agent = Agent(
        TestModel(call_tools=["ship_status"],
                  custom_output_text="All slots reported."),
        name="fabric_test",
        tools=TOOLS,
    )
    prev = mainmod._agent
    mainmod._agent = agent
    try:
        r = client.post("/chat", json={"message": "Computer, status",
                                      "department": 100})
        assert r.status_code == 200
        body = r.json()
        assert "slots" in body["reply"].lower() or "reported" in body["reply"].lower()
        assert "ship_status" in body["tools"]
        assert body["department"] == 100
    finally:
        mainmod._agent = prev


def test_chat_unknown_department_still_replies():
    agent = Agent(
        TestModel(call_tools=[], custom_output_text="Command."),
        name="fabric_test",
        tools=TOOLS,
    )
    prev = mainmod._agent
    mainmod._agent = agent
    try:
        r = client.post("/chat", json={"message": "hello", "department": 999})
        assert r.status_code == 200
        assert r.json()["department"] is None
        assert r.json()["reply"]
        assert r.json()["microapp"] is None
    finally:
        mainmod._agent = prev


def test_chat_returns_microapp(monkeypatch):
    class ToolCallPart:
        tool_name = "generate_microapp"

    class _Msg:
        parts = [ToolCallPart()]

    class _Result:
        output = "Habit tracker is on the wall."
        last_result = {
            "spec": {"id": "app_habit", "title": "7-day habit tracker"},
            "figure": {"type": "html", "html": "<div></div>", "caption": "habit"},
        }

        def all_messages(self):
            return [_Msg()]

    class _Agent:
        async def run(self, message, instructions=None):
            return _Result()

    prev = mainmod._agent
    mainmod._agent = _Agent()
    try:
        r = client.post("/chat", json={"message": "make a 7-day habit tracker"})
        assert r.status_code == 200
        body = r.json()
        assert body["microapp"]["id"] == "app_habit"
        assert body["figure"]["type"] == "html"
        assert "generate_microapp" in body["tools"]
    finally:
        mainmod._agent = prev
