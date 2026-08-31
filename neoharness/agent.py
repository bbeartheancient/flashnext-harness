"""The Hermes-style agent loop: tool calling + BPE actions in one runtime.

Per turn the duty model may emit:

- ``<tool>name {"arg": ...}</tool>`` — a tool call (prompt-side convention,
  same shape as M1 fabric/agent.py; the duty 27B has no native tool_choice)
- ``<action>verb[arg]</action>`` — a BPE harness action (EvoHarness protocol)
- ``<done>summary</done>`` — goal complete, stop the loop
- plain text with none of the above — treated as the final answer

Both tool calls and harness actions burn the same step budget (cost ledger).
BPE writes still mirror into the XYZ n-gram buffer via the Harness callback.
The message window is compacted through the latent lane (compactor.py) when
it crosses the fit budget — long-context goals do not silently drop history.

Hermes features here: tool loop, memory nudge (every NUDGE_EVERY steps the
model is prompted to consider note[...]/recall[...]), subagent delegation
(the ``delegate`` tool), trajectory recording, and skill creation (goals.py
calls skills.create_from_trajectory on success).
"""

from __future__ import annotations

import json
import os
import re

from .actions import dispatch, parse_action
from .compactor import ContextCompactor
from .harness import EpisodeResult, Harness, StepRecord
from .tools import ToolError, ToolRegistry, make_core_tools

TOOL_BLOCK_RE = re.compile(r"<tool>(.*?)</tool>", re.S)
TOOL_NAME_ARGS_RE = re.compile(r"\s*(\w+)\s*(\{.*\})?\s*", re.S)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNC_NAME_RE = re.compile(r"<(?:function|tool)=(\w+)")
PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.S)
CONTENT_BLOCK_RE = re.compile(r"<content>(.*?)</content>", re.S)
DONE_RE = re.compile(r"<done>(.*?)</done>", re.S)

NUDGE_EVERY = int(os.environ.get("NEOH_NUDGE_EVERY", "8"))
MAX_PLAIN_TURNS = int(os.environ.get("NEOH_MAX_PLAIN_TURNS", "2"))
MAX_DEPTH = 1  # subagents cannot themselves delegate


class Agent:
    def __init__(
        self,
        llm,
        harness: Harness | None = None,
        tools: ToolRegistry | None = None,
        compactor: ContextCompactor | None = None,
        max_steps: int = 40,
        nudge_every: int = NUDGE_EVERY,
        depth: int = 0,
        use_bpe: bool = True,
    ):
        self.llm = llm
        self.harness = harness or Harness(llm)
        self.tools = tools or make_core_tools()
        self.compactor = compactor or ContextCompactor(buffer=self.harness.buffer)
        self.max_steps = max_steps
        self.nudge_every = nudge_every
        self.depth = depth
        self.use_bpe = use_bpe
        self.on_step = None  # optional callback(StepRecord), e.g. CLI display
        if depth < MAX_DEPTH:
            self._register_delegate()

    # -- delegate tool (Hermes subagents) --
    def _register_delegate(self) -> None:
        from .tools import Tool

        def delegate(goal: str, context: str = "") -> str:
            sub = Agent(
                self.llm, harness=self.harness, tools=self.tools,
                compactor=self.compactor, max_steps=max(4, self.max_steps // 2),
                depth=self.depth + 1,
            )
            ep = sub.run(f"{goal}\n\nContext: {context}" if context else goal)
            return (f"subagent finished (solved={ep.solved}, "
                    f"{len(ep.steps)} steps): {ep.steps[-1].observation[:500]}"
                    if ep.steps else "subagent did nothing")

        self.tools.register(Tool(
            "delegate", "spawn a subagent for an isolated sub-goal",
            {"goal": "str", "context": "str (optional)"}, delegate,
        ))

    # -- prompt --
    def _system_prompt(self) -> str:
        base = self.harness._system_prompt() if self.use_bpe else (
            "You are a long-horizon task agent.\n"
            "Every turn must contain a <tool> call or <done> — a bare prose "
            "turn is a protocol error. Keep reasoning short; act early."
        )
        return (
            base
            + "\n\n### Tools\nInvoke one tool per turn as "
              '<tool>name {"arg": "value"}</tool> or '
              '<tool>{"name": "name", "arguments": {...}}</tool>.\n'
            + self.tools.spec_text()
            + "\n\nFor large file writes prefer "
              '<tool>write_file {"path": "f"}</tool><content>verbatim body'
              "</content> — the content block is never JSON-escaped."
            + "\nEmit <done>summary</done> when the goal is complete."
            + (" A turn may carry one <tool> call and/or one <action> block."
               if self.use_bpe else "")
        )

    # -- turn parsing --
    def _run_tool(self, reply: str) -> tuple[str, str, str] | None:
        # Qwen3-XML trained form, as emitted by Flash-Next:
        # <tool_call><function=name><parameter=k>v</parameter>...</function></tool_call>
        # (also the degenerate <tool_call><tool=name></tool></tool_call>).
        tc = TOOL_CALL_RE.search(reply)
        if tc:
            seg = tc.group(1)
            nm = FUNC_NAME_RE.search(seg)
            if not nm:
                return "?", seg, f"tool error: cannot parse tool_call: {seg[:120]}"
            name = nm.group(1)
            args: dict = {}
            for k, v in PARAM_RE.findall(seg):
                v = v.strip()
                try:
                    args[k] = json.loads(v)
                except json.JSONDecodeError:
                    args[k] = v
            if not args:
                jm = re.search(r"\{.*\}", seg, re.S)
                if jm:
                    try:
                        args = json.loads(jm.group(0))
                    except json.JSONDecodeError:
                        pass
            raw_args = json.dumps(args)
        else:
            m = TOOL_BLOCK_RE.search(reply)
            if not m:
                return None
            body = m.group(1).strip()
            if body.startswith("{"):
                # Native trained form: <tool>{"name": ..., "arguments": {...}}</tool>
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError as exc:
                    return "?", body, f"tool error: bad JSON tool call: {exc}"
                name = str(payload.get("name", ""))
                args = payload.get("arguments") or payload.get("args") or {}
                raw_args = json.dumps(args)
            else:
                # Harness form: <tool>name {"arg": ...}</tool>
                m2 = TOOL_NAME_ARGS_RE.match(body)
                if not m2 or not m2.group(1):
                    return "?", body, f"tool error: cannot parse tool call: {body[:120]}"
                name, raw_args = m2.group(1), m2.group(2) or ""
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError as exc:
                    return name, raw_args, f"tool error: bad JSON args for {name}: {exc}"
        # Large file bodies defeat JSON escaping under token pressure.
        # A trailing <content>...</content> block supplies/fills the content
        # arg verbatim (measured failure mode, seed-23 long-horizon bench).
        cm = CONTENT_BLOCK_RE.search(reply)
        if cm and isinstance(args, dict) and "content" not in args:
            args["content"] = cm.group(1)
            raw_args = json.dumps(args)
        try:
            return name, raw_args, f"tool {name}: {self.tools.call(name, args)}"
        except ToolError as exc:
            return name, raw_args, f"tool error: {exc}"

    # -- main loop --
    def run(self, goal: str) -> EpisodeResult:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"Goal: {goal}"},
        ]
        steps: list[StepRecord] = []
        tool_calls = harness_calls = invalid_calls = 0
        solved = False
        plain_streak = 0

        for t in range(self.max_steps):
            window = self.compactor.fit(messages)
            reply = self.llm.chat(window)
            messages.append({"role": "assistant", "content": reply})

            observations = []

            done = DONE_RE.search(reply)
            tool = self._run_tool(reply)
            tool_obs = tool[2] if tool else None
            if tool is not None:
                tool_calls += 1
                observations.append(tool_obs)

            if parse_action(reply) is not None and self.use_bpe:
                result = dispatch(reply, self.harness.state,
                                  on_write=self.harness._on_write)
                observations.append(result.observation)
                if result.kind == "harness":
                    harness_calls += 1
                elif result.kind == "invalid":
                    invalid_calls += 1
                steps.append(StepRecord(
                    t=t, action_text=reply,
                    kind="tool" if tool else result.kind,
                    verb=tool[0] if tool else result.verb,
                    arg=tool[1] if tool else result.arg,
                    observation="\n".join(observations),
                    axis=result.axis, detail=result.detail,
                ))
            else:
                kind = "tool" if tool is not None else "answer"
                if tool is None and done is None:
                    invalid_calls += 1
                steps.append(StepRecord(
                    t=t, action_text=reply, kind=kind,
                    verb=tool[0] if tool else "",
                    arg=tool[1] if tool else "",
                    observation="\n".join(observations),
                ))

            if done is not None:
                solved = True
                steps[-1].detail["done"] = done.group(1).strip()
            if self.on_step is not None:
                self.on_step(steps[-1])
            if done is not None:
                break
            if tool is None and not observations:
                # Bare prose turn (often a truncated thinking reply): push
                # back instead of mistaking it for the final answer.
                plain_streak += 1
                if plain_streak >= MAX_PLAIN_TURNS:
                    break
                observations.append(
                    "[harness: no <tool> call, <action> block, or <done> tag "
                    "in your last turn — a bare prose turn is a protocol "
                    "error. Keep reasoning short and emit exactly one tool "
                    "call or action now.]")
                steps[-1].observation = "\n".join(observations)
            else:
                plain_streak = 0

            feedback = "\n".join(observations)
            if self.nudge_every and (t + 1) % self.nudge_every == 0:
                feedback += ("\n\n[harness nudge: you have taken "
                             f"{t + 1} steps — consider recall[...] for prior "
                             "experience and note[...] for new insights]")
            messages.append({
                "role": "user",
                "content": feedback + (
                    "\n\n### Harness state (H_t)\n" + self.harness.state.render()
                    if self.use_bpe else ""
                ),
            })

        ep = EpisodeResult(
            task=goal, steps=steps, solved=solved,
            env_calls=tool_calls, harness_calls=harness_calls,
            invalid_calls=invalid_calls,
        )
        self.harness._record(ep)
        return ep
