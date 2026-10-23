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

from . import handoff
from .searcher import Searcher
from .actions import dispatch, parse_action
from .compactor import ContextCompactor
from .sliding_gate import SlidingGate
from .ltm import LTMStore
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

# --- the turn gate (posenc/rotor.c) ---------------------------------------------
# The rotor closes a cycle with an IDENTITY seam, so accumulated error never
# compounds; its Q2 result is the shape we borrow: a growing absolute counter
# becomes a FIXED-WIDTH bounded phase plus an unbounded dimensionless turn.
# Applied to the step loop that means delegation is a SCHEDULE, not a sampled
# behaviour: the seam opens on a turn boundary and closes inside the turn.
#
# cycle.c measured HOLD ~= INTERP at every realistic rate (HOLD actually won at
# 1 deg and 45 deg), so refreshing at the boundary beats interpolating inside
# it.  We therefore never interpolate: the gate is a boundary, nothing more.
TURN_LEN = int(os.environ.get("NEOH_TURN_LEN", "8"))   # cycle length, in steps
FANOUT = int(os.environ.get("NEOH_FANOUT", "4"))       # subagents per expansion

# The sliding window gate is OFF by default. It changes what the model sees, so
# it is opt-in until measured on a real goal; NEOH_COMPACTOR=threshold forces
# the old behaviour even when the gate is on.
SLIDING_GATE_ENABLED = os.environ.get("NEOH_WINDOW_GATE", "0") == "1"
COMPACTOR_MODE = os.environ.get("NEOH_COMPACTOR", "auto")   # auto|threshold

# Long-term recall. Defaults to the gate's own flag so enabling the gate also
# enables the store it retires into -- separating them silently produces a gate
# that discards without remembering, which is strictly worse than no gate.
RECALL_ENABLED = os.environ.get("NEOH_RECALL", "").strip() or None
if RECALL_ENABLED is None:
    RECALL_ENABLED = SLIDING_GATE_ENABLED
else:
    RECALL_ENABLED = RECALL_ENABLED not in ("0", "false", "no", "")
RECALL_K = int(os.environ.get("NEOH_RECALL_K", "3"))
RECALL_CHARS = int(os.environ.get("NEOH_RECALL_CHARS", "1200"))
# Fired on a turn that just retired spans. Kept short and imperative: it has to
# survive being the last thing before the model speaks.
RECALL_NUDGE = (
    "[harness: this turn compacted earlier context into long-term memory. "
    "If your answer depends on anything established earlier, emit "
    "<action>recall[what you need]</action> THIS turn before answering.]"
)


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
        sub_llm=None,
        fanout: int = FANOUT,
        turn_len: int = TURN_LEN,
        searcher=None,
        window_gate=None,
        ltm=None,
    ):
        self.llm = llm
        self.harness = harness or Harness(llm)
        self.tools = tools or make_core_tools()
        self.compactor = compactor or ContextCompactor(buffer=self.harness.buffer)
        # The sliding window gate (sliding_gate.py). Opt-in behind
        # NEOH_WINDOW_GATE so the threshold compactor remains the default until
        # the sine window has been measured on a real goal. When enabled,
        # run() calls gate.apply() and the compactor is bypassed.
        self.window_gate = window_gate
        if self.window_gate is None and SLIDING_GATE_ENABLED:
            self.window_gate = SlidingGate(turn_len=turn_len)
        # The readable side of retirement. Enabled by default WITH the gate
        # (NEOH_RECALL defaults to the gate's own flag) so the two cannot be
        # separated by accident: a gate that retires into nothing is the blind
        # rotation this was built to replace. Pass ltm=False to opt out.
        self.ltm = None
        if ltm is None:
            if RECALL_ENABLED and self.window_gate is not None:
                self.ltm = LTMStore()
        elif ltm is not False:
            self.ltm = ltm
        self.max_steps = max_steps
        self.nudge_every = nudge_every
        self.depth = depth
        self.use_bpe = use_bpe
        self.on_step = None  # optional callback(StepRecord), e.g. CLI display
        # the gate.  sub_llm is the pool the subagents decode on (:8081 in the
        # two-instance layout, the same :8080 batch pool in the layer-split one).
        self.sub_llm = sub_llm
        # P1: the fan-out worker. A Searcher needs no model; when one is given
        # (or NEOH_SUBAGENT=llm) we fall back to the old LLM subagent. Default is
        # deterministic, because the LLM version lost 2 of 4 workers and ran
        # 0.82x serial (see searcher.py for the measurements).
        self.fanout = max(1, fanout)
        # fanout is read here, so it must already be set (it was set first)
        self.searcher = searcher or (None if os.environ.get("NEOH_SUBAGENT") == "llm"
                                     else Searcher(tools=self.tools, fanout=self.fanout))
        self.turn_len = max(0, turn_len)
        self._turn = 0
        self._fired_turn = None      # law cron_due_fires_once: once per boundary
        self._last_expansion = None
        if depth < MAX_DEPTH:
            self._register_delegate()

    # -- delegate: the 1:4:1 turn gate (expand -> 4 subagents -> consolidate) --
    def _gate_state(self) -> str:
        """(turn, phase) and whether the seam is open right now."""
        phase = self._turn % self.turn_len if self.turn_len else 0
        return phase, phase == 0

    def _decompose(self, goal: str, context: str = "") -> list[str]:
        """The EXPAND stage: ask the agent model for `fanout` independent sub-goals.

        Falls back to framing the same goal from `fanout` angles, because a
        small model that cannot enumerate still has to produce useful fan-out.
        """
        n = self.fanout
        ask = (f"Break this goal into exactly {n} independent sub-goals that can be "
               f"pursued in parallel by separate workers, each answering one question. "
               f"One per line, numbered 1..{n}, no prose.\n\nGoal: {goal}\n"
               + (f"Context: {context}" if context else ""))
        try:
            raw = self.llm.chat([{"role": "user", "content": ask}], max_tokens=400)
        except Exception as e:                                    # noqa: BLE001
            raw = ""
        parts = [ln.strip() for ln in (raw or "").splitlines()
                 if re.match(r"\s*\d+[.)]", ln)]
        if len(parts) >= n:
            return [re.sub(r"^\s*\d+[.)]\s*", "", ln) for ln in parts[:n]]
        angles = ["map the whole surface", "find the counter-example",
                  "find the fastest working path", "check the edges and failure modes"]
        return [f"{goal} (angle: {angles[i % len(angles)]})" for i in range(n)]

    def _one_subagent(self, goal: str, context: str = "") -> str:
        """One fan-out worker.

        Deterministic by default (Searcher): it greps the workspace for the
        sub-goal's salient terms and reports verbatim lines. The LLM subagent is
        kept behind NEOH_SUBAGENT=llm for comparison."""
        if self.searcher is not None:
            return self.searcher.search(goal, context)
        llm = self.sub_llm or self.llm
        sub = Agent(llm, harness=Harness(llm), tools=self.tools,
                    compactor=ContextCompactor(), max_steps=max(4, self.max_steps // 2),
                    depth=self.depth + 1, use_bpe=False,
                    fanout=1, turn_len=0, searcher=None)
        ep = sub.run(f"{goal}\n\nContext: {context}" if context else goal)
        if not ep.steps:
            return "(subagent produced nothing)"
        return ep.steps[-1].observation[:700]

    def _consolidate(self, goal: str, reports: list[str]) -> str:
        """The CONTRACT stage: 4 reports -> 1, WITHOUT an agent-written summary.

        Backplane's handoff.bend is explicit about why:

            "It is a budgeted selection of the source thread's own messages and
             tool lines, NEVER AN AGENT-WRITTEN SUMMARY: short conversations go
             whole; longer ones keep their newest items that fit the budget,
             plus the first request, and say how many earlier items were left
             out and where to read them."

        So this does not ask the main model to merge anything. It selects the
        newest worker reports that fit the budget, verbatim, and names what it
        left out. A summary would drop facts with no trace; this does not.
        Laws: handoff_goes_first (the banner leads), handoff_never_runs_alone
        (a handoff never starts a turn - it rides on this tool observation,
        which follows the agent's own delegate call), hand_no_holes, hand_cut_says.
        """
        items = [f"[worker {i + 1}] {r}" for i, r in enumerate(reports)]
        sel = handoff.select(items, budget=handoff.BUDGET)
        body = "\n\n".join(sel.kept)
        if sel.omitted:
            head = (f"[1:4:1 expansion] {len(items)} workers -> {len(sel.kept)} carried "
                    f"verbatim; {sel.omitted} earlier worker(s) left out at the "
                    f"{handoff.BUDGET}-char budget")
        else:
            head = f"[1:4:1 expansion] {len(items)} workers -> all {len(sel.kept)} carried verbatim"
        self._last_expansion = reports
        return head + "\n\n" + body

    # -- delegate tool (Hermes subagents) --
    def _register_delegate(self) -> None:
        from .tools import Tool

        def delegate(goal: str, context: str = "") -> str:
            phase, open_now = self._gate_state()
            # law cron_due_fires_once: "fires a routine once for any number of
            # missed times, and never twice at the same moment". Two calls in
            # one turn are the same moment, so the second is refused rather than
            # run as a second expansion.
            if open_now and self._fired_turn == self._turn:
                return (f"[gate already fired] turn={self._turn}: this boundary has "
                        f"already expanded. Continue with your own tools; delegation "
                        f"reopens at the next turn boundary.")
            if not open_now:
                # The seam is identity: inside a turn the gate does not open.
                return (f"[gate closed] turn={self._turn} phase={phase}/{self.turn_len}; "
                        f"delegation opens only at phase 0. Continue with your own tools "
                        f"for {self.turn_len - phase} more step(s), then delegate.")
            self._fired_turn = self._turn
            # SEQUENTIAL on purpose. Measured: deterministic workers are I/O+regex
            # bound and the GIL serializes them, so ThreadPoolExecutor made the
            # fan-out SLOWER (0.12x-0.64x on 65-400 file corpora). Threads were
            # only ever a win for the old LLM workers, which blocked on HTTP.
            return self._consolidate(goal, [
                self._one_subagent(g, context)
                for g in self._decompose(goal, context)
            ])

        self.tools.register(Tool(
            "delegate", f"expand this goal into {self.fanout} parallel subagents and "
                        f"consolidate (opens on a turn boundary, phase 0 of {self.turn_len})",
            {"goal": "str", "context": "str (optional)"}, delegate,
        ))

    # -- prompt --
    def _system_prompt(self) -> str:
        base = self.harness._system_prompt() if self.use_bpe else (
            "You are a long-horizon task agent.\n"
            "Every turn must contain a <tool> call or <done> — a bare prose "
            "turn is a protocol error. Keep reasoning short; act early."
        )
        recall_help = ""
        if self.ltm is not None:
            # The nudge existed but was never acted on. Two reasons, both
            # measured rather than assumed:
            #
            #  1. recall[] searched only the distilled skill store, which stays
            #     empty on most runs -- the model was told to use a verb with
            #     nothing to find.
            #  2. Even once it searched LTM, a FREE-RUNNING duty model emitted
            #     ZERO <action> blocks across 12 turns. Asked directly it emits
            #     one perfectly ('<action>recall[vault code for obj_delta]
            #     </action>', verified), so the blocker is not capability --
            #     it is that nothing in the turn DEMANDED one.
            #
            # So the verb is advertised AND the obligation is explicit.
            recall_help = (
                "\n\n### Long-term recall\n"
                "Earlier turns are compacted out of your window and kept in long-term "
                "memory. To read it back, emit exactly\n"
                "  <action>recall[what you are looking for]</action>\n"
                "It searches both your distilled skills and the retired turns, and "
                "returns them as `(skill/...)` or `(ltm/turn N)` lines.\n"
                "RULE: if this turn's answer depends on any value, name, path or "
                "decision established earlier in this session, you MUST emit a "
                "recall[] for it on this turn. A <tool> call is NOT a substitute "
                "and will not retrieve it. A \"### Long-term memory\" block in your "
                "context only says what EXISTS; call recall[] to read the spans. "
                "Saying you cannot find something is correct only AFTER you have "
                "recalled for it."
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
            + recall_help
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

    def _with_recall(self, window: list[dict], block: str) -> list[dict]:
        """Splice a recall block in after the stable system head.

        Placement is deliberate. Measured on :8080: the server caches an
        unchanged prefix automatically and WARM prefill is flat at 0.05 s
        regardless of context length, so where a block sits inside the prompt
        does not change prefill materially. It sits after the system message
        because that is where standing context belongs, and it is rebuilt each
        turn because it is QUERY-DEPENDENT -- the same store returns different
        spans for different goals.

        The caller's list is not mutated; the live `messages` keeps its own
        copy so the trajectory record stays un-recalled.
        """
        out = list(window)
        head = 0
        while head < len(out) and out[head]["role"] == "system":
            head += 1
        return out[:head] + [{"role": "system", "content": block}] + out[head:]

    # -- main loop --
    def run(self, goal: str) -> EpisodeResult:
        self._fired_turn = None      # a new goal re-arms the boundary
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"Goal: {goal}"},
        ]
        steps: list[StepRecord] = []
        tool_calls = harness_calls = invalid_calls = 0
        solved = False
        plain_streak = 0

        for t in range(self.max_steps):
            # Compaction is a RUNNING PROCESS when the sliding gate is on: the
            # sine window decides how much stays live this turn, and the
            # overflow is retired to the XYZ axes rather than waiting for the
            # window to cross FIT_CHARS. Otherwise the threshold compactor.
            if self.window_gate is not None and COMPACTOR_MODE != "threshold":
                before = len(getattr(self.window_gate, "retired", ()))
                window = self.window_gate.apply(messages, t, buffer=self.harness.buffer)
                messages = window      # feed it back so retirement compounds
                # Everything the gate just retired must land in the readable
                # store, or the cycle is rotating BLINDLY -- moving tokens out
                # without any way to bring the right one back. Measured on the
                # live model (90-record haystack, 25 live, needle at 7 depths):
                # recency 5/7 wrong, +LTM recall 0/7 wrong.
                if self.ltm is not None:
                    for rec in getattr(self.window_gate, "retired", ())[before:]:
                        if rec.get("text"):
                            self.ltm.retire(rec["text"], turn=t,
                                            meta={"n_messages": rec.get("n_messages")})
                retired_this_turn = len(
                    getattr(self.window_gate, "retired", ())) > before
            else:
                retired_this_turn = False
                window = self.compactor.fit(messages)

            # Recall. Injected as a SYSTEM message immediately after the stable
            # head, ahead of the Goal line, so the model reads it as standing
            # context rather than as a turn's instruction. Skipped when the
            # store is empty: an empty "### Long-term memory" header is noise
            # that costs prompt budget and says nothing.
            if self.ltm is not None and len(self.ltm):
                block = self.ltm.render(goal, k=RECALL_K,
                                        budget_chars=RECALL_CHARS)
                if block:
                    window = self._with_recall(window, block)
                    # Per-turn obligation, not just a one-off system-prompt
                    # paragraph. A free-running model emitted zero <action>
                    # blocks in 12 turns while the paragraph was present, so
                    # the reminder has to ride on the turn that actually needs
                    # it -- otherwise the duty model simply prefers <tool>.
                    #
                    # Fired only after a turn that retired something, so it
                    # marks exactly the turns where context was just lost.
                    if retired_this_turn:
                        window = self._with_recall(window, RECALL_NUDGE)
            reply = self.llm.chat(window)
            messages.append({"role": "assistant", "content": reply})

            observations = []

            done = DONE_RE.search(reply)
            self._turn = t          # the gate reads this
            tool = self._run_tool(reply)
            tool_obs = tool[2] if tool else None
            if tool is not None:
                tool_calls += 1
                observations.append(tool_obs)

            if parse_action(reply) is not None and self.use_bpe:
                # ltm=self.ltm is what makes recall ACTIONABLE: recall[] now
                # searches the spans the gate retired, not just the distilled
                # skill store. Without it the duty model can see a recalled
                # block in its prompt but has no way to ask for more.
                result = dispatch(reply, self.harness.state,
                                  on_write=self.harness._on_write,
                                  ltm=self.ltm)
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
