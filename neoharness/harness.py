"""The NeoHarness loop: BPE state + action protocol + XYZ buffer + cost ledger.

At each step the duty model receives the observation o_t, the rendered
harness state H_t = (B_t, P_t, E_t), and the task context, then emits one
<action>. Harness actions query/update the BPE stores (and mirror writes into
the n-gram buffer as XYZ coordinates); environment actions are passed to the
caller-provided environment.

Both action kinds consume the same interaction budget; the cost ledger
(harness_calls vs env_calls per episode) is exactly the quantity the paper's
cost-aware GRPO optimizes, recorded per step into a JSONL trajectory so
rollouts can later feed SFT/GRPO offline.

Environment protocol (duck-typed):
    env.reset(task) -> observation: str
    env.step(verb, arg) -> observation: str        # raises on bad command
    env.admissible() -> list[str]                  # optional
    env.solved() -> bool                           # optional
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .actions import HARNESS_VERBS, ActionResult, dispatch
from .bpe import BPEState, state_dir
from .consolidate import ConsolidationResult, consolidate

PROTOCOL_HELP = """\
You may emit one harness action per turn, in the form <action>verb[arg]</action>.
Harness actions (always available):
  track[object|world]   - read belief (environment facts)
  commit[subgoal | status] - record progress (status: pending|attempted|done|blocked)
  recall[query]         - retrieve relevant prior experience
  note[kind | insight]  - record an insight for later (kind: general|task|mistake|prior)
When tools are available you may ALSO emit one <tool> call in the same turn.
Environment actions: any command from ADMISSIBLE COMMANDS.
Every turn must contain a <tool> call, an <action> block, or <done> —
a bare prose turn is a protocol error. Harness actions and environment
actions share the same step budget — use the harness when it saves steps,
not by habit. Keep your reasoning short; act early."""


@dataclass
class StepRecord:
    t: int
    action_text: str
    kind: str
    verb: str
    arg: str
    observation: str
    axis: str | None = None
    detail: dict = field(default_factory=dict)


@dataclass
class EpisodeResult:
    task: str
    steps: list[StepRecord]
    solved: bool
    env_calls: int
    harness_calls: int
    invalid_calls: int
    consolidation: ConsolidationResult | None = None

    @property
    def cost(self) -> dict:
        """The cost-aware ledger: every action burns the same budget."""
        total = self.env_calls + self.harness_calls + self.invalid_calls
        return {
            "total_steps": total,
            "env_calls": self.env_calls,
            "harness_calls": self.harness_calls,
            "invalid_calls": self.invalid_calls,
            "harness_fraction": self.harness_calls / total if total else 0.0,
            "solved": self.solved,
        }


class Harness:
    def __init__(
        self,
        llm,
        state: BPEState | None = None,
        buffer=None,
        max_steps: int = 50,
        trajectory_dir: str | Path | None = None,
    ):
        self.llm = llm
        self.state = state or BPEState.load()
        self.buffer = buffer  # XYZBuffer or None (text-only harness)
        self.max_steps = max_steps
        self.trajectory_dir = Path(
            trajectory_dir or state_dir() / "trajectories"
        )

    # -- prompt --
    def _system_prompt(self) -> str:
        return (
            "You are a long-horizon task agent with an external harness "
            "workspace (Belief / Progress / Experience).\n\n"
            + PROTOCOL_HELP
            + "\n\n### Harness state (H_t)\n"
            + self.state.render()
        )

    # -- vector lane callback --
    def _on_write(self, axis: str, verb: str, rendered: str, detail: dict) -> None:
        if self.buffer is None:
            return
        resp = self.buffer.store_axis(axis, rendered, persist=False)
        detail["vector_store"] = resp

    # -- episode --
    def run_episode(self, task: str, env) -> EpisodeResult:
        obs = env.reset(task)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"Task: {task}\nObservation: {obs}"},
        ]
        steps: list[StepRecord] = []
        env_calls = harness_calls = invalid_calls = 0
        solved = False

        for t in range(self.max_steps):
            reply = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": reply})

            result: ActionResult = dispatch(reply, self.state, on_write=self._on_write)

            if result.kind == "env":
                env_calls += 1
                try:
                    result.observation = env.step(result.verb, result.arg)
                except Exception as exc:  # env rejected the command
                    result.observation = f"environment error: {exc}"
            elif result.kind == "harness":
                harness_calls += 1
            else:
                invalid_calls += 1

            steps.append(StepRecord(
                t=t, action_text=reply, kind=result.kind, verb=result.verb,
                arg=result.arg, observation=result.observation,
                axis=result.axis, detail=result.detail,
            ))
            messages.append({
                "role": "user",
                "content": f"Observation:\n{result.observation}\n\n"
                           f"### Harness state (H_t)\n{self.state.render()}",
            })

            if hasattr(env, "solved") and env.solved():
                solved = True
                break

        ep = EpisodeResult(
            task=task, steps=steps, solved=solved,
            env_calls=env_calls, harness_calls=harness_calls,
            invalid_calls=invalid_calls,
        )
        self._record(ep)
        return ep

    # -- epoch boundary --
    def consolidate(self, trajectory_summary: str | None = None,
                    prune_below_usage: int = 0) -> ConsolidationResult:
        result = consolidate(
            self.state.experience,
            trajectory_summary=trajectory_summary,
            prune_below_usage=prune_below_usage,
        )
        self.state.save()
        # Persisted consolidation is the one lane allowed to write the shard
        # (persist=True) — mirrors M1's live-vs-mv2 rule.
        if self.buffer is not None:
            self.buffer.store_axis("z", self.state.experience.render(),
                                   persist=True, mode="merge")
        return result

    # -- trajectory recording --
    def _record(self, ep: EpisodeResult) -> Path:
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        path = self.trajectory_dir / f"ep-{int(time.time() * 1000)}.jsonl"
        with path.open("w") as fh:
            fh.write(json.dumps({"task": ep.task, "cost": ep.cost}) + "\n")
            for s in ep.steps:
                fh.write(json.dumps({
                    "t": s.t, "kind": s.kind, "verb": s.verb, "arg": s.arg,
                    "axis": s.axis, "observation": s.observation,
                    "action_text": s.action_text,
                }) + "\n")
        return path


__all__ = ["Harness", "EpisodeResult", "StepRecord", "PROTOCOL_HELP", "HARNESS_VERBS"]
