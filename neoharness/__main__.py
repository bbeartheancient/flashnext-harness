"""CLI runner: python -m neoharness "goal text"

Streams the agent loop to the terminal as it happens — tool calls, BPE
actions, observations, cost ledger — then prints the final harness state.

Examples:
    python -m neoharness "summarize notes.txt into summary.txt"
    python -m neoharness --latent --max-steps 20 "audit the workspace"
    python -m neoharness --subgoal "read parser" --subgoal "fix bug" "fix parser"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import Agent
from .bpe import BPEState
from .goals import GoalRunner
from .harness import Harness
from .skills import SkillStore
from .tools import make_core_tools

KIND_ICON = {"tool": ">", "harness": "#", "env": ">", "answer": "=", "invalid": "!"}


def _print_step(step) -> None:
    icon = KIND_ICON.get(step.kind, "?")
    call = step.verb + (f" {step.arg}" if step.arg else "")
    print(f"  {icon} t{step.t} [{step.kind}] {call[:120]}", flush=True)
    if step.observation:
        for line in step.observation.splitlines()[:6]:
            print(f"      {line[:160]}", flush=True)
    if "done" in step.detail:
        print(f"      done: {step.detail['done'][:160]}", flush=True)
    if "vector_error" in step.detail:
        print(f"      (latent lane: {step.detail['vector_error']})", flush=True)
    vs = step.detail.get("vector_store")
    if vs:
        print(f"      XYZ[{vs.get('axis')}] session={vs.get('session')} "
              f"rows={vs.get('table_rows')} row_cos={vs.get('row_cos'):.4f}",
              flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neoharness",
        description="BPE agentic harness for the duty LLM (tool loop + "
                    "Belief/Progress/Experience state + XYZ n-gram buffer).",
    )
    p.add_argument("goal", help="the long-context goal to accomplish")
    p.add_argument("--subgoal", action="append", default=[],
                   help="seed a Progress subgoal (repeatable)")
    p.add_argument("--workspace", default=".",
                   help="tool sandbox root (default: cwd)")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--latent", action="store_true",
                   help="enable the XYZ latent lane (NEOH_LATENT=1; touches "
                        "the live vLLM hidden-states hook and spawns the "
                        "ple-buffer-worker)")
    p.add_argument("--model", default=None,
                   help="duty model id (default: NEOH_LLM_MODEL or 'duty')")
    p.add_argument("--base-url", default=None,
                   help="vLLM base url (default: NEOH_LLM_BASE_URL)")
    p.add_argument("--quiet", action="store_true",
                   help="only print the final result")
    return p


def main(argv: list[str] | None = None, llm=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "serve":
        from .serve import main as serve_main
        return serve_main(argv[1:], llm=llm)
    args = build_parser().parse_args(argv)

    import os
    buffer = None
    if args.latent:
        os.environ["NEOH_LATENT"] = "1"
        from .vectors import XYZBuffer
        buffer = XYZBuffer()

    if llm is None:
        from .llm import DutyLLM
        llm = DutyLLM(base_url=args.base_url, model=args.model,
                      max_tokens=args.max_tokens)

    workspace = Path(args.workspace).resolve()
    harness = Harness(llm, state=BPEState.load(), buffer=buffer)
    agent = Agent(llm, harness=harness, tools=make_core_tools(workspace),
                  max_steps=args.max_steps)
    if not args.quiet:
        agent.on_step = _print_step
    runner = GoalRunner(agent, SkillStore(harness.state))

    print(f"goal: {args.goal}", flush=True)
    print(f"workspace: {workspace}  latent: {'on' if args.latent else 'off'}",
          flush=True)
    result = runner.run_goal(args.goal, subgoals=args.subgoal or None)

    print("\n--- result ---")
    print("solved:", result.solved)
    print("cost:", result.episode.cost)
    if result.skill_path:
        print("skill:", result.skill_path)
    print("\n### Harness state (final)\n" + harness.state.render())
    return 0 if result.solved else 1


if __name__ == "__main__":
    sys.exit(main())
