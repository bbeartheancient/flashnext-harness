"""Live smoke run: GoalRunner against the duty 27B on :8000.

Text-only path (NEOH_LATENT unset) — proves the agentic loop against the
real model without touching the latent lane. Workspace is a tmp dir.
"""

import sys
import os
import tempfile
import time
from pathlib import Path

from neoharness import Agent, BPEState, GoalRunner, Harness
from neoharness.llm import DutyLLM
from neoharness.skills import SkillStore
from neoharness.tools import make_core_tools

MODEL = os.environ.get(
    "SMOKE_MODEL", str(Path.home() / "models/llm/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"))

GOAL = (
    "In the workspace there is a file notes.txt. Read it, then write a new "
    "file summary.txt containing a one-line summary of notes.txt, then mark "
    "the goal done. Keep every reply short — one tool or action per turn."
)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="neoh-smoke-"))
    (tmp / "notes.txt").write_text(
        "The bicameral harness stores belief, progress, and experience as "
        "XYZ vectors in the n-gram buffer.\n"
    )
    llm = DutyLLM(model=MODEL, max_tokens=512, timeout=300)
    harness = Harness(llm, state=BPEState(), trajectory_dir=tmp / "traj")
    agent = Agent(llm, harness=harness, tools=make_core_tools(tmp), max_steps=12)
    runner = GoalRunner(agent, SkillStore(harness.state, directory=tmp / "skills"))

    print(f"workspace: {tmp}", flush=True)
    t0 = time.time()
    result = runner.run_goal(GOAL)
    dt = time.time() - t0

    print(f"\n--- {dt:.0f}s ---")
    print("solved:", result.solved)
    print("cost:", result.episode.cost)
    for s in result.episode.steps:
        print(f"  t{s.t} [{s.kind}] {s.verb} {s.arg[:80]!r} -> {s.observation[:120]!r}")
    summary = tmp / "summary.txt"
    print("summary.txt:", summary.read_text() if summary.exists() else "(missing)")
    print("skill:", result.skill_path)
    print("\nfinal harness state:\n" + harness.state.render())
    return 0 if result.solved and summary.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
