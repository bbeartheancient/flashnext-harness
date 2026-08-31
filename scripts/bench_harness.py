"""A/B/C live benchmark: raw tool loop vs BPE harness vs BPE+atlas latent rate.

Arms (identical loop machinery, toggled features):
  raw   — Agent(use_bpe=False): tools + <done>, no BPE state/actions/skills
  bpe   — GoalRunner: BPE state + skills + consolidation, persistent state
  atlas — bpe + XYZ latent lane at the atlas cache-map rate for the tap
          layer (rate_for(cmap, 63)); requires NEOH_LATENT=1

Tasks are verifiable file operations in a tmp workspace. A run counts as
VERIFIED only if ep.solved AND the task checker passes (solved-by-declaration
doesn't count). The bpe/atlas arms reuse one BPEState across tasks so the
learning loop (skills from earlier solved tasks) is measurable on the repeat
task family.

Usage:
  PYTHONPATH=. python scripts/bench_harness.py [--latent] [--out bench.json]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from neoharness import Agent, BPEState, Harness
from neoharness.goals import GoalRunner
from neoharness.llm import DutyLLM
from neoharness.skills import SkillStore
from neoharness.tools import make_core_tools

MODEL = os.environ.get(
    "BENCH_MODEL", str(Path.home() / "models/llm/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"))


def _write_fixture(ws: Path) -> None:
    (ws / "facts.txt").write_text(
        "MTP acceptance on the duty 27B runs 30-80% when healthy.\n"
        "The n-gram buffer lives on GPU1.\n"
    )
    (ws / "roster.txt").write_text("alice\nbob\ncarol\ndave\nerin\n")
    (ws / "part_a.txt").write_text("alpha: 10\n")
    (ws / "part_b.txt").write_text("beta: 20\n")


TASKS = [
    {
        "id": "summarize",
        "goal": "Read facts.txt, then write brief.txt containing a "
                "one-sentence brief of it, then finish.",
        "check": lambda ws: (ws / "brief.txt").exists()
        and "MTP" in (ws / "brief.txt").read_text(),
    },
    {
        "id": "count",
        "goal": "Count the lines in roster.txt (use grep or read_file), then "
                "write count.txt containing just the number, then finish.",
        "check": lambda ws: (ws / "count.txt").exists()
        and "5" in (ws / "count.txt").read_text(),
    },
    {
        "id": "merge",
        "goal": "Read part_a.txt and part_b.txt, then write merged.txt "
                "containing both of their lines, then finish.",
        "check": lambda ws: (ws / "merged.txt").exists()
        and "alpha" in (ws / "merged.txt").read_text()
        and "beta" in (ws / "merged.txt").read_text(),
    },
    {
        # same family as 'summarize': the BPE arm should recall the distilled
        # skill from the first pass
        "id": "summarize-repeat",
        "goal": "Read roster.txt, then write roster-brief.txt containing a "
                "one-sentence brief of it, then finish.",
        "check": lambda ws: (ws / "roster-brief.txt").exists()
        and len((ws / "roster-brief.txt").read_text()) > 20,
    },
]


def run_arm_raw(llm, ws: Path, task: dict, max_steps: int) -> dict:
    agent = Agent(llm, harness=Harness(llm, state=BPEState()),
                  tools=make_core_tools(ws), max_steps=max_steps,
                  use_bpe=False)
    t0 = time.time()
    ep = agent.run(task["goal"])
    return _result(ep, task, ws, t0, llm)


def run_arm_bpe(llm, ws: Path, task: dict, max_steps: int,
                state: BPEState, buffer=None) -> dict:
    harness = Harness(llm, state=state, buffer=buffer)
    agent = Agent(llm, harness=harness, tools=make_core_tools(ws),
                  max_steps=max_steps)
    runner = GoalRunner(agent, SkillStore(state))
    row_cos = []

    def on_step(step):
        vs = step.detail.get("vector_store")
        if vs and vs.get("row_cos"):
            row_cos.append(vs["row_cos"])

    agent.on_step = on_step
    t0 = time.time()
    result = runner.run_goal(task["goal"])
    out = _result(result.episode, task, ws, t0, llm)
    out["harness_calls"] = result.episode.harness_calls
    out["skill_written"] = bool(result.skill_path)
    out["latent_stores"] = len(row_cos)
    if row_cos:
        out["row_cos_mean"] = sum(row_cos) / len(row_cos)
    return out


def _result(ep, task, ws, t0, llm) -> dict:
    return {
        "task": task["id"],
        "solved": ep.solved,
        "verified": bool(ep.solved and task["check"](ws)),
        "steps": len(ep.steps),
        "tool_calls": ep.env_calls,
        "invalid": ep.invalid_calls,
        "wall_s": round(time.time() - t0, 1),
        "tokens": llm.usage["prompt_tokens"] + llm.usage["completion_tokens"],
    }


def _delta(results: list[dict]) -> None:
    base = 0
    for r in results:
        r["tokens_delta"] = r["tokens"] - base
        base = r["tokens"]


def _summary(arm: str, results: list[dict]) -> dict:
    n = len(results)
    return {
        "arm": arm,
        "verified": sum(r["verified"] for r in results),
        "of": n,
        "steps_total": sum(r["steps"] for r in results),
        "tokens_total": sum(r["tokens_delta"] for r in results),
        "wall_total_s": round(sum(r["wall_s"] for r in results), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", action="store_true",
                    help="add the atlas arm (NEOH_LATENT lane at map rate)")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    llm = DutyLLM(model=MODEL, max_tokens=args.max_tokens)
    report = {"arms": {}}

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        _write_fixture(ws)

        raw = [run_arm_raw(llm, ws, t, args.max_steps) for t in TASKS]
        for t in TASKS:  # reset workspace artifacts between arms
            for f in ("brief.txt", "count.txt", "merged.txt", "roster-brief.txt"):
                (ws / f).unlink(missing_ok=True)
        _delta(raw)
        report["arms"]["raw"] = raw
        print("raw:", json.dumps(_summary("raw", raw)), flush=True)

        state = BPEState()
        bpe = [run_arm_bpe(llm, ws, t, args.max_steps, state) for t in TASKS]
        for t in TASKS:
            for f in ("brief.txt", "count.txt", "merged.txt", "roster-brief.txt"):
                (ws / f).unlink(missing_ok=True)
        _delta(bpe)
        report["arms"]["bpe"] = bpe
        print("bpe:", json.dumps(_summary("bpe", bpe)), flush=True)
        report["bpe_skills"] = sorted(state.experience.skills)

        if args.latent:
            import os
            os.environ["NEOH_LATENT"] = "1"
            from neoharness import atlas as atlas_mod
            from neoharness.vectors import XYZBuffer
            cmap = atlas_mod.cache_map(atlas_mod.load())
            rate = atlas_mod.rate_for(cmap, 63)
            buffer = XYZBuffer(rate=rate)
            state2 = BPEState()
            atl = [run_arm_bpe(llm, ws, t, args.max_steps, state2, buffer=buffer)
                   for t in TASKS]
            _delta(atl)
            report["arms"]["atlas"] = atl
            report["atlas_rate"] = rate
            print("atlas:", json.dumps(_summary("atlas", atl)), flush=True)

    out = args.out or "bench_results.json"
    Path(out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    for name, res in report["arms"].items():
        print(json.dumps(_summary(name, res)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
